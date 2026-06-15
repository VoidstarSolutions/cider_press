//! Qwen3.5 / Qwen3.6 Gated-DeltaNet linear-attention mixer.
//!
//! Phase-3 lands the mixer's building blocks incrementally: the depthwise
//! causal Conv1d that opens the mixer (over the `q‖k‖v` projection) and the
//! per-(token, value-head) gates (`compute_g` — the Mamba2 forget gate — and
//! `gdn_beta`, computed in fp32). The `GatedDeltaNet` struct, the q/k norm,
//! and the gated recurrence land in later phase-3 tasks. See
//! `docs/inference/models/qwen3.6.md` and `ROADMAP.md`.

use cider_press_runtime::{DType, Tensor};

use crate::error::{Error, Result};

/// Depthwise causal Conv1d (kernel 4, groups = channels, no bias) + SiLU,
/// for the Gated-DeltaNet mixer. `qkv` `[1, T, C]`, `conv_state` `[1, 3, C]`
/// (the kernel-1 cached pre-conv tokens; zeros at prefill start), `weight`
/// `[C, 4, 1]`. Returns `(conv_out [1, T, C]` post-SiLU, `new_conv_state
/// [1, 3, C])`. Composed as `concat` (left-pad) + 4 shifted per-channel scaled
/// adds + SiLU — no conv kernel.
//
// mlx-lm `qwen3_5.py:GatedDeltaNet`: conv_input = concat([conv_state, qkv]) so
// the kernel-3 history left-pads the sequence; per output t, channel c:
// conv_out[t,c] = silu(Σ_{j=0..3} weight[c,j,0] · conv_input[t+j, c]). The new
// state is the last 3 PRE-conv rows for the next call.
#[allow(dead_code)] // wired into GatedDeltaNet in B6.
pub(crate) fn conv1d_causal(
    qkv: &Tensor,
    conv_state: &Tensor,
    weight: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let qkv_dims = qkv.shape().dims();
    let state_dims = conv_state.shape().dims();
    let w_dims = weight.shape().dims();

    if qkv_dims.len() != 3 || qkv_dims[0] != 1 {
        return Err(Error::InvalidArgument(format!(
            "conv1d_causal: qkv must be rank-3 [1, T, C]; got {qkv_dims:?}",
        )));
    }
    let (t, c) = (qkv_dims[1], qkv_dims[2]);

    if state_dims != [1, 3, c] {
        return Err(Error::InvalidArgument(format!(
            "conv1d_causal: conv_state must be [1, 3, {c}]; got {state_dims:?}",
        )));
    }
    if w_dims.len() != 3 || w_dims[0] != c || w_dims[2] != 1 {
        return Err(Error::InvalidArgument(format!(
            "conv1d_causal: weight must be [{c}, 4, 1]; got {w_dims:?}",
        )));
    }
    let kernel = w_dims[1];
    if kernel != 4 {
        return Err(Error::InvalidArgument(format!(
            "conv1d_causal: only kernel size 4 is supported; got {kernel}",
        )));
    }
    for (name, t) in [("qkv", qkv), ("conv_state", conv_state), ("weight", weight)] {
        if t.dtype() != DType::BF16 {
            return Err(Error::InvalidArgument(format!(
                "conv1d_causal: {name} must be BF16; got {:?}",
                t.dtype()
            )));
        }
    }

    // Left-pad the sequence with the 3 cached pre-conv tokens: conv_input is
    // [1, 3+T, C], so a kernel-4 window ending at output t reads inputs
    // t..t+4 and output length stays T (padding=0, the pad supplies history).
    let conv_input = Tensor::concat(&[conv_state, qkv], 1)?;

    // Sum 4 shifted, per-channel-scaled copies of conv_input. Tap j weights
    // channel c by weight[c,j,0]; reshape to [1,1,C] so it broadcasts over the
    // [1,T,C] shifted slice. Slices feed mul (binary), so copy() to a dense
    // leaf — view chains aren't accepted as binary-op inputs.
    let mut acc: Option<Tensor> = None;
    for j in 0..kernel {
        let wj = weight
            .slice(&[0..c, j..j + 1, 0..1])?
            .copy()?
            .reshape([1usize, 1, c])?;
        let xj = conv_input.slice(&[0..1, j..j + t, 0..c])?.copy()?;
        let term = xj.mul(&wj)?;
        acc = Some(match acc {
            None => term,
            Some(prev) => prev.add(&term)?,
        });
    }
    let acc = acc.expect("kernel == 4 guarantees at least one tap");

    let conv_out = crate::nn::silu(&acc)?;

    // Next state: the last 3 PRE-conv rows of conv_input (this call's tail).
    let new_state = conv_input
        .slice(&[0..1, (3 + t - 3)..(3 + t), 0..c])?
        .copy()?;

    Ok((conv_out, new_state))
}

/// Gated-DeltaNet forget gate `g = exp(-exp(A_log) * softplus(a + dt_bias))`,
/// computed entirely in fp32 (the recurrence runs in fp32). `a` `[1, T, Hv]`
/// BF16 (from a Linear), `a_log` `[Hv]` f32 (loaded f32), `dt_bias` `[Hv]`
/// BF16. Returns `g` `[1, T, Hv]` f32.
//
// mlx-lm `gated_delta.py:compute_g`:
//   exp(-exp(A_log.astype(f32)) * softplus(a + dt_bias))
// with softplus(u) = log(1 + exp(u)) (mlx uses logaddexp(u, 0); equivalent for
// these O(1) inputs). The reference promotes the whole chain to fp32, so we
// cast `a`/`dt_bias` up front and keep every op in f32. `-x` is composed as a
// multiply by a broadcast scalar (no negate primitive).
#[allow(dead_code)] // wired into GatedDeltaNet in B6.
pub(crate) fn compute_g(a: &Tensor, a_log: &Tensor, dt_bias: &Tensor) -> Result<Tensor> {
    let a_dims = a.shape().dims();
    if a_dims.len() != 3 || a_dims[0] != 1 {
        return Err(Error::InvalidArgument(format!(
            "compute_g: a must be rank-3 [1, T, Hv]; got {a_dims:?}",
        )));
    }
    let (t, hv) = (a_dims[1], a_dims[2]);
    if a.dtype() != DType::BF16 {
        return Err(Error::InvalidArgument(format!(
            "compute_g: a must be BF16; got {:?}",
            a.dtype()
        )));
    }
    if dt_bias.shape().dims() != [hv] {
        return Err(Error::InvalidArgument(format!(
            "compute_g: dt_bias must be [{hv}]; got {:?}",
            dt_bias.shape().dims()
        )));
    }
    if dt_bias.dtype() != DType::BF16 {
        return Err(Error::InvalidArgument(format!(
            "compute_g: dt_bias must be BF16; got {:?}",
            dt_bias.dtype()
        )));
    }
    if a_log.shape().dims() != [hv] {
        return Err(Error::InvalidArgument(format!(
            "compute_g: a_log must be [{hv}]; got {:?}",
            a_log.shape().dims()
        )));
    }
    if a_log.dtype() != DType::F32 {
        return Err(Error::InvalidArgument(format!(
            "compute_g: a_log must be F32; got {:?}",
            a_log.dtype()
        )));
    }

    let device = a
        .device()
        .ok_or_else(|| Error::InvalidArgument("compute_g: a has no device (placeholder)".into()))?;

    // Promote everything to fp32 to match the reference's fp32 chain.
    let a_f32 = a.cast(DType::F32)?;
    // dt_bias / a_log are per-head [Hv]; reshape to [1, 1, Hv] so they broadcast
    // over the [1, T, Hv] token axis. copy() the broadcast views to dense leaves
    // — unary/binary ops require contiguous inputs.
    let dt_bias_b = dt_bias
        .cast(DType::F32)?
        .reshape([1usize, 1, hv])?
        .broadcast_to([1usize, t, hv])?
        .copy()?;

    // softplus(u) = log(1 + exp(u)); `ones` is a scalar that broadcasts.
    let ones = Tensor::from_slice(device, &[1.0f32], [1usize])?;
    let u = a_f32.add(&dt_bias_b)?;
    let sp = u.exp()?.add(&ones)?.log()?;

    let ea = a_log
        .exp()?
        .reshape([1usize, 1, hv])?
        .broadcast_to([1usize, t, hv])?
        .copy()?;
    let prod = ea.mul(&sp)?;

    // g = exp(-prod): negate via multiply by a broadcast scalar -1.
    let neg_one = Tensor::from_slice(device, &[-1.0f32], [1usize])?;
    Ok(prod.mul(&neg_one)?.exp()?)
}

/// Gated-DeltaNet `beta = sigmoid(b)`, returned as f32 `[1, T, Hv]` for the
/// fp32 recurrence. `b` `[1, T, Hv]` BF16.
//
// mlx-lm computes `mx.sigmoid(b)` on the BF16 input, and the recurrence then
// promotes the result to fp32 — so we sigmoid first (BF16) and cast up after,
// matching the reference's BF16 sigmoid → fp32 promote order.
#[allow(dead_code)] // wired into GatedDeltaNet in B6.
pub(crate) fn gdn_beta(b: &Tensor) -> Result<Tensor> {
    if b.dtype() != DType::BF16 {
        return Err(Error::InvalidArgument(format!(
            "gdn_beta: b must be BF16; got {:?}",
            b.dtype()
        )));
    }
    Ok(b.sigmoid()?.cast(DType::F32)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cider_press_runtime::{Device, Tensor};
    use half::bf16;

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn conv1d_causal_matches_fp32_reference() {
        let device = Device::shared().unwrap();
        let (c, kernel, t) = (4usize, 4usize, 2usize);

        // Deterministic small fractions.
        let frac = |i: usize, seed: usize| bf16::from_f32(((((i + seed) % 9) as f32) - 4.0) * 0.1);

        let state_data: Vec<bf16> = (0..3 * c).map(|i| frac(i, 1)).collect();
        let qkv_data: Vec<bf16> = (0..t * c).map(|i| frac(i, 5)).collect();
        let w_data: Vec<bf16> = (0..c * kernel).map(|i| frac(i, 2)).collect();

        let conv_state = Tensor::from_slice(&device, &state_data, [1usize, 3, c]).unwrap();
        let qkv = Tensor::from_slice(&device, &qkv_data, [1usize, t, c]).unwrap();
        let weight = Tensor::from_slice(&device, &w_data, [c, kernel, 1usize]).unwrap();

        let (conv_out, new_state) = conv1d_causal(&qkv, &conv_state, &weight).unwrap();
        conv_out.eval().unwrap();
        new_state.eval().unwrap();
        let got_out = conv_out.cpu_to_vec::<bf16>().unwrap();
        let got_state = new_state.cpu_to_vec::<bf16>().unwrap();
        assert_eq!(got_out.len(), t * c);
        assert_eq!(got_state.len(), 3 * c);

        // fp32 host reference: conv_input [3+T, C] row-major.
        let sf: Vec<f32> = state_data.iter().map(|x| x.to_f32()).collect();
        let qf: Vec<f32> = qkv_data.iter().map(|x| x.to_f32()).collect();
        let wf: Vec<f32> = w_data.iter().map(|x| x.to_f32()).collect();
        let mut conv_input = vec![0f32; (3 + t) * c];
        conv_input[..3 * c].copy_from_slice(&sf);
        conv_input[3 * c..].copy_from_slice(&qf);
        let ci = |row: usize, ch: usize| conv_input[row * c + ch];
        // weight[c, j, 0] is row-major over [C, 4, 1]: index c*kernel + j.
        let wi = |ch: usize, j: usize| wf[ch * kernel + j];

        let mut ref_out = vec![0f32; t * c];
        for tt in 0..t {
            for ch in 0..c {
                let mut s = 0f32;
                for j in 0..kernel {
                    s += wi(ch, j) * ci(tt + j, ch);
                }
                let silu = s * (1.0 / (1.0 + (-s).exp())); // s * sigmoid(s)
                ref_out[tt * c + ch] = silu;
            }
        }
        // new_state = last 3 pre-conv rows of conv_input.
        let ref_state: Vec<f32> = conv_input[(3 + t - 3) * c..(3 + t) * c].to_vec();

        // conv_out: silu carries sigmoid → combined bf16-ULP bound.
        let (atol, rtol) = (1e-2f32, 2e-2f32);
        let mut max_abs = 0f32;
        for (g, r) in got_out.iter().map(|x| x.to_f32()).zip(ref_out.iter()) {
            let abs = (g - r).abs();
            max_abs = max_abs.max(abs);
            assert!(
                abs <= atol + rtol * r.abs(),
                "conv_out mismatch: got={g}, ref={r}, abs={abs}",
            );
        }
        println!("conv_out max_abs={max_abs}");

        // new_state: pure data movement → bit-exact.
        for (g, r) in got_state.iter().map(|x| x.to_f32()).zip(ref_state.iter()) {
            assert!(
                (g - r).abs() == 0.0,
                "new_state must be bit-exact: got={g}, ref={r}",
            );
        }
    }

    #[test]
    #[allow(clippy::cast_precision_loss, clippy::many_single_char_names)]
    fn gates_match_fp32_reference() {
        let device = Device::shared().unwrap();
        let (t, hv) = (2usize, 4usize);

        // Deterministic O(1) values; a_log = ln of small positives.
        let a_data: Vec<bf16> = (0..t * hv)
            .map(|i| bf16::from_f32(((i % 7) as f32 - 3.0) * 0.2))
            .collect();
        let b_data: Vec<bf16> = (0..t * hv)
            .map(|i| bf16::from_f32(((i % 5) as f32 - 2.0) * 0.3))
            .collect();
        let a_log_f32: Vec<f32> = vec![0.5f32.ln(), 2.0f32.ln(), 4.0f32.ln(), 8.0f32.ln()];
        let dt_bias_data: Vec<bf16> = (0..hv)
            .map(|i| bf16::from_f32(1.0 + (i as f32) * 0.1))
            .collect();

        let a = Tensor::from_slice(&device, &a_data, [1usize, t, hv]).unwrap();
        let b = Tensor::from_slice(&device, &b_data, [1usize, t, hv]).unwrap();
        let a_log = Tensor::from_slice(&device, &a_log_f32, [hv]).unwrap();
        let dt_bias = Tensor::from_slice(&device, &dt_bias_data, [hv]).unwrap();

        let g = compute_g(&a, &a_log, &dt_bias).unwrap();
        let beta = gdn_beta(&b).unwrap();
        g.eval().unwrap();
        beta.eval().unwrap();
        assert_eq!(g.dtype(), DType::F32);
        assert_eq!(beta.dtype(), DType::F32);
        let got_g = g.cpu_to_vec::<f32>().unwrap();
        let got_beta = beta.cpu_to_vec::<f32>().unwrap();
        assert_eq!(got_g.len(), t * hv);
        assert_eq!(got_beta.len(), t * hv);

        // fp32 host reference.
        let af: Vec<f32> = a_data.iter().map(|x| x.to_f32()).collect();
        let bf: Vec<f32> = b_data.iter().map(|x| x.to_f32()).collect();
        let dbf: Vec<f32> = dt_bias_data.iter().map(|x| x.to_f32()).collect();

        let mut ref_g = vec![0f32; t * hv];
        let mut ref_beta = vec![0f32; t * hv];
        for tt in 0..t {
            for h in 0..hv {
                let idx = tt * hv + h;
                let u = af[idx] + dbf[h];
                let softplus = (1.0 + u.exp()).ln();
                ref_g[idx] = (-a_log_f32[h].exp() * softplus).exp();
                ref_beta[idx] = 1.0 / (1.0 + (-bf[idx]).exp());
            }
        }

        // exp/log/sigmoid carrying → combined bf16-ULP bound.
        let (atol, rtol) = (1e-2f32, 2e-2f32);
        let mut max_abs = 0f32;
        for (g, r) in got_g.iter().zip(ref_g.iter()) {
            let abs = (g - r).abs();
            max_abs = max_abs.max(abs);
            assert!(
                abs <= atol + rtol * r.abs(),
                "g mismatch: got={g}, ref={r}, abs={abs}"
            );
        }
        for (g, r) in got_beta.iter().zip(ref_beta.iter()) {
            let abs = (g - r).abs();
            max_abs = max_abs.max(abs);
            assert!(
                abs <= atol + rtol * r.abs(),
                "beta mismatch: got={g}, ref={r}, abs={abs}"
            );
        }
        println!("gates max_abs={max_abs}");
    }
}
