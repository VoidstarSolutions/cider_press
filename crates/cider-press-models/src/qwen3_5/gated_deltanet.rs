//! Qwen3.5 / Qwen3.6 Gated-DeltaNet linear-attention mixer.
//!
//! Phase-3 lands the mixer's building blocks; this module currently provides
//! the depthwise causal Conv1d that opens the mixer (over the `q‖k‖v`
//! projection). The `GatedDeltaNet` struct and the gated recurrence land in
//! later phase-3 tasks. See `docs/inference/models/qwen3.6.md` and `ROADMAP.md`.

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
}
