//! Qwen3.5 / Qwen3.6 Gated-DeltaNet linear-attention mixer.
//!
//! Phase-3 lands the mixer's building blocks incrementally: the depthwise
//! causal Conv1d that opens the mixer (over the `q‖k‖v` projection), the
//! per-(token, value-head) gates (`compute_g` — the Mamba2 forget gate — and
//! `gdn_beta`, computed in fp32), and the q/k/v preparation that feeds the
//! recurrence — `split_qkv` (the `conv_out` → per-head `q`/`k`/`v` split),
//! `qk_norm_scale` (the weightless per-head RMSNorm + scalar scale applied to
//! `q` and `k`), and `gqa_repeat` (the interleaved key/value-head broadcast).
//! `recurrence` is the per-token Gated-DeltaNet delta-rule update over an fp32
//! `[Hv, Dv, Dk]` state (the parity-critical core of the mixer). The
//! [`GatedDeltaNet`] struct composes all of these into the full mixer forward
//! (project → conv → split → q/k norm+scale → GQA → gates → recurrence → gated
//! output norm → `out_proj`). See `docs/inference/models/qwen3.6.md` and
//! `ROADMAP.md`.

use crate::error::{Error, Result};
use crate::nn::{Linear, Module, repeat_axis_interleaved, scalar_tensor, silu};
use crate::qwen3_5::config::Qwen35Config;
use crate::qwen3_5::weights::Qwen35LinearAttnWeights;
use cider_press_runtime::{DType, Tensor};

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
pub(crate) fn compute_g(a: &Tensor, a_log: &Tensor, dt_bias: &Tensor) -> Result<Tensor> {
    let a_dims = a.shape().dims();
    if a_dims.len() != 3 || a_dims[0] != 1 {
        return Err(Error::InvalidArgument(format!(
            "compute_g: a must be rank-3 [1, T, Hv]; got {a_dims:?}",
        )));
    }
    let hv = a_dims[2];
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
    // dt_bias / a_log are per-head [Hv]; reshape to [1, 1, Hv] and let the
    // rank-3 strided binary op broadcast over the [1, T, Hv] token axis.
    let dt_bias_b = dt_bias.cast(DType::F32)?.reshape([1usize, 1, hv])?;

    // softplus(u) = log(1 + exp(u)); `ones` is a scalar that broadcasts.
    let ones = scalar_tensor(device, DType::F32, 1.0)?;
    let u = a_f32.add(&dt_bias_b)?;
    let sp = u.exp()?.add(&ones)?.log()?;

    let ea = a_log.exp()?.reshape([1usize, 1, hv])?;
    let prod = ea.mul(&sp)?;

    // g = exp(-prod): negate via multiply by a broadcast scalar -1.
    let neg_one = scalar_tensor(device, DType::F32, -1.0)?;
    Ok(prod.mul(&neg_one)?.exp()?)
}

/// Gated-DeltaNet `beta = sigmoid(b)`, returned as f32 `[1, T, Hv]` for the
/// fp32 recurrence. `b` `[1, T, Hv]` BF16.
//
// mlx-lm computes `mx.sigmoid(b)` on the BF16 input, and the recurrence then
// promotes the result to fp32 — so we sigmoid first (BF16) and cast up after,
// matching the reference's BF16 sigmoid → fp32 promote order.
pub(crate) fn gdn_beta(b: &Tensor) -> Result<Tensor> {
    if b.dtype() != DType::BF16 {
        return Err(Error::InvalidArgument(format!(
            "gdn_beta: b must be BF16; got {:?}",
            b.dtype()
        )));
    }
    Ok(b.sigmoid()?.cast(DType::F32)?)
}

/// Split the post-conv `conv_out` `[1, T, conv_dim]` into per-head `q`
/// `[1, T, Hk, Dk]`, `k` `[1, T, Hk, Dk]`, and `v` `[1, T, Hv, Dv]`. The split
/// boundaries are `key_dim = Hk·Dk` and `2·key_dim`; the value tail is
/// `value_dim = Hv·Dv`. Requires `conv_dim == 2·Hk·Dk + Hv·Dv`.
//
// mlx-lm `qwen3_5.py:GatedDeltaNet`: q, k, v = split(conv_out, [key_dim,
// 2*key_dim], -1) then reshaped to (.., Hk, Dk) / (.., Hk, Dk) / (.., Hv, Dv).
// Slices feed reshape (and downstream binary ops), so copy() each to a dense
// leaf — view chains aren't accepted.
pub(crate) fn split_qkv(
    conv_out: &Tensor,
    hk: usize,
    hv: usize,
    dk: usize,
    dv: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    let dims = conv_out.shape().dims();
    if dims.len() != 3 || dims[0] != 1 {
        return Err(Error::InvalidArgument(format!(
            "split_qkv: conv_out must be rank-3 [1, T, conv_dim]; got {dims:?}",
        )));
    }
    let (t, conv_dim) = (dims[1], dims[2]);
    let key_dim = hk * dk;
    let value_dim = hv * dv;
    if conv_dim != 2 * key_dim + value_dim {
        return Err(Error::InvalidArgument(format!(
            "split_qkv: conv_dim {conv_dim} != 2*Hk*Dk + Hv*Dv ({}); \
             Hk={hk}, Hv={hv}, Dk={dk}, Dv={dv}",
            2 * key_dim + value_dim,
        )));
    }

    let q = conv_out
        .slice(&[0..1, 0..t, 0..key_dim])?
        .copy()?
        .reshape([1usize, t, hk, dk])?;
    let k = conv_out
        .slice(&[0..1, 0..t, key_dim..2 * key_dim])?
        .copy()?
        .reshape([1usize, t, hk, dk])?;
    let v = conv_out
        .slice(&[0..1, 0..t, 2 * key_dim..conv_dim])?
        .copy()?
        .reshape([1usize, t, hv, dv])?;
    Ok((q, k, v))
}

/// Weightless per-head RMSNorm over the last axis (`D`) followed by a scalar
/// `scale` broadcast. `x` `[1, T, H, D]` BF16; the RMSNorm uses eps `1e-6` and
/// a unit (`gamma = 1`) weight, so it normalizes each head row without an
/// affine transform. Returns `[1, T, H, D]` BF16.
//
// mlx-lm: q = (inv**2) * rms_norm(q, None, 1e-6); k = inv * rms_norm(k, None,
// 1e-6) with inv = Dk**-0.5 — so the caller passes scale = 1/Dk for q and
// 1/sqrt(Dk) for k. `rms_norm(None)` is the weightless path: a stride-0 scalar
// `1.0` gamma read by every lane (mlx's `weight=None`), no `[D]` ones upload.
// The reshaped/sliced input is a view, so copy() it to a dense leaf first.
pub(crate) fn qk_norm_scale(x: &Tensor, scale: f32) -> Result<Tensor> {
    let dims = x.shape().dims();
    if dims.len() != 4 || dims[0] != 1 {
        return Err(Error::InvalidArgument(format!(
            "qk_norm_scale: x must be rank-4 [1, T, H, D]; got {dims:?}",
        )));
    }
    if x.dtype() != DType::BF16 {
        return Err(Error::InvalidArgument(format!(
            "qk_norm_scale: x must be BF16; got {:?}",
            x.dtype()
        )));
    }
    let (t, heads, dim) = (dims[1], dims[2], dims[3]);
    let device = x.device().ok_or_else(|| {
        Error::InvalidArgument("qk_norm_scale: x has no device (placeholder)".into())
    })?;

    // Weightless RMSNorm (gamma = 1). rms_norm rejects view inputs — copy()
    // the (possibly reshaped) x to a dense leaf first.
    let normed = x.copy()?.rms_norm(None, 1e-6)?;

    // Broadcast scalar scale. The binary op's strided dispatch is only wired up
    // to rank 3, so flatten the contiguous rms_norm output to rank-1, multiply
    // by the broadcast scalar, then restore the [1, T, H, D] shape.
    let scale_t = scalar_tensor(device, DType::BF16, scale)?;
    let scaled = normed.reshape([t * heads * dim])?.copy()?.mul(&scale_t)?;
    Ok(scaled.reshape([1usize, t, heads, dim])?)
}

/// Interleaved GQA repeat: broadcast `[1, T, Hk, D]` to `[1, T, Hv, D]` by a
/// factor `Hv / Hk` over the head axis. Output head `hv` reads input head
/// `hv / factor` — i.e. each input head `i` becomes the `factor` contiguous
/// output heads `i·factor .. (i+1)·factor`. Requires `Hv % Hk == 0`.
//
// mlx-lm `qwen3_5.py:GatedDeltaNet`: q, k = mx.repeat(., factor, axis=-2)
// inside the recurrence. mx.repeat replicates each head in place (interleaved),
// so out heads 2i and 2i+1 both equal in head i — NOT a tiled concat. Composed
// as reshape [..,Hk,1,D] → broadcast [..,Hk,factor,D] → copy → reshape
// [..,Hv,D].
pub(crate) fn gqa_repeat(x: &Tensor, hv: usize) -> Result<Tensor> {
    let dims = x.shape().dims();
    if dims.len() != 4 || dims[0] != 1 {
        return Err(Error::InvalidArgument(format!(
            "gqa_repeat: x must be rank-4 [1, T, Hk, D]; got {dims:?}",
        )));
    }
    let hk = dims[2];
    if hk == 0 || hv % hk != 0 {
        return Err(Error::InvalidArgument(format!(
            "gqa_repeat: Hv ({hv}) must be a nonzero multiple of Hk ({hk})",
        )));
    }

    // Interleaved replication over the head axis (2): out head i*factor..(i+1)*
    // factor each equal in head i.
    repeat_axis_interleaved(x, 2, hv / hk)
}

/// Gated-DeltaNet per-token recurrence (fp32 state). `q`, `k`, `v`
/// `[1, T, Hv, D]` (cast to f32 internally), `g`, `beta` `[1, T, Hv]` f32,
/// `state0` `[1, Hv, Dv, Dk]` f32 (or `None` ⇒ zeros). Returns
/// (`y` `[1, T, Hv, Dv]` bf16, `final_state` `[1, Hv, Dv, Dk]` f32). The state
/// is worked in rank-3 `[Hv, Dv, Dk]` to stay within the wired strided-binary
/// ranks.
//
// mlx-lm `gated_delta.py:_gated_delta_step_ops`, per timestep, on state
// M [Hv, Dv, Dk] (fp32), inputs q_t,k_t [Hv,Dk], v_t [Hv,Dv], g_t,beta_t [Hv]:
//   1. decay:  M = M * g_t[:, None, None]
//   2. read:   kv_mem = (M * k_t[:, None, :]).sum(-1)        # on the DECAYED M
//   3. delta:  delta = (v_t - kv_mem) * beta_t[:, None]
//   4. write:  M = M + k_t[:, None, :] * delta[:, :, None]   # rank-1 outer
//   5. out:    y_t = (M * q_t[:, None, :]).sum(-1)           # on the UPDATED M
// kv_mem reads the decayed M (after 1, before 4); y_t reads the updated M
// (after 4). Order is load-bearing. y_t cast to bf16; stacked over t.
#[allow(
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::too_many_lines
)]
pub(crate) fn recurrence(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state0: Option<&Tensor>,
) -> Result<(Tensor, Tensor)> {
    let qd = q.shape().dims();
    let kd = k.shape().dims();
    let vd = v.shape().dims();
    let gd = g.shape().dims();
    let bd = beta.shape().dims();

    if qd.len() != 4 || qd[0] != 1 {
        return Err(Error::InvalidArgument(format!(
            "recurrence: q must be rank-4 [1, T, Hv, Dk]; got {qd:?}",
        )));
    }
    let (t, hv, dk) = (qd[1], qd[2], qd[3]);
    if kd != [1, t, hv, dk] {
        return Err(Error::InvalidArgument(format!(
            "recurrence: k must be [1, {t}, {hv}, {dk}]; got {kd:?}",
        )));
    }
    if vd.len() != 4 || vd[0] != 1 || vd[1] != t || vd[2] != hv {
        return Err(Error::InvalidArgument(format!(
            "recurrence: v must be [1, {t}, {hv}, Dv]; got {vd:?}",
        )));
    }
    let dv = vd[3];
    if gd != [1, t, hv] {
        return Err(Error::InvalidArgument(format!(
            "recurrence: g must be [1, {t}, {hv}]; got {gd:?}",
        )));
    }
    if bd != [1, t, hv] {
        return Err(Error::InvalidArgument(format!(
            "recurrence: beta must be [1, {t}, {hv}]; got {bd:?}",
        )));
    }
    if g.dtype() != DType::F32 || beta.dtype() != DType::F32 {
        return Err(Error::InvalidArgument(format!(
            "recurrence: g/beta must be F32; got g={:?}, beta={:?}",
            g.dtype(),
            beta.dtype()
        )));
    }

    let device = q.device().ok_or_else(|| {
        Error::InvalidArgument("recurrence: q has no device (placeholder)".into())
    })?;

    // Cast q/k/v to f32 (the whole recurrence runs in fp32). `cast` clones on a
    // matching dtype and converts BF16; an unsupported dtype surfaces its error.
    let qf = q.cast(DType::F32)?;
    let kf = k.cast(DType::F32)?;
    let vf = v.cast(DType::F32)?;

    // State M [Hv, Dv, Dk] f32 (rank-3 to stay within wired strided binaries).
    let mut m = match state0 {
        Some(s) => {
            let sd = s.shape().dims();
            if sd != [1, hv, dv, dk] {
                return Err(Error::InvalidArgument(format!(
                    "recurrence: state0 must be [1, {hv}, {dv}, {dk}]; got {sd:?}",
                )));
            }
            if s.dtype() != DType::F32 {
                return Err(Error::InvalidArgument(format!(
                    "recurrence: state0 must be F32; got {:?}",
                    s.dtype()
                )));
            }
            s.copy()?.reshape([hv, dv, dk])?
        }
        None => Tensor::zeros(device, [hv, dv, dk], DType::F32)?,
    };

    let neg_one = scalar_tensor(device, DType::F32, -1.0)?;

    let mut ys: Vec<Tensor> = Vec::with_capacity(t);
    for ti in 0..t {
        // Per-step slices → dense rank-3/2/1 leaves.
        let q_t = qf
            .slice(&[0..1, ti..ti + 1, 0..hv, 0..dk])?
            .copy()?
            .reshape([hv, dk])?;
        let k_t = kf
            .slice(&[0..1, ti..ti + 1, 0..hv, 0..dk])?
            .copy()?
            .reshape([hv, dk])?;
        let v_t = vf
            .slice(&[0..1, ti..ti + 1, 0..hv, 0..dv])?
            .copy()?
            .reshape([hv, dv])?;
        let g_t = g.slice(&[0..1, ti..ti + 1, 0..hv])?.copy()?.reshape([hv])?;
        let beta_t = beta
            .slice(&[0..1, ti..ti + 1, 0..hv])?
            .copy()?
            .reshape([hv])?;

        // 1. decay: M = M * g_t[:, None, None]  (g over Dv, Dk).
        let g_b = g_t.reshape([hv, 1, 1])?;
        m = m.mul(&g_b)?;

        // 2. read on DECAYED M: kv_mem = (M * k_t[:, None, :]).sum(-1) → [Hv, Dv].
        let k_row = k_t.reshape([hv, 1, dk])?;
        let kv_mem = m.mul(&k_row)?.sum(-1, false)?; // [Hv, Dv]

        // 3. delta = (v_t - kv_mem) * beta_t[:, None]  → [Hv, Dv].
        //    sub composed as v_t + kv_mem * (-1).
        let diff = v_t.add(&kv_mem.mul(&neg_one)?)?;
        let beta_b = beta_t.reshape([hv, 1])?;
        let delta = diff.mul(&beta_b)?; // [Hv, Dv]

        // 4. write: M = M + k_t[:, None, :] * delta[:, :, None]  (rank-1 outer).
        //    k_row [Hv,1,Dk] * delta_col [Hv,Dv,1] → [Hv,Dv,Dk].
        let delta_col = delta.reshape([hv, dv, 1])?;
        let outer = k_row.mul(&delta_col)?;
        m = m.add(&outer)?;

        // 5. out on UPDATED M: y_t = (M * q_t[:, None, :]).sum(-1) → [Hv, Dv].
        let q_row = q_t.reshape([hv, 1, dk])?;
        let y_t = m.mul(&q_row)?.sum(-1, false)?; // [Hv, Dv]
        let y_t = y_t.cast(DType::BF16)?.reshape([1usize, 1, hv, dv])?;
        ys.push(y_t);
    }

    let y_refs: Vec<&Tensor> = ys.iter().collect();
    let y = Tensor::concat(&y_refs, 1)?; // [1, T, Hv, Dv] bf16
    let final_state = m.reshape([1usize, hv, dv, dk])?;
    Ok((y, final_state))
}

/// Qwen3.5/3.6 Gated-DeltaNet linear-attention mixer (the 3-in-4 block).
///
/// Owns the four quantized input projections (`in_proj_{qkv,z,a,b}`, all
/// bias-free) and the output projection, plus the dense per-mixer tensors: the
/// depthwise `conv1d_weight`, the f32 `a_log` decay log-rate, the `dt_bias`, and
/// the weighted gated-output `norm` gamma. [`GatedDeltaNet::forward`] composes
/// the tested helpers above: project → causal conv → q/k/v split → weightless
/// q/k norm+scale → GQA repeat → forget/`beta` gates → fp32 delta-rule
/// recurrence → gated output `RMSNorm` (× `silu(z)` in fp32) → `out_proj`.
///
/// The parity gate runs the prefill-from-zero path (`cache = None`): both the
/// conv history and the recurrence state start at zeros.
#[derive(Debug)]
pub struct GatedDeltaNet {
    in_proj_qkv: Linear,
    in_proj_z: Linear,
    in_proj_a: Linear,
    in_proj_b: Linear,
    out_proj: Linear,
    conv1d_weight: Tensor,
    a_log: Tensor,
    dt_bias: Tensor,
    norm: Tensor,
    config: Qwen35Config,
}

impl GatedDeltaNet {
    /// Build from a layer's loaded linear-attention weights and the model
    /// config. The four input projections and `out_proj` are bias-free
    /// `Linear`s; the dense tensors are cloned (cheap refcount bumps on shared
    /// device buffers).
    pub fn from_weights(w: &Qwen35LinearAttnWeights, config: &Qwen35Config) -> Result<Self> {
        Ok(Self {
            in_proj_qkv: Linear::new(w.in_proj_qkv.clone(), None)?,
            in_proj_z: Linear::new(w.in_proj_z.clone(), None)?,
            in_proj_a: Linear::new(w.in_proj_a.clone(), None)?,
            in_proj_b: Linear::new(w.in_proj_b.clone(), None)?,
            out_proj: Linear::new(w.out_proj.clone(), None)?,
            conv1d_weight: w.conv1d_weight.clone(),
            a_log: w.a_log.clone(),
            dt_bias: w.dt_bias.clone(),
            norm: w.norm.clone(),
            config: config.clone(),
        })
    }

    /// Run the Gated-DeltaNet mixer on `hidden` (`[1, T, hidden_size]`, dense
    /// BF16, already input-layernorm'd by the caller). Prefill-from-zero state
    /// (no cache). Returns `[1, T, hidden_size]`, lazy.
    #[allow(clippy::many_single_char_names)]
    pub fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let config = &self.config;
        let hidden_size = config.hidden_size;
        let hk = config.linear_num_key_heads;
        let hv = config.linear_num_value_heads;
        let dk = config.linear_key_head_dim;
        let dv = config.linear_value_head_dim;
        let value_dim = config.value_dim();
        let conv_dim = config.conv_dim();
        let eps = {
            #[allow(clippy::cast_possible_truncation)]
            let e = config.rms_norm_eps as f32;
            e
        };

        let dims = hidden.shape().dims();
        if dims.len() != 3 || dims[0] != 1 {
            return Err(Error::InvalidArgument(format!(
                "GatedDeltaNet::forward: hidden must be rank-3 [1, T, hidden_size]; got {dims:?}",
            )));
        }
        if dims[2] != hidden_size {
            return Err(Error::InvalidArgument(format!(
                "GatedDeltaNet::forward: hidden last dim ({}) != config.hidden_size ({hidden_size})",
                dims[2]
            )));
        }
        if hidden.dtype() != DType::BF16 {
            return Err(Error::InvalidArgument(format!(
                "GatedDeltaNet::forward: hidden must be BF16; got {:?}",
                hidden.dtype()
            )));
        }
        let t = dims[1];
        let device = hidden.device().ok_or_else(|| {
            Error::InvalidArgument("GatedDeltaNet::forward: hidden has no device".into())
        })?;

        // 1. Project. All bias-free Linears.
        let qkv = self.in_proj_qkv.forward(hidden)?; // [1, T, conv_dim]
        let z = self
            .in_proj_z
            .forward(hidden)?
            .reshape([1usize, t, hv, dv])?; // [1, T, Hv, Dv]
        let a = self.in_proj_a.forward(hidden)?; // [1, T, Hv]
        let b = self.in_proj_b.forward(hidden)?; // [1, T, Hv]

        // 2. Causal depthwise conv (prefill: zero history).
        let conv_state = Tensor::zeros(device, [1usize, 3, conv_dim], DType::BF16)?;
        let (conv_out, _new_state) = conv1d_causal(&qkv, &conv_state, &self.conv1d_weight)?;

        // 3. Split into per-head q/k/v.
        let (q, k, v) = split_qkv(&conv_out, hk, hv, dk, dv)?;

        // 4. Weightless q/k norm + scale: q ×(1/Dk), k ×(1/√Dk).
        #[allow(clippy::cast_precision_loss)]
        let inv = (dk as f32).powf(-0.5);
        let q = qk_norm_scale(&q, inv * inv)?;
        let k = qk_norm_scale(&k, inv)?;

        // 5. GQA repeat q/k from Hk to Hv heads.
        let q = gqa_repeat(&q, hv)?;
        let k = gqa_repeat(&k, hv)?;

        // 6. Forget gate g and beta (both fp32).
        let g = compute_g(&a, &self.a_log, &self.dt_bias)?;
        let beta = gdn_beta(&b)?;

        // 7. Delta-rule recurrence (prefill: zero state). Pre-cast q/k/v to f32.
        //    `Tensor::cast` can't take a rank-4 *view* (the cross-dtype strided
        //    copy is rank ≤ 3), so materialize each view to a dense leaf first;
        //    the dense cast path is rank-agnostic.
        let to_f32 = |x: &Tensor| -> Result<Tensor> { Ok(x.copy()?.cast(DType::F32)?) };
        let q = to_f32(&q)?;
        let k = to_f32(&k)?;
        let v = to_f32(&v)?;
        let (y, _state) = recurrence(&q, &k, &v, &g, &beta, None)?; // [1, T, Hv, Dv] bf16

        // 8. Gated output norm + out_proj. The gated-output `norm` gamma is
        //    WEIGHTED (loaded as-is). Then `_precise_swiglu` in fp32:
        //    out = silu(z_f32) * nrm_f32, cast back to bf16.
        let nrm = y.copy()?.rms_norm(Some(&self.norm), eps)?; // [1, T, Hv, Dv] bf16
        // `_precise_swiglu` in fp32: out = silu(z_f32) * nrm_f32. Flatten both to
        // rank-2 [T*Hv, Dv] before casting — the cross-dtype strided copy that
        // backs `cast` only supports rank <= 3, and a contiguous flatten keeps
        // the values bit-identical to the [1,T,Hv,Dv] layout.
        let nrm_f32 = nrm.reshape([t * hv, dv])?.copy()?.cast(DType::F32)?;
        let z_f32 = z.copy()?.reshape([t * hv, dv])?.copy()?.cast(DType::F32)?;
        let silu_z = silu(&z_f32)?;
        let out = silu_z
            .mul(&nrm_f32)?
            .cast(DType::BF16)?
            .reshape([1usize, t, value_dim])?;
        self.out_proj.forward(&out)
    }
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

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn split_qkv_maps_columns_to_heads() {
        let device = Device::shared().unwrap();
        // Hk=2, Hv=4, Dk=Dv=4 → key_dim=8, value_dim=16, conv_dim=32.
        let (hk, hv, dk, dv, t) = (2usize, 4usize, 4usize, 4usize, 2usize);
        let key_dim = hk * dk;
        let value_dim = hv * dv;
        let conv_dim = 2 * key_dim + value_dim;
        assert_eq!(conv_dim, 32);

        let data: Vec<bf16> = (0..t * conv_dim)
            .map(|i| bf16::from_f32(i as f32 * 0.25))
            .collect();
        let conv_out = Tensor::from_slice(&device, &data, [1usize, t, conv_dim]).unwrap();

        let (q, k, v) = split_qkv(&conv_out, hk, hv, dk, dv).unwrap();
        q.eval().unwrap();
        k.eval().unwrap();
        v.eval().unwrap();
        assert_eq!(q.shape().dims(), [1, t, hk, dk]);
        assert_eq!(k.shape().dims(), [1, t, hk, dk]);
        assert_eq!(v.shape().dims(), [1, t, hv, dv]);

        let got_q = q.cpu_to_vec::<bf16>().unwrap();
        let got_k = k.cpu_to_vec::<bf16>().unwrap();
        let got_v = v.cpu_to_vec::<bf16>().unwrap();

        // q = first key_dim cols of each row; k = next key_dim; v = tail.
        for tt in 0..t {
            let row = tt * conv_dim;
            for j in 0..key_dim {
                assert_eq!(got_q[tt * key_dim + j], data[row + j], "q mismatch");
                assert_eq!(
                    got_k[tt * key_dim + j],
                    data[row + key_dim + j],
                    "k mismatch"
                );
            }
            for j in 0..value_dim {
                assert_eq!(
                    got_v[tt * value_dim + j],
                    data[row + 2 * key_dim + j],
                    "v mismatch"
                );
            }
        }
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn qk_norm_scale_matches_weightless_rms_reference() {
        let device = Device::shared().unwrap();
        let (h, d) = (2usize, 4usize);
        let scale = 0.125f32;

        let data: Vec<bf16> = (0..h * d)
            .map(|i| bf16::from_f32((i as f32 - 3.0) * 0.5))
            .collect();
        let x = Tensor::from_slice(&device, &data, [1usize, 1, h, d]).unwrap();

        let out = qk_norm_scale(&x, scale).unwrap();
        out.eval().unwrap();
        assert_eq!(out.shape().dims(), [1, 1, h, d]);
        let got = out.cpu_to_vec::<bf16>().unwrap();

        // Host weightless RMSNorm over D per head: x / sqrt(mean(x²)+eps) * scale.
        let xf: Vec<f32> = data.iter().map(|x| x.to_f32()).collect();
        let mut want = vec![0f32; h * d];
        for head in 0..h {
            let row = head * d;
            let ms = (0..d).map(|j| xf[row + j] * xf[row + j]).sum::<f32>() / d as f32;
            let inv = 1.0 / (ms + 1e-6).sqrt();
            for j in 0..d {
                want[row + j] = xf[row + j] * inv * scale;
            }
        }

        let (atol, rtol) = (1e-2f32, 2e-2f32);
        let mut max_abs = 0f32;
        for (g, r) in got.iter().map(|x| x.to_f32()).zip(want.iter()) {
            let abs = (g - r).abs();
            max_abs = max_abs.max(abs);
            assert!(
                abs <= atol + rtol * r.abs(),
                "qk_norm_scale mismatch: got={g}, ref={r}, abs={abs}"
            );
        }
        println!("qk_norm_scale max_abs={max_abs}");
    }

    #[test]
    #[allow(
        clippy::cast_precision_loss,
        clippy::many_single_char_names,
        clippy::needless_range_loop
    )]
    fn recurrence_matches_fp32_reference() {
        let device = Device::shared().unwrap();
        let (t, hv, dk, dv) = (3usize, 2usize, 4usize, 4usize);

        // Deterministic, distinct, non-degenerate f32 inputs.
        // q,k,v [1,T,Hv,D]; g,beta [1,T,Hv] (g,beta in (0,1)).
        let qkv_val = |seed: usize, ti: usize, h: usize, d: usize| -> f32 {
            (((ti * 7 + h * 3 + d + seed) % 11) as f32 - 5.0) * 0.13
        };
        let q_data: Vec<f32> = (0..t)
            .flat_map(|ti| (0..hv).flat_map(move |h| (0..dk).map(move |d| qkv_val(1, ti, h, d))))
            .collect();
        let k_data: Vec<f32> = (0..t)
            .flat_map(|ti| (0..hv).flat_map(move |h| (0..dk).map(move |d| qkv_val(4, ti, h, d))))
            .collect();
        let v_data: Vec<f32> = (0..t)
            .flat_map(|ti| (0..hv).flat_map(move |h| (0..dv).map(move |d| qkv_val(8, ti, h, d))))
            .collect();
        // g in (0,1), varying per (t,h); beta in (0,1), distinct.
        let g_data: Vec<f32> = (0..t * hv).map(|i| 0.3 + ((i % 5) as f32) * 0.12).collect();
        let beta_data: Vec<f32> = (0..t * hv).map(|i| 0.2 + ((i % 4) as f32) * 0.17).collect();

        let q = Tensor::from_slice(&device, &q_data, [1usize, t, hv, dk]).unwrap();
        let k = Tensor::from_slice(&device, &k_data, [1usize, t, hv, dk]).unwrap();
        let v = Tensor::from_slice(&device, &v_data, [1usize, t, hv, dv]).unwrap();
        let g = Tensor::from_slice(&device, &g_data, [1usize, t, hv]).unwrap();
        let beta = Tensor::from_slice(&device, &beta_data, [1usize, t, hv]).unwrap();

        let (y, state) = recurrence(&q, &k, &v, &g, &beta, None).unwrap();
        y.eval().unwrap();
        state.eval().unwrap();
        assert_eq!(y.dtype(), DType::BF16);
        assert_eq!(y.shape().dims(), [1, t, hv, dv]);
        assert_eq!(state.dtype(), DType::F32);
        assert_eq!(state.shape().dims(), [1, hv, dv, dk]);
        let got_y = y.cpu_to_vec::<bf16>().unwrap();
        let got_state = state.cpu_to_vec::<f32>().unwrap();

        // Independent fp32 host oracle: a direct port of _gated_delta_step_ops.
        // Index helpers (row-major).
        let qi = |ti: usize, h: usize, d: usize| q_data[(ti * hv + h) * dk + d];
        let ki = |ti: usize, h: usize, d: usize| k_data[(ti * hv + h) * dk + d];
        let vi = |ti: usize, h: usize, d: usize| v_data[(ti * hv + h) * dv + d];
        let gi = |ti: usize, h: usize| g_data[ti * hv + h];
        let bi = |ti: usize, h: usize| beta_data[ti * hv + h];

        // M[h][dv][dk], init zeros.
        let mut m = vec![vec![vec![0f32; dk]; dv]; hv];
        let mut ref_y = vec![0f32; t * hv * dv];
        for ti in 0..t {
            for h in 0..hv {
                // 1. decay.
                for a in 0..dv {
                    for b in 0..dk {
                        m[h][a][b] *= gi(ti, h);
                    }
                }
                // 2. read on decayed M: kv_mem[dv] = Σ_dk M·k.
                let mut kv_mem = vec![0f32; dv];
                for a in 0..dv {
                    let mut s = 0f32;
                    for b in 0..dk {
                        s += m[h][a][b] * ki(ti, h, b);
                    }
                    kv_mem[a] = s;
                }
                // 3. delta = (v - kv_mem) * beta.
                let mut delta = vec![0f32; dv];
                for a in 0..dv {
                    delta[a] = (vi(ti, h, a) - kv_mem[a]) * bi(ti, h);
                }
                // 4. write: M += k[dk] * delta[dv].
                for a in 0..dv {
                    for b in 0..dk {
                        m[h][a][b] += ki(ti, h, b) * delta[a];
                    }
                }
                // 5. out on updated M: y[dv] = Σ_dk M·q.
                for a in 0..dv {
                    let mut s = 0f32;
                    for b in 0..dk {
                        s += m[h][a][b] * qi(ti, h, b);
                    }
                    ref_y[(ti * hv + h) * dv + a] = s;
                }
            }
        }

        // y: bf16 out of an accumulating recurrence → looser combined bound.
        let (atol, rtol) = (2e-2f32, 2e-2f32);
        let mut max_abs_y = 0f32;
        let mut max_rel_y = 0f32;
        for (g, r) in got_y.iter().map(|x| x.to_f32()).zip(ref_y.iter()) {
            let abs = (g - r).abs();
            max_abs_y = max_abs_y.max(abs);
            if r.abs() > 1e-6 {
                max_rel_y = max_rel_y.max(abs / r.abs());
            }
            assert!(
                abs <= atol + rtol * r.abs(),
                "y mismatch: got={g}, ref={r}, abs={abs}"
            );
        }

        // final state: fp32 throughout → tight tolerance.
        let mut ref_state = vec![0f32; hv * dv * dk];
        for h in 0..hv {
            for a in 0..dv {
                for b in 0..dk {
                    ref_state[(h * dv + a) * dk + b] = m[h][a][b];
                }
            }
        }
        let stol = 1e-4f32;
        let mut max_abs_s = 0f32;
        for (g, r) in got_state.iter().zip(ref_state.iter()) {
            let abs = (g - r).abs();
            max_abs_s = max_abs_s.max(abs);
            assert!(abs <= stol, "state mismatch: got={g}, ref={r}, abs={abs}");
        }
        println!("recurrence y max_abs={max_abs_y} max_rel={max_rel_y} state max_abs={max_abs_s}");
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn gqa_repeat_is_interleaved() {
        let device = Device::shared().unwrap();
        // Hk=2 → Hv=4, D=3. Critical mapping: out heads 2i,2i+1 == in head i.
        let (hk, hv, d, t) = (2usize, 4usize, 3usize, 1usize);
        let data: Vec<bf16> = (0..hk * d)
            .map(|i| bf16::from_f32(i as f32 + 1.0))
            .collect();
        let x = Tensor::from_slice(&device, &data, [1usize, t, hk, d]).unwrap();

        let out = gqa_repeat(&x, hv).unwrap();
        out.eval().unwrap();
        assert_eq!(out.shape().dims(), [1, t, hv, d]);
        let got = out.cpu_to_vec::<bf16>().unwrap();

        // out head hv reads in head hv/factor (factor = 2): heads 0,1 == in 0;
        // heads 2,3 == in 1. Bit-exact (pure data movement).
        let factor = hv / hk;
        for out_h in 0..hv {
            let in_h = out_h / factor;
            for j in 0..d {
                assert_eq!(
                    got[out_h * d + j],
                    data[in_h * d + j],
                    "gqa_repeat interleave: out head {out_h} must equal in head {in_h}",
                );
            }
        }
        // Explicit critical assertions: head 0==head 1==in 0, head 2==head 3==in 1.
        assert_eq!(got[0..d], got[d..2 * d], "out head 0 must equal out head 1");
        assert_eq!(
            got[2 * d..3 * d],
            got[3 * d..4 * d],
            "out head 2 must equal out head 3"
        );
        assert_eq!(got[0..d], data[0..d], "out head 0 must equal in head 0");
        assert_eq!(
            got[2 * d..3 * d],
            data[d..2 * d],
            "out head 2 must equal in head 1"
        );
    }
}
