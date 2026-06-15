//! Qwen3.5 / Qwen3.6 Gated-Attention (the 1-in-4 full-attention mixer).
//!
//! `head_dim` 256, GQA 16 q / 4 kv, per-head query‖gate split, weighted QK-norm
//! (`RMSNorm` over `head_dim`, before RoPE), partial rotary (64 of 256 dims),
//! sigmoid output gate. Phase-2 lands the helpers + mixer; see
//! `docs/inference/models/qwen3.6.md`.

use cider_press_runtime::{DType, Device, Tensor};
use half::bf16;

use crate::error::{Error, Result};
use crate::qwen3_5::config::Qwen35Config;

/// Rotary dims for partial RoPE: `floor(head_dim * partial_rotary_factor)`.
/// For the 4B/27B config that is `256 * 0.25 = 64`.
// Consumed by the `GatedAttention` forward in Task 4; only the tests exercise
// it today.
#[allow(dead_code)]
pub(crate) fn rotary_dims(config: &Qwen35Config) -> usize {
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    let dims = (config.head_dim as f64 * config.rope.partial_rotary_factor) as usize;
    dims
}

/// Partial RoPE over Q or K shaped `[1, H, T, head_dim]`: rotates the leading
/// `rotary_dims` (64 of 256) and passes the rest through. `base = rope_theta`,
/// `scale = 1`, non-traditional (`NeoX`). `offset` is a length-1 I32 tensor.
// Consumed by the `GatedAttention` forward in Task 4; only the tests exercise
// it today.
#[allow(dead_code)]
pub(crate) fn partial_rope(x: &Tensor, offset: &Tensor, config: &Qwen35Config) -> Result<Tensor> {
    #[allow(clippy::cast_possible_truncation)]
    let base = config.rope.rope_theta as f32;
    Ok(x.rope(offset, base, 1.0, rotary_dims(config))?)
}

/// Reshape `[B, H_kv, T_c, D_h]` to `[B, H_q, T_c, D_h]` by fanning the `H_kv`
/// axis out by `ratio = H_q / H_kv`. Zero-copy through `broadcast_to`, then
/// `copy()` to land a contiguous buffer that the matmul kernel can read.
// Wired in Task 4 (GatedAttention forward); only the tests exercise it today.
#[allow(dead_code, clippy::many_single_char_names)]
fn broadcast_kv_heads(
    x: &Tensor,
    b: usize,
    h_kv: usize,
    ratio: usize,
    t_cache: usize,
    head_dim: usize,
    h_q: usize,
) -> Result<Tensor> {
    if ratio == 1 {
        return Ok(x.copy()?);
    }
    let expanded = x
        .reshape([b, h_kv, 1, t_cache, head_dim])?
        .broadcast_to([b, h_kv, ratio, t_cache, head_dim])?
        .copy()?
        .reshape([b, h_q, t_cache, head_dim])?;
    Ok(expanded)
}

/// Composed scaled-dot-product attention for `head_dim` the fused steel kernel
/// doesn't instantiate (e.g. 256). GQA-broadcast K,V to `H_q`, then
/// QKᵀ/scale/(+mask)/softmax/·V. `mask` (if given) is an additive BF16
/// `[T, T_cache]` broadcast over batch·heads. Used for `head_dim`-256 prefill;
/// decode routes to the fused vector kernel via `Tensor::sdpa`.
// Wired in Task 4 (GatedAttention forward); only the tests exercise it today.
// Composed SDPA lineage: ported from qwen2's pre-steel-kernel chain (git
// be74e60); kept because the fused steel sdpa_full is head_dim-64 only and
// gated attention needs head_dim 256 for prefill.
#[allow(dead_code, clippy::many_single_char_names)]
pub(crate) fn composed_sdpa(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    mask: Option<&Tensor>,
) -> Result<Tensor> {
    let q_dims = q.shape().dims();
    let k_dims = k.shape().dims();
    let v_dims = v.shape().dims();
    if q_dims.len() != 4 || k_dims.len() != 4 || v_dims.len() != 4 {
        return Err(Error::InvalidArgument(format!(
            "qwen3_5::attention::composed_sdpa: inputs must be rank-4 [B, H, T, D_h] (got \
             Q={q_dims:?}, K={k_dims:?}, V={v_dims:?})",
        )));
    }
    let (b, h_q, t, head_dim) = (q_dims[0], q_dims[1], q_dims[2], q_dims[3]);
    let h_kv = k_dims[1];
    let t_cache = k_dims[2];
    if h_kv == 0 || h_q % h_kv != 0 {
        return Err(Error::InvalidArgument(format!(
            "qwen3_5::attention::composed_sdpa: invalid GQA shapes (H_q={h_q}, H_kv={h_kv})",
        )));
    }
    if k_dims[0] != b || k_dims[3] != head_dim {
        return Err(Error::InvalidArgument(format!(
            "qwen3_5::attention::composed_sdpa: K shape {k_dims:?} incompatible with Q {q_dims:?}",
        )));
    }
    if v_dims != [b, h_kv, t_cache, head_dim] {
        return Err(Error::InvalidArgument(format!(
            "qwen3_5::attention::composed_sdpa: V shape {v_dims:?} must match K's \
             [B, H_kv, T_cache, D_h] = [{b}, {h_kv}, {t_cache}, {head_dim}]",
        )));
    }
    let ratio = h_q / h_kv;

    // GQA broadcast: fan K, V from H_kv to H_q so head hq reads kv head hq/ratio.
    let k_full = broadcast_kv_heads(k, b, h_kv, ratio, t_cache, head_dim, h_q)?;
    let v_full = broadcast_kv_heads(v, b, h_kv, ratio, t_cache, head_dim, h_q)?;

    // Q @ K^T: permute K_full's last two dims to [B, H_q, D_h, T_cache]; copy
    // materializes the permuted view contiguous for the matmul kernel.
    let k_t = k_full.permute(&[0, 1, 3, 2])?.copy()?;
    let scores = q.matmul(&k_t)?; // [B, H_q, T, T_cache]

    // The scale/mask/softmax stretch only has rank-3 strided-broadcast binary
    // paths, so collapse `[B, H_q, T, T_cache]` to `[B*H_q, T, T_cache]` for
    // these ops and reshape back. The reshape is zero-copy (matmul/softmax
    // outputs are dense contiguous).
    let device = q.device().ok_or_else(|| {
        Error::InvalidArgument("qwen3_5::attention::composed_sdpa: Q has no device".into())
    })?;
    let scores_3d = scores.reshape([b * h_q, t, t_cache])?;

    let scale = Tensor::from_slice(device, &[bf16::from_f32(scale)], [1usize])?;
    let mut pre_softmax = scores_3d.mul(&scale)?;

    if let Some(mask) = mask {
        if mask.dtype() != DType::BF16 {
            return Err(Error::InvalidArgument(format!(
                "qwen3_5::attention::composed_sdpa: mask must be BF16 (got {:?})",
                mask.dtype(),
            )));
        }
        pre_softmax = pre_softmax.add(mask)?;
    }

    let probs_3d = pre_softmax.softmax(true)?;
    let probs = probs_3d.reshape([b, h_q, t, t_cache])?;

    let out = probs.matmul(&v_full)?; // [B, H_q, T, D_h]
    Ok(out)
}

/// Additive causal mask `[t, t_cache]` (BF16): 0 where key j attends to query i
/// (`j <= i + (t_cache - t)`), -inf above. For prefill-from-scratch `t_cache == t`
/// → lower-triangular. Each row keeps at least its diagonal, so no all-masked row.
// Wired in Task 4 (via the sdpa router); only the tests exercise it today.
#[allow(dead_code)]
fn causal_mask(device: &Device, t: usize, t_cache: usize) -> Result<Tensor> {
    let offset = t_cache - t;
    let mut data = Vec::with_capacity(t * t_cache);
    for i in 0..t {
        for j in 0..t_cache {
            data.push(if j <= i + offset {
                bf16::ZERO
            } else {
                bf16::NEG_INFINITY
            });
        }
    }
    Ok(Tensor::from_slice(device, &data, [t, t_cache])?)
}

/// SDPA for the gated-attention mixer: decode (`T_q=1`) → fused vector kernel
/// (`head_dim` 256 instantiated); prefill (`T_q>1`) → composed (steel `sdpa_full`
/// is `head_dim`-64 only). GQA factor and scale derive from the tensor shapes.
// Wired in Task 4 (GatedAttention forward); only the tests exercise it today.
#[allow(dead_code)]
pub(crate) fn sdpa(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let qd = q.shape().dims();
    let kd = k.shape().dims();
    if qd.len() != 4 || kd.len() != 4 {
        return Err(Error::InvalidArgument(format!(
            "qwen3_5::attention::sdpa: q/k must be rank-4 [1,H,T,D]; got q={qd:?}, k={kd:?}"
        )));
    }
    let head_dim = qd[3];
    #[allow(clippy::cast_precision_loss)]
    let scale = (head_dim as f32).powf(-0.5);

    let h_q = qd[1];
    let h_kv = kd[1];
    if h_kv == 0 || h_q % h_kv != 0 {
        return Err(Error::InvalidArgument(format!(
            "qwen3_5::attention::sdpa: invalid GQA shapes (H_q={h_q}, H_kv={h_kv})",
        )));
    }
    let gqa_factor = h_q / h_kv;
    let t_q = qd[2];
    let t_cache = kd[2];

    if t_q == 1 {
        Ok(Tensor::sdpa(q, k, v, None, scale, gqa_factor, false)?)
    } else {
        let device = q.device().ok_or_else(|| {
            Error::InvalidArgument("qwen3_5::attention::sdpa: Q has no device".into())
        })?;
        let mask = causal_mask(device, t_q, t_cache)?;
        composed_sdpa(q, k, v, scale, Some(&mask))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cider_press_runtime::{Device, Tensor};
    use half::bf16;

    use crate::qwen3_5::config::Qwen35Config;

    const CONFIG_JSON: &str = r#"{
        "model_type": "qwen3_5",
        "quantization": { "group_size": 64, "bits": 4, "mode": "affine" },
        "text_config": {
            "hidden_size": 2560, "num_hidden_layers": 4, "num_attention_heads": 16,
            "num_key_value_heads": 4, "head_dim": 256, "intermediate_size": 9216,
            "vocab_size": 248320, "rms_norm_eps": 1e-06, "attention_bias": false,
            "attn_output_gate": true, "full_attention_interval": 4,
            "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128,
            "linear_num_key_heads": 16, "linear_num_value_heads": 32,
            "linear_value_head_dim": 128, "tie_word_embeddings": true,
            "layer_types": ["linear_attention","linear_attention","linear_attention","full_attention"],
            "rope_parameters": { "rope_theta": 10000000, "partial_rotary_factor": 0.25, "mrope_section": [11,11,10] }
        }
    }"#;

    #[test]
    fn rotary_dims_is_quarter_head_dim() {
        let cfg = Qwen35Config::from_json_bytes(CONFIG_JSON.as_bytes()).unwrap();
        assert_eq!(rotary_dims(&cfg), 64); // 256 * 0.25
    }

    #[test]
    fn partial_rope_preserves_shape() {
        let cfg = Qwen35Config::from_json_bytes(CONFIG_JSON.as_bytes()).unwrap();
        let device = Device::shared().unwrap();
        // [1, H=16, T=3, head_dim=256] of zeros is fine for a shape check.
        let n = 16 * 3 * 256;
        let data = vec![bf16::ZERO; n];
        let x = Tensor::from_slice(&device, &data, [1usize, 16, 3, 256]).unwrap();
        let offset = Tensor::from_slice(&device, &[0i32], [1usize]).unwrap();
        let y = partial_rope(&x, &offset, &cfg).unwrap();
        y.eval().unwrap();
        assert_eq!(y.shape().dims(), &[1, 16, 3, 256]);
    }

    #[test]
    #[allow(clippy::many_single_char_names, clippy::cast_precision_loss)]
    fn composed_sdpa_matches_fp32_reference() {
        let device = Device::shared().unwrap();
        // b=1, h_q=2, T=4, D=8; h_kv=1 (GQA ratio 2), t_cache=4.
        let (h_q, h_kv, t, d) = (2usize, 1usize, 4usize, 8usize);
        let ratio = h_q / h_kv;

        let fill = |seed: usize, n: usize| -> Vec<bf16> {
            (0..n)
                .map(|i| bf16::from_f32(((((i + seed) % 7) as f32) - 3.0) * 0.1))
                .collect()
        };
        let q_data = fill(0, h_q * t * d);
        let k_data = fill(2, h_kv * t * d);
        let v_data = fill(5, h_kv * t * d);

        let q = Tensor::from_slice(&device, &q_data, [1usize, h_q, t, d]).unwrap();
        let k = Tensor::from_slice(&device, &k_data, [1usize, h_kv, t, d]).unwrap();
        let v = Tensor::from_slice(&device, &v_data, [1usize, h_kv, t, d]).unwrap();

        let scale = (d as f32).powf(-0.5);
        let mask = causal_mask(&device, t, t).unwrap();
        let got = composed_sdpa(&q, &k, &v, scale, Some(&mask)).unwrap();
        got.eval().unwrap();
        let got = got.cpu_to_vec::<bf16>().unwrap();
        assert_eq!(got.len(), h_q * t * d);

        // Independent fp32 reference.
        let qf: Vec<f32> = q_data.iter().map(|x| x.to_f32()).collect();
        let kf: Vec<f32> = k_data.iter().map(|x| x.to_f32()).collect();
        let vf: Vec<f32> = v_data.iter().map(|x| x.to_f32()).collect();
        let qi = |hq: usize, i: usize, dd: usize| qf[(hq * t + i) * d + dd];
        let ki = |hk: usize, j: usize, dd: usize| kf[(hk * t + j) * d + dd];
        let vi = |hk: usize, j: usize, dd: usize| vf[(hk * t + j) * d + dd];

        let mut reference = vec![0f32; h_q * t * d];
        for hq in 0..h_q {
            let hk = hq / ratio;
            for i in 0..t {
                let mut scores = vec![f32::NEG_INFINITY; t];
                for (j, score) in scores.iter_mut().enumerate().take(i + 1) {
                    let mut s = 0f32;
                    for dd in 0..d {
                        s += qi(hq, i, dd) * ki(hk, j, dd);
                    }
                    *score = scale * s;
                }
                let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
                let sum: f32 = exps.iter().sum();
                for dd in 0..d {
                    let mut o = 0f32;
                    for (j, e) in exps.iter().enumerate() {
                        o += (e / sum) * vi(hk, j, dd);
                    }
                    reference[(hq * t + i) * d + dd] = o;
                }
            }
        }

        let (atol, rtol) = (5e-2f32, 5e-2f32);
        let mut max_abs = 0f32;
        let mut max_rel = 0f32;
        for (g, r) in got.iter().map(|x| x.to_f32()).zip(reference.iter()) {
            let abs = (g - r).abs();
            max_abs = max_abs.max(abs);
            if r.abs() > 1e-6 {
                max_rel = max_rel.max(abs / r.abs());
            }
            assert!(
                abs <= atol + rtol * r.abs(),
                "composed_sdpa mismatch: got={g}, ref={r}, abs={abs}",
            );
        }
        println!("composed_sdpa max_abs={max_abs}, max_rel={max_rel}");
    }
}
