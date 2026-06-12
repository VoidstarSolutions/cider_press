//! Qwen2 attention building blocks.
//!
//! Contains [`rope`], [`sdpa`] (a single fused [`Tensor::sdpa`] call,
//! routed by query length to the vector decode or steel `sdpa_full`
//! prefill kernel), and
//! [`Attention`], which folds the per-head projection, [`rope`], the
//! [`cider_press_runtime::KvCache`], and [`sdpa`] into a single
//! `forward`. `rope` and `sdpa` remain free helpers (reused by
//! `Attention` and future models).
//!
//! Lives in the `qwen2` submodule rather than the generic [`crate::nn`]
//! surface because the bind-config-to-op idiom is per-model: Qwen2.5
//! uses a single `rope_theta`, but Llama-3 layers an NTK-aware scaling
//! map on top, and Cohere uses different rotary dim coverage. We keep
//! the model-specific glue close to the model's config struct.
//!
//! Matches MLX's `Qwen2Attention.__call__` rope step in
//! `mlx-lm/models/qwen2.py`.

use cider_press_runtime::{DType, KvCache, Tensor};

use crate::error::{Error, Result};
use crate::nn::{Linear, Module, PhasedLinear};

use super::{Qwen2AttentionWeights, Qwen2Config};

/// Apply rotary positional embedding to a Qwen2 attention projection
/// (Q or K) at sequence position `offset`.
///
/// Thin wrapper that pulls `rope_theta` and `head_dim` from `config`
/// and routes to `Tensor::rope` with Qwen2's `traditional=false,
/// scale=1.0` defaults. Call once for Q and once for K (they share
/// `rope_theta` and the per-head dimension, only the head count
/// differs).
///
/// Preconditions inherited from [`Tensor::rope`]: `x` must be dense
/// contiguous BF16 of rank 4 (`[B, H, T, D]`) where `D == head_dim`;
/// `offset` must be a length-1 dense contiguous I32 tensor on the
/// same device.
pub fn rope(x: &Tensor, offset: &Tensor, config: &Qwen2Config) -> Result<Tensor> {
    let head_dim = config.head_dim()?;
    let trailing = *x.shape().dims().last().ok_or_else(|| {
        Error::InvalidArgument("qwen2::attention::rope: input must have rank ≥ 1".into())
    })?;
    if trailing != head_dim {
        return Err(Error::InvalidArgument(format!(
            "qwen2::attention::rope: input's trailing dim ({trailing}) does not match \
             config.head_dim() ({head_dim})",
        )));
    }
    // Qwen2 uses standard non-traditional RoPE with no position scaling;
    // `mx.fast.rope`'s `scale` argument defaults to 1.0 for this family.
    // `rope_theta` is stored f64 in the config (HF's `config.json` is
    // JSON-numeric); rounding to f32 here is safe because MLX itself
    // converts via `std::log2(double)` → bound to the kernel's `float`
    // base parameter.
    #[allow(clippy::cast_possible_truncation)]
    let theta = config.rope_theta as f32;
    Ok(x.rope(offset, theta, 1.0, head_dim)?)
}

/// Scaled dot-product attention for Qwen2.5.
///
/// Makes a single [`Tensor::sdpa`] call; the runtime routes by query
/// length: `T_q == 1` → `sdpa_vector` (vector decode, GQA in-kernel,
/// K/V read strided in place), `T_q > 1` → steel `sdpa_full` (prefill,
/// causal via the kernel's `do_causal` function constant — no
/// materialized mask). This mirrors MLX's single
/// `ScaledDotProductAttention` primitive.
///
/// Inputs:
/// - `q`: `[B, H_q, T, D_h]`, dense contiguous BF16.
/// - `k`: `[B, H_kv, T_cache, D_h]`, dense contiguous BF16. Typically
///   the [`cider_press_runtime::KvCache::keys_view`] over the
///   populated prefix of the cache.
/// - `v`: `[B, H_kv, T_cache, D_h]`, dense contiguous BF16. Typically
///   the [`cider_press_runtime::KvCache::values_view`].
/// - `config`: pulls `H_q`, `H_kv`, and `D_h` for shape validation
///   and the GQA broadcast ratio.
///
/// Output: `[B, H_q, T, D_h]`, dense contiguous BF16.
///
/// Mirrors `mlx-lm.models.qwen2.Attention.__call__`'s post-RoPE
/// arithmetic, which itself delegates to
/// `mx.fast.scaled_dot_product_attention`.
#[allow(clippy::many_single_char_names)]
pub fn sdpa(q: &Tensor, k: &Tensor, v: &Tensor, config: &Qwen2Config) -> Result<Tensor> {
    let head_dim = config.head_dim()?;
    let h_q = config.num_attention_heads;
    let h_kv = config.num_key_value_heads;
    if h_kv == 0 || h_q % h_kv != 0 {
        return Err(Error::InvalidArgument(format!(
            "qwen2::attention::sdpa: invalid GQA config (H_q={h_q}, H_kv={h_kv})",
        )));
    }
    let t_q = q.shape().dims().get(2).copied().ok_or_else(|| {
        Error::InvalidArgument("qwen2::attention::sdpa: Q must be rank-4 [B, H, T, D_h]".into())
    })?;
    let gqa_factor = h_q / h_kv;
    #[allow(clippy::cast_precision_loss)]
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    // Mirror MLX's single ScaledDotProductAttention primitive: eval routes
    // T_q==1 → sdpa_vector (decode), T_q>1 → steel sdpa_full (prefill,
    // causal via the kernel's do_causal function constant — no materialized
    // mask). Qwen2 prefill is always causal self-attention.
    Ok(Tensor::sdpa(q, k, v, None, scale, gqa_factor, t_q > 1)?)
}

/// Qwen2 multi-head self-attention with grouped-query attention (GQA)
/// and a `RoPE`-positioned KV cache.
///
/// Owns the four projections as [`PhasedLinear`]s (`q`/`k`/`v` carry
/// the dense additive bias; `o` does not) plus a cloned [`Qwen2Config`]
/// so [`Attention::forward`] can drive the [`rope`] / [`sdpa`] helpers.
/// Each `PhasedLinear` holds a prefill `Linear` (checkpoint weights,
/// qmm path) and a decode `Linear` (K-padded twin, qmv fast-path).
///
/// Deliberately does **not** implement [`crate::nn::Module`]: its
/// forward genuinely needs the `RoPE` offset and a mutable [`KvCache`] —
/// mirroring MLX's `Attention.__call__(x, cache)`.
/// [`crate::nn::Module`] is single-input by design; auxiliary state flows
/// through a layer's own methods, and this is that case.
#[derive(Clone, Debug)]
pub struct Attention {
    q_proj: PhasedLinear,
    k_proj: PhasedLinear,
    v_proj: PhasedLinear,
    o_proj: PhasedLinear,
    config: Qwen2Config,
}

impl Attention {
    /// Build from a layer's loaded attention weights and the model
    /// config. Clones the weights (cheap refcount bumps on shared
    /// device buffers) into owned [`Linear`]s.
    pub fn from_weights(weights: &Qwen2AttentionWeights, config: &Qwen2Config) -> Result<Self> {
        Ok(Self {
            q_proj: PhasedLinear::new(
                Linear::new(weights.q_proj.clone(), Some(weights.q_bias.clone()))?,
                Linear::new(weights.q_proj_padded.clone(), Some(weights.q_bias.clone()))?,
            ),
            k_proj: PhasedLinear::new(
                Linear::new(weights.k_proj.clone(), Some(weights.k_bias.clone()))?,
                Linear::new(weights.k_proj_padded.clone(), Some(weights.k_bias.clone()))?,
            ),
            v_proj: PhasedLinear::new(
                Linear::new(weights.v_proj.clone(), Some(weights.v_bias.clone()))?,
                Linear::new(weights.v_proj_padded.clone(), Some(weights.v_bias.clone()))?,
            ),
            o_proj: PhasedLinear::new(
                Linear::new(weights.o_proj.clone(), None)?,
                Linear::new(weights.o_proj_padded.clone(), None)?,
            ),
            config: config.clone(),
        })
    }

    /// Run self-attention on `hidden` (`[1, T, hidden_size]`, dense
    /// BF16).
    ///
    /// - `offset`: length-1 I32 tensor — the number of tokens already
    ///   in `cache` (the `RoPE` position base). Must equal
    ///   `cache.position()` before this call for correct positioning.
    /// - `cache`: the KV cache; `forward` appends this step's K/V.
    ///
    /// Returns `[1, T, hidden_size]`, lazy.
    ///
    /// Batch is fixed at 1 — [`KvCache`] is single-sequence
    /// (`[max_tokens, n_kv_heads, head_dim]`, no batch axis).
    ///
    /// **Aliasing contract:** the returned tensor's graph reads the
    /// cache's backing buffer (the populated prefix). `eval()` it
    /// before the next [`KvCache::update`] on the same cache;
    /// `update` writes the shared slab in place. A decode loop
    /// satisfies this naturally (sampling forces the eval).
    pub fn forward(&self, hidden: &Tensor, offset: &Tensor, cache: &mut KvCache) -> Result<Tensor> {
        let config = &self.config;
        let head_dim = config.head_dim()?;
        let h_q = config.num_attention_heads;
        let h_kv = config.num_key_value_heads;
        let hidden_size = config.hidden_size;

        let dims = hidden.shape().dims();
        if dims.len() != 3 {
            return Err(Error::InvalidArgument(format!(
                "Attention::forward: hidden must be rank 3 [B, T, hidden_size]; got rank {}",
                dims.len()
            )));
        }
        if dims[0] != 1 {
            return Err(Error::InvalidArgument(format!(
                "Attention::forward: batch must be 1 (KvCache is single-sequence); got B={}",
                dims[0]
            )));
        }
        if dims[2] != hidden_size {
            return Err(Error::InvalidArgument(format!(
                "Attention::forward: hidden last dim ({}) != config.hidden_size ({hidden_size})",
                dims[2]
            )));
        }
        if hidden.dtype() != DType::BF16 {
            return Err(Error::InvalidArgument(format!(
                "Attention::forward: hidden must be BF16; got {:?}",
                hidden.dtype()
            )));
        }
        let t = dims[1];

        // Project + lay out per head: [1, T, H*D_h] -> [1, H, T, D_h].
        let q = project_heads(&self.q_proj, hidden, t, h_q, head_dim)?;
        let k = project_heads(&self.k_proj, hidden, t, h_kv, head_dim)?;
        let v = project_heads(&self.v_proj, hidden, t, h_kv, head_dim)?;

        // RoPE on Q and K (V is unrotated).
        let q = rope(&q, offset, config)?;
        let k = rope(&k, offset, config)?;

        // KvCache stores [step_t, n_kv_heads, head_dim]. K/V are
        // [1, H_kv, T, D_h]; permute (2,1,3,0) -> [T, H_kv, D_h, 1] strided
        // cache-write source (D axis unit-stride). No copy: slice_update reads
        // the strides; the trailing unit axis is dropped at dispatch.
        let k_upd = k.permute(&[2, 1, 3, 0])?;
        let v_upd = v.permute(&[2, 1, 3, 0])?;
        cache.update(&k_upd, &v_upd)?;

        // Read the populated prefix. keys_view()/values_view() are
        // [T_cache, H_kv, D_h] (T-outer); sdpa wants [1, H_kv, T_cache,
        // D_h]. Permute (1,0,2) → [H_kv, T_cache, D_h], then add the
        // batch axis.
        let t_cache = cache.position();
        let k_view = cache.keys_view().permute(&[1, 0, 2])?;
        let v_view = cache.values_view().permute(&[1, 0, 2])?;
        // Both decode and prefill feed the D-contiguous strided cache view
        // straight to the kernel (sdpa_vector / steel sdpa_full both read
        // K/V by stride — only the head dim must be contiguous, as MLX's
        // full path also requires). No materialization copy.
        let k_sdpa = k_view.broadcast_to([1usize, h_kv, t_cache, head_dim])?;
        let v_sdpa = v_view.broadcast_to([1usize, h_kv, t_cache, head_dim])?;

        let attn = sdpa(&q, &k_sdpa, &v_sdpa, config)?; // [1, H_q, T, D_h]

        // Collapse heads: [1, H_q, T, D_h] -> [1, T, H_q*D_h], then O proj.
        let collapsed =
            attn.permute(&[0, 2, 1, 3])?
                .copy()?
                .reshape([1usize, t, h_q * head_dim])?;
        self.o_proj.forward(&collapsed)
    }
}

/// Project `hidden` through `proj` and lay the result out per head:
/// `[1, T, H*D_h] -> [1, H, T, D_h]`. Returns a strided view (no copy):
/// the feature axis stays unit-stride, and `RoPE` (Q/K), the steel SDPA,
/// and the strided cache write all consume the permuted strides directly.
fn project_heads(
    proj: &dyn Module,
    hidden: &Tensor,
    t: usize,
    h: usize,
    head_dim: usize,
) -> Result<Tensor> {
    let out = proj.forward(hidden)?; // [1, T, H*D_h]
    Ok(out
        .reshape([1usize, t, h, head_dim])?
        .permute(&[0, 2, 1, 3])?)
}
