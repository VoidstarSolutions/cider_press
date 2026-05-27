//! Qwen2 attention building blocks.
//!
//! Branch 9 landed [`rope`]. Branch 11 lands [`sdpa`], the three-op
//! SDPA composition (`qk_matmul` → scale + mask + softmax →
//! `attn_matmul`) with Qwen2-specific GQA broadcast of K/V to match
//! Q's head count. Branch 12b lands [`Attention`], which folds the
//! former `project_qkv` / `project_output` helpers into a single
//! `forward` over the [`crate::nn::Linear`] projections plus [`rope`],
//! the [`cider_press_runtime::KvCache`], and [`sdpa`]. `rope` and
//! `sdpa` remain free helpers (reused by `Attention` and future
//! models).
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
use half::bf16;

use crate::error::{Error, Result};
use crate::nn::{Linear, Module};

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

/// Scaled dot-product attention for Qwen2.5, composed from the
/// runtime's primitive ops (no fused SDPA kernel yet — branch 15
/// will revisit if perf demands).
///
/// Composition:
///
/// ```text
/// K_g, V_g = broadcast K/V from H_kv to H_q (GQA fan-out)
/// scores   = Q @ K_g^T
/// probs    = softmax(scale * scores [+ mask], precise=true)
/// out      = probs @ V_g
/// ```
///
/// Inputs:
/// - `q`: `[B, H_q, T, D_h]`, dense contiguous BF16.
/// - `k`: `[B, H_kv, T_cache, D_h]`, dense contiguous BF16. Typically
///   the [`cider_press_runtime::KvCache::keys_view`] over the
///   populated prefix of the cache.
/// - `v`: `[B, H_kv, T_cache, D_h]`, dense contiguous BF16. Typically
///   the [`cider_press_runtime::KvCache::values_view`].
/// - `mask`: optional additive attention mask, broadcastable to
///   `[B, H_q, T, T_cache]`. For decode (T=1, full attention to the
///   cache) pass `None`; for prefill pass the causal `-inf` mask.
/// - `config`: pulls `H_q`, `H_kv`, and `D_h` for shape validation
///   and the GQA broadcast ratio.
///
/// Output: `[B, H_q, T, D_h]`, dense contiguous BF16.
///
/// Mirrors `mlx-lm.models.qwen2.Attention.__call__`'s post-RoPE
/// arithmetic, which itself delegates to
/// `mx.fast.scaled_dot_product_attention`. We compose explicitly so
/// each primitive carries its branch-by-branch parity story
/// forward.
#[allow(clippy::many_single_char_names)]
pub fn sdpa(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    config: &Qwen2Config,
) -> Result<Tensor> {
    let head_dim = config.head_dim()?;
    let h_q = config.num_attention_heads;
    let h_kv = config.num_key_value_heads;
    if h_kv == 0 || h_q % h_kv != 0 {
        return Err(Error::InvalidArgument(format!(
            "qwen2::attention::sdpa: invalid GQA config (H_q={h_q}, H_kv={h_kv})",
        )));
    }
    let ratio = h_q / h_kv;

    let q_dims = q.shape().dims();
    let k_dims = k.shape().dims();
    let v_dims = v.shape().dims();
    if q_dims.len() != 4 || k_dims.len() != 4 || v_dims.len() != 4 {
        return Err(Error::InvalidArgument(format!(
            "qwen2::attention::sdpa: inputs must be rank-4 [B, H, T, D_h] (got Q={q_dims:?}, \
             K={k_dims:?}, V={v_dims:?})",
        )));
    }
    let (b, t) = (q_dims[0], q_dims[2]);
    if q_dims[1] != h_q || q_dims[3] != head_dim {
        return Err(Error::InvalidArgument(format!(
            "qwen2::attention::sdpa: Q shape {q_dims:?} does not match config \
             [B, {h_q}, T, {head_dim}]",
        )));
    }
    let t_cache = k_dims[2];
    if k_dims[0] != b || k_dims[1] != h_kv || k_dims[3] != head_dim {
        return Err(Error::InvalidArgument(format!(
            "qwen2::attention::sdpa: K shape {k_dims:?} does not match \
             [{b}, {h_kv}, T_cache, {head_dim}]",
        )));
    }
    if v_dims != [b, h_kv, t_cache, head_dim] {
        return Err(Error::InvalidArgument(format!(
            "qwen2::attention::sdpa: V shape {v_dims:?} must match K's \
             [B, H_kv, T_cache, D_h] = [{b}, {h_kv}, {t_cache}, {head_dim}]",
        )));
    }

    // GQA broadcast: fan K, V from H_kv to H_q along a fresh axis,
    // then collapse back to the 4D `[B, H_q, T_cache, D_h]` shape.
    // `broadcast_to` is zero-copy; `copy` materializes for the
    // contiguous-only matmul kernel.
    let k_full = broadcast_kv_heads(k, b, h_kv, ratio, t_cache, head_dim, h_q)?;
    let v_full = broadcast_kv_heads(v, b, h_kv, ratio, t_cache, head_dim, h_q)?;

    // Q @ K^T: permute K_full's last two dims to land at
    // [B, H_q, D_h, T_cache]; copy materializes the permuted view
    // into a contiguous buffer for the matmul kernel.
    let k_t = k_full.permute(&[0, 1, 3, 2])?.copy()?;
    let scores = q.matmul(&k_t)?; // [B, H_q, T, T_cache]

    // The scale / mask / softmax stretch broadcasts a `[1]` (or
    // `[T, T_cache]` mask) into the score tensor. Branch 4 wired
    // `g{1,2,3}_*` strided binary paths but not the rank-4 path, so
    // we collapse `[B, H_q, T, T_cache]` to `[B*H_q, T, T_cache]`
    // (rank 3) for the broadcast ops and reshape back. The reshape
    // is zero-copy (matmul / softmax outputs are dense contiguous);
    // the rank-4 strided binary lands in a future branch.
    let device = q
        .device()
        .ok_or_else(|| Error::InvalidArgument("qwen2::attention::sdpa: Q has no device".into()))?;
    let scores_3d = scores.reshape([b * h_q, t, t_cache])?;

    #[allow(clippy::cast_precision_loss)]
    let scale_value = 1.0f32 / (head_dim as f32).sqrt();
    let scale_bf16 = bf16::from_f32(scale_value);
    let scale = Tensor::from_slice(device, &[scale_bf16], [1usize])?;
    let mut pre_softmax = scores_3d.mul(&scale)?;

    if let Some(mask) = mask {
        if mask.dtype() != DType::BF16 {
            return Err(Error::InvalidArgument(format!(
                "qwen2::attention::sdpa: mask must be BF16 (got {:?})",
                mask.dtype(),
            )));
        }
        pre_softmax = pre_softmax.add(mask)?;
    }

    // softmax(precise=true) is the standard choice for bf16 scores
    // (branch-10 default for attention; the float accumulator
    // absorbs simdgroup-sum drift before the bf16 write).
    let probs_3d = pre_softmax.softmax(true)?;
    let probs = probs_3d.reshape([b, h_q, t, t_cache])?;

    let out = probs.matmul(&v_full)?; // [B, H_q, T, D_h]
    Ok(out)
}

/// Qwen2 multi-head self-attention with grouped-query attention (GQA)
/// and a `RoPE`-positioned KV cache.
///
/// Owns the four projections as [`Linear`]s (`q`/`k`/`v` carry the
/// dense additive bias; `o` does not) plus a cloned [`Qwen2Config`]
/// so [`Attention::forward`] can drive the [`rope`] / [`sdpa`] helpers.
///
/// Deliberately does **not** implement [`crate::nn::Module`]: its
/// forward genuinely needs the attention mask, the `RoPE` offset, and a
/// mutable [`KvCache`] — mirroring MLX's `Attention.__call__(x, mask,
/// cache)`. Branch 12a kept `Module` single-input and documented that
/// auxiliary state flows through a layer's own methods; this is that
/// case.
#[derive(Clone, Debug)]
pub struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    config: Qwen2Config,
}

impl Attention {
    /// Build from a layer's loaded attention weights and the model
    /// config. Clones the weights (cheap refcount bumps on shared
    /// device buffers) into owned [`Linear`]s.
    pub fn from_weights(weights: &Qwen2AttentionWeights, config: &Qwen2Config) -> Result<Self> {
        Ok(Self {
            q_proj: Linear::new(weights.q_proj.clone(), Some(weights.q_bias.clone()))?,
            k_proj: Linear::new(weights.k_proj.clone(), Some(weights.k_bias.clone()))?,
            v_proj: Linear::new(weights.v_proj.clone(), Some(weights.v_bias.clone()))?,
            o_proj: Linear::new(weights.o_proj.clone(), None)?,
            config: config.clone(),
        })
    }

    /// Run self-attention on `hidden` (`[1, T, hidden_size]`, dense
    /// BF16).
    ///
    /// - `mask`: optional additive mask broadcastable to
    ///   `[1, H_q, T, T_cache]` (BF16). Pass `None` for full attention.
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
    pub fn forward(
        &self,
        hidden: &Tensor,
        mask: Option<&Tensor>,
        offset: &Tensor,
        cache: &mut KvCache,
    ) -> Result<Tensor> {
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
        // [1, H_kv, T, D_h]; permute (2,1,3,0) -> [T, H_kv, D_h, 1],
        // reshape to [T, H_kv, D_h] (safe: B==1). copy() lands the
        // contiguous bytes update() memcpys from.
        let k_upd = k
            .permute(&[2, 1, 3, 0])?
            .copy()?
            .reshape([t, h_kv, head_dim])?;
        let v_upd = v
            .permute(&[2, 1, 3, 0])?
            .copy()?
            .reshape([t, h_kv, head_dim])?;
        cache.update(&k_upd, &v_upd)?;

        // Read the populated prefix. keys_view()/values_view() are
        // [T_cache, H_kv, D_h] (T-outer). sdpa wants [1, H_kv, T_cache,
        // D_h], so permute the T_cache/H_kv axes (1,0,2) then add the
        // leading batch axis. A plain reshape here would transpose the
        // data incorrectly when H_kv>1 and T_cache>1.
        let t_cache = cache.position();
        let k_sdpa = cache
            .keys_view()
            .permute(&[1, 0, 2])?
            .copy()?
            .reshape([1usize, h_kv, t_cache, head_dim])?;
        let v_sdpa = cache
            .values_view()
            .permute(&[1, 0, 2])?
            .copy()?
            .reshape([1usize, h_kv, t_cache, head_dim])?;

        let attn = sdpa(&q, &k_sdpa, &v_sdpa, mask, config)?; // [1, H_q, T, D_h]

        // Collapse heads: [1, H_q, T, D_h] -> [1, T, H_q*D_h], then O proj.
        let collapsed =
            attn.permute(&[0, 2, 1, 3])?
                .copy()?
                .reshape([1usize, t, h_q * head_dim])?;
        self.o_proj.forward(&collapsed)
    }
}

/// Project `hidden` through `proj` and lay the result out per head:
/// `[1, T, H*D_h] -> [1, H, T, D_h]`. The `copy()` materializes the
/// permuted view into contiguous bytes for the downstream matmul /
/// `RoPE` kernels.
fn project_heads(
    proj: &Linear,
    hidden: &Tensor,
    t: usize,
    h: usize,
    head_dim: usize,
) -> Result<Tensor> {
    let out = proj.forward(hidden)?; // [1, T, H*D_h]
    Ok(out
        .reshape([1usize, t, h, head_dim])?
        .permute(&[0, 2, 1, 3])?
        .copy()?)
}

/// Reshape `[B, H_kv, T_c, D_h]` to `[B, H_q, T_c, D_h]` by fanning
/// the `H_kv` axis out by `ratio = H_q / H_kv`. Zero-copy through
/// `broadcast_to`, then `copy()` to land a contiguous buffer that
/// the matmul kernel can read.
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
