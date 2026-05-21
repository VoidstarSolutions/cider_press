//! Qwen2 attention building blocks.
//!
//! Branch 9 lands the first piece: [`rope`], which threads
//! [`Qwen2Config::rope_theta`] and [`Qwen2Config::head_dim`] into
//! `Tensor::rope`. The rest of the attention path (Q/K/V projections,
//! KV-cache write, SDPA, output projection) lands in branches 10–11
//! once their underlying ops do.
//!
//! Lives in the `qwen2` submodule rather than the generic [`crate::nn`]
//! surface because the bind-config-to-op idiom is per-model: Qwen2.5
//! uses a single `rope_theta`, but Llama-3 layers an NTK-aware scaling
//! map on top, and Cohere uses different rotary dim coverage. We keep
//! the model-specific glue close to the model's config struct.
//!
//! Matches MLX's `Qwen2Attention.__call__` rope step in
//! `mlx-lm/models/qwen2.py`.

use cider_press_runtime::Tensor;

use crate::error::{Error, Result};

use super::Qwen2Config;

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
        Error::InvalidArgument("qwen2::rope: input must have rank ≥ 1".into())
    })?;
    if trailing != head_dim {
        return Err(Error::InvalidArgument(format!(
            "qwen2::rope: input's trailing dim ({trailing}) does not match \
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
