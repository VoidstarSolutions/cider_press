//! Composed transformer building blocks.
//!
//! Functions here express standard neural-network operations on top
//! of `cider_press_runtime`'s lazy graph — no new kernels, just
//! orderings of existing ones. Each function schedules an op tensor
//! and returns it unevaluated; the caller chooses when to `eval()`.
//!
//! Status: branch 5 shipped [`rms_norm`]; branch 6 adds [`silu`] and
//! [`gelu`] (exact, `erf`-based — matches `mlx.nn.gelu`). `SwiGLU`,
//! `RoPE`, `SDPA`, and the rest of the Qwen2 layer building blocks
//! land as their underlying ops do.

use cider_press_runtime::{DType, Device, Tensor};
use half::{bf16, f16};

use crate::error::{Error, Result};

/// `RMSNorm`: `y = x * rsqrt(mean(x², axis=-1) + eps) * gamma`.
///
/// Composed entirely from existing ops — no fused kernel. The
/// roadmap defers `rms_norm.metal` until perf measurement on the
/// composed path identifies a real bottleneck (`docs/QWEN_PATH.md`).
///
/// Preconditions:
///
/// - `x` is dense, contiguous, and rank ≥ 1 (`hidden_size` is the
///   last axis).
/// - `gamma` is dense, contiguous, and rank-1 with size equal to
///   `x.shape().dims().last()`.
/// - `x` and `gamma` share a [`Device`] and a [`DType`] (one of f32,
///   f16, bf16).
/// - `eps` is finite and non-negative.
///
/// Output has the same shape + dtype as `x`.
///
/// `eps` flows through the graph as a host-side `[1]`-tensor on the
/// same device + dtype; the broadcast-add path handles the rank
/// alignment. Skipping the scalar-dispatch sv_/vs_ MLX path is
/// deliberate — at this scale the perf delta is essentially zero
/// (still one `MTLBuffer` alloc + one host write + one dispatch either
/// way), and the synthetic `[1]` tensor keeps the graph free of
/// scalar-binding surface area we don't yet need.
pub fn rms_norm(x: &Tensor, gamma: &Tensor, eps: f32) -> Result<Tensor> {
    if x.is_placeholder() || gamma.is_placeholder() {
        return Err(Error::InvalidArgument(
            "rms_norm: cannot apply to a placeholder (no device)".into(),
        ));
    }
    let device = x.device().expect("checked above");
    let gamma_device = gamma.device().expect("checked above");
    if !device.ptr_eq(gamma_device) {
        return Err(Error::InvalidArgument(
            "rms_norm: x and gamma are on different devices".into(),
        ));
    }
    if x.dtype() != gamma.dtype() {
        return Err(Error::InvalidArgument(format!(
            "rms_norm: dtype mismatch (x={:?}, gamma={:?})",
            x.dtype(),
            gamma.dtype(),
        )));
    }
    if x.rank() == 0 {
        return Err(Error::InvalidArgument(
            "rms_norm: x must have rank ≥ 1 (hidden size is the last axis)".into(),
        ));
    }
    let hidden = *x.shape().dims().last().expect("rank ≥ 1");
    if gamma.rank() != 1 || gamma.shape().dims() != [hidden] {
        return Err(Error::InvalidArgument(format!(
            "rms_norm: gamma must be rank-1 of size {hidden} (got rank {} shape {:?})",
            gamma.rank(),
            gamma.shape().dims(),
        )));
    }
    if !eps.is_finite() || eps < 0.0 {
        return Err(Error::InvalidArgument(format!(
            "rms_norm: eps must be finite and non-negative (got {eps})",
        )));
    }

    // mean(x², axis=-1, keep_dim=true) — shape [..., 1].
    let squared = x.square()?;
    let mean_sq = squared.mean(-1, true)?;

    // + eps (broadcast a [1] scalar tensor against the [..., 1] mean).
    let eps_tensor = scalar_tensor(device, x.dtype(), eps)?;
    let shifted = mean_sq.add(&eps_tensor)?;

    // rsqrt(...) — shape [..., 1].
    let inv_rms = shifted.rsqrt()?;

    // x * inv_rms (broadcasts inv_rms across the last axis).
    let normed = x.mul(&inv_rms)?;

    // * gamma (broadcasts gamma over the leading axes).
    Ok(normed.mul(gamma)?)
}

/// `SiLU` (a.k.a. Swish): `y = x * sigmoid(x)`. Element-wise; output
/// has the same shape, dtype, and device as `x`.
///
/// Matches `mlx.nn.silu`. Composed from existing primitives; no fused
/// kernel.
///
/// Precondition: `x` is on a device (not a placeholder) and has a
/// float dtype (`f32`, `f16`, or `bf16`).
pub fn silu(x: &Tensor) -> Result<Tensor> {
    require_float_activation_input(x, "silu")?;
    let sig = x.sigmoid()?;
    Ok(x.mul(&sig)?)
}

/// Exact GELU: `y = 0.5 * x * (1 + erf(x / sqrt(2)))`. Element-wise;
/// output has the same shape, dtype, and device as `x`.
///
/// Matches `mlx.nn.gelu` (the exact / `erf`-based variant; the
/// tanh-approximation variant from older HF checkpoints lands as a
/// separate function when its first consumer arrives).
///
/// Precondition: `x` is on a device (not a placeholder) and has a
/// float dtype.
pub fn gelu(x: &Tensor) -> Result<Tensor> {
    require_float_activation_input(x, "gelu")?;
    let device = x.device().expect("checked above");
    let dtype = x.dtype();

    // 1/sqrt(2) and 0.5 flow as host-side [1] tensors broadcast against
    // the activation, same pattern rms_norm uses for eps. Composed in
    // input dtype throughout to match mlx.nn.gelu exactly (mlx does not
    // promote to f32 internally).
    let inv_sqrt2 = scalar_tensor(device, dtype, std::f32::consts::FRAC_1_SQRT_2)?;
    let half = scalar_tensor(device, dtype, 0.5)?;
    let one = scalar_tensor(device, dtype, 1.0)?;

    let scaled = x.mul(&inv_sqrt2)?;
    let e = scaled.erf()?;
    let inner = e.add(&one)?;
    let half_x = x.mul(&half)?;
    Ok(half_x.mul(&inner)?)
}

fn require_float_activation_input(x: &Tensor, op: &str) -> Result<()> {
    if x.is_placeholder() {
        return Err(Error::InvalidArgument(format!(
            "{op}: cannot apply to a placeholder (no device)",
        )));
    }
    match x.dtype() {
        DType::F32 | DType::F16 | DType::BF16 => Ok(()),
        dtype @ (DType::I32 | DType::U32) => Err(Error::InvalidArgument(format!(
            "{op}: dtype {dtype:?} is not supported (float dtypes only)",
        ))),
    }
}

/// Build a `[1]` host-side tensor on `device` with the given value
/// rounded into `dtype`. Used by [`rms_norm`] to flow `eps` into the
/// graph as a regular leaf rather than wiring scalar-binding kernels.
fn scalar_tensor(device: &Device, dtype: DType, value: f32) -> Result<Tensor> {
    match dtype {
        DType::F32 => Ok(Tensor::from_slice(device, &[value], [1])?),
        DType::F16 => Ok(Tensor::from_slice(device, &[f16::from_f32(value)], [1])?),
        DType::BF16 => Ok(Tensor::from_slice(device, &[bf16::from_f32(value)], [1])?),
        DType::I32 | DType::U32 => Err(Error::InvalidArgument(format!(
            "scalar_tensor: integer dtype {dtype:?} is not supported here",
        ))),
    }
}
