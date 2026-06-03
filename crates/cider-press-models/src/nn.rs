//! Composed transformer building blocks.
//!
//! Functions here express standard neural-network operations on top
//! of `cider_press_runtime`'s lazy graph — no new kernels, just
//! orderings of existing ones. Each function schedules an op tensor
//! and returns it unevaluated; the caller chooses when to `eval()`.
//!
//! Includes: [`rms_norm`]; [`silu`] and [`gelu`] (exact, `erf`-based
//! — matches `mlx.nn.gelu`); [`embed_tokens`] (gather + dequantize
//! composition over a [`QuantizedWeight`] embedding table); the
//! [`Module`] trait and stateful building blocks — [`Linear`] (a
//! quantized projection with optional dense bias) and [`RmsNormLayer`]
//! (the composed [`rms_norm`] carrying its gamma weight). Additional
//! building blocks land as their underlying ops are added.

use cider_press_runtime::{DType, Device, QuantizedWeight, Tensor};
use half::{bf16, f16};

use crate::error::{Error, Result};

/// `RMSNorm`: `y = x * rsqrt(mean(x², axis=-1) + eps) * gamma`.
///
/// Composed entirely from existing ops — no fused kernel. The
/// roadmap defers `rms_norm.metal` until perf measurement on the
/// composed path identifies a real bottleneck (`docs/ARCHITECTURE.md`).
///
/// Preconditions:
///
/// - `x` is dense, contiguous, and rank ≥ 1 (`hidden_size` is the
///   last axis).
/// - `gamma` is dense, contiguous, and rank-1 with size equal to
///   `x.shape().dims().last()`.
/// - `x` and `gamma` share a [`Device`] and are [`DType::BF16`].
/// - `eps` is finite and non-negative.
///
/// Output has the same shape + dtype as `x`.
///
/// Delegates to the fused [`Tensor::rms_norm`] kernel — one dispatch of
/// MLX's `rms_single_row` (reduced per row in fp32), replacing the former
/// square→mean→rsqrt→mul→mul composition. The fused kernel is what
/// `mlx.nn.RMSNorm` itself uses (`mx.fast.rms_norm`); the composition was a
/// six-dispatch deviation. `Tensor::rms_norm` carries the validation.
pub fn rms_norm(x: &Tensor, gamma: &Tensor, eps: f32) -> Result<Tensor> {
    Ok(x.rms_norm(gamma, eps)?)
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
    let _ = float_activation_device(x, "silu")?;
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
    let device = float_activation_device(x, "gelu")?;
    let dtype = x.dtype();

    // 1/sqrt(2) and 0.5 flow as host-side [1] tensors broadcast against
    // the activation, same pattern rms_norm uses for eps. Composed in
    // input dtype throughout to follow mlx.nn.gelu's algorithmic steps
    // (mlx does not promote to f32 internally). f32/f16 match bit-exactly;
    // bf16 exhibits small drift because mlx writes `x / sqrt(2)` while we
    // reciprocal-multiply by a bf16-rounded `1/sqrt(2)` (no divide op yet),
    // so the bf16 constants differ in the last few bits.
    let inv_sqrt2 = scalar_tensor(device, dtype, std::f32::consts::FRAC_1_SQRT_2)?;
    let half = scalar_tensor(device, dtype, 0.5)?;
    let one = scalar_tensor(device, dtype, 1.0)?;

    let scaled = x.mul(&inv_sqrt2)?;
    let e = scaled.erf()?;
    let inner = e.add(&one)?;
    let half_x = x.mul(&half)?;
    Ok(half_x.mul(&inner)?)
}

fn float_activation_device<'a>(x: &'a Tensor, op: &str) -> Result<&'a Device> {
    let device = x.device().ok_or_else(|| {
        Error::InvalidArgument(format!("{op}: cannot apply to a placeholder (no device)"))
    })?;
    match x.dtype() {
        DType::F32 | DType::F16 | DType::BF16 => Ok(device),
        dtype @ (DType::I32 | DType::U32) => Err(Error::InvalidArgument(format!(
            "{op}: dtype {dtype:?} is not supported (float dtypes only)",
        ))),
    }
}

/// Embedding lookup against a quantized embedding table:
/// `out[t, ..] = dequantize(weight[input_ids[t], ..])`.
///
/// Composed entirely from existing runtime ops — no new kernel. The
/// composition mirrors MLX-LM's quantized embedding path: gather the
/// rows of each component of the quantized triple (packed `w_q` as
/// `U32`, `scales`/`biases` as `BF16`), then run a single
/// `affine_dequantize` over the gathered triple. Lazy throughout —
/// the four constituent ops land in a single `eval()`.
///
/// Preconditions:
///
/// - `input_ids` is a dense, contiguous, rank-1 [`DType::U32`] tensor
///   on the same device as `weight`.
/// - `weight` is shaped `[vocab, hidden]` with quantization config
///   `(group_size=64, bits=4)` (the only instantiation wired so far,
///   matching Qwen2's production config).
///
/// Output is a dense `[input_ids.shape()[0], hidden]` `BF16` tensor.
pub fn embed_tokens(input_ids: &Tensor, weight: &QuantizedWeight) -> Result<Tensor> {
    let q = weight.quantization();
    let (w_q, scales, biases) = weight.components();
    let gathered_w = w_q.gather(input_ids)?;
    let gathered_s = scales.gather(input_ids)?;
    let gathered_b = biases.gather(input_ids)?;
    Ok(Tensor::dequantize_affine(
        &gathered_w,
        &gathered_s,
        &gathered_b,
        q.group_size(),
        q.bits(),
    )?)
}

/// A forward-only transformer building block.
///
/// `forward` schedules ops onto the runtime's lazy graph and returns
/// the output tensor *unevaluated* — the caller decides when to
/// `eval()`. This is the uniform shape that the stateful layers in
/// this crate implement: [`Linear`] and [`RmsNormLayer`] here, then
/// `Attention`, `Mlp`, and `TransformerBlock`,
/// so a model is "just" a composition of `Module::forward` calls.
///
/// Single-input/single-output by design — it covers every Qwen2 layer.
/// Modules that need auxiliary state (attention's `RoPE` offset, the
/// [`cider_press_runtime::KvCache`]) take it through their own typed
/// constructors / methods rather than widening this signature.
pub trait Module {
    /// Schedule this module's computation on `x`, returning the lazy
    /// output tensor.
    fn forward(&self, x: &Tensor) -> Result<Tensor>;
}

/// A quantized linear projection: `y = x @ W^T (+ bias)`.
///
/// Wraps [`Tensor::quantized_matmul`] (MLX's `affine_qmv` for the M=1
/// decode path, `affine_qmm_t` for M>1 prefill) plus an optional dense
/// additive bias that broadcasts over the leading dims. Qwen2's Q/K/V
/// projections carry a bias; the O and MLP projections do not.
///
/// Owns its weights (refcounted device buffers — `Clone` is cheap), so
/// a loaded checkpoint can be moved into a tree of modules.
#[derive(Clone, Debug)]
pub struct Linear {
    weight: QuantizedWeight,
    bias: Option<Tensor>,
}

impl Linear {
    /// Construct from a quantized weight and an optional dense bias.
    ///
    /// `weight` is logical `[N, K]`; `forward` maps `[..., K]` to
    /// `[..., N]`. If `bias` is `Some`, it must be a rank-1 tensor of
    /// length `N` (the weight's output dim) so it broadcasts against the
    /// `[..., N]` projection output. A shape mismatch is rejected here
    /// rather than deferred to `forward`.
    pub fn new(weight: QuantizedWeight, bias: Option<Tensor>) -> Result<Self> {
        if let Some(bias) = &bias {
            let n = weight.shape().dims()[0];
            if bias.rank() != 1 || bias.shape().dims() != [n] {
                return Err(Error::InvalidArgument(format!(
                    "Linear: bias must be rank-1 of size {n} (weight output dim); \
                     got rank {} shape {:?}",
                    bias.rank(),
                    bias.shape().dims(),
                )));
            }
            // The bias adds to the BF16 projection output on the weight's
            // device, so it must match both — otherwise `forward`'s `add`
            // would fail late. Reject the mismatch at construction.
            let weight_device = weight.tensor().device().ok_or_else(|| {
                Error::InvalidArgument("Linear: weight has no device (placeholder?)".into())
            })?;
            let bias_device = bias.device().ok_or_else(|| {
                Error::InvalidArgument("Linear: bias has no device (placeholder?)".into())
            })?;
            if !weight_device.ptr_eq(bias_device) {
                return Err(Error::InvalidArgument(
                    "Linear: bias and weight are on different devices".into(),
                ));
            }
            if bias.dtype() != DType::BF16 {
                return Err(Error::InvalidArgument(format!(
                    "Linear: bias must be BF16 to match the projection output; got {:?}",
                    bias.dtype(),
                )));
            }
        }
        Ok(Self { weight, bias })
    }

    /// The quantized projection weight (logical `[N, K]`).
    #[must_use]
    pub fn weight(&self) -> &QuantizedWeight {
        &self.weight
    }

    /// The dense additive bias, if this projection has one.
    #[must_use]
    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }
}

impl Module for Linear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = x.quantized_matmul(&self.weight, true)?;
        match &self.bias {
            Some(bias) => Ok(y.add(bias)?),
            None => Ok(y),
        }
    }
}

/// `RMSNorm` as a [`Module`]: carries the gamma weight and epsilon and
/// applies the composed [`rms_norm`] on `forward`.
///
/// Owns its gamma weight (refcounted device buffer — `Clone` is cheap).
#[derive(Clone, Debug)]
pub struct RmsNormLayer {
    gamma: Tensor,
    eps: f32,
}

impl RmsNormLayer {
    /// Construct from a gamma weight (rank-1 `[hidden_size]`, validated
    /// against the input at `forward`) and a normalization epsilon.
    #[must_use]
    pub fn new(gamma: Tensor, eps: f32) -> Self {
        Self { gamma, eps }
    }

    /// The norm's gamma (scale) weight.
    #[must_use]
    pub fn gamma(&self) -> &Tensor {
        &self.gamma
    }

    /// The normalization epsilon.
    #[must_use]
    pub fn eps(&self) -> f32 {
        self.eps
    }
}

impl Module for RmsNormLayer {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        rms_norm(x, &self.gamma, self.eps)
    }
}

/// Qwen2-family `SwiGLU` MLP as a [`Module`].
///
/// Owns three bias-free quantized projections (enforced by
/// [`Mlp::new`]). `forward` computes `down(silu(gate(x)) * up(x))` —
/// mirroring MLX's `swiglu(a, b) = silu(a) * b`
/// (`mlx_lm/models/qwen2.py::MLP`).
///
/// Generic over the three [`Linear`]s rather than bound to a specific
/// checkpoint's weight struct, matching how [`Linear`] / [`RmsNormLayer`]
/// stay model-agnostic. The caller binds checkpoint weights into the
/// `Linear`s (e.g. `Linear::new(weights.gate_proj.clone(), None)`).
#[derive(Clone, Debug)]
pub struct Mlp {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl Mlp {
    /// Construct from the three bias-free projections. Rejects any
    /// [`Linear`] carrying an additive bias — Qwen2's `SwiGLU` MLP
    /// projections are bias-free, and mixing in a biased projection
    /// would silently produce a different op.
    pub fn new(gate: Linear, up: Linear, down: Linear) -> Result<Self> {
        for (which, lin) in [("gate", &gate), ("up", &up), ("down", &down)] {
            if lin.bias().is_some() {
                return Err(Error::InvalidArgument(format!(
                    "Mlp::new: {which} must be a bias-free Linear",
                )));
            }
        }
        Ok(Self { gate, up, down })
    }

    /// The gate projection.
    #[must_use]
    pub fn gate(&self) -> &Linear {
        &self.gate
    }

    /// The up projection.
    #[must_use]
    pub fn up(&self) -> &Linear {
        &self.up
    }

    /// The down projection.
    #[must_use]
    pub fn down(&self) -> &Linear {
        &self.down
    }
}

impl Module for Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gated = silu(&self.gate.forward(x)?)?;
        let up = self.up.forward(x)?;
        self.down.forward(&gated.mul(&up)?)
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

/// Build an additive causal attention mask of shape `[T, T]`, BF16.
///
/// `0.0` on/below the diagonal, `-1e4` above. The `-1e4` magnitude
/// keeps the softmax accumulator inside bf16's representable range
/// while still acting as a hard mask (softmax subtracts the max
/// before exp).
///
/// Rank-2 because `sdpa`'s scores tensor is rank-3
/// `[B*H_q, T, T_cache]`; the wired binary path is `g3_*`. Rank-4
/// strided binary is deferred.
pub fn causal_mask(device: &Device, t: usize) -> Result<Tensor> {
    let neg = bf16::from_f32(-1.0e4);
    let zero = bf16::ZERO;
    let mut data = vec![zero; t * t];
    for row in 0..t {
        for col in (row + 1)..t {
            data[row * t + col] = neg;
        }
    }
    Ok(Tensor::from_slice(device, &data, [t, t])?)
}
