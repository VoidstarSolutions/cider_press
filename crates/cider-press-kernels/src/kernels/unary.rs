// Dispatch routine derived from MLX (https://github.com/ml-explore/mlx).
// Copyright © 2023-2024 Apple Inc. Released under the MIT license — see
// the full upstream notice at `kernels-mlx/COPYING`.

//! Element-wise unary dispatch over MLX's `unary.metal`.
//!
//! Exposes the dense same-dtype `v_<Op><in_tname><out_tname>` family
//! (the `N = 1` instantiation of `unary_v`) for `Square`, `Rsqrt`,
//! `Sigmoid`, `Erf`, `Exp`, and `Log` across `f32`, `f16`, and `bf16`.
//!
//! `Square` + `Rsqrt` cover the `rms_norm` composition. `Sigmoid` and
//! `Erf` are the primitives MLX itself uses to compose `silu` and
//! `gelu` in `mlx.nn` — `SiLU` = `x * sigmoid(x)`, exact `GELU` =
//! `0.5 * x * (1 + erf(x / sqrt(2)))`. `Exp` + `Log` are the primitives
//! for the Gated-DeltaNet recurrence — `compute_g` (exp gate) and
//! `softplus` (log). The rest of MLX's unary surface (Sin / Cos /
//! Tanh / …) lands the same way when its first consumer arrives.
//!
//! Higher-rank strided (`gn{1,4}large_`), 2D-large contiguous
//! (`v2_`), and work-per-thread (`vn_`) variants are deferred — the
//! rmsnorm reduction outputs are always dense and small enough to fit
//! the 1D grid + 32-bit element count, so the simpler `v_` path
//! suffices. Add the others when profiling justifies the surface.
//!
//! Analogue of MLX's `unary_op_gpu_inplace`
//! (`backend/metal/unary.cpp`).

use std::ffi::c_void;
use std::ptr::NonNull;

use half::{bf16, f16};
use objc2_metal::{MTLComputeCommandEncoder, MTLSize};

use crate::buffer::Buffer;
use crate::commands::Commands;
use crate::error::{Error, Result};
use crate::library::KernelLibrary;

/// Element-wise `out = x * x` for dense `f32` input.
///
/// `library` must be a [`KernelLibrary::unary`]. Encodes one dispatch
/// into `commands`; the caller flushes via [`Commands::commit_and_wait`].
pub fn v_square_f32(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    src: &Buffer<f32>,
    dst: &mut Buffer<f32>,
) -> Result<()> {
    encode_v(commands, library, "v_Squarefloat32float32", src, dst)
}

/// `f16` analogue of [`v_square_f32`].
pub fn v_square_f16(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    src: &Buffer<f16>,
    dst: &mut Buffer<f16>,
) -> Result<()> {
    encode_v(commands, library, "v_Squarefloat16float16", src, dst)
}

/// `bf16` analogue of [`v_square_f32`].
pub fn v_square_bf16(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    src: &Buffer<bf16>,
    dst: &mut Buffer<bf16>,
) -> Result<()> {
    encode_v(commands, library, "v_Squarebfloat16bfloat16", src, dst)
}

/// Element-wise `out = 1 / sqrt(x)` for dense `f32` input. The Metal
/// implementation maps directly to the GPU's hardware `rsqrt`
/// (`metal::rsqrt`), which is what MLX's `Rsqrt` op uses.
pub fn v_rsqrt_f32(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    src: &Buffer<f32>,
    dst: &mut Buffer<f32>,
) -> Result<()> {
    encode_v(commands, library, "v_Rsqrtfloat32float32", src, dst)
}

/// `f16` analogue of [`v_rsqrt_f32`].
pub fn v_rsqrt_f16(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    src: &Buffer<f16>,
    dst: &mut Buffer<f16>,
) -> Result<()> {
    encode_v(commands, library, "v_Rsqrtfloat16float16", src, dst)
}

/// `bf16` analogue of [`v_rsqrt_f32`].
pub fn v_rsqrt_bf16(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    src: &Buffer<bf16>,
    dst: &mut Buffer<bf16>,
) -> Result<()> {
    encode_v(commands, library, "v_Rsqrtbfloat16bfloat16", src, dst)
}

/// Element-wise `out = 1 / (1 + exp(-x))` for dense `f32` input. Maps
/// to MLX's `Sigmoid` op (`backend/metal/kernels/unary_ops.h`), which
/// uses a sign-aware formulation to avoid overflow on large negative
/// `x`.
pub fn v_sigmoid_f32(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    src: &Buffer<f32>,
    dst: &mut Buffer<f32>,
) -> Result<()> {
    encode_v(commands, library, "v_Sigmoidfloat32float32", src, dst)
}

/// `f16` analogue of [`v_sigmoid_f32`].
pub fn v_sigmoid_f16(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    src: &Buffer<f16>,
    dst: &mut Buffer<f16>,
) -> Result<()> {
    encode_v(commands, library, "v_Sigmoidfloat16float16", src, dst)
}

/// `bf16` analogue of [`v_sigmoid_f32`].
pub fn v_sigmoid_bf16(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    src: &Buffer<bf16>,
    dst: &mut Buffer<bf16>,
) -> Result<()> {
    encode_v(commands, library, "v_Sigmoidbfloat16bfloat16", src, dst)
}

/// Element-wise Gauss error function for dense `f32` input. MLX's
/// `Erf` op routes through a vendored polynomial approximation in
/// `erf.h`; the dispatch slot binding is identical to every other
/// `v_*` kernel here.
pub fn v_erf_f32(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    src: &Buffer<f32>,
    dst: &mut Buffer<f32>,
) -> Result<()> {
    encode_v(commands, library, "v_Erffloat32float32", src, dst)
}

/// `f16` analogue of [`v_erf_f32`].
pub fn v_erf_f16(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    src: &Buffer<f16>,
    dst: &mut Buffer<f16>,
) -> Result<()> {
    encode_v(commands, library, "v_Erffloat16float16", src, dst)
}

/// `bf16` analogue of [`v_erf_f32`].
pub fn v_erf_bf16(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    src: &Buffer<bf16>,
    dst: &mut Buffer<bf16>,
) -> Result<()> {
    encode_v(commands, library, "v_Erfbfloat16bfloat16", src, dst)
}

/// Element-wise `out = exp(x)` for dense `f32` input. Maps to MLX's
/// `Exp` op (`metal::exp`); the dispatch slot binding is identical to
/// every other `v_*` kernel here.
pub fn v_exp_f32(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    src: &Buffer<f32>,
    dst: &mut Buffer<f32>,
) -> Result<()> {
    encode_v(commands, library, "v_Expfloat32float32", src, dst)
}

/// `f16` analogue of [`v_exp_f32`].
pub fn v_exp_f16(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    src: &Buffer<f16>,
    dst: &mut Buffer<f16>,
) -> Result<()> {
    encode_v(commands, library, "v_Expfloat16float16", src, dst)
}

/// `bf16` analogue of [`v_exp_f32`].
pub fn v_exp_bf16(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    src: &Buffer<bf16>,
    dst: &mut Buffer<bf16>,
) -> Result<()> {
    encode_v(commands, library, "v_Expbfloat16bfloat16", src, dst)
}

/// Element-wise natural log `out = ln(x)` for dense `f32` input. Maps
/// to MLX's `Log` op (`metal::log`); the dispatch slot binding is
/// identical to every other `v_*` kernel here.
pub fn v_log_f32(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    src: &Buffer<f32>,
    dst: &mut Buffer<f32>,
) -> Result<()> {
    encode_v(commands, library, "v_Logfloat32float32", src, dst)
}

/// `f16` analogue of [`v_log_f32`].
pub fn v_log_f16(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    src: &Buffer<f16>,
    dst: &mut Buffer<f16>,
) -> Result<()> {
    encode_v(commands, library, "v_Logfloat16float16", src, dst)
}

/// `bf16` analogue of [`v_log_f32`].
pub fn v_log_bf16(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    src: &Buffer<bf16>,
    dst: &mut Buffer<bf16>,
) -> Result<()> {
    encode_v(commands, library, "v_Logbfloat16bfloat16", src, dst)
}

fn encode_v<T>(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    kernel_name: &str,
    src: &Buffer<T>,
    dst: &mut Buffer<T>,
) -> Result<()> {
    let n = dst.len();
    if src.len() != n {
        return Err(Error::InvalidArgument(format!(
            "unary v: element-count mismatch (src={}, dst={n})",
            src.len(),
        )));
    }
    if n == 0 {
        return Ok(());
    }
    let size = u32::try_from(n).map_err(|_| {
        Error::InvalidArgument(format!(
            "unary v: element count {n} does not fit in u32; use the (deferred) v2_ variant",
        ))
    })?;

    let pipeline = library.pipeline(kernel_name)?;

    {
        let encoder = commands.encoder()?;
        encoder.setComputePipelineState(pipeline.metal_pipeline_state());
        // SAFETY: slots / dtypes match `unary_v` in `unary.h` — two
        // device buffers then `constant uint& size`. Buffer typing is
        // enforced by the typed `Buffer<T>` and the call-site `kernel_name`.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(src.metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(dst.metal_buffer()), 0, 1);
            let size_ptr: NonNull<c_void> = NonNull::from(&size).cast();
            encoder.setBytes_length_atIndex(size_ptr, std::mem::size_of::<u32>(), 2);
        }
    }

    // `v_` is the N=1 instantiation — one thread per element.
    let grid = MTLSize {
        width: n,
        height: 1,
        depth: 1,
    };
    let threadgroup = MTLSize {
        width: n.min(1024),
        height: 1,
        depth: 1,
    };
    commands.dispatch_threads(grid, threadgroup)
}
