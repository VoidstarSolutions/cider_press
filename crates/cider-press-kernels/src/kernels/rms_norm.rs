// Dispatch routine derived from MLX (https://github.com/ml-explore/mlx).
// Copyright © 2023-2024 Apple Inc. Released under the MIT license — see
// the full upstream notice at `kernels-mlx/COPYING`.

//! Fused last-axis `RMSNorm` over bf16: `out = w * x * rsqrt(mean(x²) + eps)`,
//! reduced per row in fp32. Wraps MLX's `rms_single_row` (axis ≤ 4096,
//! [`rms_norm_bf16`]) and `rms_looped` (axis > 4096, [`rms_norm_looped_bf16`])
//! from `rms_norm.metal`, collapsing the composed square→mean→rsqrt→mul→mul
//! chain into one dispatch. Mirrors `mlx/backend/metal/normalization.cpp`'s
//! `RMSNorm::eval_gpu`.
//!
//! Explained in the inference guide: `docs/inference/rmsnorm.md`.

use half::bf16;
use objc2_metal::{MTLComputeCommandEncoder, MTLSize};
use std::ffi::c_void;
use std::ptr::NonNull;

use crate::buffer::Buffer;
use crate::commands::Commands;
use crate::error::{Error, Result};
use crate::library::KernelLibrary;

/// MLX `RMS_N_READS` — elements summed per thread.
const N_READS: usize = 4;
/// MLX `RMS_LOOPED_LIMIT` — axes wider than this use the looped kernel
/// ([`rms_norm_looped_bf16`]); at or below it, the single-row kernel
/// ([`rms_norm_bf16`]) fits a row in one threadgroup. Qwen2.5-0.5B's
/// hidden size (896) is single-row; Qwen3.6-27B's hidden 5120 is looped.
pub const LOOPED_LIMIT: usize = 4096;
/// Metal SIMD width — the threadgroup size is rounded to a multiple.
const SIMD_SIZE: usize = 32;

/// Fused `RMSNorm` over the last (contiguous) axis of a row-major
/// `[n_rows, axis_size]` bf16 input, scaled by a weight. Single dispatch of
/// MLX's `rms_single_row`; one threadgroup per row.
///
/// `w_stride` is the per-element stride into `w` (matching MLX's
/// `(w.ndim() == 1) ? w.strides()[0] : 0`): pass `1` with a rank-1
/// `[axis_size]` weight for the standard affine gamma, or `0` with a
/// 1-element `[1.0]` scalar for the weightless (`weight=None`) case — every
/// lane then reads `w[0]`. The forward kernel always multiplies by
/// `w[w_stride * i]`; there is no separate weightless code path (`has_w` is
/// consulted only by the unused VJP variants).
///
/// `x` and `out` are `n_rows * axis_size` long; `w` is `axis_size` long when
/// `w_stride == 1`, or `1` long when `w_stride == 0`.
#[allow(clippy::too_many_arguments)]
pub fn rms_norm_bf16(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    x: &Buffer<bf16>,
    w: &Buffer<bf16>,
    out: &mut Buffer<bf16>,
    axis_size: usize,
    n_rows: usize,
    eps: f32,
    w_stride: u32,
) -> Result<()> {
    if axis_size == 0 || n_rows == 0 {
        return Err(Error::InvalidArgument(
            "rms_norm_bf16: axis_size and n_rows must be positive".to_owned(),
        ));
    }
    if axis_size > LOOPED_LIMIT {
        return Err(Error::InvalidArgument(format!(
            "rms_norm_bf16: axis_size {axis_size} exceeds the single-row limit \
             ({LOOPED_LIMIT}); use rms_norm_looped_bf16"
        )));
    }
    let expected = n_rows * axis_size;
    if x.len() != expected {
        return Err(Error::InvalidArgument(format!(
            "rms_norm_bf16: x.len() ({}) != n_rows * axis_size ({expected})",
            x.len()
        )));
    }
    if out.len() != expected {
        return Err(Error::InvalidArgument(format!(
            "rms_norm_bf16: out.len() ({}) != n_rows * axis_size ({expected})",
            out.len()
        )));
    }
    // Only stride 0 (weightless scalar) and 1 (contiguous `[axis_size]`
    // gamma) are valid: the kernel reads `w[w_stride * i]`, so any other
    // stride would index out of bounds while still passing the length check
    // below (`expected_w == axis_size`).
    if w_stride > 1 {
        return Err(Error::InvalidArgument(format!(
            "rms_norm_bf16: w_stride must be 0 (weightless scalar) or 1 \
             (contiguous gamma); got {w_stride}"
        )));
    }
    // Weighted (stride 1): `w` is the `[axis_size]` gamma. Weightless
    // (stride 0): `w` is a 1-element `[1.0]` scalar read by every lane.
    let expected_w = if w_stride == 0 { 1 } else { axis_size };
    if w.len() != expected_w {
        return Err(Error::InvalidArgument(format!(
            "rms_norm_bf16: w.len() ({}) != {expected_w} (w_stride={w_stride})",
            w.len()
        )));
    }

    let pipeline = library.pipeline("rmsbfloat16")?;

    let axis_size_u32 = u32::try_from(axis_size).map_err(|_| {
        Error::InvalidArgument(format!(
            "rms_norm_bf16: axis_size {axis_size} overflows u32"
        ))
    })?;

    {
        let encoder = commands.encoder()?;
        encoder.setComputePipelineState(pipeline.metal_pipeline_state());
        // SAFETY: bind slots/dtypes match `rms_single_row`'s MSL signature
        // (x@0, w@1, out@2, eps float@3, axis_size uint@4, w_stride uint@5).
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(x.metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(w.metal_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(out.metal_buffer()), 0, 2);
            encoder.setBytes_length_atIndex(NonNull::from(&eps).cast::<c_void>(), 4, 3);
            encoder.setBytes_length_atIndex(NonNull::from(&axis_size_u32).cast::<c_void>(), 4, 4);
            encoder.setBytes_length_atIndex(NonNull::from(&w_stride).cast::<c_void>(), 4, 5);
        }
    }

    // One threadgroup per row, sized to cover the axis at N_READS/thread,
    // rounded up to a simd multiple. `dispatch_threads` grid is total
    // threads (= n_rows × threadgroup_size). Mirrors RMSNorm::eval_gpu.
    let threadgroup_needed = axis_size.div_ceil(N_READS);
    let threadgroup_size = threadgroup_needed.div_ceil(SIMD_SIZE) * SIMD_SIZE;
    let grid = MTLSize {
        width: n_rows * threadgroup_size,
        height: 1,
        depth: 1,
    };
    let threadgroup = MTLSize {
        width: threadgroup_size,
        height: 1,
        depth: 1,
    };
    commands.dispatch_threads(grid, threadgroup)
}

/// Fused `RMSNorm` for trailing axes wider than [`LOOPED_LIMIT`] (the
/// single-row kernel's threadgroup can't cover the row). Single dispatch
/// of MLX's `rms_looped`: a fixed-size threadgroup per row loops the axis
/// in `lsize * N_READS` chunks. Bind slots and the per-row layout are
/// identical to [`rms_norm_bf16`] — including the `w_stride` weighted /
/// weightless contract; only the kernel and threadgroup size differ.
/// Qwen3.6-27B's hidden 5120 uses this path.
///
/// `x` and `out` are `n_rows * axis_size` long; `w` is `axis_size` long when
/// `w_stride == 1`, or `1` long when `w_stride == 0`.
#[allow(clippy::too_many_arguments)]
pub fn rms_norm_looped_bf16(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    x: &Buffer<bf16>,
    w: &Buffer<bf16>,
    out: &mut Buffer<bf16>,
    axis_size: usize,
    n_rows: usize,
    eps: f32,
    w_stride: u32,
) -> Result<()> {
    if axis_size == 0 || n_rows == 0 {
        return Err(Error::InvalidArgument(
            "rms_norm_looped_bf16: axis_size and n_rows must be positive".to_owned(),
        ));
    }
    let expected = n_rows * axis_size;
    if x.len() != expected {
        return Err(Error::InvalidArgument(format!(
            "rms_norm_looped_bf16: x.len() ({}) != n_rows * axis_size ({expected})",
            x.len()
        )));
    }
    if out.len() != expected {
        return Err(Error::InvalidArgument(format!(
            "rms_norm_looped_bf16: out.len() ({}) != n_rows * axis_size ({expected})",
            out.len()
        )));
    }
    if w_stride > 1 {
        return Err(Error::InvalidArgument(format!(
            "rms_norm_looped_bf16: w_stride must be 0 (weightless scalar) or 1 \
             (contiguous gamma); got {w_stride}"
        )));
    }
    let expected_w = if w_stride == 0 { 1 } else { axis_size };
    if w.len() != expected_w {
        return Err(Error::InvalidArgument(format!(
            "rms_norm_looped_bf16: w.len() ({}) != {expected_w} (w_stride={w_stride})",
            w.len()
        )));
    }

    let pipeline = library.pipeline("rms_loopedbfloat16")?;

    let axis_size_u32 = u32::try_from(axis_size).map_err(|_| {
        Error::InvalidArgument(format!(
            "rms_norm_looped_bf16: axis_size {axis_size} overflows u32"
        ))
    })?;

    {
        let encoder = commands.encoder()?;
        encoder.setComputePipelineState(pipeline.metal_pipeline_state());
        // SAFETY: bind slots/dtypes match `rms_looped`'s MSL signature —
        // identical to `rms_single_row` (x@0, w@1, out@2, eps float@3,
        // axis_size uint@4, w_stride uint@5); the kernel's `lsize` comes
        // from the threadgroup dims, not a bound buffer.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(x.metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(w.metal_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(out.metal_buffer()), 0, 2);
            encoder.setBytes_length_atIndex(NonNull::from(&eps).cast::<c_void>(), 4, 3);
            encoder.setBytes_length_atIndex(NonNull::from(&axis_size_u32).cast::<c_void>(), 4, 4);
            encoder.setBytes_length_atIndex(NonNull::from(&w_stride).cast::<c_void>(), 4, 5);
        }
    }

    // One threadgroup per row; the kernel loops the axis. MLX sizes the looped
    // threadgroup at the pipeline's max threads. The simd-rounded need for any
    // axis > 4096 (the only case routed here) already exceeds that max, so the
    // `.min()` below always selects `max_threads` — matching MLX exactly. The
    // computed value is kept only to mirror MLX's intent explicitly.
    let max_threads = pipeline.max_total_threads_per_threadgroup();
    let threadgroup_needed = axis_size.div_ceil(N_READS);
    let computed = threadgroup_needed.div_ceil(SIMD_SIZE) * SIMD_SIZE;
    let threadgroup_size = computed.min(max_threads);
    let grid = MTLSize {
        width: n_rows * threadgroup_size,
        height: 1,
        depth: 1,
    };
    let threadgroup = MTLSize {
        width: threadgroup_size,
        height: 1,
        depth: 1,
    };
    commands.dispatch_threads(grid, threadgroup)
}
