// Dispatch routine derived from MLX (https://github.com/ml-explore/mlx).
// Copyright © 2023-2024 Apple Inc. Released under the MIT license — see
// the full upstream notice at `kernels-mlx/COPYING`.

//! Fused last-axis `RMSNorm` over bf16: `out = w * x * rsqrt(mean(x²) + eps)`,
//! reduced per row in fp32. Wraps MLX's `rms_single_row` (`rms_norm.metal`),
//! collapsing the composed square→mean→rsqrt→mul→mul chain into one
//! dispatch. Mirrors `mlx/backend/metal/normalization.cpp`'s `RMSNorm::eval_gpu`.

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
/// MLX `RMS_LOOPED_LIMIT` — axes wider than this need the looped kernel,
/// which this dispatch does not (yet) implement. Qwen2.5-0.5B's hidden
/// size (896) is well within the single-row regime.
const LOOPED_LIMIT: usize = 4096;

/// Fused `RMSNorm` over the last (contiguous) axis of a row-major
/// `[n_rows, axis_size]` bf16 input, scaled by a rank-1 `[axis_size]`
/// weight. Single dispatch of MLX's `rms_single_row`; one threadgroup per
/// row.
///
/// `x` and `out` are `n_rows * axis_size` long; `w` is `axis_size` long.
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
) -> Result<()> {
    if axis_size == 0 || n_rows == 0 {
        return Err(Error::InvalidArgument(
            "rms_norm_bf16: axis_size and n_rows must be positive".to_owned(),
        ));
    }
    if axis_size > LOOPED_LIMIT {
        return Err(Error::InvalidArgument(format!(
            "rms_norm_bf16: axis_size {axis_size} exceeds the single-row limit \
             ({LOOPED_LIMIT}); the looped kernel is not implemented"
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
    if w.len() != axis_size {
        return Err(Error::InvalidArgument(format!(
            "rms_norm_bf16: w.len() ({}) != axis_size ({axis_size})",
            w.len()
        )));
    }

    let pipeline = library.pipeline("rmsbfloat16")?;
    let encoder = commands.encoder()?;
    encoder.setComputePipelineState(pipeline.metal_pipeline_state());

    let axis_size_u32 = u32::try_from(axis_size).map_err(|_| {
        Error::InvalidArgument(format!(
            "rms_norm_bf16: axis_size {axis_size} overflows u32"
        ))
    })?;
    // Contiguous rank-1 weight ⇒ unit stride (matches MLX's
    // `(w.ndim() == 1) ? w.strides()[0] : 0`).
    let w_stride: u32 = 1;

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

    // One threadgroup per row, sized to cover the axis at N_READS/thread,
    // rounded up to a simd multiple. `dispatch_threads` grid is total
    // threads (= n_rows × threadgroup_size). Mirrors RMSNorm::eval_gpu.
    let threadgroup_needed = axis_size.div_ceil(N_READS);
    let threadgroup_size = threadgroup_needed.div_ceil(32) * 32;
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
    encoder.dispatchThreads_threadsPerThreadgroup(grid, threadgroup);
    Ok(())
}
