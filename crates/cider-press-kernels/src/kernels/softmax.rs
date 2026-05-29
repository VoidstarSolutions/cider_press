// Dispatch routine derived from MLX (https://github.com/ml-explore/mlx).
// Copyright © 2023-2024 Apple Inc. Released under the MIT license — see
// the full upstream notice at `kernels-mlx/COPYING`.

//! Last-axis softmax dispatch over MLX's `softmax.metal`.
//!
//! Exposes two `bf16` entry points:
//!
//! - [`dispatch_block_softmax_bf16`] — `block_softmax_bfloat16`,
//!   `AccT = bf16`. Matches `mx.softmax(x, precise=False)`.
//! - [`dispatch_block_softmax_precise_bf16`] —
//!   `block_softmax_precise_bfloat16`, `AccT = float`. Matches
//!   `mx.softmax(x, precise=True)`; the standard choice for bf16
//!   attention scores where the running `sum(exp)` benefits from a
//!   float accumulator.
//!
//! Scope cuts (defer until consumer justifies):
//!
//! - The `looped_softmax_*` family (axis > 4096) — Qwen2 attention
//!   scores stay ≤ 4096 in the shapes we target.
//! - `f32` / `f16` dtypes — bf16 is the only inference dtype that
//!   reaches softmax in the current dispatch.
//! - Non-last-axis softmax — softmax is overwhelmingly last-axis;
//!   non-last consumers permute first.
//!
//! Analogue of MLX's `Softmax::eval_gpu` (`backend/metal/softmax.cpp`).

#![allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]

use std::ffi::c_void;
use std::ptr::NonNull;

use half::bf16;
use objc2_metal::{MTLComputeCommandEncoder, MTLSize};

use crate::buffer::Buffer;
use crate::commands::Commands;
use crate::error::{Error, Result};
use crate::library::KernelLibrary;

/// MLX's per-thread fan-out for the softmax kernels (`SOFTMAX_N_READS`
/// in `backend/metal/kernels/defines.h`). Each lane reads / writes
/// `N_READS` consecutive elements per pass; the host needs the same
/// value to compute the threadgroup size.
const N_READS: usize = 4;

/// Apple Silicon's SIMD width — the simdgroup size MLX's softmax
/// kernels reduce within. Hard-coded at 32 in `softmax.h`'s
/// `constexpr int SIMD_SIZE = 32`.
const SIMD_SIZE: usize = 32;

/// Upper bound on `axis_size` for which `block_softmax_*` is the
/// correct variant. From MLX's
/// `constexpr int SOFTMAX_LOOPED_LIMIT = 4096` in
/// `backend/metal/softmax.cpp`. Above this, `looped_softmax_*` is
/// required — not yet wired.
pub const BLOCK_AXIS_LIMIT: usize = 4096;

/// `block_softmax_bfloat16`: numerically-stable softmax along the
/// trailing axis, with the running `max` and `sum(exp)` carried in
/// bf16. Matches `mx.softmax(x, precise=False)`.
///
/// `n_rows` is the product of every dim *except* the trailing softmax
/// axis (i.e. `src.elem_count() / axis_size`); the dispatch fans out
/// one threadgroup per row.
///
/// Precondition: `axis_size <= BLOCK_AXIS_LIMIT`. Above the limit the
/// in-block reduction can't fit the row, and MLX swaps to the looped
/// variant (not wired yet). The check is hard-errored here rather
/// than silently producing wrong output.
pub fn dispatch_block_softmax_bf16(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    src: &Buffer<bf16>,
    dst: &mut Buffer<bf16>,
    axis_size: usize,
    n_rows: usize,
) -> Result<()> {
    encode_block(
        commands,
        library,
        "block_softmax_bfloat16",
        src,
        dst,
        axis_size,
        n_rows,
    )
}

/// `block_softmax_precise_bfloat16`: same as
/// [`dispatch_block_softmax_bf16`] but carries the running `max` and
/// `sum(exp)` in `float`, casting only at the final write. Matches
/// `mx.softmax(x, precise=True)`. The standard choice for bf16
/// attention scores in inference — the bf16 accumulator path can lose
/// precision on the `sum(exp)` when several lanes contribute
/// similarly-large values.
pub fn dispatch_block_softmax_precise_bf16(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    src: &Buffer<bf16>,
    dst: &mut Buffer<bf16>,
    axis_size: usize,
    n_rows: usize,
) -> Result<()> {
    encode_block(
        commands,
        library,
        "block_softmax_precise_bfloat16",
        src,
        dst,
        axis_size,
        n_rows,
    )
}

fn encode_block(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    kernel_name: &str,
    src: &Buffer<bf16>,
    dst: &mut Buffer<bf16>,
    axis_size: usize,
    n_rows: usize,
) -> Result<()> {
    if axis_size == 0 || n_rows == 0 {
        return Ok(());
    }
    if axis_size > BLOCK_AXIS_LIMIT {
        return Err(Error::InvalidArgument(format!(
            "softmax block: axis_size {axis_size} exceeds BLOCK_AXIS_LIMIT={BLOCK_AXIS_LIMIT}; \
             looped variant is not wired yet",
        )));
    }
    let need = axis_size
        .checked_mul(n_rows)
        .ok_or_else(|| Error::InvalidArgument("softmax: row count overflows usize".into()))?;
    if src.len() < need || dst.len() < need {
        return Err(Error::InvalidArgument(format!(
            "softmax: src/dst too small (need {need}, got src={}, dst={})",
            src.len(),
            dst.len(),
        )));
    }
    let axis_size_i32 = i32::try_from(axis_size).map_err(|_| {
        Error::InvalidArgument(format!(
            "softmax: axis_size {axis_size} does not fit in i32"
        ))
    })?;

    // Threadgroup sizing follows MLX's `Softmax::eval_gpu`:
    //   threadgroup_needed = ceil(axis_size / N_READS)
    //   simds_needed       = ceil(threadgroup_needed / SIMD_SIZE)
    //   threadgroup_size   = SIMD_SIZE * simds_needed   (≤ maxTotalThreadsPerThreadgroup)
    //   n_threads          = n_rows * threadgroup_size  (1D grid)
    let threadgroup_needed = axis_size.div_ceil(N_READS);
    let simds_needed = threadgroup_needed.div_ceil(SIMD_SIZE).max(1);
    let threadgroup_size = SIMD_SIZE * simds_needed;
    let n_threads = n_rows
        .checked_mul(threadgroup_size)
        .ok_or_else(|| Error::InvalidArgument("softmax: thread count overflows usize".into()))?;

    let pipeline = library.pipeline(kernel_name)?;
    let encoder = commands.encoder()?;
    encoder.setComputePipelineState(pipeline.metal_pipeline_state());

    // SAFETY: slot / dtype layout matches the `softmax_single_row`
    // kernel signature in softmax.h:
    //   0  device const T* in   (bf16, n_rows * axis_size)
    //   1  device T* out        (bf16, n_rows * axis_size)
    //   2  constant int& axis_size
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(src.metal_buffer()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(dst.metal_buffer()), 0, 1);
        let axis_ptr: NonNull<c_void> = NonNull::from(&axis_size_i32).cast();
        encoder.setBytes_length_atIndex(axis_ptr, std::mem::size_of::<i32>(), 2);
    }

    let grid = MTLSize {
        width: n_threads,
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
