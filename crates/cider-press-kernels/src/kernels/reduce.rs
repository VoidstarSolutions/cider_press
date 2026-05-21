// Dispatch routine derived from MLX (https://github.com/ml-explore/mlx).
// Copyright © 2023-2024 Apple Inc. Released under the MIT license — see
// the full upstream notice at `kernels-mlx/COPYING`.

//! Reduction dispatch over MLX's `reduce.metal`.
//!
//! Exposes one public function per (op, dtype) for last-axis row
//! reductions:
//!
//! - `row_reduce_{sum,prod,min,max}_{f32,f16,bf16}` — 12 fns.
//!
//! The "cleaner API than mirroring kernel host names" call: every
//! logical reduction is internally two dispatches (an `init_reduce_*`
//! that seeds the output buffer with the op's identity element, then
//! the actual `row_reduce_*` that accumulates). Hiding that split
//! plus the kernel-variant selection inside one Rust fn keeps the
//! caller saying "reduce this tensor along axis -1," not "first
//! dispatch this init kernel, then pick simple vs looped vs small."
//! Mirrors the entry-point structure of MLX's
//! `row_reduce_general_dispatch` (`backend/metal/reduce.cpp`).
//!
//! **Variant scope:** branch 5 wires the `row_reduce_looped`
//! variant only. It's correct for every shape `RMSNorm` and softmax
//! produce (small batch and large prefill alike); it's just less
//! optimal than `row_reduce_simple` when `out_size ≥ 32` and the
//! reduction axis is contiguous. The simple/small/large variants
//! follow the same dispatch transcription pattern and land when
//! perf measurement justifies the surface.
//!
//! **Axis scope:** branch 5 only handles the last-axis case
//! (`axis == src_shape.len() - 1`). Reducing along a non-last axis
//! routes through `col_reduce.metal`, which is deferred. The runtime
//! layer can permute first if a non-last reduction is needed.
//!
//! Mean is not a kernel-level op — at the runtime layer
//! `Tensor::mean` is `sum / N`.

use std::ffi::c_void;
use std::ptr::NonNull;

use half::{bf16, f16};
use objc2_metal::{MTLComputeCommandEncoder, MTLSize};

use crate::buffer::Buffer;
use crate::commands::Commands;
use crate::error::{Error, Result};
use crate::library::KernelLibrary;

// =====================================================================
// Public dispatch fns — one per (op, dtype).
// =====================================================================

/// Sum-reduce `src` along its last axis into `dst`. `src_shape` is the
/// logical (contiguous, row-major) shape of `src`; `dst` must be sized
/// for the product of all dims except the last.
///
/// `library` must be a [`KernelLibrary::reduce`]. Encodes two
/// dispatches (init + reduce) into `commands`; the caller flushes via
/// [`Commands::commit_and_wait`].
pub fn row_reduce_sum_f32(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    src: &Buffer<f32>,
    src_shape: &[usize],
    dst: &mut Buffer<f32>,
) -> Result<()> {
    encode_row_reduce(commands, library, "sum", "float32", src, src_shape, dst)
}

/// `f16` analogue of [`row_reduce_sum_f32`].
pub fn row_reduce_sum_f16(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    src: &Buffer<f16>,
    src_shape: &[usize],
    dst: &mut Buffer<f16>,
) -> Result<()> {
    encode_row_reduce(commands, library, "sum", "float16", src, src_shape, dst)
}

/// `bf16` analogue of [`row_reduce_sum_f32`].
pub fn row_reduce_sum_bf16(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    src: &Buffer<bf16>,
    src_shape: &[usize],
    dst: &mut Buffer<bf16>,
) -> Result<()> {
    encode_row_reduce(commands, library, "sum", "bfloat16", src, src_shape, dst)
}

/// Prod-reduce `src` along its last axis. See [`row_reduce_sum_f32`].
pub fn row_reduce_prod_f32(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    src: &Buffer<f32>,
    src_shape: &[usize],
    dst: &mut Buffer<f32>,
) -> Result<()> {
    encode_row_reduce(commands, library, "prod", "float32", src, src_shape, dst)
}

/// `f16` analogue of [`row_reduce_prod_f32`].
pub fn row_reduce_prod_f16(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    src: &Buffer<f16>,
    src_shape: &[usize],
    dst: &mut Buffer<f16>,
) -> Result<()> {
    encode_row_reduce(commands, library, "prod", "float16", src, src_shape, dst)
}

/// `bf16` analogue of [`row_reduce_prod_f32`].
pub fn row_reduce_prod_bf16(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    src: &Buffer<bf16>,
    src_shape: &[usize],
    dst: &mut Buffer<bf16>,
) -> Result<()> {
    encode_row_reduce(commands, library, "prod", "bfloat16", src, src_shape, dst)
}

/// Min-reduce `src` along its last axis. See [`row_reduce_sum_f32`].
pub fn row_reduce_min_f32(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    src: &Buffer<f32>,
    src_shape: &[usize],
    dst: &mut Buffer<f32>,
) -> Result<()> {
    encode_row_reduce(commands, library, "min", "float32", src, src_shape, dst)
}

/// `f16` analogue of [`row_reduce_min_f32`].
pub fn row_reduce_min_f16(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    src: &Buffer<f16>,
    src_shape: &[usize],
    dst: &mut Buffer<f16>,
) -> Result<()> {
    encode_row_reduce(commands, library, "min", "float16", src, src_shape, dst)
}

/// `bf16` analogue of [`row_reduce_min_f32`].
pub fn row_reduce_min_bf16(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    src: &Buffer<bf16>,
    src_shape: &[usize],
    dst: &mut Buffer<bf16>,
) -> Result<()> {
    encode_row_reduce(commands, library, "min", "bfloat16", src, src_shape, dst)
}

/// Max-reduce `src` along its last axis. See [`row_reduce_sum_f32`].
pub fn row_reduce_max_f32(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    src: &Buffer<f32>,
    src_shape: &[usize],
    dst: &mut Buffer<f32>,
) -> Result<()> {
    encode_row_reduce(commands, library, "max", "float32", src, src_shape, dst)
}

/// `f16` analogue of [`row_reduce_max_f32`].
pub fn row_reduce_max_f16(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    src: &Buffer<f16>,
    src_shape: &[usize],
    dst: &mut Buffer<f16>,
) -> Result<()> {
    encode_row_reduce(commands, library, "max", "float16", src, src_shape, dst)
}

/// `bf16` analogue of [`row_reduce_max_f32`].
pub fn row_reduce_max_bf16(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    src: &Buffer<bf16>,
    src_shape: &[usize],
    dst: &mut Buffer<bf16>,
) -> Result<()> {
    encode_row_reduce(commands, library, "max", "bfloat16", src, src_shape, dst)
}

// =====================================================================
// Encoders
// =====================================================================

/// Last-axis reduction for a contiguous row-major input. Internally
/// issues `init_reduce_<op><tname>` then
/// `row_reduce_looped_1_reduce_<op><tname>`.
fn encode_row_reduce<T>(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    op_name: &str,
    tname: &str,
    src: &Buffer<T>,
    src_shape: &[usize],
    dst: &mut Buffer<T>,
) -> Result<()> {
    if src_shape.is_empty() {
        return Err(Error::InvalidArgument(
            "row reduce: rank-0 (scalar) input is not supported".into(),
        ));
    }
    let row_size = *src_shape.last().expect("rank ≥ 1");
    let out_size: usize = src_shape[..src_shape.len() - 1].iter().product();
    let total_elems: usize = src_shape.iter().product();
    if src.len() != total_elems {
        return Err(Error::InvalidArgument(format!(
            "row reduce: src.len() ({}) != product(src_shape) ({total_elems})",
            src.len(),
        )));
    }
    if dst.len() != out_size {
        return Err(Error::InvalidArgument(format!(
            "row reduce: dst.len() ({}) != out_size ({out_size})",
            dst.len(),
        )));
    }
    if total_elems > i32::MAX as usize {
        return Err(Error::InvalidArgument(format!(
            "row reduce: input size {total_elems} exceeds i32::MAX; \
             the 'large' kernel variant is not yet wired",
        )));
    }
    if out_size == 0 {
        return Ok(()); // no rows to write; dst is zero-length too.
    }

    // 1) Init pass: seed dst with the op's identity element. Runs even
    //    when `row_size == 0` (e.g. shape `[B, 0]`) so empty rows still
    //    get a well-defined identity value rather than stale memory.
    encode_init_reduce(commands, library, op_name, tname, dst, out_size)?;
    if row_size == 0 {
        return Ok(());
    }

    // 2) Reduce pass: row_reduce_looped, reduce_ndim = 0 → host_name
    //    template parameter `dim` is 1 (per MLX's
    //    `get_kernel_reduce_ndim(0)`).
    encode_row_reduce_looped(
        commands, library, op_name, tname, src, src_shape, dst, row_size, out_size,
    )?;
    Ok(())
}

fn encode_init_reduce<T>(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    op_name: &str,
    tname: &str,
    dst: &mut Buffer<T>,
    n: usize,
) -> Result<()> {
    let kernel_name = format!("init_reduce_{op_name}{tname}");
    let pipeline = library.pipeline(&kernel_name)?;
    let encoder = commands.encoder()?;
    encoder.setComputePipelineState(pipeline.metal_pipeline_state());

    // SAFETY: matches `init_reduce` in `reduction/reduce_init.h`:
    // single output buffer at slot 0, one thread per element.
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(dst.metal_buffer()), 0, 0);
    }
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
    encoder.dispatchThreads_threadsPerThreadgroup(grid, threadgroup);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn encode_row_reduce_looped<T>(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    op_name: &str,
    tname: &str,
    src: &Buffer<T>,
    src_shape: &[usize],
    dst: &mut Buffer<T>,
    row_size: usize,
    out_size: usize,
) -> Result<()> {
    // get_kernel_reduce_ndim(0) → 1.
    let kernel_name = format!("row_reduce_looped_1_reduce_{op_name}{tname}");
    let pipeline = library.pipeline(&kernel_name)?;
    let encoder = commands.encoder()?;
    encoder.setComputePipelineState(pipeline.metal_pipeline_state());

    // Non-reduce shape/strides (all axes except the last). Strides are
    // *element* strides for a contiguous row-major input.
    let pre_dims = &src_shape[..src_shape.len() - 1];
    let mut pre_shape_i32: Vec<i32> = pre_dims
        .iter()
        .map(|&d| {
            i32::try_from(d).map_err(|_| {
                Error::InvalidArgument(format!("row reduce: shape dim {d} does not fit in i32"))
            })
        })
        .collect::<Result<_>>()?;
    let mut pre_strides_i64: Vec<i64> = contiguous_strides_with_inner(pre_dims, row_size);
    let mut ndim_i32: i32 = i32::try_from(pre_dims.len())
        .map_err(|_| Error::InvalidArgument("row reduce: pre-rank > i32::MAX".into()))?;
    // Match MLX's "push 0s to avoid encoding empty vectors" trick when
    // the non-reduce part is empty (i.e. the input is a single row).
    if pre_dims.is_empty() {
        pre_shape_i32.push(0);
        pre_strides_i64.push(0);
        ndim_i32 = 0;
    }

    // Reduce-axes shape/strides: reduce_ndim = 0 (we only reduce the
    // row axis, which the kernel handles implicitly via row_size).
    // MLX still binds a single 0-valued element so the
    // `constant int*` / `constant int64_t*` slots are non-empty.
    let reduce_shape_i32: [i32; 1] = [0];
    let reduce_strides_i64: [i64; 1] = [0];
    let reduce_ndim_i32: i32 = 0;

    let row_size_i64 = i64::try_from(row_size).map_err(|_| {
        Error::InvalidArgument(format!(
            "row reduce: row_size {row_size} does not fit in i64"
        ))
    })?;
    let non_row_reductions_i64: i64 = 1; // No axes reduced beyond the row axis itself.

    // SAFETY: bind slots match `row_reduce_looped` in
    // `kernels-mlx/.../reduction/reduce_row.h`.
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(src.metal_buffer()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(dst.metal_buffer()), 0, 1);

        let row_size_ptr: NonNull<c_void> = NonNull::from(&row_size_i64).cast();
        encoder.setBytes_length_atIndex(row_size_ptr, std::mem::size_of::<i64>(), 2);

        let nrr_ptr: NonNull<c_void> = NonNull::from(&non_row_reductions_i64).cast();
        encoder.setBytes_length_atIndex(nrr_ptr, std::mem::size_of::<i64>(), 3);

        let shape_ptr: NonNull<c_void> = NonNull::from(pre_shape_i32.as_slice()).cast();
        encoder.setBytes_length_atIndex(
            shape_ptr,
            std::mem::size_of_val(pre_shape_i32.as_slice()),
            4,
        );

        let strides_ptr: NonNull<c_void> = NonNull::from(pre_strides_i64.as_slice()).cast();
        encoder.setBytes_length_atIndex(
            strides_ptr,
            std::mem::size_of_val(pre_strides_i64.as_slice()),
            5,
        );

        let ndim_ptr: NonNull<c_void> = NonNull::from(&ndim_i32).cast();
        encoder.setBytes_length_atIndex(ndim_ptr, std::mem::size_of::<i32>(), 6);

        let reduce_shape_ptr: NonNull<c_void> = NonNull::from(reduce_shape_i32.as_slice()).cast();
        encoder.setBytes_length_atIndex(
            reduce_shape_ptr,
            std::mem::size_of_val(reduce_shape_i32.as_slice()),
            7,
        );

        let reduce_strides_ptr: NonNull<c_void> =
            NonNull::from(reduce_strides_i64.as_slice()).cast();
        encoder.setBytes_length_atIndex(
            reduce_strides_ptr,
            std::mem::size_of_val(reduce_strides_i64.as_slice()),
            8,
        );

        let reduce_ndim_ptr: NonNull<c_void> = NonNull::from(&reduce_ndim_i32).cast();
        encoder.setBytes_length_atIndex(reduce_ndim_ptr, std::mem::size_of::<i32>(), 9);
    }

    // Grid: (threadgroup_size, out_grid.width, out_grid.height).
    // For our case the output is a flat 1D vector of length out_size
    // — out_grid.width = out_size, out_grid.height = 1. MLX's
    // get_2d_grid_dims would split tall outputs across .height, but
    // 1D is correct for the shapes `RMSNorm` + softmax produce here.
    let threadgroup_size = threadgroup_size_from_row_size(row_size);
    let grid = MTLSize {
        width: threadgroup_size,
        height: out_size,
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

fn contiguous_strides_with_inner(pre_dims: &[usize], row_size: usize) -> Vec<i64> {
    // Build strides for the non-reduce axes assuming the input is
    // row-major contiguous over `pre_dims + [row_size]`. The innermost
    // pre-axis has stride `row_size`; outer axes multiply.
    let mut strides = vec![0i64; pre_dims.len()];
    let mut running: i64 = i64::try_from(row_size).unwrap_or(i64::MAX);
    for i in (0..pre_dims.len()).rev() {
        strides[i] = running;
        let dim = i64::try_from(pre_dims[i]).unwrap_or(i64::MAX);
        running = running.saturating_mul(dim);
    }
    strides
}

/// Port of MLX `threadgroup_size_from_row_size` (`reduce.cpp:223`).
fn threadgroup_size_from_row_size(row_size: usize) -> usize {
    const REDUCE_N_READS: usize = 4;
    if row_size <= 512 {
        return 32;
    }
    if row_size <= 1024 {
        return 128;
    }
    let mut tg = row_size.div_ceil(REDUCE_N_READS);
    tg = tg.div_ceil(32) * 32;
    tg.min(1024)
}
