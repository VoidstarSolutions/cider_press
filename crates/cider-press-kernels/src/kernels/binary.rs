// Dispatch routine derived from MLX (https://github.com/ml-explore/mlx).
// Copyright © 2023-2024 Apple Inc. Released under the MIT license — see
// the full upstream notice at `kernels-mlx/COPYING`.

//! Element-wise binary dispatch over MLX's `binary.metal`.
//!
//! Exposes two families for `Add` and `Multiply` across `f32`, `f16`,
//! and `bf16`:
//!
//! - `vv_*` — `BinaryOpType::VectorVector`: both inputs are dense and
//!   the same shape as the output. 1D grid over `out.len()`. Maps to
//!   MLX's `vv_<Op><dtype>` (`N = 1` instantiation; no work-per-thread
//!   tiling).
//! - `g1_/g2_/g3_*` — `BinaryOpType::General` for output rank 1/2/3:
//!   inputs are strided (typically because one side is a broadcast
//!   view, e.g. a per-channel bias). Maps to MLX's `g{n}_<Op><dtype>`
//!   rank-specialised templates.
//!
//! Higher-rank (`gn2_`/`gn4large_`), large-buffer (`g{n}large_`), and
//! scalar fast paths (`sv_`, `vs_`, `ss_`, plus the `*n_`
//! work-per-thread tilings) are deferred — the runtime collapses
//! contiguous dims first and the dropped ranks are unreachable for
//! Qwen2-scale tensors. Add them the same way when profiling justifies
//! the surface area.
//!
//! Output is assumed dense, row-major, fully owned; the strided
//! kernels only walk inputs through `a_strides` / `b_strides` and write
//! `c` at the dense logical index.
//!
//! Analogue of MLX's `binary_op_gpu_inplace` (`backend/metal/binary.cpp`).

use std::ffi::c_void;
use std::ptr::NonNull;

use half::{bf16, f16};
use objc2_metal::{MTLComputeCommandEncoder, MTLSize};

use crate::buffer::Buffer;
use crate::commands::Commands;
use crate::error::{Error, Result};
use crate::library::KernelLibrary;

// =====================================================================
// vv_ — both inputs dense + same shape as output.
// =====================================================================

/// Element-wise `c = a + b` for dense, same-shape `f32` inputs.
///
/// `library` must be a [`KernelLibrary::binary`]. Encodes one dispatch
/// into `commands`; the caller flushes via [`Commands::commit_and_wait`].
pub fn vv_add_f32(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    a: &Buffer<f32>,
    b: &Buffer<f32>,
    c: &mut Buffer<f32>,
) -> Result<()> {
    encode_vv(commands, library, "vv_Addfloat32", a, b, c)
}

/// `f16` analogue of [`vv_add_f32`].
pub fn vv_add_f16(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    a: &Buffer<f16>,
    b: &Buffer<f16>,
    c: &mut Buffer<f16>,
) -> Result<()> {
    encode_vv(commands, library, "vv_Addfloat16", a, b, c)
}

/// `bf16` analogue of [`vv_add_f32`].
pub fn vv_add_bf16(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    a: &Buffer<bf16>,
    b: &Buffer<bf16>,
    c: &mut Buffer<bf16>,
) -> Result<()> {
    encode_vv(commands, library, "vv_Addbfloat16", a, b, c)
}

/// Element-wise `c = a * b` for dense, same-shape `f32` inputs. See
/// [`vv_add_f32`] for semantics.
pub fn vv_mul_f32(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    a: &Buffer<f32>,
    b: &Buffer<f32>,
    c: &mut Buffer<f32>,
) -> Result<()> {
    encode_vv(commands, library, "vv_Multiplyfloat32", a, b, c)
}

/// `f16` analogue of [`vv_mul_f32`].
pub fn vv_mul_f16(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    a: &Buffer<f16>,
    b: &Buffer<f16>,
    c: &mut Buffer<f16>,
) -> Result<()> {
    encode_vv(commands, library, "vv_Multiplyfloat16", a, b, c)
}

/// `bf16` analogue of [`vv_mul_f32`].
pub fn vv_mul_bf16(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    a: &Buffer<bf16>,
    b: &Buffer<bf16>,
    c: &mut Buffer<bf16>,
) -> Result<()> {
    encode_vv(commands, library, "vv_Multiplybfloat16", a, b, c)
}

// =====================================================================
// g1_/g2_/g3_ — strided general path, rank-specialised.
// =====================================================================

/// Rank-1 strided `c = a + b` for `f32`. Output is dense over `shape[0]`;
/// `a_stride` / `b_stride` are *element* strides for the single axis
/// (zero ⇒ broadcast).
#[allow(clippy::too_many_arguments)]
pub fn g1_add_f32(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    a: &Buffer<f32>,
    b: &Buffer<f32>,
    c: &mut Buffer<f32>,
    a_stride: i64,
    b_stride: i64,
    shape: i32,
) -> Result<()> {
    encode_g1(
        commands,
        library,
        "g1_Addfloat32",
        a,
        b,
        c,
        a_stride,
        b_stride,
        shape,
    )
}

/// Rank-1 strided `add`, `f16`. See [`g1_add_f32`].
#[allow(clippy::too_many_arguments)]
pub fn g1_add_f16(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    a: &Buffer<f16>,
    b: &Buffer<f16>,
    c: &mut Buffer<f16>,
    a_stride: i64,
    b_stride: i64,
    shape: i32,
) -> Result<()> {
    encode_g1(
        commands,
        library,
        "g1_Addfloat16",
        a,
        b,
        c,
        a_stride,
        b_stride,
        shape,
    )
}

/// Rank-1 strided `add`, `bf16`. See [`g1_add_f32`].
#[allow(clippy::too_many_arguments)]
pub fn g1_add_bf16(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    a: &Buffer<bf16>,
    b: &Buffer<bf16>,
    c: &mut Buffer<bf16>,
    a_stride: i64,
    b_stride: i64,
    shape: i32,
) -> Result<()> {
    encode_g1(
        commands,
        library,
        "g1_Addbfloat16",
        a,
        b,
        c,
        a_stride,
        b_stride,
        shape,
    )
}

/// Rank-1 strided `mul`, `f32`. See [`g1_add_f32`].
#[allow(clippy::too_many_arguments)]
pub fn g1_mul_f32(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    a: &Buffer<f32>,
    b: &Buffer<f32>,
    c: &mut Buffer<f32>,
    a_stride: i64,
    b_stride: i64,
    shape: i32,
) -> Result<()> {
    encode_g1(
        commands,
        library,
        "g1_Multiplyfloat32",
        a,
        b,
        c,
        a_stride,
        b_stride,
        shape,
    )
}

/// Rank-1 strided `mul`, `f16`. See [`g1_add_f32`].
#[allow(clippy::too_many_arguments)]
pub fn g1_mul_f16(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    a: &Buffer<f16>,
    b: &Buffer<f16>,
    c: &mut Buffer<f16>,
    a_stride: i64,
    b_stride: i64,
    shape: i32,
) -> Result<()> {
    encode_g1(
        commands,
        library,
        "g1_Multiplyfloat16",
        a,
        b,
        c,
        a_stride,
        b_stride,
        shape,
    )
}

/// Rank-1 strided `mul`, `bf16`. See [`g1_add_f32`].
#[allow(clippy::too_many_arguments)]
pub fn g1_mul_bf16(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    a: &Buffer<bf16>,
    b: &Buffer<bf16>,
    c: &mut Buffer<bf16>,
    a_stride: i64,
    b_stride: i64,
    shape: i32,
) -> Result<()> {
    encode_g1(
        commands,
        library,
        "g1_Multiplybfloat16",
        a,
        b,
        c,
        a_stride,
        b_stride,
        shape,
    )
}

/// Rank-2 strided `add`, `f32`. `shape = [outer, inner]`; strides have
/// matching length. Output is dense row-major.
#[allow(clippy::too_many_arguments)]
pub fn g2_add_f32(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    a: &Buffer<f32>,
    b: &Buffer<f32>,
    c: &mut Buffer<f32>,
    a_strides: [i64; 2],
    b_strides: [i64; 2],
    shape: [i32; 2],
) -> Result<()> {
    encode_g2(
        commands,
        library,
        "g2_Addfloat32",
        a,
        b,
        c,
        a_strides,
        b_strides,
        shape,
    )
}

/// Rank-2 strided `add`, `f16`. See [`g2_add_f32`].
#[allow(clippy::too_many_arguments)]
pub fn g2_add_f16(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    a: &Buffer<f16>,
    b: &Buffer<f16>,
    c: &mut Buffer<f16>,
    a_strides: [i64; 2],
    b_strides: [i64; 2],
    shape: [i32; 2],
) -> Result<()> {
    encode_g2(
        commands,
        library,
        "g2_Addfloat16",
        a,
        b,
        c,
        a_strides,
        b_strides,
        shape,
    )
}

/// Rank-2 strided `add`, `bf16`. See [`g2_add_f32`].
#[allow(clippy::too_many_arguments)]
pub fn g2_add_bf16(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    a: &Buffer<bf16>,
    b: &Buffer<bf16>,
    c: &mut Buffer<bf16>,
    a_strides: [i64; 2],
    b_strides: [i64; 2],
    shape: [i32; 2],
) -> Result<()> {
    encode_g2(
        commands,
        library,
        "g2_Addbfloat16",
        a,
        b,
        c,
        a_strides,
        b_strides,
        shape,
    )
}

/// Rank-2 strided `mul`, `f32`. See [`g2_add_f32`].
#[allow(clippy::too_many_arguments)]
pub fn g2_mul_f32(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    a: &Buffer<f32>,
    b: &Buffer<f32>,
    c: &mut Buffer<f32>,
    a_strides: [i64; 2],
    b_strides: [i64; 2],
    shape: [i32; 2],
) -> Result<()> {
    encode_g2(
        commands,
        library,
        "g2_Multiplyfloat32",
        a,
        b,
        c,
        a_strides,
        b_strides,
        shape,
    )
}

/// Rank-2 strided `mul`, `f16`. See [`g2_add_f32`].
#[allow(clippy::too_many_arguments)]
pub fn g2_mul_f16(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    a: &Buffer<f16>,
    b: &Buffer<f16>,
    c: &mut Buffer<f16>,
    a_strides: [i64; 2],
    b_strides: [i64; 2],
    shape: [i32; 2],
) -> Result<()> {
    encode_g2(
        commands,
        library,
        "g2_Multiplyfloat16",
        a,
        b,
        c,
        a_strides,
        b_strides,
        shape,
    )
}

/// Rank-2 strided `mul`, `bf16`. See [`g2_add_f32`].
#[allow(clippy::too_many_arguments)]
pub fn g2_mul_bf16(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    a: &Buffer<bf16>,
    b: &Buffer<bf16>,
    c: &mut Buffer<bf16>,
    a_strides: [i64; 2],
    b_strides: [i64; 2],
    shape: [i32; 2],
) -> Result<()> {
    encode_g2(
        commands,
        library,
        "g2_Multiplybfloat16",
        a,
        b,
        c,
        a_strides,
        b_strides,
        shape,
    )
}

/// Rank-3 strided `add`, `f32`. `shape = [outer, mid, inner]`.
#[allow(clippy::too_many_arguments)]
pub fn g3_add_f32(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    a: &Buffer<f32>,
    b: &Buffer<f32>,
    c: &mut Buffer<f32>,
    a_strides: [i64; 3],
    b_strides: [i64; 3],
    shape: [i32; 3],
) -> Result<()> {
    encode_g3(
        commands,
        library,
        "g3_Addfloat32",
        a,
        b,
        c,
        a_strides,
        b_strides,
        shape,
    )
}

/// Rank-3 strided `add`, `f16`. See [`g3_add_f32`].
#[allow(clippy::too_many_arguments)]
pub fn g3_add_f16(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    a: &Buffer<f16>,
    b: &Buffer<f16>,
    c: &mut Buffer<f16>,
    a_strides: [i64; 3],
    b_strides: [i64; 3],
    shape: [i32; 3],
) -> Result<()> {
    encode_g3(
        commands,
        library,
        "g3_Addfloat16",
        a,
        b,
        c,
        a_strides,
        b_strides,
        shape,
    )
}

/// Rank-3 strided `add`, `bf16`. See [`g3_add_f32`].
#[allow(clippy::too_many_arguments)]
pub fn g3_add_bf16(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    a: &Buffer<bf16>,
    b: &Buffer<bf16>,
    c: &mut Buffer<bf16>,
    a_strides: [i64; 3],
    b_strides: [i64; 3],
    shape: [i32; 3],
) -> Result<()> {
    encode_g3(
        commands,
        library,
        "g3_Addbfloat16",
        a,
        b,
        c,
        a_strides,
        b_strides,
        shape,
    )
}

/// Rank-3 strided `mul`, `f32`. See [`g3_add_f32`].
#[allow(clippy::too_many_arguments)]
pub fn g3_mul_f32(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    a: &Buffer<f32>,
    b: &Buffer<f32>,
    c: &mut Buffer<f32>,
    a_strides: [i64; 3],
    b_strides: [i64; 3],
    shape: [i32; 3],
) -> Result<()> {
    encode_g3(
        commands,
        library,
        "g3_Multiplyfloat32",
        a,
        b,
        c,
        a_strides,
        b_strides,
        shape,
    )
}

/// Rank-3 strided `mul`, `f16`. See [`g3_add_f32`].
#[allow(clippy::too_many_arguments)]
pub fn g3_mul_f16(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    a: &Buffer<f16>,
    b: &Buffer<f16>,
    c: &mut Buffer<f16>,
    a_strides: [i64; 3],
    b_strides: [i64; 3],
    shape: [i32; 3],
) -> Result<()> {
    encode_g3(
        commands,
        library,
        "g3_Multiplyfloat16",
        a,
        b,
        c,
        a_strides,
        b_strides,
        shape,
    )
}

/// Rank-3 strided `mul`, `bf16`. See [`g3_add_f32`].
#[allow(clippy::too_many_arguments)]
pub fn g3_mul_bf16(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    a: &Buffer<bf16>,
    b: &Buffer<bf16>,
    c: &mut Buffer<bf16>,
    a_strides: [i64; 3],
    b_strides: [i64; 3],
    shape: [i32; 3],
) -> Result<()> {
    encode_g3(
        commands,
        library,
        "g3_Multiplybfloat16",
        a,
        b,
        c,
        a_strides,
        b_strides,
        shape,
    )
}

// =====================================================================
// Encoders
// =====================================================================

fn encode_vv<T>(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    kernel_name: &str,
    a: &Buffer<T>,
    b: &Buffer<T>,
    c: &mut Buffer<T>,
) -> Result<()> {
    let n = c.len();
    if a.len() != n || b.len() != n {
        return Err(Error::InvalidArgument(format!(
            "binary vv: element-count mismatch (a={}, b={}, c={n})",
            a.len(),
            b.len(),
        )));
    }
    let size = u32::try_from(n).map_err(|_| {
        Error::InvalidArgument(format!(
            "binary vv: element count {n} does not fit in u32; use the (deferred) vv2_ variant",
        ))
    })?;

    let pipeline = library.pipeline(kernel_name)?;

    {
        let encoder = commands.encoder()?;
        encoder.setComputePipelineState(pipeline.metal_pipeline_state());
        // SAFETY: slots / dtypes match `binary_vv` in `binary.h` — three
        // device buffers then `constant uint& size`. Buffer typing is
        // enforced by the typed `Buffer<T>` and call-site `kernel_name`.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(a.metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(b.metal_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(c.metal_buffer()), 0, 2);
            let size_ptr: NonNull<c_void> = NonNull::from(&size).cast();
            encoder.setBytes_length_atIndex(size_ptr, std::mem::size_of::<u32>(), 3);
        }
    }

    // `vv_` is the N=1 instantiation — one thread per element.
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

#[allow(clippy::too_many_arguments)]
fn encode_g1<T>(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    kernel_name: &str,
    a: &Buffer<T>,
    b: &Buffer<T>,
    c: &mut Buffer<T>,
    a_stride: i64,
    b_stride: i64,
    shape: i32,
) -> Result<()> {
    if shape <= 0 {
        return Ok(());
    }
    let pipeline = library.pipeline(kernel_name)?;

    {
        let encoder = commands.encoder()?;
        encoder.setComputePipelineState(pipeline.metal_pipeline_state());
        // SAFETY: matches `binary_g_nd1` in `binary.h`: buffers a,b,c then
        // two `constant int64_t&` strides.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(a.metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(b.metal_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(c.metal_buffer()), 0, 2);
            let a_ptr: NonNull<c_void> = NonNull::from(&a_stride).cast();
            encoder.setBytes_length_atIndex(a_ptr, std::mem::size_of::<i64>(), 3);
            let b_ptr: NonNull<c_void> = NonNull::from(&b_stride).cast();
            encoder.setBytes_length_atIndex(b_ptr, std::mem::size_of::<i64>(), 4);
        }
    }

    #[allow(clippy::cast_sign_loss)]
    let width = shape as usize;
    let grid = MTLSize {
        width,
        height: 1,
        depth: 1,
    };
    let threadgroup = MTLSize {
        width: width.min(1024),
        height: 1,
        depth: 1,
    };
    commands.dispatch_threads(grid, threadgroup)
}

#[allow(clippy::too_many_arguments)]
fn encode_g2<T>(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    kernel_name: &str,
    a: &Buffer<T>,
    b: &Buffer<T>,
    c: &mut Buffer<T>,
    a_strides: [i64; 2],
    b_strides: [i64; 2],
    shape: [i32; 2],
) -> Result<()> {
    if shape.iter().any(|&d| d <= 0) {
        return Ok(());
    }
    let pipeline = library.pipeline(kernel_name)?;

    {
        let encoder = commands.encoder()?;
        encoder.setComputePipelineState(pipeline.metal_pipeline_state());
        // SAFETY: matches `binary_g_nd2` — three buffers then two
        // `constant int64_t[2]` stride arrays.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(a.metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(b.metal_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(c.metal_buffer()), 0, 2);
            let a_ptr: NonNull<c_void> = NonNull::from(&a_strides).cast();
            encoder.setBytes_length_atIndex(a_ptr, std::mem::size_of_val(&a_strides), 3);
            let b_ptr: NonNull<c_void> = NonNull::from(&b_strides).cast();
            encoder.setBytes_length_atIndex(b_ptr, std::mem::size_of_val(&b_strides), 4);
        }
    }

    // Grid: (inner, outer). Matches MLX `binary.cpp` ndim<=3 path —
    // shape is implicit in the grid and `get_block_dims` picks the
    // threadgroup.
    #[allow(clippy::cast_sign_loss)]
    let dim0 = shape[1] as usize;
    #[allow(clippy::cast_sign_loss)]
    let dim1 = shape[0] as usize;
    let grid = MTLSize {
        width: dim0,
        height: dim1,
        depth: 1,
    };
    let threadgroup = get_block_dims(dim0, dim1, 1, 10);
    commands.dispatch_threads(grid, threadgroup)
}

#[allow(clippy::too_many_arguments)]
fn encode_g3<T>(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    kernel_name: &str,
    a: &Buffer<T>,
    b: &Buffer<T>,
    c: &mut Buffer<T>,
    a_strides: [i64; 3],
    b_strides: [i64; 3],
    shape: [i32; 3],
) -> Result<()> {
    if shape.iter().any(|&d| d <= 0) {
        return Ok(());
    }
    let pipeline = library.pipeline(kernel_name)?;

    {
        let encoder = commands.encoder()?;
        encoder.setComputePipelineState(pipeline.metal_pipeline_state());
        // SAFETY: matches `binary_g_nd3` — three buffers then two
        // `constant int64_t[3]` stride arrays.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(a.metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(b.metal_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(c.metal_buffer()), 0, 2);
            let a_ptr: NonNull<c_void> = NonNull::from(&a_strides).cast();
            encoder.setBytes_length_atIndex(a_ptr, std::mem::size_of_val(&a_strides), 3);
            let b_ptr: NonNull<c_void> = NonNull::from(&b_strides).cast();
            encoder.setBytes_length_atIndex(b_ptr, std::mem::size_of_val(&b_strides), 4);
        }
    }

    #[allow(clippy::cast_sign_loss)]
    let dim0 = shape[2] as usize;
    #[allow(clippy::cast_sign_loss)]
    let dim1 = shape[1] as usize;
    #[allow(clippy::cast_sign_loss)]
    let dim2 = shape[0] as usize;
    let grid = MTLSize {
        width: dim0,
        height: dim1,
        depth: dim2,
    };
    let threadgroup = get_block_dims(dim0, dim1, dim2, 10);
    commands.dispatch_threads(grid, threadgroup)
}

/// Port of MLX `get_block_dims_common` (`common/utils.cpp:117`).
///
/// Picks a power-of-two threadgroup shape with total log2 bits ≤ `pow2`
/// (10 → 1024 threads max), greedily allocating to the largest axis
/// first. Duplicated from `kernels::copy`; will be hoisted to a shared
/// helper once a third caller appears.
fn get_block_dims(dim0: usize, dim1: usize, dim2: usize, max_log2: u32) -> MTLSize {
    let extents = [dim0, dim1, dim2];
    let mut log2_dims = [0u32; 3];
    let mut total: u32 = 0;
    loop {
        let prev_total = total;
        for axis in 0..3 {
            if extents[axis] >= (1usize << (log2_dims[axis] + 1)) && total < max_log2 {
                log2_dims[axis] += 1;
                total += 1;
            }
            if total == max_log2 {
                break;
            }
        }
        if total == prev_total || total == max_log2 {
            break;
        }
    }
    MTLSize {
        width: 1usize << log2_dims[0],
        height: 1usize << log2_dims[1],
        depth: 1usize << log2_dims[2],
    }
}
