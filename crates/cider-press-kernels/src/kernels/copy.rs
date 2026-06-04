// Dispatch routine derived from MLX (https://github.com/ml-explore/mlx).
// Copyright © 2023-2024 Apple Inc. Released under the MIT license — see
// the full upstream notice at `kernels-mlx/COPYING`.

//! Copy dispatch over MLX's `copy.metal`.
//!
//! Exposes the same-dtype identity-copy variants the runtime layer
//! needs: `f32`, `f16`, `bf16`, `i32`, `u32`. Two flavours:
//!
//! - `copy_v_*` — `CopyType::Vector` (element-wise contiguous copy,
//!   used when src and dst are both dense row-major with matching byte
//!   layouts);
//! - `copy_g_*` — `CopyType::GeneralGeneral` (strided gather from
//!   arbitrary src shape/strides/byte-offset into arbitrary dst
//!   shape/strides/byte-offset). Used by the runtime to materialise
//!   non-contiguous views (transpose, broadcast, sliced inner dim).
//!
//! Additional dtype combinations from MLX's `instantiate_copy_*`
//! macros (cross-dtype casts, complex, smaller integer widths) can be
//! added the same way when callers ask for them.
//!
//! Analogue of MLX's `mlx::core::metal::copy_gpu_inplace`.

use std::ffi::c_void;
use std::ptr::NonNull;

use half::{bf16, f16};
use objc2_metal::{MTLComputeCommandEncoder, MTLSize};

use crate::buffer::Buffer;
use crate::commands::Commands;
use crate::error::{Error, Result};
use crate::library::KernelLibrary;

/// Identity-copy `src` into `dst`, both `f32`.
///
/// Encodes one dispatch into `commands`; the caller must invoke
/// [`Commands::commit_and_wait`] to
/// flush it to the GPU.
///
/// `library` must be a [`KernelLibrary::copy`].
pub fn copy_v_f32(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    src: &Buffer<f32>,
    dst: &mut Buffer<f32>,
) -> Result<()> {
    encode_v_copy(commands, library, "v_copyfloat32float32", src, dst)
}

/// Identity-copy `src` into `dst`, both `f16`.
///
/// See [`copy_v_f32`] for semantics; `library` must be a
/// [`KernelLibrary::copy`].
pub fn copy_v_f16(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    src: &Buffer<f16>,
    dst: &mut Buffer<f16>,
) -> Result<()> {
    encode_v_copy(commands, library, "v_copyfloat16float16", src, dst)
}

/// Identity-copy `src` into `dst`, both `bf16`.
///
/// See [`copy_v_f32`] for semantics; `library` must be a
/// [`KernelLibrary::copy`].
pub fn copy_v_bf16(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    src: &Buffer<bf16>,
    dst: &mut Buffer<bf16>,
) -> Result<()> {
    encode_v_copy(commands, library, "v_copybfloat16bfloat16", src, dst)
}

/// Identity-copy `src` into `dst`, both `i32`.
///
/// See [`copy_v_f32`] for semantics; `library` must be a
/// [`KernelLibrary::copy`].
pub fn copy_v_i32(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    src: &Buffer<i32>,
    dst: &mut Buffer<i32>,
) -> Result<()> {
    encode_v_copy(commands, library, "v_copyint32int32", src, dst)
}

/// Identity-copy `src` into `dst`, both `u32`.
///
/// See [`copy_v_f32`] for semantics; `library` must be a
/// [`KernelLibrary::copy`].
pub fn copy_v_u32(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    src: &Buffer<u32>,
    dst: &mut Buffer<u32>,
) -> Result<()> {
    encode_v_copy(commands, library, "v_copyuint32uint32", src, dst)
}

/// Strided same-dtype copy: `dst[logical_idx] = src[logical_idx]` for
/// every logical index, walking arbitrary src/dst element strides and
/// honoring per-buffer byte offsets.
///
/// Dispatches MLX's `gg{1,2,3}_copy<T><T>` (specialised by rank) for
/// rank ≤ 3, and `ggn2_copy<T><T>` for rank ≥ 4. `f32` variant. See
/// the module docstring for which kernels these are.
///
/// `src_strides` / `dst_strides` are *element* strides (one entry per
/// axis); shape is the logical shape (one entry per axis). Byte
/// offsets are applied at bind time. Caller is responsible for
/// ensuring every reachable element index falls within the underlying
/// buffer bounds.
#[allow(clippy::too_many_arguments)]
pub fn copy_g_f32(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    src: &Buffer<f32>,
    src_byte_offset: usize,
    src_strides: &[i64],
    dst: &mut Buffer<f32>,
    dst_byte_offset: usize,
    dst_strides: &[i64],
    shape: &[i32],
) -> Result<()> {
    encode_g_copy(
        commands,
        library,
        "float32float32",
        src.metal_buffer(),
        src_byte_offset,
        src_strides,
        dst.metal_buffer(),
        dst_byte_offset,
        dst_strides,
        shape,
    )
}

/// `f16` strided same-dtype copy. See [`copy_g_f32`] for semantics.
#[allow(clippy::too_many_arguments)]
pub fn copy_g_f16(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    src: &Buffer<f16>,
    src_byte_offset: usize,
    src_strides: &[i64],
    dst: &mut Buffer<f16>,
    dst_byte_offset: usize,
    dst_strides: &[i64],
    shape: &[i32],
) -> Result<()> {
    encode_g_copy(
        commands,
        library,
        "float16float16",
        src.metal_buffer(),
        src_byte_offset,
        src_strides,
        dst.metal_buffer(),
        dst_byte_offset,
        dst_strides,
        shape,
    )
}

/// `bf16` strided same-dtype copy. See [`copy_g_f32`] for semantics.
#[allow(clippy::too_many_arguments)]
pub fn copy_g_bf16(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    src: &Buffer<bf16>,
    src_byte_offset: usize,
    src_strides: &[i64],
    dst: &mut Buffer<bf16>,
    dst_byte_offset: usize,
    dst_strides: &[i64],
    shape: &[i32],
) -> Result<()> {
    encode_g_copy(
        commands,
        library,
        "bfloat16bfloat16",
        src.metal_buffer(),
        src_byte_offset,
        src_strides,
        dst.metal_buffer(),
        dst_byte_offset,
        dst_strides,
        shape,
    )
}

/// `i32` strided same-dtype copy. See [`copy_g_f32`] for semantics.
#[allow(clippy::too_many_arguments)]
pub fn copy_g_i32(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    src: &Buffer<i32>,
    src_byte_offset: usize,
    src_strides: &[i64],
    dst: &mut Buffer<i32>,
    dst_byte_offset: usize,
    dst_strides: &[i64],
    shape: &[i32],
) -> Result<()> {
    encode_g_copy(
        commands,
        library,
        "int32int32",
        src.metal_buffer(),
        src_byte_offset,
        src_strides,
        dst.metal_buffer(),
        dst_byte_offset,
        dst_strides,
        shape,
    )
}

/// `u32` strided same-dtype copy. See [`copy_g_f32`] for semantics.
#[allow(clippy::too_many_arguments)]
pub fn copy_g_u32(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    src: &Buffer<u32>,
    src_byte_offset: usize,
    src_strides: &[i64],
    dst: &mut Buffer<u32>,
    dst_byte_offset: usize,
    dst_strides: &[i64],
    shape: &[i32],
) -> Result<()> {
    encode_g_copy(
        commands,
        library,
        "uint32uint32",
        src.metal_buffer(),
        src_byte_offset,
        src_strides,
        dst.metal_buffer(),
        dst_byte_offset,
        dst_strides,
        shape,
    )
}

const MAX_COPY_SPECIALIZED_DIMS: usize = 3;

/// Shared encoder for the `copy_gg*` (general-general, same-dtype)
/// template family. Mirrors the `CopyType::GeneralGeneral` branch of
/// MLX's `copy_gpu_inplace`.
#[allow(clippy::too_many_arguments)]
fn encode_g_copy(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    dtype_suffix: &str,
    src_buffer: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
    src_byte_offset: usize,
    src_strides: &[i64],
    dst_buffer: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
    dst_byte_offset: usize,
    dst_strides: &[i64],
    shape: &[i32],
) -> Result<()> {
    let ndim = shape.len();
    if ndim == 0 {
        return Err(Error::InvalidArgument(
            "copy_g: rank-0 (scalar) strided copy is not supported; use a vector copy".into(),
        ));
    }
    if src_strides.len() != ndim || dst_strides.len() != ndim {
        return Err(Error::InvalidArgument(format!(
            "copy_g: strides/shape rank mismatch (shape={ndim}, src_strides={}, dst_strides={})",
            src_strides.len(),
            dst_strides.len(),
        )));
    }
    if shape.iter().any(|&d| d <= 0) {
        // Empty dim → no work, but still well-defined. MLX's
        // `copy_gpu_inplace` short-circuits on `out.size() == 0`.
        return Ok(());
    }

    let kernel_name = if ndim <= MAX_COPY_SPECIALIZED_DIMS {
        format!("gg{ndim}_copy{dtype_suffix}")
    } else {
        format!("ggn2_copy{dtype_suffix}")
    };
    let pipeline = library.pipeline(&kernel_name)?;

    // Grid layout (mirrors `copy.cpp:131–163`):
    //   dim0 = shape[ndim-1], dim1 = shape[ndim-2] (or 1), rest = product of remaining.
    // Negative / zero dims were rejected above, so the `as usize` casts are safe.
    #[allow(clippy::cast_sign_loss)]
    let dim0 = shape[ndim - 1] as usize;
    #[allow(clippy::cast_sign_loss)]
    let dim1 = if ndim > 1 {
        shape[ndim - 2] as usize
    } else {
        1
    };
    #[allow(clippy::cast_sign_loss)]
    let rest: usize = shape[..ndim.saturating_sub(2)]
        .iter()
        .map(|&d| d as usize)
        .product::<usize>()
        .max(1);

    {
        let encoder = commands.encoder()?;
        encoder.setComputePipelineState(pipeline.metal_pipeline_state());
        // SAFETY: bind slots / dtypes match the `copy_gg*` MSL signatures
        // in `kernels-mlx/.../copy.h`. Element strides are passed as
        // `int64_t`; byte offsets ride along on the buffer bind.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(src_buffer), src_byte_offset, 0);
            encoder.setBuffer_offset_atIndex(Some(dst_buffer), dst_byte_offset, 1);

            if ndim > MAX_COPY_SPECIALIZED_DIMS {
                let shape_ptr: NonNull<c_void> = NonNull::from(shape).cast();
                encoder.setBytes_length_atIndex(shape_ptr, std::mem::size_of_val(shape), 2);
            }
            let src_strides_ptr: NonNull<c_void> = NonNull::from(src_strides).cast();
            encoder.setBytes_length_atIndex(src_strides_ptr, std::mem::size_of_val(src_strides), 3);
            let dst_strides_ptr: NonNull<c_void> = NonNull::from(dst_strides).cast();
            encoder.setBytes_length_atIndex(dst_strides_ptr, std::mem::size_of_val(dst_strides), 4);
            if ndim > MAX_COPY_SPECIALIZED_DIMS {
                let ndim_i32 = i32::try_from(ndim).map_err(|_| {
                    Error::InvalidArgument(format!("copy_g: ndim {ndim} does not fit in i32"))
                })?;
                let ndim_ptr: NonNull<c_void> = NonNull::from(&ndim_i32).cast();
                encoder.setBytes_length_atIndex(ndim_ptr, std::mem::size_of::<i32>(), 5);
            }
        }
    }

    let threadgroup = get_block_dims(dim0, dim1, rest, 10);
    let grid = MTLSize {
        width: dim0,
        height: dim1,
        depth: rest,
    };
    commands.dispatch_threads(grid, threadgroup)
}

/// Port of MLX `get_block_dims_common` (`common/utils.cpp:117`).
///
/// Picks a power-of-two threadgroup shape with total log2 bits ≤ `pow2`
/// (10 → 1024 threads max), greedily allocating to the largest axis
/// first.
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

/// Shared encoder for the `copy_v<T, T, 1>` template family.
fn encode_v_copy<T>(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    kernel_name: &str,
    src: &Buffer<T>,
    dst: &mut Buffer<T>,
) -> Result<()> {
    if src.len() != dst.len() {
        return Err(Error::InvalidArgument(format!(
            "copy_v: src.len() ({}) != dst.len() ({})",
            src.len(),
            dst.len(),
        )));
    }
    let size = u32::try_from(src.len()).map_err(|_| {
        Error::InvalidArgument(format!(
            "copy_v: element count {} does not fit in u32",
            src.len(),
        ))
    })?;

    let pipeline = library.pipeline(kernel_name)?;

    {
        let encoder = commands.encoder()?;
        encoder.setComputePipelineState(pipeline.metal_pipeline_state());
        // SAFETY: src/dst are typed `Buffer<T>` whose element type matches the
        // kernel's MSL scalar type, validated at the call-site by choice of
        // `kernel_name`. Offsets are 0 and bind-slot indices are fixed by the
        // kernel signature (see MLX `copy.h::copy_v`).
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(src.metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(dst.metal_buffer()), 0, 1);
            // `constant uint& size` auto-binds to slot 2 (after the two buffers).
            let bytes_ptr: NonNull<c_void> = NonNull::from(&size).cast();
            encoder.setBytes_length_atIndex(bytes_ptr, std::mem::size_of::<u32>(), 2);
        }
    }

    let grid = MTLSize {
        width: src.len(),
        height: 1,
        depth: 1,
    };
    let threadgroup = MTLSize {
        width: 64,
        height: 1,
        depth: 1,
    };
    commands.dispatch_threads(grid, threadgroup)
}
