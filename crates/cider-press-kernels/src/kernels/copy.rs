// Dispatch routine derived from MLX (https://github.com/ml-explore/mlx).
// Copyright © 2023-2024 Apple Inc. Released under the MIT license — see
// the full upstream notice at `kernels-mlx/COPYING`.

//! Vector-copy dispatch over MLX's `copy.metal`.
//!
//! Currently exposes only the same-dtype identity-copy variants the
//! spike used (`v_copyfloat32float32`, `v_copyfloat16float16`).
//! Additional dtype combinations from MLX's instantiate macros can be
//! added the same way as the runtime layer needs them.
//!
//! Analogue of MLX's `mlx::core::metal::copy_gpu_inplace` (the
//! `CopyType::Vector` branch).

use std::ffi::c_void;
use std::ptr::NonNull;

use half::f16;
use objc2_metal::{MTLComputeCommandEncoder, MTLSize};

use crate::buffer::Buffer;
use crate::commands::Commands;
use crate::error::Result;
use crate::library::KernelLibrary;

/// Identity-copy `src` into `dst`, both `f32`.
///
/// Encodes one dispatch into `commands`; the caller must invoke
/// [`Commands::commit_and_wait`](crate::Commands::commit_and_wait) to
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

/// Shared encoder for the `copy_v<T, T, 1>` template family.
fn encode_v_copy<T>(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    kernel_name: &str,
    src: &Buffer<T>,
    dst: &mut Buffer<T>,
) -> Result<()> {
    assert_eq!(
        src.len(),
        dst.len(),
        "copy_v: src.len() ({}) != dst.len() ({})",
        src.len(),
        dst.len(),
    );

    let pipeline = library.pipeline(kernel_name)?;
    let size = u32::try_from(src.len()).expect("element count fits in u32");
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
    encoder.dispatchThreads_threadsPerThreadgroup(grid, threadgroup);
    Ok(())
}
