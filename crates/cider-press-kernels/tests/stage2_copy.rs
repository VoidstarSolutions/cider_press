//! Stage 2 of the development spike — see `CLAUDE.md`.
//!
//! Loads MLX's `copy.metal` (vendored under `kernels-mlx/`, flattened by
//! `build.rs`), compiles it at runtime, and dispatches the simplest
//! vector-copy instantiation against 1D `f32` and `f16` inputs. Asserts
//! the output is bit-identical to the input — the kernel is identity.
//!
//! Acceptance: passes under `cargo test --release` for both dtypes.

#![cfg(target_os = "macos")]
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use std::ffi::c_void;
use std::ptr::NonNull;

use cider_press_kernels::{Buffer, Device, KernelLibrary, Pipeline};
use half::f16;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLSize,
};

const N: usize = 1024;

#[test]
fn copy_v_f32_identity() {
    let device = Device::system_default().expect("Metal device");
    let library = KernelLibrary::copy(&device).expect("compile copy.metal");
    let pipeline = library.pipeline("v_copyfloat32float32").expect("kernel");

    let mut src = vec![0_f32; N];
    for (i, v) in src.iter_mut().enumerate() {
        *v = i as f32 * 0.5 - 17.0;
    }
    let dst = run_copy(&device, &pipeline, &src);

    for (i, (&a, &b)) in src.iter().zip(dst.iter()).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "f32 mismatch at index {i}");
    }
}

#[test]
fn copy_v_f16_identity() {
    let device = Device::system_default().expect("Metal device");
    let library = KernelLibrary::copy(&device).expect("compile copy.metal");
    let pipeline = library.pipeline("v_copyfloat16float16").expect("kernel");

    let mut src = vec![f16::ZERO; N];
    for (i, v) in src.iter_mut().enumerate() {
        *v = f16::from_f32(i as f32 * 0.25 - 8.0);
    }
    let dst = run_copy(&device, &pipeline, &src);

    for (i, (&a, &b)) in src.iter().zip(dst.iter()).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "f16 mismatch at index {i}");
    }
}

/// Run one dispatch of `copy_v<T, T, 1>` and return the destination
/// slice. `T` must match the MSL scalar type of `pipeline`.
fn run_copy<T: Copy + Default>(device: &Device, pipeline: &Pipeline, src: &[T]) -> Vec<T> {
    let in_buf: Buffer<T> = device.upload(src).expect("upload src");
    let mut out_buf: Buffer<T> = device.alloc_buffer(src.len()).expect("alloc dst");
    let size = u32::try_from(src.len()).expect("size fits in u32");

    let cmd = device.metal_queue().commandBuffer().expect("commandBuffer");
    let encoder = cmd.computeCommandEncoder().expect("computeCommandEncoder");
    encoder.setComputePipelineState(pipeline.metal_pipeline_state());
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(in_buf.metal_buffer()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(out_buf.metal_buffer()), 0, 1);
        // `constant uint& size` auto-binds to the next free buffer slot (2).
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
    encoder.endEncoding();

    cmd.commit();
    cmd.waitUntilCompleted();

    let mut out = vec![T::default(); src.len()];
    // SAFETY: waitUntilCompleted returned; GPU has finished writing out_buf.
    unsafe { out.copy_from_slice(out_buf.as_mut_slice()) };
    out
}
