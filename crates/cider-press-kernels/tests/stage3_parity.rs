//! Stage 3 of the development spike — see `CLAUDE.md`.
//!
//! Reference parity harness. Loads pre-generated input/expected-output
//! safetensors (produced by `scripts/gen_stage3_fixtures.py`), runs the
//! MLX `v_copy` kernel via cider-press, and compares against the MLX
//! reference with dtype-appropriate tolerance.
//!
//! The harness exercised here is the rig Stage 4 uses to validate
//! `qmv` against MLX's `quantized_matmul`. For copy the reference
//! equals the input, so this stage mostly smoke-tests the rig itself.

#![cfg(target_os = "macos")]
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use std::ffi::c_void;
use std::path::PathBuf;
use std::ptr::NonNull;

use approx::assert_relative_eq;
use cider_press_kernels::{Buffer, Device, KernelLibrary, Pipeline};
use half::f16;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLSize,
};
use safetensors::SafeTensors;

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("stage3_copy.safetensors")
}

#[test]
fn parity_v_copy_f32() {
    let path = fixture_path();
    if !path.exists() {
        eprintln!(
            "skipping: fixture missing at {}; run `uv run scripts/gen_stage3_fixtures.py`",
            path.display(),
        );
        return;
    }

    let bytes = std::fs::read(&path).expect("read fixture");
    let st = SafeTensors::deserialize(&bytes).expect("parse safetensors");
    let src: Vec<f32> = read_tensor_f32(&st, "src_f32");
    let expected: Vec<f32> = read_tensor_f32(&st, "dst_f32");
    assert_eq!(src.len(), expected.len());

    let device = Device::system_default().expect("Metal device");
    let library = KernelLibrary::copy(&device).expect("compile copy.metal");
    let pipeline = library.pipeline("v_copyfloat32float32").expect("kernel");
    let actual = run_copy(&device, &pipeline, &src);

    for (a, e) in actual.iter().zip(expected.iter()) {
        assert_relative_eq!(*a, *e, max_relative = 1e-6, epsilon = 1e-7);
    }
}

#[test]
fn parity_v_copy_f16() {
    let path = fixture_path();
    if !path.exists() {
        eprintln!(
            "skipping: fixture missing at {}; run `uv run scripts/gen_stage3_fixtures.py`",
            path.display(),
        );
        return;
    }

    let bytes = std::fs::read(&path).expect("read fixture");
    let st = SafeTensors::deserialize(&bytes).expect("parse safetensors");
    let src: Vec<f16> = read_tensor_f16(&st, "src_f16");
    let expected: Vec<f16> = read_tensor_f16(&st, "dst_f16");
    assert_eq!(src.len(), expected.len());

    let device = Device::system_default().expect("Metal device");
    let library = KernelLibrary::copy(&device).expect("compile copy.metal");
    let pipeline = library.pipeline("v_copyfloat16float16").expect("kernel");
    let actual = run_copy(&device, &pipeline, &src);

    for (a, e) in actual.iter().zip(expected.iter()) {
        assert_relative_eq!(a.to_f32(), e.to_f32(), max_relative = 1e-3, epsilon = 1e-4);
    }
}

fn read_tensor_f32(st: &SafeTensors, name: &str) -> Vec<f32> {
    let view = st
        .tensor(name)
        .unwrap_or_else(|e| panic!("tensor {name}: {e}"));
    assert_eq!(view.dtype(), safetensors::Dtype::F32);
    let bytes = view.data();
    assert!(bytes.len() % 4 == 0);
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn read_tensor_f16(st: &SafeTensors, name: &str) -> Vec<f16> {
    let view = st
        .tensor(name)
        .unwrap_or_else(|e| panic!("tensor {name}: {e}"));
    assert_eq!(view.dtype(), safetensors::Dtype::F16);
    let bytes = view.data();
    assert!(bytes.len() % 2 == 0);
    bytes
        .chunks_exact(2)
        .map(|c| f16::from_le_bytes([c[0], c[1]]))
        .collect()
}

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
