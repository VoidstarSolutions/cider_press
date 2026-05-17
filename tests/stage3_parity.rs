//! Stage 3 of the development spike — see `CLAUDE.md`.
//!
//! Reference parity harness. Loads pre-generated input/expected-output
//! safetensors (produced by `scripts/gen_stage3_fixtures.py`), runs the
//! MLX `v_copy` kernel via cider-press, and compares against the MLX
//! reference with dtype-appropriate tolerance.
//!
//! The harness exercised here is the rig Stage 4 will use to validate
//! `qmv` against MLX's `quantized_matmul`. For copy the reference equals
//! the input, so this stage is mostly a smoke test of the rig itself.

#![cfg(target_os = "macos")]
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use std::ffi::c_void;
use std::path::PathBuf;
use std::ptr::NonNull;

use approx::assert_relative_eq;
use half::f16;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary,
    MTLResourceOptions, MTLSize,
};
use safetensors::SafeTensors;

#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {}

const COPY_SOURCE: &str = include_str!(concat!(env!("OUT_DIR"), "/copy_inlined.metal"));

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("stage3_copy.safetensors")
}

fn mlx_source_unavailable() -> bool {
    COPY_SOURCE.trim().starts_with("// MLX checkout not found")
}

#[test]
fn parity_v_copy_f32() {
    if mlx_source_unavailable() {
        eprintln!("skipping: MLX checkout not found at build time");
        return;
    }
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

    let ctx = MetalCtx::new();
    let pipeline = ctx.pipeline("v_copyfloat32float32");
    let actual = run_copy(&ctx, &pipeline, &src);

    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_relative_eq!(*a, *e, max_relative = 1e-6, epsilon = 1e-7);
        // index i is included on assert failure via panic context below
        let _ = i;
    }
}

#[test]
fn parity_v_copy_f16() {
    if mlx_source_unavailable() {
        eprintln!("skipping: MLX checkout not found at build time");
        return;
    }
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

    let ctx = MetalCtx::new();
    let pipeline = ctx.pipeline("v_copyfloat16float16");
    let actual = run_copy(&ctx, &pipeline, &src);

    for (a, e) in actual.iter().zip(expected.iter()) {
        let a32 = a.to_f32();
        let e32 = e.to_f32();
        assert_relative_eq!(a32, e32, max_relative = 1e-3, epsilon = 1e-4);
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

struct MetalCtx {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
}

impl MetalCtx {
    fn new() -> Self {
        let device = MTLCreateSystemDefaultDevice().expect("no Metal device available");
        let queue = device.newCommandQueue().expect("command queue");
        let source = NSString::from_str(COPY_SOURCE);
        let library = device
            .newLibraryWithSource_options_error(&source, None)
            .unwrap_or_else(|err| panic!("MLX copy.metal compile: {err}"));
        Self {
            device,
            queue,
            library,
        }
    }

    fn pipeline(&self, name: &str) -> Retained<ProtocolObject<dyn MTLComputePipelineState>> {
        let ns_name = NSString::from_str(name);
        let function = self
            .library
            .newFunctionWithName(&ns_name)
            .unwrap_or_else(|| panic!("kernel {name} not found"));
        self.device
            .newComputePipelineStateWithFunction_error(&function)
            .unwrap_or_else(|err| panic!("pipeline {name}: {err}"))
    }
}

fn run_copy<T: Copy + Default>(
    ctx: &MetalCtx,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    src: &[T],
) -> Vec<T> {
    let bytes = std::mem::size_of_val(src);

    let in_buf = ctx
        .device
        .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
        .expect("input buffer");
    let out_buf = ctx
        .device
        .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
        .expect("output buffer");

    unsafe {
        let ptr: NonNull<T> = in_buf.contents().cast();
        std::slice::from_raw_parts_mut(ptr.as_ptr(), src.len()).copy_from_slice(src);
    }

    let size = u32::try_from(src.len()).expect("size fits in u32");

    let cmd = ctx.queue.commandBuffer().expect("commandBuffer");
    let encoder = cmd.computeCommandEncoder().expect("computeCommandEncoder");

    encoder.setComputePipelineState(pipeline);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(&in_buf), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(&out_buf), 0, 1);
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
    unsafe {
        let ptr: NonNull<T> = out_buf.contents().cast();
        out.copy_from_slice(std::slice::from_raw_parts(ptr.as_ptr(), src.len()));
    }
    out
}
