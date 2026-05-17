//! Stage 2 of the development spike — see `CLAUDE.md`.
//!
//! Loads the real MLX `copy.metal` source (flattened by `build.rs`),
//! compiles it at runtime, and dispatches the simplest vector-copy
//! instantiation against 1D `f32` and `f16` inputs. Asserts the output
//! is bit-identical to the input — the kernel is an identity copy.
//!
//! Acceptance: passes under `cargo test --release` for both dtypes.

#![cfg(target_os = "macos")]
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use std::ffi::c_void;
use std::ptr::NonNull;

use half::f16;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary,
    MTLResourceOptions, MTLSize,
};

#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {}

/// MLX `copy.metal` plus all its transitive project-relative includes,
/// flattened by `build.rs` into one self-contained source string.
const COPY_SOURCE: &str = include_str!(concat!(env!("OUT_DIR"), "/copy_inlined.metal"));

const N: usize = 1024;

#[test]
fn copy_v_f32_identity() {
    if mlx_source_unavailable() {
        eprintln!("skipping: MLX checkout not found at build time (set CIDER_MLX_DIR)");
        return;
    }

    let ctx = MetalCtx::new();
    let pipeline = ctx.pipeline("v_copyfloat32float32");

    let mut src = vec![0_f32; N];
    for (i, v) in src.iter_mut().enumerate() {
        *v = i as f32 * 0.5 - 17.0;
    }
    let dst = run_copy(&ctx, &pipeline, &src);

    for (i, (&a, &b)) in src.iter().zip(dst.iter()).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "f32 mismatch at index {i}");
    }
}

#[test]
fn copy_v_f16_identity() {
    if mlx_source_unavailable() {
        eprintln!("skipping: MLX checkout not found at build time (set CIDER_MLX_DIR)");
        return;
    }

    let ctx = MetalCtx::new();
    let pipeline = ctx.pipeline("v_copyfloat16float16");

    let mut src = vec![f16::ZERO; N];
    for (i, v) in src.iter_mut().enumerate() {
        *v = f16::from_f32(i as f32 * 0.25 - 8.0);
    }
    let dst = run_copy(&ctx, &pipeline, &src);

    for (i, (&a, &b)) in src.iter().zip(dst.iter()).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "f16 mismatch at index {i}");
    }
}

fn mlx_source_unavailable() -> bool {
    // build.rs writes a stub when the MLX checkout is missing; detect that.
    COPY_SOURCE.trim().starts_with("// MLX checkout not found")
}

struct MetalCtx {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
}

impl MetalCtx {
    fn new() -> Self {
        let device = MTLCreateSystemDefaultDevice().expect("no Metal device available");
        let queue = device
            .newCommandQueue()
            .expect("failed to create command queue");
        let source = NSString::from_str(COPY_SOURCE);
        let library = device
            .newLibraryWithSource_options_error(&source, None)
            .unwrap_or_else(|err| panic!("MLX copy.metal failed to compile: {err}"));
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
            .unwrap_or_else(|| panic!("kernel {name} not found in compiled library"));
        self.device
            .newComputePipelineStateWithFunction_error(&function)
            .unwrap_or_else(|err| panic!("pipeline build for {name} failed: {err}"))
    }
}

/// Run one dispatch of `copy_v<T, T, 1>` and return the destination slice.
/// `T` must be `Copy + Default` and have the same memory layout as the MSL
/// scalar type associated with `pipeline`.
fn run_copy<T: Copy + Default>(
    ctx: &MetalCtx,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    src: &[T],
) -> Vec<T> {
    let bytes = std::mem::size_of_val(src);

    let in_buf = ctx
        .device
        .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
        .expect("failed to allocate input buffer");
    let out_buf = ctx
        .device
        .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
        .expect("failed to allocate output buffer");

    unsafe {
        let ptr: NonNull<T> = in_buf.contents().cast();
        std::slice::from_raw_parts_mut(ptr.as_ptr(), src.len()).copy_from_slice(src);
    }

    let size: u32 = u32::try_from(src.len()).expect("size fits in u32");

    let cmd = ctx
        .queue
        .commandBuffer()
        .expect("commandBuffer returned nil");
    let encoder = cmd
        .computeCommandEncoder()
        .expect("computeCommandEncoder returned nil");

    encoder.setComputePipelineState(pipeline);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(&in_buf), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(&out_buf), 0, 1);
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
    unsafe {
        let ptr: NonNull<T> = out_buf.contents().cast();
        out.copy_from_slice(std::slice::from_raw_parts(ptr.as_ptr(), src.len()));
    }
    out
}
