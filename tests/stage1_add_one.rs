//! Stage 1 of the development spike — see `CLAUDE.md`.
//!
//! Compiles a trivial inline Metal kernel at runtime, dispatches it on 1024
//! `f32` elements via shared-memory `MTLBuffer`s, and verifies the result.
//!
//! Acceptance: passes under `cargo test --release`, and a single dispatch
//! round-trip (after JIT warm-up) completes in well under 5 ms.

#![cfg(target_os = "macos")]
// Test code values clarity over the pedantic numeric-cast lints.
// Bit-exact f32 comparison is intentional here: `add_one` on integer-valued
// inputs produces results that are exactly representable.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::float_cmp
)]

use std::ptr::NonNull;
use std::time::Instant;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary,
    MTLResourceOptions, MTLSize,
};

// `MTLCreateSystemDefaultDevice` requires CoreGraphics at link time
// (see the objc2-metal crate docs).
#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {}

const KERNEL_SOURCE: &str = "
#include <metal_stdlib>
using namespace metal;
kernel void add_one(
    device const float* x [[buffer(0)]],
    device float* y       [[buffer(1)]],
    uint i [[thread_position_in_grid]]
) { y[i] = x[i] + 1.0; }
";

const N: usize = 1024;
const BYTES: usize = N * std::mem::size_of::<f32>();

#[test]
fn add_one_round_trip() {
    let device: Retained<ProtocolObject<dyn MTLDevice>> =
        MTLCreateSystemDefaultDevice().expect("no Metal device available");

    let queue = device
        .newCommandQueue()
        .expect("failed to create command queue");

    let source = NSString::from_str(KERNEL_SOURCE);
    let library = device
        .newLibraryWithSource_options_error(&source, None)
        .expect("kernel compilation failed");

    let function_name = NSString::from_str("add_one");
    let function = library
        .newFunctionWithName(&function_name)
        .expect("add_one not found in compiled library");

    let pipeline = device
        .newComputePipelineStateWithFunction_error(&function)
        .expect("failed to build compute pipeline");

    let in_buf = device
        .newBufferWithLength_options(BYTES, MTLResourceOptions::StorageModeShared)
        .expect("failed to allocate input buffer");
    let out_buf = device
        .newBufferWithLength_options(BYTES, MTLResourceOptions::StorageModeShared)
        .expect("failed to allocate output buffer");

    // Shared storage on Apple Silicon means GPU and CPU see the same memory —
    // no `did_modify_range` call required after a host write.
    unsafe {
        let ptr: NonNull<f32> = in_buf.contents().cast();
        let slice = std::slice::from_raw_parts_mut(ptr.as_ptr(), N);
        for (i, v) in slice.iter_mut().enumerate() {
            *v = i as f32;
        }
    }

    // Warm-up dispatch so the timing below excludes one-time costs (driver
    // setup, GPU power-up, command-buffer caches).
    dispatch(&queue, &pipeline, &in_buf, &out_buf);

    let start = Instant::now();
    dispatch(&queue, &pipeline, &in_buf, &out_buf);
    let elapsed = start.elapsed();

    let out: &[f32] = unsafe {
        let ptr: NonNull<f32> = out_buf.contents().cast();
        std::slice::from_raw_parts(ptr.as_ptr(), N)
    };
    for (i, &v) in out.iter().enumerate() {
        assert_eq!(v, i as f32 + 1.0, "mismatch at index {i}");
    }

    eprintln!("add_one dispatch round-trip: {elapsed:?}");
    assert!(
        elapsed.as_secs_f64() * 1e3 < 5.0,
        "dispatch took {elapsed:?}; acceptance is < 5 ms",
    );
}

fn dispatch(
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    in_buf: &ProtocolObject<dyn MTLBuffer>,
    out_buf: &ProtocolObject<dyn MTLBuffer>,
) {
    let cmd = queue.commandBuffer().expect("commandBuffer returned nil");
    let encoder = cmd
        .computeCommandEncoder()
        .expect("computeCommandEncoder returned nil");

    encoder.setComputePipelineState(pipeline);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(in_buf), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(out_buf), 0, 1);
    }

    let grid = MTLSize {
        width: N,
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
}
