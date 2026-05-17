//! Stage 1 of the development spike — see `CLAUDE.md`.
//!
//! Compiles a trivial inline Metal kernel at runtime, dispatches it on 1024
//! `f32` elements via shared-memory `MTLBuffer`s, and verifies the result.
//!
//! Acceptance: passes under `cargo test --release`, and a single dispatch
//! round-trip (after JIT warm-up) completes in well under 5 ms.
//!
//! Post-spike: uses [`Device`] + [`Buffer`] from the crate. The library
//! compile / pipeline lookup / dispatch path still lives inline in this
//! file pending [`KernelLibrary`] and [`kernels`] modules in later
//! commits of the port series.

#![cfg(target_os = "macos")]
// Test code values clarity over the pedantic numeric-cast lints.
// Bit-exact f32 comparison is intentional here: `add_one` on integer-valued
// inputs produces results that are exactly representable.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::float_cmp
)]

use std::time::Instant;

use cider_press_kernels::{Buffer, Device};
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLDevice, MTLLibrary, MTLSize,
};

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

#[test]
fn add_one_round_trip() {
    let device = Device::system_default().expect("no Metal device available");

    // KernelLibrary + per-kernel dispatch fns are upcoming in this port
    // series; for now compile inline against the underlying MTLDevice.
    let source = NSString::from_str(KERNEL_SOURCE);
    let library = device
        .metal_device()
        .newLibraryWithSource_options_error(&source, None)
        .expect("kernel compilation failed");
    let function = library
        .newFunctionWithName(&NSString::from_str("add_one"))
        .expect("add_one not found in compiled library");
    let pipeline = device
        .metal_device()
        .newComputePipelineStateWithFunction_error(&function)
        .expect("failed to build compute pipeline");

    let input: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let in_buf: Buffer<f32> = device.upload(&input).expect("upload input");
    let mut out_buf: Buffer<f32> = device.alloc_buffer(N).expect("alloc output");

    // Warm-up dispatch so the timing below excludes one-time costs
    // (driver setup, GPU power-up, command-buffer caches).
    dispatch(&device, &pipeline, &in_buf, &out_buf);

    let start = Instant::now();
    dispatch(&device, &pipeline, &in_buf, &out_buf);
    let elapsed = start.elapsed();

    // SAFETY: dispatch above is sync (commit + waitUntilCompleted), so the
    // GPU is done writing out_buf before we read.
    let out = unsafe { out_buf.as_mut_slice() };
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
    device: &Device,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    in_buf: &Buffer<f32>,
    out_buf: &Buffer<f32>,
) {
    let cmd = device
        .metal_queue()
        .commandBuffer()
        .expect("commandBuffer returned nil");
    let encoder = cmd
        .computeCommandEncoder()
        .expect("computeCommandEncoder returned nil");

    encoder.setComputePipelineState(pipeline);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(in_buf.metal_buffer()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(out_buf.metal_buffer()), 0, 1);
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
