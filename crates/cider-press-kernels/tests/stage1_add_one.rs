//! Stage 1 of the development spike — see `CLAUDE.md`.
//!
//! Compiles a trivial inline Metal kernel at runtime, dispatches it on 1024
//! `f32` elements via shared-memory `MTLBuffer`s, and verifies the result.
//!
//! Acceptance: passes under `cargo test --release`, and a single dispatch
//! round-trip (after JIT warm-up) completes in well under 5 ms.
//!
//! Post-spike: uses `Device` + `Buffer` + `KernelLibrary` from the crate.
//! The dispatch path itself still lives inline pending the `Commands`
//! handle and `kernels::` modules in upcoming commits.

#![cfg(target_os = "macos")]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::float_cmp
)]

use std::time::Instant;

use cider_press_kernels::{Buffer, Device, KernelLibrary, Pipeline};
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLSize,
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
    let library = KernelLibrary::from_source(&device, KERNEL_SOURCE).expect("compile kernel");
    let pipeline = library.pipeline("add_one").expect("look up add_one");

    let input: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let in_buf: Buffer<f32> = device.upload(&input).expect("upload input");
    let mut out_buf: Buffer<f32> = device.alloc_buffer(N).expect("alloc output");

    // Warm-up dispatch — excludes one-time costs (driver setup,
    // pipeline cache miss, GPU power-up) from the timing below.
    dispatch(&device, &pipeline, &in_buf, &out_buf);

    let start = Instant::now();
    dispatch(&device, &pipeline, &in_buf, &out_buf);
    let elapsed = start.elapsed();

    // SAFETY: dispatch() above is sync; GPU has finished writing out_buf.
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

fn dispatch(device: &Device, pipeline: &Pipeline, in_buf: &Buffer<f32>, out_buf: &Buffer<f32>) {
    let cmd = device
        .metal_queue()
        .commandBuffer()
        .expect("commandBuffer returned nil");
    let encoder = cmd
        .computeCommandEncoder()
        .expect("computeCommandEncoder returned nil");

    encoder.setComputePipelineState(pipeline.metal_pipeline_state());
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
