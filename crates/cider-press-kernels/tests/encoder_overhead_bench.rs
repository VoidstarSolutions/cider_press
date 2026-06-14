//! Microbench: per-dispatch GPU cost of a dependent kernel chain under
//! competing encoder models.
//!
//! Decode is a ~530-dispatch dependent chain per token. cider-press encodes
//! it as ONE serial compute encoder; MLX encodes ~10 concurrent-dispatch
//! encoders with explicit `memoryBarrier(BarrierScopeBuffers)` between
//! hazarded dispatches. The xctrace comparison (`docs/inference/execution-model.md` follow-up)
//! shows both runtimes GPU-dense running the same kernels, with cider ~300 µs
//! slower per token — i.e. ~0.5–1 µs/dispatch of extra serialization. This
//! bench isolates that per-link cost with identical kernels and a strictly
//! dependent ping-pong chain, timed by command-buffer GPU timestamps
//! (unperturbed — no counter sampling).
//!
//! Run: cargo test -p cider-press-kernels --release --test
//! `encoder_overhead_bench` -- --ignored --nocapture

#![cfg(target_os = "macos")]
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use cider_press_kernels::{Buffer, Device, KernelLibrary, Pipeline};
use objc2_metal::{
    MTLBarrierScope, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLComputePassDescriptor, MTLDispatchType, MTLSize,
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

/// Hidden-dim-sized elementwise op: matches the decode residual/bias adds.
const N: usize = 896;
/// Dependent-chain length: matches decode's ~530 dispatches/token.
const CHAIN: usize = 512;
const REPS: usize = 7;

struct Ctx<'d> {
    device: &'d Device,
    pipeline: Pipeline,
    ping: Buffer<f32>,
    pong: Buffer<f32>,
}

fn encode_dispatch(
    ctx: &Ctx<'_>,
    encoder: &objc2::runtime::ProtocolObject<dyn MTLComputeCommandEncoder>,
    i: usize,
) {
    let (src, dst) = if i % 2 == 0 {
        (&ctx.ping, &ctx.pong)
    } else {
        (&ctx.pong, &ctx.ping)
    };
    encoder.setComputePipelineState(ctx.pipeline.metal_pipeline_state());
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(src.metal_buffer()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(dst.metal_buffer()), 0, 1);
    }
    let grid = MTLSize {
        width: N,
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: 64,
        height: 1,
        depth: 1,
    };
    encoder.dispatchThreads_threadsPerThreadgroup(grid, tg);
}

/// One command buffer, one serial encoder (the production cider-press model).
fn run_serial(ctx: &Ctx<'_>) -> f64 {
    let cmd = ctx.device.metal_queue().commandBuffer().expect("cmdbuf");
    let enc = cmd.computeCommandEncoder().expect("encoder");
    for i in 0..CHAIN {
        encode_dispatch(ctx, &enc, i);
    }
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();
    cmd.GPUEndTime() - cmd.GPUStartTime()
}

/// One concurrent-dispatch encoder + a global buffer-scope memory barrier
/// before every dependent dispatch (MLX's in-chain hazard model).
fn run_concurrent_barrier(ctx: &Ctx<'_>) -> f64 {
    let cmd = ctx.device.metal_queue().commandBuffer().expect("cmdbuf");
    let descriptor = MTLComputePassDescriptor::computePassDescriptor();
    descriptor.setDispatchType(MTLDispatchType::Concurrent);
    let enc = cmd
        .computeCommandEncoderWithDescriptor(&descriptor)
        .expect("encoder");
    for i in 0..CHAIN {
        if i > 0 {
            enc.memoryBarrierWithScope(MTLBarrierScope::Buffers);
        }
        encode_dispatch(ctx, &enc, i);
    }
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();
    cmd.GPUEndTime() - cmd.GPUStartTime()
}

/// One concurrent encoder, no barriers. Races on the ping-pong buffers are
/// irrelevant — this measures the no-dependency launch floor, not values.
fn run_concurrent_free(ctx: &Ctx<'_>) -> f64 {
    let cmd = ctx.device.metal_queue().commandBuffer().expect("cmdbuf");
    let descriptor = MTLComputePassDescriptor::computePassDescriptor();
    descriptor.setDispatchType(MTLDispatchType::Concurrent);
    let enc = cmd
        .computeCommandEncoderWithDescriptor(&descriptor)
        .expect("encoder");
    for i in 0..CHAIN {
        encode_dispatch(ctx, &enc, i);
    }
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();
    cmd.GPUEndTime() - cmd.GPUStartTime()
}

/// The chain split across `splits` command buffers (serial encoder each),
/// committed back-to-back, waited at the end — MLX's multi-encoder
/// structure. Tracked (default) hazard mode orders the buffers; the
/// question is whether the seams overlap on the GPU.
fn run_split_cmdbufs(ctx: &Ctx<'_>, splits: usize) -> f64 {
    let mut cmds = Vec::with_capacity(splits);
    for s in 0..splits {
        let cmd = ctx.device.metal_queue().commandBuffer().expect("cmdbuf");
        let enc = cmd.computeCommandEncoder().expect("encoder");
        // Balanced ranges covering all CHAIN dispatches even when splits
        // doesn't divide CHAIN (CHAIN / splits alone would drop the
        // remainder and under-count the per-dispatch normalization).
        for i in s * CHAIN / splits..(s + 1) * CHAIN / splits {
            encode_dispatch(ctx, &enc, i);
        }
        enc.endEncoding();
        cmd.commit();
        cmds.push(cmd);
    }
    cmds.last().expect("nonempty").waitUntilCompleted();
    cmds.last().expect("nonempty").GPUEndTime() - cmds.first().expect("nonempty").GPUStartTime()
}

/// The chain as `splits` serial encoders within ONE command buffer.
fn run_split_encoders(ctx: &Ctx<'_>, splits: usize) -> f64 {
    let cmd = ctx.device.metal_queue().commandBuffer().expect("cmdbuf");
    for s in 0..splits {
        let enc = cmd.computeCommandEncoder().expect("encoder");
        // Balanced ranges covering all CHAIN dispatches (see run_split_cmdbufs).
        for i in s * CHAIN / splits..(s + 1) * CHAIN / splits {
            encode_dispatch(ctx, &enc, i);
        }
        enc.endEncoding();
    }
    cmd.commit();
    cmd.waitUntilCompleted();
    cmd.GPUEndTime() - cmd.GPUStartTime()
}

fn bench(name: &str, mut f: impl FnMut() -> f64) {
    // Warm-up.
    f();
    let mut samples: Vec<f64> = (0..REPS).map(|_| f()).collect();
    samples.sort_by(f64::total_cmp);
    let median = samples[REPS / 2];
    let floor = samples[0];
    eprintln!(
        "{name:24} gpu total median {:8.1} us  floor {:8.1} us  median/dispatch {:6.3} us",
        median * 1e6,
        floor * 1e6,
        median * 1e6 / CHAIN as f64,
    );
}

#[test]
#[ignore = "manual microbench; run with --ignored --nocapture"]
fn encoder_overhead() {
    let device = Device::system_default().expect("no Metal device");
    let library = KernelLibrary::from_source(&device, KERNEL_SOURCE).expect("compile");
    let pipeline = library.pipeline("add_one").expect("pipeline");
    let input: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let ping: Buffer<f32> = device.upload(&input).expect("upload");
    let scratch: Buffer<f32> = device.alloc_buffer(N).expect("alloc");
    let ctx = Ctx {
        device: &device,
        pipeline,
        ping,
        pong: scratch,
    };

    eprintln!("dependent chain: {CHAIN} dispatches of add_one[{N}] (f32)");
    bench("serial (production)", || run_serial(&ctx));
    bench("concurrent + barrier", || run_concurrent_barrier(&ctx));
    bench("concurrent, no barrier", || run_concurrent_free(&ctx));
    bench("split 10 cmdbufs", || run_split_cmdbufs(&ctx, 10));
    bench("split 10 encoders", || run_split_encoders(&ctx, 10));

    let _ = &ctx.device;
}
