//! Stage 5 of the development spike — see `CLAUDE.md`.
//!
//! Times 1000 dispatches of `affine_qmv_fast_bfloat16_t_gs_64_b_4_batch_0`
//! at two shapes (parity dim K=N=512 and production dim K=N=4096) and
//! reports per-dispatch latency. The companion script
//! `scripts/measure_stage5_mlx.py` runs the equivalent timing under
//! MLX Python; comparing the two numbers tells us whether MLX's perf
//! advantage on Apple Silicon lives in the kernels (it doesn't — Stage 4
//! got bit-exact output) or in the graph/dispatch layer.
//!
//! Correctness is *not* asserted here; buffers are zero-initialized
//! since the kernel runtime is data-independent at these sizes.

#![cfg(target_os = "macos")]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::too_many_lines
)]

use std::ffi::c_void;
use std::ptr::NonNull;
use std::time::Instant;

use cider_press_kernels::{Buffer, Device, KernelLibrary, Pipeline};
use half::bf16;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLSize,
};

const KERNEL_NAME: &str = "affine_qmv_fast_bfloat16_t_gs_64_b_4_batch_0";
const GROUP_SIZE: usize = 64;
const BITS: usize = 4;
const BN: i32 = 8;
const BK: i32 = 32;
const WARMUP: usize = 50;
const TIMED: usize = 1000;

#[test]
#[ignore = "perf benchmark; run with --ignored"]
fn perf_affine_qmv_fast_bf16_gs64_b4() {
    let device = Device::system_default().expect("Metal device");
    let library = KernelLibrary::quantized(&device).expect("compile quantized.metal");
    let pipeline = library.pipeline(KERNEL_NAME).expect("affine_qmv_fast");

    eprintln!();
    eprintln!(
        "stage 5: qmv dispatch latency (kernel={KERNEL_NAME}, warmup={WARMUP}, timed={TIMED})",
    );
    eprintln!("  shape (K=N)   total          per-dispatch    rate");
    for &dim in &[512_i32, 4096_i32] {
        bench(&device, &pipeline, dim);
    }
}

fn bench(device: &Device, pipeline: &Pipeline, dim: i32) {
    let k = dim;
    let n = dim;
    let k_usize = k as usize;
    let n_usize = n as usize;

    // Stage 4 already validated correctness; zero-init is fine for timing.
    let w_buf: Buffer<u32> = device
        .alloc_buffer(n_usize * (k_usize * BITS / 32))
        .expect("alloc w");
    let s_buf: Buffer<bf16> = device
        .alloc_buffer(n_usize * (k_usize / GROUP_SIZE))
        .expect("alloc scales");
    let b_buf: Buffer<bf16> = device
        .alloc_buffer(n_usize * (k_usize / GROUP_SIZE))
        .expect("alloc biases");
    let x_buf: Buffer<bf16> = device.alloc_buffer(k_usize).expect("alloc x");
    let y_buf: Buffer<bf16> = device.alloc_buffer(n_usize).expect("alloc y");

    let dispatch = || {
        let cmd = device.metal_queue().commandBuffer().expect("commandBuffer");
        let encoder = cmd.computeCommandEncoder().expect("encoder");
        encoder.setComputePipelineState(pipeline.metal_pipeline_state());
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(w_buf.metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(s_buf.metal_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(b_buf.metal_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(x_buf.metal_buffer()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(y_buf.metal_buffer()), 0, 4);
            let k_ptr: NonNull<c_void> = NonNull::from(&k).cast();
            let n_ptr: NonNull<c_void> = NonNull::from(&n).cast();
            encoder.setBytes_length_atIndex(k_ptr, std::mem::size_of::<i32>(), 5);
            encoder.setBytes_length_atIndex(n_ptr, std::mem::size_of::<i32>(), 6);
        }
        let grid = MTLSize {
            width: 1,
            height: ((n + BN - 1) / BN) as usize,
            depth: 1,
        };
        let threadgroup = MTLSize {
            width: BK as usize,
            height: 2,
            depth: 1,
        };
        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, threadgroup);
        encoder.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
    };

    for _ in 0..WARMUP {
        dispatch();
    }
    let start = Instant::now();
    for _ in 0..TIMED {
        dispatch();
    }
    let elapsed = start.elapsed();
    let per_us = (elapsed.as_secs_f64() * 1e6) / TIMED as f64;
    let rate = TIMED as f64 / elapsed.as_secs_f64();
    eprintln!("  {k:>5}x{n:<5}   {elapsed:>10.2?}   {per_us:>9.2} us   {rate:>9.0} disp/s");
}
