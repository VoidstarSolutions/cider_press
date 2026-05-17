//! Stage 5 of the development spike — see `CLAUDE.md`.
//!
//! Times 1000 dispatches of `affine_qmv_fast_bfloat16_t_gs_64_b_4_batch_0`
//! at two shapes (parity dim K=N=512 and production dim K=N=4096) and
//! reports per-dispatch latency. The companion script
//! `scripts/measure_stage5_mlx.py` runs the equivalent timing under
//! MLX Python; comparing the two numbers tells us whether MLX's perf
//! advantage on Apple Silicon lives in the kernels (it doesn't — we
//! got bit-exact output in Stage 4) or in the graph/dispatch layer.
//!
//! Correctness is *not* asserted here. Stage 4 already proved bit-exact
//! parity; for timing we only care that the dispatch geometry is right.
//! Buffers are filled with deterministic but otherwise meaningless data.

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

const QUANTIZED_SOURCE: &str = include_str!(concat!(env!("OUT_DIR"), "/quantized_inlined.metal"));

const KERNEL_NAME: &str = "affine_qmv_fast_bfloat16_t_gs_64_b_4_batch_0";
const GROUP_SIZE: usize = 64;
const BITS: usize = 4;
const BN: i32 = 8;
const BK: i32 = 32;
const WARMUP: usize = 50;
const TIMED: usize = 1000;

fn mlx_source_unavailable() -> bool {
    QUANTIZED_SOURCE
        .trim()
        .starts_with("// MLX checkout not found")
}

// Perf measurement, not a regression test: 2 shapes × 1050 sync
// dispatches each adds ~1 s of noise to default `cargo test`. Run
// explicitly with `cargo test --release --test stage5_perf -- --ignored
// --nocapture`.
#[test]
#[ignore = "perf benchmark; run with --ignored"]
fn perf_affine_qmv_fast_bf16_gs64_b4() {
    if mlx_source_unavailable() {
        eprintln!("skipping: MLX checkout not found at build time");
        return;
    }

    let device = MTLCreateSystemDefaultDevice().expect("no Metal device");
    let queue = device.newCommandQueue().expect("command queue");
    let source = NSString::from_str(QUANTIZED_SOURCE);
    let library = device
        .newLibraryWithSource_options_error(&source, None)
        .unwrap_or_else(|err| panic!("quantized.metal compile: {err}"));
    let ns_name = NSString::from_str(KERNEL_NAME);
    let function = library
        .newFunctionWithName(&ns_name)
        .unwrap_or_else(|| panic!("kernel {KERNEL_NAME} not in library"));
    let pipeline = device
        .newComputePipelineStateWithFunction_error(&function)
        .unwrap_or_else(|err| panic!("pipeline {KERNEL_NAME}: {err}"));

    eprintln!();
    eprintln!(
        "stage 5: qmv dispatch latency (kernel={KERNEL_NAME}, warmup={WARMUP}, timed={TIMED})"
    );
    eprintln!("  shape (K=N)   total          per-dispatch    rate");
    for &dim in &[512_i32, 4096_i32] {
        bench(&device, &queue, &pipeline, dim);
    }
}

fn bench(
    device: &ProtocolObject<dyn MTLDevice>,
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    dim: i32,
) {
    let k = dim;
    let n = dim;
    let k_usize = k as usize;
    let n_usize = n as usize;

    // Allocate buffers at the right shape. Random/garbage data is fine —
    // Stage 4 already validated correctness; we're only measuring dispatch.
    let w_bytes = n_usize * (k_usize * BITS / 32) * std::mem::size_of::<u32>();
    let s_bytes = n_usize * (k_usize / GROUP_SIZE) * std::mem::size_of::<u16>();
    let b_bytes = s_bytes;
    let x_bytes = k_usize * std::mem::size_of::<u16>();
    let y_bytes = n_usize * std::mem::size_of::<u16>();

    let w_buf = alloc_zeroed(device, w_bytes);
    let s_buf = alloc_zeroed(device, s_bytes);
    let b_buf = alloc_zeroed(device, b_bytes);
    let x_buf = alloc_zeroed(device, x_bytes);
    let y_buf = alloc_zeroed(device, y_bytes);

    let dispatch = || {
        let cmd = queue.commandBuffer().expect("commandBuffer");
        let encoder = cmd.computeCommandEncoder().expect("encoder");
        encoder.setComputePipelineState(pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&w_buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&s_buf), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&b_buf), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&x_buf), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(&y_buf), 0, 4);
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
    let per = elapsed / (TIMED as u32);
    let per_us = per.as_secs_f64() * 1e6;
    let rate = TIMED as f64 / elapsed.as_secs_f64();
    eprintln!("  {k:>5}x{n:<5}   {elapsed:>10.2?}   {per_us:>9.2} us   {rate:>9.0} disp/s");
}

fn alloc_zeroed(
    device: &ProtocolObject<dyn MTLDevice>,
    bytes: usize,
) -> Retained<ProtocolObject<dyn MTLBuffer>> {
    let buf = device
        .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
        .expect("allocate buffer");
    unsafe {
        let ptr: NonNull<u8> = buf.contents().cast();
        std::ptr::write_bytes(ptr.as_ptr(), 0, bytes);
    }
    buf
}
