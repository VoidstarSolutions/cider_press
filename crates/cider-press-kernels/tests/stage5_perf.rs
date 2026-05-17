//! Stage 5 of the development spike — see `CLAUDE.md`.
//!
//! Times 1000 dispatches of `kernels::qmv::affine_qmv_bf16` at two
//! shapes (parity dim K=N=512, production dim K=N=4096) and reports
//! per-dispatch latency. The companion script
//! `scripts/measure_stage5_mlx.py` does the same against MLX Python;
//! the ratio between the two is the spike's headline number.
//!
//! Stage 4 already validated correctness; this test only measures, so
//! buffers are zero-initialized.

#![cfg(target_os = "macos")]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::many_single_char_names
)]

use std::time::Instant;

use cider_press_kernels::{Buffer, Device, KernelLibrary, kernels};
use half::bf16;

const GROUP_SIZE: u32 = 64;
const BITS: u32 = 4;
const WARMUP: usize = 50;
const TIMED: usize = 1000;

#[test]
#[ignore = "perf benchmark; run with --ignored"]
fn perf_affine_qmv_fast_bf16_gs64_b4() {
    let device = Device::system_default().expect("Metal device");
    let library = KernelLibrary::quantized(&device).expect("compile quantized.metal");

    eprintln!();
    eprintln!(
        "stage 5: affine_qmv_bf16 dispatch latency (gs={GROUP_SIZE}, bits={BITS}, \
         warmup={WARMUP}, timed={TIMED})",
    );
    eprintln!("  shape (K=N)   total          per-dispatch    rate");
    for &dim in &[512_usize, 4096_usize] {
        bench(&device, &library, dim);
    }
}

fn bench(device: &Device, library: &KernelLibrary, dim: usize) {
    let k = dim;
    let n = dim;

    let w: Buffer<u32> = device
        .alloc_buffer(n * (k * BITS as usize / 32))
        .expect("alloc w");
    let s: Buffer<bf16> = device
        .alloc_buffer(n * (k / GROUP_SIZE as usize))
        .expect("alloc scales");
    let b: Buffer<bf16> = device
        .alloc_buffer(n * (k / GROUP_SIZE as usize))
        .expect("alloc biases");
    let x: Buffer<bf16> = device.alloc_buffer(k).expect("alloc x");
    let mut y: Buffer<bf16> = device.alloc_buffer(n).expect("alloc y");

    let dispatch = |y: &mut Buffer<bf16>| {
        let mut cmds = device.commands().expect("commands");
        kernels::qmv::affine_qmv_bf16(&mut cmds, library, &w, &s, &b, &x, y, GROUP_SIZE, BITS)
            .expect("dispatch");
        cmds.commit_and_wait().expect("commit");
    };

    for _ in 0..WARMUP {
        dispatch(&mut y);
    }
    let start = Instant::now();
    for _ in 0..TIMED {
        dispatch(&mut y);
    }
    let elapsed = start.elapsed();
    let per_us = (elapsed.as_secs_f64() * 1e6) / TIMED as f64;
    let rate = TIMED as f64 / elapsed.as_secs_f64();
    eprintln!("  {k:>5}x{n:<5}   {elapsed:>10.2?}   {per_us:>9.2} us   {rate:>9.0} disp/s");
}
