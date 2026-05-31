//! qmv GPU-execution + effective-bandwidth benchmark.
//!
//! Times `kernels::qmv::affine_qmv_bf16` at the five Qwen2.5-0.5B decode
//! shapes plus a forced-fast control, reporting GPU-execution ns (via
//! stage-boundary counters) and absolute effective GB/s per shape. The
//! companion script `scripts/measure_qmv_mlx.py` provides the MLX
//! comparison and is the primary comparator for localizing the decode qmv
//! gap. See `docs/superpowers/specs/2026-05-31-qmv-audit-design.md`.
//!
//! The qmv parity test already validated correctness; this test only
//! measures, so buffers are left at whatever values
//! `Device::alloc_buffer` produced (Metal does not zero shared-storage
//! allocations). The qmv kernel is data-independent in its runtime
//! cost, so the resulting outputs are garbage but the timings are still
//! meaningful.

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

/// Bytes moved by one `affine_qmv_bf16` at `(k, n)`: packed 4-bit weights
/// dominate, plus bf16 scales+biases, the x vector, and the y output.
fn qmv_bytes(k: usize, n: usize, group_size: usize, bits: usize) -> usize {
    let weights = n * k * bits / 8;
    let scales_biases = 2 * (n * (k / group_size)) * std::mem::size_of::<bf16>();
    let x = k * std::mem::size_of::<bf16>();
    let y = n * std::mem::size_of::<bf16>();
    weights + scales_biases + x + y
}

/// Average GPU-execution time per `affine_qmv_bf16` dispatch, in
/// nanoseconds, via stage-boundary counter sampling. All `reps`
/// dispatches share one command buffer — the same back-to-back regime as
/// a decode step. Returns `None` when the device lacks counter sampling
/// (caller falls back to the wall-clock number).
#[allow(clippy::too_many_arguments)]
fn qmv_gpu_ns(
    device: &Device,
    library: &KernelLibrary,
    w: &Buffer<u32>,
    s: &Buffer<bf16>,
    b: &Buffer<bf16>,
    x: &Buffer<bf16>,
    y: &mut Buffer<bf16>,
    group_size: u32,
    bits: u32,
    reps: usize,
) -> Option<f64> {
    if !device.supports_stage_boundary_sampling() {
        return None;
    }
    let mut cmds = device.commands_profiled(reps).expect("profiled commands");
    for _ in 0..reps {
        cmds.begin_profiled_op("qmv");
        kernels::qmv::affine_qmv_bf16(&mut cmds, library, w, s, b, x, y, group_size, bits)
            .expect("dispatch");
    }
    let segments = cmds.commit_wait_resolve().expect("resolve");
    let period_ns = device.gpu_timestamp_period_ns();
    let total_ns: f64 = segments
        .iter()
        .map(|seg| seg.ticks() as f64 * period_ns)
        .sum();
    Some(total_ns / segments.len() as f64)
}

fn bench(device: &Device, library: &KernelLibrary, k: usize, n: usize) {
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
    let wall_us = (start.elapsed().as_secs_f64() * 1e6) / TIMED as f64;

    let fast = n % 8 == 0 && k % 512 == 0;
    let variant = if fast { "fast" } else { "gen " };
    let bytes = qmv_bytes(k, n, GROUP_SIZE as usize, BITS as usize);

    // 256 dispatches share one command buffer (the decode regime); enough to average out, few enough to bound counter-segment overhead.
    match qmv_gpu_ns(device, library, &w, &s, &b, &x, &mut y, GROUP_SIZE, BITS, 256) {
        Some(gpu_ns) => {
            let eff_gbps = bytes as f64 / (gpu_ns / 1e9) / 1e9;
            eprintln!(
                "  {k:>5}x{n:<6} {variant}  wall {wall_us:>7.2}us  gpu {gpu_ns:>8.0}ns  \
                 {eff_gbps:>6.0} GB/s",
            );
        }
        None => {
            eprintln!(
                "  {k:>5}x{n:<6} {variant}  wall {wall_us:>7.2}us  gpu  (no counters)",
            );
        }
    }
}

#[test]
#[ignore = "perf benchmark; run with --ignored"]
fn perf_affine_qmv_fast_bf16_gs64_b4() {
    let device = Device::system_default().expect("Metal device");
    let library = KernelLibrary::quantized(&device).expect("compile quantized.metal");

    eprintln!();
    eprintln!(
        "qmv GPU-execution benchmark: affine_qmv_bf16 (gs={GROUP_SIZE}, bits={BITS}, \
         warmup={WARMUP}, wall-timed={TIMED})",
    );
    eprintln!("  shape        var   wall          gpu          eff BW");
    // Qwen2.5-0.5B decode qmv shapes (K=in, N=out), then forced-fast control.
    let shapes: &[(usize, usize, &str)] = &[
        (896, 896, "q/o_proj"),
        (896, 128, "k/v_proj"),
        (896, 4864, "gate/up_proj"),
        (4864, 896, "down_proj"),
        (896, 151_936, "lm_head"),
        (1024, 1024, "forced-fast control"),
    ];
    for &(k, n, label) in shapes {
        eprintln!("  -- {label}");
        bench(&device, &library, k, n);
    }
}
