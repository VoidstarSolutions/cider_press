//! qmm GPU-execution + effective-bandwidth benchmark.
//!
//! Times `kernels::qmm::affine_qmm_t_bf16` at the eight Qwen2.5-0.5B prefill
//! shapes (M=39), reporting GPU-execution ns (via stage-boundary counters) and
//! absolute effective GB/s per shape. The companion script
//! `scripts/measure_qmm_mlx.py` provides the MLX comparison and is the
//! primary comparator for localising the prefill qmm gap.
//!
//! The qmm parity test already validated correctness; this test only
//! measures, so buffers are left at whatever values
//! `Device::alloc_buffer` produced (Metal does not zero shared-storage
//! allocations). The qmm kernel is data-independent in its runtime
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
const TIMED: usize = 200;
const M: usize = 39;

/// Bytes moved by one `affine_qmm_t_bf16` at `(m, k, n)`: packed 4-bit
/// weights, bf16 scales+biases, the x matrix, and the y output.
fn qmm_bytes(m: usize, k: usize, n: usize, group_size: usize, bits: usize) -> usize {
    let weights = n * k * bits / 8;
    let scales_biases = 2 * (n * (k / group_size)) * std::mem::size_of::<bf16>();
    let x = m * k * std::mem::size_of::<bf16>();
    let y = m * n * std::mem::size_of::<bf16>();
    weights + scales_biases + x + y
}

/// Average GPU-execution time per `affine_qmm_t_bf16` dispatch, in
/// nanoseconds, via stage-boundary counter sampling. All `reps`
/// dispatches share one command buffer. Returns `None` when the device
/// lacks counter sampling (caller falls back to the wall-clock number).
#[allow(clippy::too_many_arguments)]
fn qmm_gpu_ns(
    device: &Device,
    library: &KernelLibrary,
    w: &Buffer<u32>,
    s: &Buffer<bf16>,
    b: &Buffer<bf16>,
    x: &Buffer<bf16>,
    y: &mut Buffer<bf16>,
    m: usize,
    n: usize,
    k: usize,
    group_size: u32,
    bits: u32,
    reps: usize,
) -> Option<f64> {
    if !device.supports_stage_boundary_sampling() {
        return None;
    }
    let mut cmds = device.commands_profiled(reps).expect("profiled commands");
    for _ in 0..reps {
        cmds.begin_profiled_op("qmm");
        kernels::qmm::affine_qmm_t_bf16(
            &mut cmds, library, w, s, b, x, y, m, n, k, group_size, bits,
        )
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

fn bench(device: &Device, library: &KernelLibrary, k: usize, n: usize, label: &str) {
    let w: Buffer<u32> = device
        .alloc_buffer(n * (k * BITS as usize / 32))
        .expect("alloc w");
    let s: Buffer<bf16> = device
        .alloc_buffer(n * (k / GROUP_SIZE as usize))
        .expect("alloc scales");
    let b: Buffer<bf16> = device
        .alloc_buffer(n * (k / GROUP_SIZE as usize))
        .expect("alloc biases");
    let x: Buffer<bf16> = device.alloc_buffer(M * k).expect("alloc x");
    let mut y: Buffer<bf16> = device.alloc_buffer(M * n).expect("alloc y");

    let dispatch = |y: &mut Buffer<bf16>| {
        let mut cmds = device.commands().expect("commands");
        kernels::qmm::affine_qmm_t_bf16(
            &mut cmds, library, &w, &s, &b, &x, y, M, n, k, GROUP_SIZE, BITS,
        )
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

    let bytes = qmm_bytes(M, k, n, GROUP_SIZE as usize, BITS as usize);

    // 64 dispatches per profiled command buffer — enough to average out noise,
    // few enough to bound counter-segment overhead for large shapes.
    match qmm_gpu_ns(
        device, library, &w, &s, &b, &x, &mut y, M, n, k, GROUP_SIZE, BITS, 64,
    ) {
        Some(gpu_ns) => {
            let eff_gbps = bytes as f64 / (gpu_ns / 1e9) / 1e9;
            eprintln!(
                "  {label:<10}  M={M:>2} K={k:>4} N={n:>6}  wall {wall_us:>8.2}us  \
                 gpu {gpu_ns:>8.0}ns  {eff_gbps:>6.0} GB/s"
            );
        }
        None => {
            eprintln!(
                "  {label:<10}  M={M:>2} K={k:>4} N={n:>6}  wall {wall_us:>8.2}us  \
                 gpu  (no counters)"
            );
        }
    }
}

#[test]
#[ignore = "perf benchmark; run with --ignored"]
fn perf_affine_qmm_t_bf16_prefill_shapes() {
    let device = Device::system_default().expect("Metal device");
    let library = KernelLibrary::quantized(&device).expect("compile quantized.metal");

    eprintln!();
    eprintln!(
        "qmm GPU-execution benchmark: affine_qmm_t_bf16 (M={M}, gs={GROUP_SIZE}, bits={BITS}, \
         warmup={WARMUP}, wall-timed={TIMED})",
    );
    eprintln!("  shape                           wall            gpu            eff BW");
    // Qwen2.5-0.5B prefill qmm shapes (K=in, N=out, M=T=39).
    let shapes: &[(usize, usize, &str)] = &[
        (896, 896, "q_proj"),
        (896, 128, "k_proj"),
        (896, 128, "v_proj"),
        (896, 896, "o_proj"),
        (896, 4864, "gate"),
        (896, 4864, "up"),
        (4864, 896, "down"),
        (896, 151_936, "lm_head"),
    ];
    for &(k, n, label) in shapes {
        bench(&device, &library, k, n, label);
    }
}
