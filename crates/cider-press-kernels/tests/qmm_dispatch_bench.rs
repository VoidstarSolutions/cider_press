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

/// Average GPU-execution time per `affine_qmm_t_bf16` (or, with `nax`,
/// `affine_qmm_t_nax_bf16`) dispatch, in nanoseconds, via stage-boundary
/// counter sampling. All `reps` dispatches share one command buffer.
/// Returns `None` when the device lacks counter sampling (caller falls back
/// to the wall-clock number).
#[allow(clippy::too_many_arguments)]
fn qmm_gpu_ns(
    device: &Device,
    library: &KernelLibrary,
    nax: bool,
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
        if nax {
            kernels::qmm::affine_qmm_t_nax_bf16(
                &mut cmds, library, w, s, b, x, y, m, n, k, group_size, bits,
            )
        } else {
            kernels::qmm::affine_qmm_t_bf16(
                &mut cmds, library, w, s, b, x, y, m, n, k, group_size, bits,
            )
        }
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
        device, library, false, &w, &s, &b, &x, &mut y, M, n, k, GROUP_SIZE, BITS, 64,
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

/// GPU-execution time per dispatch (64 per profiled command buffer, so the
/// commit/wait floor is excluded) at prefill-sized M for both kernels on the
/// Qwen3-Embedding-0.6B and Qwen2.5-0.5B projection shapes; wall-clock per
/// isolated dispatch alongside for reference.
fn bench_m512(
    device: &Device,
    classic: &KernelLibrary,
    nax: Option<&KernelLibrary>,
    k: usize,
    n: usize,
    label: &str,
) {
    const M512: usize = 512;
    let w: Buffer<u32> = device
        .alloc_buffer(n * (k * BITS as usize / 32))
        .expect("alloc w");
    let s: Buffer<bf16> = device
        .alloc_buffer(n * (k / GROUP_SIZE as usize))
        .expect("alloc scales");
    let b: Buffer<bf16> = device
        .alloc_buffer(n * (k / GROUP_SIZE as usize))
        .expect("alloc biases");
    let x: Buffer<bf16> = device.alloc_buffer(M512 * k).expect("alloc x");
    let mut y: Buffer<bf16> = device.alloc_buffer(M512 * n).expect("alloc y");
    let flops = 2.0 * M512 as f64 * k as f64 * n as f64;

    let mut measure = |use_nax: bool| -> Option<(f64, Option<f64>)> {
        let library = if use_nax { nax? } else { classic };
        let dispatch = |y: &mut Buffer<bf16>| {
            let mut cmds = device.commands().expect("commands");
            if use_nax {
                kernels::qmm::affine_qmm_t_nax_bf16(
                    &mut cmds, library, &w, &s, &b, &x, y, M512, n, k, GROUP_SIZE, BITS,
                )
            } else {
                kernels::qmm::affine_qmm_t_bf16(
                    &mut cmds, library, &w, &s, &b, &x, y, M512, n, k, GROUP_SIZE, BITS,
                )
            }
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
        let wall_s = start.elapsed().as_secs_f64() / TIMED as f64;
        let gpu_s = qmm_gpu_ns(
            device, library, use_nax, &w, &s, &b, &x, &mut y, M512, n, k, GROUP_SIZE, BITS, 64,
        )
        .map(|ns| ns / 1e9);
        Some((wall_s, gpu_s))
    };
    let fmt = |(wall_s, gpu_s): (f64, Option<f64>)| match gpu_s {
        Some(g) => format!(
            "gpu {:>7.1}us {:>5.1} TFLOPS (wall {:>7.1}us)",
            g * 1e6,
            flops / g / 1e12,
            wall_s * 1e6
        ),
        None => format!("gpu (no counters) (wall {:>7.1}us)", wall_s * 1e6),
    };
    let classic_m = measure(false).expect("classic");
    let line = format!(
        "  {label:<11} M=512 K={k:>4} N={n:>5}  classic {}",
        fmt(classic_m)
    );
    match measure(true) {
        Some(nax_m) => {
            let speedup = match (classic_m.1, nax_m.1) {
                (Some(c), Some(x)) => c / x,
                _ => classic_m.0 / nax_m.0,
            };
            eprintln!("{line}   nax {}   x{speedup:.2}", fmt(nax_m));
        }
        None => eprintln!("{line}   nax (unavailable)"),
    }
}

#[test]
#[ignore = "perf benchmark; run with --ignored"]
fn perf_affine_qmm_t_nax_vs_classic_m512() {
    let device = Device::system_default().expect("Metal device");
    let classic = KernelLibrary::quantized(&device).expect("compile quantized.metal");
    let nax = device
        .nax_available()
        .then(|| KernelLibrary::quantized_nax(&device).expect("compile quantized_nax.metal"));
    eprintln!();
    eprintln!(
        "qmm M=512: classic affine_qmm_t vs affine_qmm_t_nax (gpu = counter-sampled kernel time, 64 dispatches/buffer; wall = isolated dispatch incl. commit+wait, warmup={WARMUP}, timed={TIMED})"
    );
    for &(k, n, label) in &[
        (1024usize, 2048usize, "emb q_proj"),
        (1024, 1024, "emb kv"),
        (2048, 1024, "emb o_proj"),
        (1024, 3072, "emb gate/up"),
        (3072, 1024, "emb down"),
        (896, 896, "q25 q_proj"),
        (896, 4864, "q25 gate/up"),
        (4864, 896, "q25 down"),
    ] {
        bench_m512(&device, &classic, nax.as_ref(), k, n, label);
    }
}
