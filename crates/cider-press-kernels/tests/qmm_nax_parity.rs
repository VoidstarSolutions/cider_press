//! Parity tests for `kernels::qmm::affine_qmm_t_nax_bf16` — MLX's tensor-core
//! `affine_qmm_t_nax` (bf16 / int4 / gs 64, transpose, unbatched).
//!
//! Fixtures are materialised on demand by `dump_mlx_op qmm`. On a NAX machine
//! MLX's `mx.quantized_matmul` dispatches this same kernel
//! (`quantized.cpp`: `is_nax_available() && transpose && K % 64 == 0`), so the
//! comparison is **bit-exact**. Skips where NAX is unavailable.
//!
//! Shape caveat: MLX routes transposed, unbatched `qmm` through `qmm_splitk`
//! first, which takes a classic split-K kernel (32×32 tiles, fp32 partials —
//! not bit-identical to NAX) whenever `⌈N/32⌉·⌈M/32⌉ < 512` and `K` splits
//! into whole 64-wide partitions. Every shape below therefore has at least
//! 512 classic tiles so MLX's reference really is `qmm_t_nax`; e.g.
//! `(M, N, K) = (150, 896, 896)` would get `split_k = 2` from MLX and differ
//! in ~37 % of elements.

#![cfg(target_os = "macos")]
#![allow(clippy::many_single_char_names)]

use cider_press_kernels::{Buffer, Device, KernelLibrary, kernels};
use cider_press_test_utils::{dump_mlx_op, read_bf16, read_u32, tempdir};
use half::bf16;
use safetensors::SafeTensors;

const GROUP_SIZE: u32 = 64;
const BITS: u32 = 4;

fn run_case(tag: &str, m: usize, n: usize, k: usize) {
    let device = Device::system_default().expect("Metal device");
    if !device.nax_available() {
        eprintln!("skipping {tag}: NAX unavailable on this GPU/OS");
        return;
    }
    let tmp = tempdir(tag);
    let path = tmp.join("qmm_nax.safetensors");
    let (m_s, n_s, k_s) = (m.to_string(), n.to_string(), k.to_string());
    dump_mlx_op(
        &path,
        &[
            "qmm",
            "--m",
            &m_s,
            "--n",
            &n_s,
            "--k",
            &k_s,
            "--group-size",
            "64",
            "--bits",
            "4",
            "--dtype",
            "bf16",
        ],
    );
    let bytes = std::fs::read(&path).expect("read fixture");
    let st = SafeTensors::deserialize(&bytes).expect("parse safetensors");
    let w_q = read_u32(&st, "w_q");
    let scales = read_bf16(&st, "scales");
    let biases = read_bf16(&st, "biases");
    let x = read_bf16(&st, "x");
    let y_ref = read_bf16(&st, "y");
    assert_eq!(y_ref.len(), m * n);

    let library = KernelLibrary::quantized_nax(&device).expect("compile quantized_nax.metal");
    let w_buf: Buffer<u32> = device.upload(&w_q).expect("upload w_q");
    let s_buf: Buffer<bf16> = device.upload(&scales).expect("upload scales");
    let b_buf: Buffer<bf16> = device.upload(&biases).expect("upload biases");
    let x_buf: Buffer<bf16> = device.upload(&x).expect("upload x");
    let mut y_buf: Buffer<bf16> = device.alloc_buffer(m * n).expect("alloc y");

    let mut cmds = device.commands().expect("commands");
    kernels::qmm::affine_qmm_t_nax_bf16(
        &mut cmds, &library, &w_buf, &s_buf, &b_buf, &x_buf, &mut y_buf, m, n, k, GROUP_SIZE, BITS,
    )
    .expect("dispatch");
    cmds.commit_and_wait().expect("commit");
    // SAFETY: commit_and_wait blocked until the GPU finished writing y_buf.
    let y_out: Vec<bf16> = unsafe { y_buf.as_mut_slice() }.to_vec();

    let mismatches = y_out
        .iter()
        .zip(y_ref.iter())
        .filter(|(a, e)| a.to_bits() != e.to_bits())
        .count();
    assert_eq!(
        mismatches,
        0,
        "{tag}: {mismatches}/{} elements differ from MLX",
        y_ref.len()
    );
}

#[test]
fn parity_affine_qmm_t_nax_bf16_m512_k1024_n2048_aligned() {
    run_case("kernels-qmm-nax-aligned", 512, 2048, 1024);
}

#[test]
fn parity_affine_qmm_t_nax_bf16_unaligned_n1000() {
    run_case("kernels-qmm-nax-unaligned", 512, 1000, 1024);
}

#[test]
fn parity_affine_qmm_t_nax_bf16_m150_partial_row_tile() {
    // Qwen2.5-0.5B gate/up at T = 150: 152 × 5 classic tiles, so MLX takes
    // the NAX path; M = 150 leaves a 22-row partial tile at bm = 64.
    run_case("kernels-qmm-nax-m150", 150, 4864, 896);
}

#[test]
fn parity_affine_qmm_t_nax_bf16_m16_bm32_variant() {
    // M ≤ 32 selects the 32-row instantiation (MLX: `bm = 32` when one block
    // covers all of M). 304 classic tiles → MLX's split-K gate is a no-op and
    // its reference is this same kernel; K = 2560 / N = 9728 keep M = 16
    // above MLX's qmv batch limit (13 for these dims).
    run_case("kernels-qmm-nax-m16", 16, 9728, 2560);
}

#[test]
fn nax_rejects_k_not_multiple_of_64() {
    let device = Device::system_default().expect("Metal device");
    if !device.nax_available() {
        eprintln!("skipping: NAX unavailable on this GPU/OS");
        return;
    }
    let library = KernelLibrary::quantized_nax(&device).expect("compile");
    let (m, n, k) = (8usize, 64usize, 96usize);
    let w: Buffer<u32> = device.alloc_buffer(n * k * 4 / 32).expect("w");
    let s: Buffer<bf16> = device.alloc_buffer(n * k / 64).expect("s");
    let b: Buffer<bf16> = device.alloc_buffer(n * k / 64).expect("b");
    let x: Buffer<bf16> = device.alloc_buffer(m * k).expect("x");
    let mut y: Buffer<bf16> = device.alloc_buffer(m * n).expect("y");
    let mut cmds = device.commands().expect("commands");
    let err = kernels::qmm::affine_qmm_t_nax_bf16(
        &mut cmds, &library, &w, &s, &b, &x, &mut y, m, n, k, GROUP_SIZE, BITS,
    )
    .expect_err("K=96 must be rejected");
    assert!(err.to_string().contains("multiple of 64"), "{err}");
}

#[test]
fn nax_rejects_mismatched_x_len() {
    let device = Device::system_default().expect("Metal device");
    if !device.nax_available() {
        eprintln!("skipping: NAX unavailable on this GPU/OS");
        return;
    }
    let library = KernelLibrary::quantized_nax(&device).expect("compile");
    let (m, n, k) = (8usize, 64usize, 128usize);
    let w: Buffer<u32> = device.alloc_buffer(n * k * 4 / 32).expect("w");
    let s: Buffer<bf16> = device.alloc_buffer(n * k / 64).expect("s");
    let b: Buffer<bf16> = device.alloc_buffer(n * k / 64).expect("b");
    // An A1-padded weight pairs a K_physical > K_logical activation: x is too short.
    let x: Buffer<bf16> = device.alloc_buffer(m * (k - 64)).expect("x");
    let mut y: Buffer<bf16> = device.alloc_buffer(m * n).expect("y");
    let mut cmds = device.commands().expect("commands");
    kernels::qmm::affine_qmm_t_nax_bf16(
        &mut cmds, &library, &w, &s, &b, &x, &mut y, m, n, k, GROUP_SIZE, BITS,
    )
    .expect_err("short x must be rejected before dispatch");
}
