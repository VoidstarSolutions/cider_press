//! Parity test for the naive bf16 batched matmul kernel
//! (`kernels/matmul.metal`, branch 11).
//!
//! Drives `gemm_bfloat16` against `mx.matmul`. The kernel uses a
//! float accumulator (same as MLX's GEMMs) and processes each
//! `[Batch, M, K] × [Batch, K, N] → [Batch, M, N]` slice
//! independently, so the only source of drift is `fma`-vs-`mul+add`
//! ordering inside the per-row dot product. At Qwen2.5-0.5B SDPA
//! head sizes the bf16-cast at the final write dominates, and we
//! land at a tight bf16-ULP tolerance — looser than bit-exact, same
//! ballpark as branch-10 softmax (`precise`).
//!
//! Two shape regimes:
//! - Small SDPA decode-ish shape `[14, 1, 64] × [14, 64, 4]` —
//!   exercises the "single-row Q against 4-token K" pattern.
//! - Medium SDPA prefill-ish shape `[14, 32, 64] × [14, 64, 32]` —
//!   exercises Q@K^T at modest prefill width.

#![cfg(target_os = "macos")]
#![allow(clippy::cast_precision_loss)]

use cider_press_kernels::{Buffer, Device, KernelLibrary, kernels};
use cider_press_test_utils::{dump_mlx_op, read_bf16, tempdir};
use half::bf16;
use safetensors::SafeTensors;

#[test]
fn parity_gemm_bf16_small() {
    parity(14, 1, 64, 4, 0.02, 0.02);
}

#[test]
fn parity_gemm_bf16_medium() {
    parity(14, 32, 64, 32, 0.05, 0.05);
}

#[test]
fn parity_gemm_bf16_attn_shape() {
    // attn @ V: scores [14, T, T] @ V [14, T, D_h]
    parity(14, 8, 8, 64, 0.05, 0.05);
}

fn parity(batch: usize, m: usize, k: usize, n: usize, abs_tol: f32, rel_tol: f32) {
    let (lhs, rhs, out_ref) = fixture(batch, m, k, n);
    let device = Device::system_default().expect("Metal device");
    let library = KernelLibrary::matmul(&device).expect("compile matmul.metal");

    let src_a: Buffer<bf16> = device.upload(&lhs).expect("upload A");
    let src_b: Buffer<bf16> = device.upload(&rhs).expect("upload B");
    let mut dst: Buffer<bf16> = device.alloc_buffer(batch * m * n).expect("alloc C");

    let mut cmds = device.commands().expect("commands");
    kernels::matmul::dispatch_gemm_bf16(
        &mut cmds,
        &library,
        &src_a,
        &src_b,
        &mut dst,
        m,
        n,
        k,
        batch,
        m * k,
        k * n,
        m * n,
    )
    .expect("dispatch");
    cmds.commit_and_wait().expect("commit");

    // SAFETY: commit_and_wait synchronised; GPU is done with `dst`.
    let out: Vec<bf16> = unsafe { dst.as_mut_slice() }.to_vec();
    assert_within_tolerance(&out, &out_ref, abs_tol, rel_tol);
}

fn assert_within_tolerance(got: &[bf16], expected: &[bf16], abs_tol: f32, rel_tol: f32) {
    assert_eq!(got.len(), expected.len(), "matmul: length mismatch");
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut mismatches = 0usize;
    let mut samples: Vec<(usize, f32, f32, f32)> = Vec::new();
    for (i, (a, b)) in got.iter().zip(expected.iter()).enumerate() {
        if a == b {
            continue;
        }
        mismatches += 1;
        let af = a.to_f32();
        let bf = b.to_f32();
        let abs = (af - bf).abs();
        let rel = if bf.abs() > 1e-6 { abs / bf.abs() } else { 0.0 };
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(rel);
        if samples.len() < 4 {
            samples.push((i, af, bf, abs));
        }
    }
    if max_abs > abs_tol || max_rel > rel_tol {
        eprintln!(
            "matmul: {mismatches}/{} mismatches, max_abs={max_abs:.6}, max_rel={max_rel:.6}",
            got.len(),
        );
        for (i, g, e, abs) in samples {
            eprintln!("  [{i}] got={g} expected={e} |diff|={abs}");
        }
        panic!(
            "matmul: tolerance exceeded (max_abs={max_abs} > {abs_tol} \
             or max_rel={max_rel} > {rel_tol})"
        );
    }
}

fn fixture(batch: usize, m: usize, k: usize, n: usize) -> (Vec<bf16>, Vec<bf16>, Vec<bf16>) {
    let tmp = tempdir("matmul-parity");
    let path = tmp.join(format!("matmul-{batch}x{m}x{k}x{n}.safetensors"));
    let lhs_shape = format!("{batch},{m},{k}");
    let rhs_shape = format!("{batch},{k},{n}");
    dump_mlx_op(
        &path,
        &[
            "matmul",
            "--lhs-shape",
            &lhs_shape,
            "--rhs-shape",
            &rhs_shape,
            "--dtype",
            "bf16",
        ],
    );
    let bytes = std::fs::read(&path).expect("read fixture");
    let st = SafeTensors::deserialize(&bytes).expect("parse safetensors");
    (
        read_bf16(&st, "lhs"),
        read_bf16(&st, "rhs"),
        read_bf16(&st, "out"),
    )
}
