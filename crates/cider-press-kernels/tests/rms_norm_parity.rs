//! Parity test for the fused `rms_norm.metal` dispatch.
//!
//! Drives `rms_norm_bf16` (MLX's `rms_single_row`) against
//! `mx.fast.rms_norm(x, gamma, eps)` at Qwen2.5-0.5B norm shapes
//! (hidden=896) for one row (decode, T=1) and several rows (prefill).
//! The kernel reduces in fp32 and finishes with `metal::precise::rsqrt`,
//! and it is MLX's own kernel run on the same inputs — so we hold a tight
//! bf16-ULP tolerance (the rsqrt + lane-ordered `simd_sum` can drift a
//! ULP or two across Apple-Silicon generations, per `CLAUDE.md`).

#![cfg(target_os = "macos")]
#![allow(clippy::cast_precision_loss)]

use cider_press_kernels::{Buffer, Device, KernelLibrary, kernels};
use cider_press_test_utils::{dump_mlx_op, read_bf16, tempdir};
use half::bf16;
use safetensors::SafeTensors;

const EPS: f32 = 1e-6;

#[test]
fn parity_rms_norm_decode_row() {
    parity(1, 896);
}

#[test]
fn parity_rms_norm_prefill_rows() {
    parity(8, 896);
}

#[test]
fn parity_rms_norm_small() {
    parity(2, 64);
}

fn parity(n_rows: usize, hidden: usize) {
    let (x, gamma, out_ref) = fixture(n_rows, hidden);
    let device = Device::system_default().expect("Metal device");
    let library = KernelLibrary::rms_norm(&device).expect("compile rms_norm.metal");

    let src: Buffer<bf16> = device.upload(&x).expect("upload x");
    let w: Buffer<bf16> = device.upload(&gamma).expect("upload gamma");
    let mut dst: Buffer<bf16> = device.alloc_buffer(n_rows * hidden).expect("alloc out");

    let mut cmds = device.commands().expect("commands");
    kernels::rms_norm::rms_norm_bf16(&mut cmds, &library, &src, &w, &mut dst, hidden, n_rows, EPS)
        .expect("dispatch rms_norm");
    cmds.commit_and_wait().expect("commit");

    // SAFETY: commit_and_wait synchronised; GPU is done with `dst`.
    let out: Vec<bf16> = unsafe { dst.as_mut_slice() }.to_vec();
    assert_within_tolerance(
        &format!("rms_norm[{n_rows}x{hidden}]"),
        &out,
        &out_ref,
        0.005,
        0.01,
    );
}

fn fixture(n_rows: usize, hidden: usize) -> (Vec<bf16>, Vec<bf16>, Vec<bf16>) {
    let tmp = tempdir("rms-norm-parity");
    let path = tmp.join(format!("rms_norm-{n_rows}x{hidden}.safetensors"));
    let shape = format!("{n_rows},{hidden}");
    let eps_str = format!("{EPS}");
    dump_mlx_op(
        &path,
        &[
            "rms_norm",
            "--x-shape",
            &shape,
            "--eps",
            &eps_str,
            "--dtype",
            "bf16",
        ],
    );
    let bytes = std::fs::read(&path).expect("read fixture");
    let st = SafeTensors::deserialize(&bytes).expect("parse safetensors");
    (
        read_bf16(&st, "x"),
        read_bf16(&st, "gamma"),
        read_bf16(&st, "out"),
    )
}

fn assert_within_tolerance(
    label: &str,
    got: &[bf16],
    expected: &[bf16],
    abs_tol: f32,
    rel_tol: f32,
) {
    assert_eq!(got.len(), expected.len(), "{label}: length mismatch");
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
            "{label}: {mismatches}/{} mismatches, max_abs={max_abs:.6}, max_rel={max_rel:.6}",
            got.len(),
        );
        for (i, g, e, abs) in samples {
            eprintln!("  [{i}] got={g} expected={e} |diff|={abs}");
        }
        panic!(
            "{label}: tolerance exceeded (max_abs={max_abs} > {abs_tol} \
             or max_rel={max_rel} > {rel_tol})"
        );
    }
}
