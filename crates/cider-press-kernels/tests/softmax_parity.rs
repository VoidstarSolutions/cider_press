//! Parity test for `softmax.metal` block dispatch.
//!
//! Drives both `block_softmax_bfloat16` (default accumulator) and
//! `block_softmax_precise_bfloat16` (float accumulator) against
//! `mx.softmax(x, axis=-1, precise=...)`. We assert a tight ULP
//! tolerance rather than bit-exact: softmax does the running
//! `max` / `sum(exp)` reduction via `simd_max` / `simd_sum` which
//! depend on lane-order summation, and `metal::fast::exp` carries a
//! relative-precision bound — same hardware-drift story as the
//! sigmoid case in `unary_parity`. The precise variant is meaningfully
//! tighter than the default since its float accumulator absorbs the
//! lane-summation drift before the final bf16 cast.
//!
//! Drives a Qwen2-attention-shape input `[1, 14, T, T]` for two
//! values of T (4 and 256) to exercise both small (single-simd
//! reduction) and medium (multi-simd in-block reduction) shapes
//! within the `block_softmax` regime.

#![cfg(target_os = "macos")]
#![allow(clippy::cast_precision_loss)]

use cider_press_kernels::{Buffer, Device, KernelLibrary, kernels};
use cider_press_test_utils::{dump_mlx_op, read_bf16, tempdir};
use half::bf16;
use safetensors::SafeTensors;

const H: usize = 14;
const B: usize = 1;

#[test]
fn parity_block_softmax_bf16_small() {
    parity(4, false, 0.01, 0.02);
}

#[test]
fn parity_block_softmax_bf16_medium() {
    parity(256, false, 0.01, 0.04);
}

#[test]
fn parity_block_softmax_precise_bf16_small() {
    parity(4, true, 0.005, 0.01);
}

#[test]
fn parity_block_softmax_precise_bf16_medium() {
    parity(256, true, 0.005, 0.01);
}

fn parity(seq_len: usize, precise: bool, abs_tol: f32, rel_tol: f32) {
    let (input, out_ref) = fixture(seq_len, precise);
    let device = Device::system_default().expect("Metal device");
    let library = KernelLibrary::softmax(&device).expect("compile softmax.metal");

    let elem = B * H * seq_len * seq_len;
    let src: Buffer<bf16> = device.upload(&input).expect("upload");
    let mut dst: Buffer<bf16> = device.alloc_buffer(elem).expect("alloc out");
    let n_rows = B * H * seq_len;

    let mut cmds = device.commands().expect("commands");
    if precise {
        kernels::softmax::dispatch_block_softmax_precise_bf16(
            &mut cmds, &library, &src, &mut dst, seq_len, n_rows,
        )
        .expect("dispatch precise");
    } else {
        kernels::softmax::dispatch_block_softmax_bf16(
            &mut cmds, &library, &src, &mut dst, seq_len, n_rows,
        )
        .expect("dispatch default");
    }
    cmds.commit_and_wait().expect("commit");

    // SAFETY: commit_and_wait synchronised; GPU is done with `dst`.
    let out: Vec<bf16> = unsafe { dst.as_mut_slice() }.to_vec();
    assert_within_tolerance(
        if precise {
            "block_softmax_precise_bfloat16"
        } else {
            "block_softmax_bfloat16"
        },
        &out,
        &out_ref,
        abs_tol,
        rel_tol,
    );
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

fn fixture(seq_len: usize, precise: bool) -> (Vec<bf16>, Vec<bf16>) {
    let tmp = tempdir("softmax-parity");
    let path = tmp.join(format!(
        "softmax-{}-{seq_len}.safetensors",
        if precise { "precise" } else { "default" }
    ));
    let shape = format!("{B},{H},{seq_len},{seq_len}");
    let mut args: Vec<&str> = vec!["softmax", "--lhs-shape", &shape, "--dtype", "bf16"];
    if precise {
        args.push("--precise");
    }
    dump_mlx_op(&path, &args);
    let bytes = std::fs::read(&path).expect("read fixture");
    let st = SafeTensors::deserialize(&bytes).expect("parse safetensors");
    (read_bf16(&st, "lhs"), read_bf16(&st, "out"))
}
