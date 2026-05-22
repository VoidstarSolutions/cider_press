//! Runtime-level parity test for `Tensor::softmax` against
//! `mx.softmax(x, axis=-1, precise=...)`.
//!
//! Kernels-crate parity already pins the dispatch within the bf16
//! ULP-tolerance window. This test confirms the runtime wires the
//! op through correctly — flattening a multi-dim input into
//! `n_rows × axis_size` and routing on the `precise` flag.

#![cfg(target_os = "macos")]
#![allow(clippy::cast_precision_loss)]

use cider_press_runtime::{Device, Tensor};
use cider_press_test_utils::{dump_mlx_op, read_bf16, tempdir};
use half::bf16;
use safetensors::SafeTensors;

const H: usize = 14;
const B: usize = 1;

#[test]
fn softmax_through_runtime_matches_mlx_default() {
    parity(4, false, 0.01, 0.02);
}

#[test]
fn softmax_through_runtime_matches_mlx_precise() {
    parity(4, true, 0.005, 0.01);
}

#[test]
fn softmax_through_runtime_matches_mlx_medium_precise() {
    parity(256, true, 0.005, 0.01);
}

fn parity(seq_len: usize, precise: bool, abs_tol: f32, rel_tol: f32) {
    let (input_host, out_ref) = fixture(seq_len, precise);
    let device = Device::system_default().expect("device");
    let input = Tensor::from_slice(&device, &input_host, [B, H, seq_len, seq_len]).expect("input");

    let out = input.softmax(precise).expect("schedule softmax");
    out.eval().expect("eval");
    assert_eq!(out.shape().dims(), &[B, H, seq_len, seq_len]);
    let got: Vec<bf16> = out.cpu_to_vec().expect("dense out");

    let label = if precise {
        "Tensor::softmax(precise=true)"
    } else {
        "Tensor::softmax(precise=false)"
    };
    assert_within_tolerance(label, &got, &out_ref, abs_tol, rel_tol);
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
    for (a, b) in got.iter().zip(expected.iter()) {
        if a == b {
            continue;
        }
        let af = a.to_f32();
        let bf = b.to_f32();
        let abs = (af - bf).abs();
        let rel = if bf.abs() > 1e-6 { abs / bf.abs() } else { 0.0 };
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(rel);
    }
    assert!(
        max_abs <= abs_tol && max_rel <= rel_tol,
        "{label}: tolerance exceeded (max_abs={max_abs} > {abs_tol} \
         or max_rel={max_rel} > {rel_tol})"
    );
}

fn fixture(seq_len: usize, precise: bool) -> (Vec<bf16>, Vec<bf16>) {
    let tmp = tempdir("runtime-softmax-parity");
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
