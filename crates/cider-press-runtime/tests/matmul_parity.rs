//! Runtime-level parity test for `Tensor::matmul` against
//! `mx.matmul`.
//!
//! Kernels-crate parity already pins the bf16-ULP tolerance; this
//! test confirms the runtime correctly wires `OpKind::MatMul` —
//! shape inference for the leading dims, allocation, and the
//! kernels-layer dispatch with the right `(M, N, K, batch)`.

#![cfg(target_os = "macos")]
#![allow(clippy::cast_precision_loss)]

use cider_press_runtime::{Device, Tensor};
use cider_press_test_utils::{dump_mlx_op, read_bf16, tempdir};
use half::bf16;
use safetensors::SafeTensors;

#[test]
fn matmul_through_runtime_decode_shape() {
    // Decode-step Q @ K^T: Q [1, 14, 1, 64] @ K^T [1, 14, 64, 4]
    parity(&[1, 14, 1, 64], &[1, 14, 64, 4], 0.02, 0.02);
}

#[test]
fn matmul_through_runtime_attn_at_v_shape() {
    // Decode-step attn @ V: attn [1, 14, 1, 4] @ V [1, 14, 4, 64]
    parity(&[1, 14, 1, 4], &[1, 14, 4, 64], 0.02, 0.02);
}

#[test]
fn matmul_through_runtime_prefill_shape() {
    // Prefill QK shape: [1, 14, 8, 64] @ [1, 14, 64, 8]
    parity(&[1, 14, 8, 64], &[1, 14, 64, 8], 0.05, 0.05);
}

fn parity(lhs_shape: &[usize], rhs_shape: &[usize], abs_tol: f32, rel_tol: f32) {
    let (lhs_host, rhs_host, out_ref) = fixture(lhs_shape, rhs_shape);
    let device = Device::system_default().expect("device");
    let lhs = Tensor::from_slice(&device, &lhs_host, lhs_shape.to_vec()).expect("lhs");
    let rhs = Tensor::from_slice(&device, &rhs_host, rhs_shape.to_vec()).expect("rhs");

    let out = lhs.matmul(&rhs).expect("schedule matmul");
    out.eval().expect("eval");

    let mut expected_shape: Vec<usize> = lhs_shape[..lhs_shape.len() - 1].to_vec();
    expected_shape.push(rhs_shape[rhs_shape.len() - 1]);
    assert_eq!(out.shape().dims(), expected_shape.as_slice());

    let got: Vec<bf16> = out.cpu_to_vec().expect("dense out");
    assert_within_tolerance(&got, &out_ref, abs_tol, rel_tol);
}

fn assert_within_tolerance(got: &[bf16], expected: &[bf16], abs_tol: f32, rel_tol: f32) {
    assert_eq!(got.len(), expected.len(), "matmul: length mismatch");
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
        "Tensor::matmul: tolerance exceeded (max_abs={max_abs} > {abs_tol} \
         or max_rel={max_rel} > {rel_tol})"
    );
}

fn fixture(lhs_shape: &[usize], rhs_shape: &[usize]) -> (Vec<bf16>, Vec<bf16>, Vec<bf16>) {
    let tmp = tempdir("runtime-matmul-parity");
    let lhs_csv = shape_csv(lhs_shape);
    let rhs_csv = shape_csv(rhs_shape);
    let path = tmp.join(format!("matmul-{lhs_csv}__{rhs_csv}.safetensors"));
    dump_mlx_op(
        &path,
        &[
            "matmul",
            "--lhs-shape",
            &lhs_csv,
            "--rhs-shape",
            &rhs_csv,
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

fn shape_csv(s: &[usize]) -> String {
    s.iter().map(usize::to_string).collect::<Vec<_>>().join(",")
}
