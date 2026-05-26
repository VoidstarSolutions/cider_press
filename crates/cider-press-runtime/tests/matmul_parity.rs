//! Runtime-level parity test for `Tensor::matmul` against
//! `mx.matmul`.
//!
//! Kernels-crate parity already pins the bf16-ULP tolerance; this
//! test confirms the runtime correctly wires `OpKind::MatMul` —
//! shape inference for the leading dims, allocation, and the
//! kernels-layer dispatch with the right `(M, N, K, batch)`.

#![cfg(target_os = "macos")]
#![allow(clippy::cast_precision_loss)]

use approx::assert_relative_eq;
use cider_press_runtime::{Device, Tensor};
use cider_press_test_utils::{dump_mlx_op, read_bf16, tempdir};
use half::bf16;
use safetensors::SafeTensors;

#[test]
fn matmul_through_runtime_decode_shape() {
    // Decode-step Q @ K^T: Q [1, 14, 1, 64] @ K^T [1, 14, 64, 4]
    parity(&[1, 14, 1, 64], &[1, 14, 64, 4]);
}

#[test]
fn matmul_through_runtime_attn_at_v_shape() {
    // Decode-step attn @ V: attn [1, 14, 1, 4] @ V [1, 14, 4, 64]
    parity(&[1, 14, 1, 4], &[1, 14, 4, 64]);
}

#[test]
fn matmul_through_runtime_prefill_shape() {
    // Prefill QK shape: [1, 14, 8, 64] @ [1, 14, 64, 8]
    parity(&[1, 14, 8, 64], &[1, 14, 64, 8]);
}

fn parity(lhs_shape: &[usize], rhs_shape: &[usize]) {
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
    assert_eq!(got.len(), out_ref.len(), "matmul: length mismatch");
    for (a, b) in got.iter().zip(out_ref.iter()) {
        assert_relative_eq!(a.to_f32(), b.to_f32(), max_relative = 1e-2, epsilon = 5e-2);
    }
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
