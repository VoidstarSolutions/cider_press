//! Runtime-level parity tests for `Tensor::square` / `Tensor::rsqrt` /
//! `Tensor::sigmoid` / `Tensor::erf` / `Tensor::sum` / `Tensor::mean`
//! against MLX.
//!
//! The kernels-crate tests already validate each underlying dispatch
//! bit-exactly; these tests validate that the runtime threads inputs +
//! shape metadata through the lazy graph correctly for each op family
//! and that `Tensor::mean` (composed as `sum * (1/N)`) produces the
//! expected bf16-rounded result.

#![cfg(target_os = "macos")]
#![allow(clippy::cast_precision_loss)]

use cider_press_runtime::{Device, Tensor};
use cider_press_test_utils::{dump_mlx_op, read_bf16, tempdir};
use half::bf16;
use safetensors::SafeTensors;

const S: usize = 8;
const H: usize = 896;

#[test]
fn square_through_runtime_matches_mlx() {
    let (lhs, out_ref) = fixture("square");
    let device = Device::system_default().expect("device");
    let t = Tensor::from_slice(&device, &lhs, [1, S, H]).expect("lhs");
    let sq = t.square().expect("schedule square");
    sq.eval().expect("eval");
    let out: Vec<bf16> = sq.cpu_to_vec().expect("dense out");
    assert_eq!(
        out, out_ref,
        "Tensor::square must match mx.square bit-exactly"
    );
}

#[test]
fn rsqrt_through_runtime_matches_mlx() {
    let (lhs, out_ref) = fixture("rsqrt");
    let device = Device::system_default().expect("device");
    let t = Tensor::from_slice(&device, &lhs, [1, S, H]).expect("lhs");
    let inv = t.rsqrt().expect("schedule rsqrt");
    inv.eval().expect("eval");
    let out: Vec<bf16> = inv.cpu_to_vec().expect("dense out");
    assert_eq!(
        out, out_ref,
        "Tensor::rsqrt must match mx.rsqrt bit-exactly"
    );
}

#[test]
fn sigmoid_through_runtime_matches_mlx_within_ulp_tolerance() {
    let (lhs, out_ref) = fixture("sigmoid");
    let device = Device::system_default().expect("device");
    let t = Tensor::from_slice(&device, &lhs, [1, S, H]).expect("lhs");
    let sig = t.sigmoid().expect("schedule sigmoid");
    sig.eval().expect("eval");
    let out: Vec<bf16> = sig.cpu_to_vec().expect("dense out");
    // Sigmoid composes `metal::exp(metal::abs(x))`, which drifts 1–2
    // bf16 ULPs across Apple Silicon generations — see the kernel-layer
    // parity test for the analysis. The runtime layer adds no new math
    // on top, so the same tolerance applies here.
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for (a, b) in out.iter().zip(out_ref.iter()) {
        let af = a.to_f32();
        let bf = b.to_f32();
        let abs = (af - bf).abs();
        let rel = if bf.abs() > 1e-6 { abs / bf.abs() } else { 0.0 };
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(rel);
    }
    assert!(
        max_abs <= 0.005,
        "Tensor::sigmoid max_abs {max_abs} exceeded bf16-ULP tolerance 0.005"
    );
    assert!(
        max_rel <= 0.01,
        "Tensor::sigmoid max_rel {max_rel} exceeded bf16-ULP tolerance 0.01"
    );
}

#[test]
fn erf_through_runtime_matches_mlx() {
    let (lhs, out_ref) = fixture("erf");
    let device = Device::system_default().expect("device");
    let t = Tensor::from_slice(&device, &lhs, [1, S, H]).expect("lhs");
    let e = t.erf().expect("schedule erf");
    e.eval().expect("eval");
    let out: Vec<bf16> = e.cpu_to_vec().expect("dense out");
    assert_eq!(out, out_ref, "Tensor::erf must match mx.erf bit-exactly");
}

#[test]
fn sum_axis_minus_one_matches_mlx() {
    let (lhs, out_ref) = fixture("row_sum");
    let device = Device::system_default().expect("device");
    let t = Tensor::from_slice(&device, &lhs, [1, S, H]).expect("lhs");
    let summed = t.sum(-1, false).expect("schedule sum");
    summed.eval().expect("eval");
    // Output shape is [1, 8] (last axis dropped).
    assert_eq!(summed.shape().dims(), &[1, S]);
    let out: Vec<bf16> = summed.cpu_to_vec().expect("dense out");
    assert_eq!(
        out, out_ref,
        "Tensor::sum(-1) must match mx.sum bit-exactly"
    );
}

#[test]
fn sum_keep_dim_preserves_rank() {
    let (lhs, _) = fixture("row_sum");
    let device = Device::system_default().expect("device");
    let t = Tensor::from_slice(&device, &lhs, [1, S, H]).expect("lhs");
    let summed = t.sum(-1, true).expect("schedule sum keep_dim");
    summed.eval().expect("eval");
    assert_eq!(summed.shape().dims(), &[1, S, 1]);
    // Bit-exact equivalence with keep_dim=false is the same bytes,
    // since the storage layout doesn't change.
    let (_, out_ref) = fixture("row_sum");
    let out: Vec<bf16> = summed.cpu_to_vec().expect("dense out");
    assert_eq!(out, out_ref);
}

#[test]
fn mean_composes_to_sum_times_recip() {
    // We don't have an MLX `mean` parity fixture (mean() at the
    // runtime layer composes sum + mul_by_1/N), so the reference is
    // CPU-computed bf16(sum * (1/H)) per row.
    let (lhs, sum_ref) = fixture("row_sum");
    let device = Device::system_default().expect("device");
    let t = Tensor::from_slice(&device, &lhs, [1, S, H]).expect("lhs");
    let avg = t.mean(-1, true).expect("schedule mean");
    avg.eval().expect("eval");
    assert_eq!(avg.shape().dims(), &[1, S, 1]);

    let inv_h = bf16::from_f32(1.0_f32 / H as f32);
    let expected: Vec<bf16> = sum_ref
        .iter()
        .map(|s| bf16::from_f32(s.to_f32() * inv_h.to_f32()))
        .collect();
    let got: Vec<bf16> = avg.cpu_to_vec().expect("dense out");
    assert_eq!(
        got, expected,
        "Tensor::mean must equal bf16(sum * (1/N)) per row"
    );
}

// ---------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------

fn fixture(op: &str) -> (Vec<bf16>, Vec<bf16>) {
    let tmp = tempdir("runtime-unary-reduce");
    let path = tmp.join(format!("{op}.safetensors"));
    let shape = format!("1,{S},{H}");
    dump_mlx_op(&path, &[op, "--lhs-shape", &shape, "--dtype", "bf16"]);
    let bytes = std::fs::read(&path).expect("read fixture");
    let st = SafeTensors::deserialize(&bytes).expect("parse safetensors");
    (read_bf16(&st, "lhs"), read_bf16(&st, "out"))
}
