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

use std::path::{Path, PathBuf};
use std::process::Command;

use cider_press_runtime::{Device, Tensor};
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
fn sigmoid_through_runtime_matches_mlx() {
    let (lhs, out_ref) = fixture("sigmoid");
    let device = Device::system_default().expect("device");
    let t = Tensor::from_slice(&device, &lhs, [1, S, H]).expect("lhs");
    let sig = t.sigmoid().expect("schedule sigmoid");
    sig.eval().expect("eval");
    let out: Vec<bf16> = sig.cpu_to_vec().expect("dense out");
    assert_eq!(
        out, out_ref,
        "Tensor::sigmoid must match mx.sigmoid bit-exactly"
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
    let tmp = tempdir();
    let path = tmp.join(format!("{op}.safetensors"));
    dump_mlx(op, &path, &format!("1,{S},{H}"));
    let bytes = std::fs::read(&path).expect("read fixture");
    let st = SafeTensors::deserialize(&bytes).expect("parse safetensors");
    (read_bf16(&st, "lhs"), read_bf16(&st, "out"))
}

fn dump_mlx(op: &str, out: &Path, lhs_shape: &str) {
    let script = workspace_root().join("scripts").join("dump_mlx_op.py");
    let status = Command::new("uv")
        .arg("run")
        .arg(&script)
        .arg("--output")
        .arg(out)
        .arg("--seed")
        .arg("0")
        .arg(op)
        .arg("--lhs-shape")
        .arg(lhs_shape)
        .arg("--dtype")
        .arg("bf16")
        .status();
    let status = match status {
        Ok(s) => s,
        Err(err) => panic!(
            "failed to invoke `uv run {}`: {err}. This test requires `uv` on PATH; \
             CI installs it via astral-sh/setup-uv.",
            script.display()
        ),
    };
    assert!(status.success(), "dump_mlx_op.py {op} exited {status}");
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("workspace root")
        .to_path_buf()
}

fn read_bf16(st: &SafeTensors, name: &str) -> Vec<bf16> {
    let view = st
        .tensor(name)
        .unwrap_or_else(|e| panic!("tensor {name}: {e}"));
    assert_eq!(view.dtype(), safetensors::Dtype::BF16);
    let bytes = view.data();
    assert!(bytes.len() % 2 == 0);
    bytes
        .chunks_exact(2)
        .map(|c| bf16::from_le_bytes([c[0], c[1]]))
        .collect()
}

fn tempdir() -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("SystemTime")
        .as_nanos();
    let pid = std::process::id();
    let dir = std::env::temp_dir().join(format!("cider-press-runtime-unary-reduce-{pid}-{nanos}"));
    std::fs::create_dir_all(&dir).expect("mktemp");
    dir
}
