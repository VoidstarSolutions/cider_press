//! Runtime-level parity tests for `Tensor::{add, mul}` against MLX's
//! `mx.add` / `mx.multiply`.
//!
//! Drives the dispatch through the runtime API end-to-end:
//! `from_slice` two operands, schedule the op, `eval()`, read host
//! bytes. Two cases per op mirror the kernels-crate test in
//! `crates/cider-press-kernels/tests/binary_parity.rs`:
//!
//! - **vv** — `[1, S, H] op [1, S, H]` bf16. Output should be bit-exact
//!   identical to MLX since bf16 add/mul is a single rounded op and the
//!   runtime threads the same MLX kernel.
//! - **broadcast** — `[1, S, H] op [H]`. The runtime's `Tensor::binary`
//!   inserts a `broadcast_to` view on the rhs which makes the
//!   dispatcher pick the `g3_{Add,Mul}bfloat16` variant.
//!
//! Like the kernels-crate parity test, fixtures are generated at test
//! time via `uv run scripts/dump_mlx_op.py {add,mul}`; CI installs uv
//! via astral-sh/setup-uv. See that file's module docstring for the
//! rationale for not checking in fixture bytes for branch 4.

#![cfg(target_os = "macos")]

use std::path::{Path, PathBuf};
use std::process::Command;

use cider_press_runtime::{Device, Tensor};
use half::bf16;
use safetensors::SafeTensors;

const S: usize = 8;
const H: usize = 896;

#[test]
fn add_vv_through_runtime_matches_mlx() {
    let tmp = tempdir();
    let fixture = tmp.join("add_vv_bf16.safetensors");
    dump_mlx_add(&fixture, &format!("1,{S},{H}"), &format!("1,{S},{H}"));

    let bytes = std::fs::read(&fixture).expect("read fixture");
    let st = SafeTensors::deserialize(&bytes).expect("parse safetensors");
    let lhs = read_bf16(&st, "lhs");
    let rhs = read_bf16(&st, "rhs");
    let out_ref = read_bf16(&st, "out");

    let device = Device::system_default().expect("Metal device");
    let a = Tensor::from_slice(&device, &lhs, [1, S, H]).expect("lhs tensor");
    let b = Tensor::from_slice(&device, &rhs, [1, S, H]).expect("rhs tensor");
    let c = a.add(&b).expect("schedule add");
    c.eval().expect("eval");

    let out: Vec<bf16> = c.cpu_to_vec().expect("dense output");
    assert_eq!(out, out_ref, "Tensor::add vv must match MLX bit-exactly");
}

#[test]
fn add_broadcast_through_runtime_matches_mlx() {
    let tmp = tempdir();
    let fixture = tmp.join("add_bcast_bf16.safetensors");
    dump_mlx_add(&fixture, &format!("1,{S},{H}"), &format!("{H}"));

    let bytes = std::fs::read(&fixture).expect("read fixture");
    let st = SafeTensors::deserialize(&bytes).expect("parse safetensors");
    let lhs = read_bf16(&st, "lhs");
    let rhs = read_bf16(&st, "rhs");
    let out_ref = read_bf16(&st, "out");

    let device = Device::system_default().expect("Metal device");
    let a = Tensor::from_slice(&device, &lhs, [1, S, H]).expect("lhs tensor");
    let b = Tensor::from_slice(&device, &rhs, [H]).expect("rhs tensor (rank 1)");
    let c = a.add(&b).expect("schedule add (broadcast)");
    c.eval().expect("eval");

    let out: Vec<bf16> = c.cpu_to_vec().expect("dense output");
    assert_eq!(
        out, out_ref,
        "Tensor::add broadcast must match MLX bit-exactly"
    );
}

#[test]
fn mul_vv_through_runtime_matches_mlx() {
    let tmp = tempdir();
    let fixture = tmp.join("mul_vv_bf16.safetensors");
    dump_mlx_mul(&fixture, &format!("1,{S},{H}"), &format!("1,{S},{H}"));

    let bytes = std::fs::read(&fixture).expect("read fixture");
    let st = SafeTensors::deserialize(&bytes).expect("parse safetensors");
    let lhs = read_bf16(&st, "lhs");
    let rhs = read_bf16(&st, "rhs");
    let out_ref = read_bf16(&st, "out");

    let device = Device::system_default().expect("Metal device");
    let a = Tensor::from_slice(&device, &lhs, [1, S, H]).expect("lhs tensor");
    let b = Tensor::from_slice(&device, &rhs, [1, S, H]).expect("rhs tensor");
    let c = a.mul(&b).expect("schedule mul");
    c.eval().expect("eval");

    let out: Vec<bf16> = c.cpu_to_vec().expect("dense output");
    assert_eq!(out, out_ref, "Tensor::mul vv must match MLX bit-exactly");
}

#[test]
fn mul_broadcast_through_runtime_matches_mlx() {
    let tmp = tempdir();
    let fixture = tmp.join("mul_bcast_bf16.safetensors");
    dump_mlx_mul(&fixture, &format!("1,{S},{H}"), &format!("{H}"));

    let bytes = std::fs::read(&fixture).expect("read fixture");
    let st = SafeTensors::deserialize(&bytes).expect("parse safetensors");
    let lhs = read_bf16(&st, "lhs");
    let rhs = read_bf16(&st, "rhs");
    let out_ref = read_bf16(&st, "out");

    let device = Device::system_default().expect("Metal device");
    let a = Tensor::from_slice(&device, &lhs, [1, S, H]).expect("lhs tensor");
    let b = Tensor::from_slice(&device, &rhs, [H]).expect("rhs tensor (rank 1)");
    let c = a.mul(&b).expect("schedule mul (broadcast)");
    c.eval().expect("eval");

    let out: Vec<bf16> = c.cpu_to_vec().expect("dense output");
    assert_eq!(
        out, out_ref,
        "Tensor::mul broadcast must match MLX bit-exactly"
    );
}

fn dump_mlx_add(out: &Path, lhs_shape: &str, rhs_shape: &str) {
    dump_mlx_op("add", out, lhs_shape, rhs_shape);
}

fn dump_mlx_mul(out: &Path, lhs_shape: &str, rhs_shape: &str) {
    dump_mlx_op("mul", out, lhs_shape, rhs_shape);
}

fn dump_mlx_op(op: &str, out: &Path, lhs_shape: &str, rhs_shape: &str) {
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
        .arg("--rhs-shape")
        .arg(rhs_shape)
        .arg("--dtype")
        .arg("bf16")
        .status();
    let status = match status {
        Ok(s) => s,
        Err(err) => panic!(
            "failed to invoke `uv run {}`: {err}. \
             This test requires `uv` (https://docs.astral.sh/uv) on PATH \
             so MLX can generate the parity fixture; CI installs it via \
             astral-sh/setup-uv.",
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
    let dir = workspace_root()
        .join("target")
        .join(format!("cider-press-runtime-binary-parity-{pid}-{nanos}"));
    std::fs::create_dir_all(&dir).expect("mktemp");
    dir
}
