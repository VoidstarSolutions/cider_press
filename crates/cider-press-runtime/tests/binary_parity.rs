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
//! rationale for not checking in fixture bytes.

#![cfg(target_os = "macos")]

use std::path::Path;

use cider_press_runtime::{Device, Tensor};
use cider_press_test_utils::{dump_mlx_op, read_bf16, tempdir};
use half::bf16;
use safetensors::SafeTensors;

const S: usize = 8;
const H: usize = 896;

#[test]
fn add_vv_through_runtime_matches_mlx() {
    let tmp = tempdir("runtime-binary-parity");
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
    let tmp = tempdir("runtime-binary-parity");
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
    let tmp = tempdir("runtime-binary-parity");
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
    let tmp = tempdir("runtime-binary-parity");
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
    dump_mlx_op(
        out,
        &[
            "add",
            "--lhs-shape",
            lhs_shape,
            "--rhs-shape",
            rhs_shape,
            "--dtype",
            "bf16",
        ],
    );
}

fn dump_mlx_mul(out: &Path, lhs_shape: &str, rhs_shape: &str) {
    dump_mlx_op(
        out,
        &[
            "mul",
            "--lhs-shape",
            lhs_shape,
            "--rhs-shape",
            rhs_shape,
            "--dtype",
            "bf16",
        ],
    );
}
