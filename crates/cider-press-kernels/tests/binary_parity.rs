//! Parity test for `binary.metal` Add dispatch.
//!
//! Unlike the stage-3/4 parity tests, the fixture is generated at test
//! time by shelling out to `uv run scripts/dump_mlx_op.py add ...`
//! against MLX's own reference implementation. Fixture bytes are kept
//! out of the repo so the activation-dump harness stays the single
//! source of truth for the op's reference output across both the
//! kernels-crate parity test and the later models-crate parity test.
//!
//! Two cases:
//!
//! - **vv** — same-shape `[1, S, H]` bf16 add, exercises `vv_Addbfloat16`.
//! - **broadcast** — `[1, S, H] + [H]` bf16 add, exercises
//!   `g3_Addbfloat16` with rhs strides `[0, 0, 1]`. This is the
//!   per-channel-bias shape that rmsnorm and the Q/K/V/O Linear
//!   biases hit.
//!
//! Bit-exact: bf16 add is a single rounded op, so MLX and us must
//! agree on every byte. If this drifts to a tolerance check, something
//! is wrong with the dispatch — investigate before relaxing.

#![cfg(target_os = "macos")]

use cider_press_kernels::{Buffer, Device, KernelLibrary, kernels};
use cider_press_test_utils::{dump_mlx_op, read_bf16, tempdir};
use half::bf16;
use safetensors::SafeTensors;

const S: usize = 8;
const H: usize = 896;

fn dump_mlx_add(out: &std::path::Path, lhs_shape: &str, rhs_shape: &str) {
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

#[test]
fn parity_vv_add_bf16() {
    let tmp = tempdir("kernels-binary-parity");
    let fixture = tmp.join("add_vv_bf16.safetensors");
    dump_mlx_add(&fixture, &format!("1,{S},{H}"), &format!("1,{S},{H}"));

    let bytes = std::fs::read(&fixture).expect("read fixture");
    let st = SafeTensors::deserialize(&bytes).expect("parse safetensors");
    let lhs = read_bf16(&st, "lhs");
    let rhs = read_bf16(&st, "rhs");
    let out_ref = read_bf16(&st, "out");
    let n = S * H;
    assert_eq!(lhs.len(), n);
    assert_eq!(rhs.len(), n);
    assert_eq!(out_ref.len(), n);

    let device = Device::system_default().expect("Metal device");
    let library = KernelLibrary::binary(&device).expect("compile binary.metal");

    let a: Buffer<bf16> = device.upload(&lhs).expect("upload lhs");
    let b: Buffer<bf16> = device.upload(&rhs).expect("upload rhs");
    let mut c: Buffer<bf16> = device.alloc_buffer(n).expect("alloc out");

    let mut cmds = device.commands().expect("commands");
    kernels::binary::vv_add_bf16(&mut cmds, &library, &a, &b, &mut c).expect("dispatch");
    cmds.commit_and_wait().expect("commit");

    // SAFETY: commit_and_wait synchronised; GPU is done with `c`.
    let out: Vec<bf16> = unsafe { c.as_mut_slice() }.to_vec();
    assert_eq!(out, out_ref, "vv_Addbfloat16 must be bit-exact vs MLX");
}

#[test]
fn parity_g3_add_bf16_broadcast() {
    let tmp = tempdir("kernels-binary-parity");
    let fixture = tmp.join("add_bcast_bf16.safetensors");
    dump_mlx_add(&fixture, &format!("1,{S},{H}"), &format!("{H}"));

    let bytes = std::fs::read(&fixture).expect("read fixture");
    let st = SafeTensors::deserialize(&bytes).expect("parse safetensors");
    let lhs = read_bf16(&st, "lhs"); // [1, S, H]
    let rhs = read_bf16(&st, "rhs"); // [H]
    let out_ref = read_bf16(&st, "out"); // [1, S, H]
    let n = S * H;
    assert_eq!(lhs.len(), n);
    assert_eq!(rhs.len(), H);
    assert_eq!(out_ref.len(), n);

    let device = Device::system_default().expect("Metal device");
    let library = KernelLibrary::binary(&device).expect("compile binary.metal");

    let a: Buffer<bf16> = device.upload(&lhs).expect("upload lhs");
    let b: Buffer<bf16> = device.upload(&rhs).expect("upload rhs");
    let mut c: Buffer<bf16> = device.alloc_buffer(n).expect("alloc out");

    // Padded to rank 3, both operands. lhs is contiguous over the
    // logical [1, S, H] shape; rhs is broadcast across the leading two
    // axes so its strides on those axes are zero.
    #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
    let a_strides: [i64; 3] = [(S * H) as i64, H as i64, 1];
    let b_strides: [i64; 3] = [0, 0, 1];
    #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
    let shape: [i32; 3] = [1, S as i32, H as i32];

    let mut cmds = device.commands().expect("commands");
    kernels::binary::g3_add_bf16(
        &mut cmds, &library, &a, &b, &mut c, a_strides, b_strides, shape,
    )
    .expect("dispatch");
    cmds.commit_and_wait().expect("commit");

    // SAFETY: commit_and_wait synchronised; GPU is done with `c`.
    let out: Vec<bf16> = unsafe { c.as_mut_slice() }.to_vec();
    assert_eq!(
        out, out_ref,
        "g3_Addbfloat16 broadcast must be bit-exact vs MLX"
    );
}
