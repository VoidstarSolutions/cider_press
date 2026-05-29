//! Parity test for `reduce.metal` last-axis Sum dispatch.
//!
//! Generates fixtures via `uv run scripts/dump_mlx_op.py row_sum`
//! and asserts bit-exact equality vs MLX for bf16 at `[1, 8, 896]`
//! reducing axis -1.
//!
//! Bit-exactness rationale: the kernels-crate `row_reduce_sum_bf16`
//! dispatches the same MLX `row_reduce_looped_1_reduce_sumbfloat16`
//! kernel that `mx.sum` does. The accumulation order is deterministic
//! within the kernel (simdgroup tree-reduce → threadgroup tree-reduce
//! → write), so identical kernel + identical inputs ⇒ identical bytes.
//!
//! If this test starts failing, suspect either dispatch transcription
//! drift (our bind order or grid math) or an MLX-side change to the
//! reduce kernel itself.

#![cfg(target_os = "macos")]

use cider_press_kernels::{Buffer, Device, KernelLibrary, kernels};
use cider_press_test_utils::{dump_mlx_op, read_bf16, tempdir};
use half::bf16;
use safetensors::SafeTensors;

const S: usize = 8;
const H: usize = 896;

#[test]
fn parity_row_reduce_sum_bf16() {
    let tmp = tempdir("reduce-parity");
    let fixture = tmp.join("row_sum.safetensors");
    let shape = format!("1,{S},{H}");
    dump_mlx_op(
        &fixture,
        &["row_sum", "--lhs-shape", &shape, "--dtype", "bf16"],
    );

    let bytes = std::fs::read(&fixture).expect("read fixture");
    let st = SafeTensors::deserialize(&bytes).expect("parse safetensors");
    let lhs = read_bf16(&st, "lhs");
    let out_ref = read_bf16(&st, "out");
    assert_eq!(lhs.len(), S * H);
    assert_eq!(out_ref.len(), S);

    let device = Device::system_default().expect("Metal device");
    let library = KernelLibrary::reduce(&device).expect("compile reduce.metal");

    let src: Buffer<bf16> = device.upload(&lhs).expect("upload lhs");
    let mut dst: Buffer<bf16> = device.alloc_buffer(S).expect("alloc out");

    let mut cmds = device.commands().expect("commands");
    kernels::reduce::row_reduce_sum_bf16(&mut cmds, &library, &src, &[1, S, H], &mut dst)
        .expect("dispatch");
    cmds.commit_and_wait().expect("commit");

    // SAFETY: commit_and_wait synchronised.
    let out: Vec<bf16> = unsafe { dst.as_mut_slice() }.to_vec();
    assert_eq!(
        out, out_ref,
        "row_reduce_sum_bf16 must be bit-exact vs mx.sum"
    );
}
