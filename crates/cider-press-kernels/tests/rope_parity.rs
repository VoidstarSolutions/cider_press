//! Parity test for `rope.metal` `rope_bfloat16` dispatch (branch 9).
//!
//! Generates a fixture at test time via `uv run scripts/dump_mlx_op.py
//! rope` and asserts bit-exact equality with MLX for the Qwen2-shape
//! input. The kernel does the rotation in float internally before
//! casting back to bf16, so a bit-exact bar is reasonable — same as
//! square / rsqrt / erf, which also go through `static_cast<float>` →
//! op → `static_cast<T>`.
//!
//! The single specialization exercised is the Qwen2-inference one:
//! `forward=true, traditional=false, hs_transpose=false`,
//! `with_freqs=false`, `large=false`. See
//! `crates/cider-press-kernels/src/kernels/rope.rs` for the deferred-
//! variant rationale.

#![cfg(target_os = "macos")]

use cider_press_kernels::{Buffer, Device, KernelLibrary, kernels};
use cider_press_test_utils::{dump_mlx_op, read_bf16, tempdir};
use half::bf16;
use safetensors::SafeTensors;

const B: i32 = 1;
const H: i32 = 14;
const T: i32 = 4;
const D: i32 = 64;
const BASE: f32 = 1_000_000.0;

#[test]
fn parity_rope_bf16_qwen2_q_shape_offset_zero() {
    run_parity(0);
}

#[test]
fn parity_rope_bf16_qwen2_q_shape_decode_offset() {
    // Mid-decode position (offset=37) exercises the `L = scale * (T_pos
    // + batch_offset)` arithmetic that a zero offset masks.
    run_parity(37);
}

fn run_parity(start_pos: i32) {
    let (lhs, out_ref) = fixture(start_pos);
    let device = Device::system_default().expect("Metal device");
    let library = KernelLibrary::rope(&device).expect("compile rope.metal");

    let elem = (B * H * T * D) as usize;
    let src: Buffer<bf16> = device.upload(&lhs).expect("upload lhs");
    let mut dst: Buffer<bf16> = device.alloc_buffer(elem).expect("alloc out");
    let offset: Buffer<i32> = device.upload(&[start_pos]).expect("upload offset");

    let args = kernels::rope::RopeArgs {
        batch: B,
        n_head: H,
        seq_len: T,
        head_dim: D,
        rotary_dims: D,
        scale: 1.0,
        base_log2: BASE.ln() / std::f32::consts::LN_2,
        strides: [i64::from(T * D), i64::from(D), 1],
        out_strides: [i64::from(T * D), i64::from(D), 1],
        offset_stride: 0,
    };

    let mut cmds = device.commands().expect("commands");
    kernels::rope::dispatch_rope_bf16(&mut cmds, &library, &src, &mut dst, &offset, args)
        .expect("dispatch rope");
    cmds.commit_and_wait().expect("commit");

    // SAFETY: commit_and_wait synchronised; GPU is done with `dst`.
    let out: Vec<bf16> = unsafe { dst.as_mut_slice() }.to_vec();
    assert_eq!(
        out, out_ref,
        "rope_bfloat16 (offset={start_pos}) must be bit-exact vs MLX"
    );
}

fn fixture(start_pos: i32) -> (Vec<bf16>, Vec<bf16>) {
    let tmp = tempdir("rope-parity");
    let path = tmp.join(format!("rope-{start_pos}.safetensors"));
    let shape = format!("{B},{H},{T},{D}");
    let dims_str = D.to_string();
    let base_str = format!("{BASE}");
    let offset_str = start_pos.to_string();
    dump_mlx_op(
        &path,
        &[
            "rope",
            "--lhs-shape",
            &shape,
            "--dims",
            &dims_str,
            "--base",
            &base_str,
            "--offset",
            &offset_str,
            "--dtype",
            "bf16",
        ],
    );
    let bytes = std::fs::read(&path).expect("read fixture");
    let st = SafeTensors::deserialize(&bytes).expect("parse safetensors");
    (read_bf16(&st, "lhs"), read_bf16(&st, "out"))
}
