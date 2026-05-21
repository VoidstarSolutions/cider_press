//! Runtime-level parity test for `Tensor::rope` against MLX's
//! `mx.fast.rope`.
//!
//! The kernels-crate test already validates `dispatch_rope_bf16` bit-
//! exactly; this test validates that the runtime threads inputs +
//! stride metadata correctly through `OpKind::Rope`, including the
//! `base.log2()` host-side preprocessing.

#![cfg(target_os = "macos")]

use cider_press_runtime::{DType, Device, Tensor};
use cider_press_test_utils::{dump_mlx_op, read_bf16, tempdir};
use half::bf16;
use safetensors::SafeTensors;

const B: usize = 1;
const H: usize = 14;
const T: usize = 4;
const D: usize = 64;
const BASE: f32 = 1_000_000.0;

#[test]
fn rope_through_runtime_matches_mlx_bit_exact_offset_zero() {
    run(0);
}

#[test]
fn rope_through_runtime_matches_mlx_bit_exact_decode_offset() {
    run(37);
}

fn run(start_pos: i32) {
    let (input_host, out_ref) = fixture(start_pos);
    let device = Device::system_default().expect("device");

    let input = Tensor::from_slice(&device, &input_host, [B, H, T, D]).expect("input");
    let offset = Tensor::from_slice(&device, &[start_pos], [1]).expect("offset");
    assert_eq!(offset.dtype(), DType::I32);

    let out = input.rope(&offset, BASE, 1.0, D).expect("schedule rope");
    out.eval().expect("eval");
    assert_eq!(out.shape().dims(), &[B, H, T, D]);
    let got: Vec<bf16> = out.cpu_to_vec().expect("dense out");
    assert_eq!(
        got, out_ref,
        "Tensor::rope (offset={start_pos}) must match mx.fast.rope bit-exactly",
    );
}

fn fixture(start_pos: i32) -> (Vec<bf16>, Vec<bf16>) {
    let tmp = tempdir("runtime-rope-parity");
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
