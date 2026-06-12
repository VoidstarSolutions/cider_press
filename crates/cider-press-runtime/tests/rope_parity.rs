//! Runtime-level parity test for `Tensor::rope` against MLX's
//! `mx.fast.rope`.
//!
//! The kernels-crate test already validates `dispatch_rope_bf16`;
//! this test validates that the runtime threads inputs + stride
//! metadata correctly through `OpKind::Rope`, including the
//! `base.log2()` host-side preprocessing. Tolerance mirrors the
//! kernel test: bit-exact at offset=0 (identity rotation), tight
//! bf16-ULP tolerance at non-zero offsets because `metal::cos` /
//! `metal::sin` drift 1–2 bf16 ULPs across Apple Silicon generations.

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
fn rope_through_runtime_matches_mlx_decode_offset() {
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
    if start_pos == 0 {
        assert_eq!(
            got, out_ref,
            "Tensor::rope (offset=0) must match mx.fast.rope bit-exactly",
        );
    } else {
        assert_within_tolerance(
            &format!("Tensor::rope (offset={start_pos})"),
            &got,
            &out_ref,
            0.005,
            0.01,
        );
    }
}

#[test]
fn rope_strided_view_matches_contiguous() {
    // [1, T, H, D] dense source; permute to [1, H, T, D] (strided, feature
    // axis stays contiguous) vs the same permuted-then-copied (contiguous).
    let (batch, seq, heads, dim) = (1usize, 6usize, 4usize, 64usize);
    let device = Device::system_default().expect("device");

    let len = batch * seq * heads * dim;
    let host: Vec<bf16> = (0..len)
        .map(|i| {
            // Deterministic spread in [-0.5, 0.5) without an int->float cast.
            let f = f32::from(u16::try_from(i % 1000).expect("0..1000")) * 0.001 - 0.5;
            bf16::from_f32(f)
        })
        .collect();
    let base = Tensor::from_slice(&device, &host, [batch, seq, heads, dim]).expect("base");
    let offset = Tensor::from_slice(&device, &[0i32], [1]).expect("offset");

    let strided = base.permute(&[0, 2, 1, 3]).expect("permute"); // [1,H,T,D] strided
    let contiguous = strided.copy().expect("copy");

    let out_strided = strided.rope(&offset, BASE, 1.0, dim).expect("rope strided");
    let out_contig = contiguous
        .rope(&offset, BASE, 1.0, dim)
        .expect("rope contig");
    out_strided.eval().expect("eval s");
    out_contig.eval().expect("eval c");

    let got_strided: Vec<bf16> = out_strided.cpu_to_vec().expect("dense strided out");
    let got_contig: Vec<bf16> = out_contig.cpu_to_vec().expect("dense contig out");
    assert_eq!(
        got_strided, got_contig,
        "rope on a strided permuted view must match rope on its contiguous copy",
    );
}

#[test]
fn rope_rejects_non_rank4_without_panicking() {
    // A rank-0 (scalar) input must return InvalidArgument, not panic on the
    // last-axis stride probe (which runs only after the rank check).
    let device = Device::system_default().expect("device");
    let scalar = Tensor::from_slice(&device, &[bf16::from_f32(1.0)], []).expect("scalar");
    let offset = Tensor::from_slice(&device, &[0i32], [1]).expect("offset");
    let err = scalar
        .rope(&offset, BASE, 1.0, D)
        .expect_err("rank-0 rope must be rejected");
    assert!(
        err.to_string().contains("rank 4"),
        "expected a rank-4 error, got: {err}"
    );
}

fn assert_within_tolerance(
    label: &str,
    got: &[bf16],
    expected: &[bf16],
    abs_tol: f32,
    rel_tol: f32,
) {
    assert_eq!(got.len(), expected.len(), "{label}: length mismatch");
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
        "{label}: tolerance exceeded (max_abs={max_abs} > {abs_tol} or max_rel={max_rel} > {rel_tol})"
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
