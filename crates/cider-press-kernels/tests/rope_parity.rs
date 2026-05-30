//! Parity test for `rope.metal` `rope_bfloat16` dispatch.
//!
//! Generates a fixture at test time via `uv run scripts/dump_mlx_op.py
//! rope` and compares against MLX for the Qwen2-shape input. The
//! offset=0 case is bit-exact (cos=1 / sin=0; the rotation is the
//! identity, so the only operations are the float→bf16 casts that
//! square / rsqrt / erf also exercise bit-exactly).
//!
//! The non-zero-offset case uses a tight bf16-ULP tolerance: the kernel
//! invokes `metal::cos` / `metal::sin`, which drift 1–2 bf16 ULPs
//! across Apple Silicon generations (M-series local vs macos-15 CI
//! runners), even though the kernel source matches MLX byte-for-byte.
//! Same situation as the sigmoid kernel — see `unary_parity.rs`.
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
    if start_pos == 0 {
        assert_eq!(
            out, out_ref,
            "rope_bfloat16 (offset=0) must be bit-exact vs MLX (identity rotation)"
        );
    } else {
        assert_within_tolerance(
            &format!("rope_bfloat16 (offset={start_pos})"),
            &out,
            &out_ref,
            0.005,
            0.01,
        );
    }
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
    let mut mismatches = 0usize;
    let mut samples: Vec<(usize, f32, f32, f32)> = Vec::new();
    for (i, (a, b)) in got.iter().zip(expected.iter()).enumerate() {
        if a == b {
            continue;
        }
        mismatches += 1;
        let af = a.to_f32();
        let bf = b.to_f32();
        let abs = (af - bf).abs();
        let rel = if bf.abs() > 1e-6 { abs / bf.abs() } else { 0.0 };
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(rel);
        if samples.len() < 4 {
            samples.push((i, af, bf, abs));
        }
    }
    if max_abs > abs_tol || max_rel > rel_tol {
        eprintln!(
            "{label}: {mismatches}/{} mismatches, max_abs={max_abs:.6}, max_rel={max_rel:.6}",
            got.len(),
        );
        for (i, g, e, abs) in samples {
            eprintln!("  [{i}] got={g} expected={e} |diff|={abs}");
        }
        panic!(
            "{label}: tolerance exceeded (max_abs={max_abs} > {abs_tol} \
             or max_rel={max_rel} > {rel_tol})"
        );
    }
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
