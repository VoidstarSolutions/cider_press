//! Parity test for `Tensor::rms_norm` (the runtime op + eval dispatch).
//!
//! The kernels-crate `rms_norm_parity` validates the bare dispatch; this
//! exercises the op-graph path — `OpKind::RmsNorm` construction, the
//! eval-side `n_rows`/`axis_size` derivation, and buffer extraction — against
//! `mx.fast.rms_norm` at a Qwen2.5-0.5B decode shape `[1, T, 896]` and a
//! small shape. Tight bf16-ULP tolerance (MLX's own kernel; rsqrt +
//! lane-ordered reduction can drift a ULP or two).

#![cfg(target_os = "macos")]

use cider_press_runtime::{Device, Tensor};
use cider_press_test_utils::{dump_mlx_op, read_bf16, tempdir};
use half::bf16;
use safetensors::SafeTensors;

const EPS: f32 = 1e-6;

#[test]
fn rms_norm_op_decode_shape() {
    // [1, 4, 896] — leading dims collapse to n_rows=4, axis=896.
    parity(&[1, 4, 896]);
}

#[test]
fn rms_norm_op_small() {
    parity(&[3, 64]);
}

/// Weightless `rms_norm` (gamma = `None`, mlx's `weight=None`) must be
/// **bit-exact** with the weighted path fed an explicit `[axis]` ones-vector:
/// it's the same kernel over the same values — a stride-0 `w[0] = 1.0` read by
/// every lane vs a `[D]` ones gamma `w[i] = 1.0`. This is the optional-gamma
/// surface the Gated-DeltaNet q/k norm (`qk_norm_scale`) consumes.
#[test]
fn rms_norm_weightless_matches_explicit_ones() {
    let device = Device::shared().expect("device");
    // [3,64] (small), [1,4,128] (the GDN linear_key_head_dim), and
    // the single-row/looped boundary (4096 / 4097) plus a wide looped axis.
    for shape in [
        vec![3usize, 64],
        vec![1usize, 4, 128],
        vec![1usize, 1, 4096],
        vec![1usize, 1, 4097],
        vec![1usize, 1, 5120],
    ] {
        let hidden = *shape.last().unwrap();
        let rows: usize = shape[..shape.len() - 1].iter().product();
        // Deterministic, varied input across roughly [-2, 2).
        let x: Vec<bf16> = (0..rows * hidden)
            .map(|i| {
                let m = u8::try_from(i * 37 % 101).expect("< 101 fits u8");
                bf16::from_f32(f32::from(m) / 25.0 - 2.0)
            })
            .collect();
        let xt = Tensor::from_slice(&device, &x, shape.clone()).expect("x tensor");

        let ones = vec![bf16::ONE; hidden];
        let gt = Tensor::from_slice(&device, &ones, [hidden]).expect("ones gamma");
        let weighted = xt.rms_norm(Some(&gt), EPS).expect("weighted rms_norm");
        weighted.eval().expect("eval weighted");
        let weighted = weighted.cpu_to_vec::<bf16>().expect("read weighted");

        let weightless = xt.rms_norm(None, EPS).expect("weightless rms_norm");
        weightless.eval().expect("eval weightless");
        let weightless = weightless.cpu_to_vec::<bf16>().expect("read weightless");

        assert_eq!(
            weighted, weightless,
            "weightless != weighted-with-ones for shape {shape:?}"
        );
    }
}

/// Qwen3.5-4B hidden 2560 — single-row path.
#[test]
fn rms_norm_op_4b_hidden() {
    parity(&[1, 1, 2560]);
}

/// Qwen3.6-27B hidden 5120 — wider than a single threadgroup covers, so this
/// dispatches the looped kernel.
#[test]
fn rms_norm_op_27b_hidden_looped() {
    parity(&[1, 1, 5120]);
}

#[test]
fn rms_norm_op_widest_single_row() {
    parity(&[1, 1, 4096]);
}

#[test]
fn rms_norm_op_narrowest_looped() {
    parity(&[1, 1, 4097]);
}

fn parity(shape: &[usize]) {
    let hidden = *shape.last().unwrap();
    let rows: usize = shape[..shape.len() - 1].iter().product();
    let (x, gamma, out_ref) = fixture(rows, hidden);

    let device = Device::shared().expect("device");
    let xt = Tensor::from_slice(&device, &x, shape).expect("x tensor");
    let gt = Tensor::from_slice(&device, &gamma, [hidden]).expect("gamma tensor");

    let out = xt.rms_norm(Some(&gt), EPS).expect("rms_norm op");
    out.eval().expect("eval");
    let got = out.cpu_to_vec::<bf16>().expect("read out");

    // Wider axes accumulate a longer fp32 reduction; allow a slightly
    // looser bf16-ULP bound for the looped regime (> 4096).
    let (abs_tol, rel_tol) = if hidden > 4096 {
        (0.01, 0.02)
    } else {
        (0.005, 0.01)
    };
    assert_within_tolerance(
        &format!("rms_norm{shape:?}"),
        &got,
        &out_ref,
        abs_tol,
        rel_tol,
    );
}

fn fixture(rows: usize, hidden: usize) -> (Vec<bf16>, Vec<bf16>, Vec<bf16>) {
    let tmp = tempdir("rms-norm-op");
    let path = tmp.join(format!("rms_norm-{rows}x{hidden}.safetensors"));
    let shape = format!("{rows},{hidden}");
    let eps_str = format!("{EPS}");
    dump_mlx_op(
        &path,
        &[
            "rms_norm",
            "--x-shape",
            &shape,
            "--eps",
            &eps_str,
            "--dtype",
            "bf16",
        ],
    );
    let bytes = std::fs::read(&path).expect("read fixture");
    let st = SafeTensors::deserialize(&bytes).expect("parse safetensors");
    (
        read_bf16(&st, "x"),
        read_bf16(&st, "gamma"),
        read_bf16(&st, "out"),
    )
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
        let (af, bf) = (a.to_f32(), b.to_f32());
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
