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

fn parity(shape: &[usize]) {
    let hidden = *shape.last().unwrap();
    let rows: usize = shape[..shape.len() - 1].iter().product();
    let (x, gamma, out_ref) = fixture(rows, hidden);

    let device = Device::shared().expect("device");
    let xt = Tensor::from_slice(&device, &x, shape).expect("x tensor");
    let gt = Tensor::from_slice(&device, &gamma, [hidden]).expect("gamma tensor");

    let out = xt.rms_norm(&gt, EPS).expect("rms_norm op");
    out.eval().expect("eval");
    let got = out.cpu_to_vec::<bf16>().expect("read out");

    assert_within_tolerance(&format!("rms_norm{shape:?}"), &got, &out_ref, 0.005, 0.01);
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
