//! Parity tests for the branch-12a [`Module`] wrappers — [`Linear`]
//! and [`RmsNormLayer`] — against MLX.
//!
//! Both wrappers are thin: `Linear::forward` is `quantized_matmul (+
//! bias)` and `RmsNormLayer::forward` is the composed `nn::rms_norm`.
//! So we validate them against the *same* MLX fixtures their
//! underlying ops use (`qmm`, `rms_norm` from `dump_mlx_op.py`),
//! confirming the wrapper adds no behavioral drift:
//!
//! - `Linear` (no bias) reproduces `mx.quantized_matmul` at Qwen2.5's
//!   `[896, 896]` projection shape, decode (`M=1`) and prefill (`M=8`).
//! - `Linear` (with bias) adds a synthetic dense bias on top; the
//!   reference is the MLX qmm output plus the same bias broadcast in
//!   bf16 (what `mlx.nn.QuantizedLinear` computes).
//! - `RmsNormLayer` reproduces `mx.fast.rms_norm` at `[1, 8, 896]`.

#![cfg(target_os = "macos")]
#![allow(clippy::cast_precision_loss)]

use cider_press_models::nn::{Linear, Module, RmsNormLayer};
use cider_press_runtime::{Device, Quantization, QuantizedWeight, Tensor};
use cider_press_test_utils::{dump_mlx_op, read_bf16, read_u32, tempdir};
use half::bf16;
use safetensors::SafeTensors;

// Qwen2.5-0.5B q_proj weight shape: [H_q * D_h, hidden_size] = [896, 896].
const N: usize = 896;
const K: usize = 896;

// Matches the qmm parity bars (1-2 bf16 ULP cross-generation drift plus
// the kernel's bf16 accumulation rounding; epsilon floor covers near-zero
// elements where relative error is unstable).
const QMM_ATOL: f32 = 5e-2;
const QMM_RTOL: f32 = 1e-2;

// bf16-compose tolerance for the rms_norm path (same bar as rms_norm.rs).
const RMS_TOL: f32 = 0.02;

fn u32_to_bytes(vals: &[u32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn bf16_to_bytes(vals: &[bf16]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn assert_close(label: &str, out: &[bf16], reference: &[bf16], atol: f32, rtol: f32) {
    assert_eq!(
        out.len(),
        reference.len(),
        "{label}: output length {} != reference length {}",
        out.len(),
        reference.len()
    );
    let mut worst_excess = 0.0f32;
    let mut worst_at = 0usize;
    let mut worst_pair = (0.0f32, 0.0f32);
    for (i, (a, b)) in out.iter().zip(reference.iter()).enumerate() {
        if a == b {
            continue;
        }
        let af = a.to_f32();
        let bf = b.to_f32();
        assert!(
            !af.is_nan() && !bf.is_nan(),
            "{label}: NaN at element [{i}] got={af} want={bf}"
        );
        let diff = (af - bf).abs();
        let bound = atol + rtol * bf.abs();
        let excess = diff - bound;
        if excess > worst_excess {
            worst_excess = excess;
            worst_at = i;
            worst_pair = (af, bf);
        }
    }
    assert!(
        worst_excess <= 0.0,
        "{label}: tolerance exceeded — worst element [{worst_at}] got={} want={} |diff|={} \
         > atol={atol} + rtol={rtol}*|want|",
        worst_pair.0,
        worst_pair.1,
        (worst_pair.0 - worst_pair.1).abs(),
    );
}

/// Load the `qmm` MLX fixture for a given `M`, returning the device,
/// the quantized weight, the activation tensor, and the MLX reference
/// output `y` (bias-free `mx.quantized_matmul`).
fn load_qmm_fixture(m: usize) -> (Device, QuantizedWeight, Tensor, Vec<bf16>) {
    let tmp = tempdir("models-linear-qmm");
    let path = tmp.join(format!("qmm-m{m}-n{N}-k{K}.safetensors"));
    let m_s = m.to_string();
    let n_s = N.to_string();
    let k_s = K.to_string();
    dump_mlx_op(
        &path,
        &[
            "qmm",
            "--m",
            &m_s,
            "--n",
            &n_s,
            "--k",
            &k_s,
            "--group-size",
            "64",
            "--bits",
            "4",
            "--dtype",
            "bf16",
        ],
    );

    let bytes = std::fs::read(&path).expect("read fixture");
    let st = SafeTensors::deserialize(&bytes).expect("parse safetensors");

    let device = Device::system_default().expect("device");

    let qw = QuantizedWeight::from_bytes(
        &device,
        [N, K],
        Quantization::Q4_GS64,
        &u32_to_bytes(&read_u32(&st, "w_q")),
        &bf16_to_bytes(&read_bf16(&st, "scales")),
        &bf16_to_bytes(&read_bf16(&st, "biases")),
    )
    .expect("QuantizedWeight::from_bytes");

    let x_vals = read_bf16(&st, "x");
    let x = if m == 1 {
        Tensor::from_slice(&device, &x_vals, [K]).expect("x tensor [K]")
    } else {
        Tensor::from_slice(&device, &x_vals, [m, K]).expect("x tensor [M, K]")
    };

    let y = read_bf16(&st, "y");
    (device, qw, x, y)
}

fn run_linear_no_bias(m: usize) {
    let (_device, qw, x, reference) = load_qmm_fixture(m);

    let linear = Linear::new(qw, None).expect("Linear::new");
    let y = linear.forward(&x).expect("Linear::forward");
    y.eval().expect("eval");
    let got: Vec<bf16> = y.cpu_to_vec().expect("cpu_to_vec");

    let label = format!("Linear (no bias) M={m} N={N} K={K}");
    assert_close(&label, &got, &reference, QMM_ATOL, QMM_RTOL);
}

#[test]
fn linear_no_bias_decode_path() {
    run_linear_no_bias(1);
}

#[test]
fn linear_no_bias_prefill_path() {
    run_linear_no_bias(8);
}

#[test]
fn linear_with_bias_matches_qmm_plus_bias() {
    let m = 8;
    let (device, qw, x, qmm_ref) = load_qmm_fixture(m);

    // Synthetic dense bias [N], same shape Qwen2's q_proj.bias carries.
    let bias_host: Vec<bf16> = (0..N)
        .map(|i| bf16::from_f32(((i as f32) * 0.01) - 0.3))
        .collect();
    let bias = Tensor::from_slice(&device, &bias_host, [N]).expect("bias tensor");

    let linear = Linear::new(qw, Some(bias)).expect("Linear::new with bias");
    let y = linear.forward(&x).expect("Linear::forward");
    y.eval().expect("eval");
    let got: Vec<bf16> = y.cpu_to_vec().expect("cpu_to_vec");

    // Reference: MLX qmm output + bias broadcast over rows, in bf16 —
    // exactly what mlx.nn.QuantizedLinear(bias=True) produces.
    let mut expected = vec![bf16::ZERO; qmm_ref.len()];
    for row in 0..m {
        for col in 0..N {
            let v = qmm_ref[row * N + col].to_f32() + bias_host[col].to_f32();
            expected[row * N + col] = bf16::from_f32(v);
        }
    }

    assert_close(
        &format!("Linear (with bias) M={m} N={N} K={K}"),
        &got,
        &expected,
        QMM_ATOL,
        QMM_RTOL,
    );
}

#[test]
fn linear_rejects_mismatched_bias() {
    let (device, qw, _x, _y) = load_qmm_fixture(1);
    // Bias of the wrong length (N-1) must be rejected at construction.
    let bad_bias = Tensor::from_slice(&device, &vec![bf16::ZERO; N - 1], [N - 1]).expect("bias");
    assert!(Linear::new(qw, Some(bad_bias)).is_err());
}

#[test]
fn rms_norm_layer_matches_mlx_fast_rms_norm() {
    const S: usize = 8;
    const H: usize = 896;
    const EPS: f32 = 1e-6;

    let tmp = tempdir("models-rmsnorm-layer");
    let fixture = tmp.join("rms_norm.safetensors");
    let shape = format!("1,{S},{H}");
    let eps_str = format!("{EPS}");
    dump_mlx_op(
        &fixture,
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

    let bytes = std::fs::read(&fixture).expect("read fixture");
    let st = SafeTensors::deserialize(&bytes).expect("parse safetensors");
    let x_host = read_bf16(&st, "x");
    let gamma_host = read_bf16(&st, "gamma");
    let out_ref = read_bf16(&st, "out");

    let device = Device::system_default().expect("device");
    let x = Tensor::from_slice(&device, &x_host, [1, S, H]).expect("x");
    let gamma = Tensor::from_slice(&device, &gamma_host, [H]).expect("gamma");

    let norm = RmsNormLayer::new(gamma, EPS);
    let y = norm.forward(&x).expect("RmsNormLayer::forward");
    y.eval().expect("eval");
    let got: Vec<bf16> = y.cpu_to_vec().expect("dense out");

    assert_close("RmsNormLayer", &got, &out_ref, RMS_TOL, RMS_TOL);
}
