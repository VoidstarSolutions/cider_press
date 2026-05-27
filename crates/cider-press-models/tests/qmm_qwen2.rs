//! Parity test driving `Tensor::quantized_matmul` at Qwen2.5-0.5B
//! projection shapes against `mx.quantized_matmul`.
//!
//! Covers two regimes:
//! - **Decode** (`M=1`, rank-1 activation `[K]`): exercises the `qmv`
//!   path in the MLX kernel (matvec).
//! - **Prefill** (`M=8`, rank-2 activation `[M, K]`): exercises the
//!   `qmm_t` path.
//!
//! `N=896, K=896` matches Qwen2.5-0.5B's Q-projection weight shape
//! (`[H_q * D_h, hidden_size]` = `[14 * 64, 896]` = `[896, 896]`).
//!
//! Tolerance: 0.05 abs / 0.01 rel (np.allclose-style combined bound).

#![cfg(target_os = "macos")]

use cider_press_runtime::{Device, Quantization, QuantizedWeight, Tensor};
use cider_press_test_utils::{dump_mlx_op, read_bf16, read_u32, tempdir};
use half::bf16;
use safetensors::SafeTensors;

// Qwen2.5-0.5B q_proj weight shape: [H_q * D_h, hidden_size] = [896, 896].
const N: usize = 896;
const K: usize = 896;

// Tolerance matches the existing kernel- and runtime-layer qmm parity bars:
// 1-2 bf16 ULP drift across Apple Silicon generations, plus the inherent
// bf16 accumulation rounding in the kernel. The epsilon floor (5e-2) covers
// near-zero elements where relative error is unstable.
const ATOL: f32 = 5e-2;
const RTOL: f32 = 1e-2;

fn u32_to_bytes(vals: &[u32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn bf16_to_bytes(vals: &[bf16]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn assert_close(label: &str, out: &[bf16], reference: &[bf16]) {
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
        let diff = (af - bf).abs();
        let bound = ATOL + RTOL * bf.abs();
        let excess = diff - bound;
        if excess > worst_excess {
            worst_excess = excess;
            worst_at = i;
            worst_pair = (af, bf);
        }
    }
    assert!(
        worst_excess <= 0.0,
        "{label}: tolerance exceeded — worst element [{worst_at}] \
         got={} want={} |diff|={} > atol={ATOL} + rtol={RTOL}*|want|",
        worst_pair.0,
        worst_pair.1,
        (worst_pair.0 - worst_pair.1).abs(),
    );
}

fn run_case(m: usize) {
    let tmp = tempdir("models-qmm-qwen2");
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

    // Build a QuantizedWeight from the fixture triple.
    let w_q_vals = read_u32(&st, "w_q");
    let scales_vals = read_bf16(&st, "scales");
    let biases_vals = read_bf16(&st, "biases");

    let qw = QuantizedWeight::from_bytes(
        &device,
        [N, K],
        Quantization::Q4_GS64,
        &u32_to_bytes(&w_q_vals),
        &bf16_to_bytes(&scales_vals),
        &bf16_to_bytes(&biases_vals),
    )
    .expect("QuantizedWeight::from_bytes");

    // Build activation tensor.
    let x_vals = read_bf16(&st, "x");
    let x = if m == 1 {
        Tensor::from_slice(&device, &x_vals, [K]).expect("x tensor [K]")
    } else {
        Tensor::from_slice(&device, &x_vals, [m, K]).expect("x tensor [M, K]")
    };

    // Run quantized_matmul and eval.
    let y = x.quantized_matmul(&qw, true).expect("quantized_matmul");
    y.eval().expect("eval");
    let got: Vec<bf16> = y.cpu_to_vec().expect("cpu_to_vec");

    let reference = read_bf16(&st, "y");
    let label = format!("qmm M={m} N={N} K={K}");
    assert_close(&label, &got, &reference);
}

#[test]
fn qmm_qwen2_decode_path() {
    run_case(1);
}

#[test]
fn qmm_qwen2_prefill_path() {
    run_case(8);
}
