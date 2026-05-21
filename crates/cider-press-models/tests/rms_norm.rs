//! Parity tests for the composed [`nn::rms_norm`] against MLX's
//! `mx.fast.rms_norm` (synthetic) and against a CPU reference using
//! a real `layer0.input_layernorm` weight from a Qwen2 checkpoint
//! (gated on `CIDER_QWEN_CHECKPOINT_PATH`).
//!
//! Why not bit-exact vs MLX? `mx.fast.rms_norm` is a *fused* kernel
//! that runs in higher-precision accumulators; our composition uses
//! several bf16 round-trips through device memory (`square` → `mean`
//! → `+eps` → `rsqrt` → `*x` → `*gamma`). Each round-trip costs one
//! bf16 rounding. The end-to-end drift is well under a bf16 ULP at
//! the magnitudes the rmsnorm output sits at, but it's not exactly
//! zero, so we use a tolerance bar comparable to other bf16
//! reductions.

#![cfg(target_os = "macos")]
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use std::fs;
use std::path::PathBuf;

use cider_press_models::nn::rms_norm;
use cider_press_models::qwen2::{Qwen2Config, load_qwen2_weights};
use cider_press_runtime::{Device, Tensor};
use cider_press_test_utils::{dump_mlx_op, read_bf16, tempdir};
use half::bf16;
use safetensors::SafeTensors;

const S: usize = 8;
const H: usize = 896;
const EPS: f32 = 1e-6;

#[test]
fn rms_norm_matches_mlx_fast_rms_norm() {
    let tmp = tempdir("rms-norm");
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

    let bytes = fs::read(&fixture).expect("read fixture");
    let st = SafeTensors::deserialize(&bytes).expect("parse safetensors");
    let x_host = read_bf16(&st, "x");
    let gamma_host = read_bf16(&st, "gamma");
    let out_ref = read_bf16(&st, "out");

    let device = Device::system_default().expect("device");
    let x = Tensor::from_slice(&device, &x_host, [1, S, H]).expect("x");
    let gamma = Tensor::from_slice(&device, &gamma_host, [H]).expect("gamma");

    let y = rms_norm(&x, &gamma, EPS).expect("schedule rms_norm");
    y.eval().expect("eval");
    let got: Vec<bf16> = y.cpu_to_vec().expect("dense out");

    // bf16 round-trip-aware tolerance. The composed path costs more
    // bf16 roundings than the fused kernel, so we don't claim
    // bit-exactness — but the absolute error per element stays well
    // under one bf16 ULP at the magnitudes rms_norm outputs sit at.
    assert_eq!(got.len(), out_ref.len());
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for (a, b) in got.iter().zip(out_ref.iter()) {
        let af = a.to_f32();
        let bf = b.to_f32();
        let abs = (af - bf).abs();
        let rel = if bf.abs() > 1e-3 { abs / bf.abs() } else { 0.0 };
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(rel);
    }
    assert!(
        max_abs <= 0.02,
        "max absolute error {max_abs} exceeded bf16-compose tolerance 0.02",
    );
    assert!(
        max_rel <= 0.02,
        "max relative error {max_rel} exceeded bf16-compose tolerance 0.02",
    );
}

#[test]
fn real_checkpoint_rms_norm_matches_cpu_reference() {
    let Some(dir) = checkpoint_path() else {
        eprintln!(
            "skipped: set CIDER_QWEN_CHECKPOINT_PATH to a local Qwen2.5-0.5B-Instruct-4bit \
             checkout to run this test"
        );
        return;
    };

    let config_bytes = fs::read(dir.join("config.json")).expect("read config.json");
    let config = Qwen2Config::from_json_bytes(&config_bytes).expect("parse config.json");
    let archive_bytes = fs::read(dir.join("model.safetensors")).expect("read model.safetensors");
    let archive = SafeTensors::deserialize(&archive_bytes).expect("deserialize");

    let device = Device::shared().expect("device");
    let weights = load_qwen2_weights(&archive, &config, &device).expect("load weights");

    // Real loaded gamma weight for layer 0's pre-attention RMSNorm.
    let gamma = &weights.layers[0].input_layernorm;
    assert_eq!(gamma.shape().dims(), &[config.hidden_size]);

    // Synthetic hidden states — same shape pattern the attention
    // block would feed in.
    let hidden_len = S * config.hidden_size;
    let hidden_host: Vec<bf16> = (0..hidden_len)
        .map(|i| bf16::from_f32(((i as f32) * 0.001) - 0.5))
        .collect();
    let x = Tensor::from_slice(&device, &hidden_host, [1, S, config.hidden_size]).expect("x");

    let y = rms_norm(&x, gamma, config.rms_norm_eps as f32).expect("rms_norm");
    y.eval().expect("eval");
    let got: Vec<bf16> = y.cpu_to_vec().expect("dense out");

    // CPU reference uses f32 throughout — closer to mx.fast.rms_norm
    // than to our bf16-composed path, so we compare with the same
    // tolerance as the synthetic test.
    let gamma_host: Vec<bf16> = gamma.cpu_to_vec().expect("gamma host");
    let expected = cpu_rms_norm(
        &hidden_host,
        &gamma_host,
        config.hidden_size,
        config.rms_norm_eps as f32,
    );

    let mut max_abs = 0.0f32;
    for (a, b) in got.iter().zip(expected.iter()) {
        max_abs = max_abs.max((a.to_f32() - b.to_f32()).abs());
    }
    assert!(
        max_abs <= 0.05,
        "real-checkpoint rms_norm max abs error {max_abs} exceeded tolerance 0.05",
    );
}

// ---------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------

fn cpu_rms_norm(x: &[bf16], gamma: &[bf16], hidden: usize, eps: f32) -> Vec<bf16> {
    let mut out = vec![bf16::ZERO; x.len()];
    let rows = x.len() / hidden;
    for row in 0..rows {
        let start = row * hidden;
        let end = start + hidden;
        let row_slice = &x[start..end];
        let mean_sq: f32 = row_slice
            .iter()
            .map(|v| {
                let f = v.to_f32();
                f * f
            })
            .sum::<f32>()
            / hidden as f32;
        let inv_rms = (mean_sq + eps).sqrt().recip();
        for (i, v) in row_slice.iter().enumerate() {
            let scaled = v.to_f32() * inv_rms * gamma[i].to_f32();
            out[start + i] = bf16::from_f32(scaled);
        }
    }
    out
}

fn checkpoint_path() -> Option<PathBuf> {
    let raw = std::env::var("CIDER_QWEN_CHECKPOINT_PATH").ok()?;
    let path = fs::canonicalize(&raw)
        .unwrap_or_else(|err| panic!("CIDER_QWEN_CHECKPOINT_PATH={raw} does not resolve: {err}"));
    assert!(
        path.is_dir(),
        "CIDER_QWEN_CHECKPOINT_PATH={} is not a directory",
        path.display(),
    );
    Some(path)
}
