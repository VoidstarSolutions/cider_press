//! Models-side parity for `Tensor::add` / `Tensor::mul` at Qwen2 shapes.
//!
//! Two layers of coverage:
//!
//! 1. **Synthetic** (always runs): shapes derived from
//!    `Qwen2.5-0.5B-Instruct`'s config (`hidden_size = 896`). Generates
//!    fixtures at test time via `uv run scripts/dump_mlx_op.py`,
//!    drives the dispatch through `cider_press_runtime::Tensor`, and
//!    asserts bit-exact equality with MLX for both same-shape and
//!    broadcast cases of `add` and `mul`. This is the smoke that
//!    catches surface-area drift between branches.
//! 2. **Gated real-checkpoint** (requires `CIDER_QWEN_CHECKPOINT_PATH`):
//!    loads weights via [`load_qwen2_weights`] and adds a real
//!    `input_layernorm` weight (shape `[hidden_size]`, bf16) to a
//!    synthetic hidden-states tensor `[1, S, hidden_size]`. Validates
//!    bit-exactly against a CPU reference (`bf16(h_f32 + g_f32)`),
//!    since bf16 add is one rounded op. This is the integration
//!    canary for the load → eval → host-read end-to-end path.
//!
//! Like the kernels- and runtime-crate parity tests, the synthetic
//! layer shells out to `uv`; CI installs it via astral-sh/setup-uv.

#![cfg(target_os = "macos")]
#![allow(clippy::cast_precision_loss)]

use std::fs;
use std::path::PathBuf;

use cider_press_models::qwen2::{Qwen2Config, load_qwen2_weights};
use cider_press_runtime::{Device, Tensor};
use cider_press_test_utils::{dump_mlx_op, read_bf16, tempdir};
use half::bf16;
use safetensors::SafeTensors;

/// Sequence length used for the residual-shaped test. Small enough to
/// stay cheap, large enough to populate the rank-3 axis.
const S: usize = 8;

/// Qwen2.5-0.5B-Instruct hidden size. Matches `config.hidden_size`
/// for the pinned revision; the synthetic tests don't need to read
/// the config file (the dimension is stable across Qwen2 0.5B
/// variants).
const QWEN2_05B_HIDDEN_SIZE: usize = 896;

#[test]
fn add_at_qwen2_hidden_size_matches_mlx() {
    parity_case(
        "add",
        &format!("1,{S},{QWEN2_05B_HIDDEN_SIZE}"),
        &format!("1,{S},{QWEN2_05B_HIDDEN_SIZE}"),
        false,
    );
}

#[test]
fn add_with_bias_broadcast_matches_mlx() {
    parity_case(
        "add",
        &format!("1,{S},{QWEN2_05B_HIDDEN_SIZE}"),
        &format!("{QWEN2_05B_HIDDEN_SIZE}"),
        true,
    );
}

#[test]
fn mul_at_qwen2_hidden_size_matches_mlx() {
    parity_case(
        "mul",
        &format!("1,{S},{QWEN2_05B_HIDDEN_SIZE}"),
        &format!("1,{S},{QWEN2_05B_HIDDEN_SIZE}"),
        false,
    );
}

#[test]
fn mul_with_bias_broadcast_matches_mlx() {
    parity_case(
        "mul",
        &format!("1,{S},{QWEN2_05B_HIDDEN_SIZE}"),
        &format!("{QWEN2_05B_HIDDEN_SIZE}"),
        true,
    );
}

/// Drives a synthetic add/mul case at the given shapes via the
/// runtime and compares against MLX. `broadcast` controls whether rhs
/// is the same rank as lhs or a rank-1 vector (which forces the
/// strided dispatch path).
fn parity_case(op: &str, lhs_shape: &str, rhs_shape: &str, broadcast: bool) {
    let tmp = tempdir("models-binary-parity");
    let fixture = tmp.join(format!("{op}_qwen2.safetensors"));
    dump_mlx_op(
        &fixture,
        &[
            op,
            "--lhs-shape",
            lhs_shape,
            "--rhs-shape",
            rhs_shape,
            "--dtype",
            "bf16",
        ],
    );

    let bytes = fs::read(&fixture).expect("read fixture");
    let st = SafeTensors::deserialize(&bytes).expect("parse safetensors");
    let lhs = read_bf16(&st, "lhs");
    let rhs = read_bf16(&st, "rhs");
    let out_ref = read_bf16(&st, "out");

    let device = Device::system_default().expect("Metal device");
    let a = Tensor::from_slice(&device, &lhs, [1, S, QWEN2_05B_HIDDEN_SIZE]).expect("lhs");
    let b = if broadcast {
        Tensor::from_slice(&device, &rhs, [QWEN2_05B_HIDDEN_SIZE]).expect("rhs (bcast)")
    } else {
        Tensor::from_slice(&device, &rhs, [1, S, QWEN2_05B_HIDDEN_SIZE]).expect("rhs")
    };
    let c = match op {
        "add" => a.add(&b),
        "mul" => a.mul(&b),
        other => panic!("unhandled op {other}"),
    }
    .expect("schedule op");
    c.eval().expect("eval");

    let out: Vec<bf16> = c.cpu_to_vec().expect("dense output");
    assert_eq!(
        out, out_ref,
        "{op} ({lhs_shape}, {rhs_shape}, broadcast={broadcast}) must match MLX bit-exactly"
    );
}

#[test]
fn real_checkpoint_layernorm_add_broadcast() {
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
    let archive = SafeTensors::deserialize(&archive_bytes).expect("deserialize safetensors");

    let device = Device::shared().expect("system default device");
    let weights = load_qwen2_weights(&archive, &config, &device).expect("load weights");

    // input_layernorm is bf16, shape [hidden_size] — a natural
    // broadcast-add operand against a [1, S, hidden_size] hidden-states
    // tensor. We don't actually run rmsnorm here; we
    // just use this weight as a known-good loaded bf16 vector.
    let gamma = &weights.layers[0].input_layernorm;
    assert_eq!(gamma.shape().dims(), &[config.hidden_size]);

    let h_len = S * config.hidden_size;
    let hidden_host: Vec<bf16> = (0..h_len)
        .map(|i| bf16::from_f32(((i as f32) * 0.001) - 0.5))
        .collect();
    let hidden =
        Tensor::from_slice(&device, &hidden_host, [1, S, config.hidden_size]).expect("hidden");

    let summed = hidden.add(gamma).expect("schedule add");
    summed.eval().expect("eval");
    let got: Vec<bf16> = summed.cpu_to_vec().expect("dense output");

    let gamma_host: Vec<bf16> = gamma.cpu_to_vec().expect("gamma host");
    assert_eq!(gamma_host.len(), config.hidden_size);

    // bf16 add is a single rounded op; the expected output is exactly
    // bf16(h_f32 + g_f32), with broadcasting across the leading axes
    // matching the strided dispatch's stride-0 expansion.
    let mut expected = Vec::with_capacity(h_len);
    for (i, &h) in hidden_host.iter().enumerate() {
        let g = gamma_host[i % config.hidden_size];
        expected.push(bf16::from_f32(h.to_f32() + g.to_f32()));
    }
    assert_eq!(
        got, expected,
        "add(hidden, input_layernorm) must equal the bf16-rounded CPU sum"
    );
}

// ---------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------

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
