//! Layer-0 `SwiGLU` MLP real-checkpoint parity test.
//!
//! Loads Qwen2.5-0.5B layer 0 (gated on `CIDER_QWEN_CHECKPOINT_PATH`),
//! builds an `Mlp` from the layer's `gate`/`up`/`down` projections, and
//! compares `Mlp::forward` to MLX's `MLP.__call__`
//! (`down(silu(gate(x)) * up(x))`) at bf16-compose tolerance.
//!
//! Skips with a PASS + `eprintln!` when the env var is unset.

#![cfg(target_os = "macos")]
#![allow(clippy::cast_precision_loss)]

use std::fs;
use std::path::{Path, PathBuf};

use cider_press_models::nn::{Linear, Mlp, Module};
use cider_press_models::qwen2::{Qwen2Config, load_qwen2_weights};
use cider_press_runtime::{Device, Tensor};
use cider_press_test_utils::{dump_mlx_op, read_bf16, tempdir};
use half::bf16;
use safetensors::SafeTensors;

// SwiGLU = silu ∘ mul ∘ 3 projections; matches the attention_layer0 bar.
const ATOL: f32 = 5e-2;
const RTOL: f32 = 5e-2;

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

fn assert_close(label: &str, got: &[bf16], expected: &[bf16]) {
    assert_eq!(got.len(), expected.len(), "{label}: length mismatch");
    let mut worst_excess = 0.0f32;
    let mut worst_at = 0usize;
    let mut worst_pair = (0.0f32, 0.0f32);
    for (i, (a, b)) in got.iter().zip(expected.iter()).enumerate() {
        if a == b {
            continue;
        }
        let af = a.to_f32();
        let bfv = b.to_f32();
        let diff = (af - bfv).abs();
        let bound = ATOL + RTOL * bfv.abs();
        let excess = diff - bound;
        if excess > worst_excess {
            worst_excess = excess;
            worst_at = i;
            worst_pair = (af, bfv);
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

fn run_layer0_mlp(checkpoint: &Path, seq_len: usize) {
    let tmp = tempdir("models-mlp-layer0");
    let fixture_path = tmp.join(format!("mlp-layer0-seq{seq_len}.safetensors"));
    let checkpoint_str = checkpoint.to_str().expect("checkpoint path is UTF-8");
    let seq_len_s = seq_len.to_string();

    dump_mlx_op(
        &fixture_path,
        &[
            "mlp_layer0",
            "--checkpoint",
            checkpoint_str,
            "--seq-len",
            &seq_len_s,
        ],
    );

    let fixture_bytes = fs::read(&fixture_path).expect("read fixture");
    let st = SafeTensors::deserialize(&fixture_bytes).expect("parse safetensors");
    let x_ref = read_bf16(&st, "x");
    let y_ref = read_bf16(&st, "y");

    let config_bytes = fs::read(checkpoint.join("config.json")).expect("read config.json");
    let config = Qwen2Config::from_json_bytes(&config_bytes).expect("parse config.json");

    let archive_bytes =
        fs::read(checkpoint.join("model.safetensors")).expect("read model.safetensors");
    let archive = SafeTensors::deserialize(&archive_bytes).expect("deserialize safetensors");

    let device = Device::shared().expect("system default device");
    let weights = load_qwen2_weights(&archive, &config, &device).expect("load weights");
    let mlp_w = &weights.layers[0].mlp;

    let mlp = Mlp::new(
        Linear::new(mlp_w.gate_proj.clone(), None).expect("gate Linear"),
        Linear::new(mlp_w.up_proj.clone(), None).expect("up Linear"),
        Linear::new(mlp_w.down_proj.clone(), None).expect("down Linear"),
    );

    let hidden = Tensor::from_slice(&device, &x_ref, [1usize, seq_len, config.hidden_size])
        .expect("hidden tensor");

    let out = mlp.forward(&hidden).expect("mlp forward");
    out.eval().expect("eval");
    let got: Vec<bf16> = out.cpu_to_vec().expect("cpu_to_vec");

    assert_close(&format!("mlp_layer0 seq_len={seq_len}"), &got, &y_ref);
}

#[test]
fn mlp_layer0_prefill() {
    let Some(checkpoint) = checkpoint_path() else {
        eprintln!(
            "skipped: set CIDER_QWEN_CHECKPOINT_PATH to a local \
             Qwen2.5-0.5B-Instruct-4bit MLX checkout to run this test"
        );
        return;
    };
    run_layer0_mlp(&checkpoint, 8);
}

#[test]
fn mlp_layer0_decode() {
    let Some(checkpoint) = checkpoint_path() else {
        eprintln!(
            "skipped: set CIDER_QWEN_CHECKPOINT_PATH to a local \
             Qwen2.5-0.5B-Instruct-4bit MLX checkout to run this test"
        );
        return;
    };
    run_layer0_mlp(&checkpoint, 1);
}
