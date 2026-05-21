//! Parity test driving `Tensor::softmax` at Qwen2 attention-score
//! shapes derived from the pinned `Qwen2Config`.
//!
//! Branch 10 doesn't add a models-crate softmax wrapper — softmax is
//! a primitive `Tensor::softmax(precise)` call and `mlx.nn` doesn't
//! layer anything algorithmic on top of `mx.softmax`. The
//! models-crate test exists for the integration seam: it parses
//! `Qwen2Config`, computes the attention-score shape `[B, H_q, T, T]`
//! from `num_attention_heads`, and runs the runtime op against
//! `mx.softmax(x, axis=-1, precise=...)`.
//!
//! `precise=True` is the case to care about — that's the
//! configuration `mx.fast.scaled_dot_product_attention` selects for
//! bf16 attention scores in inference.

#![cfg(target_os = "macos")]
#![allow(clippy::cast_precision_loss)]

use cider_press_models::qwen2::Qwen2Config;
use cider_press_runtime::{Device, Tensor};
use cider_press_test_utils::{dump_mlx_op, read_bf16, tempdir};
use half::bf16;
use safetensors::SafeTensors;

/// Same pinned config snippet that `rope_qwen2.rs` uses.
const PINNED_CONFIG_JSON: &str = r#"{
    "architectures": ["Qwen2ForCausalLM"],
    "hidden_size": 896,
    "intermediate_size": 4864,
    "num_attention_heads": 14,
    "num_key_value_heads": 2,
    "num_hidden_layers": 24,
    "max_position_embeddings": 32768,
    "rope_theta": 1000000.0,
    "rms_norm_eps": 1e-06,
    "vocab_size": 151936,
    "tie_word_embeddings": true,
    "quantization": { "group_size": 64, "bits": 4 }
}"#;

const B: usize = 1;
const T: usize = 32;

#[test]
fn qwen2_attention_score_softmax_precise_matches_mlx() {
    let config = Qwen2Config::from_json_bytes(PINNED_CONFIG_JSON.as_bytes()).expect("parse config");
    let h_q = config.num_attention_heads;
    let (input_host, out_ref) = fixture(h_q, true);

    let device = Device::system_default().expect("device");
    let scores = Tensor::from_slice(&device, &input_host, [B, h_q, T, T]).expect("scores");
    let out = scores.softmax(true).expect("schedule softmax");
    out.eval().expect("eval");
    let got: Vec<bf16> = out.cpu_to_vec().expect("dense out");
    assert_within_tolerance(
        "Qwen2 attention softmax (precise)",
        &got,
        &out_ref,
        0.005,
        0.01,
    );
}

#[test]
fn qwen2_attention_score_softmax_default_matches_mlx() {
    let config = Qwen2Config::from_json_bytes(PINNED_CONFIG_JSON.as_bytes()).expect("parse config");
    let h_q = config.num_attention_heads;
    let (input_host, out_ref) = fixture(h_q, false);

    let device = Device::system_default().expect("device");
    let scores = Tensor::from_slice(&device, &input_host, [B, h_q, T, T]).expect("scores");
    let out = scores.softmax(false).expect("schedule softmax");
    out.eval().expect("eval");
    let got: Vec<bf16> = out.cpu_to_vec().expect("dense out");
    assert_within_tolerance(
        "Qwen2 attention softmax (default)",
        &got,
        &out_ref,
        0.01,
        0.04,
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
        "{label}: tolerance exceeded (max_abs={max_abs} > {abs_tol} \
         or max_rel={max_rel} > {rel_tol})"
    );
}

fn fixture(h_q: usize, precise: bool) -> (Vec<bf16>, Vec<bf16>) {
    let tmp = tempdir("models-softmax-qwen2");
    let path = tmp.join(format!(
        "softmax-{}.safetensors",
        if precise { "precise" } else { "default" }
    ));
    let shape = format!("{B},{h_q},{T},{T}");
    let mut args: Vec<&str> = vec!["softmax", "--lhs-shape", &shape, "--dtype", "bf16"];
    if precise {
        args.push("--precise");
    }
    dump_mlx_op(&path, &args);
    let bytes = std::fs::read(&path).expect("read fixture");
    let st = SafeTensors::deserialize(&bytes).expect("parse safetensors");
    (read_bf16(&st, "lhs"), read_bf16(&st, "out"))
}
