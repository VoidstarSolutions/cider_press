//! Parity test for `qwen2::attention::rope` against `mx.fast.rope`,
//! exercising both Q (14 heads) and K (2 heads) projection shapes.
//!
//! The models-layer surface is a thin config-binding wrapper
//! (`qwen2::attention::rope`); the algorithmic correctness was already
//! pinned bit-exact at the kernels + runtime layers. This test makes
//! sure the wrapper threads `rope_theta` and `head_dim` from
//! `Qwen2Config` correctly — running with the canonical pinned
//! config (`rope_theta = 1_000_000`, `head_dim = 64`) and checking
//! both Q and K shapes simultaneously.
//!
//! There is no "real-checkpoint" layer test because rope applies to
//! projection *outputs*, and the full layer-0 Q-projection → rope
//! flow is covered by the `attention_layer0` integration test.
//! The Qwen2-shape synthetic test is the right bar here.

#![cfg(target_os = "macos")]

use cider_press_models::qwen2::{Qwen2Config, attention};
use cider_press_runtime::{Device, Tensor};
use cider_press_test_utils::{dump_mlx_op, read_bf16, tempdir};
use half::bf16;
use safetensors::SafeTensors;

const B: usize = 1;
const T: usize = 4;
const D: usize = 64;
const H_Q: usize = 14;
const H_KV: usize = 2;

/// Subset of the pinned Qwen2.5-0.5B config that drives rope. Kept
/// inline so the test doesn't depend on `CIDER_QWEN_CHECKPOINT_PATH`
/// — the algorithmic parameters (`hidden_size`, `num_attention_heads`,
/// `rope_theta`) are all the helper consults.
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

#[test]
fn qwen2_rope_q_shape_matches_mlx() {
    parity_for(H_Q, 0);
}

#[test]
fn qwen2_rope_kv_shape_matches_mlx() {
    parity_for(H_KV, 0);
}

#[test]
fn qwen2_rope_q_shape_decode_offset() {
    parity_for(H_Q, 37);
}

fn parity_for(n_heads: usize, start_pos: i32) {
    let config = Qwen2Config::from_json_bytes(PINNED_CONFIG_JSON.as_bytes()).expect("parse config");
    assert_eq!(config.head_dim().unwrap(), D);

    let (input_host, out_ref) = fixture(n_heads, start_pos);
    let device = Device::system_default().expect("device");
    let input = Tensor::from_slice(&device, &input_host, [B, n_heads, T, D]).expect("input");
    let offset = Tensor::from_slice(&device, &[start_pos], [1]).expect("offset");

    let out = attention::rope(&input, &offset, &config).expect("schedule rope");
    out.eval().expect("eval");
    let got: Vec<bf16> = out.cpu_to_vec().expect("dense out");
    if start_pos == 0 {
        assert_eq!(
            got, out_ref,
            "qwen2::attention::rope (n_heads={n_heads}, offset=0) \
             must match mx.fast.rope bit-exactly",
        );
    } else {
        // metal::cos / metal::sin drift 1–2 bf16 ULPs across Apple
        // Silicon generations (M-series local vs macos-15 CI runners);
        // same situation as the sigmoid tolerance (hardware drift).
        assert_within_tolerance(
            &format!("qwen2::attention::rope (n_heads={n_heads}, offset={start_pos})"),
            &got,
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

fn fixture(n_heads: usize, start_pos: i32) -> (Vec<bf16>, Vec<bf16>) {
    let tmp = tempdir("qwen2-rope");
    let path = tmp.join(format!("rope-{n_heads}-{start_pos}.safetensors"));
    let shape = format!("{B},{n_heads},{T},{D}");
    let dims_str = D.to_string();
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
            "1000000.0",
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
