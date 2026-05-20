//! Gated integration test against the real
//! `mlx-community/Qwen2.5-0.5B-Instruct-4bit` checkpoint pinned in
//! [`cider_press_models::qwen2::HF_REVISION`].
//!
//! Skipped by default; runs only when `CIDER_QWEN_CHECKPOINT_PATH`
//! points at a local checkout of the HF repo at the pinned revision.
//! Use:
//!
//! ```sh
//! CIDER_QWEN_CHECKPOINT_PATH=/path/to/Qwen2.5-0.5B-Instruct-4bit \
//!     cargo test -p cider-press-models --test qwen2_real_checkpoint -- --nocapture
//! ```
//!
//! The directory must contain `config.json` and `model.safetensors`.
//! No download is performed by this test.

use std::fs;
use std::path::PathBuf;

use cider_press_models::qwen2::{Qwen2Config, load_qwen2_weights};
use cider_press_runtime::Device;
use safetensors::SafeTensors;

/// Skip-or-load helper. Returns `None` if the env var isn't set,
/// otherwise returns the canonicalized absolute path. Panics if the
/// env var is set but the path doesn't resolve to a directory —
/// that's almost always operator error worth surfacing immediately.
fn checkpoint_path() -> Option<PathBuf> {
    let raw = std::env::var("CIDER_QWEN_CHECKPOINT_PATH").ok()?;
    let path = fs::canonicalize(&raw).unwrap_or_else(|err| {
        panic!("CIDER_QWEN_CHECKPOINT_PATH={raw} does not resolve: {err}")
    });
    assert!(
        path.is_dir(),
        "CIDER_QWEN_CHECKPOINT_PATH={} is not a directory",
        path.display(),
    );
    Some(path)
}

#[test]
fn real_checkpoint_loads_and_byte_round_trips() {
    let Some(dir) = checkpoint_path() else {
        eprintln!(
            "skipped: set CIDER_QWEN_CHECKPOINT_PATH to a local Qwen2.5-0.5B-Instruct-4bit \
             checkout to run this test"
        );
        return;
    };

    let config_bytes = fs::read(dir.join("config.json")).expect("read config.json");
    let config = Qwen2Config::from_json_bytes(&config_bytes).expect("parse config.json");
    assert!(
        config.quantization.is_some(),
        "checkpoint is missing quantization metadata; expected MLX 4-bit",
    );

    let archive_bytes = fs::read(dir.join("model.safetensors")).expect("read model.safetensors");
    let archive = SafeTensors::deserialize(&archive_bytes).expect("deserialize safetensors");

    let device = Device::shared().expect("system default device");
    let weights = load_qwen2_weights(&archive, &config, &device).expect("load");

    // Structural sanity: layer count matches config.
    assert_eq!(weights.layers.len(), config.num_hidden_layers);

    // Spot-check a few bytes-match assertions on layer 0 (the layer
    // every later op branch validates against).
    let layer0 = &weights.layers[0];
    let dense_match = |key: &str, t: &cider_press_runtime::Tensor| {
        let arc = archive.tensor(key).expect("present");
        assert_eq!(t.cpu_bytes().expect("dense bytes"), arc.data(), "{key}");
    };
    dense_match("model.norm.weight", &weights.norm);
    dense_match(
        "model.layers.0.input_layernorm.weight",
        &layer0.input_layernorm,
    );
    dense_match(
        "model.layers.0.post_attention_layernorm.weight",
        &layer0.post_attention_layernorm,
    );
    dense_match(
        "model.layers.0.self_attn.q_proj.bias",
        &layer0.attention.q_bias,
    );

    let q_match = |base: &str, qw: &cider_press_runtime::QuantizedWeight| {
        let (w, s, b) = qw.component_bytes();
        assert_eq!(
            w,
            archive.tensor(&format!("{base}.weight")).unwrap().data(),
            "{base}.weight"
        );
        assert_eq!(
            s,
            archive.tensor(&format!("{base}.scales")).unwrap().data(),
            "{base}.scales"
        );
        assert_eq!(
            b,
            archive.tensor(&format!("{base}.biases")).unwrap().data(),
            "{base}.biases"
        );
    };
    q_match("model.embed_tokens", &weights.embed);
    q_match("model.layers.0.self_attn.q_proj", &layer0.attention.q_proj);
    q_match("model.layers.0.mlp.gate_proj", &layer0.mlp.gate_proj);
}
