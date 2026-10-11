//! Gated integration test against the real
//! `mlx-community/Qwen2.5-0.5B-Instruct-4bit` checkpoint pinned in
//! [`cider_press_models::qwen2::HF_REVISION`].
//!
//! `#[ignore]`d by default; run with `--ignored` and
//! `CIDER_QWEN_CHECKPOINT_PATH` pointing at a local checkout of the HF repo
//! at the pinned revision.
//! Use:
//!
//! ```sh
//! CIDER_QWEN_CHECKPOINT_PATH=/path/to/Qwen2.5-0.5B-Instruct-4bit \
//!     cargo test -p cider-press-models --test qwen2_real_checkpoint -- --nocapture
//! ```
//!
//! The directory must contain `config.json` and `model.safetensors`.
//! No download is performed by this test.

use cider_press_models::safetensors_io::{ShardSet, TensorLookup};
use cider_press_test_utils::require_checkpoint_dir;
use std::fs;
use std::path::PathBuf;

use cider_press_models::qwen2::{Qwen2Config, load_qwen2_weights};
use cider_press_runtime::Device;

/// Returns the canonicalized checkpoint directory. Panics if the env var is
/// unset or the path doesn't resolve to a directory.
fn checkpoint_path() -> PathBuf {
    require_checkpoint_dir(
        "CIDER_QWEN_CHECKPOINT_PATH",
        "a local Qwen2.5-0.5B-Instruct-4bit checkpoint directory",
    )
}

#[test]
#[ignore = "requires a local checkpoint; run with --ignored (see panic message for the env var)"]
fn real_checkpoint_loads_and_byte_round_trips() {
    let dir = checkpoint_path();

    let config_bytes = fs::read(dir.join("config.json")).expect("read config.json");
    let config = Qwen2Config::from_json_bytes(&config_bytes).expect("parse config.json");
    assert!(
        config.quantization.is_some(),
        "checkpoint is missing quantization metadata; expected MLX 4-bit",
    );

    let shards = ShardSet::read(&dir).expect("load checkpoint");
    let archive = shards.archive().expect("load checkpoint");

    let device = Device::shared().expect("system default device");
    let weights = load_qwen2_weights(&archive, &config, &device).expect("load");

    // Structural sanity: layer count matches config.
    assert_eq!(weights.layers.len(), config.num_hidden_layers);

    // Spot-check a few bytes-match assertions on layer 0 (the layer
    // every later op branch validates against).
    let layer0 = &weights.layers[0];
    let dense_match = |key: &str, t: &cider_press_runtime::Tensor| {
        let arc = archive.view(key).expect("present");
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
            archive.view(&format!("{base}.weight")).unwrap().data(),
            "{base}.weight"
        );
        assert_eq!(
            s,
            archive.view(&format!("{base}.scales")).unwrap().data(),
            "{base}.scales"
        );
        assert_eq!(
            b,
            archive.view(&format!("{base}.biases")).unwrap().data(),
            "{base}.biases"
        );
    };
    q_match("model.embed_tokens", &weights.embed);
    q_match("model.layers.0.self_attn.q_proj", &layer0.attention.q_proj);
    q_match("model.layers.0.mlp.gate_proj", &layer0.mlp.gate_proj);
}
