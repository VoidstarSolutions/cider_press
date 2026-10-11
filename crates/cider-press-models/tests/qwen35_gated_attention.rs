#![cfg(target_os = "macos")]

use cider_press_models::safetensors_io::ShardSet;
use std::{fs, path::PathBuf};

use cider_press_models::nn::rms_norm;
use cider_press_models::qwen3_5::{
    GatedAttention, Qwen35Config, Qwen35DecoderLayer, Qwen35LayerCache, Qwen35MixerWeights,
    load_qwen35_weights,
};
use cider_press_runtime::{DType, Device, KvCache, Tensor};
use cider_press_test_utils::{read_bf16, require_checkpoint_dir, require_file};
use half::bf16;
use safetensors::SafeTensors;

const ATOL: f32 = 5e-2;
const RTOL: f32 = 5e-2;
const FIXTURES: &str = "/tmp/qwen3_5_4b_fixtures.safetensors";
const FIXTURES_REGEN: &str = "uv run scripts/dump_qwen35_fixtures.py";

fn checkpoint_path() -> PathBuf {
    require_checkpoint_dir(
        "CIDER_QWEN35_CHECKPOINT_PATH",
        "a local Qwen3.5-4B-4bit MLX checkpoint directory",
    )
}

fn assert_close(label: &str, got: &[bf16], expected: &[bf16]) {
    assert_eq!(got.len(), expected.len(), "{label}: length mismatch");
    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    for (i, (a, b)) in got.iter().zip(expected.iter()).enumerate() {
        let (a, b) = (a.to_f32(), b.to_f32());
        let abs = (a - b).abs();
        max_abs = max_abs.max(abs);
        if b.abs() > 1e-6 {
            max_rel = max_rel.max(abs / b.abs());
        }
        let bound = ATOL + RTOL * b.abs();
        assert!(
            abs <= bound,
            "{label}[{i}]: got={a} want={b} > bound={bound}"
        );
    }
    println!("{label}: max_abs={max_abs}, max_rel={max_rel}");
}

#[test]
#[ignore = "requires a local checkpoint; run with --ignored (see panic message for the env var)"]
fn gated_attention_layer3_matches_mlx() {
    let dir = checkpoint_path();
    let fixtures = require_file(FIXTURES, FIXTURES_REGEN);
    let config =
        Qwen35Config::from_json_bytes(&fs::read(dir.join("config.json")).unwrap()).unwrap();
    let shards = ShardSet::read(&dir).unwrap();
    let archive = shards.archive().unwrap();
    let device = Device::shared().unwrap();
    let weights = load_qwen35_weights(&archive, &config, &device).unwrap();

    let fx_bytes = fs::read(&fixtures).unwrap();
    let fx = SafeTensors::deserialize(&fx_bytes).unwrap();
    let layer3_in = read_bf16(&fx, "layer3_in"); // [1,6,2560]
    let expected = read_bf16(&fx, "layer3_attn_out"); // [1,6,2560]
    let t = layer3_in.len() / config.hidden_size;

    let x = Tensor::from_slice(&device, &layer3_in, [1usize, t, config.hidden_size]).unwrap();
    // input_layernorm (layer 3 gamma) — the fixture taps self_attn(input_layernorm(x)).
    #[allow(clippy::cast_possible_truncation)]
    let eps = config.rms_norm_eps as f32;
    let normed = rms_norm(&x, &weights.layers[3].input_layernorm, eps).unwrap();

    let Qwen35MixerWeights::FullAttn(fa) = &weights.layers[3].mixer else {
        panic!("layer 3 must be full-attention");
    };
    let attn = GatedAttention::from_weights(fa, &config).unwrap();
    let mut cache = KvCache::new(
        &device,
        t,
        config.num_key_value_heads,
        config.head_dim,
        DType::BF16,
    )
    .unwrap();
    let offset = Tensor::from_slice(&device, &[0i32], [1usize]).unwrap();

    let out = attn.forward(&normed, &offset, &mut cache).unwrap();
    out.eval().unwrap();
    let got: Vec<bf16> = out.cpu_to_vec().unwrap();
    assert_close("gated_attention layer3", &got, &expected);
}

#[test]
#[ignore = "requires a local checkpoint; run with --ignored (see panic message for the env var)"]
fn decoder_layer3_matches_mlx() {
    let dir = checkpoint_path();
    let fixtures = require_file(FIXTURES, FIXTURES_REGEN);
    let config =
        Qwen35Config::from_json_bytes(&fs::read(dir.join("config.json")).unwrap()).unwrap();
    let shards = ShardSet::read(&dir).unwrap();
    let archive = shards.archive().unwrap();
    let device = Device::shared().unwrap();
    let weights = load_qwen35_weights(&archive, &config, &device).unwrap();

    let fx_bytes = fs::read(&fixtures).unwrap();
    let fx = SafeTensors::deserialize(&fx_bytes).unwrap();
    let layer3_in = read_bf16(&fx, "layer3_in");
    let expected = read_bf16(&fx, "layer3_out");
    let t = layer3_in.len() / config.hidden_size;

    let x = Tensor::from_slice(&device, &layer3_in, [1usize, t, config.hidden_size]).unwrap();
    let layer = Qwen35DecoderLayer::from_weights(&weights.layers[3], &config).unwrap();
    let mut cache = Qwen35LayerCache::Full(
        KvCache::new(
            &device,
            t,
            config.num_key_value_heads,
            config.head_dim,
            DType::BF16,
        )
        .unwrap(),
    );
    let offset = Tensor::from_slice(&device, &[0i32], [1usize]).unwrap();

    let out = layer.forward(&x, &offset, &mut cache).unwrap();
    out.eval().unwrap();
    let got: Vec<bf16> = out.cpu_to_vec().unwrap();
    assert_close("decoder layer3", &got, &expected);
}
