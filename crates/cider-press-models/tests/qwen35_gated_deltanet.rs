#![cfg(target_os = "macos")]

use std::{fs, path::PathBuf};

use cider_press_models::nn::rms_norm;
use cider_press_models::qwen3_5::{
    GatedDeltaNet, Qwen35Config, Qwen35MixerWeights, load_qwen35_weights,
};
use cider_press_runtime::{Device, Tensor};
use cider_press_test_utils::read_bf16;
use half::bf16;
use safetensors::SafeTensors;

const ATOL: f32 = 5e-2;
const RTOL: f32 = 5e-2;
const FIXTURES: &str = "/tmp/qwen3_5_4b_fixtures.safetensors";

fn checkpoint_path() -> Option<PathBuf> {
    std::env::var_os("CIDER_QWEN35_CHECKPOINT_PATH").map(PathBuf::from)
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
            "{label}[{i}]: got={a} want={b} > bound={bound} (max_abs so far={max_abs})"
        );
    }
    println!("{label}: max_abs={max_abs}, max_rel={max_rel}");
}

/// Layer-0 Gated-DeltaNet mixer parity vs mlx-lm. The fixture taps
/// `linear_attn(input_layernorm(layer0_in), ssm_mask, cache=None)` — the
/// isolated GDN mixer in prefill-from-zero state.
#[test]
fn gated_deltanet_layer0_matches_mlx() {
    let Some(dir) = checkpoint_path() else {
        eprintln!("skipping: CIDER_QWEN35_CHECKPOINT_PATH unset");
        return;
    };
    if !std::path::Path::new(FIXTURES).exists() {
        eprintln!("skipping: {FIXTURES} missing — run scripts/dump_qwen35_fixtures.py");
        return;
    }
    let config =
        Qwen35Config::from_json_bytes(&fs::read(dir.join("config.json")).unwrap()).unwrap();
    let archive_bytes = fs::read(dir.join("model.safetensors")).unwrap();
    let archive = SafeTensors::deserialize(&archive_bytes).unwrap();
    let device = Device::shared().unwrap();
    let weights = load_qwen35_weights(&archive, &config, &device).unwrap();

    let fx_bytes = fs::read(FIXTURES).unwrap();
    let fx = SafeTensors::deserialize(&fx_bytes).unwrap();
    let layer0_in = read_bf16(&fx, "layer0_in"); // [1, T, 2560]
    let expected = read_bf16(&fx, "layer0_attn_out"); // [1, T, 2560]
    let t = layer0_in.len() / config.hidden_size;

    let x = Tensor::from_slice(&device, &layer0_in, [1usize, t, config.hidden_size]).unwrap();
    // input_layernorm (layer 0 gamma) — the fixture taps linear_attn(input_layernorm(x)).
    #[allow(clippy::cast_possible_truncation)]
    let eps = config.rms_norm_eps as f32;
    let normed = rms_norm(&x, &weights.layers[0].input_layernorm, eps).unwrap();

    let Qwen35MixerWeights::LinearAttn(la) = &weights.layers[0].mixer else {
        panic!("layer 0 must be linear");
    };
    let gdn = GatedDeltaNet::from_weights(la, &config).unwrap();
    let out = gdn.forward(&normed).unwrap();
    out.eval().unwrap();
    let got: Vec<bf16> = out.cpu_to_vec().unwrap();
    assert_close("gated_deltanet layer0", &got, &expected);
}
