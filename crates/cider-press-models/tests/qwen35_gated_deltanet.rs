#![cfg(target_os = "macos")]

use std::{fs, path::PathBuf};

use cider_press_models::nn::rms_norm;
use cider_press_models::qwen3_5::{
    GatedDeltaNet, LinearState, Qwen35Config, Qwen35DecoderLayer, Qwen35MixerWeights,
    load_qwen35_weights,
};
use cider_press_runtime::{DType, Device, Tensor};
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
    let out = gdn.forward(&normed, None).unwrap();
    out.eval().unwrap();
    let got: Vec<bf16> = out.cpu_to_vec().unwrap();
    assert_close("gated_deltanet layer0", &got, &expected);
}

/// Whole linear (GDN) decoder layer parity vs mlx-lm. Layer 0 is a linear
/// mixer; `forward_linear` runs pre-norm GDN + residual, then pre-norm SwiGLU
/// MLP + residual (prefill from zero state, no KV cache). The fixture taps the
/// full decoder layer output `layer0_out`.
#[test]
fn decoder_layer0_matches_mlx() {
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
    let expected = read_bf16(&fx, "layer0_out"); // [1, T, 2560]
    let t = layer0_in.len() / config.hidden_size;

    let x = Tensor::from_slice(&device, &layer0_in, [1usize, t, config.hidden_size]).unwrap();
    let layer = Qwen35DecoderLayer::from_weights(&weights.layers[0], &config).unwrap();
    let out = layer.forward_linear(&x).unwrap();
    out.eval().unwrap();
    let got: Vec<bf16> = out.cpu_to_vec().unwrap();
    assert_close("decoder layer0", &got, &expected);
}

/// Prefill-decode cache-correctness gate. Runs the layer-0 GDN mixer two ways
/// over the same `T`-token input: (1) one-shot over `[0..T]` with no cache; (2)
/// split — prefill `[0..T-1]` threading a `LinearState`, then a single decode
/// step over `[T-1..T]` threading the updated cache. The last row of the
/// one-shot output must match the single decode row, proving the conv-window +
/// recurrent-state threading reconstructs the one-shot result.
#[test]
#[allow(clippy::many_single_char_names)]
fn gated_deltanet_prefill_decode_equivalence() {
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
    let t = layer0_in.len() / config.hidden_size;
    assert!(t >= 2, "need T >= 2 for prefill+decode split; got T={t}");

    let x = Tensor::from_slice(&device, &layer0_in, [1usize, t, config.hidden_size]).unwrap();
    #[allow(clippy::cast_possible_truncation)]
    let eps = config.rms_norm_eps as f32;
    let normed = rms_norm(&x, &weights.layers[0].input_layernorm, eps).unwrap();

    let Qwen35MixerWeights::LinearAttn(la) = &weights.layers[0].mixer else {
        panic!("layer 0 must be linear");
    };
    let gdn = GatedDeltaNet::from_weights(la, &config).unwrap();

    // (1) One-shot, no cache.
    let y_full = gdn.forward(&normed, None).unwrap();
    y_full.eval().unwrap();
    assert_eq!(y_full.shape().dims(), [1, t, config.hidden_size]);
    let full: Vec<bf16> = y_full.cpu_to_vec().unwrap();

    // (2) Split: prefill [0..T-1] then decode [T-1..T], threading a cache.
    let hv = config.linear_num_value_heads;
    let dv = config.linear_value_head_dim;
    let dk = config.linear_key_head_dim;
    let mut st = LinearState {
        conv_state: Tensor::zeros(&device, [1usize, 3, config.conv_dim()], DType::BF16).unwrap(),
        recurrent_state: Tensor::zeros(&device, [1usize, hv, dv, dk], DType::F32).unwrap(),
    };

    // Build compact prefill/decode inputs from the host rows. (A device-side
    // row-prefix slice+copy elides to a zero-copy view over the full backing
    // buffer, which the qmm reads as the full T — so round-trip through host
    // data to get genuinely compact [1, T-1, H] / [1, 1, H] leaves.)
    normed.eval().unwrap();
    let normed_host: Vec<bf16> = normed.cpu_to_vec().unwrap();
    let h = config.hidden_size;
    let normed_prefill =
        Tensor::from_slice(&device, &normed_host[..(t - 1) * h], [1usize, t - 1, h]).unwrap();
    let normed_dec =
        Tensor::from_slice(&device, &normed_host[(t - 1) * h..t * h], [1usize, 1, h]).unwrap();

    let _ = gdn.forward(&normed_prefill, Some(&mut st)).unwrap();
    // CRITICAL: commit the lazy cache Tensors before the decode reads them
    // (producer→consumer split across evals — the un-evaluated conv/recurrent
    // writes would otherwise be dropped by the next call).
    st.conv_state.eval().unwrap();
    st.recurrent_state.eval().unwrap();

    let y_dec = gdn.forward(&normed_dec, Some(&mut st)).unwrap();
    y_dec.eval().unwrap();
    assert_eq!(y_dec.shape().dims(), [1, 1, config.hidden_size]);
    let dec: Vec<bf16> = y_dec.cpu_to_vec().unwrap();

    // Compare the last row of the one-shot output to the single decode row.
    // atol/rtol 0.02: the recurrence accumulates over T steps, so the split path
    // (T-1 prefill steps + 1 decode step) drifts a little from the one-shot path.
    let last_row = &full[(t - 1) * config.hidden_size..t * config.hidden_size];
    let (atol, rtol) = (0.02f32, 0.02f32);
    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    for (i, (g, r)) in dec.iter().zip(last_row.iter()).enumerate() {
        let (g, r) = (g.to_f32(), r.to_f32());
        let abs = (g - r).abs();
        max_abs = max_abs.max(abs);
        if r.abs() > 1e-6 {
            max_rel = max_rel.max(abs / r.abs());
        }
        let bound = atol + rtol * r.abs();
        assert!(
            abs <= bound,
            "gdn prefill-decode equivalence[{i}]: decode={g} one-shot={r} > bound={bound} \
             (max_abs so far={max_abs})"
        );
    }
    println!("gdn prefill-decode equivalence: max_abs={max_abs}, max_rel={max_rel}");
}
