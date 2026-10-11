#![cfg(target_os = "macos")]

use cider_press_models::safetensors_io::ShardSet;
use std::fs;

use cider_press_models::nn::rms_norm;
use cider_press_models::qwen3_5::{
    GatedDeltaNet, Qwen35Config, Qwen35DecoderLayer, Qwen35MixerWeights, load_qwen35_weights,
};
use cider_press_runtime::{Device, Tensor};
use cider_press_test_utils::{assert_allclose, qwen35_4b_checkpoint_dir, read_bf16, require_file};
use half::bf16;
use safetensors::SafeTensors;

const ATOL: f32 = 5e-2;
const RTOL: f32 = 5e-2;
const FIXTURES: &str = "/tmp/qwen3_5_4b_fixtures.safetensors";
const FIXTURES_REGEN: &str = "uv run scripts/dump_qwen35_fixtures.py";

/// Layer-0 Gated-DeltaNet mixer parity vs mlx-lm. The fixture taps
/// `linear_attn(input_layernorm(layer0_in), ssm_mask, cache=None)` — the
/// isolated GDN mixer in prefill-from-zero state.
#[test]
#[ignore = "requires a local checkpoint; run with --ignored (see panic message for the env var)"]
fn gated_deltanet_layer0_matches_mlx() {
    let dir = qwen35_4b_checkpoint_dir();
    let fixtures = require_file(FIXTURES, FIXTURES_REGEN);
    let config =
        Qwen35Config::from_json_bytes(&fs::read(dir.join("config.json")).unwrap()).unwrap();
    let shards = ShardSet::read(&dir).unwrap();
    let archive = shards.archive().unwrap();
    let device = Device::shared().unwrap();
    let weights = load_qwen35_weights(&archive, &config, &device).unwrap();

    let fx_bytes = fs::read(&fixtures).unwrap();
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
    let mut state = gdn.make_cache(&device).unwrap();
    let out = gdn.forward(&normed, &mut state).unwrap();
    out.eval().unwrap();
    let got: Vec<bf16> = out.cpu_to_vec().unwrap();
    assert_allclose("gated_deltanet layer0", &got, &expected, ATOL, RTOL);
}

/// Whole linear (GDN) decoder layer parity vs mlx-lm. Layer 0 is a linear
/// mixer; the unified `forward` dispatches on a `Linear` cache, running
/// pre-norm GDN + residual, then pre-norm SwiGLU MLP + residual (prefill from
/// zero state). The `offset` is ignored on the linear path. The fixture taps
/// the full decoder layer output `layer0_out`.
#[test]
#[ignore = "requires a local checkpoint; run with --ignored (see panic message for the env var)"]
fn decoder_layer0_matches_mlx() {
    let dir = qwen35_4b_checkpoint_dir();
    let fixtures = require_file(FIXTURES, FIXTURES_REGEN);
    let config =
        Qwen35Config::from_json_bytes(&fs::read(dir.join("config.json")).unwrap()).unwrap();
    let shards = ShardSet::read(&dir).unwrap();
    let archive = shards.archive().unwrap();
    let device = Device::shared().unwrap();
    let weights = load_qwen35_weights(&archive, &config, &device).unwrap();

    let fx_bytes = fs::read(&fixtures).unwrap();
    let fx = SafeTensors::deserialize(&fx_bytes).unwrap();
    let layer0_in = read_bf16(&fx, "layer0_in"); // [1, T, 2560]
    let expected = read_bf16(&fx, "layer0_out"); // [1, T, 2560]
    let t = layer0_in.len() / config.hidden_size;

    let x = Tensor::from_slice(&device, &layer0_in, [1usize, t, config.hidden_size]).unwrap();
    let layer = Qwen35DecoderLayer::from_weights(&weights.layers[0], &config).unwrap();
    let mut cache = layer.make_cache(t, &device).unwrap();
    let offset = Tensor::from_slice(&device, &[0i32], [1usize]).unwrap();
    let out = layer.forward(&x, &offset, &mut cache).unwrap();
    out.eval().unwrap();
    let got: Vec<bf16> = out.cpu_to_vec().unwrap();
    assert_allclose("decoder layer0", &got, &expected, ATOL, RTOL);
}

/// Prefill-decode cache-correctness gate. Runs the layer-0 GDN mixer two ways
/// over the same `T`-token input: (1) one-shot over `[0..T]` from a fresh state; (2)
/// split — prefill `[0..T-1]` threading a `LinearState`, then a single decode
/// step over `[T-1..T]` threading the updated cache. The last row of the
/// one-shot output must match the single decode row, proving the conv-window +
/// recurrent-state threading reconstructs the one-shot result.
#[test]
#[ignore = "requires a local checkpoint; run with --ignored (see panic message for the env var)"]
#[allow(clippy::many_single_char_names)]
fn gated_deltanet_prefill_decode_equivalence() {
    let dir = qwen35_4b_checkpoint_dir();
    let fixtures = require_file(FIXTURES, FIXTURES_REGEN);
    let config =
        Qwen35Config::from_json_bytes(&fs::read(dir.join("config.json")).unwrap()).unwrap();
    let shards = ShardSet::read(&dir).unwrap();
    let archive = shards.archive().unwrap();
    let device = Device::shared().unwrap();
    let weights = load_qwen35_weights(&archive, &config, &device).unwrap();

    let fx_bytes = fs::read(&fixtures).unwrap();
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

    // (1) One-shot from a fresh state.
    let mut st_full = gdn.make_cache(&device).unwrap();
    let y_full = gdn.forward(&normed, &mut st_full).unwrap();
    y_full.eval().unwrap();
    assert_eq!(y_full.shape().dims(), [1, t, config.hidden_size]);
    let full: Vec<bf16> = y_full.cpu_to_vec().unwrap();

    // (2) Split: prefill [0..T-1] then decode [T-1..T], threading a cache.
    let mut st = gdn.make_cache(&device).unwrap();

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

    let _ = gdn.forward(&normed_prefill, &mut st).unwrap();
    // CRITICAL: commit the lazy cache Tensors before the decode reads them
    // (producer→consumer split across evals — the un-evaluated conv/recurrent
    // writes would otherwise be dropped by the next call).
    st.conv_state().eval().unwrap();
    st.recurrent_state().eval().unwrap();

    let y_dec = gdn.forward(&normed_dec, &mut st).unwrap();
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
