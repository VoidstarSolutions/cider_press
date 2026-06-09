//! Smoke tests for [`Generator::profiled_prefill`].
//!
//! Synthetic-model construction is inlined from `generator_smoke.rs` so this
//! file is independently runnable without a shared `tests/common/` module.

use std::collections::HashSet;

use cider_press_models::qwen2::{
    Qwen2AttentionWeights, Qwen2Config, Qwen2LayerWeights, Qwen2MlpWeights, Qwen2Model,
    Qwen2Weights,
};
use cider_press_runtime::{Device, Quantization, QuantizedWeight, Tensor};
use half::bf16;

const HIDDEN: usize = 128;
const VOCAB: usize = 128;
const LAYERS: usize = 2;
const HEADS: usize = 2;
const KV_HEADS: usize = 1;
const HEAD_DIM: usize = 64;
const INTERMEDIATE: usize = 128;
const GROUP_SIZE: usize = 64;
const BITS: usize = 4;

fn synthetic_config() -> Qwen2Config {
    let json = serde_json::json!({
        "hidden_size": HIDDEN,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "intermediate_size": INTERMEDIATE,
        "vocab_size": VOCAB,
        "max_position_embeddings": 256,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1.0e-6,
        "tie_word_embeddings": true,
        "quantization": {
            "group_size": GROUP_SIZE,
            "bits": BITS,
        },
    });
    Qwen2Config::from_json_bytes(&serde_json::to_vec(&json).unwrap())
        .expect("synthetic Qwen2Config")
}

/// Deterministic-but-non-trivial byte pattern (avoids all-zero weights
/// which produce all-zero logits and ambiguous argmaxes).
fn fill(seed: u8, len: usize) -> Vec<u8> {
    #[allow(clippy::cast_possible_truncation)]
    (0..len)
        .map(|i| seed.wrapping_add((i as u8) ^ 0xA5))
        .collect()
}

fn random_qweight(device: &Device, n: usize, k: usize, seed: u8) -> QuantizedWeight {
    let q = Quantization::Q4_GS64; // 4-bit, group_size=64
    let elem_count = n * k;
    let w_bytes = (elem_count * BITS / 8).max(4);
    let aux_bytes = (elem_count / GROUP_SIZE) * std::mem::size_of::<bf16>();
    QuantizedWeight::from_bytes(
        device,
        [n, k],
        q,
        &fill(seed, w_bytes),
        &fill(seed.wrapping_add(1), aux_bytes),
        &fill(seed.wrapping_add(2), aux_bytes),
    )
    .expect("from_bytes")
}

fn random_dense(device: &Device, dim: usize, seed: u8) -> Tensor {
    let data: Vec<bf16> = (0..dim)
        .map(|i| {
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_possible_wrap,
                clippy::cast_precision_loss
            )]
            let v = (i32::from(seed) + i as i32) as f32 * 1e-3;
            bf16::from_f32(v)
        })
        .collect();
    Tensor::from_slice(device, &data, [dim]).expect("from_slice")
}

fn synthetic_model(device: &Device) -> Qwen2Model {
    let cfg = synthetic_config();
    let q_dim = HEADS * HEAD_DIM;
    let kv_dim = KV_HEADS * HEAD_DIM;
    let layers = (0..LAYERS)
        .map(|li| {
            #[allow(clippy::cast_possible_truncation)]
            let s = (li as u8) * 16;
            Qwen2LayerWeights {
                input_layernorm: random_dense(device, HIDDEN, s + 1),
                post_attention_layernorm: random_dense(device, HIDDEN, s + 2),
                attention: Qwen2AttentionWeights {
                    q_proj: random_qweight(device, q_dim, HIDDEN, s + 3),
                    k_proj: random_qweight(device, kv_dim, HIDDEN, s + 4),
                    v_proj: random_qweight(device, kv_dim, HIDDEN, s + 5),
                    o_proj: random_qweight(device, HIDDEN, q_dim, s + 6),
                    q_bias: random_dense(device, q_dim, s + 7),
                    k_bias: random_dense(device, kv_dim, s + 8),
                    v_bias: random_dense(device, kv_dim, s + 9),
                    // The padded decode twins. HIDDEN=128 is too small for
                    // qmv_fast (requires K%512==0 and buffer ≥1024B), so use
                    // the unpadded shape (k_physical==k_logical). PhasedLinear
                    // still routes T=1 decode through these twins; fast-path
                    // selection is a real-model concern, not a smoke-test one.
                    q_proj_padded: random_qweight(device, q_dim, HIDDEN, s + 3),
                    k_proj_padded: random_qweight(device, kv_dim, HIDDEN, s + 4),
                    v_proj_padded: random_qweight(device, kv_dim, HIDDEN, s + 5),
                    o_proj_padded: random_qweight(device, HIDDEN, q_dim, s + 6),
                },
                mlp: Qwen2MlpWeights {
                    gate_proj: random_qweight(device, INTERMEDIATE, HIDDEN, s + 10),
                    up_proj: random_qweight(device, INTERMEDIATE, HIDDEN, s + 11),
                    down_proj: random_qweight(device, HIDDEN, INTERMEDIATE, s + 12),
                },
            }
        })
        .collect();
    let weights = Qwen2Weights {
        embed: random_qweight(device, VOCAB, HIDDEN, 100),
        norm: random_dense(device, HIDDEN, 200),
        layers,
    };
    Qwen2Model::from_weights(weights, cfg).expect("from_weights")
}

#[test]
fn profiled_prefill_runs_on_synthetic_model() {
    let device = Device::shared().expect("device");
    // `profiled_eval` needs GPU stage-boundary counter sampling; CI runners
    // (and some Apple-Silicon generations) lack it. Skip there — `prefill_sync`
    // exercises the same forward without counters.
    if !device.supports_stage_boundary_sampling() {
        eprintln!("SKIP: no stage-boundary sampling (profiled_eval unavailable)");
        return;
    }
    let model = synthetic_model(&device);
    let eos = HashSet::from([999u32]);
    let mut generator =
        cider_press_models::generator::Generator::new(model, 64, eos).expect("generator");
    // T = 5 prompt (> 1, exercises the composed prefill attention path).
    let ids = [1u32, 2, 3, 4, 5];
    generator
        .profiled_prefill(&ids)
        .expect("profiled prefill runs");
}

#[test]
fn profiled_prefill_rejects_empty_ids() {
    let device = Device::shared().expect("device");
    let model = synthetic_model(&device);
    let eos = HashSet::from([999u32]);
    let mut generator =
        cider_press_models::generator::Generator::new(model, 64, eos).expect("generator");
    assert!(generator.profiled_prefill(&[]).is_err());
}

#[test]
fn prefill_sync_runs_on_synthetic_model() {
    let device = Device::shared().expect("device");
    let model = synthetic_model(&device);
    let eos = HashSet::from([999u32]);
    let mut generator =
        cider_press_models::generator::Generator::new(model, 64, eos).expect("generator");
    generator
        .prefill_sync(&[1, 2, 3, 4, 5])
        .expect("prefill_sync runs");
}

#[test]
fn prefill_sync_rejects_empty_ids() {
    let device = Device::shared().expect("device");
    let model = synthetic_model(&device);
    let eos = HashSet::from([999u32]);
    let mut generator =
        cider_press_models::generator::Generator::new(model, 64, eos).expect("generator");
    assert!(generator.prefill_sync(&[]).is_err());
}
