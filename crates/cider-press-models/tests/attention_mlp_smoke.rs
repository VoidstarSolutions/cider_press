//! Synthetic (no-checkpoint) tests for the branch-12b model layers:
//! shape/dtype/lazy smoke coverage plus `Attention::forward`'s input
//! rejection paths (the negative coverage carried over from the removed
//! `projections_smoke.rs`).
//!
//! Numerical parity against MLX is covered by the gated `mlp_layer0.rs`
//! and `attention_layer0.rs` integration tests. The positive tests here
//! just confirm the lazy graph builds with the right output shape/dtype
//! on zero-filled Qwen2.5-0.5B-shaped weights.

#![cfg(target_os = "macos")]

use cider_press_models::Error;
use cider_press_models::nn::{Linear, Mlp, Module};
use cider_press_models::qwen2::{Qwen2AttentionWeights, Qwen2Config, attention::Attention};
use cider_press_runtime::{DType, Device, KvCache, Quantization, QuantizedWeight, Tensor};
use half::bf16;

// Qwen2.5-0.5B-Instruct dimensions.
const HIDDEN_SIZE: usize = 896;
const INTERMEDIATE: usize = 4864;
const T: usize = 4;
const NUM_Q_HEADS: usize = 14;
const NUM_KV_HEADS: usize = 2;
const HEAD_DIM: usize = 64;

/// Zero-filled `QuantizedWeight` with logical `[n, k]`, q4-gs64.
fn zero_qw(device: &Device, n: usize, k: usize) -> QuantizedWeight {
    let elem_count = n * k;
    let w_bytes = elem_count * 4 / 32 * 4;
    let aux_bytes = (elem_count / 64) * 2;
    QuantizedWeight::from_bytes(
        device,
        [n, k],
        Quantization::Q4_GS64,
        &vec![0u8; w_bytes],
        &vec![0u8; aux_bytes],
        &vec![0u8; aux_bytes],
    )
    .expect("zero QuantizedWeight")
}

fn synthetic_config() -> Qwen2Config {
    // Qwen2.5-0.5B-Instruct architecture fields. Constructed via
    // `from_json_bytes` because `Qwen2Config` is `#[non_exhaustive]`.
    Qwen2Config::from_json_bytes(
        br#"{
            "hidden_size": 896,
            "num_hidden_layers": 1,
            "num_attention_heads": 14,
            "num_key_value_heads": 2,
            "intermediate_size": 4864,
            "vocab_size": 151936,
            "max_position_embeddings": 32768,
            "rope_theta": 1000000.0,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": true,
            "quantization": { "group_size": 64, "bits": 4 }
        }"#,
    )
    .expect("synthetic config")
}

/// Build a zero-filled dense BF16 tensor of the given 1D length.
fn zero_bf16_1d(device: &Device, len: usize) -> Tensor {
    Tensor::from_slice(device, &vec![bf16::ZERO; len], [len]).expect("zero bf16 tensor")
}

fn synthetic_attention_weights(device: &Device) -> Qwen2AttentionWeights {
    let q_dim = NUM_Q_HEADS * HEAD_DIM; // 896
    let kv_dim = NUM_KV_HEADS * HEAD_DIM; // 128

    Qwen2AttentionWeights {
        q_proj: zero_qw(device, q_dim, HIDDEN_SIZE),
        k_proj: zero_qw(device, kv_dim, HIDDEN_SIZE),
        v_proj: zero_qw(device, kv_dim, HIDDEN_SIZE),
        o_proj: zero_qw(device, HIDDEN_SIZE, q_dim),
        q_bias: zero_bf16_1d(device, q_dim),
        k_bias: zero_bf16_1d(device, kv_dim),
        v_bias: zero_bf16_1d(device, kv_dim),
    }
}

#[test]
fn attention_forward_shape_dtype_and_lazy() {
    let device = Device::shared().expect("device");
    let config = synthetic_config();
    let weights = synthetic_attention_weights(&device);
    let attn = Attention::from_weights(&weights, &config).expect("from_weights");

    let hidden = Tensor::from_slice(
        &device,
        &vec![bf16::ZERO; T * HIDDEN_SIZE],
        [1usize, T, HIDDEN_SIZE],
    )
    .expect("hidden");
    let offset = Tensor::from_slice(&device, &[0i32], [1usize]).expect("offset");
    let mut cache =
        KvCache::new(&device, T + 4, NUM_KV_HEADS, HEAD_DIM, DType::BF16).expect("KvCache::new");

    let out = attn
        .forward(&hidden, None, &offset, &mut cache)
        .expect("attention forward");

    assert_eq!(out.shape().dims(), &[1, T, HIDDEN_SIZE], "attn out shape");
    assert_eq!(out.dtype(), DType::BF16, "attn out dtype");
    assert!(!out.is_materialized(), "attn out should be lazy");
}

/// Build an `Attention` plus a fresh cache + offset for the rejection
/// tests, returning the pieces the caller drives `forward` with.
fn attention_fixture() -> (Attention, Tensor, KvCache) {
    let device = Device::shared().expect("device");
    let config = synthetic_config();
    let weights = synthetic_attention_weights(&device);
    let attn = Attention::from_weights(&weights, &config).expect("from_weights");
    let offset = Tensor::from_slice(&device, &[0i32], [1usize]).expect("offset");
    let cache =
        KvCache::new(&device, T + 4, NUM_KV_HEADS, HEAD_DIM, DType::BF16).expect("KvCache::new");
    (attn, offset, cache)
}

#[test]
fn attention_forward_rejects_rank2_hidden() {
    let device = Device::shared().expect("device");
    let (attn, offset, mut cache) = attention_fixture();
    // Rank-2 input: missing the batch dimension.
    let hidden = Tensor::from_slice(
        &device,
        &vec![bf16::ZERO; T * HIDDEN_SIZE],
        [T, HIDDEN_SIZE],
    )
    .expect("rank-2 hidden");
    let err = attn
        .forward(&hidden, None, &offset, &mut cache)
        .expect_err("should reject rank-2 hidden");
    assert!(
        matches!(err, Error::InvalidArgument(_)),
        "expected InvalidArgument, got {err:?}"
    );
}

#[test]
fn attention_forward_rejects_batch_gt_1() {
    let device = Device::shared().expect("device");
    let (attn, offset, mut cache) = attention_fixture();
    // B=2: KvCache is single-sequence, so batch > 1 must be rejected.
    let hidden = Tensor::from_slice(
        &device,
        &vec![bf16::ZERO; 2 * T * HIDDEN_SIZE],
        [2usize, T, HIDDEN_SIZE],
    )
    .expect("B=2 hidden");
    let err = attn
        .forward(&hidden, None, &offset, &mut cache)
        .expect_err("should reject batch > 1");
    assert!(
        matches!(err, Error::InvalidArgument(_)),
        "expected InvalidArgument, got {err:?}"
    );
}

#[test]
fn attention_forward_rejects_wrong_hidden_size() {
    let device = Device::shared().expect("device");
    let (attn, offset, mut cache) = attention_fixture();
    // Correct rank-3, B=1, but wrong hidden size (512 != 896).
    let hidden = Tensor::from_slice(&device, &vec![bf16::ZERO; T * 512], [1usize, T, 512])
        .expect("wrong-hidden hidden");
    let err = attn
        .forward(&hidden, None, &offset, &mut cache)
        .expect_err("should reject wrong hidden size");
    assert!(
        matches!(err, Error::InvalidArgument(_)),
        "expected InvalidArgument, got {err:?}"
    );
}

#[test]
fn mlp_forward_shape_dtype_and_lazy() {
    let device = Device::shared().expect("device");

    let gate = Linear::new(zero_qw(&device, INTERMEDIATE, HIDDEN_SIZE), None).expect("gate");
    let up = Linear::new(zero_qw(&device, INTERMEDIATE, HIDDEN_SIZE), None).expect("up");
    let down = Linear::new(zero_qw(&device, HIDDEN_SIZE, INTERMEDIATE), None).expect("down");
    let mlp = Mlp::new(gate, up, down).expect("mlp");

    let x = Tensor::from_slice(
        &device,
        &vec![bf16::ZERO; T * HIDDEN_SIZE],
        [1usize, T, HIDDEN_SIZE],
    )
    .expect("hidden");

    let out = mlp.forward(&x).expect("mlp forward");
    assert_eq!(out.shape().dims(), &[1, T, HIDDEN_SIZE], "mlp out shape");
    assert_eq!(out.dtype(), DType::BF16, "mlp out dtype");
    assert!(!out.is_materialized(), "mlp out should be lazy");
}

#[test]
fn mlp_new_rejects_biased_linear() {
    let device = Device::shared().expect("device");

    let gate_qw = zero_qw(&device, INTERMEDIATE, HIDDEN_SIZE);
    let up_qw = zero_qw(&device, INTERMEDIATE, HIDDEN_SIZE);
    let down_qw = zero_qw(&device, HIDDEN_SIZE, INTERMEDIATE);

    let biased_gate = Linear::new(
        gate_qw.clone(),
        Some(zero_bf16_1d(&device, INTERMEDIATE)),
    )
    .expect("biased gate Linear");
    let up = Linear::new(up_qw.clone(), None).expect("up");
    let down = Linear::new(down_qw.clone(), None).expect("down");
    assert!(
        Mlp::new(biased_gate, up, down).is_err(),
        "Mlp::new must reject a biased gate projection",
    );
}
