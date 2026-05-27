//! Smoke tests for [`qwen2::attention::project_qkv`] and
//! [`qwen2::attention::project_output`].
//!
//! Verifies that the helpers compile, construct the correct output
//! shapes through the lazy graph, and reject invalid inputs — without
//! calling `eval()`. Full numerical parity against MLX is covered by
//! Task 4's gated integration test (real checkpoint +
//! `mx.fast.scaled_dot_product_attention`).

#![cfg(target_os = "macos")]

use cider_press_models::qwen2::{Qwen2AttentionWeights, Qwen2Config, attention};
use cider_press_runtime::{DType, Device, Quantization, QuantizedWeight, Tensor};
use half::bf16;

// Qwen2.5-0.5B-Instruct dimensions.
const HIDDEN_SIZE: usize = 896;
const NUM_Q_HEADS: usize = 14;
const NUM_KV_HEADS: usize = 2;
const HEAD_DIM: usize = 64; // hidden_size / num_attention_heads
const T: usize = 4; // small sequence length for smoke tests

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

/// Build a zero-filled `QuantizedWeight` with logical `[n, k]` shape
/// at `group_size=64`, `bits=4`.
fn zero_qw(device: &Device, n: usize, k: usize) -> QuantizedWeight {
    let elem_count = n * k;
    // packed u32 words: elem_count * bits / 32 words × 4 bytes/word
    let w_bytes = elem_count * 4 / 32 * 4;
    // groups × 2 bytes (bf16) for scales and biases
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

/// Build a zero-filled dense BF16 tensor of the given 1D length.
fn zero_bf16_1d(device: &Device, len: usize) -> Tensor {
    Tensor::from_slice(device, &vec![bf16::ZERO; len], [len]).expect("zero bf16 tensor")
}

fn synthetic_weights(device: &Device) -> Qwen2AttentionWeights {
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
fn project_qkv_output_shapes_dtype_and_lazy() {
    let device = Device::shared().expect("device");
    let config = synthetic_config();
    let weights = synthetic_weights(&device);

    let hidden = Tensor::from_slice(
        &device,
        &vec![bf16::ZERO; T * HIDDEN_SIZE],
        [1usize, T, HIDDEN_SIZE],
    )
    .expect("hidden tensor");

    let (q, k, v) =
        attention::project_qkv(&hidden, &weights, &config).expect("project_qkv should succeed");

    // Output shapes: [B, H, T, D_h]
    assert_eq!(q.shape().dims(), &[1, NUM_Q_HEADS, T, HEAD_DIM], "q shape");
    assert_eq!(k.shape().dims(), &[1, NUM_KV_HEADS, T, HEAD_DIM], "k shape");
    assert_eq!(v.shape().dims(), &[1, NUM_KV_HEADS, T, HEAD_DIM], "v shape");

    // Dtypes are BF16.
    assert_eq!(q.dtype(), DType::BF16, "q dtype");
    assert_eq!(k.dtype(), DType::BF16, "k dtype");
    assert_eq!(v.dtype(), DType::BF16, "v dtype");

    // Outputs are lazy: the graph was constructed without calling eval().
    assert!(!q.is_materialized(), "q should be lazy");
    assert!(!k.is_materialized(), "k should be lazy");
    assert!(!v.is_materialized(), "v should be lazy");
}

#[test]
fn project_output_shape_dtype_and_lazy() {
    let device = Device::shared().expect("device");
    let config = synthetic_config();
    let weights = synthetic_weights(&device);

    // attn_out: [B, H_q, T, D_h]
    let attn_out = Tensor::from_slice(
        &device,
        &vec![bf16::ZERO; NUM_Q_HEADS * T * HEAD_DIM],
        [1usize, NUM_Q_HEADS, T, HEAD_DIM],
    )
    .expect("attn_out tensor");

    let out = attention::project_output(&attn_out, &weights, &config)
        .expect("project_output should succeed");

    // Output shape: [B, T, hidden_size]
    assert_eq!(out.shape().dims(), &[1, T, HIDDEN_SIZE], "out shape");
    assert_eq!(out.dtype(), DType::BF16, "out dtype");

    // Output is lazy.
    assert!(!out.is_materialized(), "out should be lazy");
}

#[test]
fn project_qkv_rejects_rank2_hidden() {
    let device = Device::shared().expect("device");
    let config = synthetic_config();
    let weights = synthetic_weights(&device);

    // Rank-2 input: missing the batch dimension.
    let hidden = Tensor::from_slice(
        &device,
        &vec![bf16::ZERO; T * HIDDEN_SIZE],
        [T, HIDDEN_SIZE],
    )
    .expect("rank-2 hidden");

    let err = attention::project_qkv(&hidden, &weights, &config)
        .expect_err("should reject rank-2 hidden");
    assert!(
        matches!(err, cider_press_models::Error::InvalidArgument(_)),
        "expected InvalidArgument, got {err:?}"
    );
}

#[test]
fn project_qkv_rejects_wrong_last_dim() {
    let device = Device::shared().expect("device");
    let config = synthetic_config();
    let weights = synthetic_weights(&device);

    // Correct rank-3, but wrong hidden_size (512 instead of 896).
    let hidden = Tensor::from_slice(&device, &vec![bf16::ZERO; T * 512], [1usize, T, 512])
        .expect("rank-3 wrong-dim hidden");

    let err = attention::project_qkv(&hidden, &weights, &config)
        .expect_err("should reject wrong last dim");
    assert!(
        matches!(err, cider_press_models::Error::InvalidArgument(_)),
        "expected InvalidArgument, got {err:?}"
    );
}

#[test]
fn project_output_rejects_rank3_input() {
    let device = Device::shared().expect("device");
    let config = synthetic_config();
    let weights = synthetic_weights(&device);

    // Rank-3 instead of rank-4.
    let bad = Tensor::from_slice(
        &device,
        &vec![bf16::ZERO; T * (NUM_Q_HEADS * HEAD_DIM)],
        [1usize, T, NUM_Q_HEADS * HEAD_DIM],
    )
    .expect("rank-3 attn_out");

    let err =
        attention::project_output(&bad, &weights, &config).expect_err("should reject rank-3 input");
    assert!(
        matches!(err, cider_press_models::Error::InvalidArgument(_)),
        "expected InvalidArgument, got {err:?}"
    );
}
