//! Qwen3.5 / Qwen3.6 Gated-Attention (the 1-in-4 full-attention mixer).
//!
//! `head_dim` 256, GQA 16 q / 4 kv, per-head query‖gate split, weighted QK-norm
//! (`RMSNorm` over `head_dim`, before RoPE), partial rotary (64 of 256 dims),
//! sigmoid output gate. Phase-2 lands the helpers + mixer; see
//! `docs/inference/models/qwen3.6.md`.

use cider_press_runtime::Tensor;

use crate::error::Result;
use crate::qwen3_5::config::Qwen35Config;

/// Rotary dims for partial RoPE: `floor(head_dim * partial_rotary_factor)`.
/// For the 4B/27B config that is `256 * 0.25 = 64`.
// Consumed by the `GatedAttention` forward in Task 4; only the tests exercise
// it today.
#[allow(dead_code)]
pub(crate) fn rotary_dims(config: &Qwen35Config) -> usize {
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    let dims = (config.head_dim as f64 * config.rope.partial_rotary_factor) as usize;
    dims
}

/// Partial RoPE over Q or K shaped `[1, H, T, head_dim]`: rotates the leading
/// `rotary_dims` (64 of 256) and passes the rest through. `base = rope_theta`,
/// `scale = 1`, non-traditional (`NeoX`). `offset` is a length-1 I32 tensor.
// Consumed by the `GatedAttention` forward in Task 4; only the tests exercise
// it today.
#[allow(dead_code)]
pub(crate) fn partial_rope(x: &Tensor, offset: &Tensor, config: &Qwen35Config) -> Result<Tensor> {
    #[allow(clippy::cast_possible_truncation)]
    let base = config.rope.rope_theta as f32;
    Ok(x.rope(offset, base, 1.0, rotary_dims(config))?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cider_press_runtime::{Device, Tensor};
    use half::bf16;

    use crate::qwen3_5::config::Qwen35Config;

    const CONFIG_JSON: &str = r#"{
        "model_type": "qwen3_5",
        "quantization": { "group_size": 64, "bits": 4, "mode": "affine" },
        "text_config": {
            "hidden_size": 2560, "num_hidden_layers": 4, "num_attention_heads": 16,
            "num_key_value_heads": 4, "head_dim": 256, "intermediate_size": 9216,
            "vocab_size": 248320, "rms_norm_eps": 1e-06, "attention_bias": false,
            "attn_output_gate": true, "full_attention_interval": 4,
            "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128,
            "linear_num_key_heads": 16, "linear_num_value_heads": 32,
            "linear_value_head_dim": 128, "tie_word_embeddings": true,
            "layer_types": ["linear_attention","linear_attention","linear_attention","full_attention"],
            "rope_parameters": { "rope_theta": 10000000, "partial_rotary_factor": 0.25, "mrope_section": [11,11,10] }
        }
    }"#;

    #[test]
    fn rotary_dims_is_quarter_head_dim() {
        let cfg = Qwen35Config::from_json_bytes(CONFIG_JSON.as_bytes()).unwrap();
        assert_eq!(rotary_dims(&cfg), 64); // 256 * 0.25
    }

    #[test]
    fn partial_rope_preserves_shape() {
        let cfg = Qwen35Config::from_json_bytes(CONFIG_JSON.as_bytes()).unwrap();
        let device = Device::shared().unwrap();
        // [1, H=16, T=3, head_dim=256] of zeros is fine for a shape check.
        let n = 16 * 3 * 256;
        let data = vec![bf16::ZERO; n];
        let x = Tensor::from_slice(&device, &data, [1usize, 16, 3, 256]).unwrap();
        let offset = Tensor::from_slice(&device, &[0i32], [1usize]).unwrap();
        let y = partial_rope(&x, &offset, &cfg).unwrap();
        y.eval().unwrap();
        assert_eq!(y.shape().dims(), &[1, 16, 3, 256]);
    }
}
