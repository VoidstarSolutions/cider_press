//! Per-layer cache for the Qwen3.5/3.6 model: a KV cache for full-attention
//! layers, a two-part (conv-window + recurrent-state) cache for GDN layers.
//!
//! The model alternates 3 linear (Gated-DeltaNet) : 1 full-attention layer,
//! selected by [`Qwen35Config::is_linear`]. Full layers carry a standard
//! [`KvCache`]; linear layers carry a [`LinearState`] — a depthwise-conv
//! sliding window plus the fp32 delta-rule recurrent state.

use cider_press_runtime::{DType, Device, KvCache, Tensor};

use crate::error::Result;
use crate::qwen3_5::config::Qwen35Config;

/// Gated-DeltaNet two-part state: the conv sliding window + the fp32 recurrent
/// state.
pub struct LinearState {
    /// `[1, linear_conv_kernel_dim - 1, conv_dim]` bf16 — the trailing pre-conv
    /// `q‖k‖v` rows that the next step's depthwise causal conv reads.
    pub conv_state: Tensor,
    /// `[1, Hv, Dv, Dk]` f32 — the delta-rule recurrent state. Held in fp32 for
    /// numerical stability of the recurrence.
    pub recurrent_state: Tensor,
}

/// Per-layer cache, variant chosen by [`Qwen35Config::is_linear`].
pub enum Qwen35LayerCache {
    /// Full-attention KV cache.
    Full(KvCache),
    /// Gated-DeltaNet conv-window + recurrent state.
    Linear(LinearState),
}

impl Qwen35LayerCache {
    /// Build the cache for layer `i` ([`Linear`](Self::Linear) or
    /// [`Full`](Self::Full) per [`Qwen35Config::is_linear`]).
    pub fn for_layer(
        config: &Qwen35Config,
        i: usize,
        device: &Device,
        max_tokens: usize,
    ) -> Result<Self> {
        if config.is_linear(i) {
            Ok(Self::Linear(LinearState::new(config, device)?))
        } else {
            Ok(Self::Full(KvCache::new(
                device,
                max_tokens,
                config.num_key_value_heads,
                config.head_dim,
                DType::BF16,
            )?))
        }
    }

    /// Reset to the initial state for a fresh sequence: [`Full`](Self::Full)
    /// rewinds the [`KvCache`] to position 0; [`Linear`](Self::Linear) re-zeros
    /// both state tensors. `device` is the allocation device for the re-zeroed
    /// linear tensors (the conv/recurrent shapes are read back from the
    /// existing tensors).
    pub fn reset(&mut self, device: &Device) -> Result<()> {
        match self {
            Self::Full(kv) => Ok(kv.reset()?),
            Self::Linear(state) => {
                state.conv_state = Tensor::zeros(
                    device,
                    state.conv_state.shape().dims().to_vec(),
                    DType::BF16,
                )?;
                state.recurrent_state = Tensor::zeros(
                    device,
                    state.recurrent_state.shape().dims().to_vec(),
                    DType::F32,
                )?;
                Ok(())
            }
        }
    }
}

impl LinearState {
    /// Allocate a zeroed Gated-DeltaNet state for one linear layer.
    fn new(config: &Qwen35Config, device: &Device) -> Result<Self> {
        let conv_window = config.linear_conv_kernel_dim - 1;
        let hv = config.linear_num_value_heads;
        let dv = config.linear_value_head_dim;
        let dk = config.linear_key_head_dim;
        Ok(Self {
            conv_state: Tensor::zeros(device, [1, conv_window, config.conv_dim()], DType::BF16)?,
            recurrent_state: Tensor::zeros(device, [1, hv, dv, dk], DType::F32)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tiny structural config: 4 layers, 3 linear : 1 full, dims small but
    /// valid. `conv_dim = 2 * key_dim + value_dim = 2 * 8 + 8 = 24`.
    const CONFIG_JSON: &str = r#"{
        "model_type": "qwen3_5",
        "text_config": {
            "hidden_size": 32,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "intermediate_size": 64,
            "vocab_size": 256,
            "rms_norm_eps": 1e-06,
            "full_attention_interval": 4,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 4,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 2,
            "linear_value_head_dim": 4,
            "tie_word_embeddings": true,
            "layer_types": [
                "linear_attention", "linear_attention",
                "linear_attention", "full_attention"
            ],
            "rope_parameters": {
                "rope_theta": 10000000,
                "partial_rotary_factor": 0.25,
                "mrope_section": [11, 11, 10]
            }
        }
    }"#;

    #[test]
    fn for_layer_picks_variant_and_shapes() {
        let cfg = Qwen35Config::from_json_bytes(CONFIG_JSON.as_bytes()).unwrap();
        let device = Device::shared().unwrap();

        // conv_dim = 2 * (2 * 4) + (2 * 4) = 24.
        assert_eq!(cfg.conv_dim(), 24);

        for i in 0..3 {
            let cache = Qwen35LayerCache::for_layer(&cfg, i, &device, 8).unwrap();
            match cache {
                Qwen35LayerCache::Linear(state) => {
                    assert_eq!(state.conv_state.shape().dims(), &[1, 3, 24]);
                    assert_eq!(state.conv_state.dtype(), DType::BF16);
                    assert_eq!(state.recurrent_state.shape().dims(), &[1, 2, 4, 4]);
                    assert_eq!(state.recurrent_state.dtype(), DType::F32);
                }
                Qwen35LayerCache::Full(_) => panic!("layer {i} should be Linear"),
            }
        }

        let full = Qwen35LayerCache::for_layer(&cfg, 3, &device, 8).unwrap();
        match full {
            Qwen35LayerCache::Full(kv) => {
                assert_eq!(kv.head_dim(), cfg.head_dim);
                assert_eq!(kv.dtype(), DType::BF16);
            }
            Qwen35LayerCache::Linear(_) => panic!("layer 3 should be Full"),
        }
    }

    #[test]
    fn reset_rezeros_linear_state() {
        let cfg = Qwen35Config::from_json_bytes(CONFIG_JSON.as_bytes()).unwrap();
        let device = Device::shared().unwrap();
        let mut cache = Qwen35LayerCache::for_layer(&cfg, 0, &device, 8).unwrap();
        cache.reset(&device).unwrap();
        match cache {
            Qwen35LayerCache::Linear(state) => {
                assert_eq!(state.conv_state.shape().dims(), &[1, 3, 24]);
                assert_eq!(state.recurrent_state.shape().dims(), &[1, 2, 4, 4]);
            }
            Qwen35LayerCache::Full(_) => unreachable!(),
        }
    }
}
