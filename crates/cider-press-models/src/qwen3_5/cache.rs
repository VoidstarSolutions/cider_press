//! Per-layer cache for the Qwen3.5/3.6 model: a KV cache for full-attention
//! layers, a two-part (conv-window + recurrent-state) cache for GDN layers.
//!
//! The model alternates 3 linear (Gated-DeltaNet) : 1 full-attention layer.
//! Full layers carry a standard [`KvCache`]; linear layers carry a
//! [`LinearState`] — a depthwise-conv sliding window plus the fp32 delta-rule
//! recurrent state. Each [`Qwen35DecoderLayer`](crate::qwen3_5::Qwen35DecoderLayer)
//! builds its own cache, so the variant matches the layer's mixer.

use cider_press_runtime::{DType, Device, KvCache, Tensor};

use crate::error::{Error, Result};
use crate::qwen3_5::config::Qwen35Config;

/// Gated-DeltaNet two-part state: the conv sliding window + the fp32 recurrent
/// state.
#[derive(Debug)]
pub struct LinearState {
    pub(crate) conv_state: Tensor,
    pub(crate) recurrent_state: Tensor,
}

impl LinearState {
    /// Allocate a zeroed Gated-DeltaNet state for one linear layer — the start
    /// state of a fresh sequence.
    ///
    /// # Errors
    /// Device allocation failure.
    pub(crate) fn zeros(config: &Qwen35Config, device: &Device) -> Result<Self> {
        let conv_window = config.linear_conv_kernel_dim - 1;
        let hv = config.linear_num_value_heads;
        let dv = config.linear_value_head_dim;
        let dk = config.linear_key_head_dim;
        Ok(Self {
            conv_state: Tensor::zeros(device, [1, conv_window, config.conv_dim()], DType::BF16)?,
            recurrent_state: Tensor::zeros(device, [1, hv, dv, dk], DType::F32)?,
        })
    }

    /// `[1, linear_conv_kernel_dim - 1, conv_dim]` bf16 — the trailing pre-conv
    /// `q‖k‖v` rows that the next step's depthwise causal conv reads.
    #[must_use]
    pub fn conv_state(&self) -> &Tensor {
        &self.conv_state
    }

    /// `[1, Hv, Dv, Dk]` f32 — the delta-rule recurrent state, held in fp32 for
    /// numerical stability of the recurrence.
    #[must_use]
    pub fn recurrent_state(&self) -> &Tensor {
        &self.recurrent_state
    }

    fn rezero(&mut self) -> Result<()> {
        let device = self.conv_state.device().ok_or_else(|| {
            Error::InvalidArgument("LinearState::rezero: conv_state has no device".into())
        })?;
        let conv_state =
            Tensor::zeros(device, self.conv_state.shape().dims().to_vec(), DType::BF16)?;
        let recurrent_state = Tensor::zeros(
            device,
            self.recurrent_state.shape().dims().to_vec(),
            DType::F32,
        )?;
        self.conv_state = conv_state;
        self.recurrent_state = recurrent_state;
        Ok(())
    }
}

/// Per-layer cache: [`Full`](Self::Full) for a full-attention layer,
/// [`Linear`](Self::Linear) for a Gated-DeltaNet layer.
#[derive(Debug)]
pub enum Qwen35LayerCache {
    /// Full-attention KV cache.
    Full(KvCache),
    /// Gated-DeltaNet conv-window + recurrent state.
    Linear(LinearState),
}

impl Qwen35LayerCache {
    /// Reset to the initial state for a fresh sequence: [`Full`](Self::Full)
    /// rewinds the [`KvCache`] to position 0; [`Linear`](Self::Linear) re-zeros
    /// both state tensors (same shapes, dtypes, and device).
    ///
    /// # Errors
    /// A [`KvCache`] reset failure (an un-committed prior update), or device
    /// allocation failure.
    pub fn reset(&mut self) -> Result<()> {
        match self {
            Self::Full(kv) => Ok(kv.reset()?),
            Self::Linear(state) => state.rezero(),
        }
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
    fn linear_state_zeros_shapes_and_dtypes() {
        let cfg = Qwen35Config::from_json_bytes(CONFIG_JSON.as_bytes()).unwrap();
        let device = Device::shared().unwrap();

        // conv_dim = 2 * (2 * 4) + (2 * 4) = 24.
        assert_eq!(cfg.conv_dim(), 24);

        let state = LinearState::zeros(&cfg, &device).unwrap();
        assert_eq!(state.conv_state().shape().dims(), &[1, 3, 24]);
        assert_eq!(state.conv_state().dtype(), DType::BF16);
        assert_eq!(state.recurrent_state().shape().dims(), &[1, 2, 4, 4]);
        assert_eq!(state.recurrent_state().dtype(), DType::F32);
    }

    #[test]
    fn reset_rezeros_linear_state() {
        let cfg = Qwen35Config::from_json_bytes(CONFIG_JSON.as_bytes()).unwrap();
        let device = Device::shared().unwrap();
        let mut state = LinearState::zeros(&cfg, &device).unwrap();
        state.conv_state =
            Tensor::from_slice(&device, &[half::bf16::ONE; 3 * 24], [1usize, 3, 24]).unwrap();
        state.recurrent_state =
            Tensor::from_slice(&device, &[1f32; 2 * 4 * 4], [1usize, 2, 4, 4]).unwrap();
        let mut cache = Qwen35LayerCache::Linear(state);

        cache.reset().unwrap();

        let Qwen35LayerCache::Linear(state) = cache else {
            unreachable!()
        };
        assert_eq!(state.conv_state().shape().dims(), &[1, 3, 24]);
        assert_eq!(state.conv_state().dtype(), DType::BF16);
        assert_eq!(state.recurrent_state().shape().dims(), &[1, 2, 4, 4]);
        assert_eq!(state.recurrent_state().dtype(), DType::F32);
        state.conv_state().eval().unwrap();
        state.recurrent_state().eval().unwrap();
        let conv: Vec<half::bf16> = state.conv_state().cpu_to_vec().unwrap();
        let rec: Vec<f32> = state.recurrent_state().cpu_to_vec().unwrap();
        assert!(conv.iter().all(|v| v.to_f32() == 0.0));
        assert!(rec.iter().all(|&v| v == 0.0));
    }
}
