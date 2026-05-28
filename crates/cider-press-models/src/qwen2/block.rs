//! Qwen2 transformer block: pre-norm + attention + residual,
//! pre-norm + MLP + residual.

use cider_press_runtime::{KvCache, Tensor};

use super::attention::Attention;
use super::config::Qwen2Config;
use super::weights::Qwen2LayerWeights;
use crate::error::Result;
use crate::nn::{Linear, Mlp, Module, RmsNormLayer};

/// One layer of Qwen2: pre-norm self-attention with a residual, then
/// pre-norm `SwiGLU` MLP with a residual.
///
/// `forward` mirrors [`Attention::forward`]'s signature
/// (`hidden, mask, offset, cache`) — auxiliary state flows through
/// the signature, so this type does not implement [`crate::nn::Module`].
#[derive(Debug)]
pub struct TransformerBlock {
    input_layernorm: RmsNormLayer,
    attention: Attention,
    post_attention_layernorm: RmsNormLayer,
    mlp: Mlp,
}

impl TransformerBlock {
    /// Build a block from a layer's loaded weights and the model
    /// config. Clones the weights (cheap refcount bumps on shared
    /// device buffers) into owned `Linear`s.
    pub fn from_weights(weights: &Qwen2LayerWeights, config: &Qwen2Config) -> Result<Self> {
        // HF config.json emits eps as JSON numeric (f64); the value (~1e-6) is exactly representable in f32.
        #[allow(clippy::cast_possible_truncation)]
        let eps = config.rms_norm_eps as f32;
        let input_layernorm = RmsNormLayer::new(weights.input_layernorm.clone(), eps);
        let post_attention_layernorm =
            RmsNormLayer::new(weights.post_attention_layernorm.clone(), eps);
        let attention = Attention::from_weights(&weights.attention, config)?;
        let mlp = Mlp::new(
            Linear::new(weights.mlp.gate_proj.clone(), None)?,
            Linear::new(weights.mlp.up_proj.clone(), None)?,
            Linear::new(weights.mlp.down_proj.clone(), None)?,
        )?;
        Ok(Self {
            input_layernorm,
            attention,
            post_attention_layernorm,
            mlp,
        })
    }

    /// Forward through the block.
    ///
    /// - `hidden`: `[1, T, hidden_size]` dense BF16.
    /// - `mask`, `offset`, `cache`: forwarded to [`Attention::forward`].
    ///
    /// Returns `[1, T, hidden_size]`, lazy.
    pub fn forward(
        &self,
        hidden: &Tensor,
        mask: Option<&Tensor>,
        offset: &Tensor,
        cache: &mut KvCache,
    ) -> Result<Tensor> {
        let normed = self.input_layernorm.forward(hidden)?;
        let attn_out = self.attention.forward(&normed, mask, offset, cache)?;
        let after_attn = hidden.add(&attn_out)?;

        let mlp_input = self.post_attention_layernorm.forward(&after_attn)?;
        let mlp_out = self.mlp.forward(&mlp_input)?;
        Ok(after_attn.add(&mlp_out)?)
    }
}
