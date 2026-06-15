//! Qwen3.5 / Qwen3.6 decoder layer (full-attention variant): pre-norm gated
//! attention + residual, pre-norm SwiGLU MLP + residual. The linear (GDN)
//! variant lands in Phase 3.

use cider_press_runtime::{KvCache, Tensor};

use crate::error::{Error, Result};
use crate::nn::{Linear, Mlp, Module, RmsNormLayer};
use crate::qwen3_5::attention::GatedAttention;
use crate::qwen3_5::config::Qwen35Config;
use crate::qwen3_5::weights::{Qwen35LayerWeights, Qwen35MixerWeights};

/// One full-attention decoder layer of Qwen3.5/3.6.
#[derive(Debug)]
pub struct Qwen35DecoderLayer {
    input_layernorm: RmsNormLayer,
    attn: GatedAttention,
    post_attention_layernorm: RmsNormLayer,
    mlp: Mlp,
}

impl Qwen35DecoderLayer {
    /// Build from a layer's weights. Only accepts a FULL-attention layer;
    /// the linear (Gated-DeltaNet) mixer is Phase 3.
    pub fn from_weights(weights: &Qwen35LayerWeights, config: &Qwen35Config) -> Result<Self> {
        // HF config.json emits eps as JSON numeric (f64); the value (~1e-6) is exactly representable in f32.
        #[allow(clippy::cast_possible_truncation)]
        let eps = config.rms_norm_eps as f32;
        let Qwen35MixerWeights::FullAttn(fa) = &weights.mixer else {
            return Err(Error::InvalidArgument(
                "Qwen35DecoderLayer::from_weights: expected a full-attention layer (the linear \
                 Gated-DeltaNet mixer is Phase 3)"
                    .into(),
            ));
        };
        let input_layernorm = RmsNormLayer::new(weights.input_layernorm.clone(), eps);
        let attn = GatedAttention::from_weights(fa, config)?;
        let post_attention_layernorm =
            RmsNormLayer::new(weights.post_attention_layernorm.clone(), eps);
        let mlp = Mlp::new(
            Linear::new(weights.mlp.gate_proj.clone(), None)?,
            Linear::new(weights.mlp.up_proj.clone(), None)?,
            Linear::new(weights.mlp.down_proj.clone(), None)?,
        )?;
        Ok(Self {
            input_layernorm,
            attn,
            post_attention_layernorm,
            mlp,
        })
    }

    /// Forward: pre-norm gated attention + residual, pre-norm SwiGLU MLP + residual.
    ///
    /// - `hidden`: `[1, T, hidden_size]` dense BF16.
    /// - `offset`, `cache`: forwarded to [`GatedAttention::forward`].
    ///
    /// Returns `[1, T, hidden_size]`, lazy.
    pub fn forward(&self, hidden: &Tensor, offset: &Tensor, cache: &mut KvCache) -> Result<Tensor> {
        let normed = self.input_layernorm.forward(hidden)?;
        let attn_out = self.attn.forward(&normed, offset, cache)?;
        let after_attn = hidden.add(&attn_out)?;
        let mlp_input = self.post_attention_layernorm.forward(&after_attn)?;
        let mlp_out = self.mlp.forward(&mlp_input)?;
        Ok(after_attn.add(&mlp_out)?)
    }
}
