//! Qwen3.5 / Qwen3.6 decoder layer: pre-norm mixer + residual, pre-norm SwiGLU
//! MLP + residual. The layer generalizes over both mixers — the full
//! Gated-Attention mixer ([`GatedAttention`]) and the linear Gated-DeltaNet
//! mixer ([`GatedDeltaNet`]) — selected per layer by the loaded weights.
//!
//! The two mixers carry different state needs: the full mixer threads a RoPE
//! `offset` and a [`KvCache`]; the linear (GDN) mixer (Phase 3) prefills from a
//! zero recurrent state and needs neither. The layer therefore exposes two
//! entry points — [`Qwen35DecoderLayer::forward`] (full attention, needs the
//! cache) and [`Qwen35DecoderLayer::forward_linear`] (GDN prefill, no cache) —
//! each erroring if the layer's mixer is the other variant. Phase 4 unifies
//! cache handling across the full model (a per-layer cache enum threaded
//! through one `forward`); until then the caller dispatches on layer type.

use cider_press_runtime::{KvCache, Tensor};

use crate::error::{Error, Result};
use crate::nn::{Linear, Mlp, Module, RmsNormLayer};
use crate::qwen3_5::attention::GatedAttention;
use crate::qwen3_5::config::Qwen35Config;
use crate::qwen3_5::gated_deltanet::GatedDeltaNet;
use crate::qwen3_5::weights::{Qwen35LayerWeights, Qwen35MixerWeights};

/// The token-mixing sublayer: either the full Gated-Attention mixer or the
/// linear Gated-DeltaNet mixer, selected per layer by the checkpoint.
#[derive(Debug)]
enum Mixer {
    Full(GatedAttention),
    Linear(GatedDeltaNet),
}

/// One decoder layer of Qwen3.5/3.6 — generic over the mixer variant.
#[derive(Debug)]
pub struct Qwen35DecoderLayer {
    input_layernorm: RmsNormLayer,
    mixer: Mixer,
    post_attention_layernorm: RmsNormLayer,
    mlp: Mlp,
}

impl Qwen35DecoderLayer {
    /// Build from a layer's weights, selecting the mixer variant from
    /// `weights.mixer`: a full-attention layer builds [`GatedAttention`]; a
    /// linear layer builds [`GatedDeltaNet`].
    pub fn from_weights(weights: &Qwen35LayerWeights, config: &Qwen35Config) -> Result<Self> {
        // HF config.json emits eps as JSON numeric (f64); the value (~1e-6) is exactly representable in f32.
        #[allow(clippy::cast_possible_truncation)]
        let eps = config.rms_norm_eps as f32;
        let mixer = match &weights.mixer {
            Qwen35MixerWeights::FullAttn(fa) => {
                Mixer::Full(GatedAttention::from_weights(fa, config)?)
            }
            Qwen35MixerWeights::LinearAttn(la) => {
                Mixer::Linear(GatedDeltaNet::from_weights(la, config)?)
            }
        };
        let input_layernorm = RmsNormLayer::new(weights.input_layernorm.clone(), eps);
        let post_attention_layernorm =
            RmsNormLayer::new(weights.post_attention_layernorm.clone(), eps);
        let mlp = Mlp::new(
            Linear::new(weights.mlp.gate_proj.clone(), None)?,
            Linear::new(weights.mlp.up_proj.clone(), None)?,
            Linear::new(weights.mlp.down_proj.clone(), None)?,
        )?;
        Ok(Self {
            input_layernorm,
            mixer,
            post_attention_layernorm,
            mlp,
        })
    }

    /// Forward for a FULL-attention layer: pre-norm gated attention + residual,
    /// pre-norm SwiGLU MLP + residual.
    ///
    /// - `hidden`: `[1, T, hidden_size]` dense BF16.
    /// - `offset`, `cache`: forwarded to [`GatedAttention::forward`].
    ///
    /// Returns `[1, T, hidden_size]`, lazy. Errors if the layer's mixer is the
    /// linear (Gated-DeltaNet) variant — call [`Self::forward_linear`] for that.
    pub fn forward(&self, hidden: &Tensor, offset: &Tensor, cache: &mut KvCache) -> Result<Tensor> {
        let Mixer::Full(attn) = &self.mixer else {
            return Err(Error::InvalidArgument(
                "Qwen35DecoderLayer::forward: this is a linear (Gated-DeltaNet) layer; call \
                 forward_linear"
                    .into(),
            ));
        };
        let normed = self.input_layernorm.forward(hidden)?;
        let mixer_out = attn.forward(&normed, offset, cache)?;
        self.mlp_residual(hidden, &mixer_out)
    }

    /// Forward for a LINEAR (Gated-DeltaNet) layer: pre-norm GDN mixer +
    /// residual, pre-norm SwiGLU MLP + residual. Prefills from a zero recurrent
    /// state (no KV cache, no RoPE offset).
    ///
    /// - `hidden`: `[1, T, hidden_size]` dense BF16.
    ///
    /// Returns `[1, T, hidden_size]`, lazy. Errors if the layer's mixer is the
    /// full-attention variant — call [`Self::forward`] for that.
    pub fn forward_linear(&self, hidden: &Tensor) -> Result<Tensor> {
        let Mixer::Linear(gdn) = &self.mixer else {
            return Err(Error::InvalidArgument(
                "Qwen35DecoderLayer::forward_linear: this is a full-attention layer; call forward"
                    .into(),
            ));
        };
        let normed = self.input_layernorm.forward(hidden)?;
        let mixer_out = gdn.forward(&normed)?;
        self.mlp_residual(hidden, &mixer_out)
    }

    /// Shared tail: mixer residual, then pre-norm SwiGLU MLP + residual.
    fn mlp_residual(&self, hidden: &Tensor, mixer_out: &Tensor) -> Result<Tensor> {
        let after_mixer = hidden.add(mixer_out)?;
        let mlp_input = self.post_attention_layernorm.forward(&after_mixer)?;
        let mlp_out = self.mlp.forward(&mlp_input)?;
        Ok(after_mixer.add(&mlp_out)?)
    }
}
