//! Qwen3.5 / Qwen3.6 decoder layer: pre-norm mixer + residual, pre-norm SwiGLU
//! MLP + residual. The layer generalizes over both mixers — the full
//! Gated-Attention mixer ([`GatedAttention`]) and the linear Gated-DeltaNet
//! mixer ([`GatedDeltaNet`]) — selected per layer by the loaded weights.
//!
//! The two mixers carry different state needs: the full mixer threads a RoPE
//! `offset` and a [`KvCache`](cider_press_runtime::KvCache); the linear (GDN)
//! mixer threads a depthwise-conv window plus an fp32 recurrent state. One
//! [`Qwen35DecoderLayer::forward`] serves both, dispatching on a per-layer
//! [`Qwen35LayerCache`] enum built by [`Qwen35DecoderLayer::make_cache`], so
//! the cache variant matches the mixer by construction (a hand-built
//! mismatched cache errors). The `offset` is threaded for the full path's
//! RoPE; the linear (GDN) mixer carries no position and ignores it (matching
//! mlx-lm, where the GDN layer takes no offset).

use cider_press_runtime::{Device, Tensor};

use crate::error::{Error, Result};
use crate::nn::{Linear, Mlp, Module, RmsNormLayer};
use crate::qwen3_5::attention::GatedAttention;
use crate::qwen3_5::cache::Qwen35LayerCache;
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

    /// A fresh cache for this layer's mixer: a [`KvCache`](cider_press_runtime::KvCache)
    /// sized for `max_tokens` positions for full attention, a zeroed
    /// [`LinearState`](crate::qwen3_5::LinearState) for Gated-DeltaNet.
    ///
    /// # Errors
    /// Device allocation failure.
    pub fn make_cache(&self, max_tokens: usize, device: &Device) -> Result<Qwen35LayerCache> {
        Ok(match &self.mixer {
            Mixer::Full(attn) => Qwen35LayerCache::Full(attn.make_cache(max_tokens, device)?),
            Mixer::Linear(gdn) => Qwen35LayerCache::Linear(gdn.make_cache(device)?),
        })
    }

    /// Forward over either mixer, dispatching on the per-layer cache: pre-norm
    /// mixer + residual, pre-norm SwiGLU MLP + residual.
    ///
    /// - `hidden`: `[1, T, hidden_size]` dense BF16.
    /// - `offset`: RoPE offset, forwarded to [`GatedAttention::forward`] on the
    ///   full path. The linear (Gated-DeltaNet) mixer carries no position and
    ///   ignores it (matching mlx-lm, where the GDN layer takes no offset).
    /// - `cache`: this layer's [`Qwen35LayerCache`] (from
    ///   [`make_cache`](Self::make_cache)) — [`Full`](Qwen35LayerCache::Full) threads the [`KvCache`](cider_press_runtime::KvCache);
    ///   [`Linear`](Qwen35LayerCache::Linear) threads the GDN conv-window +
    ///   recurrent state.
    ///
    /// Returns `[1, T, hidden_size]`, lazy.
    ///
    /// # Errors
    /// The mixer and `cache` variants disagree (possible only with a cache not
    /// built by [`make_cache`](Self::make_cache)), or the mixer forward fails.
    pub fn forward(
        &self,
        hidden: &Tensor,
        offset: &Tensor,
        cache: &mut Qwen35LayerCache,
    ) -> Result<Tensor> {
        let normed = self.input_layernorm.forward(hidden)?;
        let mixer_out = match (&self.mixer, cache) {
            (Mixer::Full(attn), Qwen35LayerCache::Full(kv)) => attn.forward(&normed, offset, kv)?,
            (Mixer::Linear(gdn), Qwen35LayerCache::Linear(st)) => gdn.forward(&normed, st)?,
            _ => {
                return Err(Error::InvalidArgument(
                    "Qwen35DecoderLayer::forward: mixer/cache type mismatch (Full mixer needs \
                     Full cache, Linear needs Linear)"
                        .into(),
                ));
            }
        };
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
