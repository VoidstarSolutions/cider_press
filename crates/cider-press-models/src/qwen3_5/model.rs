//! Full Qwen3.5 / Qwen3.6 text model: embed → N mixed decoder layers →
//! final norm → tied LM head.
//!
//! Structurally identical to [`Qwen2Model`](crate::qwen2::Qwen2Model); the two
//! differences are (1) the layers are [`Qwen35DecoderLayer`]s whose `forward`
//! dispatches on a per-layer mixer (full attention vs Gated-DeltaNet), so the
//! model is mixer-agnostic — it zips layers with caches; (2) the caches are
//! [`Qwen35LayerCache`]s, not a flat `KvCache` per layer.

use cider_press_runtime::{QuantizedWeight, Tensor};

use super::block::Qwen35DecoderLayer;
use super::cache::Qwen35LayerCache;
use super::config::Qwen35Config;
use super::weights::Qwen35Weights;
use crate::error::{Error, Result};
use crate::nn::{Module, RmsNormLayer, embed_tokens};

/// Qwen3.5 / Qwen3.6 causal language model.
///
/// Owns the embedding weight, all (mixed full-attention / Gated-DeltaNet)
/// decoder layers, and the final `RMSNorm`. The embedding is a plain lookup
/// (no `√hidden` scale). The LM head is **tied**: `forward` re-projects the
/// final hidden states through the embedding weight via
/// `Tensor::quantized_matmul(.., transpose=true)`. The 4B dev vehicle ships
/// `tie_word_embeddings=true`; the untied 27B `lm_head` is deferred to Phase 5
/// ([`from_weights`](Self::from_weights) errors if `tie_word_embeddings` is
/// false).
///
/// `forward` takes caller-owned `&mut [Qwen35LayerCache]` (one per layer) so a
/// decode loop can persist them across calls.
#[derive(Debug)]
pub struct Qwen35Model {
    embed: QuantizedWeight,
    layers: Vec<Qwen35DecoderLayer>,
    final_norm: RmsNormLayer,
    config: Qwen35Config,
}

impl Qwen35Model {
    /// Move loaded weights into the module tree.
    ///
    /// # Errors
    /// - `weights.layers.len()` disagrees with `config.num_hidden_layers`.
    /// - `config.tie_word_embeddings` is false (untied 27B `lm_head` is
    ///   Phase 5).
    /// - a layer fails to build (mixer-weight / config mismatch).
    pub fn from_weights(weights: Qwen35Weights, config: Qwen35Config) -> Result<Self> {
        if weights.layers.len() != config.num_hidden_layers {
            return Err(Error::InvalidArgument(format!(
                "Qwen35Model: weights has {} layers but config says {}",
                weights.layers.len(),
                config.num_hidden_layers
            )));
        }
        if !config.tie_word_embeddings {
            return Err(Error::InvalidArgument(
                "Qwen35Model: untied lm_head not yet supported (27B is Phase 5)".into(),
            ));
        }
        let layers = weights
            .layers
            .iter()
            .map(|lw| Qwen35DecoderLayer::from_weights(lw, &config))
            .collect::<Result<Vec<_>>>()?;
        // HF config.json emits eps as JSON numeric (f64); the value (~1e-6) is exactly representable in f32.
        #[allow(clippy::cast_possible_truncation)]
        let final_norm = RmsNormLayer::new(weights.norm, config.rms_norm_eps as f32);
        Ok(Self {
            embed: weights.embed,
            layers,
            final_norm,
            config,
        })
    }

    /// The config the model was built from.
    #[must_use]
    pub fn config(&self) -> &Qwen35Config {
        &self.config
    }

    /// Forward through the full model.
    ///
    /// - `input_ids`: `[1, T]` U32 token ids. Flattened to rank-1 for the
    ///   embedding gather, then reshaped back to `[1, T, hidden_size]`
    ///   before the decoder layers.
    /// - `offset`: length-1 I32 — the number of tokens already in each
    ///   layer's cache (the `RoPE` base / starting attention position).
    /// - `caches`: one [`Qwen35LayerCache`] per layer; length must equal
    ///   `self.config().num_hidden_layers`.
    ///
    /// Returns `[1, T, vocab_size]` BF16, lazy.
    ///
    /// # Errors
    /// `caches.len()` disagrees with the layer count, or any layer / projection
    /// op fails.
    pub fn forward(
        &self,
        input_ids: &Tensor,
        offset: &Tensor,
        caches: &mut [Qwen35LayerCache],
    ) -> Result<Tensor> {
        let hidden = self.hidden_states(input_ids, offset, caches)?;
        self.project_logits(&hidden)
    }

    /// Like [`forward`](Self::forward), but projects **only the last
    /// position** to logits: returns `[1, 1, vocab_size]`.
    ///
    /// The decoder layers still run over all `T` tokens (writing every layer's
    /// mixer state for all positions into `caches`); only the final norm and
    /// the tied LM head are computed, for the final position alone. That is the
    /// single logits row greedy generation consumes — projecting the whole
    /// `[1, T, vocab]` is wasted work. Numerically identical to
    /// `forward(..)[:, -1, :]` because `RMSNorm` and the LM head are
    /// per-position.
    ///
    /// # Errors
    /// `input_ids` is empty, or any underlying op fails.
    pub fn forward_last(
        &self,
        input_ids: &Tensor,
        offset: &Tensor,
        caches: &mut [Qwen35LayerCache],
    ) -> Result<Tensor> {
        let t = input_ids.elem_count();
        if t == 0 {
            return Err(Error::InvalidArgument(
                "Qwen35Model::forward_last: empty input_ids (need at least one token)".into(),
            ));
        }
        let hidden = self.hidden_states(input_ids, offset, caches)?;
        let last = if t == 1 {
            hidden
        } else {
            // Slice to the last token before norm/head; copy() materialises
            // the strided view dense (the head reads inputs as dense leaves).
            hidden
                .slice(&[0..1, t - 1..t, 0..self.config.hidden_size])?
                .copy()?
        };
        self.project_logits(&last)
    }

    /// Post-final-norm hidden states `[1, T, hidden_size]` (before the
    /// `lm_head`). Mirrors mlx-lm's `lm.norm(x)`; used for backbone parity
    /// localization — a failure here points at the 32 decoder layers (vs the
    /// tied head, exercised only by [`forward`](Self::forward)).
    ///
    /// # Errors
    /// `caches.len()` disagrees with the layer count, or any layer / norm op
    /// fails.
    pub fn forward_hidden(
        &self,
        input_ids: &Tensor,
        offset: &Tensor,
        caches: &mut [Qwen35LayerCache],
    ) -> Result<Tensor> {
        let hidden = self.hidden_states(input_ids, offset, caches)?;
        self.final_norm.forward(&hidden)
    }

    /// Embed `input_ids` and run every decoder layer, returning the final
    /// hidden states `[1, T, hidden_size]` (before the final norm / LM head).
    /// Writes each layer's mixer state for all `T` positions into `caches`.
    fn hidden_states(
        &self,
        input_ids: &Tensor,
        offset: &Tensor,
        caches: &mut [Qwen35LayerCache],
    ) -> Result<Tensor> {
        if caches.len() != self.layers.len() {
            return Err(Error::InvalidArgument(format!(
                "Qwen35Model::hidden_states: got {} caches, expected one per layer ({})",
                caches.len(),
                self.layers.len()
            )));
        }

        // embed_tokens takes rank-1 ids and returns [T, hidden]; the layers
        // require [1, T, hidden]. Flatten on the way in, restore the leading
        // batch axis on the way out. Qwen3.5 uses a plain lookup (no √hidden
        // scale).
        let t = input_ids.elem_count();
        // reshape returns a view; gather and the downstream unary/reduce
        // dispatchers read inputs as dense leaves and don't resolve view
        // chains, so materialise both ends with copy().
        let flat_ids = input_ids.reshape([t])?.copy()?;
        let embedded = embed_tokens(&flat_ids, &self.embed)?;
        let mut hidden = embedded
            .reshape([1usize, t, self.config.hidden_size])?
            .copy()?;
        for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
            hidden = layer.forward(&hidden, offset, cache)?;
        }
        Ok(hidden)
    }

    /// Final norm + tied LM head over `hidden` (`[1, S, hidden_size]`),
    /// returning `[1, S, vocab_size]` BF16, lazy.
    fn project_logits(&self, hidden: &Tensor) -> Result<Tensor> {
        let normed = self.final_norm.forward(hidden)?;
        // Tied LM head: the embedding table is [vocab, D], so the standard
        // y = x @ W^T direction (transpose=true) projects hidden states to
        // vocab-size logits without a separate weight.
        Ok(normed.quantized_matmul(&self.embed, true)?)
    }
}
