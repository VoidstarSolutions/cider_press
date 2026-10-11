//! Full Qwen3.5 / Qwen3.6 text model: embed → N mixed decoder layers →
//! final norm → LM head (tied embedding on the 4B, separate untied head on
//! the 27B).
//!
//! Structurally identical to [`Qwen2Model`](crate::qwen2::Qwen2Model); the two
//! differences are (1) the layers are [`Qwen35DecoderLayer`]s whose `forward`
//! dispatches on a per-layer mixer (full attention vs Gated-DeltaNet), so the
//! model is mixer-agnostic — it zips layers with caches; (2) the caches are
//! [`Qwen35LayerCache`]s, not a flat `KvCache` per layer.

use cider_press_runtime::{Device, QuantizedWeight, Tensor};

use super::block::Qwen35DecoderLayer;
use super::cache::Qwen35LayerCache;
use super::config::Qwen35Config;
use super::weights::Qwen35Weights;
use crate::error::{Error, Result};
use crate::language_model::LanguageModel;
use crate::nn::{Module, RmsNormLayer, embed_tokens, rope_offset};

/// Prompt tokens per prefill chunk. The per-token Gated-DeltaNet recurrence
/// keeps ~5 fp32 state-sized temporaries per token per GDN layer alive until
/// its eval completes (~0.75 GB/token on the 27B's 48 GDN layers; ~0.24
/// GB/token measured on the 4B), so 16-token chunks bound the transient to
/// 16 × that per-token cost at any prompt length. Halving it to 8 cost ~7% 4B
/// prefill throughput.
const QWEN35_PREFILL_CHUNK: usize = 16;

/// Qwen3.5 / Qwen3.6 causal language model.
///
/// Owns the embedding weight, all (mixed full-attention / Gated-DeltaNet)
/// decoder layers, and the final `RMSNorm`. The embedding is a plain lookup
/// (no `√hidden` scale). The LM head is chosen at load time from the checkpoint weights:
///
/// - **Tied** (`tie_word_embeddings=true`, the 4B dev vehicle): `lm_head` is
///   `None`; `forward` re-projects the final hidden states through the
///   embedding weight via `Tensor::quantized_matmul(.., transpose=true)`.
/// - **Untied** (`tie_word_embeddings=false`, the 27B target): `lm_head` is
///   `Some` separate `[vocab, hidden]` quantized head, projected with the same
///   `y = x·Wᵀ` direction.
///
/// `forward` takes caller-owned `&mut [Qwen35LayerCache]` (one per layer, from
/// [`make_caches`](Self::make_caches)) so a decode loop can persist them
/// across calls.
#[derive(Debug)]
pub struct Qwen35Model {
    embed: QuantizedWeight,
    /// Untied LM head (`[vocab, hidden]`, quantized) on the 27B; `None` on the
    /// tied 4B, where logits project through `embed`.
    lm_head: Option<QuantizedWeight>,
    layers: Vec<Qwen35DecoderLayer>,
    final_norm: RmsNormLayer,
    config: Qwen35Config,
}

impl Qwen35Model {
    /// Move loaded weights into the module tree.
    ///
    /// # Errors
    /// - `weights.layers.len()` disagrees with `config.num_hidden_layers`.
    /// - a layer fails to build (mixer-weight / config mismatch).
    pub fn from_weights(weights: Qwen35Weights, config: Qwen35Config) -> Result<Self> {
        if weights.layers.len() != config.num_hidden_layers {
            return Err(Error::InvalidArgument(format!(
                "Qwen35Model: weights has {} layers but config says {}",
                weights.layers.len(),
                config.num_hidden_layers
            )));
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
            lm_head: weights.lm_head,
            layers,
            final_norm,
            config,
        })
    }

    /// One fresh cache per layer, each matching its layer's mixer; full-attention
    /// caches are sized for `max_tokens` positions.
    ///
    /// # Errors
    /// Device allocation failure.
    pub fn make_caches(&self, max_tokens: usize, device: &Device) -> Result<Vec<Qwen35LayerCache>> {
        self.layers
            .iter()
            .map(|layer| layer.make_cache(max_tokens, device))
            .collect()
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
    /// - `caches`: one [`Qwen35LayerCache`] per layer; length must equal
    ///   `self.config().num_hidden_layers`. The full-attention caches' shared
    ///   position — the number of tokens already cached — is the `RoPE`
    ///   offset.
    ///
    /// Returns `[1, T, vocab_size]` BF16, lazy.
    ///
    /// # Errors
    /// `caches.len()` disagrees with the layer count, the full-attention caches
    /// are absent or disagree on their position, or any layer / projection op
    /// fails.
    pub fn forward(&self, input_ids: &Tensor, caches: &mut [Qwen35LayerCache]) -> Result<Tensor> {
        let hidden = self.hidden_states(input_ids, caches)?;
        self.project_logits(&hidden)
    }

    /// Like [`forward`](Self::forward), but projects **only the last
    /// position** to logits: returns `[1, 1, vocab_size]`.
    ///
    /// The decoder layers still run over all `T` tokens (writing every layer's
    /// mixer state for all positions into `caches`); only the final norm and
    /// the LM head are computed, for the final position alone. That is the
    /// single logits row greedy generation consumes — projecting the whole
    /// `[1, T, vocab]` is wasted work. Numerically identical to
    /// `forward(..)[:, -1, :]` because `RMSNorm` and the LM head are
    /// per-position.
    ///
    /// # Errors
    /// `input_ids` is empty, or as for [`forward`](Self::forward).
    pub fn forward_last(
        &self,
        input_ids: &Tensor,
        caches: &mut [Qwen35LayerCache],
    ) -> Result<Tensor> {
        let t = input_ids.elem_count();
        if t == 0 {
            return Err(Error::InvalidArgument(
                "Qwen35Model::forward_last: empty input_ids (need at least one token)".into(),
            ));
        }
        let hidden = self.hidden_states(input_ids, caches)?;
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

    /// Run every decoder layer over `input_ids`, advancing all
    /// `caches` by `T`, **without** the final norm / LM head — the
    /// non-final-chunk step of chunked prefill.
    ///
    /// Returns the final hidden states `[1, T, hidden_size]`, lazy. Every
    /// piece of cache state the next call reads is an ancestor of it: each
    /// full-attention layer's output reads the K/V `slice_update` it wrote,
    /// each GDN layer's last output token reads the final recurrent state,
    /// and the conv window is a view of the conv input those outputs consume.
    /// `eval()`ing it therefore commits the whole chunk — which the caller
    /// must do before the next call (an un-committed `KvCache` write is
    /// dropped by the next update).
    ///
    /// # Errors
    /// `input_ids` is empty, `caches.len()` disagrees with the layer count, the
    /// full-attention caches are absent or disagree on their position, or any
    /// underlying op fails.
    pub fn fill_cache(
        &self,
        input_ids: &Tensor,
        caches: &mut [Qwen35LayerCache],
    ) -> Result<Tensor> {
        if input_ids.elem_count() == 0 {
            return Err(Error::InvalidArgument(
                "Qwen35Model::fill_cache: empty input_ids (need at least one token)".into(),
            ));
        }
        self.hidden_states(input_ids, caches)
    }

    /// Post-final-norm hidden states `[1, T, hidden_size]` (before the
    /// `lm_head`). Mirrors mlx-lm's `lm.norm(x)`; used for backbone parity
    /// localization — a failure here points at the decoder layers (vs the
    /// LM head, exercised only by [`forward`](Self::forward)).
    ///
    /// # Errors
    /// `caches.len()` disagrees with the layer count, the full-attention caches
    /// are absent or disagree on their position, or any layer / norm op fails.
    pub fn forward_hidden(
        &self,
        input_ids: &Tensor,
        caches: &mut [Qwen35LayerCache],
    ) -> Result<Tensor> {
        let hidden = self.hidden_states(input_ids, caches)?;
        self.final_norm.forward(&hidden)
    }

    /// Embed `input_ids` and run every decoder layer, returning the final
    /// hidden states `[1, T, hidden_size]` (before the final norm / LM head).
    /// Writes each layer's mixer state for all `T` positions into `caches`.
    fn hidden_states(&self, input_ids: &Tensor, caches: &mut [Qwen35LayerCache]) -> Result<Tensor> {
        if caches.len() != self.layers.len() {
            return Err(Error::InvalidArgument(format!(
                "Qwen35Model::hidden_states: got {} caches, expected one per layer ({})",
                caches.len(),
                self.layers.len()
            )));
        }

        let offset = rope_offset(
            caches.iter().filter_map(|cache| match cache {
                Qwen35LayerCache::Full(kv) => Some(kv),
                Qwen35LayerCache::Linear(_) => None,
            }),
            "Qwen35Model",
        )?;

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
            hidden = layer.forward(&hidden, &offset, cache)?;
        }
        Ok(hidden)
    }

    /// Final norm + LM head over `hidden` (`[1, S, hidden_size]`), returning
    /// `[1, S, vocab_size]` BF16, lazy.
    fn project_logits(&self, hidden: &Tensor) -> Result<Tensor> {
        let normed = self.final_norm.forward(hidden)?;
        // Both heads are [vocab, D], so the standard y = x @ W^T direction
        // (transpose=true) projects hidden states to vocab-size logits. Untied
        // (27B): the separate `lm_head`. Tied (4B): the embedding table itself.
        let head = self.lm_head.as_ref().unwrap_or(&self.embed);
        Ok(normed.quantized_matmul(head, true)?)
    }
}

impl LanguageModel for Qwen35Model {
    type Cache = Qwen35LayerCache;

    fn make_caches(&self, max_tokens: usize, device: &Device) -> Result<Vec<Qwen35LayerCache>> {
        Qwen35Model::make_caches(self, max_tokens, device)
    }

    fn forward_last(&self, input_ids: &Tensor, caches: &mut [Qwen35LayerCache]) -> Result<Tensor> {
        Qwen35Model::forward_last(self, input_ids, caches)
    }

    fn fill_cache(&self, input_ids: &Tensor, caches: &mut [Qwen35LayerCache]) -> Result<Tensor> {
        Qwen35Model::fill_cache(self, input_ids, caches)
    }

    fn prefill_chunk_len(&self) -> Option<usize> {
        Some(QWEN35_PREFILL_CHUNK)
    }

    fn reset_caches(&self, caches: &mut [Qwen35LayerCache]) -> Result<()> {
        for cache in caches {
            cache.reset()?;
        }
        Ok(())
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }
}

#[cfg(test)]
mod tests {
    use cider_press_runtime::DType;
    use safetensors::SafeTensors;

    use super::*;
    use crate::qwen3_5::load_qwen35_weights;
    use crate::qwen3_5::test_support::{Fill, TINY_CONFIG_JSON, synthesize_archive_bytes_with};

    #[test]
    fn make_caches_follow_each_layer_mixer() {
        let device = Device::shared().unwrap();
        let config = Qwen35Config::from_json_bytes(TINY_CONFIG_JSON.as_bytes()).unwrap();
        let bytes = synthesize_archive_bytes_with(&config, false, Fill::Zeros, |_, _| None);
        let archive = SafeTensors::deserialize(&bytes).unwrap();
        let weights = load_qwen35_weights(&archive, &config, &device).unwrap();
        let model = Qwen35Model::from_weights(weights, config.clone()).unwrap();

        let caches = model.make_caches(8, &device).unwrap();
        assert_eq!(caches.len(), config.num_hidden_layers);
        for (i, cache) in caches.iter().enumerate() {
            match cache {
                Qwen35LayerCache::Linear(state) => {
                    assert!(
                        config.is_linear(i),
                        "layer {i}: Linear cache on a full layer"
                    );
                    assert_eq!(
                        state.conv_state().shape().dims(),
                        &[1, config.linear_conv_kernel_dim - 1, config.conv_dim()]
                    );
                    assert_eq!(state.recurrent_state().dtype(), DType::F32);
                }
                Qwen35LayerCache::Full(kv) => {
                    assert!(
                        !config.is_linear(i),
                        "layer {i}: Full cache on a linear layer"
                    );
                    assert_eq!(kv.head_dim(), config.head_dim);
                    assert_eq!(kv.dtype(), DType::BF16);
                }
            }
        }
    }
}
