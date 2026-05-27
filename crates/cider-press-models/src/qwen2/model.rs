//! Full Qwen2 model: embed → N transformer blocks → final norm →
//! tied LM head.

use cider_press_runtime::{KvCache, QuantizedWeight, Tensor};

use super::block::TransformerBlock;
use super::config::Qwen2Config;
use super::weights::Qwen2Weights;
use crate::error::{Error, Result};
use crate::nn::{embed_tokens, Module, RmsNormLayer};

/// Qwen2-family causal language model.
///
/// Owns the embedding weight, all transformer blocks, and the final
/// `RMSNorm`. The LM head is **tied**: `forward` re-projects the final
/// hidden states through the embedding weight via
/// `Tensor::quantized_matmul(.., transpose=true)`. Qwen2.5 ships
/// `tie_word_embeddings=true`; this loader/runtime assumes the tied
/// configuration.
///
/// `forward` takes caller-owned `&mut [KvCache]` (one per layer) so a
/// decode loop in branch 14 can persist them across calls.
#[derive(Debug)]
pub struct Qwen2Model {
    embed: QuantizedWeight,
    layers: Vec<TransformerBlock>,
    final_norm: RmsNormLayer,
    config: Qwen2Config,
}

impl Qwen2Model {
    /// Move loaded weights into the module tree.
    pub fn from_weights(weights: Qwen2Weights, config: Qwen2Config) -> Result<Self> {
        if weights.layers.len() != config.num_hidden_layers {
            return Err(Error::InvalidArgument(format!(
                "Qwen2Model: weights has {} layers but config says {}",
                weights.layers.len(),
                config.num_hidden_layers
            )));
        }
        let layers = weights
            .layers
            .iter()
            .map(|lw| TransformerBlock::from_weights(lw, &config))
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
    pub fn config(&self) -> &Qwen2Config {
        &self.config
    }

    /// Forward through the full model.
    ///
    /// - `input_ids`: `[1, T]` U32 token ids.
    /// - `mask`: optional additive attention mask broadcastable to
    ///   `[1, H_q, T, T_cache]` (BF16). Pass `None` for full attention
    ///   (decode of T=1 typically passes `None`).
    /// - `offset`: length-1 I32 — the number of tokens already in each
    ///   layer's cache (the `RoPE` base / starting attention position).
    /// - `caches`: one [`KvCache`] per layer; length must equal
    ///   `self.config().num_hidden_layers`.
    ///
    /// Returns `[1, T, vocab_size]` BF16, lazy.
    pub fn forward(
        &self,
        input_ids: &Tensor,
        mask: Option<&Tensor>,
        offset: &Tensor,
        caches: &mut [KvCache],
    ) -> Result<Tensor> {
        if caches.len() != self.layers.len() {
            return Err(Error::InvalidArgument(format!(
                "Qwen2Model::forward: got {} caches, expected one per layer ({})",
                caches.len(),
                self.layers.len()
            )));
        }

        let mut hidden = embed_tokens(input_ids, &self.embed)?;
        for (block, cache) in self.layers.iter().zip(caches.iter_mut()) {
            hidden = block.forward(&hidden, mask, offset, cache)?;
        }
        let normed = self.final_norm.forward(&hidden)?;
        // Tied LM head: the embedding table is [vocab, D], so the standard
        // y = x @ W^T direction (transpose=true) projects hidden states to
        // vocab-size logits without a separate weight.
        Ok(normed.quantized_matmul(&self.embed, true)?)
    }
}
