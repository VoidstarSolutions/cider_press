//! Full Qwen2 model: embed → N transformer blocks → final norm →
//! tied LM head.

use cider_press_runtime::{KvCache, QuantizedWeight, Tensor};

use super::block::TransformerBlock;
use super::config::Qwen2Config;
use super::weights::Qwen2Weights;
use crate::error::{Error, Result};
use crate::nn::{Module, RmsNormLayer, embed_tokens};

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
/// decode loop can persist them across calls.
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
    /// - `input_ids`: `[1, T]` U32 token ids. Flattened to rank-1 for the
    ///   embedding gather, then reshaped back to `[1, T, hidden_size]`
    ///   before the transformer blocks.
    /// - `offset`: length-1 I32 — the number of tokens already in each
    ///   layer's cache (the `RoPE` base / starting attention position).
    /// - `caches`: one [`KvCache`] per layer; length must equal
    ///   `self.config().num_hidden_layers`.
    ///
    /// Returns `[1, T, vocab_size]` BF16, lazy.
    pub fn forward(
        &self,
        input_ids: &Tensor,
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

        // embed_tokens takes rank-1 ids and returns [T, hidden]; the blocks
        // require [1, T, hidden]. Flatten on the way in, restore the leading
        // batch axis on the way out.
        let t = input_ids.elem_count();
        // reshape returns a view; gather and the downstream unary/reduce
        // dispatchers read inputs as dense leaves and don't resolve view
        // chains, so materialise both ends with copy().
        let flat_ids = input_ids.reshape([t])?.copy()?;
        let embedded = embed_tokens(&flat_ids, &self.embed)?;
        let mut hidden = embedded
            .reshape([1usize, t, self.config.hidden_size])?
            .copy()?;
        for (block, cache) in self.layers.iter().zip(caches.iter_mut()) {
            hidden = block.forward(&hidden, offset, cache)?;
        }
        let normed = self.final_norm.forward(&hidden)?;
        // Tied LM head: the embedding table is [vocab, D], so the standard
        // y = x @ W^T direction (transpose=true) projects hidden states to
        // vocab-size logits without a separate weight.
        Ok(normed.quantized_matmul(&self.embed, true)?)
    }
}
