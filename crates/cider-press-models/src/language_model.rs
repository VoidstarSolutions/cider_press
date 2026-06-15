//! The minimal model interface the greedy [`Generator`](crate::generator::Generator)
//! drives: build per-layer caches, run a last-position forward, and reset the
//! caches between sequences.
//!
//! Lifting the generator to a trait waited until a second architecture
//! arrived; with both [`Qwen2Model`] (cache =
//! [`KvCache`](cider_press_runtime::KvCache)) and
//! [`Qwen35Model`] (cache =
//! [`Qwen35LayerCache`](crate::qwen3_5::Qwen35LayerCache)) in the tree, the
//! generator is generic over this trait and its associated `Cache` type. The
//! async-pipelined decode loop only needs `forward_last` + argmax + the cache
//! vector, all of which are cache-type-agnostic.

use cider_press_runtime::{Device, Tensor};

use crate::error::Result;
use crate::qwen2::Qwen2Model;
use crate::qwen3_5::Qwen35Model;

/// A causal language model the greedy generator can drive: it builds and resets
/// its own per-layer caches and exposes a last-position forward.
///
/// The generator never names a concrete cache type — it holds a
/// `Vec<M::Cache>` and threads it through [`forward_last`](Self::forward_last).
/// Both prefill (`T = prompt_len`) and each decode step (`T = 1`) go through
/// `forward_last`, which projects only the final position to `[1, 1, vocab]`
/// (for `T = 1` that is the whole sequence).
pub trait LanguageModel {
    /// Per-layer cache the model writes K/V (or mixer) state into. `Debug` so
    /// the generator and its iterator can derive `Debug` uniformly.
    type Cache: std::fmt::Debug;

    /// Allocate one cache per layer, each sized for `max_tokens` positions.
    /// The model supplies its own config (layer count / per-layer dims).
    ///
    /// # Errors
    /// Cache allocation failure.
    fn make_caches(&self, max_tokens: usize, device: &Device) -> Result<Vec<Self::Cache>>;

    /// Forward over `input_ids` `[1, T]` at `offset` (length-1 I32 = the number
    /// of tokens already in each cache), writing all `T` positions' state into
    /// `caches` and returning the **last-position** logits `[1, 1, vocab]`.
    ///
    /// # Errors
    /// Empty `input_ids`, a cache-count mismatch, or any underlying op failure.
    fn forward_last(
        &self,
        input_ids: &Tensor,
        offset: &Tensor,
        caches: &mut [Self::Cache],
    ) -> Result<Tensor>;

    /// Reset every cache to position 0 for a fresh sequence. `device` is the
    /// allocation device for caches that re-zero state tensors (the attention
    /// KV caches merely rewind; the linear caches re-allocate).
    ///
    /// # Errors
    /// Propagates a cache reset failure (e.g. an un-committed prior update).
    fn reset_caches(&self, caches: &mut [Self::Cache], device: &Device) -> Result<()>;

    /// Vocabulary size — the logits inner dimension, used to validate the
    /// argmax input shape.
    fn vocab_size(&self) -> usize;
}

impl LanguageModel for Qwen2Model {
    type Cache = cider_press_runtime::KvCache;

    fn make_caches(&self, max_tokens: usize, device: &Device) -> Result<Vec<Self::Cache>> {
        let cfg = self.config();
        let head_dim = cfg.head_dim()?;
        (0..cfg.num_hidden_layers)
            .map(|_| {
                cider_press_runtime::KvCache::new(
                    device,
                    max_tokens,
                    cfg.num_key_value_heads,
                    head_dim,
                    cider_press_runtime::DType::BF16,
                )
                .map_err(Into::into)
            })
            .collect()
    }

    fn forward_last(
        &self,
        input_ids: &Tensor,
        offset: &Tensor,
        caches: &mut [Self::Cache],
    ) -> Result<Tensor> {
        Qwen2Model::forward_last(self, input_ids, offset, caches)
    }

    fn reset_caches(&self, caches: &mut [Self::Cache], _device: &Device) -> Result<()> {
        for cache in caches {
            cache.reset()?;
        }
        Ok(())
    }

    fn vocab_size(&self) -> usize {
        self.config().vocab_size
    }
}

impl LanguageModel for Qwen35Model {
    type Cache = crate::qwen3_5::Qwen35LayerCache;

    fn make_caches(&self, max_tokens: usize, device: &Device) -> Result<Vec<Self::Cache>> {
        let cfg = self.config();
        (0..cfg.num_hidden_layers)
            .map(|i| crate::qwen3_5::Qwen35LayerCache::for_layer(cfg, i, device, max_tokens))
            .collect()
    }

    fn forward_last(
        &self,
        input_ids: &Tensor,
        offset: &Tensor,
        caches: &mut [Self::Cache],
    ) -> Result<Tensor> {
        Qwen35Model::forward_last(self, input_ids, offset, caches)
    }

    fn reset_caches(&self, caches: &mut [Self::Cache], device: &Device) -> Result<()> {
        for cache in caches {
            cache.reset(device)?;
        }
        Ok(())
    }

    fn vocab_size(&self) -> usize {
        self.config().vocab_size
    }
}
