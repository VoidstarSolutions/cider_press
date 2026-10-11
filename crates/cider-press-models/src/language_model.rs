//! The minimal model interface the greedy [`Generator`](crate::generator::Generator)
//! drives: build per-layer caches, run a last-position forward, and reset the
//! caches between sequences.
//!
//! The generator is generic over this trait and its associated `Cache` type:
//! [`Qwen2Model`] uses [`KvCache`](cider_press_runtime::KvCache) and
//! [`Qwen35Model`] uses
//! [`Qwen35LayerCache`](crate::qwen3_5::Qwen35LayerCache). The async-pipelined
//! decode loop only needs `forward_last` + argmax + the cache vector, all of
//! which are cache-type-agnostic.

use cider_press_runtime::{Device, Tensor};

use crate::error::Result;
use crate::qwen2::Qwen2Model;
use crate::qwen3_5::Qwen35Model;

/// A causal language model the greedy generator can drive: it builds and resets
/// its own per-layer caches and exposes a last-position forward.
///
/// The generator never names a concrete cache type — it holds a
/// `Vec<M::Cache>` and threads it through [`forward_last`](Self::forward_last).
/// Prefill's final (or only) chunk and each decode step (`T = 1`) go through
/// `forward_last`, which projects only the final position to `[1, 1, vocab]`
/// (for `T = 1` that is the whole sequence). A model that opts into chunked
/// prefill via [`prefill_chunk_len`](Self::prefill_chunk_len) has its earlier
/// prompt chunks run through [`fill_cache`](Self::fill_cache), each evaluated
/// before the next.
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

    /// Forward over `input_ids` `[1, T]`, writing all `T` positions' state into
    /// `caches` and returning the **last-position** logits `[1, 1, vocab]`.
    /// The model derives the `RoPE` / attention offset from the caches' own
    /// position (the tokens already fed), so the caller never supplies it.
    ///
    /// # Errors
    /// Empty `input_ids`, a cache-count mismatch, caches that disagree on
    /// their position, or any underlying op failure.
    fn forward_last(&self, input_ids: &Tensor, caches: &mut [Self::Cache]) -> Result<Tensor>;

    /// Run every layer over `input_ids` `[1, T]` at the caches' current
    /// position, advancing all `caches` by `T`, without the final norm / LM
    /// head. Returns a lazy handle whose `eval()` commits every cache write for
    /// the chunk; the caller must evaluate it before the next cache-writing
    /// call.
    ///
    /// # Errors
    /// Empty `input_ids`, a cache-count mismatch, caches that disagree on
    /// their position, or any underlying op failure.
    fn fill_cache(&self, input_ids: &Tensor, caches: &mut [Self::Cache]) -> Result<Tensor>;

    /// Maximum prompt tokens per prefill eval, or `None` to prefill the whole
    /// prompt in one forward. A model whose prefill transient grows with `T`
    /// (beyond the caches themselves) returns `Some` to bound peak memory; the
    /// generator then feeds longer prompts chunk by chunk.
    fn prefill_chunk_len(&self) -> Option<usize> {
        None
    }

    /// Reset every cache to position 0 for a fresh sequence.
    ///
    /// # Errors
    /// Propagates a cache reset failure (e.g. an un-committed prior update).
    fn reset_caches(&self, caches: &mut [Self::Cache]) -> Result<()>;

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

    fn forward_last(&self, input_ids: &Tensor, caches: &mut [Self::Cache]) -> Result<Tensor> {
        Qwen2Model::forward_last(self, input_ids, caches)
    }

    fn fill_cache(&self, input_ids: &Tensor, caches: &mut [Self::Cache]) -> Result<Tensor> {
        Qwen2Model::fill_cache(self, input_ids, caches)
    }

    fn reset_caches(&self, caches: &mut [Self::Cache]) -> Result<()> {
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
        Qwen35Model::make_caches(self, max_tokens, device)
    }

    fn forward_last(&self, input_ids: &Tensor, caches: &mut [Self::Cache]) -> Result<Tensor> {
        Qwen35Model::forward_last(self, input_ids, caches)
    }

    fn fill_cache(&self, input_ids: &Tensor, caches: &mut [Self::Cache]) -> Result<Tensor> {
        Qwen35Model::fill_cache(self, input_ids, caches)
    }

    fn prefill_chunk_len(&self) -> Option<usize> {
        Some(crate::qwen3_5::QWEN35_PREFILL_CHUNK)
    }

    fn reset_caches(&self, caches: &mut [Self::Cache]) -> Result<()> {
        for cache in caches {
            cache.reset()?;
        }
        Ok(())
    }

    fn vocab_size(&self) -> usize {
        self.config().vocab_size
    }
}
