//! Greedy-decode driver over [`Qwen2Model`] + [`KvCache`].
//!
//! Concrete-on-Qwen2Model for now; lifting to a `LanguageModel`
//! trait waits until a second architecture arrives. The iterator
//! yields `Result<u32>` ids; UTF-8 detokenization is the caller's
//! problem (compose `Tokenizer::decode_stream` on top).

use std::collections::HashSet;

use cider_press_runtime::{DType, Device, KvCache, Tensor};

use crate::error::{Error, Result};
use crate::nn::causal_mask;
use crate::qwen2::Qwen2Model;

/// Owns a model + pre-allocated `KvCache`s + the EOS set. One instance
/// drives one or more `generate` calls; caches are reset on every
/// call so previous runs don't leak into the next prompt.
#[derive(Debug)]
pub struct Generator {
    model: Qwen2Model,
    caches: Vec<KvCache>,
    eos_ids: HashSet<u32>,
    context_window: usize,
}

impl Generator {
    /// Construct a Generator. Pre-allocates one [`KvCache`] per layer
    /// sized to `context_window` tokens.
    ///
    /// # Errors
    /// - `context_window == 0`
    /// - `eos_ids.is_empty()` — without an EOS the loop has no
    ///   natural terminator besides `max_new_tokens`; we reject so a
    ///   caller can't accidentally rely on that.
    pub fn new(model: Qwen2Model, context_window: usize, eos_ids: HashSet<u32>) -> Result<Self> {
        if context_window == 0 {
            return Err(Error::InvalidArgument(
                "Generator::new: context_window must be > 0".into(),
            ));
        }
        if eos_ids.is_empty() {
            return Err(Error::InvalidArgument(
                "Generator::new: eos_ids must be non-empty".into(),
            ));
        }
        let device = Device::shared()?;
        let cfg = model.config();
        let head_dim = cfg.head_dim()?;
        let caches = (0..cfg.num_hidden_layers)
            .map(|_| {
                KvCache::new(
                    &device,
                    context_window,
                    cfg.num_key_value_heads,
                    head_dim,
                    DType::BF16,
                )
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(Self {
            model,
            caches,
            eos_ids,
            context_window,
        })
    }

    /// Reset all `KvCache`s to position 0. Called automatically at the
    /// start of [`Self::generate`].
    ///
    /// # Errors
    /// Propagates [`KvCache::reset`] if a cache's last update has not been
    /// evaluated (it must be, to keep outstanding K/V views from replaying
    /// stale writes into the reused slab).
    pub fn reset(&mut self) -> Result<()> {
        for cache in &mut self.caches {
            cache.reset()?;
        }
        Ok(())
    }

    /// Run one decode-step forward (`T = 1`, no mask, offset = the current
    /// cache position) through the profiling-only [`Tensor::profiled_eval`],
    /// recording per-OpKind GPU time into the `profile` GPU store.
    ///
    /// `last_id` is the most recently sampled token; the caches must hold
    /// the full preceding context (call this *after* draining a
    /// [`GenerateIter`]), so the graph is a realistic full-context decode
    /// step rather than a cold `T = 1`. Note this advances the caches by one
    /// row as a side effect — intended for one-shot profiling, not resumable
    /// generation.
    ///
    /// # Errors
    /// Tensor construction, forward, or `profiled_eval` failure.
    #[doc(hidden)]
    pub fn profiled_decode_step(&mut self, last_id: u32) -> Result<()> {
        let device = Device::shared()?;
        let pos = self.caches.first().map_or(0, KvCache::position);
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let pos_i32 = pos as i32;
        let ids = Tensor::from_slice(&device, &[last_id], [1, 1])?;
        let offset = Tensor::from_slice(&device, &[pos_i32], [1])?;
        let logits = self.model.forward(&ids, None, &offset, &mut self.caches)?;
        logits.profiled_eval()?;
        Ok(())
    }

    /// Greedy-decode up to `max_new_tokens` ids from `input_ids`.
    ///
    /// Runs prefill (one forward at `T = input_ids.len()`, offset=0,
    /// causal mask) then yields sampled ids one at a time. Iterator
    /// terminates on EOS or after `max_new_tokens` items.
    ///
    /// # Errors
    ///
    /// `generate()` itself returns `Err` for construction-time validation:
    /// - `input_ids.is_empty()`
    /// - `max_new_tokens == 0`
    /// - `input_ids.len() + max_new_tokens` overflows `usize`
    /// - `input_ids.len() + max_new_tokens > context_window`
    /// - Device acquisition or prefill-forward failure (rare; surfaces
    ///   construction-time runtime errors as the `?` propagates).
    ///
    /// Per-step forward-pass errors during decode surface as
    /// `Some(Err(...))` from the returned iterator, NOT as an `Err` on
    /// `generate()`. Callers iterating manually (rather than with
    /// `.collect::<Result<_, _>>()`) need to handle the `Some(Err)` case.
    pub fn generate(
        &mut self,
        input_ids: &[u32],
        max_new_tokens: usize,
    ) -> Result<GenerateIter<'_>> {
        if input_ids.is_empty() {
            return Err(Error::InvalidArgument(
                "Generator::generate: input_ids must be non-empty".into(),
            ));
        }
        if max_new_tokens == 0 {
            return Err(Error::InvalidArgument(
                "Generator::generate: max_new_tokens must be > 0 \
                 (the iterator yields the prefill argmax as its first item)"
                    .into(),
            ));
        }
        let total = input_ids.len().checked_add(max_new_tokens).ok_or_else(|| {
            Error::InvalidArgument(
                "Generator::generate: input_ids.len() + max_new_tokens overflow".into(),
            )
        })?;
        if total > self.context_window {
            return Err(Error::InvalidArgument(format!(
                "Generator::generate: input_ids.len() ({}) + max_new_tokens ({}) > context_window ({})",
                input_ids.len(),
                max_new_tokens,
                self.context_window,
            )));
        }
        self.reset()?;
        let device = Device::shared()?;
        let prefill_len = input_ids.len();

        // Prefill: [1, T] ids, offset=0, explicit causal mask.
        let ids_tensor = Tensor::from_slice(&device, input_ids, [1, prefill_len])?;
        let offset = Tensor::from_slice(&device, &[0i32], [1])?;
        let mask = causal_mask(&device, prefill_len)?;
        let logits = self
            .model
            .forward(&ids_tensor, Some(&mask), &offset, &mut self.caches)?;
        let first = argmax_last_position(&logits, self.model.config().vocab_size)?;

        Ok(GenerateIter {
            owner: self,
            device,
            prefill_len,
            next_input: Some(first),
            step: 0,
            max_new_tokens,
        })
    }
}

/// Iterator yielding `Result<u32>` sampled ids. Borrows `&mut Generator`.
#[derive(Debug)]
pub struct GenerateIter<'a> {
    owner: &'a mut Generator,
    device: Device,
    prefill_len: usize,
    /// The id to yield next (and to feed into the next forward). `None`
    /// once the iterator has terminated.
    next_input: Option<u32>,
    step: usize,
    max_new_tokens: usize,
}

impl Iterator for GenerateIter<'_> {
    type Item = Result<u32>;

    fn next(&mut self) -> Option<Self::Item> {
        let id = self.next_input.take()?;
        // `next_input` always holds the id we're about to yield — it was
        // pre-computed during the previous call's `advance()` (or during
        // `Generator::generate()` for the very first yield, which is the
        // prefill argmax). The pattern per call: (1) take that pre-decided
        // id, (2) increment `step`, (3) check whether this id terminates
        // the sequence (EOS or max_new_tokens hit), (4) if not terminal,
        // run one decode `advance()` to pre-fetch the *next* id into
        // `next_input` for the following call, then (5) yield the id we
        // took. Terminal cases yield the id but leave `next_input = None`
        // so the subsequent `next()` returns None.
        self.step += 1;
        if id_is_terminal(id, &self.owner.eos_ids) || self.step >= self.max_new_tokens {
            // Don't run another forward; iterator ends after we hand this id back.
            return Some(Ok(id));
        }
        // Forward T=1 to produce the *next* id, store it for the next call.
        let next_id = match self.advance(id) {
            Ok(n) => n,
            Err(e) => return Some(Err(e)),
        };
        self.next_input = Some(next_id);
        Some(Ok(id))
    }
}

impl GenerateIter<'_> {
    fn advance(&mut self, prev_id: u32) -> Result<u32> {
        let ids = Tensor::from_slice(&self.device, &[prev_id], [1, 1])?;
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let pos = (self.prefill_len + self.step - 1) as i32;
        let offset = Tensor::from_slice(&self.device, &[pos], [1])?;
        let logits = self
            .owner
            .model
            .forward(&ids, None, &offset, &mut self.owner.caches)?;
        argmax_last_position(&logits, self.owner.model.config().vocab_size)
    }
}

fn id_is_terminal(id: u32, eos_ids: &HashSet<u32>) -> bool {
    eos_ids.contains(&id)
}

/// Argmax over the last position of a `[1, T, vocab]` BF16 logits tensor.
fn argmax_last_position(logits: &Tensor, vocab: usize) -> Result<u32> {
    // logits is [1, T, vocab]; select the last position's row, argmax
    // over the vocab axis on the GPU (same command buffer as lm_head),
    // and read back the single index.
    let t = logits.shape().dims()[1];
    let last = logits.slice(&[0..1, t - 1..t, 0..vocab])?.copy()?; // [1, 1, vocab] dense
    let idx = last.argmax(2)?; // [1, 1] U32
    idx.eval()?;
    let _span = cider_press_runtime::profile::span("argmax");
    let ids = idx
        .cpu_slice::<u32>()
        .ok_or_else(|| Error::InvalidArgument("argmax: cpu_slice::<u32> returned None".into()))?;
    Ok(ids[0])
}
