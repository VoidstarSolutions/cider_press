//! Greedy-decode driver over [`Qwen2Model`] + [`KvCache`].
//!
//! Concrete-on-Qwen2Model for now; lifting to a `LanguageModel`
//! trait waits until a second architecture arrives. The iterator
//! yields `Result<u32>` ids; UTF-8 detokenization is the caller's
//! problem (compose `Tokenizer::decode_stream` on top).

use std::collections::{HashSet, VecDeque};

use cider_press_runtime::{DType, Device, KvCache, PendingEval, Tensor};

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
    inflight_depth: usize,
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
            inflight_depth: 1,
        })
    }

    /// Set the decode pipeline lookahead distance: the number of decode
    /// forwards committed ahead of the token currently being read back. The
    /// target in-flight count is `depth + 1`. Default 1 — one token's GPU
    /// work overlaps the readback. Greedy decode is inherently depth-1
    /// (there is no second independent token to compute ahead), so values
    /// above 1 do not speed up greedy generation; the knob exists for the
    /// general async-eval infra and future speculative/batched paths. A
    /// depth of 0 disables lookahead (fully synchronous per token).
    ///
    /// The depth is snapshotted into the iterator when [`Self::generate`] is
    /// called, so changing it after `generate()` does not affect an already-
    /// returned iterator; it takes effect on the next `generate()` call.
    pub fn set_inflight_depth(&mut self, depth: usize) {
        self.inflight_depth = depth;
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

        let vocab = self.model.config().vocab_size;
        let idx0 = argmax_last_position_lazy(&logits, vocab)?;
        let pending0 = idx0.eval_async()?;

        let depth = self.inflight_depth;
        let mut inflight: VecDeque<(PendingEval, Tensor)> = VecDeque::new();
        inflight.push_back((pending0, idx0.clone()));

        Ok(GenerateIter {
            owner: self,
            device,
            prefill_len,
            inflight,
            depth,
            committed: 0,
            yielded: 0,
            done: false,
            max_new_tokens,
        })
    }
}

/// Iterator yielding `Result<u32>` sampled ids. Borrows `&mut Generator`.
///
/// Runs a commit-ahead pipeline: each `next()` first commits lookahead
/// tokens (up to `depth` ahead of the read position) so their CPU encode
/// overlaps the prior token's GPU work, then waits and reads the oldest
/// committed token. Ids are chained on-GPU — the argmax tensor feeds the
/// next forward's embedding gather directly, never round-tripping the CPU.
#[derive(Debug)]
pub struct GenerateIter<'a> {
    owner: &'a mut Generator,
    device: Device,
    prefill_len: usize,
    /// Committed-but-unread tokens in order: (handle, its argmax tensor).
    inflight: VecDeque<(PendingEval, Tensor)>,
    /// Lookahead distance; target in-flight count is `depth + 1`.
    depth: usize,
    /// Decode forwards committed so far (the `RoPE` / cache offset source).
    committed: usize,
    /// Tokens handed back so far (caps at `max_new_tokens`).
    yielded: usize,
    /// EOS seen or max hit — stop committing further lookaheads.
    done: bool,
    max_new_tokens: usize,
}

impl Iterator for GenerateIter<'_> {
    type Item = Result<u32>;

    fn next(&mut self) -> Option<Self::Item> {
        // Once terminal, the sequence is finished; any still-in-flight
        // lookahead (committed before we knew it was terminal — at most
        // `depth` wasted forwards) is discarded when the iterator drops,
        // its PendingEval blocking on completion before freeing buffers.
        // Those lookahead forwards also advanced KvCache.position; reusing
        // the same Generator for another generate() is safe only because
        // generate() calls reset() unconditionally, and reset()'s precondition
        // (prior_update_evaluated) is satisfied because each dropped PendingEval
        // blocks until its command buffer completes, materializing the KvCache's
        // latest update before the slab is reused.
        if self.done {
            return None;
        }

        // Commit-ahead: keep `depth + 1` tokens in flight (the one we are
        // about to read plus `depth` ahead of it), bounded by max_new_tokens.
        // Each eval_async here overlaps the prior token's running GPU work.
        // Chain from the most recently committed token's on-GPU argmax.
        while self.inflight.len() < self.depth + 1
            && self.yielded + self.inflight.len() < self.max_new_tokens
        {
            let feed = match self.inflight.back() {
                Some((_, idx)) => idx.clone(),
                None => break, // nothing committed to chain from
            };
            match self.advance(&feed) {
                Ok((pending, idx)) => {
                    self.inflight.push_back((pending, idx));
                    self.committed += 1;
                }
                Err(e) => {
                    self.done = true;
                    return Some(Err(e));
                }
            }
        }

        // Drain the oldest committed token: wait its GPU work, read the id.
        let (pending, idx) = self.inflight.pop_front()?;
        if let Err(e) = pending.wait() {
            self.done = true;
            return Some(Err(Error::Runtime(e)));
        }
        let id = match read_id(&idx) {
            Ok(id) => id,
            Err(e) => {
                self.done = true;
                return Some(Err(e));
            }
        };
        self.yielded += 1;
        if id_is_terminal(id, &self.owner.eos_ids) || self.yielded >= self.max_new_tokens {
            self.done = true;
        }
        Some(Ok(id))
    }
}

impl GenerateIter<'_> {
    /// One decode forward from the on-GPU `feed` id tensor: build the graph,
    /// commit it async, and return the handle plus the lazy next-argmax
    /// tensor. Does not block — readback happens later in `next()`.
    fn advance(&mut self, feed: &Tensor) -> Result<(PendingEval, Tensor)> {
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let pos = (self.prefill_len + self.committed) as i32;
        let offset = Tensor::from_slice(&self.device, &[pos], [1])?;
        let logits = self
            .owner
            .model
            .forward(feed, None, &offset, &mut self.owner.caches)?;
        let vocab = self.owner.model.config().vocab_size;
        let idx = argmax_last_position_lazy(&logits, vocab)?;
        let pending = idx.eval_async()?;
        Ok((pending, idx))
    }
}

fn id_is_terminal(id: u32, eos_ids: &HashSet<u32>) -> bool {
    eos_ids.contains(&id)
}

/// Build the lazy argmax over the last position of a `[1, T, vocab]` BF16
/// logits tensor, returning a `[1, 1]` U32 index tensor **without**
/// evaluating it — the caller commits it via `eval_async` and reads it back
/// after waiting.
fn argmax_last_position_lazy(logits: &Tensor, vocab: usize) -> Result<Tensor> {
    let t = logits.shape().dims()[1];
    let last = if t == 1 {
        logits.clone()
    } else {
        logits.slice(&[0..1, t - 1..t, 0..vocab])?.copy()?
    };
    // argmax(2) over [1, 1, vocab] removes the axis -> [1, 1] U32, which is
    // exactly forward()'s [1, T] input-ids shape for the next token.
    Ok(last.argmax(2)?)
}

/// Read a single token id from an argmax tensor whose `PendingEval` has
/// already been waited.
fn read_id(idx: &Tensor) -> Result<u32> {
    let _span = cider_press_runtime::profile::span("argmax.readback");
    let ids = idx.cpu_slice::<u32>().ok_or_else(|| {
        Error::InvalidArgument(
            "argmax: result not materialised or wrong dtype (expected U32)".into(),
        )
    })?;
    Ok(ids[0])
}
