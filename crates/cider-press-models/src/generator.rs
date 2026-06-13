//! Greedy-decode driver over [`Qwen2Model`] + [`KvCache`].
//!
//! Concrete-on-Qwen2Model for now; lifting to a `LanguageModel`
//! trait waits until a second architecture arrives. The iterator
//! yields `Result<u32>` ids; UTF-8 detokenization is the caller's
//! problem (compose `Tokenizer::decode_stream` on top).

use std::collections::{HashSet, VecDeque};

use cider_press_runtime::{DType, Device, KvCache, PendingEval, Tensor};

use crate::error::{Error, Result};
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

    /// Run one decode-step forward (`T = 1`, offset = the current
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
        let logits = self.model.forward(&ids, &offset, &mut self.caches)?;
        logits.profiled_eval()?;
        Ok(())
    }

    /// mlx_lm-faithful prefill: fill the cache with the first `T-1` tokens
    /// (eval'd to commit the KV writes), then build and return the **lazy**
    /// last-position logits (`[1, 1, vocab]`) from a `T=1` decode-style step
    /// over the final token. Resets the caches first; advances them to
    /// `input_ids.len()` as a side effect.
    ///
    /// `T == 1`: no cache-fill — the lone `T=1` `forward` is already the step.
    ///
    /// The `T=1` step routes through `sdpa_vector` + the qmv decode twin via
    /// the existing T-length dispatch selection, exactly as `mlx_lm`'s `_step`
    /// runs the final prompt token through the M=1 quantized matvec path.
    fn prefill_step_logits(&mut self, input_ids: &[u32]) -> Result<Tensor> {
        if input_ids.is_empty() {
            return Err(Error::InvalidArgument(
                "Generator prefill: input_ids must be non-empty".into(),
            ));
        }
        self.reset()?;
        let device = Device::shared()?;
        let t = input_ids.len();

        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let last_offset = if t > 1 {
            // Cache-fill over the first T-1 tokens at offset 0; eval to commit
            // the slice_update KV writes before the T=1 step reads them.
            let fill_ids = Tensor::from_slice(&device, &input_ids[..t - 1], [1, t - 1])?;
            let zero = Tensor::from_slice(&device, &[0i32], [1])?;
            let hidden = self.model.fill_cache(&fill_ids, &zero, &mut self.caches)?;
            hidden.eval()?;
            (t - 1) as i32
        } else {
            0
        };

        // Final token as a T=1 step at offset = number of cached tokens.
        let ids = Tensor::from_slice(&device, &[input_ids[t - 1]], [1, 1])?;
        let offset = Tensor::from_slice(&device, &[last_offset], [1])?;
        self.model.forward(&ids, &offset, &mut self.caches)
    }

    /// Profile the **prefill-path** forward: the cache-fill over the first
    /// `T-1` tokens (the composed/qmm + `sdpa_full` dispatches). The final
    /// `T=1` token is a decode step — see [`Self::profiled_decode_step`]. For a
    /// 1-token prompt there is no cache-fill, so the lone `T=1` step is
    /// profiled instead. Resets the caches; advances them as a side effect.
    ///
    /// # Errors
    /// `input_ids` empty, or device / forward / `profiled_eval` failure.
    #[doc(hidden)]
    pub fn profiled_prefill(&mut self, input_ids: &[u32]) -> Result<()> {
        if input_ids.is_empty() {
            return Err(Error::InvalidArgument(
                "Generator prefill: input_ids must be non-empty".into(),
            ));
        }
        self.reset()?;
        let device = Device::shared()?;
        let t = input_ids.len();
        let n = if t > 1 { t - 1 } else { 1 };
        let ids = Tensor::from_slice(&device, &input_ids[..n], [1, n])?;
        let offset = Tensor::from_slice(&device, &[0i32], [1])?;
        if t > 1 {
            let hidden = self.model.fill_cache(&ids, &offset, &mut self.caches)?;
            hidden.profiled_eval()?;
        } else {
            let logits = self.model.forward(&ids, &offset, &mut self.caches)?;
            logits.profiled_eval()?;
        }
        Ok(())
    }

    /// Run mlx_lm-faithful prefill — `fill_cache` over the first `T-1` tokens
    /// (eval'd) then a `T=1` step over the last token — and **synchronously**
    /// `eval()` the last-position `[1, 1, vocab]` logits, waiting for GPU
    /// completion. Two command-buffer commits, matching `mlx_lm`'s
    /// `mx.eval([c.state])` + `_step`. Apples-to-apples prefill wall-clock vs
    /// `scripts/profile_mlx_prefill.py`. Resets + advances the caches as a
    /// side effect — one-shot, not resumable.
    ///
    /// # Errors
    /// `input_ids` empty, or device / forward / eval failure.
    pub fn prefill_sync(&mut self, input_ids: &[u32]) -> Result<()> {
        let logits = self.prefill_step_logits(input_ids)?;
        logits.eval()?;
        Ok(())
    }

    /// Greedy-decode up to `max_new_tokens` ids from `input_ids`.
    ///
    /// Runs mlx_lm-faithful prefill (cache-fill over the first `T-1` tokens,
    /// then a `T=1` forward for the final token) then yields sampled ids one at a
    /// time. Iterator
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
    /// A failed speculative lookahead is deferred until earlier committed
    /// tokens have drained, so the yielded prefix is the same at every
    /// `inflight_depth`.
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
        // mlx_lm-faithful prefill: cache-fill over the first T-1 tokens
        // (eval'd inside the helper), then the last token as a T=1 step.
        // Only the last token's logits drive the first sampled id.
        let prefill_len = input_ids.len();
        let logits = self.prefill_step_logits(input_ids)?;
        let device = Device::shared()?;

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
            next_id_gpu: idx0,
            inflight,
            depth,
            committed: 0,
            yielded: 0,
            done: false,
            deferred_err: None,
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
    /// Most recent argmax tensor (`[1, 1]` U32), fed into the next forward.
    /// Persists across the in-flight queue draining (needed at depth 0,
    /// where the queue empties between commits), so it is the chain source
    /// rather than `inflight.back()`.
    next_id_gpu: Tensor,
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
    /// A speculative lookahead `advance()` failed. Stashed (not returned
    /// immediately) so already-committed earlier tokens still drain first —
    /// the yielded prefix must not depend on `depth`. Surfaced once the
    /// in-flight queue is empty; dropped if EOS lands before then (depth 0
    /// would never have computed the failing forward).
    deferred_err: Option<Error>,
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
        // (prior_update_committed) is satisfied because each dropped PendingEval
        // blocks until its command buffer completes, materializing the KvCache's
        // latest update before the slab is reused.
        if self.done {
            return None;
        }

        // Commit-ahead: keep `depth + 1` tokens in flight (the one we are
        // about to read plus `depth` ahead of it), bounded by max_new_tokens
        // (saturating: an absurd depth just means "commit everything").
        // Each eval_async here overlaps the prior token's running GPU work.
        // Chain from the most recently committed token's on-GPU argmax.
        while self.deferred_err.is_none()
            && self.inflight.len() < self.depth.saturating_add(1)
            && self.yielded + self.inflight.len() < self.max_new_tokens
        {
            let feed = self.next_id_gpu.clone();
            match self.advance(&feed) {
                Ok((pending, idx)) => {
                    self.next_id_gpu = idx.clone();
                    self.inflight.push_back((pending, idx));
                    self.committed += 1;
                }
                // Stash, don't surface: earlier committed tokens are still
                // valid, and a deeper pipeline must not hide tokens depth 0
                // would yield before failing. Stop committing new work.
                Err(e) => self.deferred_err = Some(e),
            }
        }

        // Drain the oldest committed token: wait its GPU work, read the id.
        let Some((pending, idx)) = self.inflight.pop_front() else {
            // Queue drained — surface a stashed lookahead error, once.
            self.done = true;
            return self.deferred_err.take().map(Err);
        };
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
            .forward(feed, &offset, &mut self.owner.caches)?;
        let vocab = self.owner.model.config().vocab_size;
        let idx = argmax_last_position_lazy(&logits, vocab)?;
        let pending = idx.eval_async()?;
        Ok((pending, idx))
    }
}

fn id_is_terminal(id: u32, eos_ids: &HashSet<u32>) -> bool {
    eos_ids.contains(&id)
}

/// Build the lazy argmax over a single-position `[1, 1, vocab]` BF16 logits
/// tensor, returning a `[1, 1]` U32 index tensor **without** evaluating it —
/// the caller commits it via `eval_async` and reads it back after waiting.
///
/// Both callers feed last-position-only logits: decode runs `forward` on a
/// `T = 1` token and prefill projects the LM head for the final position alone
/// (`Qwen2Model::forward_last`). `vocab` is accepted to validate that shape.
fn argmax_last_position_lazy(logits: &Tensor, vocab: usize) -> Result<Tensor> {
    // A real runtime check, not `debug_assert_eq!`: release builds (the perf
    // test/bench path) compile that out, which would let a non-`[1,1,vocab]`
    // tensor silently argmax the wrong position instead of failing cleanly.
    if logits.shape().dims() != [1, 1, vocab] {
        return Err(Error::InvalidArgument(format!(
            "argmax_last_position_lazy expects last-position-only logits [1, 1, {vocab}], got {:?}",
            logits.shape().dims(),
        )));
    }
    // argmax(2) over [1, 1, vocab] removes the axis -> [1, 1] U32, which is
    // exactly forward()'s [1, T] input-ids shape for the next token.
    Ok(logits.argmax(2)?)
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
