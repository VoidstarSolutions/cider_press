//! Greedy-decode driver over a [`LanguageModel`] + its per-layer caches.
//!
//! Generic over the model: [`Generator<M>`] drives any
//! [`LanguageModel`] (currently [`Qwen2Model`](crate::qwen2::Qwen2Model) with
//! [`KvCache`](cider_press_runtime::KvCache) and
//! [`Qwen35Model`](crate::qwen3_5::Qwen35Model) with
//! [`Qwen35LayerCache`](crate::qwen3_5::Qwen35LayerCache)) — the async-
//! pipelined decode loop is cache-type-agnostic. The iterator yields
//! `Result<u32>` ids; UTF-8 detokenization is the caller's problem (compose
//! `Tokenizer::decode_stream` on top).

#![doc = include_str!("../../../docs/inference/sampling.md")]

use std::collections::{HashSet, VecDeque};

use cider_press_runtime::{Device, PendingEval, Tensor};

use crate::error::{Error, Result};
use crate::language_model::LanguageModel;

/// Owns a model + pre-allocated per-layer caches + the EOS set. One instance
/// drives one or more `generate` calls; caches are reset on every
/// call so previous runs don't leak into the next prompt.
#[derive(Debug)]
pub struct Generator<M: LanguageModel> {
    model: M,
    caches: Vec<M::Cache>,
    eos_ids: HashSet<u32>,
    context_window: usize,
    inflight_depth: usize,
    /// Tokens fed through the caches so far — the `RoPE` / cache offset of
    /// the next forward. Advanced by every path that feeds tokens.
    position: usize,
}

impl<M: LanguageModel> Generator<M> {
    /// Construct a Generator. Pre-allocates one cache per layer
    /// sized to `context_window` tokens.
    ///
    /// # Errors
    /// - `context_window == 0`
    /// - `eos_ids.is_empty()` — without an EOS the loop has no
    ///   natural terminator besides `max_new_tokens`; we reject so a
    ///   caller can't accidentally rely on that.
    pub fn new(model: M, context_window: usize, eos_ids: HashSet<u32>) -> Result<Self> {
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
        let caches = model.make_caches(context_window, &device)?;
        Ok(Self {
            model,
            caches,
            eos_ids,
            context_window,
            inflight_depth: 1,
            position: 0,
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

    /// Reset all per-layer caches to position 0. Called automatically at the
    /// start of [`Self::generate`].
    ///
    /// # Errors
    /// Propagates the cache reset (e.g. for `KvCache`, if a cache's last update
    /// has not been evaluated — it must be, to keep outstanding K/V views from
    /// replaying stale writes into the reused slab).
    pub fn reset(&mut self) -> Result<()> {
        self.model.reset_caches(&mut self.caches)?;
        self.position = 0;
        Ok(())
    }

    /// Shared prefill graph builder for the profiling/timing entry points:
    /// resets the caches, then prefills `input_ids` from offset 0 via
    /// [`prefill_last_logits`] (chunked when the model asks for it). Returns
    /// the lazy **last-position** logits (`[1, 1, vocab]`) — the blocks still
    /// run over all `T` tokens, but only the final token is projected through
    /// the LM head, as greedy generation (and `mlx_lm`'s prefill) does.
    /// Advances the caches to `input_ids.len()` as a side effect.
    fn prefill_logits(&mut self, input_ids: &[u32]) -> Result<Tensor> {
        if input_ids.is_empty() {
            return Err(Error::InvalidArgument(
                "Generator prefill: input_ids must be non-empty".into(),
            ));
        }
        self.reset()?;
        let device = Device::shared()?;
        let chunk_len = self.model.prefill_chunk_len();
        let logits =
            prefill_last_logits(&self.model, &mut self.caches, &device, input_ids, chunk_len)?;
        self.position = input_ids.len();
        Ok(logits)
    }

    /// Run a **prefill** (`T = input_ids.len()`, offset = 0, causal via the
    /// fused sdpa kernel; chunked when the model asks for it, in which case
    /// earlier chunks fill the caches unprofiled and only the final chunk's
    /// forward is profiled) through the profiling-only
    /// [`Tensor::profiled_eval`], recording per-OpKind GPU time into the
    /// `profile` GPU store.
    ///
    /// The T>1 analog of [`Self::profiled_decode_step`]. Resets the caches
    /// first; advances them to `input_ids.len()` as a side effect — intended
    /// for one-shot profiling, not resumable generation.
    ///
    /// # Errors
    /// `input_ids` empty, or device / forward / `profiled_eval` failure.
    #[doc(hidden)]
    pub fn profiled_prefill(&mut self, input_ids: &[u32]) -> Result<()> {
        let logits = self.prefill_logits(input_ids)?;
        logits.profiled_eval()?;
        Ok(())
    }

    /// Run a prefill (`T = input_ids.len()`, offset 0, causal via the fused
    /// sdpa kernel; chunked when the model asks for it) and **synchronously**
    /// `eval()` the last-position `[1, 1, vocab]` logits (see `prefill_logits`),
    /// waiting for GPU completion. Mirrors `mlx_lm`'s prefill, which evals
    /// only the KV cache for the prompt and projects the LM head for a single
    /// token — so this is an apples-to-apples prefill wall-clock measurement
    /// (the production `generate()` path uses `eval_async` and does not wait).
    /// Resets + advances the caches as a side effect — one-shot, not resumable.
    ///
    /// # Errors
    /// `input_ids` empty, or device / forward / eval failure.
    pub fn prefill_sync(&mut self, input_ids: &[u32]) -> Result<()> {
        let logits = self.prefill_logits(input_ids)?;
        logits.eval()?;
        Ok(())
    }

    /// Greedy-decode up to `max_new_tokens` ids from `input_ids`.
    ///
    /// Runs prefill (one forward at `T = input_ids.len()`, offset=0,
    /// causal via the fused sdpa kernel — or, for a model with a
    /// [`prefill_chunk_len`](LanguageModel::prefill_chunk_len), synchronously
    /// evaluated chunks followed by that forward over the last chunk) then
    /// yields sampled ids one at a time. Iterator
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
    ) -> Result<GenerateIter<'_, M>> {
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
        let chunk_len = self.model.prefill_chunk_len();
        let logits =
            prefill_last_logits(&self.model, &mut self.caches, &device, input_ids, chunk_len)?;
        self.position = prefill_len;

        let vocab = self.model.vocab_size();
        let idx0 = argmax_last_position_lazy(&logits, vocab)?;
        let pending0 = idx0.eval_async()?;

        let depth = self.inflight_depth;
        let mut inflight: VecDeque<(PendingEval, Tensor)> = VecDeque::new();
        inflight.push_back((pending0, idx0.clone()));

        Ok(GenerateIter {
            owner: self,
            device,
            next_id_gpu: idx0,
            inflight,
            depth,
            yielded: 0,
            done: false,
            deferred_err: None,
            max_new_tokens,
        })
    }

    /// Run one decode-step forward (`T = 1`, offset = the generator's
    /// committed position) through the profiling-only
    /// [`Tensor::profiled_eval`], recording per-OpKind GPU time into the
    /// `profile` GPU store.
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
        let ids = Tensor::from_slice(&device, &[last_id], [1, 1])?;
        let offset = position_tensor(&device, self.position)?;
        let logits = self.model.forward_last(&ids, &offset, &mut self.caches)?;
        self.position += 1;
        logits.profiled_eval()?;
        Ok(())
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
pub struct GenerateIter<'a, M: LanguageModel> {
    owner: &'a mut Generator<M>,
    device: Device,
    /// Most recent argmax tensor (`[1, 1]` U32), fed into the next forward.
    /// Persists across the in-flight queue draining (needed at depth 0,
    /// where the queue empties between commits), so it is the chain source
    /// rather than `inflight.back()`.
    next_id_gpu: Tensor,
    /// Committed-but-unread tokens in order: (handle, its argmax tensor).
    inflight: VecDeque<(PendingEval, Tensor)>,
    /// Lookahead distance; target in-flight count is `depth + 1`.
    depth: usize,
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

impl<M: LanguageModel> Iterator for GenerateIter<'_, M> {
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

impl<M: LanguageModel> GenerateIter<'_, M> {
    /// One decode forward from the on-GPU `feed` id tensor: build the graph,
    /// commit it async, and return the handle plus the lazy next-argmax
    /// tensor. Does not block — readback happens later in `next()`.
    ///
    /// The step is `T = 1`, so `forward_last` projects the whole (single-row)
    /// sequence — identical to a full `forward` but routed through the one
    /// forward entry point the trait exposes.
    fn advance(&mut self, feed: &Tensor) -> Result<(PendingEval, Tensor)> {
        let offset = position_tensor(&self.device, self.owner.position)?;
        let logits = self
            .owner
            .model
            .forward_last(feed, &offset, &mut self.owner.caches)?;
        self.owner.position += 1;
        let vocab = self.owner.model.vocab_size();
        let idx = argmax_last_position_lazy(&logits, vocab)?;
        let pending = idx.eval_async()?;
        Ok((pending, idx))
    }
}

/// Prefill `input_ids` into `caches` (which must be at position 0) and return
/// the lazy last-position logits `[1, 1, vocab]`.
///
/// With `chunk_len = Some(c)` and a prompt longer than `c`, every chunk but
/// the last runs through [`LanguageModel::fill_cache`] and is evaluated
/// synchronously before the next is built, so peak memory tracks `c` rather
/// than the prompt length. The last chunk (a full `c` tokens when the prompt is
/// an exact multiple) goes through [`LanguageModel::forward_last`] like an
/// unchunked prefill. `None`, `Some(0)`, or a prompt of at most `c` tokens is a
/// single `forward_last` over the whole prompt. Causality comes from the fused
/// sdpa kernel's `do_causal` — no materialized mask.
fn prefill_last_logits<M: LanguageModel>(
    model: &M,
    caches: &mut [M::Cache],
    device: &Device,
    input_ids: &[u32],
    chunk_len: Option<usize>,
) -> Result<Tensor> {
    let offset_tensor = |start: usize| position_tensor(device, start);
    let mut start = 0;
    if let Some(chunk) = chunk_len.filter(|&c| c > 0) {
        while input_ids.len() - start > chunk {
            let ids = Tensor::from_slice(device, &input_ids[start..start + chunk], [1, chunk])?;
            model
                .fill_cache(&ids, &offset_tensor(start)?, caches)?
                .eval()?;
            start += chunk;
        }
    }
    let tail = &input_ids[start..];
    let ids = Tensor::from_slice(device, tail, [1, tail.len()])?;
    model.forward_last(&ids, &offset_tensor(start)?, caches)
}

fn position_tensor(device: &Device, position: usize) -> Result<Tensor> {
    let position = i32::try_from(position)
        .map_err(|_| Error::InvalidArgument(format!("offset {position} overflows i32")))?;
    Ok(Tensor::from_slice(device, &[position], [1])?)
}

fn id_is_terminal(id: u32, eos_ids: &HashSet<u32>) -> bool {
    eos_ids.contains(&id)
}

/// Build the lazy argmax over a single-position `[1, 1, vocab]` BF16 logits
/// tensor, returning a `[1, 1]` U32 index tensor **without** evaluating it —
/// the caller commits it via `eval_async` and reads it back after waiting.
///
/// Both callers feed last-position-only logits: decode runs a `T = 1` token
/// and prefill projects the LM head for the final position alone
/// ([`LanguageModel::forward_last`]). `vocab` is accepted to validate that shape.
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

#[cfg(test)]
mod tests {
    use half::bf16;
    use safetensors::SafeTensors;

    use super::*;
    use crate::qwen3_5::test_support::{Fill, TINY_CONFIG_JSON, synthesize_archive_bytes_with};
    use crate::qwen3_5::{Qwen35Config, Qwen35LayerCache, Qwen35Model, load_qwen35_weights};

    fn tiny_seeded_qwen35(device: &Device) -> Qwen35Model {
        let config = Qwen35Config::from_json_bytes(TINY_CONFIG_JSON.as_bytes()).expect("config");
        let bytes =
            synthesize_archive_bytes_with(&config, false, Fill::Seeded(0x5eed), |_, _| None);
        let archive = SafeTensors::deserialize(&bytes).expect("deserialize");
        let weights = load_qwen35_weights(&archive, &config, device).expect("load");
        Qwen35Model::from_weights(weights, config).expect("model")
    }

    fn host<T: cider_press_runtime::Scalar>(t: &Tensor) -> Vec<T> {
        let dense = t.copy().expect("copy");
        dense.eval().expect("eval");
        dense.cpu_to_vec::<T>().expect("host read")
    }

    /// Every piece of per-layer state as `(position, values widened to f32)`;
    /// GDN states carry position 0.
    type CacheSnapshot = Vec<(usize, Vec<f32>)>;

    fn cache_snapshot(caches: &[Qwen35LayerCache]) -> CacheSnapshot {
        let widen = |v: Vec<bf16>| v.into_iter().map(bf16::to_f32).collect::<Vec<_>>();
        caches
            .iter()
            .flat_map(|cache| match cache {
                Qwen35LayerCache::Full(kv) => vec![
                    (kv.position(), widen(host(&kv.keys_view()))),
                    (kv.position(), widen(host(&kv.values_view()))),
                ],
                Qwen35LayerCache::Linear(st) => vec![
                    (0, widen(host(st.conv_state()))),
                    (0, host::<f32>(st.recurrent_state())),
                ],
            })
            .collect()
    }

    /// Prefill `ids` with and without chunking from fresh caches; return the
    /// last-position logits and the resulting cache state of each run.
    fn prefill_both_ways(
        model: &Qwen35Model,
        device: &Device,
        ids: &[u32],
        chunk_len: usize,
    ) -> [(Vec<f32>, CacheSnapshot); 2] {
        [None, Some(chunk_len)].map(|chunk| {
            let mut caches = model.make_caches(32, device).expect("caches");
            let logits =
                prefill_last_logits(model, &mut caches, device, ids, chunk).expect("prefill");
            assert_eq!(logits.shape().dims(), &[1, 1, model.vocab_size()]);
            let logits: Vec<f32> = host::<bf16>(&logits)
                .into_iter()
                .map(bf16::to_f32)
                .collect();
            (logits, cache_snapshot(&caches))
        })
    }

    /// Evaluating the `fill_cache` handle alone must materialize every piece of
    /// state the next chunk reads; anything left lazy would be recomputed (or,
    /// for a `KvCache` write, dropped) by the next chunk's graph.
    #[test]
    fn fill_cache_eval_commits_every_cache_state() {
        let device = Device::shared().expect("device");
        let model = tiny_seeded_qwen35(&device);
        let mut caches = model.make_caches(32, &device).expect("caches");
        let ids = Tensor::from_slice(&device, &[3u32, 1, 4, 1, 5], [1, 5]).expect("ids");
        let offset = Tensor::from_slice(&device, &[0i32], [1]).expect("offset");
        model
            .fill_cache(&ids, &offset, &mut caches)
            .expect("fill_cache")
            .eval()
            .expect("eval");
        for (i, cache) in caches.iter().enumerate() {
            let states = match cache {
                Qwen35LayerCache::Full(kv) => [kv.keys_view(), kv.values_view()],
                Qwen35LayerCache::Linear(st) => {
                    [st.conv_state().clone(), st.recurrent_state().clone()]
                }
            };
            for state in &states {
                assert!(state.is_materialized(), "layer {i}: state left lazy");
            }
        }
    }

    #[test]
    fn chunked_prefill_matches_single_forward() {
        let device = Device::shared().expect("device");
        let model = tiny_seeded_qwen35(&device);
        // 11 = 4 + 4 + 3 (ragged last chunk); 12 = 4 + 4 + 4 (exact multiple:
        // the last chunk is a full chunk through forward_last).
        for len in [11u32, 12] {
            let ids: Vec<u32> = (0..len).map(|i| (i * 37 + 5) % 256).collect();
            let [(plain_logits, plain_state), (chunked_logits, chunked_state)] =
                prefill_both_ways(&model, &device, &ids, 4);

            let spread = plain_logits
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max)
                - plain_logits.iter().copied().fold(f32::INFINITY, f32::min);
            assert!(
                spread > 0.0,
                "len {len}: degenerate logits, the test proves nothing"
            );

            assert_eq!(
                chunked_logits, plain_logits,
                "len {len}: last-position logits"
            );
            assert_eq!(chunked_state.len(), plain_state.len());
            for (i, (c, p)) in chunked_state.iter().zip(&plain_state).enumerate() {
                assert_eq!(c.0, p.0, "len {len}: state {i} position");
                assert_eq!(c.1, p.1, "len {len}: state {i} contents");
            }
        }
    }

    #[test]
    fn position_tracks_prefill_and_reset() {
        let device = Device::shared().expect("device");
        let mut generator =
            Generator::new(tiny_seeded_qwen35(&device), 32, HashSet::from([u32::MAX]))
                .expect("generator");
        let ids: Vec<u32> = (0..6).map(|i| (i * 37 + 5) % 256).collect();

        generator.prefill_sync(&ids).expect("prefill");
        assert_eq!(generator.position, 6);

        generator.reset().expect("reset");
        assert_eq!(generator.position, 0);
        // max_new_tokens = 1 yields the prefill argmax with no decode forward.
        assert_eq!(generator.generate(&ids, 1).expect("generate").count(), 1);
        assert_eq!(generator.position, 6);

        generator.reset().expect("reset");
        assert_eq!(generator.position, 0);
    }
}
