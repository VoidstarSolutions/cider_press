# KV cache

Autoregressive decode generates one token at a time, and each new token attends
back over every token before it. The naïve way to do that is to re-run the whole
prefix through attention every step — but the keys and values for the prefix
never change once computed, so recomputing them is pure waste. The KV cache is
the data structure that stores them: each step computes K/V for *only* the new
token and reads the rest back from the cache. This page is about what the cache
holds, why decode is O(1) in history because of it, and how cider-press makes the
write a lazy in-graph op rather than a host copy.

## Concept

Inside [attention](./attention.md), a query at position `t` scores against the
keys at every position `0..=t` and mixes their values by those scores. The keys
and values at positions `0..t` depend only on those earlier tokens and their
weights — they are *identical* on every later step. So computing them once and
keeping them is strictly better than recomputing: without a cache, step `t` would
re-project all `t` prior tokens to K/V (an O(T) cost that climbs every step,
O(T²) over a sequence); with a cache, step `t` projects only the *one* new token's
K/V and reads the `t` prior rows straight from storage. Each decode step becomes
**O(1) in history** — the only per-token work that scales is the attention read
over the cached prefix, not a re-projection of it.

The cache is filled in two phases, mirroring the two phases of generation:

- **Prefill** runs the whole prompt through the model at once, so it writes all
  `T` prompt rows of K/V into the cache in a single update — one wide append.
- **Decode** then generates one token per step, so each step **appends a single
  row** of K/V at the current position and reads back the populated prefix.

What the cache stores is exactly the per-layer keys and values — one cache per
decoder layer — laid out per KV head. Because the keys and values are what grows
with context, they are also what [grouped-query attention](./attention.md)
shrinks: GQA narrows the *number* of KV heads, so the cache is the structure GQA
exists to make cheaper.

## In cider-press

[`KvCache`](../../crates/cider-press-runtime/src/kv_cache.rs) is the **first
runtime abstraction that is not a `Tensor`**. A `Tensor` leaf is set-once — its
storage fills exactly once, modeled by an `OnceLock<LeafStorage>` — but a cache is
*set-by-position-monotonically*: row after row, the same slab written at growing
offsets. That doesn't fit set-once, so the cache is its own type. See
[the execution model](./execution-model.md) for the lazy graph and `eval()`
boundary this builds on, and why the cache sits beside `Tensor` rather than inside
it.

Each cache holds a pair of pre-allocated slabs shaped
`[max_tokens, n_kv_heads, head_dim]` (T-outer, so an append is a contiguous row
write). Writes are **lazy in-graph `slice_update`** ops, not eager host
`memcpy`s: [`KvCache::update`](../../crates/cider-press-runtime/src/kv_cache.rs)
records one `slice_update` per slab — writing this step's K/V rows into the slab
at the current row offset — and returns without evaluating. The write lands in the
**same command buffer as the SDPA read that consumes it**, which is what collapses
a decode step to a single GPU submission. `keys_view`/`values_view` then expose
the populated prefix `[0..position]` as zero-copy `Tensor` slices of the slab,
fed straight into [attention](./attention.md).

Two invariants keep the lazy write sound, and both **fail loud** rather than
silently corrupt:

- **Drop K/V views before the next `update`.** The views alias the slab's
  `MTLBuffer`; the cache is append-only (each `update` writes a fresh, disjoint
  row range and `position` only grows), so an outstanding view always reads a
  still-valid prefix — but callers drop their K/V views before appending again.
- **Successive `update`s need an `eval` between them.** Each `update` bases its
  `slice_update` directly on the **slab leaf**, keeping only the latest op (not
  chaining off the prior one — see Performance). An un-committed prior write would
  therefore be dropped, so a second `update` before the first is committed returns
  `Error::InvalidArgument`. Decode commits each token; prefill is a single wide
  update. (A still-in-flight async commit qualifies — the write is encoded and
  submitted, so it can neither be dropped nor replayed.)

The slabs are **zero-filled** at construction. Reads only ever target the
populated prefix, so under serial dispatch the fill is invisible; but concurrent
dispatch over hazard-untracked buffers can lose the automatic write-before-read
ordering at a fence-chain gap (e.g. the first step before any write has
committed), and a deterministically-zero slab makes a premature read of an
unwritten row **benign** rather than garbage. The cache is BF16-only — the slab
write path (`slice_update`) supports no other dtype.

**Model variations.**

- **GQA / MQA shrink the KV-head count.** The slab's `n_kv_heads` is the GQA KV
  count, not the query-head count — Qwen2.5-0.5B caches 2 KV heads against 14
  query heads (a 7× group), so the cache holds ⅐ the keys/values a full
  multi-head model would. MQA is the extreme (`n_kv_heads = 1`).
- **Sliding-window attention caps the slab length.** Families that bound each
  position's lookback to a fixed window (Mistral, Gemma2 alternating layers) need
  only a window-sized slab, not one sized to the full context. Qwen2.5-0.5B is
  full causal, so its slab is sized to the context window.
- **Quantized KV cache** — storing K/V at lower precision to fit longer contexts
  in the same memory — is a future variation; today the slab is dense BF16.

## Performance

The cache write is where the decode pipeline's biggest single enabler lives, and
it is a structural one, not a kernel one. Each `update` bases its `SliceUpdate`
op **directly on the slab leaf** rather than chaining it off the prior step's
op-tensor. That sounds like a detail, but it is what unblocked the cross-token
buffer pool: with the old chaining form, `keys_latest`/`values_latest` held an
`Arc` to the previous step's op, which transitively **pinned that step's entire
projection graph** for the whole decode, so the pool could never recycle per-step
scratch. Slab-basing lets each step's graph free once the next `update`
reassigns those fields. The pool was inert on its own (~3% hit) until this fix
landed; **together they hit ~98% pool hit-rate**, and the cross-token pool took
decode **~120 → ~210 tok/s** (~1.75×), cutting the `tensor.eval` CPU-encode share
from ~38% to ~11% and resolving a +32% peak-RSS regression (~1192 → ~900 MiB)
(see the [execution model](./execution-model.md) page's decode-pipelining and
buffer-pool notes).

The lazy in-graph write itself — landing the K/V `SliceUpdate` in the same
command buffer as the consuming SDPA read, rather than an eager host `memcpy` +
forced `eval` — is the change that first collapsed each decode step to **one
command buffer per token** (~1.6×), the foundation the later pipelining built on
(see the [execution model](./execution-model.md) page). The zero-filled slabs cost one host
memset at construction over shared storage, and buy deterministic-rather-than-
garbage reads under the concurrent dispatch the async pipeline relies on.

## Open levers

- **Chunked prefill via `fill_cache`.** Today prefill writes all `T` prompt rows
  in one update; the MLX-faithful T−1-fill + T=1-step split was measured a
  pessimization for the short-prompt regime and reverted, but
  [`fill_cache`](../../crates/cider-press-models/src/qwen2/model.rs) (and its
  equivalence test) was **kept as the building block** for chunked prefill — the
  one regime where the split actually helps: long prompts under bounded memory,
  where filling the cache in chunks caps peak working set. This is the
  prefill-concurrency direction the [execution model](./execution-model.md)
  points at, not a decode-kernel lever.
- **Quantized KV cache** for longer contexts — storing the slab at lower
  precision so a given memory budget holds more tokens. The slab is dense BF16
  today (the ~30 MiB pre-allocated to the full context window is the main
  cache-side memory cost at scale, see the [execution model](./execution-model.md)
  page's buffer-pool note); quantizing
  it is the lever for genuinely long sessions.
