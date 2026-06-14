# Attention

Attention is the only operation in a transformer that lets one token read
another. Everything else — the norms, the projections, the MLP — works on each
position independently; attention is where information moves *across* positions.
This page is about the scaled dot-product attention at the heart of each block:
what it computes, how grouped-query attention shrinks its cost, and why the same
math routes to two completely different kernels in decode versus prefill.

## Concept

Attention scores every position against every other and mixes their values by
those scores. For a single query, key, and value set it is one expression:

```text
attention(Q, K, V) = softmax(QKᵀ / √d + mask) · V
```

Read left to right: `QKᵀ` is every query's dot product against every key — a
`[T, T]` score matrix saying how much each position wants to attend to each
other position. Dividing by `√d` (the per-head dimension) keeps those scores
from growing with `d` and saturating the softmax. The **mask** is what makes the
model autoregressive: a position may attend to itself and everything *before*
it, never after, so the mask sets the upper triangle to `-∞` before the softmax
zeroes it out — this is **causal masking**. The softmax turns each query's row of
scores into a probability distribution, and multiplying by `V` takes the
score-weighted average of the value vectors. The output is a new vector per
position, each one a blend of the positions it was allowed to see.

**Multi-head** attention runs that whole computation `H` times in parallel on
slices of the model dimension — each head gets its own `Q`, `K`, `V` projection
and learns to attend on a different relationship (syntax, coreference, position),
and their outputs are concatenated. `H` independent `[T, T]` attentions, not one
big one.

**Grouped-query attention (GQA)** is the cost lever. Naïvely each of the `H`
query heads has its own key/value head, but the keys and values are what the
[KV cache](./kv-cache.md) stores, and at long context that cache dominates
memory and bandwidth. GQA keeps the full `H_q` query heads but uses **fewer**
key/value heads `H_kv`, with each KV head shared (broadcast) across a group of
`H_q / H_kv` query heads. Qwen2.5-0.5B runs `H_q = 14` query heads against
`H_kv = 2` KV heads — a 7× group — so the cache holds ⅐ the keys and values a
full multi-head model would, at a small quality cost. The query side is
untouched; only the KV side is narrowed.

The shape of the computation differs sharply between the two phases of
generation, and this is what splits the implementation:

- **Prefill** processes the whole prompt at once, so `Q` is `[H_q, T, d]` with
  `T` > 1 — a genuine matrix, the full `[T, T]` causal score matrix, tiled like
  any GEMM.
- **Decode** generates one token at a time, so the new `Q` is a single row,
  `[H_q, 1, d]` — `T_q = 1`. There is only one query against the *whole* cached
  history of keys/values; the score "matrix" is a vector, and the op is
  bandwidth-bound on streaming the cache, not compute-bound on a GEMM.

Same softmax-of-scores math; two different performance regimes, and (below) two
different kernels.

## In cider-press

[`Attention`](../../crates/cider-press-models/src/qwen2/attention.rs) folds the
whole per-block attention sub-layer into one `forward`: it projects the residual
stream to `Q`/`K`/`V` (the four [quantized matmul](./quantized-matmul.md)
projections), lays each out per head, applies [`rope`](./rope.md) to `Q` and `K`,
writes the new `K`/`V` into the [`KvCache`](./kv-cache.md) and reads back the
populated prefix, runs `sdpa`, then collapses the heads and projects out through
`o_proj`. The per-head layout, RoPE, the strided cache write, and the strided
cache read are all done as **views** — `reshape`/`permute`/`broadcast_to` over
the same storage, not materialized copies (the head dimension stays unit-stride,
which is all the kernels require).

The actual attention math is a single [`Tensor::sdpa`] call, and the runtime
routes it by query length at `eval()`:

- **Decode** (`T_q == 1`) → the **vector** kernel
  [`sdpa_vector_only.metal`](../../kernels-mlx/mlx/backend/metal/kernels/sdpa_vector_only.metal),
  one threadgroup per (batch·query-head), reading `K`/`V` strided in place. No
  mask, no causal, no materialized scores — just the single query against the
  cached history.
- **Prefill** (`T_q > 1`) → the fused **steel** kernel
  [`steel_attention.metal`](../../kernels-mlx/mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.metal)
  (`sdpa_full`), which tiles the `QKᵀ`/scale/softmax/`·V` chain into one
  dispatch and applies causality via a `do_causal` function constant — no
  materialized mask tensor.

Both are dispatched through
[`sdpa.rs`](../../crates/cider-press-kernels/src/kernels/sdpa.rs), with the
softmax inside the steel kernel mirroring vendored
[`softmax.metal`](../../kernels-mlx/mlx/backend/metal/kernels/softmax.metal). In
**both** paths the GQA broadcast is handled **in-kernel** via a `gqa_factor`
(each query head reads KV head `q_head / gqa_factor`), so the 7× narrower KV
heads are never expanded into materialized `K`/`V` copies. This mirrors MLX's
single `ScaledDotProductAttention` primitive, which routes the same way.

**Model variations.**

- **MHA / GQA / MQA** — the `H_q : H_kv` ratio. Full multi-head is `H_kv = H_q`;
  multi-query (MQA) is the extreme `H_kv = 1` (one shared KV head). Qwen2.5-0.5B
  is GQA at 14:2. The `gqa_factor` plumbing covers all three with no kernel
  change.
- **Sliding-window vs. full causal** — some families (Mistral, Gemma2 on
  alternate layers) cap each position's lookback to a fixed window rather than
  the full causal prefix. Qwen2.5-0.5B is full causal.
- **ALiBi vs. RoPE positioning** — where positional information enters. cider
  applies [RoPE](./rope.md) to `Q`/`K` before attention; ALiBi instead adds a
  distance-based bias *into the scores*. Qwen2.5 is RoPE.
- **QK-norm** — some families (e.g. newer Qwen, Gemma2) normalize the query and
  key projections before scoring. Qwen2.5-0.5B does not.

See [models/qwen2.5.md](./models/qwen2.5.md) for Qwen2.5's concrete head counts
and the RoPE/norm choices.

## Performance

The prefill attention story is the **fused steel `sdpa_full`** landing. The old
prefill path was a composed chain — per-layer GQA-broadcast `K`/`V` copies, a
`Kᵀ` transpose copy, then separate `QKᵀ`/scale/mask/softmax/`·V` dispatches.
That whole chain is deleted, replaced by one `sdpa` dispatch per layer (the new
`gpu.sdpa`, 24 = 1/layer, subsumes the 48 `gpu.matmul` score/value matmuls and
`gpu.softmax`). Synchronous prefill went **10.19 → ~9.3 ms**, gap vs MLX
**1.30× → ~1.19×**; decode unchanged. The realized **~0.86 ms** win tracks the
eliminated **copy traffic** (the 3/layer score-chain copies, `gpu.copy`
266 → 194) — the `QKᵀ`/softmax/`·V` *compute* did not vanish, it moved *into*
the kernel.

The next lever, eliminating the **KV-cache-read copies** (the strided cache view
fed straight to the kernel instead of `copy()`-ing it contiguous, `gpu.copy`
194 → 146), then **measured a no-op**: those tiny `[1, H_kv=2, T, 64]` copies
are off the qmm-bound critical path (all copies together ~4.8% of prefill GPU vs
quantized matmul's ~83%) and overlap the qmm rather than serializing behind it.
The change was kept (correct, less traffic, matches MLX's strided-input
behavior) but the wall-clock did not move — the copy lever is **closed**.

**There was no real prefill gap.** Chased to its end, the "gap" dissolved rather
than closing: after the last-position LM-head fix, cider's structure-for-
structure prefill is **~7.76 ms**, and a same-session re-measure exposed the old
MLX "~7.27 ms" reference as a stale single-forward proxy. Against MLX's *real*
`generate_step` prefill — which evals the KV-cache state and defers the head —
MLX measures **8.98 ms**, so cider's prefill already **leads**. The residual was
baseline scheduling variance (~7% run-to-run on this prompt), not op sequence,
encode overhead, or chunking.

On the decode side, the T=1 attention permutes (per-head layout, cache read,
head collapse) are **zero-copy views**: the S1 copy-elimination recognized that
at `T_q = 1` every permute only moves the size-1 `T` axis, so the result is
already contiguous modulo unit dims and needs no materialization. That dropped
`gpu.copy` from **146 → 2 dispatches per token**, contributing to the decode
lead over `mlx_lm`.

## Open levers

- **Prefill concurrency, not fewer dispatches.** Prefill runs synchronously per
  `eval()` at under 50% GPU occupancy. What once looked like a ~2.2 ms
  dispatch-induced bubble turned out to be **real data-dependency latency** —
  the per-output fence-map experiment proved cider serializes exactly the
  fences the dependency chain already requires, so it is not over-serializing
  independent work. Every dispatch-*count* lever was refuted: cadence
  (fewer/bigger command buffers), copy-count reduction, and the fence map all
  measured no-ops. Beating the prefill floor needs genuine **concurrency**
  (overlapping independent work, e.g. the chunked-prefill regime), not a leaner
  dispatch stream — see the [execution model](./execution-model.md).
- **The copy lever is closed.** The remaining prefill copies are qmm-hidden
  (off the critical path, overlapping the bandwidth-bound matmuls), so further
  strided-copy elimination buys correctness/traffic but no measured wall-clock.
