# Sampling

The forward pass ends with a vector of logits — one score per vocabulary
token — and the job of this last step is to turn that vector into a single
concrete token id, append it to the sequence, and feed it back into the next
forward. This page is about that turn: how logits become a token, why
`cider-press` picks the argmax on the GPU rather than scanning it on the CPU,
and why prefill only ever needs the *last* position's logits.

## Concept

A decoder-only transformer's final hidden state is projected to a score for
every token in the vocabulary. Two things stand between the last block's
output and that score vector:

1. a final **[RMSNorm](./rmsnorm.md)** of the hidden state, the same
   normalization the blocks use internally, applied once more before the head;
2. the **LM head** — a `[vocab, D]` projection that maps the `D`-dimensional
   hidden state to a `vocab`-wide logit vector, one
   [quantized matmul](./quantized-matmul.md) against the (transposed) weight.

The logits are an unnormalized log-probability distribution. Choosing a token
from them is the *decoding strategy*, and the strategies form a spectrum:

- **Greedy (argmax)** — take the single highest-scoring token, every step. It
  is deterministic and is what you want when reproducing a reference run
  byte-for-byte. It is also the cheapest: no softmax, no sort, no RNG, just the
  index of the maximum.
- **Temperature / top-k / top-p sampling** — instead of always taking the
  maximum, draw from the distribution. A *temperature* divides the logits
  before the softmax (higher = flatter = more random); *top-k* restricts the
  draw to the `k` highest-scoring tokens; *top-p* (nucleus) restricts it to the
  smallest set of tokens whose probabilities sum to `p`. These trade
  determinism for diversity and are what production chat sampling uses.

The other half of the concept is *which* logits you actually need. A causal
transformer produces a logit vector for **every** position in its input, but
in autoregressive generation you only ever continue from the **last** one — the
next token is conditioned on the whole prompt, which is exactly the final
position's hidden state. So at prefill, where the prompt is `T` tokens wide,
the blocks must run over all `T` positions (every position's key/value has to
land in the [KV cache](./kv-cache.md) so later tokens can attend to it), but
the norm + head only need to run on the single last row. Projecting all `T`
rows through the head would compute `T - 1` logit vectors that are immediately
discarded.

## In cider-press

The head lives in
[`Qwen2Model::project_logits`](../../crates/cider-press-models/src/qwen2/model.rs):
a final [`rms_norm`](./rmsnorm.md) followed by the tied LM head — a
[`quantized_matmul(.., transpose=true)`](./quantized-matmul.md) against the
**embedding table itself**. Qwen2.5 ships `tie_word_embeddings = true`, so the
`[vocab, D]` table used for the input [embedding](./embedding.md) is reused,
transposed, as the output projection; there is no separate head weight.

At prefill,
[`Qwen2Model::forward_last`](../../crates/cider-press-models/src/qwen2/model.rs)
slices the hidden states to the last row *before* the norm/head, so only one
position is projected — numerically identical to `forward(..)[:, -1, :]`
because RMSNorm and the head are per-position, but without computing the `T - 1`
discarded rows. The blocks above it still run over all `T` tokens to fill the
caches.

Token selection is then **on-GPU**. The `[1, 1, vocab]` logits feed an in-graph
argmax — `argmax_last_position_lazy` in
[`generator.rs`](../../crates/cider-press-models/src/generator.rs) records a
single `argmax` op over the vocab axis, which dispatches MLX's
[`argmax_bfloat16`](../../kernels-mlx/mlx/backend/metal/kernels/arg_reduce.metal)
via
[`arg_reduce.rs`](../../crates/cider-press-kernels/src/kernels/arg_reduce.rs) —
inside the **same command buffer as the LM head**. The GPU reduces the vocab
row to a single `u32` index; only that 4-byte index is read back to the CPU.
[`generator.rs`](../../crates/cider-press-models/src/generator.rs) drives the
loop: the argmax tensor is kept on-GPU and fed straight into the next forward's
embedding gather (it is already in the `[1, 1]` U32 shape the next step's input
ids want), so the per-token next-id never round-trips the CPU. The tie-break is
locked to the lower index, matching MLX's `ArgMax`.

**Model variations.**

- **Tied vs. untied LM head.** Qwen2.5 ties the head to the embedding table;
  many models (larger Qwen variants, some Llama configs) ship a *separate*
  `lm_head` weight. The loader, not the forward pass, decides — an untied
  checkpoint would load a distinct head weight; the projection op is identical.
- **Logit soft-capping.** Gemma2 scales-and-`tanh`-clamps the final logits
  (`logits = cap * tanh(logits / cap)`) before sampling; a model that needs it
  inserts that elementwise step between the head and the argmax.
- **Repetition penalties.** Many sampling configs down-weight logits of tokens
  already in the context before the draw — a logit-processor step that lives in
  the decoding strategy, not the head.

## Performance

Two perf results converge on this step.

**GPU argmax** (`docs/QWEN_PERF.md` § "GPU argmax") moved next-token selection
off the CPU. Greedy decode used to `eval()` the logits, read back the full
302 kB `[vocab]` row, and scan ~151 k entries on the CPU to find the max —
~444 µs of serial CPU work per token (~9.5% of the decode step). That is
replaced by the in-graph `ArgReduce` above: a sub-µs memory-bound GPU pass over
one vocab row, folded into the same command buffer as the head, with only a
4-byte index read back. The `argmax` span dropped from **~444 µs/token to
~0.8 µs/token**, and decode throughput moved **~217 → ~243 tok/s** at the time
of that change. Crucially, removing the logits readback is what *enabled*
keeping the next-token id on-GPU to feed the following embedding gather —
eliminating the per-token CPU↔GPU sync (see the
[execution model](./execution-model.md) for why that sync mattered to the
decode pipeline). Greedy output is unchanged: the kernel is MLX's own, so it
matches `mlx_lm` by construction.

**Last-position LM-head** (`docs/QWEN_PERF.md` § "Prefill parity resolved")
addresses the prefill side. Projecting only the last prompt row through the
norm + head, rather than all `T`, cut synchronous prefill to **~7.76 ms** — at
or ahead of `mlx_lm`'s equivalent real prefill (~7.84–7.99 ms single-forward,
~8.98 ms for the actual `generate_step` split). That measurement is what closed
the prefill-gap investigation: the historical "gap" turned out to be a stale
single-forward `mlx` proxy, not a real op-sequence deficit.

## Open levers

- **Temperature / top-k / top-p sampling — the clearest open feature.** Only
  greedy argmax is wired today; there is no temperature, top-k, or top-p path
  at all. The [`generator.rs`](../../crates/cider-press-models/src/generator.rs)
  loop hard-codes the `argmax` op as its selection step, so adding stochastic
  sampling means a new decoding-strategy seam (a logit-processor + a sampled
  draw) ahead of — or in place of — that argmax, plus the supporting kernels
  (softmax over the vocab row, a top-k/top-p restriction, an RNG draw). Greedy
  is deliberate for now: it keeps decode byte-identical to the `mlx_lm`
  reference, which is the parity canary. (Note: the
  [`gpu_sampler.rs`](../../crates/cider-press-kernels/src/gpu_sampler.rs)
  module is *GPU counter-sampling for profiling*, unrelated to token sampling —
  no token-sampling kernel beyond argmax exists yet.)
- **Untied / soft-capped / penalized heads** are loader- and
  strategy-level extensions (see Model variations) that land with their first
  consuming model.
