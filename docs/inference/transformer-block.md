# Transformer block

The block is the unit the network repeats. Everything else in this guide —
the norm, the projections, attention, the MLP — is a piece *inside* one
block, and a model is just `N` of these stacked, each with its own weights
but the identical shape. For Qwen2.5-0.5B that is `N = 24`. This page is
about the wiring that holds those pieces together: the residual stream they
all read from and write back to, and the pre-norm placement that decides
*where* each piece sees its input.

## Concept

A decoder-only transformer is a stack of identical blocks threaded by a
single tensor called the **residual stream** — a `[B, T, D]` activation that
enters the network from the embedding and is carried, unbroken, to the final
norm. Each block does not *replace* that stream; it *adds* to it. A block
reads the stream, computes a correction, and writes `stream + correction`
back out. The stream is the backbone; every sub-layer is a side road that
merges back in.

That additive structure is the **residual connection**, and it is the reason
deep transformers train and run at all. The key property at inference is the
**identity path**: if a sub-layer computes nothing useful, the block degrades
to `x → x`, so adding layers can never make the network strictly worse — depth
is safe to add, and the activation magnitudes flowing down a deep stack stay
stable (the same clean path is also what keeps such a stack trainable, but
that is a training-time concern). At inference we only see the forward half —
the stream as a running accumulator that each block nudges.

A block contains **two** sub-layers, applied in series, each wrapped in its
own residual:

```text
x = x + attention(norm(x))     # token mixing — moves information across positions
x = x + feed_forward(norm(x))  # channel mixing — transforms each position independently
```

Attention is the only operation that lets one position read another; the
feed-forward network reshapes each position's vector on its own. Alternating
them — mix across tokens, then mix across channels, then repeat — is the
whole transformer recipe.

The `norm(...)` placement is the **pre-norm** choice, and it is not
incidental. There are two options for where the normalization sits relative
to the residual add:

- **Pre-norm** — normalize the sub-layer's *input*, add the raw sub-layer
  output back to the *un-normalized* stream: `x + sublayer(norm(x))`. The
  residual stream itself is never normalized, so it passes through every
  block untouched — a clean identity highway from embedding to final norm.
- **Post-norm** (the original 2017 transformer) — add first, normalize the
  *sum*: `norm(x + sublayer(x))`. The normalization sits *on* the residual
  path, so the stream is re-scaled at every block.

Pre-norm won out because it keeps the residual path an exact identity, which
both stabilizes the activation magnitudes flowing down a deep stack and
preserves that clean gradient path during training. Nearly every modern
decoder-only model — Llama, Qwen, GPT-NeoX — is pre-norm. Qwen2.5 is classic
pre-norm with the two sub-layers in series.

## In cider-press

[`TransformerBlock`](../../crates/cider-press-models/src/qwen2/block.rs) is
the composition, and its `forward` reads almost exactly like the two-line
recipe above. The struct holds four owned pieces — two norms bracketing two
mixers:

- [`RmsNormLayer`](./rmsnorm.md) `input_layernorm` — the pre-attention norm.
- [`Attention`](./attention.md) — GQA self-attention (the token mixer).
- [`RmsNormLayer`](./rmsnorm.md) `post_attention_layernorm` — the pre-FFN
  norm. Named "post-attention" after its *position* in the layer, but it is a
  *pre*-norm: it normalizes the input to the FFN sub-layer, not the output of
  the attention sub-layer.
- [`Mlp`](./feed-forward.md) — the SwiGLU gated MLP (the channel mixer).

`forward(hidden, offset, cache)` then threads the residual stream through
them:

```text,ignore
let normed     = input_layernorm.forward(hidden)?;      // pre-attn norm
let attn_out   = attention.forward(&normed, offset, cache)?;
let after_attn = hidden.add(&attn_out)?;                 // residual #1

let mlp_input  = post_attention_layernorm.forward(&after_attn)?;  // pre-FFN norm
let mlp_out    = mlp.forward(&mlp_input)?;
after_attn.add(&mlp_out)                                 // residual #2
```

The structural detail to see: the norm output (`normed`, `mlp_input`) feeds
the *sub-layer*, while the residual add takes the **un-normalized** input
(`hidden`, `after_attn`). That is pre-norm in code — the stream `hidden →
after_attn → return` is never itself normalized, exactly the identity highway
from the concept section. The block does not implement the
[`Module`](../../crates/cider-press-models/src/nn.rs) trait: attention needs
the RoPE `offset` and the [`KvCache`](./kv-cache.md), which the
single-input `Module::forward` signature deliberately does not carry, so the
block takes them through its own typed `forward` instead. The mixers
themselves are built from the same model-agnostic
[building blocks](../../crates/cider-press-models/src/nn.rs) the rest of the
crate uses — `Linear`, `Mlp`, `RmsNormLayer` — so the block is "just" four
sub-modules wired by two `add`s; nothing Qwen-specific lives at this layer
except which pieces it picks. The full stack of 24 of these, plus the
embedding and the tied head, is assembled in
[`Qwen2Model`](../../crates/cider-press-models/src/qwen2/model.rs).

Everything in `forward` is **lazy** — each call schedules ops onto the
runtime graph and returns an unevaluated `Tensor`; no GPU work happens until
the model `eval()`s. So a 24-block forward records the entire per-layer body
×24 into one graph before a single kernel dispatches. See the
[execution model](./execution-model.md).

**Model variations.** The block is where several family-level knobs live:

- **Norm placement** — pre-norm (here) vs. post-norm (the original
  transformer). Some models (e.g. Gemma2) place norms on *both* sides of each
  sub-layer.
- **Serial vs. parallel sub-layers** — Qwen2.5 runs attention *then* FFN in
  series (each adding to the stream in turn). GPT-NeoX and Cohere instead run
  them **in parallel** off the *same* normed input —
  `x + attention(norm(x)) + feed_forward(norm(x))` — trading a small quality
  cost for one fewer sequential dependency.
- **Extra norms** — some families add normalizations *inside* the sub-layer,
  e.g. **QK-norm** (normalizing the query and key projections before
  attention). Qwen2.5-0.5B carries none of these; its only norms are the two
  pre-norms shown.

See [models/qwen2.5.md](./models/qwen2.5.md) for Qwen2.5's concrete choices.

## Performance

The block is not separately profiled — it has no kernels of its own. Its cost
*is* the cost of its sub-layers, repeated `N = 24` times, and that per-layer
cost is dominated by the seven quantized matmuls inside it: the four attention
projections (q, k, v, o) and the three FFN projections (gate, up, down). In
decode these are memory-bandwidth-bound — each one streams its whole weight
matrix through memory to project a single activation vector — so the block's
wall-clock is essentially "how many weight bytes does one layer move," ×24.
That is why quantization (¼ the bytes) and the qmv/qmm kernels carry the
decode story, not the block wiring; see
[quantized matmul](./quantized-matmul.md) and
[weights and quantization](./weights-and-quantization.md).

The two residual `add`s and the two norms are cheap by comparison —
element-wise / per-row passes over a `[B, T, 896]` activation, with no large
weight to stream. They are **not** folded into the matmul kernels: in the
graph each `hidden.add(&attn_out)` and `after_attn.add(&mlp_out)` is its own
elementwise dispatch, two per block (this is read off `forward` directly —
the adds are explicit `Tensor::add` calls, not a kernel epilogue). That is by
design rather than an oversight: the residual operands come from *different*
sub-layers (the block input and the attention output are produced by
separate kernel chains), so there is no single matmul whose epilogue could
absorb the add.

## Open levers

- **Cross-layer fusion is not pursued.** The blocks are strictly
  data-dependent — block `i+1`'s input is block `i`'s output (that is what
  the residual stream *is*), so there is no independent work to overlap
  *across* layers; the chain is genuinely sequential. The concurrency wins
  that did land are *within* a layer / *across* command-buffer seams (the
  encoder/fence-chain work in the [execution model](./execution-model.md)),
  not block-level. Fusing two layers into one mega-kernel would buy nothing
  the within-eval graph batching doesn't already get, since the dependency is
  real.
- **Residual-add dispatch placement.** The two residual adds sit as separate
  elementwise dispatches (above). They are off every measured critical path,
  so collapsing them — e.g. by teaching a future fused attention/FFN kernel
  to take a residual operand and add in its epilogue — is a *potential*
  micro-optimization with no current evidence it would move the needle.
  Noted for completeness, not open: the projection-side levers
  ([quantized matmul](./quantized-matmul.md)) and the prefill execution
  structure ([execution model](./execution-model.md)) are where the measured
  gaps live.
