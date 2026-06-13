# Embedding

The transformer blocks operate on dense vectors, but a prompt arrives as a
sequence of integer token IDs. Embedding is the one-step bridge: turn each
ID into the vector the rest of the network expects. It is the second stage
of the pipeline, run once over the whole prompt in prefill and once per
generated token in decode.

## Concept

A model's vocabulary is a fixed set of tokens, and the embedding table is a
learned matrix with one row per token — shape `[vocab, D]`, where `D` is the
model's hidden dimension. Embedding a token is a pure table lookup: take the
ID, read out that row. There is no arithmetic, just indexing. A batch of
token IDs shaped `[B, T]` produces `[B, T, D]` — every ID replaced by its
`D`-wide row.

The row is *learned*: training nudges each token's vector so that tokens
used in similar contexts land near each other, and so the geometry of the
space carries the information the attention and feed-forward layers go on to
mix. That is the whole reason the lookup exists rather than, say, a one-hot
encoding — the table *is* the model's first representation of what each
token means, and it is trained jointly with everything downstream.

## In cider-press

The lookup is
[`embed_tokens`](../../crates/cider-press-models/src/nn.rs), called at the
top of the forward pass from
[`Qwen2Model`](../../crates/cider-press-models/src/qwen2/model.rs) (its
`hidden_states` flattens `input_ids` to rank-1, embeds, then reshapes back
to `[1, T, D]` before the blocks). The wrinkle is that Qwen2.5's embedding
table is itself **4-bit quantized** — the same `[vocab, D]` weight the tied
LM head reuses (see [weights and quantization](./weights-and-quantization.md)).
So a plain row-read won't do: each row has to be gathered *and* dequantized.

`embed_tokens` composes that from existing runtime ops, mirroring MLX-LM's
quantized embedding path — no new kernel:

1. **Gather** the rows of each component of the quantized triple — the
   packed `u32` lanes `w_q`, the `bf16` `scales`, and the `bf16` `biases` —
   indexed by the token IDs. Each is one
   [`gather`](../../crates/cider-press-kernels/src/kernels/gather.rs)
   dispatch (the JIT-assembled MLX gather kernel; the `bf16_u32_rank1` and
   `u32_u32_rank1` instantiations cover the value and packed-lane reads).
2. **Dequantize** the gathered triple back to dense `bf16` in one
   [`dequantize`](../../crates/cider-press-kernels/src/kernels/dequantize.rs)
   dispatch (`out = w * scale + bias` per group of 64).

The result is a dense `[T, D]` `bf16` tensor — the same vectors a non-quantized
embedding would produce, just reconstructed on the fly. Everything is lazy:
the three gathers and the dequantize land in a single `eval()`.

**Model variations.** Two knobs the family leaves open. *Tied vs. untied
embeddings:* Qwen2.5 sets `tie_word_embeddings = true`, so this one table
feeds *both* the input lookup here and the output projection at the end —
the LM head re-projects through the same weight transposed (see
[sampling](./sampling.md)). Untied models ship a separate `lm_head` and the
two tables are independent. *Embedding scale:* some families (notably the
Gemma line) multiply the gathered vectors by `sqrt(D)` before the first
block; Qwen2.5 does not, so there is no scale step here. *Quantized vs.
dense table:* quantizing the embedding (as Qwen2.5-0.5B does) is what forces
the gather-then-dequantize shape above; a model with a dense `bf16` table
would gather the rows directly and skip the dequantize.

## Performance

Embedding is negligible, and the measurements bear that out. The prefill
GPU-compute trace records the three gathers and the
single dequantize at **~0%** of GPU compute. A (perturbed) decode trace puts
the three gathers at **~0.01 ms** and the dequantize at **~0.003 ms** —
together well under half a percent of the step. The conclusion holds either
way: the step never shows up in the wall-clock. That is the expected shape:
the lookup touches only `T · D` values, a rounding error next to the matmuls
that stream the whole weight set every step (the quantized projections and
the LM head dominate; see [quantized matmul](./quantized-matmul.md)). It is a
memory-bound op — indexing and a per-group affine reconstruction, no real
arithmetic — but the footprint is so small the bandwidth never shows up on
the critical path.

## Open levers

None significant. The gather is memory-bound, but at `T · D` it is far too
small to be worth optimizing — there is no plausible win here next to the
matmul-dominated stages. Noted for completeness, not open.
