# How LLM inference works in cider-press

This folder is a step-by-step walk through autoregressive transformer
inference, the way `cider-press` actually runs it. Each file covers one
stage of the forward pass — what the stage computes, why it exists, and
which Rust module and Metal kernel implement it. Qwen2.5-0.5B-Instruct-4bit
is the worked example throughout, but the structure is deliberately
model-agnostic: the pipeline below is what *every* decoder-only transformer
does, and the per-model specifics live on their own pages under
[`models/`](#models). Each of these files is also pulled into its module as
rustdoc via `#[doc = include_str!]`, so the same prose serves GitHub
readers, `cargo doc` readers, and anyone reading the source.

## The pipeline

A language model turns a sequence of tokens into a probability
distribution over the next token, then we sample one and repeat. The whole
loop, in reading order:

1. [Tokenize](./tokenization.md) the prompt text into integer token IDs.
2. [Embed](./embedding.md) each token ID into a vector — a lookup into the
   (quantized) embedding table, dequantizing the gathered rows.
3. For each of the `N` transformer blocks ([transformer block](./transformer-block.md)),
   in sequence:
   1. [RMSNorm](./rmsnorm.md) the input (pre-attention norm).
   2. Project to queries, keys, and values via
      [quantized matmul](./quantized-matmul.md) (QKV projection), then
      reshape the result per attention head.
   3. Apply [RoPE](./rope.md) (rotary position embedding) to Q and K.
   4. Write the new K and V into the [KV cache](./kv-cache.md), and read
      back the full cached history.
   5. Compute [attention](./attention.md) — scaled dot-product attention
      over the cached keys/values.
   6. [RMSNorm](./rmsnorm.md) again (pre-FFN norm).
   7. Run the [feed-forward](./feed-forward.md) network (gated MLP — SwiGLU).
4. [RMSNorm](./rmsnorm.md) the final block output.
5. Project to vocabulary logits via the [LM head](./quantized-matmul.md)
   (often the embedding table transposed — Qwen2.5 ties them).
6. [Sample](./sampling.md) the next token from the logits.
7. Append it to the sequence and loop back to step 3 until a stop token or
   length limit.

Steps 1–2 and 4–6 run once per generated token (in decode; in prefill
steps 1–2 run once over the whole prompt and the head only over the last
position — see the [execution model](./execution-model.md)); the per-layer
body of step 3 runs `N` times. The diagram below traces a single forward
pass:

```text
tokens [B, T]
  → embed (gather embedding table → dequantize)            [B, T, D]
  → for layer in 0..N:
        residual = x
        x = rmsnorm(x, attn_norm_weight)
        q, k, v = quantized_matmul(x, W_{q,k,v}) (+ bias)
        reshape per-head; apply RoPE to q, k
        kv_cache.update(k, v); read cached_k, cached_v
        attn = sdpa(q, cached_k, cached_v, mask, scale)     # GQA broadcast
        x = residual + quantized_matmul(attn, W_o)
        residual = x
        x = rmsnorm(x, ffn_norm_weight)
        x = residual + quantized_matmul(silu(gate(x)) * up(x), W_down)
  → x = rmsnorm(x, final_norm_weight)
  → logits = quantized_matmul(x, W_lm_head)                 [B, T, vocab]
  → sample → next token
```

(`B` = batch, `T` = sequence length, `D` = model/hidden dimension, `N` =
number of layers. The names above are generic; for the concrete Qwen2.5
shapes and weight names see [models/qwen2.5.md](./models/qwen2.5.md).)

## The crates

The codebase is a Cargo workspace split into crates, one per layer of the
stack. Dependencies flow strictly **upward** — a higher crate may depend on
the ones below it, never the reverse:

- **`cider-press-kernels`** — the vendored MLX `.metal` sources (flattened
  at build time) plus per-kernel Rust dispatch. The bottom of the stack; it
  knows Metal, not tensors.
- **`cider-press-runtime`** — the lazy `Tensor`, the op graph, and `eval()`
  (the synchronous dispatch boundary), plus the `KvCache` and the
  feature-gated profiling facility. Owns dtypes, layouts/views, and
  quantized weights.
- **`cider-press-models`** — transformer architectures (Qwen2) composed
  from runtime ops, plus the tokenizer, chat template, and the greedy
  `Generator`.
- **`cider-press`** — the facade and CLI (`chat` and `bench` subcommands).
  Re-exports the runtime and model layers.

So a single attention op flows top-down: the models crate composes it from
runtime `Tensor` ops, the runtime records those into a graph and dispatches
them at `eval()`, and the kernels crate is what each dispatch actually
runs on the GPU.

## The execution model

`cider-press` does not compute eagerly. Building a `Tensor` expression
records nodes into a lazy graph; nothing touches the GPU until `eval()`,
which is the one synchronous dispatch boundary. This is what lets the
runtime fuse work and batch command buffers, and it is why "prefill" (the
first pass over the whole prompt, `T` > 1) and "decode" (each subsequent
single-token step, `T` = 1) have such different performance characteristics.
The [execution model](./execution-model.md) page covers the lazy graph, the
`eval()` boundary, and the prefill/decode split in detail — read it once and
the rest of the guide makes more sense.

## Models

Each supported model gets its own page: the concrete shapes, weight names,
RoPE/norm choices, and any quirks that deviate from the generic pipeline
above. The page also collects "Model variations" notes — the small
architectural knobs (GQA ratio, norm placement, activation, tied vs.
untied embeddings) that distinguish one model family from another.

- [Qwen2.5](./models/qwen2.5.md) — the current worked example
  (Qwen2.5-0.5B-Instruct-4bit).

More models land as a new per-model page plus any "Model variations" notes
they introduce.

## Steps

- [Tokenization](./tokenization.md) — text → token IDs (and back).
- [Weights and quantization](./weights-and-quantization.md) — how
  4-bit affine-quantized weights are stored, loaded, and dequantized.
- [Embedding](./embedding.md) — token IDs → vectors via the embedding
  table.
- [Transformer block](./transformer-block.md) — how the per-layer pieces
  compose into one block, with residual connections.
- [RMSNorm](./rmsnorm.md) — root-mean-square layer normalization.
- [Feed-forward](./feed-forward.md) — the gated MLP (gate/up/down with
  SiLU).
- [Quantized matmul](./quantized-matmul.md) — the qmv/qmm kernels behind
  every projection and the LM head.
- [Attention](./attention.md) — scaled dot-product attention with GQA.
- [RoPE](./rope.md) — rotary position embeddings on Q and K.
- [KV cache](./kv-cache.md) — caching past keys/values so decode is
  O(1) per token instead of O(T).
- [Sampling](./sampling.md) — turning logits into the next token (greedy
  today).
- [Execution model](./execution-model.md) — the lazy graph, the `eval()`
  boundary, and prefill vs. decode.
