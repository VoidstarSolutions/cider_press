# Architecture

`cider-press` runs Qwen2.5-0.5B-Instruct-4bit on Apple Silicon: a lazy
`Tensor` graph over vendored MLX Metal kernels, with model architectures
composed on top. This doc is the mechanical map and the forward roadmap.
For *why* the runtime is shaped as it is (lazy eval, Views, KvCache), see
`RUNTIME_DESIGN.md`; for measured performance, `QWEN_PERF.md`.

## Layers

- **`cider-press-kernels`** — vendored MLX `.metal` sources (flattened by
  `build.rs`) plus per-kernel Rust dispatch. The bottom of the stack;
  knows Metal, not tensors.
- **`cider-press-runtime`** — lazy `Tensor` + op graph + `eval()` (the
  synchronous dispatch boundary), plus `KvCache` and the feature-gated
  `profile` facility. Owns dtypes, layouts/views, quantized weights.
- **`cider-press-models`** — transformer architectures (Qwen2), composed
  from runtime ops; the tokenizer, chat template, and greedy `Generator`.
- **`cider-press`** — facade + CLI (`chat` and `bench` subcommands).

## The forward pass

Qwen2.5 is a Llama-family transformer with Qwen RoPE/norm choices. One
forward pass:

```text
tokens [B, T]
  → embed (gather quantized embed_tokens → dequantize)   [B, T, D]
  → for layer in 0..N:
        residual = x
        x = rmsnorm(x, ln1_gamma)
        q,k,v = quantized_matmul(x, W_{q,k,v}) (+ bias)
        reshape per-head; apply RoPE to q, k
        kv_cache.update(k, v); read cached_k, cached_v
        attn = sdpa(q, cached_k, cached_v, mask, scale)   # GQA broadcast
        x = residual + quantized_matmul(attn, W_o)
        residual = x
        x = rmsnorm(x, ln2_gamma)
        x = residual + quantized_matmul(silu(gate(x)) * up(x), W_down)
  → x = rmsnorm(x, final_gamma)
  → logits = quantized_matmul(x, W_embed, transpose=true)  # tied embedding
  → argmax → next token
```

Qwen2.5-0.5B specifics: `D=896`, `N=24` layers, `H_q=14`, `H_kv=2` (GQA
7:1), `D_h=64`, `FFN=4864`, vocab≈151k. 4-bit affine quant,
`group_size=64`.

`mlx_lm`'s Qwen2.5 port is the parity reference; weights are the
HuggingFace / mlx-community 4-bit checkpoint.

## Performance backlog

Decode is dispatch-bound (`QWEN_PERF.md`): ~89% of each step is in
`Tensor::eval` over ~49 synchronous `commit + wait` cycles per token,
~10× slower than `mlx_lm`. In measurement-justified priority:

1. **Cross-eval command-buffer batching** — the dominant lever. Fold a
   token's ~49 `eval()` calls into far fewer (ideally one) pipelined
   command buffers with async commit. Removes most of the per-token
   dispatch tax.
2. **Buffer pool / allocator** — peak RSS sits well above post-load from
   per-`eval()` scratch churn; a pool amortizes it.
3. **KvCache same-eval batching** — the host `memcpy` is free, but
   `update` forcing two synchronous evals per layer feeds the dispatch
   problem; fold the cache write into the forward's command buffer.
4. **GPU argmax** — ~2.7% of decode (the `[vocab]` cpu_to_vec + scan).
   After the big levers.
5. **`.metallib` precompilation** — removes the one-time cold-start JIT
   (~tens of seconds) from first-token latency in a shipped binary.
6. **Top-k / top-p / temperature sampling** — greedy-only today.
7. **Strided-aware kernels** (MLX's `g_copy` family) — we materialize
   strided views via `Copy`; strided kernels are a perf-only improvement.

## Carry-forward items

- `LanguageModel` trait extraction — `Generator` is concrete on
  `Qwen2Model`; lift when a second architecture lands.
- `quantized_matmul` `transpose=false` direction — still unimplemented;
  Qwen2.5 ties the LM head, so it may never be needed.
- Generic non-aligned `affine_qmm` — deferred to its first unaligned
  consumer.
- Interactive / multi-turn REPL — `cider-press chat` streams a single
  turn today.
