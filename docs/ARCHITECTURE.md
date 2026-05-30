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

Decode is per-eval-cost-bound (`QWEN_PERF.md`): one `Tensor::eval` per
token (~10 ms/token, ~89% of the step), ~6.2× slower than `mlx_lm`.
The dispatch round-trip tax has been removed — see items #1 and #3 below.
The `tensor.eval` encode/wait split measures the remainder: **~62% GPU
execution, ~38% serialized CPU encode**. Both halves individually exceed
`mlx_lm`'s whole ~1.8 ms/token. In measurement-justified priority:

1. **Cross-eval command-buffer batching — DONE (~1.6×).** The in-graph
   KV write (SliceUpdate, `feat/command-buffer-batching`) collapsed the
   token's ~49 synchronous `commit + wait` round-trips to **one command
   buffer per token**. `kvcache.update` cost dropped from ~1874 ms to
   0.56 ms over the 120-step timed window; decode improved ~55 → ~90
   tok/s. Measured finding: the dispatch-collapse was worth ~1.6×; the
   dominant remaining lever has shifted to per-eval cost (see below).
   Numbers: `QWEN_PERF.md` and
   `docs/superpowers/specs/2026-05-29-kv-slice-update-perf-design.md`
   (spike results section).
2. **Buffer pool / allocator** — ~1000 fresh `MTLBuffer` allocations per
   token (one per op output, no pool); peak RSS is ~1192 MiB (+32% from
   the one-eval-holds-all-intermediates regime). A pool recycles scratch
   across tokens and lifts allocation out of the serial encode path.
3. **KvCache same-eval batching — DONE.** `KvCache::update` is now a
   lazy SliceUpdate op; the forced `k.eval()` + `v.eval()` per layer is
   gone. Metal hazard tracking serializes the slab write → SDPA read
   within the single per-token command buffer.
4. **Fused attention / fewer copies (GPU, ~62%)** — the unfused SDPA
   forces ~11 `permute().copy()`s per layer to feed the contiguous-only
   matmul kernel; a flash-attention kernel reading strided K/V in place
   removes them. The heavy compute is `qmv`, not `gemm_bfloat16` (which
   runs only the small attention score matmuls), so audit `qmv`'s launch
   config before assuming the kernel itself is the gap.
5. **Pipelining / fewer dispatches (CPU encode, ~38%)** — ~1000 dispatches
   are encoded per token (not the "~49" of an earlier draft — that was the
   pre-SliceUpdate `commit + wait` count), and `commit_and_wait` is
   synchronous, so that ~3.8 ms of CPU encode never overlaps the GPU.
   Multiple command buffers in flight, fused kernels (fewer ops), or
   Approach-C plumbing threading `Commands` through the forward hide or
   shrink it; the buffer pool (item 2) attacks the allocation portion.
6. **GPU argmax** — ~4.5% of decode (the `[vocab]` cpu_to_vec + scan).
   After the big levers.
7. **`.metallib` precompilation** — removes the one-time cold-start JIT
   (~tens of seconds) from first-token latency in a shipped binary.
8. **Top-k / top-p / temperature sampling** — greedy-only today.
9. **Strided-aware kernels** (MLX's `g_copy` family) — we materialize
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
