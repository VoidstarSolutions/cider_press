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

Decode is **critical-path-bound** (`QWEN_PERF.md`): one `Tensor::eval` per
token (~3.7 ms/token, ~90% of the step), ~2.3× slower than `mlx_lm`. The
dispatch round-trip tax (items #1, #3), the CPU-side allocation tax (item
#2), and the post-eval CPU vocab scan (item #6, GPU argmax) have been
removed; the `tensor.eval` encode/wait split is now **~86% GPU execution,
~14% CPU encode**. The GPU wait is a deep chain of ~900 *dependent*
dispatches/token (the 4-bit `qmv` matvecs plus the small ops gated behind
them). A concurrent-encoder spike confirmed this is genuine
dependency-latency, **not** removable serialization — correct concurrent
dispatch is a net loss (item #4). So the levers shorten the chain (fusion)
or speed each link (`qmv`), plus pipelining. In measurement-justified
priority:

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
2. **Buffer pool / allocator — DONE (~1.75×, `feat/buffer-pool`).** A
   cross-token free-list (`buffer_pool.rs`) recycles op-output scratch via
   a `PooledBuffer` that returns its `MTLBuffer` to the pool on drop;
   decode hits it ~98% of the time, lifting ~1000 fresh allocations/token
   off the serial encode path (CPU-encode share ~38% → ~11%). The pool was
   inert on its own (3% hit) until the **enabling KvCache fix**:
   `KvCache::update` now bases each `SliceUpdate` on the slab leaf instead
   of chaining off the prior op-tensor, which had pinned every step's
   projection graph for the whole decode. Together: decode ~120 → ~210
   tok/s, peak RSS ~1192 → ~900 MiB (+32% regression resolved — the cap is
   non-binding; the freed-graph reclaim is what bounds it). Narrowed
   contract: successive `update`s must be separated by an `eval`
   (fail-loud guard). Design:
   `docs/superpowers/specs/2026-05-31-buffer-pool-design.md` and
   `…-kvcache-collapse-design.md`. **Remaining:** within-eval reuse
   (mid-eval freeing, as MLX does) toward `mlx_lm`'s ~329 MiB peak.
3. **KvCache same-eval batching — DONE.** `KvCache::update` is a lazy
   SliceUpdate op (slab-based since `feat/buffer-pool`); the forced
   `k.eval()` + `v.eval()` per layer is gone. Metal hazard tracking
   serializes the slab write → SDPA read within the single per-token
   command buffer.
4. **Decode GPU wait (the ~900-dependent-dispatch critical path).** This is
   the dominant remaining cost. Three sub-levers, in priority:

   *(a) Kernel fusion — leading lever, untried.* The wait is a deep chain of
   ~975 dispatches/token, ~900 of them dependent. Fewer dependent dispatches
   ⇒ a shorter critical path ⇒ less inter-dispatch latency. Targets: fuse
   RoPE into the projection epilogue, the rms-norm reduction, SwiGLU's
   `silu*up`, and the residual adds. MLX runs the same model in far fewer
   dispatches, likely the bulk of its ~2.3× lead. Parity-sensitive (kernel
   changes); highest-potential, highest-effort. Decode attention is already
   fused (`sdpa_vector`); the score-matmul/softmax/copy chain is gone.

   *(b) Faster `qmv` kernels — speeds each link.* The 4-bit `qmv` matvecs are
   the heaviest dispatches. The audit (`docs/QWEN_PERF.md` `## qmv audit`,
   Approaches A + B) found the attained bandwidth is **N-dependent**: large-N
   linears (gate/up/down ~272–306 GB/s, lm_head ~464) are roofline-bound near
   the M4 Max ~410–546 GB/s spec, but small-N decode projections (k/v_proj
   ~13 GB/s, q/o_proj ~85) are occupancy-starved by the generic
   `(1, ceil(N/8), 1)` grid — lever (b), compounded by fast-path absence
   (lever a — no Qwen2.5-0.5B shape has a 512-multiple K). A small-N
   launch-config fix (split the output dim across more threadgroups) plus
   fast-path-via-padding are dispatch-side, parity-preserving, no edits to
   vendored `quantized.metal`. Shortens each `qmv` link but not the dispatch
   *count* (that is fusion).

   *(c) Concurrent dispatch encoder — tested, REJECTED.* A spike
   (`spike/concurrent-encoder`) flipped the serial encoder
   (`computeCommandEncoder`) to concurrent dispatch with correct
   buffer-level hazard barriers. Output is byte-identical to serial but
   decode is ~13% *slower*: ~900/975 dispatches need a barrier (the graph is
   ~92% a sequential chain), so almost nothing overlaps and the explicit
   barriers add cost. The no-barrier "floor" (~510 tok/s) only computes wrong
   answers (it ignores the dependencies). The serial encoder is **not** the
   bottleneck; the dependency chain is. Full analysis: `QWEN_PERF.md`
   `## Concurrent-encoder spike`. A defensive prerequisite did land — the KV
   slabs are now zero-filled (`KvCache::new`) so any future reordering reads
   deterministic zero, not garbage.
5. **Pipelining / per-token sync removal (CPU encode, now ~12%)** —
   `commit_and_wait` is synchronous, so the (now ~0.5 ms) CPU encode never
   overlaps the GPU, and each token still round-trips the next-token id
   through the CPU. Multiple command buffers in flight, or Approach-C
   plumbing threading `Commands` through the forward and chaining the GPU
   argmax id (item 6) straight into the next embedding gather, would hide
   it and eliminate the per-token sync. Lower priority now that the buffer
   pool (item 2) removed the allocation portion and encode is ~12% of the
   step. **Within-eval buffer reuse** (mid-eval freeing) is the related RSS
   lever toward `mlx_lm`'s ~329 MiB peak.
6. **GPU argmax — DONE (~210 → ~243 tok/s).** Next-token selection moved
   on-GPU: an in-graph `OpKind::ArgReduce` (MLX's `argmax_bfloat16` from
   `arg_reduce.metal`) reduces the last logits row to a `u32` index inside
   the same per-token command buffer as the lm_head. The ~444 µs/token CPU
   `cpu_to_vec` + 151 k-wide scan and the 302 kB logits readback are gone,
   replaced by a sub-µs GPU reduction folded into `tensor.eval.wait` + a
   4-byte index readback (the `argmax` span fell ~444 → ~0.8 µs/token).
   Greedy output is unchanged (MLX's own kernel ⇒ matches `mlx_lm` by
   construction; `generator_parity_greedy_qwen2` passes). Design:
   `docs/superpowers/specs/2026-05-31-gpu-argmax-design.md`. This is the
   **enabler** for eliminating the per-token CPU↔GPU sync (item 5): with
   the logits readback gone, the next-token id can stay on-GPU and feed the
   following embedding gather directly (command-buffer chaining) — that
   remains future work.
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
