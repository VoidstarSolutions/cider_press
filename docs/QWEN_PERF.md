# Qwen2.5-0.5B perf

End-to-end performance of Qwen2.5-0.5B-Instruct-4bit under `cider-press`,
measured via `cider-press bench`, compared against `mlx_lm` on the same
prompt and token budget. Methodology: warm up, then time a steady-state
window; see `docs/ARCHITECTURE.md` for the architecture and forward backlog.

The headline: decode runs **~10× slower than MLX** at the full-model
scale, far wider than the single-kernel 1.5× baseline gap. The profiling
breakdown shows why — and it's exactly where the project hypothesis
predicted: the synchronous per-dispatch `commit + wait` boundary, paid
~49 times per token, not the kernels.

## Environment

- Machine: Apple M4 Max
- OS: macOS 26.5 (build 25F71)
- Checkpoint: `mlx-community/Qwen2.5-0.5B-Instruct-4bit` (local copy),
  4-bit affine quant, `group_size=64`
- Prompt: `"Write a short paragraph about the city of Seattle."`
  (39 tokens after chat-template rendering)
- `max_tokens`: 128, `warmup`: 8 (so 120 timed decode steps)
- cider-press build: `cargo run --release --features profiling`
- MLX: `mlx_lm` 0.31.3 / `mlx` 0.31.2
- 5 cider-press runs (run 1 cold, runs 2–5 warm); 3 `mlx_lm` runs.
  Figures below are representative; ranges noted where variance is
  material.

## Throughput

| phase   | cider-press | mlx_lm  | ratio (mlx ÷ cp) |
|---------|------------:|--------:|-----------------:|
| prefill | ~290 tok/s  | ~1346 tok/s | ~4.6× |
| decode  | ~55 tok/s   | ~560 tok/s  | ~10× |

- **Prefill** processes the 39-token prompt in one forward. cider-press
  ranged 263–319 tok/s across warm runs (~290 median); `mlx_lm` 1195–1369
  (~1346). The gap is narrower here because a single 39-token forward
  amortizes the per-dispatch overhead over more compute.
- **Decode** is the per-token steady state (the perf-dominant path).
  cider-press ranged 43–65 tok/s across runs (~55 median; laptop
  thermal/scheduling variance is real at this speed); `mlx_lm` was tight
  at 555–568 (~560). This is the **~10× gap** and the number that
  matters.
- Load time (excluded from tok/s): ~0.17 s.

**Cold-start note:** the very first forward of a fresh process triggers
Metal JIT compilation of the quantized kernels — run 1's prefill measured
**42.6 s** (0.9 tok/s) for this reason. Metal caches compiled libraries
to disk cross-process, so runs 2+ prefill in ~130 ms. The 42.6 s is a
one-time first-token-latency cost (a `.metallib` precompile would
amortize it; deferred — see `docs/ARCHITECTURE.md`), not representative of
prefill compute, and is excluded from the table above.

## Decode-step breakdown

Per-span totals over the timed decode window (120 steps), from the
profiling facility (representative warm run, decode wall ≈ 2185 ms /
54.9 tok/s). Spans **overlap by construction**: `kvcache.update` includes
the `k.eval()` / `v.eval()` that also land in `tensor.eval`, and
`tensor.eval` covers all GPU dispatch+wait wherever triggered. `argmax`
is the post-eval CPU `cpu_to_vec` + scan only.

| span           | total (ms) | hits | µs/hit | hits/token | share of decode |
|----------------|-----------:|-----:|-------:|-----------:|----------------:|
| tensor.eval    | 1951 | 5831 | 335 | 48.6 | ~89% |
| kvcache.update | 1874 | 2856 | 656 | 23.8 | ~86% (overlaps eval) |
| argmax         |   60 |  119 | 501 |  1.0 | ~2.7% |
| kvcache.memcpy |    5 | 2856 |   2 | 23.8 | ~0.24% |

Reading it:

- **`tensor.eval` is ~89% of the decode step, spread over ~49 separate
  `eval()` calls per token** (5831 ÷ 120). Each `eval()` is one
  `Commands` opened, encoded, committed, and **waited on synchronously**
  (`commit_and_wait`). At ~335 µs/call that's ~16 ms/token just in
  dispatch round-trips — the dominant cost.
- **`kvcache.update` (~86%) is almost entirely its two input evals**, not
  the cache write: the write itself (`kvcache.memcpy`) is **0.24%**
  (~2 µs/hit). The unified-memory host `memcpy` is effectively free; the
  cost is that `update` forces a `k.eval()` + `v.eval()` per layer
  (2 of the ~49 evals/token × 24 layers ≈ the 23.8 hits/token here).
- **`argmax` is ~2.7%** — the bf16 `cpu_to_vec` + scan over the ~151 k
  vocab, ~500 µs/token. Real but not the bottleneck.

## Memory

| mark        | cider-press RSS |
|-------------|----------------:|
| pre-load    | ~14 MiB  |
| post-load   | ~683 MiB |
| post-decode | ~650–893 MiB |
| peak        | ~902 MiB |

`mlx_lm` peak (Metal allocations, `GenerationResponse.peak_memory`):
**~329 MiB**.

These measure **different things** — cider-press samples process RSS
(via mach `task_info`; includes the 278 MB safetensors read into a host
`Vec` during load, the dequantized weight buffers, and per-`eval()`
scratch allocations with no buffer pool yet), while `mlx_lm` reports the
Metal buffer high-water mark. So compare trends, not absolute equality:
cider-press' ~902 MiB peak vs ~683 MiB post-load points at per-`eval()`
allocation churn that a buffer pool would absorb.

## Hot spots & next levers

The decode step is **dispatch-bound, not compute-bound**. ~89% of it is
`tensor.eval`, and the kernels themselves are known-good (bit-exact vs
MLX). The ~10× decode gap to MLX is the synchronous
per-dispatch `commit + wait` boundary paid ~49× per token, versus MLX
batching a whole forward into pipelined command buffers. In measurement-
justified priority order:

- **Cross-eval command-buffer batching — the dominant lever.** Folding a
  token's ~49 `eval()` calls into far fewer command buffers (ideally one
  per forward) with async commit removes most of the ~16 ms/token
  dispatch tax. This is the predicted per-dispatch gap compounding at
  full-model scale, and the single highest-value follow-up.
- **Buffer pool / allocator.** Peak RSS (~902 MiB) well above post-load
  (~683 MiB) reflects per-`eval()` scratch allocation. A pool amortizes
  it and likely shaves dispatch-prep time too.
- **KV-cache same-eval batching.** `kvcache.memcpy` is free (0.24%), but
  `update` forcing two synchronous evals per layer feeds directly into
  the dispatch problem above; folding the cache write into the forward's
  command buffer removes those evals from the critical path.
- **GPU argmax.** ~2.7% of decode; worth doing after the big levers, not
  before. The `[vocab]` `cpu_to_vec` + scan is ~500 µs/token.
- **`.metallib` precompilation.** Doesn't touch steady-state tok/s, but
  removes the ~43 s cold-start JIT from first-token latency in a shipped
  binary.

None of these are implemented in this measurement pass. They are the
starting backlog for the perf work the numbers now justify.

## Caveats on the comparison

- **Decode windowing differs slightly between the two tools.**
  cider-press discards 8 warmup steps and times the next 120;
  `mlx_lm`'s `generation_tps` averages over all generated tokens (its
  `--warmup` flag is accepted only for CLI symmetry and is informational).
  At 128 tokens on a pipelined backend the skew is small and, if
  anything, flatters cider-press — so the ~10× gap is not an artifact of
  windowing.
- **Run-to-run variance** on cider-press decode (43–65 tok/s) is larger
  than MLX's (555–568) because at ~55 tok/s the synchronous per-dispatch
  path is more exposed to OS scheduling and thermal effects on a laptop.
  The MLX figures are tight.

## Reproduce

```bash
CKPT=/path/to/qwen2.5-0.5b-instruct-4bit
cargo run --release -p cider-press --features profiling -- bench \
  --checkpoint "$CKPT" --max-tokens 128 --warmup 8
uv run scripts/measure_qwen_mlx.py \
  --checkpoint "$CKPT" --max-tokens 128 --warmup 8
```

Run the cider-press bench at least twice; the first invocation of a fresh
process pays the one-time Metal JIT compile in prefill.
