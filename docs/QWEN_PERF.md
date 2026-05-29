# Qwen2.5-0.5B perf

End-to-end performance of Qwen2.5-0.5B-Instruct-4bit under `cider-press`,
measured via `cider-press bench`, compared against `mlx_lm` on the same
prompt and token budget. Methodology: warm up, then time a steady-state
window; see `docs/ARCHITECTURE.md` for the architecture and forward backlog.

The headline: decode runs **~6.2× slower than MLX** at the full-model
scale. The in-graph KV write (SliceUpdate) collapsed each decode step from
~49 synchronous `commit + wait` round-trips to **one command buffer per
token**, delivering ~1.6×. The remaining ~6× gap is per-eval cost (kernel
efficiency, CPU-side dispatch encoding, per-eval scratch allocation) — not
dispatch round-trips.

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
| decode  | ~90 tok/s   | ~560 tok/s  | ~6.2× |

- **Prefill** processes the 39-token prompt in one forward. cider-press
  ranged 263–319 tok/s across warm runs (~290 median); `mlx_lm` 1195–1369
  (~1346). The gap is narrower here because a single 39-token forward
  amortizes the per-dispatch overhead over more compute.
- **Decode** is the per-token steady state (the perf-dominant path).
  The in-graph KV write (SliceUpdate) collapsed each decode step to one
  command buffer per token (was ~49 synchronous `commit + wait` round-trips),
  improving decode from ~55 tok/s to ~90 tok/s (~1.6×). The **remaining
  ~6.2× gap** to `mlx_lm` is per-eval cost — kernel efficiency, CPU-side
  dispatch encoding, and per-eval scratch allocation.
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
profiling facility (representative warm run, decode wall ≈ 1338 ms /
89.6 tok/s). Spans **overlap by construction**: `kvcache.update` is now
fully lazy (builds the SliceUpdate op, no forced eval), so it no longer
overlaps with `tensor.eval`. `tensor.eval` covers all GPU dispatch+wait.
`argmax` is the post-eval CPU `cpu_to_vec` + scan only.

| span           | total (ms) | hits | µs/hit | hits/token | share of decode |
|----------------|-----------:|-----:|-------:|-----------:|----------------:|
| tensor.eval    | 1200 |  120 | 10000 |  1.0 | ~90% |
| kvcache.update |    0.56 | — | — |  — | ~0% (lazy op build only) |
| argmax         |   60 |  119 | 501 |  1.0 | ~4.5% |

Reading it:

- **`tensor.eval` is ~90% of the decode step, with exactly one `eval()`
  per token** (120 hits over 120 steps). Each token's entire forward — all
  24 layers, including the KV cache writes — is encoded into a single
  command buffer and waited on once. At ~10 ms/token this is ~6× slower
  than `mlx_lm`'s ~1.8 ms/token; the cost is per-eval: kernel compute
  (naive `gemm_bfloat16`, no steel tiling/fusion), CPU-side encoding of
  ~49 dispatches/token, and per-eval scratch allocation.
- **`kvcache.update` is now ~0%** — the forced `k.eval()` + `v.eval()`
  per layer (formerly ~1874 ms total, ~86% of decode) is gone. The
  SliceUpdate op is built lazily; Metal's hazard tracking serializes the
  slab write → SDPA read within the single per-token command buffer.
- **`argmax` is ~4.5%** — the bf16 `cpu_to_vec` + scan over the ~151 k
  vocab, ~500 µs/token. Real but not the bottleneck.

## Memory

| mark        | cider-press RSS |
|-------------|----------------:|
| pre-load    | ~14 MiB  |
| post-load   | ~683 MiB |
| post-decode | ~650–893 MiB |
| peak        | ~1192 MiB |

`mlx_lm` peak (Metal allocations, `GenerationResponse.peak_memory`):
**~329 MiB**.

These measure **different things** — cider-press samples process RSS
(via mach `task_info`; includes the 278 MB safetensors read into a host
`Vec` during load, the dequantized weight buffers, and per-`eval()`
scratch allocations with no buffer pool yet), while `mlx_lm` reports the
Metal buffer high-water mark. The peak rose from ~902 MiB to ~1192 MiB
(+32%) with the SliceUpdate change: collapsing to one `eval()` per token
means all forward intermediates of the whole 24-layer pass are live
simultaneously, rather than being freed after each prior synchronous eval.
This is a known regression; a buffer pool will absorb it.

## Hot spots & next levers

The decode step is **per-eval-cost-bound, not dispatch-round-trip-bound**.
The synchronous per-dispatch tax has been removed — the in-graph KV write
(SliceUpdate) collapsed ~49 `commit + wait` round-trips to one command
buffer per token, delivering ~1.6× and eliminating `kvcache.update` as a
cost (~1874 ms → 0.56 ms). But `tensor.eval` remains ~90% of decode at
~10 ms/token, ~6× slower than `mlx_lm`'s ~1.8 ms/token. That residual is
per-eval cost, not dispatch topology. In measurement-justified priority:

- **Buffer pool / allocator** — peak RSS (+32% to ~1192 MiB) directly
  reflects the one-eval-holds-all-intermediates regime. A pool recycles
  scratch buffers across tokens and likely shaves per-eval setup time.
- **Faster GEMM / kernel fusion** — the naive `gemm_bfloat16` dominates
  kernel compute; steel-tiled matmul and fused attention (flash-attn style)
  are the path to closing the ~6× kernel efficiency gap.
- **Per-eval dispatch encoding overhead** — ~49 kernel dispatches are
  encoded per token inside the single command buffer. Reducing encode
  overhead (fused kernels, fewer ops, or Approach-C lower-level plumbing)
  attacks the CPU-side portion of the per-eval cost.
- **GPU argmax** — ~4.5% of decode (the `[vocab]` cpu_to_vec + scan,
  ~500 µs/token). Worth doing after the big levers.
- **`.metallib` precompilation** — doesn't touch steady-state tok/s, but
  removes the ~43 s cold-start JIT from first-token latency in a shipped
  binary.

## Caveats on the comparison

- **Decode windowing differs slightly between the two tools.**
  cider-press discards 8 warmup steps and times the next 120;
  `mlx_lm`'s `generation_tps` averages over all generated tokens (its
  `--warmup` flag is accepted only for CLI symmetry and is informational).
  At 128 tokens on a pipelined backend the skew is small and, if
  anything, flatters cider-press — so the ~6.2× gap is not an artifact of
  windowing.
- **Run-to-run variance** on cider-press decode is larger than MLX's
  because the synchronous per-eval path is more exposed to OS scheduling
  and thermal effects on a laptop. The MLX figures are tight.

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
