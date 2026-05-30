# Qwen2.5-0.5B perf

End-to-end performance of Qwen2.5-0.5B-Instruct-4bit under `cider-press`,
measured via `cider-press bench`, compared against `mlx_lm` on the same
prompt and token budget. Methodology: warm up, then time a steady-state
window; see `docs/ARCHITECTURE.md` for the architecture and forward backlog.

The headline: decode runs **~6.2× slower than MLX** at the full-model
scale. The in-graph KV write (SliceUpdate) collapsed each decode step from
~49 synchronous `commit + wait` round-trips to **one command buffer per
token**, delivering ~1.6×. The remaining gap is per-eval cost, and the
`tensor.eval` encode/wait split now measures it: **~62% GPU execution,
~38% CPU-side command-buffer construction** (topo walk, per-op buffer
allocation, dispatch encoding) — the latter serialized, because
`commit_and_wait` is synchronous. It is not dispatch round-trips.

## Environment

- Machine: Apple M4 Max
- OS: macOS 26.5 (build 25F71)
- Checkpoint: `mlx-community/Qwen2.5-0.5B-Instruct-4bit`, 4-bit affine
  quant, `group_size=64`. The loader requires **BF16** scales/biases; if
  the upstream snapshot ships F16 (it has since these numbers were first
  taken), convert a BF16-scale copy with `mlx_lm.convert`
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

Per-span totals over the timed decode window (120 decode steps; the
profiler is reset after warmup, so the first timed step's eval lands in
the 119 hits below). Representative warm run, decode wall ≈ 1343 ms /
89.3 tok/s. `tensor.eval` is split into two **nested, non-overlapping**
sub-spans: `tensor.eval.encode` (CPU-side command-buffer construction)
and `tensor.eval.wait` (the synchronous GPU `commit_and_wait`); their
sum is `tensor.eval` minus the cheap cache-population pass. `argmax` is
the post-eval CPU `cpu_to_vec` + scan; `kvcache.update` is the lazy
SliceUpdate op-build (24 hits/token, no forced eval).

| span                 | total (ms) | hits | µs/hit | share of decode |
|----------------------|-----------:|-----:|-------:|----------------:|
| tensor.eval          | 1202 |  119 | 10101 | ~89% |
| → tensor.eval.encode |  454 |  119 |  3818 | ~34% |
| → tensor.eval.wait   |  744 |  119 |  6249 | ~55% |
| argmax               |   56 |  119 |   470 | ~4% |
| kvcache.update       | 0.83 | 2856 |  0.29 | ~0% (lazy op build) |

Reading it:

- **`tensor.eval` is ~89% of the decode step, one `eval()` per token.**
  Each token's entire forward — all 24 layers, including the KV writes —
  is encoded into a single command buffer and waited on once. At
  ~10.1 ms/token this is ~6× slower than `mlx_lm`'s ~1.8 ms/token, and
  the encode/wait split locates the cost: **~38% CPU encode, ~62% GPU
  wait.**
- **`tensor.eval.encode` (~3.8 ms/token, ~38% of eval) is CPU-side.**
  Building one command buffer means a topological walk, a fresh
  `MTLBuffer` allocation per op output (no buffer pool — see Memory), and
  encoding every dispatch. The decode forward is **~45 dispatches per
  layer** (per-head qmv projections + bias, RoPE, the KV-write copies,
  the unfused-SDPA copy/matmul/softmax chain, the MLP), so **on the order
  of ~1000 dispatches per token** across 24 layers plus embedding and the
  tied head — not the "~49" an earlier draft cited (that was the
  pre-SliceUpdate `commit + wait` count, ~2 per layer). Because
  `commit_and_wait` is synchronous, this CPU time is fully serialized
  ahead of the GPU.
- **`tensor.eval.wait` (~6.2 ms/token, ~62% of eval) is GPU execution.**
  The heavy compute is the 4-bit `qmv` weight matvecs — every projection
  and MLP linear at T=1 — plus the **~11 materializing `permute().copy()`s
  per layer** that the contiguous-only matmul kernel forces in the
  unfused attention path. `gemm_bfloat16` runs only the two small
  attention score matmuls (`q@kᵀ`, `probs@v`); it is **not** the dominant
  decode kernel, contrary to an earlier draft.
- **`kvcache.update` is ~0%** — the forced `k.eval()` + `v.eval()` per
  layer (formerly ~1874 ms, ~86% of decode) is gone. The SliceUpdate op
  is built lazily; Metal's hazard tracking serializes the slab write →
  SDPA read within the single per-token command buffer.
- **`argmax` is ~4%** — the bf16 `cpu_to_vec` + scan over the ~151 k
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
The synchronous per-dispatch tax is gone — the in-graph KV write
(SliceUpdate) collapsed ~49 `commit + wait` round-trips to one command
buffer per token (~1.6×) and erased `kvcache.update` (~1874 ms → ~0.8 ms).
What remains is the ~10 ms `tensor.eval`, and the encode/wait split shows
it is **~62% GPU execution / ~38% serialized CPU encode**. Both halves
individually exceed `mlx_lm`'s entire ~1.8 ms/token, so no single lever
closes the gap. In measurement-justified priority:

- **Fused attention / fewer materializing copies (GPU, ~62%)** — the
  unfused SDPA forces ~11 `permute().copy()`s per layer to feed the
  contiguous-only matmul kernel. A flash-attention-style kernel that
  reads strided K/V in place removes those copies — both the GPU
  bandwidth they burn and the encode/alloc they add. The heavy compute is
  `qmv`, not `gemm`; `qmv` is the vendored MLX kernel, so audit its launch
  config before assuming the kernel itself is the gap.
- **Buffer pool / allocator (CPU encode, part of ~38%)** — ~1000 fresh
  `MTLBuffer` allocations per token (one per op output, no pool); peak RSS
  (+32% to ~1192 MiB) is the same root cause. A pool recycles scratch
  across tokens and lifts allocation out of the serial encode path.
- **Pipelining (CPU encode, the rest of ~38%)** — `commit_and_wait` is
  synchronous, so the ~3.8 ms of CPU encode never overlaps the GPU.
  Multiple command buffers in flight (or Approach-C lower-level plumbing
  threading `Commands` through the forward) would hide encode behind the
  GPU work it currently blocks.
- **GPU argmax** — ~4% of decode (the `[vocab]` cpu_to_vec + scan,
  ~500 µs/token). After the big levers.
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
