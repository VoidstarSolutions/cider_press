# Qwen2.5-0.5B perf

End-to-end performance of Qwen2.5-0.5B-Instruct-4bit under `cider-press`,
measured via `cider-press bench`, compared against `mlx_lm` on the same
prompt and token budget. Methodology: warm up, then time a steady-state
window; see `docs/ARCHITECTURE.md` for the architecture and forward backlog.

The headline: decode runs **~4.7× slower than MLX** at the full-model
scale. Two changes drove it there. First, the in-graph KV write
(SliceUpdate) collapsed each decode step from ~49 synchronous
`commit + wait` round-trips to **one command buffer per token** (~1.6×).
Second, the decode attention path is now a fused `sdpa_vector` kernel that
reads K/V strided in place — no GQA-broadcast copy, no Kᵀ copy, no
score-matmul/softmax/attn-matmul chain — which lifted decode from ~90 to
**~120 tok/s** (~1.33×) and collapsed the ~36% copy+score-matmul attention
share into a small `gpu.sdpa` bucket. The remaining gap is per-eval cost:
the `tensor.eval` encode/wait split measures **~62% GPU execution, ~38%
CPU-side command-buffer construction** (topo walk, per-op buffer
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
| decode  | ~120 tok/s  | ~560 tok/s  | ~4.7× |

- **Prefill** processes the 39-token prompt in one forward. cider-press
  ranged 263–358 tok/s across warm runs (~290 median); `mlx_lm` 1195–1369
  (~1346). The gap is narrower here because a single 39-token forward
  amortizes the per-dispatch overhead over more compute. Prefill still
  runs the **composed** attention path (the fused vector kernel is the
  decode/T=1 path; prefill fusion is deferred — see `docs/ARCHITECTURE.md`).
- **Decode** is the per-token steady state (the perf-dominant path). Two
  changes lifted it. The in-graph KV write (SliceUpdate) collapsed each
  decode step to one command buffer per token (was ~49 synchronous
  `commit + wait` round-trips), taking decode ~55 → ~90 tok/s (~1.6×).
  Fusing the decode attention into the `sdpa_vector` kernel — which reads
  K/V strided in place, dropping the per-layer GQA-broadcast/Kᵀ copies and
  the q@kᵀ/softmax/probs@v chain — then took decode ~90 → **~120 tok/s**
  (~1.33×). The **remaining ~4.7× gap** to `mlx_lm` is per-eval cost —
  CPU-side dispatch encoding and per-eval scratch allocation.
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
  encoding every dispatch. With decode attention fused into `sdpa_vector`,
  the forward is **~38 dispatches per layer** (per-head qmv projections +
  bias, RoPE, the KV slab write, one fused `sdpa_vector`, the MLP), so
  **on the order of ~900 dispatches per token** across 24 layers plus
  embedding and the tied head — down from ~1000 before fusion. Because
  `commit_and_wait` is synchronous, this CPU time is fully serialized
  ahead of the GPU.
- **`tensor.eval.wait` (~6.2 ms/token, ~62% of eval) is GPU execution.**
  The heavy compute is the 4-bit `qmv` weight matvecs — every projection
  and MLP linear at T=1. The attention score matmuls and softmax that
  previously ran here (`q@kᵀ`, `probs@v`, the two `gemm_bfloat16` calls
  plus softmax) and the ~11 materializing `permute().copy()`s per layer
  are gone: the fused `sdpa_vector` kernel subsumes them, reading strided
  K/V in place.
- **`kvcache.update` is ~0%** — the forced `k.eval()` + `v.eval()` per
  layer (formerly ~1874 ms, ~86% of decode) is gone. The SliceUpdate op
  is built lazily; Metal's hazard tracking serializes the slab write →
  SDPA read within the single per-token command buffer.
- **`argmax` is ~4%** — the bf16 `cpu_to_vec` + scan over the ~151 k
  vocab, ~500 µs/token. Real but not the bottleneck.

### GPU breakdown by op kind (counter-sampled)

`bench --gpu-profile` (with `--features profiling`) runs one real decode
token through per-dispatch stage-boundary counter sampling, attributing
GPU time to each `OpKind`. This is a **perturbed regime**: Apple Silicon
exposes only stage-boundary (not dispatch-boundary) counter sampling, so
each sampled dispatch gets its own compute encoder — extra encoders, and
the lost cross-dispatch overlap inflate absolute time. The shares below
are therefore **within-table** (how the GPU's own work divides), **not**
comparable to the production `tensor.eval.wait` total. Representative
warm run (the first invocation pays the one-time Metal JIT compile;
numbers are from the second):

Post-fusion (decode attention through `sdpa_vector`):

| kind             | total (ms) | dispatches | µs/dispatch | % gpu |
|------------------|-----------:|-----------:|------------:|------:|
| quantized_matmul |      2.301 |        169 |       13.61 | 41.4% |
| binary           |      1.334 |        364 |        3.67 | 24.0% |
| copy             |      0.623 |        146 |        4.27 | 11.2% |
| unary            |      0.373 |        122 |        3.06 |  6.7% |
| reduce           |      0.305 |         49 |        6.22 |  5.5% |
| rope             |      0.233 |         48 |        4.86 |  4.2% |
| sdpa             |      0.213 |         24 |        8.88 |  3.8% |
| slice_update     |      0.155 |         48 |        3.22 |  2.8% |
| gather           |      0.012 |          3 |        4.03 |  0.2% |
| dequantize       |      0.003 |          1 |        2.79 |  0.1% |

Reading it:

- **The attention copy/score-matmul chain has collapsed into `gpu.sdpa`.**
  The old `matmul` (~16%) and `softmax` (~1.3%) buckets are gone entirely:
  the two SDPA score matmuls (`q@kᵀ`, `probs@v`) and the softmax now run
  inside the fused `sdpa_vector` kernel (24 dispatches, one per layer; the
  whole bucket is ~3.8%). `copy` dropped from ~20% to ~11% — the ~11
  materializing `permute().copy()`s per layer that fed the contiguous-only
  matmul kernel are gone, because `sdpa_vector` reads K/V strided in place
  with no GQA-broadcast or Kᵀ copy. The ~36% copy+score-matmul attention
  share has been replaced by a single ~3.8% `sdpa` bucket plus the residual
  copies that remain. Total dispatches/token dropped to ~924 (from ~1166).
- **`quantized_matmul` (qmv) is now the dominant bucket at ~41%** — the
  4-bit weight matvecs (every projection and MLP linear at T=1). Its
  absolute time is unchanged; its *share* rose because the attention
  overhead it competed with is gone. These are the vendored MLX kernel and
  memory-bound, near the bandwidth roofline; little headroom in the kernel
  itself.
- **`binary` (~24%, 364 dispatches)** is the elementwise add/mul scatter
  across residuals, RoPE, and SwiGLU — broadly distributed, not a single
  target. Note this is the **vector/decode path**; prefill still runs the
  composed attention (copy + gemm + softmax), so a prefill `--gpu-profile`
  would still show the old buckets until Plan B fuses the steel/nax
  prefill path.

These are perturbed-regime GPU-*internal* shares. The production
decode-step split is still the ~62% GPU / ~38% CPU-encode from the
encode/wait table above; this subsection only resolves how that GPU half
divides across op kinds.

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

- **Fused decode attention — done.** The decode path now runs through the
  `sdpa_vector` kernel, which reads strided K/V in place and removes the
  ~11 `permute().copy()`s per layer plus the q@kᵀ/softmax/probs@v chain.
  This took decode ~90 → ~120 tok/s and collapsed the ~36% copy+matmul
  attention share into a ~3.8% `gpu.sdpa` bucket. **Remaining:** prefill
  still uses the composed path; fusing it (steel/nax `sdpa_full`, Plan B)
  is the next attention lever. The heavy decode compute is now `qmv`, not
  `gemm`; `qmv` is the vendored MLX kernel and near its bandwidth floor.
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
