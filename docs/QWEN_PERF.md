# Qwen2.5-0.5B perf

End-to-end performance of Qwen2.5-0.5B-Instruct-4bit under `cider-press`,
measured via `cider-press bench`, compared against `mlx_lm` on the same
prompt and token budget. Methodology: warm up, then time a steady-state
window; see `docs/ARCHITECTURE.md` for the architecture and forward backlog.

The headline: decode runs **~1.20× slower than MLX** at the full-model
scale (~473 vs ~569 tok/s), down from ~1.35× before decode copy-elimination
and from ~2.3× two branches ago — the fused `RMSNorm`, **async decode
pipelining**, and **decode copy-elimination (S1)** stack to **~1.95×** over
the pre-fusion ~243 baseline. Three
prior changes set the stage — the in-graph KV write (SliceUpdate)
collapsed each step to **one command buffer per token** (~1.6×), the
fused `sdpa_vector` decode-attention kernel lifted decode to ~120 tok/s
(~1.33×), and a **cross-token buffer pool** (recycling op-output scratch
instead of allocating ~1000 fresh `MTLBuffer`s per token) took decode to
~210 tok/s. The pool was inert on its own (3% hit) until the enabling fix:
**`KvCache::update` no longer chains each `SliceUpdate` off the prior
op-tensor** (it bases on the slab leaf), which stopped the cache from
pinning every step's projection graph for the whole decode — together they
hit **98% pool hit-rate**, cut the `tensor.eval` CPU-encode share from ~38%
to ~11%, and resolved the +32% peak-RSS regression (**~1192 → ~900 MiB**).
This pass then moved **next-token selection on-GPU** (in-graph `ArgReduce`):
the ~444 µs/token CPU `cpu_to_vec` + 151 k-wide scan and the 302 kB logits
readback are gone, replaced by a sub-µs GPU reduction folded into the decode
command buffer + a 4-byte index readback, lifting decode **~217 → ~243
tok/s**, and the fused `RMSNorm` then **~243 → ~302** (~1.26×, first fusion
landed). **Async decode pipelining** then committed the next token's command
buffer without waiting and chained the argmax id on-GPU, so token N+1's CPU
graph-build + encode overlaps token N's GPU execution. Re-measured on this
branch with both stacked: decode is **~420 tok/s** at the production depth-1
vs **~317 tok/s** synchronous (depth-0) on the same post-`RMSNorm` base — a
**~1.33×** pipelining overlap (down from the ~1.40× measured on the pre-fusion
~243 baseline: `RMSNorm` already shortened the GPU critical path, so there is
less CPU encode left to hide). Decode copy-elimination (S1) then lifted
decode to **~473 tok/s** (depth-1) / **~352 tok/s** (depth-0). See
**Async decode pipelining** and **Decode copy elimination (S1)** below.

The remaining gap is **critical-path-bound**: `tensor.eval` is ~90% of the
decode step, and its GPU wait is a deep chain of **dependent dispatches** —
the 4-bit `qmv` matvecs plus the many small ops gated behind them. A
concurrent-encoder spike (see the **Concurrent-encoder spike** section) tested
whether the serial Metal encoder was throttling this and found it is **not**:
with correct hazard barriers, overlapping dispatch is a net loss. The levers
are fusion (fewer dependent dispatches — `RMSNorm` done, decode `copy`s now
elided by S1), faster kernels, and pipelining (now landed) — not the dispatch
model.

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
| prefill | ~376 tok/s  | ~1346 tok/s | ~3.6× |
| decode  | ~473 tok/s  | ~569 tok/s  | ~1.20× |

- **Prefill** processes the 39-token prompt in one forward. cider-press
  ranged 346–391 tok/s across warm runs (~376 median); `mlx_lm` ~1346.
  Note this branch does **not** target prefill — a single forward with an
  empty pool gets no cross-eval reuse, so the figure is within run-to-run
  variance of the prior ~290 (263–358) baseline, not a pool effect.
  Prefill still runs the **composed** attention path (the fused vector
  kernel is the decode/T=1 path; prefill fusion is deferred — see
  `docs/ARCHITECTURE.md`).
- **Decode** is the per-token steady state (the perf-dominant path). The
  prior SliceUpdate (~55 → ~90 tok/s, ~1.6×) and fused `sdpa_vector`
  (~90 → ~120 tok/s, ~1.33×) changes preceded this branch. The
  **cross-token buffer pool** then took decode **~120 → ~210 tok/s**
  (~1.75×, range 203–217) by recycling op-output scratch (98% hit-rate),
  lifting the ~1000-allocations-per-token cost out of the serialized
  command-buffer encode. The pool only paid off once `KvCache::update`
  stopped chaining `SliceUpdate`s (which had pinned every step's graph for
  the whole decode — see Decode-step breakdown). The **GPU argmax** then
  lifted decode **~217 → ~243 tok/s** (range 238–246) by removing the
  ~444 µs/token CPU vocab scan + 302 kB logits readback (see **GPU argmax**).
  The **fused `RMSNorm`** then lifted decode **~243 → ~302 tok/s** (~1.26×)
  by collapsing the 6-dispatch composition (×49 calls/token) to one
  `rms_single_row` each (see **Hot spots & next levers**). **Async
  decode pipelining** lifted the production path to **~420 tok/s** (depth-1,
  range 413–423; ~317 tok/s synchronous depth-0 on the same post-`RMSNorm`
  base, a ~1.33× overlap). **Decode copy-elimination (S1)** then lifted
  **~420 → ~473 tok/s** (~1.13×) by eliding the ~144 degenerate
  attention-permute copies (zero-copy views, no kernel work); depth-0 likewise
  rose from ~317 → ~352 tok/s. The **remaining ~1.20× gap** to `mlx_lm`
  (~569) is critical-path-bound — the chain of dependent dispatches in
  `tensor.eval.wait`, not CPU encode.
- Load time (excluded from tok/s): ~0.17 s.

**Cold-start note:** the very first forward of a fresh process triggers
Metal JIT compilation of the quantized kernels — run 1's prefill measured
**42.6 s** (0.9 tok/s) for this reason. Metal caches compiled libraries
to disk cross-process, so runs 2+ prefill in ~130 ms. The 42.6 s is a
one-time first-token-latency cost (a `.metallib` precompile would
amortize it; deferred — see `docs/ARCHITECTURE.md`), not representative of
prefill compute, and is excluded from the table above.

## Async decode pipelining (detach-on-eval)

Decode was synchronous: one `commit_and_wait` per token, and the next
token's input was the previous argmax read back to the CPU — so the CPU
graph-build + encode of token N+1 could not overlap token N's GPU work.
This branch makes `eval` non-blocking (`Tensor::eval_async` →
`commit` + a `PendingEval` handle waited later) and **chains the argmax id
on-GPU** (the `[1,1]` U32 argmax feeds the next forward's embedding gather
directly — no readback), so token N+1 is encoded and committed while token
N runs. Cross-command-buffer ordering rests on the serial queue's automatic
hazard tracking; the generator drives a depth-1 lookahead (greedy decode is
inherently depth-1 — `--inflight-depth` exposes the knob).

**The retention trap.** On-GPU chaining first *regressed* decode: feeding
the previous argmax (an op tensor) into the next forward made each token's
graph hold an `Arc` to the prior token's argmax, which transitively pinned
its entire upstream graph — a backward chain through the whole decode.
Buffer-pool reuse collapsed **98% → 0%** (every ~975 op-output buffers
re-allocated per token), `tensor.eval_async` CPU encode ballooned
**~0.55 → ~3.2 ms/token**, and depth-1 ran **~234 tok/s — below the ~243
synchronous baseline**. The fix mirrors MLX's eval semantics:
**detach-on-eval** — after `eval`/`eval_async` populate a node's cache, clear
its op inputs so an evaluated tensor stops retaining its producers (MLX's
`array::detach()`). The generator needed no change.

Measured (M4 Max, warm, interleaved A/B vs the pre-branch synchronous
generator at `260fa97`; `--max-tokens 256 --warmup 16`, 101 timed steps,
median of 3):

| build | decode tok/s | pool hit | `eval_async` encode |
|-------|-------------:|---------:|--------------------:|
| synchronous baseline (`260fa97`) | ~244 | 98.1% | — |
| depth-1, on-GPU chain, **no detach** | ~234 | **0.0%** | ~3.2 ms |
| depth-0 (synchronous, detach) | ~237 | 98.1% | ~0.52 ms |
| **depth-1 (pipelined, detach)** | **~341** | 97.3% | ~0.57 ms |

Detach restores the pool (depth-0 ≈ baseline, confirming the `eval_async`
machinery itself is neutral); the depth-1 gain is the pipelining overlap:
**~244 → ~341 tok/s, ~1.40×**. The win exceeds the original ~1.14× estimate
because pipelining hides *all* per-token CPU work — the `forward()` lazy
graph-build, not just the `tensor.eval.encode` slice — under the GPU. At
depth-1 the blocking wait drops to ~1.9 ms/token (vs ~3.2 ms synchronous),
the GPU time not yet hidden. Greedy decode is depth-1; `--inflight-depth 2`
measured no further gain. Design:
`docs/superpowers/specs/2026-06-03-async-pipelining-design.md` and
`…-eval-detach-design.md`.

**Re-measured post-`RMSNorm`** (this branch rebased onto fused `RMSNorm`; M4
Max warm, `--max-tokens 128 --warmup 8`, median of 5 — the A/B table above
predates the fusion, which now lives in the base):

| build | decode tok/s | pool hit |
|-------|-------------:|---------:|
| depth-0 (synchronous, detach) | ~317 (308–323) | 97.2% |
| **depth-1 (pipelined, detach)** | **~420 (413–423)** | 97.2% |

The pipelining overlap is now **~1.33×** (was ~1.40× pre-fusion): the fused
`RMSNorm` shortened the GPU critical path, so there is less CPU encode left to
hide under GPU execution. `mlx_lm` on the same prompt measured **~566 tok/s**
(560–577, median of 5) — gap **~1.35×** (pre-S1 snapshot; see **Decode copy
elimination (S1)** for the updated figures).

## Decode copy elimination (S1)

At decode (T=1) every attention `permute` is contiguous-modulo-size-1: the
permute only moves the size-1 T axis, so the resulting strides are already
contiguous if unit-sized dimensions are ignored. The exact-contiguity check
in `Tensor::copy()` was the blocker — it required *every* stride to match the
canonical row-major layout, so it always materialized these permutes as
`OpKind::Copy` dispatches. MLX's `is_row_contiguous` ignores unit dims; it
never paid this cost. The fix adds `Strides::is_contiguous_ignoring_unit_dims`
and returns a zero-copy canonical-strided view over the same storage when the
check passes. `rope` was the only downstream consumer that required a
materialised copy (it previously checked for the non-contiguous path); it was
taught to accept a contiguous view directly. A byte-offset-0 guard preserves
the existing invariant that a copy starting mid-buffer still materializes (not
a unit-dim case). Genuine reorders — prefill (T>1) and any permute that truly
rearranges data — still materialize. No model-layer changes were needed.

The measured effect (M4 Max, warm, `--max-tokens 128 --warmup 8`, median of 5,
branch `perf/strided-copy-elim`): `gpu.copy` fell from **146 dispatches /
11.2% GPU** to **2 dispatches / 0.2%**, and total dispatches/token from
**~680 to ~536**. CPU encode time fell from **~426 → ~308 µs/token**, the
async wait from **~1810 → ~1624 µs/token**. Production decode (depth-1) rose
from **~420 → ~473 tok/s** (466–481, median of 5); synchronous decode
(depth-0) from **~317 → ~352 tok/s** (349–356, median of 3), a ~1.34×
pipelining overlap. `mlx_lm` on the same prompt measured **~569 tok/s**
(568.5–570.6, median of 3) — gap **~1.20×** (was ~1.35× pre-S1). The greedy
parity gate passed: copy-elision is byte-identical (the views are over the
same storage). Buffer pool hit rate 96.9%; peak RSS ~878 MiB (was ~888).

S1 only targets the decode path. Prefill (T>1) permutes are genuine reorders
and still materialize via `OpKind::Copy`; the prefill strided-kernel work
(rope strides, `slice_update` strided src, gemm transpose+broadcast, qmv
strided — S2–S5) is pending. Design:
`docs/superpowers/specs/2026-06-03-strided-copy-elimination-design.md`.

## Decode-step breakdown

Per-span totals over the timed decode window (~109 timed steps; the
profiler is reset after warmup, so the spans below show 108 hits).
Representative warm run, decode wall ≈ 446 ms / 243 tok/s — a
**pre-fusion, pre-pipelining snapshot** (the headline ~420 tok/s reflects
the fused `RMSNorm` and async pipelining that landed after this breakdown
was captured; the span *shape* below is from the synchronous path — RMSNorm
fusion removed five dispatches from the `tensor.eval.encode` side, and
pipelining moved the wait off the critical path). `tensor.eval` is split into two **nested,
non-overlapping** sub-spans: `tensor.eval.encode` (CPU-side command-buffer
construction) and `tensor.eval.wait` (the synchronous GPU
`commit_and_wait`); their sum is `tensor.eval` minus the cheap
cache-population pass. `argmax` now times only the **4-byte index
readback** after `eval()` — the GPU `ArgMax` reduction itself is folded
into `tensor.eval.wait` (see **GPU argmax** below); `kvcache.update` is
the lazy slab-based SliceUpdate op-build (24 hits/token, no forced eval).

| span                 | total (ms) | hits | µs/hit | share of decode |
|----------------------|-----------:|-----:|-------:|----------------:|
| tensor.eval          |  401 |  108 |  3709 | ~90% |
| → tensor.eval.encode |   55 |  108 |   512 | ~12% |
| → tensor.eval.wait   |  342 |  108 |  3166 | ~77% |
| argmax               | 0.08 |  108 |  0.78 | ~0.02% (4-byte readback) |
| kvcache.update       | 15.9 | 2592 |   6.1 | ~4% (lazy op build) |

Reading it:

- **`tensor.eval` is ~90% of the decode step, one `eval()` per token,**
  and is now **GPU-bound**: the encode/wait split is **~14% CPU encode,
  ~86% GPU wait** (was ~38%/62% before the buffer pool). The pool moved the
  decode bottleneck off the CPU encode path; the GPU argmax then removed the
  post-eval CPU scan, so `tensor.eval`'s *share* of the (smaller) step rose.
- **`tensor.eval.encode` (~0.51 ms/token, ~12% of the step) is CPU-side**
  and now small. Building one command buffer still means a topo walk and
  encoding every dispatch (~974 dispatches/token across 24 layers plus
  embedding and the tied head, now plus the one `ArgReduce`), but the per-op
  `MTLBuffer` allocation — formerly ~1000 fresh `newBufferWithLength:` calls
  per token — is now a **buffer-pool hit ~98% of the time** (see Memory), so
  allocation is no longer the dominant encode cost.
- **`tensor.eval.wait` (~3.2 ms/token, ~77% of the step) is GPU execution**
  and is the bottleneck. The heavy compute is the 4-bit `qmv` weight
  matvecs — every projection and MLP linear at T=1 — the vendored MLX
  kernel. The audit (see `## qmv audit`) found these are *not* uniformly
  roofline-bound: the large-N linears (gate/up/down, lm_head) saturate near
  the ~464 GB/s empirical ceiling, but the small-N decode projections
  (k/v_proj ~13 GB/s, q/o_proj ~85 GB/s) are occupancy-starved by the
  generic launch grid and carry real headroom.
- **`kvcache.update` is ~3%** — the lazy SliceUpdate op-build, now based on
  the slab leaf rather than chained off the prior op-tensor. Chaining had
  pinned every step's K/V projection graph (and its residual-path
  intermediates) for the whole decode, which kept the buffer pool from ever
  reclaiming per-step scratch (3% hit). Basing on the slab frees each
  step's graph one step later, lifting the pool to 98% hit. The slab-base
  build is marginally costlier per call (~5.7 vs ~0.3 µs) but still ~3% of
  the step. Successive `update`s must now be separated by an `eval`
  (decode evals each token; a fail-loud guard enforces it).
- **`argmax` is now ~0.02% (~0.8 µs/token)** — next-token selection moved
  on-GPU (in-graph `ArgReduce`). The span now times only the 4-byte index
  readback; the actual reduction over the ~151 k vocab is folded into
  `tensor.eval.wait` (sub-µs, memory-bound), and the old 302 kB logits
  readback + bf16 `cpu_to_vec` + 151 k-wide CPU scan (~444 µs/token) are
  gone. See **GPU argmax** below.

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
  materializing `permute().copy()`s that fed the contiguous-only matmul
  kernel are gone (copy bucket: 266→146 dispatches, ~120 total / ~5 per
  layer): the GQA-broadcast, Kᵀ-transpose, and the two K/V SDPA-read
  copies per layer, because `sdpa_vector` reads K/V strided in place. The ~36% copy+score-matmul attention
  share has been replaced by a single ~3.8% `sdpa` bucket plus the residual
  copies that remain. Total dispatches/token dropped to ~974 (from ~1166; see table).
- **`quantized_matmul` (qmv) is now the dominant bucket at ~41%** — the
  4-bit weight matvecs (every projection and MLP linear at T=1). Its
  absolute time is unchanged; its *share* rose because the attention
  overhead it competed with is gone. These are the vendored MLX kernel, and
  per the `## qmv audit` the bandwidth they attain is **N-dependent**: the
  large-N linears are roofline-bound (~464 GB/s empirical ceiling, near the
  M4 Max ~410–546 GB/s spec) with little kernel headroom, but the small-N
  decode projections (k/v_proj ~13 GB/s, q/o_proj ~85 GB/s) are
  occupancy-starved by the generic `(1, ceil(N/8), 1)` grid and *do* have
  headroom — so "little headroom" holds only for the large-N shapes, not the
  decode-dominant small-N projections.
- **`binary` (~24%, 364 dispatches)** is the elementwise add/mul scatter
  across residuals, RoPE, and SwiGLU — broadly distributed, not a single
  target. Note this is the **vector/decode path**; prefill still runs the
  composed attention (copy + gemm + softmax), so a prefill `--gpu-profile`
  would still show the old buckets until Plan B fuses the steel/nax
  prefill path.

These are perturbed-regime GPU-*internal* shares. The production
decode-step split is now ~86% GPU wait / ~14% CPU-encode (the buffer pool
moved allocation off the encode path); this subsection only resolves how
that GPU half divides across op kinds. The op-kind breakdown itself is
unchanged by the buffer pool and the KvCache slab-base collapse — both act
on CPU-side allocation/retention, not the GPU dispatches.

**Post copy-elimination (S1)** (M4 Max, warm, `--max-tokens 128 --warmup 8`,
`perf/strided-copy-elim`; counter-sampled, perturbed regime, one token):

| kind             | total (ms) | dispatches | µs/dispatch | % gpu |
|------------------|-----------:|-----------:|------------:|------:|
| quantized_matmul |      1.667 |        169 |        9.86 | 58.4% |
| binary           |      0.430 |        168 |        2.56 | 15.1% |
| rope             |      0.166 |         48 |        3.46 |  5.8% |
| sdpa             |      0.195 |         24 |        8.13 |  6.8% |
| reduce (rms_norm)|      0.162 |         49 |        3.31 |  5.7% |
| slice_update     |      0.149 |         48 |        3.10 |  5.2% |
| unary            |      0.064 |         24 |        2.67 |  2.2% |
| copy             |      0.006 |          2 |        3.00 |  0.2% |
| gather           |      0.013 |          3 |        4.33 |  0.5% |
| dequantize       |      0.003 |          1 |        3.00 |  0.1% |

Total dispatches/token: **~536** (was ~680). The `copy` bucket fell from 146
dispatches / 11.2% to **2 dispatches / 0.2%** — the ~144 degenerate
attention-permute copies are now elided as zero-copy views. `quantized_matmul`
rose from ~41% to **~58%** only because the copy work it competed with is
gone; its absolute time is lower (~1.667 vs ~2.301 ms) because the GPU is no
longer serialized behind the copy chain. CPU encode likewise fell from ~426 to
~308 µs/token, and the GPU wait from ~1810 to ~1624 µs/token.

### GPU argmax

Next-token selection moved on-GPU. Greedy decode previously did
`logits.eval()` → a 302 kB `cpu_to_vec` of the `[vocab]` row → a 151 k-wide
CPU `fold` to find the max index — ~444 µs of serial CPU work per token
(~9.5% of the decode step). That is replaced by an in-graph `ArgReduce`
(MLX's `argmax_bfloat16` from `arg_reduce.metal`) that evaluates in the
**same command buffer as the lm_head**: the GPU reduces the vocab row to a
single `u32` index, and only that 4-byte index is read back.

Effect on the spans:

- The `argmax` profile span now sits *after* `eval()` and times only the
  4-byte readback — it dropped from **~444 µs/token to ~0.8 µs/token**
  (~0.73–0.94 µs across warm runs).
- The GPU reduction is a tiny, memory-bound pass over one vocab row, folded
  into `tensor.eval.wait` (sub-µs; not separately visible against the ~3.2 ms
  qmv-dominated wait).
- The 302 kB per-token logits readback is gone.

Decode throughput moved **~217 → ~243 tok/s** (warm runs measured
238–245.5 tok/s; representative ~243). The gain is close to the old argmax
share — removing ~444 µs of serial CPU work from a ~4.6 ms step — and lands
where predicted. Greedy output is unchanged: the kernel is MLX's own, so it
matches `mlx_lm` by construction (`generator_parity_greedy_qwen2` passes).
This also removes the logits-readback dependency, which is the **enabler**
for keeping the next-token id on-GPU into the following embedding gather
(eliminating the per-token CPU↔GPU sync entirely — future work, backlog #5).

## Memory

| mark        | cider-press RSS |
|-------------|----------------:|
| pre-load    | ~14 MiB  |
| post-load   | ~678 MiB |
| post-decode | ~900 MiB |
| peak        | ~900 MiB |

`mlx_lm` peak (Metal allocations, `GenerationResponse.peak_memory`):
**~329 MiB**.

These measure **different things** — cider-press samples process RSS
(via mach `task_info`; includes the 278 MB safetensors read into a host
`Vec` during load and the dequantized weight buffers), while `mlx_lm`
reports the Metal buffer high-water mark. The +32% regression (peak had
risen ~902 → ~1192 MiB when SliceUpdate collapsed the token to one
`eval()`, holding all 24-layer intermediates live at once) is **resolved**:
peak is back to **~900 MiB**. The fix was not the pool's byte cap (the
free-list high-water is only ~94 MiB, well under the 256 MiB cap) but the
KvCache slice-update collapse — once per-step graphs stop being pinned by
the cache chain, each token's intermediates free as the next token starts,
so they no longer accumulate. The remaining gap to `mlx_lm`'s ~329 MiB is
**within-eval reuse**: MLX frees intermediates *during* eval (depth-first)
and reuses buffers within a single token, while cider-press submits a token
as one command buffer with every intermediate live simultaneously, so the
pool reclaims cross-token only. Mid-eval freeing is a separate future
lever.

## Hot spots & next levers

The decode step is **critical-path-bound**. The CPU-side taxes are gone —
the SliceUpdate collapsed the dispatch round-trips, the buffer pool moved
per-op allocation off the serial encode path (encode ~12% of the step), and
the GPU argmax removed the post-eval vocab scan. What remains is
`tensor.eval.wait` (~77% of decode): a deep chain of ~900 *dependent*
dispatches/token — the 4-bit `qmv` matvecs plus the small ops gated behind
them — where each link pays GPU launch + inter-dispatch latency. The
**Concurrent-encoder spike** confirmed this is not removable serialization:
correct concurrent dispatch is a net loss. So the levers shorten or speed
the critical path. In measurement-justified priority:

- **Kernel fusion / copy elision — the leading lever (in progress).** Fewer
  dependent dispatches ⇒ a shorter critical path ⇒ less inter-dispatch
  latency. **`RMSNorm` is fused (done, ~1.26×):** the composed
  square→mean→rsqrt→mul→mul (6 dispatches × 49 calls = ~294/token) is now
  one `rms_single_row` dispatch each (~49/token), dropping decode to ~730
  dispatches/token — `tensor.eval.wait` ~3290 → ~2902 µs, encode ~440 → ~293
  µs, decode ~240 → ~302 tok/s. This also realigned us with `mlx.nn.RMSNorm`,
  which uses the same fused `mx.fast.rms_norm`. **Decode copy-elimination S1
  (done, ~1.13×):** the ~144 degenerate attention-permute `copy` dispatches at
  T=1 are now elided as zero-copy views (`Strides::is_contiguous_ignoring_unit_dims`),
  dropping decode from ~420 → ~473 tok/s and total dispatches/token from ~680
  → ~536; `gpu.copy` is now 2 dispatches / 0.2% GPU (was 146 / 11.2%). The
  **remaining ~1.20× gap** to `mlx_lm` (~569) is still critical-path-bound.
  What remains of backlog #9 (strided-aware kernels) is the **prefill** strided
  path (S2–S5): rope strides, `slice_update` strided src, gemm
  transpose+broadcast, qmv strided — none of these affect decode, which is
  down to 2 copy dispatches. Note SwiGLU's `silu*up` and RoPE are *not* fusion
  targets: MLX composes SwiGLU itself, and we already dispatch RoPE/decode-SDPA
  fused.
- **Concurrent dispatch encoder — tested, rejected.** Flipping serial →
  concurrent dispatch and inserting correct hazard barriers is byte-identical
  to serial but ~13% *slower* (~900/975 dispatches need a barrier; almost
  nothing overlaps). The serial encoder is not the bottleneck. See
  **Concurrent-encoder spike**.
- **Buffer pool / allocator — done (~1.75× with the KvCache fix).** A
  cross-token free-list recycles op-output scratch (98% decode hit), and
  the enabling `KvCache::update` slab-base change stopped the cache
  pinning per-step graphs. Together: decode ~120 → ~210 tok/s, CPU-encode
  share ~38% → ~11%, peak RSS ~1192 → ~900 MiB. The pool reclaims
  **cross-token** only (one command buffer per token keeps a token's whole
  graph live); **within-eval reuse** (mid-eval freeing, as MLX does) is the
  next RSS lever toward `mlx_lm`'s ~329 MiB.
- **Faster `qmv` kernels — speeds each critical-path link.** The 4-bit
  `qmv` weight matvecs are the heaviest dispatches in the chain. The
  `## qmv audit` localized the headroom: the large-N linears are
  roofline-bound (~464 GB/s, near the M4 Max spec), but the small-N decode
  projections (k/v_proj ~13 GB/s, q/o_proj ~85) are occupancy-starved by the
  generic launch grid — lever (b), with fast-path absence (lever a)
  compounding. A small-N qmv launch-config fix (and/or fast-path-via-padding)
  is dispatch-side and parity-preserving; it shortens each `qmv` link's
  GPU time but does not reduce the *number* of dependent dispatches (that is
  fusion, above).
- **Pipelining (overlap CPU encode with GPU) — done (~1.34× post-S1).**
  The depth-1 async pipeline overlaps token N+1's CPU graph-build + encode
  with token N's GPU execution. Post-S1, encode is ~308 µs/token and the
  pipelining overlap is ~1.34× (~352 → ~473 tok/s). Lower priority now that
  encode is small relative to the GPU wait (~1624 µs/token).
- **Fused prefill attention** — prefill still uses the composed path;
  fusing it (steel/nax `sdpa_full`, Plan B) is the prefill attention lever.
- **GPU argmax — done (~217 → ~243 tok/s).** Next-token selection moved
  on-GPU (in-graph `ArgReduce`): the ~444 µs/token CPU `cpu_to_vec` + 151 k
  scan and the 302 kB logits readback are gone, replaced by a sub-µs GPU
  reduction folded into `tensor.eval.wait` + a 4-byte index readback (the
  `argmax` span fell to ~0.8 µs/token). This also clears the logits-readback
  dependency, the enabler for keeping the id on-GPU into the next embedding
  gather (per-token sync removal — future work).
- **`.metallib` precompilation** — doesn't touch steady-state tok/s, but
  removes the ~43 s cold-start JIT from first-token latency in a shipped
  binary.

## Concurrent-encoder spike (tested, rejected)

A spike (branch `spike/concurrent-encoder`, 2026-06-02, M4 Max) asked
whether the serial Metal compute encoder was throttling decode. The
production encoder is opened serial (`computeCommandEncoder` =
`MTLDispatchTypeSerial`), which inserts an implicit barrier between every
dispatch; the hypothesis was that the ~975 decode dispatches/token were
needlessly serialized, with MLX's concurrent encoder
(`computeCommandEncoderWithDispatchType(Concurrent)`) overlapping them.
**The hypothesis was wrong** — decode is critical-path-bound, so concurrent
dispatch does not help.

Three measurements settle it:

| variant                                 | `tensor.eval.wait` | decode     | output                   |
|-----------------------------------------|-------------------:|-----------:|--------------------------|
| serial (baseline)                       |       ~3.29 ms/tok | ~240 tok/s | correct                  |
| concurrent, **no** barriers             |       ~1.03 ms/tok | ~510 tok/s | **garbage**              |
| concurrent, **correct** hazard barriers |       ~3.59 ms/tok | ~210 tok/s | byte-identical to serial |

- The **no-barrier** run is fast only because it ignores data
  dependencies — it computes wrong answers. Its ~1.03 ms is the throughput
  floor *if the work had no dependencies*; it is **not** an achievable
  target. (It misled an earlier draft of this doc into calling the serial
  encoder the dominant lever — corrected here.)
- With **correct** buffer-level hazard tracking (a barrier before any
  dispatch that reads a buffer an earlier dispatch wrote, MLX's
  `set_input_array`/`maybeInsertBarrier` model), the output is
  byte-identical to serial — but **~900 of the ~975 dispatches/token need a
  barrier**. The decode graph (the residual stream plus each layer's
  q→rope→slice-update→sdpa→o-proj and rms→gate→silu→mul→down chains) is
  ~92% a sequential dependency chain. Only ~7.7% of dispatches are
  independent, and `memoryBarrierWithScope` is a *global* barrier that
  over-serializes anyway. Net effect: concurrent dispatch is **~13% slower**
  than serial (extra explicit-barrier + hazard-tracking cost, ~nothing
  overlaps).

**Conclusion: the serial encoder is not the bottleneck.** Decode is bound
by the *critical path* through a deep chain of dependent dispatches — each
link's GPU launch plus inter-dispatch latency, summed over ~900 ordered
ops. The ~2 ms between the predicted pure-`qmv` compute (~1.27 ms, see
`## qmv audit`) and the ~3.3 ms wait is that dependent-dispatch latency,
**not** removable serialization. This re-points the remaining work at
*shortening and speeding the critical path* (fusion, faster kernels,
pipelining — see **Hot spots & next levers**), not the dispatch model.

One defensive change landed from the spike regardless: the KV slabs are
zero-filled at construction (`KvCache::new`), so any future dispatch
reordering reads deterministic zero rather than uninitialized garbage. The
concurrent encoder itself is **shelved**.

## Caveats on the comparison

- **Decode windowing differs slightly between the two tools.**
  cider-press discards 8 warmup steps and times the remainder (~109 of
  the 117 generated for this prompt); `mlx_lm`'s `generation_tps` averages
  over all generated tokens (its `--warmup` flag is accepted only for CLI
  symmetry and is informational). On a pipelined backend the skew is small
  and, if anything, flatters cider-press — so the ~1.20× gap is not an
  artifact of windowing.
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

## qmv audit

This audits backlog item #4 (the decode 4-bit `qmv` matvecs). It runs two
microbenches at the five Qwen2.5-0.5B decode shapes plus a forced-fast
control: cider-press `affine_qmv_bf16` (`crates/cider-press-kernels/tests/qmv_dispatch_bench.rs`)
and MLX `mx.quantized_matmul` (`scripts/measure_qmv_mlx.py`), gs=64, bits=4.

Read the columns carefully — the two harnesses measure different things:

- **Per-dispatch wall µs is the only apples-to-apples cider-vs-MLX
  comparator.** Both harnesses do one submit + one wait per dispatch
  (cider: `commit_and_wait`; MLX: `mx.eval`), so both are
  **dispatch-latency-bound** — the ~90–360 µs is dominated by the
  submit+wait floor, not the matvec.
- **cider's GPU-counter `eff BW` is the kernel-only truth** — pure GPU
  execution from stage-boundary counters, excluding submit/wait. This is
  the kernel-internal signal. (Small-shape GPU-counter readings vary
  run-to-run from thermal/scheduling perturbation; the floor — lowest ns
  over several runs — is the representative kernel time and is reported
  below.)
- MLX's wall-derived BW is **overhead-bound** (the ~100 µs eval floor
  dominates small shapes), so it is **not** comparable to cider's
  GPU-counter BW — see the note under the table.

| shape (K×N) | linear       | variant | cider GPU-counter eff BW | cider wall µs | MLX wall µs |
|-------------|--------------|---------|-------------------------:|--------------:|------------:|
| 896×896     | q/o_proj     | gen     |  85 GB/s |  ~93 | ~98  |
| 896×128     | k/v_proj     | gen     |  13 GB/s |  ~92 | ~97  |
| 896×4864    | gate/up_proj | gen     | 306 GB/s | ~100 | ~106 |
| 4864×896    | down_proj    | gen     | 272 GB/s | ~105 | ~109 |
| 896×151936  | lm_head      | gen     | 464 GB/s | ~257 | ~268 |
| 1024×1024   | forced-fast  | fast    | 151 GB/s |  ~92 | ~99  |

(MLX wall-clock BW, overhead-bound, for context only — not comparable to
cider's GPU-counter column: q/o ~5, k/v ~1, gate/up ~23, down ~23, lm_head
~287, fast ~6 GB/s. The ~100 µs `mx.eval` floor swamps every shape but
lm_head, which is why these are tiny.)

Findings:

1. **The qmv dispatch path is not where cider-press loses to mlx_lm.** On
   the only comparable axis — per-dispatch wall µs — cider-press is at or
   slightly ahead of MLX at every shape (~92 vs ~98 µs at q/o_proj, ~257 vs
   ~268 at lm_head). Both are latency-bound on the submit+wait floor; the
   matvec itself is a small fraction of that wall time.
2. **Kernel-internal bandwidth (cider GPU-counter) scales with N (output
   dim).** Small-N projections are badly under-utilized — k/v_proj (N=128)
   at ~13 GB/s, q/o_proj (N=896) at ~85 — while large-N linears saturate:
   gate/up (N=4864) ~306, down ~272, lm_head (N=151936) ~464 GB/s. The
   generic qmv grid is `(1, ceil(N/8), 1)` threadgroups, so small N
   launches too few threadgroups to fill the GPU. This is **occupancy
   starvation (lever b)**.
3. **The forced-fast control adds a second lever.** At a comparable size,
   the fast path (1024×1024, ~151 GB/s) runs ~1.7× the generic q/o_proj
   (896×896, ~85 GB/s). Since every Qwen2.5-0.5B decode shape has
   `K ∈ {896, 4864}` (neither a 512-multiple), the fast path is never
   selected — **fast-path absence (lever a)** also contributes.
4. **The empirical kernel ceiling is ~464 GB/s** (lm_head GPU-counter),
   approaching the M4 Max ~410–546 GB/s spec roofline — not the bogus
   ~125 GB/s `copy_v_bf16` probe (which several qmv shapes exceed, so it is
   not bandwidth-saturating; see the plan's execution amendment). So the
   large-N linears are essentially roofline-bound, and the headroom is
   concentrated in the small-N projections.
5. **This microbench cannot arbitrate the decode gap** — it is
   latency-bound (one submit+wait per call), whereas decode runs many ops
   in one command buffer with no per-op wait. The decode-gap arbiter is
   Approach B (next): a steady-state attribution of a real decode token,
   comparing Σ(per-shape qmv GPU time) against the `tensor.eval.wait`
   bucket. That determines whether the residual is kernel/config (levers
   a/b) or inter-dispatch serialization (lever c).

Reproduce: `cargo test -p cider-press-kernels --release --test
qmv_dispatch_bench -- --ignored --nocapture` and `uv run
scripts/measure_qmv_mlx.py`.

### Approach B: steady-state decode attribution

Approach A is latency-bound (one submit+wait per call); Approach B
arbitrates the decode gap by attributing a *real* decode token. Running
`bench --gpu-profile --features profiling` on Qwen2.5-0.5B-Instruct-4bit-bf16
(M4 Max, warm, representative): decode **~217 tok/s**, `tensor.eval.wait`
**~3.22 ms/token** (the synchronous GPU `commit_and_wait`), and the
per-op-kind GPU `quantized_matmul` bucket **~1.66–1.68 ms** across 169
dispatches (24 layers × 7 qmv + 1 lm_head). Note the GPU-counter breakdown
is the perturbed stage-boundary regime (each sampled dispatch gets its own
encoder), so the qmv bucket is **not** directly comparable to the
production `tensor.eval.wait` total; within that breakdown qmv is the
single largest kind at ~40% of GPU time.

Predicting the per-token qmv GPU time from the Approach A GPU-counter
bandwidths — for each layer Σ over {q,k,v,o} (896×896 / 896×128) +
{gate,up} (896×4864) + down (4864×896), ×24 layers, plus one lm_head
(896×151936) — gives **Σ ≈ 1.27 ms**. This is the same order of magnitude
as the measured ~1.67 ms bucket (Σ ≈ 0.76× bucket). Given the small-N
GPU-counter readings are noisy run-to-run (Approach A reports the floor)
and the per-shape bandwidths are approximate, **Σ ≈ bucket**: there is no
large residual that would indicate inter-dispatch serialization. The qmv
cost is genuine kernel/config time, **not** lever (c) bubbles — so the
addressable headroom is in the kernel launch, not in pipelining.

**Dominant lever: small-N occupancy starvation (lever b)**, compounded by
fast-path absence (lever a). Approach A already localized the cost — the
generic `(1, ceil(N/8), 1)` grid leaves the small-N decode projections
badly under-utilized (k/v_proj N=128 at ~13 GB/s, q/o_proj N=896 at
~85 GB/s) while the large-N linears saturate near the ~464 GB/s empirical
ceiling. These small-N projections, which carry the bulk of the
per-layer qmv dispatch count, are the addressable headroom. The forced-fast
control (~151 vs ~85 GB/s at comparable size) shows lever (a) compounds it:
no Qwen2.5-0.5B shape has a 512-multiple K, so the fast path is never
selected.

**Recommended next move:** a small-N qmv launch-config fix — split the
output dimension across more threadgroups (e.g. a 2-D `(split_k_or_n, …)`
grid) so small-N projections fill the GPU rather than launching a handful
of threadgroups — measured against the Approach A sweep and the
`qmv_parity` tests, with fast-path-via-padding (pad K to a 512-multiple so
the fast variant is selected) as a secondary, parity-checked lever. Both
are dispatch-side changes that do not touch the vendored `quantized.metal`;
they are promoted to their own follow-up branch (see `docs/ARCHITECTURE.md`
backlog item #4).
