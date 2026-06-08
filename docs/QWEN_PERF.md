# Qwen2.5-0.5B perf

End-to-end performance of Qwen2.5-0.5B-Instruct-4bit under `cider-press`,
measured via `cider-press bench`, compared against `mlx_lm` on the same
prompt and token budget. Methodology: warm up, then time a steady-state
window; see `docs/ARCHITECTURE.md` for the architecture and forward backlog.

The headline: decode now **edges past MLX** at the full-model scale
(**~599 vs ~568 tok/s, ~1.05×**) after the **qmv fast-path padding (A1)**
change padded the q/k/v/o decode projections from K=896 to K=1024, making the
vendored `qmv_fast` variant selectable — lifting decode **~561 → ~599 tok/s**
over the prior encoder/sync-model parity point (**~561 vs ~564, ~1.00×**). See
**qmv fast-path padding (A1)** below. The path to parity, down from ~1.20×
before the **encoder/sync-model port** and ~2.3× several branches ago — the fused
`RMSNorm`, **async decode pipelining**, **decode copy-elimination (S1)**, and
the **encoder/sync-model port** stack to **~2.31×** over the pre-fusion ~243
baseline. The port replaced Metal's serial-encoder + automatic-hazard-tracking
dispatch model with MLX's own — hazard-untracked buffers, an explicit
`MTLFence` chain across encoders, concurrent-dispatch encoders, and
graph-level barrier elision scheduled in Kahn-wave order — lifting decode
**~473 → ~561 tok/s** and closing the last of the gap to `mlx_lm`. See
**Dispatch-serialization gap** below for the measured trajectory. Three
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

The decode step is dominated by `tensor.eval` (~90%), whose GPU wait is a
deep chain of **dependent dispatches** — the 4-bit `qmv` matvecs plus the many
small ops gated behind them. An early concurrent-encoder spike (see the
**Concurrent-encoder spike** section) concluded the dispatch model was *not*
the lever; a 2026-06-04 trace-level comparison **overturned that** — the
remaining ~1.20× *was* the dispatch model (the spike had tested a hybrid
configuration paying both tracked-hazard and barrier costs). The
**encoder/sync-model port** that followed (hazard-untracked buffers + fence
chain + concurrent encoders + graph-level barrier elision in Kahn-wave order)
closed it: decode **~473 → ~561 tok/s, ~1.00× of `mlx_lm`**. See
**Dispatch-serialization gap** for the full measured story. The remaining
levers are now kernel-side (small-N `qmv` launch config), prefill strided work
(S2–S5), within-eval reuse for RSS, and `.metallib` precompile — not the
dispatch model, which is done.

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
| prefill | ~917 tok/s† | ~1390 tok/s† | ~1.52×† |
| decode  | ~599 tok/s  | ~568 tok/s  | **~0.95× (cider ahead)** |

(Decode reflects **qmv fast-path padding (A1)**; the pre-A1 encoder/sync-model
parity point was ~561 vs ~564, ~1.00×.)

† **These prefill figures are not comparable** — see **Prefill gap
analysis** below. cider's ~917 was wall-clock around `generate()`, which
`eval_async`s the prefill and does *not* wait for the GPU; the ~1390 came
from mlx_lm's own prompt-eval reporting. Measured apples-to-apples
(synchronous forward + eval-wait, both sides), the true prefill gap is
**~1.30×** (cider 10.19 ms vs mlx 7.82 ms at T=39), not 1.52×.

- **Prefill** processes the 39-token prompt in one forward. cider-press
  ranged 880–943 tok/s across warm runs (~917 median); `mlx_lm` ~1390.
  This branch does **not** target prefill, but the encoder/sync-model port
  lifted it incidentally (the prior baseline was ~376) — the same
  untracked-buffer + fence-chain dispatch model that pays off in decode
  also unblocks the prefill command buffer. Prefill still runs the
  **composed** attention path (the fused vector kernel is the decode/T=1
  path; prefill fusion is deferred — see `docs/ARCHITECTURE.md`). The
  prefill gap was later **measured** (not inferred) — see **Prefill gap
  analysis**: the true synchronous gap is ~1.30×, qmm is at parity, and
  the composed attention is the primary lever.
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
  rose from ~317 → ~352 tok/s. The **encoder/sync-model port** then closed the
  remaining gap, lifting decode **~473 → ~561 tok/s** (depth-1, range
  553–568) and depth-0 **~352 → ~447 tok/s** by replacing Metal's
  serial-encoder + automatic-hazard dispatch model with MLX's
  untracked-buffer + fence-chain + concurrent-encoder model under graph-level
  barrier elision (see **Dispatch-serialization gap**). At **~561 vs ~564**
  the decode gap to `mlx_lm` is **~1.00× — essentially closed**. The
  **qmv fast-path padding (A1)** change then took decode **~561 → ~599 tok/s**
  (depth-1, range 578–603, median 599) by padding the q/k/v/o decode
  projections to K=1024 so the `qmv_fast` variant fires — **past** `mlx_lm`'s
  ~568 (cider ~1.05×). See **qmv fast-path padding (A1)** below.
- Load time (excluded from tok/s): ~0.17 s.

**Cold-start note:** the very first forward of a fresh process triggers
Metal JIT compilation of the quantized kernels — run 1's prefill measured
**42.6 s** (0.9 tok/s) for this reason. Metal caches compiled libraries
to disk cross-process, so runs 2+ prefill in ~130 ms. The 42.6 s is a
one-time first-token-latency cost (a `.metallib` precompile would
amortize it; deferred — see `docs/ARCHITECTURE.md`), not representative of
prefill compute, and is excluded from the table above.

## Prefill gap analysis

Decode leads `mlx_lm`; prefill is the largest remaining gap. This section
**measures** it (the prior attribution was inferred from the decode
profile). Methodology, per the project's habit: diff the pipeline against
MLX structurally, *then* profile to confirm — don't hunt slow spots blind.
Full working notes: `docs/superpowers/notes/2026-06-08-prefill-diff-findings.md`.
Tooling added: `Generator::profiled_prefill`, `bench --gpu-profile-prefill`,
`bench --prefill-iters` (synchronous timing), `qmm_dispatch_bench.rs`,
`scripts/profile_mlx_prefill.py`, `scripts/measure_qmm_mlx.py`.

### The true gap is ~1.30×, not 1.52× (methodology fix)

The 1.52× headline was a **measurement artifact**. cider's ~917 tok/s was
wall-clock around `generate()`, which `eval_async`s the prefill and does
not wait for the GPU; the ~1390 mlx figure came from mlx_lm's own
prompt-eval reporting. Measured apples-to-apples — both sides build the
prefill graph, forward, and **synchronously** eval the full logits
(`logits.eval()` / `mx.eval(logits)`), warm, mean of 5, same T=39 prompt
and 4-bit checkpoint, same M4 Max:

| side  | method                             | prefill (T=39) | tok/s  |
|-------|------------------------------------|---------------:|-------:|
| cider | `forward` + `logits.eval()` (sync) |      10.19 ms  | 3829   |
| mlx   | `model(x)` + `mx.eval(logits)`     |       7.82 ms  | 4988   |

**True gap: ~1.30× — a 2.37 ms delta.**

### First prefill GPU profile (T=39, counter-sampled, perturbed)

`gpu.quantized_matmul` 76.3% · `gpu.copy` 8.2% (266 dispatches) ·
`gpu.binary` 5.3% · `gpu.matmul` 4.8% (48 = 2/layer) · `gpu.rope` 1.5% ·
`gpu.rms_norm` 1.4% · `gpu.slice_update` 1.0% · `gpu.softmax` 0.8% ·
rest <1%. (Decode contrast: `quantized_matmul` 58.4%, `copy` 0.2%, the
score chain fused into one `gpu.sdpa`.) The 266 copies reconcile exactly
to 11/layer × 24 + 2 (9 of the 11 per-layer copies are fusible; the 2
KV-cache-write copies are structural). RoPE is at parity (no extra copy);
the causal-mask add is a separate `binary` dispatch MLX folds in-kernel.

### qmm is at parity — it is *not* the gap

`quantized_matmul` dominates GPU time but is the **same `affine_qmm_t`
algorithm on both sides** (cider routes M>1 → `qmm`; MLX routes
M ≥ `vector_limit` → `qmm`). A microbench at all 8 prefill shapes (M=39),
cider GPU-counter ns vs **amortized** mlx timing (200 dispatches per
`mx.eval`, so the per-call host round-trip is amortized to GPU throughput):

| shape | cider GPU µs | mlx amortized µs | ratio |
|-------|----:|----:|----:|
| q_proj | 42.1 | 30.2 | 1.40× |
| k_proj | 24.3 | 12.0 | 2.02× |
| v_proj | 18.9 | 11.8 | 1.59× |
| o_proj | 23.2 | 22.7 | 1.02× |
| gate | 54.1 | 57.8 | 0.94× |
| up | 53.8 | 58.4 | 0.92× |
| down | 109.2 | 66.9 | 1.63× |
| lm_head | 1339 | 1463 | 0.91× |

Summed one-of-each: cider ~1664 µs vs mlx ~1723 µs — **parity**. The
large shapes that dominate the budget are at parity or cider-faster; the
apparent 1.4–2.0× on the tiny q/k/v projections is **counter-sampling
perturbation** (a roughly-fixed sampler overhead that is a large fraction
of a 12–24 µs dispatch but negligible on a 1.3 ms one — lm_head, fully
GPU-bound, shows cider *faster* at 0.91× despite carrying that perturbation).
qmm is ruled out as the gap driver.

### Reconciliation & ranked backlog

| contributor | est. cost | basis |
|---|---|---|
| composed attention (9 fusible copies/layer + QKᵀ/scale/mask/softmax/·V) | ~1.5–1.7 ms | GPU shares: copy 8.2% + matmul 4.8% + softmax 0.8% + partial binary, of 10.19 ms |
| encode / dispatch overhead (residual) | ~0.7–0.9 ms | gap minus the above; unconfirmed (needs Metal trace) |
| qmm projections | ~0 | microbench parity |

1. **Fused prefill attention** (steel `sdpa_full` / Plan B) — the primary
   lever (~1.5–1.7 ms, ≈ most of the gap). D_h=64 is a supported steel
   head dim and T=39 > 8 clears the fused-path threshold, so it is
   reachable; it deletes 9 of 11 per-layer copies plus the entire
   QKᵀ/scale/mask/softmax/·V chain in one kernel.
2. **Encode / dispatch overhead** — the ~0.7–0.9 ms residual; size it with
   a Metal System Trace (both binaries; the `.gputrace` capture exists)
   before committing to a fix. Deferred (smaller, and interactive).
3. **Small-projection qmm (q/k/v)** — *not* a real lever (parity once
   perturbation is credited); explicitly de-prioritized.

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
| post-load   | ~697 MiB |
| post-decode | ~913 MiB |
| peak        | ~913 MiB |

`mlx_lm` peak (Metal allocations, `GenerationResponse.peak_memory`):
**~329 MiB**.

The peak rose **~900 → ~913 MiB** with **qmv fast-path padding (A1)**: the
phase-split q/k/v/o projections keep both the original (prefill) and K-padded
(decode) weight sets resident, adding ~28 MiB of weight bytes (~3%) — the
dual-weight cost the A1 plan predicted. Freeing the prefill twin after the
first token in a serving context is possible but YAGNI for now.

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
and reuses buffers within a single token, while cider-press keeps every
intermediate of a token live simultaneously across its command-buffer chunks,
so the pool reclaims cross-token only (the encoder/sync-model port chunks the
token into ~11 command buffers, but they are committed back-to-back with all
outputs retained, so peak is unchanged). Mid-eval freeing is a separate future
lever.

## Hot spots & next levers

The decode step is **GPU-bound on a chain of dependent dispatches**, and
the dispatch-model overhead on that chain is now **gone** — the
encoder/sync-model port (below) closed the last ~1.20× to `mlx_lm`. The
CPU-side taxes were already gone (SliceUpdate collapsed the round-trips,
the buffer pool moved per-op allocation off the encode path, GPU argmax
removed the post-eval scan). What remains is the ~536 *dependent*
dispatches/token themselves — the 4-bit `qmv` matvecs plus the small ops
gated behind them — at MLX's own per-dispatch cost. The levers that remain
shorten the chain or speed each link. In measurement-justified priority:

- **MLX encoder/sync model — done (~1.19×, ~473 → ~561 tok/s).** Ported
  MLX's dispatch model: hazard-untracked buffers, an explicit `MTLFence`
  chain across encoders (signalled at encoder *close* — captures only
  commands encoded before it), concurrent-dispatch encoders, and
  **graph-level barrier elision** (a `memoryBarrier` only before dispatches
  that actually hazard a buffer written since the last barrier), scheduled in
  **Kahn-wave** order so independent ops sit adjacent and overlap. This
  closed the dispatch-serialization gap — decode is now **~1.00× of
  `mlx_lm`**. The trajectory, falsified hypotheses (barrier-per-dispatch
  worse than serial on real kernels; untracked-alone flat; the `updateFence`
  no-op bug), and the elision/wave measurements are in
  **Dispatch-serialization gap § Measured result**. Escape hatches:
  `CIDER_PRESS_TRACKED_BUFFERS=1`, `CIDER_PRESS_NO_BARRIER_ELISION=1`.
- **Small-N `qmv` — fast-path padding (A1) done; launch-config (A2) the
  remaining lever.** With the dispatch model closed, the heaviest dispatches
  in the chain are the 4-bit `qmv` matvecs (~58% of GPU time, counter-sampled).
  The `## qmv audit` localized the headroom to the small-N decode projections.
  **A1 (done, ~1.07×, ~561 → ~599 tok/s):** padding q/k/v/o to K=1024 makes the
  `qmv_fast` variant selectable — q/o ~85 → ~139 GB/s, k/v ~13 → ~22 GB/s,
  bit-exact (zero-scale/zero-bias pad groups), vendored kernels untouched (see
  **qmv fast-path padding (A1)**). **A2 (gated, conditional GO — k/v only):**
  a cider-owned small-N kernel that fixes occupancy (N=128 launches just 16
  threadgroups); q/o is near the fast-path ceiling so A1 suffices there, but
  k/v padded still runs ~7× under the kernel's own N=1024 bandwidth. Scoped to
  k/v on its own branch — see the A1 section's **A2 go/no-go**.
- **Within-eval reuse (RSS) and prefill strided work (S2–S5)** —
  orthogonal levers (memory, and the composed prefill path); see the
  per-section notes below.

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
  ~1.20× gap that remained after S1 was then closed by the encoder/sync-model
  port (above), not by further fusion.
  What remains of backlog #9 (strided-aware kernels) is the **prefill** strided
  path (S2–S5): rope strides, `slice_update` strided src, gemm
  transpose+broadcast, qmv strided — none of these affect decode, which is
  down to 2 copy dispatches. Note SwiGLU's `silu*up` and RoPE are *not* fusion
  targets: MLX composes SwiGLU itself, and we already dispatch RoPE/decode-SDPA
  fused.
- **Concurrent dispatch encoder — superseded (the full port shipped).**
  An early spike rejected concurrent dispatch + hazard barriers *on top of
  tracked buffers in one monolithic command buffer* (byte-identical to serial
  but ~13% slower). That negative result held only for the hybrid
  configuration; the **full MLX model** — untracked buffers + fence chain +
  concurrent encoders + graph-level barrier elision in Kahn-wave order — is
  what shipped and what closed the gap (~1.19×, see the done lever above).
  Barrier-per-dispatch (the spike's model) really is slower than serial on
  real kernels; *rare* barriers via elision are what pay.
- **Buffer pool / allocator — done (~1.75× with the KvCache fix).** A
  cross-token free-list recycles op-output scratch (98% decode hit), and
  the enabling `KvCache::update` slab-base change stopped the cache
  pinning per-step graphs. Together: decode ~120 → ~210 tok/s, CPU-encode
  share ~38% → ~11%, peak RSS ~1192 → ~900 MiB. The pool reclaims
  **cross-token** only (a token's ~11 chunks are committed back-to-back
  with every output retained until cache population, so its whole graph
  stays live); **within-eval reuse** (mid-eval freeing, as MLX does) is the
  next RSS lever toward `mlx_lm`'s ~329 MiB.
- **Faster `qmv` kernels — speeds each critical-path link (A1 done).** The
  4-bit `qmv` weight matvecs are the heaviest dispatches in the chain. The
  `## qmv audit` localized the headroom to the small-N projections; **A1
  (fast-path padding, done)** addressed lever (a) by padding q/k/v/o to K=1024
  (`qmv_fast` selectable) — q/o ~85 → ~139 GB/s, k/v ~13 → ~22 GB/s, bit-exact,
  ~561 → ~599 tok/s. Lever (b) (occupancy) remains for k/v only as **A2** (a
  cider-owned small-N kernel; q/o is now near the fast-path ceiling). These
  shorten each `qmv` link's GPU time but do not reduce the *number* of
  dependent dispatches (that is fusion, above).
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

## Dispatch-serialization gap (root cause of the remaining ~1.20×)

A trace-level comparison (2026-06-04, M4 Max, `xctrace` Metal System
Trace on both runtimes, post-S1) localized the remaining decode gap.
Headline: **the toll is the per-dispatch serialization model, not any
kernel** — same kernels, same dispatch counts, both runtimes 100%
GPU-dense, but cider's single serial encoder pays ~0.6–1 µs more GPU
time per dependent dispatch than MLX's encoder stack, summed over ~530
dispatches/token.

Evidence chain:

1. **Decode is GPU-saturated.** Per token: step ~2123 µs = CPU
   (encode ~358 + graph build ~160) fully hidden under the depth-1
   pipeline + blocked wait ~1607. The trace confirms: one dense
   ~2055 µs compute interval per token, inter-token gap ~0.6 µs. The
   entire gap to `mlx_lm` (~1757 µs/token) is GPU-timeline time.
2. **Same work on both sides.** Dispatch counts are essentially
   identical (~536 vs ~510 — same qmv/binary/rope/sdpa/rms/slice-update
   composition), and the kernels are the same MLX binaries. The qmv +
   sdpa heavy compute (~1.46 ms predicted) is common to both; the
   difference sits in the ~370 tiny dispatches and the links between
   them.
3. **MLX's structure differs.** Its trace shows **~10 compute encoders
   per token** — MLX commits a command buffer every **50 ops** on
   M-class (`get_max_ops_mb_per_buffer`, `device.cpp`) — each ~150–220
   µs, back-to-back with ~0.7 µs seams and ~14% genuine overlap
   (cross-token via `mlx_lm`'s async eval, plus seam pipelining).
4. **Microbench** (`encoder_overhead_bench` in the kernels crate; a
   512-dispatch dependent chain of hidden-dim-sized adds, command-buffer
   GPU timestamps): one serial encoder ~1.6–3.2 µs/dispatch; concurrent
   encoder + `memoryBarrier` every dispatch (MLX's in-chain model) ~2.1;
   the chain split across 10 command buffers ~1.3; no-dependency floor
   ~0.22.

Two shortcuts were tested and **falsified**:

- **Encoder rollover within one command buffer** (end/reopen the serial
  encoder every 50 ops): the driver **merges consecutive serial compute
  passes** — the trace still shows one ~2120 µs interval per token. No
  GPU effect; +135 µs/token CPU encode.
- **Command-buffer chunking alone** (commit every 50 ops; landed on this
  branch as the structural first half): per-chunk GPU cost drops to
  MLX's (~170 µs per 50-dispatch chunk), but Metal's **automatic hazard
  tracking between command buffers** serializes the seams coarsely —
  chunks sit channel-resident blocked (interval sum ~4× the wall span)
  and decode is unchanged (~462–465 tok/s).

**Why MLX doesn't pay:** its buffers are allocated
hazard-**untracked**; it does its own synchronization — per-encoder
`MTLFence`s (wait on the fences of prior encoders whose outputs
intersect the new encoder's inputs, signal its own at end) plus
in-encoder global `memoryBarrier(BarrierScopeBuffers)` on a
**concurrent**-dispatch encoder before any dispatch whose input was
written since the last barrier. That removes both costs at once: no
tracked-resource analysis at command-buffer seams, and a cheaper
per-link barrier inside the chain (microbench: ~2.1 vs ~3.2
µs/dispatch).

This **revises the concurrent-encoder spike's conclusion** (next
section). The spike tested concurrent dispatch + hazard barriers *on
top of tracked buffers inside one monolithic command buffer* — paying
tracked-resource bookkeeping *plus* explicit barriers — and measured a
~13% loss. The microbench isolates the encoder model with everything
else held constant and finds the opposite ordering. The spike's
negative result stands only for that hybrid configuration; it does not
condemn MLX's full model (untracked + fences + concurrent + chunking),
which the traces show running the same dispatch stream ~300 µs/token
faster.

**Estimated win for porting the full model:** ~2055 → ~1760 µs/token,
i.e. decode ~470 → ~550 tok/s, closing most of the remaining ~1.20×.
Design/plan: `docs/superpowers/plans/2026-06-04-encoder-dispatch-model.md`
(branch `perf/encoder-dispatch-model`).

### Measured result (2026-06-04, post-port)

The port landed and the estimate held — decode finished at **~561 tok/s**
vs `mlx_lm` **~564** (same-day, same machine), **~1.00×**. But the path was
not the one the plan drew, and several intermediate hypotheses were
falsified along the way. The trajectory (M4 Max, warm, `--max-tokens 128
--warmup 8`, median of 3–5, depth-1 unless noted):

| stage                                            | decode tok/s |
|--------------------------------------------------|-------------:|
| S1 baseline (single serial encoder/token)        | ~473 |
| + 50-op command-buffer chunking (tracked)        | ~463 (neutral) |
| + concurrent encoders, barrier-per-dispatch (tracked) | ~432 (regression) |
| + hazard-untracked buffers                        | ~430 (flat) |
| + graph-level barrier elision, **DFS** order      | ~457 |
| + `updateFence`-at-close bug fix                  | ~461 |
| + **Kahn-wave** scheduling                        | **~561** |

What each step taught:

- **Chunking alone was neutral (~463).** Splitting the per-token command
  buffer every 50 ops dropped per-chunk GPU cost to MLX's (~170 µs/chunk),
  but with buffers still hazard-**tracked**, Metal's automatic
  command-buffer-seam analysis stalled the chunks channel-resident (interval
  sum ~4× the wall span). No wall-clock change.
- **Concurrent + barrier-per-dispatch was a transient regression
  (~432).** On the concurrent encoder, a global `memoryBarrier` before every
  dependent dispatch costs *more* than a serial encoder's implicit
  per-dispatch serialization on real production kernels. The
  `encoder_overhead_bench` microbench had predicted the opposite (concurrent
  ~2.1 vs serial ~3.2 µs/dispatch) — but it chained hidden-dim adds at
  sub-µs kernel scale, where the barrier is cheap relative to a real qmv;
  **the microbench misled.** On production kernels the ordering inverts:
  barrier-per-dispatch is the slow path. The barrier had to become *rare*,
  not merely cheap — which is what elision does.
- **Untracked buffers alone were flat (~430).** The hypothesis that the
  tracked-hazard seam analysis was the wall-clock stall was **falsified at
  the wall-clock level**: flipping `HazardTrackingModeUntracked` on top of
  the barrier-per-dispatch model changed nothing (the xctrace timeline stayed
  fully dense, GPU/token ~2390 µs). The tracked-seam analysis *was* real in
  the trace, but it was not the binding wall-clock cost — the
  barrier-per-dispatch tax dominated it.
- **Graph-level barrier elision (DFS order) was the first real win
  (+8%, ~457 vs ~424 with elision off).** Modeled on MLX's window-reset
  hazard tracking: a `memoryBarrier` orders everything before it against
  everything after, so at each encode point only buffers written *since the
  last barrier* can race — an op missing that working set elides its barrier.
  But under the existing **DFS post-order** `build_order`, dependent ops are
  emitted back-to-back, so consecutive ops almost always hazard: measured
  elision rate **82/537 dispatches = 15%**. The lever was throttled by
  schedule order, not by the elision logic.
- **The `updateFence` placement bug (~461, and it made the chain
  real).** The fence chain was a **no-op** the whole time: `updateFence` was
  encoded at encoder *open*, but Metal's fence captures only commands encoded
  **before** the update — at open there are none, so it signalled
  immediately and the next encoder's `waitForFence` waited on nothing. The
  untracked chunks had been racing on timing luck (and passing parity by
  luck). Moving `updateFence` to `close_encoder` before `endEncoding` (MLX's
  `end_encoding` placement) made the chain actually order the chunks. The
  plan had explicitly mis-specified this ("`updateFence` may be encoded at
  any point — equivalent"); it is not.
- **Kahn-wave scheduling was the unlock (+22%, ~461 → ~561).** Switching
  `build_order` from DFS post-order to **Kahn (level/wave) order** places
  independent ops adjacent: within a wave the first op barriers and the rest
  elide and overlap. With elision off this configuration runs ~441–447; with
  elision on, **~561** — the elision rate climbs from 15% and the
  independent-op overlap is what pays. Wave ordering was **not in the plan**;
  it was the discovered unlock.

The escape hatches both survive in the shipped build:
`CIDER_PRESS_TRACKED_BUFFERS=1` restores Metal's automatic hazard tracking
(measured ~557 — essentially neutral now that the fence chain is real, but
kept for bisecting sync bugs), and `CIDER_PRESS_NO_BARRIER_ELISION=1`
disables elision (measured ~442, i.e. the +27% elision win is the dominant
contributor). Final span split at depth-1: `tensor.eval_async` (CPU encode)
~670 µs/token, `tensor.eval_async.wait` ~948 µs/token; depth-0 (synchronous)
~447 tok/s, a ~1.26× pipelining overlap. Peak RSS ~886 MiB, pool 96.9%.

The chunk size was swept ({25, 50, 100} ops/buffer); all three land within
run-to-run variance (~558/~550/~547 median, and a confirmation re-run of 25
vs 50 was a tie at ~550), so the const holds at MLX's **50**.

## Concurrent-encoder spike (tested, rejected)

> **Superseded (2026-06-04).** This section's conclusion — "the serial
> encoder is not the bottleneck" — was overturned by the trace comparison
> and the encoder/sync-model port (see **Dispatch-serialization gap §
> Measured result**). It is preserved as a dated snapshot: its specific
> finding holds (concurrent + barrier-per-dispatch *on tracked buffers* is
> ~13% slower than serial), but that hybrid is not MLX's full model. The
> shipped port — untracked buffers + fence chain + *rare* barriers via
> elision in Kahn-wave order — measures ~1.19× faster, not slower.

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
  and, if anything, flatters cider-press — so the ~1.00× near-parity is not an
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

## qmv fast-path padding (A1)

The audit above named two compounding small-N levers: occupancy starvation
(lever b) and **fast-path absence** (lever a — no Qwen2.5-0.5B shape has a
512-multiple K, so the faster `qmv_fast` variant is never selected). A1 takes
lever (a) the cheap, parity-preserving way: **pad the q/k/v/o decode
projection weights from K=896 to K=1024** so `K % 512 == 0` and the
`qmv_fast` variant becomes selectable, with **no edits to the vendored
`quantized.metal`** (per project scope guard).

**Bit-parity by construction.** The 128 pad columns carry zero packed
weights and **zero (bf16) scale AND zero bias**, so each pad group
dequantizes to exactly `0·q + 0 = 0` regardless of the activation tail — the
dot product is bit-identical to the unpadded matvec for *any* activation in
those columns. The padded activation therefore needs only *capacity*
(≥1024 elements), not zeroing. Runtime support: `QuantizedWeight` carries a
logical-K (896) distinct from its physical/padded K (1024) — activations
validate against the logical K, the kernel dispatches against the physical K
— and pooled allocations round up to 512 B so the kernel's padded-K read
(1024×bf16 = 2048 B) stays in-bounds over a 896×bf16 = 1792 B decode row. The
bit-parity gate (`crates/cider-press-runtime/tests/qmv_padded_parity.rs`)
proves padded == unpadded byte-for-byte at both decode shapes, and the
end-to-end greedy parity gate stays byte-identical to the MLX fixtures.

**Phase-split execution.** A new `PhasedLinear` (models layer) holds two
weight sets: a **prefill** `Linear` (original K=896 layout, qmm path) and a
**decode** `Linear` (K-padded twin, qmv `qmv_fast` path), selected by sequence
length (T==1 ⇒ decode). This is the first decode-specialized execution in the
models layer. Only q/k/v/o are padded — `lm_head` (already roofline-bound at
~464 GB/s) would *regress* on +14% weight bytes, and gate/up/down are
near-roofline; both are left unpadded (re-evaluate with data).

**Kernel-level effect** (`qmv_dispatch_bench`, M4 Max, GPU-counter eff BW,
warm steady-state floor over several runs — early runs are thermal/scheduling
noise, see the audit's caveat):

| shape (K×N) | linear   | unpadded (gen) | padded (fast) | GPU-time speedup |
|-------------|----------|---------------:|--------------:|-----------------:|
| 896→1024 × 896 | q/o_proj | ~85 GB/s | **~139 GB/s** | ~1.37× |
| 896→1024 × 128 | k/v_proj | ~13 GB/s | **~22 GB/s**  | ~1.43× |
| 1024 × 1024 (control) | forced-fast | — | ~156 GB/s | — |

**End-to-end effect** (M4 Max, warm, `--max-tokens 128 --warmup 8`, depth-1,
median of 5): decode **~561 → ~599 tok/s** (578.6–602.7, median 598.6);
depth-0 (synchronous) ~454 tok/s (450.7–454.7, median of 3). `mlx_lm` same-day
on the same prompt **~568 tok/s** (552.7–568.6, median 567.7) — cider-press is
now **~1.05× ahead**. Greedy output is byte-identical to MLX (parity gate
passes; the padding is provably zero-contribution). Peak RSS rose **~900 →
~913 MiB**: the dual q/k/v/o weights add ~28 MiB of resident weight bytes
(~3%), the dual-weight cost the plan predicted. Buffer pool hit-rate ~97%.

### A2 go/no-go (cider-owned small-N qmv kernel)

A2 is the deferred decision: a cider-owned, MIT-attributed small-N `qmv`
kernel variant (a NEW file — the "adapt vendored kernels" model-shift) that
splits the output dim across more threadgroups to attack lever (b). A1's
measurement gates it, and the numbers split the two shapes:

- **q/o_proj (N=896): near-done by A1, low A2 upside.** Padded q/o already
  attains ~139 GB/s — close to the `qmv_fast` small-N ceiling (the 1024×1024
  control tops out at ~156, *not* the ~464 roofline; at N≈896–1024 there isn't
  enough output work to saturate the GPU regardless of launch config). A
  custom kernel could claim at most ~139 → ~156, ~12%. **Not worth a new
  kernel.**
- **k/v_proj (N=128): the real A2 target.** Even padded, k/v sits at
  ~22 GB/s — **~7× below** the ~156 GB/s the *same* `qmv_fast` kernel reaches
  at N=1024, and ~20× below the ~464 roofline. N=128 launches only
  `ceil(128/8) = 16` threadgroups, leaving the GPU mostly idle; the fast-path
  swap can't fix occupancy. A small-N-specialized launch (split-K across the
  inner dim, or co-launching k+v as one N=256 dispatch) is where the headroom
  lives.

**Recommendation: A2 is a conditional GO, scoped narrowly to the k/v
(N=128) shape only** — q/o is adequately served by A1's padding. The k/v
projections carry a large per-layer dispatch share and run ~7× under the
kernel's own demonstrated N=1024 bandwidth, so the occupancy fix has real,
measurable room there. Sequence it as its own branch (vendored kernels
untouched; new MIT-attributed file), with the `qmv_padded_parity` and greedy
gates as the correctness bar. q/o-shape padding is the shipped answer; do not
build a custom kernel for it.
