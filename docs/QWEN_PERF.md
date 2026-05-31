# Qwen2.5-0.5B perf

End-to-end performance of Qwen2.5-0.5B-Instruct-4bit under `cider-press`,
measured via `cider-press bench`, compared against `mlx_lm` on the same
prompt and token budget. Methodology: warm up, then time a steady-state
window; see `docs/ARCHITECTURE.md` for the architecture and forward backlog.

The headline: decode runs **~2.7× slower than MLX** at the full-model
scale (~210 vs ~560 tok/s), down from ~4.7× before the buffer pool. Two
prior changes set the stage — the in-graph KV write (SliceUpdate)
collapsed each step to **one command buffer per token** (~1.6×), and the
fused `sdpa_vector` decode-attention kernel lifted decode to ~120 tok/s
(~1.33×). This pass then attacked the per-eval CPU cost with a
**cross-token buffer pool** that recycles op-output scratch instead of
allocating ~1000 fresh `MTLBuffer`s per token. The pool was inert on its
own (3% hit) until the enabling fix: **`KvCache::update` no longer chains
each `SliceUpdate` off the prior op-tensor** (it bases on the slab leaf),
which stopped the cache from pinning every step's projection graph for the
whole decode. Together they took decode **~120 → ~210 tok/s** (~1.75×) at
**98% pool hit-rate**, cut the `tensor.eval` CPU-encode share from ~38% to
**~11%**, and resolved the +32% peak-RSS regression (**~1192 → ~900
MiB**). The remaining ~2.7× gap is now GPU-execution-bound: `tensor.eval`
is ~83% of the decode step and ~89% of *that* is the synchronous GPU
`commit_and_wait` (the 4-bit `qmv` weight matvecs), with the CPU encode
down to ~9% of the step.

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
| decode  | ~210 tok/s  | ~560 tok/s  | ~2.7× |

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
  the whole decode — see Decode-step breakdown). The **remaining ~2.7×
  gap** to `mlx_lm` is now dominated by GPU execution (the 4-bit `qmv`
  matvecs in `tensor.eval.wait`), not CPU encode.
- Load time (excluded from tok/s): ~0.17 s.

**Cold-start note:** the very first forward of a fresh process triggers
Metal JIT compilation of the quantized kernels — run 1's prefill measured
**42.6 s** (0.9 tok/s) for this reason. Metal caches compiled libraries
to disk cross-process, so runs 2+ prefill in ~130 ms. The 42.6 s is a
one-time first-token-latency cost (a `.metallib` precompile would
amortize it; deferred — see `docs/ARCHITECTURE.md`), not representative of
prefill compute, and is excluded from the table above.

## Decode-step breakdown

Per-span totals over the timed decode window (~109 timed steps; the
profiler is reset after warmup, so the spans below show 108 hits).
Representative warm run, decode wall ≈ 502 ms / 217 tok/s. `tensor.eval` is split into two **nested,
non-overlapping** sub-spans: `tensor.eval.encode` (CPU-side command-buffer
construction) and `tensor.eval.wait` (the synchronous GPU
`commit_and_wait`); their sum is `tensor.eval` minus the cheap
cache-population pass. `argmax` is the post-eval CPU `cpu_to_vec` + scan;
`kvcache.update` is the lazy slab-based SliceUpdate op-build (24 hits/token,
no forced eval).

| span                 | total (ms) | hits | µs/hit | share of decode |
|----------------------|-----------:|-----:|-------:|----------------:|
| tensor.eval          |  420 |  108 |  3886 | ~83% |
| → tensor.eval.encode |   46 |  108 |   426 | ~9% |
| → tensor.eval.wait   |  371 |  108 |  3434 | ~74% |
| argmax               |   48 |  108 |   444 | ~9.5% |
| kvcache.update       | 14.7 | 2592 |   5.7 | ~3% (lazy op build) |

Reading it:

- **`tensor.eval` is ~83% of the decode step, one `eval()` per token,**
  and is now **GPU-bound**: the encode/wait split is **~11% CPU encode,
  ~89% GPU wait** (was ~38%/62% before the buffer pool). The pool moved the
  decode bottleneck off the CPU encode path.
- **`tensor.eval.encode` (~0.43 ms/token, ~9% of the step) is CPU-side**
  and now small. Building one command buffer still means a topo walk and
  encoding every dispatch (~974 dispatches/token across 24 layers plus
  embedding and the tied head), but the per-op `MTLBuffer` allocation —
  formerly ~1000 fresh `newBufferWithLength:` calls per token — is now a
  **buffer-pool hit ~98% of the time** (see Memory), so allocation is no
  longer the dominant encode cost.
- **`tensor.eval.wait` (~3.4 ms/token, ~74% of the step) is GPU execution**
  and is the bottleneck. The heavy compute is the 4-bit `qmv` weight
  matvecs — every projection and MLP linear at T=1 — which are the vendored
  MLX kernel, memory-bound near the bandwidth roofline.
- **`kvcache.update` is ~3%** — the lazy SliceUpdate op-build, now based on
  the slab leaf rather than chained off the prior op-tensor. Chaining had
  pinned every step's K/V projection graph (and its residual-path
  intermediates) for the whole decode, which kept the buffer pool from ever
  reclaiming per-step scratch (3% hit). Basing on the slab frees each
  step's graph one step later, lifting the pool to 98% hit. The slab-base
  build is marginally costlier per call (~5.7 vs ~0.3 µs) but still ~3% of
  the step. Successive `update`s must now be separated by an `eval`
  (decode evals each token; a fail-loud guard enforces it).
- **`argmax` is ~9.5%** — the bf16 `cpu_to_vec` + scan over the ~151 k
  vocab, ~440 µs/token. Its *share* rose (the step got faster around it);
  a GPU argmax is the next small lever.

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
decode-step split is now ~89% GPU wait / ~11% CPU-encode (the buffer pool
moved allocation off the encode path); this subsection only resolves how
that GPU half divides across op kinds. The op-kind breakdown itself is
unchanged by the buffer pool and the KvCache slab-base collapse — both act
on CPU-side allocation/retention, not the GPU dispatches.

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

The decode step is now **GPU-execution-bound**. The CPU-side taxes are
gone — the SliceUpdate collapsed the dispatch round-trips, and the buffer
pool moved per-op allocation off the serial encode path (encode is down to
~9% of the step). What remains is `tensor.eval.wait` (~74% of decode), the
synchronous GPU run of the 4-bit `qmv` matvecs. In measurement-justified
priority:

- **Buffer pool / allocator — done (~1.75× with the KvCache fix).** A
  cross-token free-list recycles op-output scratch (98% decode hit), and
  the enabling `KvCache::update` slab-base change stopped the cache
  pinning per-step graphs. Together: decode ~120 → ~210 tok/s, CPU-encode
  share ~38% → ~11%, peak RSS ~1192 → ~900 MiB. The pool reclaims
  **cross-token** only (one command buffer per token keeps a token's whole
  graph live); **within-eval reuse** (mid-eval freeing, as MLX does) is the
  next RSS lever toward `mlx_lm`'s ~329 MiB.
- **GPU execution (`qmv`) — the dominant remaining cost.** `tensor.eval.wait`
  is ~74% of decode and is the vendored 4-bit `qmv` weight matvecs, which
  are memory-bound near the bandwidth roofline. Closing more of the ~2.7×
  gap means either faster matvecs or hiding the wait (pipelining, below).
- **Pipelining (overlap CPU encode with GPU)** — `commit_and_wait` is
  synchronous, so the (now small) CPU encode never overlaps the GPU.
  Multiple command buffers in flight (or Approach-C plumbing threading
  `Commands` through the forward) would hide it. Lower priority now that
  encode is ~9% of the step.
- **Fused prefill attention** — prefill still uses the composed path;
  fusing it (steel/nax `sdpa_full`, Plan B) is the prefill attention lever.
- **GPU argmax** — ~9.5% of decode (the `[vocab]` cpu_to_vec + scan,
  ~440 µs/token). Its share rose as the step got faster; a worthwhile small
  lever now.
- **`.metallib` precompilation** — doesn't touch steady-state tok/s, but
  removes the ~43 s cold-start JIT from first-token latency in a shipped
  binary.

## Caveats on the comparison

- **Decode windowing differs slightly between the two tools.**
  cider-press discards 8 warmup steps and times the remainder (~109 of
  the 117 generated for this prompt); `mlx_lm`'s `generation_tps` averages
  over all generated tokens (its `--warmup` flag is accepted only for CLI
  symmetry and is informational). On a pipelined backend the skew is small
  and, if anything, flatters cider-press — so the ~2.7× gap is not an
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
