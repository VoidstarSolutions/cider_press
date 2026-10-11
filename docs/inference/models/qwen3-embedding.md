# Qwen3-Embedding (0.6B / 4B / 8B)

Status: **phase 0 complete** (ceiling measurements, 2026-10-11); the `qwen3`
model family and the `Embedder` API are not written yet. Design:
`docs/superpowers/specs/2026-10-11-qwen3-embeddings-design.md` (local).

## Workload

omniscient (code-search MCP) sends serial batches of ≤ 64 texts / 32 000
bytes to a llama-server `/v1/embeddings` endpoint running
Qwen3-Embedding-0.6B Q8_0 with last-token pooling and `-c/-b/-ub 32000`.
omniscient passes no `-np`, so the server runs its default **4 slots** and
packs at most 4 sequences per forward (`total_slots: 4` in the recorded
`/props`), not the whole 64-text batch. Corpus used here: 640
line-window chunks of the omniscient repo (`scripts/make_embed_corpus.py`,
20-line windows, ≤ 2000 bytes), mean 676 bytes, 197.8 GGUF tokens per text,
14 batches under omniscient's limits.

Checkpoint facts verified for phase 1: the mlx-community
`Qwen3-Embedding-{0.6B,4B}-4bit-DWQ` safetensors carry **BF16** scales,
biases and norms (the loader's requirement — unlike the mlx-community
Qwen2.5 snapshot, whose F16 scales it rejects), ship **no `lm_head`**, and
carry `model.layers.N.self_attn.{q,k}_norm.weight`.

## Phase 0 — measured ceilings

Machine: Apple M5 Max, 128 GiB, GPU architecture `applegpu_g17s`, macOS 27.0.1.
llama.cpp build b15ca938a (10050, Homebrew, backend BLAS+MTL, 6 threads,
`-ngl 99 -fa on`), GGUF sha256
`06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439`.
cider `df71713`, `--release`, `OPS_PER_COMMAND_BUFFER = 50`, proxy model
Qwen2.5-0.5B-Instruct-4bit (bf16-scale copy). **Normalisation:** linear
layers only, 0.716 GFLOP/token (24 layers × 14.91 M params) vs 0.881 for
Qwen3-Embedding-0.6B (28 × 15.73 M), so cider tok/s × **0.81** =
embedding-equivalent at short T; causal attention (24 × 896 wide vs 28 × 2048)
pulls the factor to ≈ 0.80 at T = 512 and ≈ 0.72 at T = 2048, so the
T = 2048 equivalents below are ~12 % optimistic. MLX 0.32.3 / mlx-lm on the
same proxy checkpoint (`scripts/measure_mlx_prefill_warm.py`: full
`model(x)`, including `lm_head` over all T, so conservative for MLX).
Median of 5, spread in parentheses; every cider number excludes the cold
Metal JIT (31.8 s).

### 1. Single-sequence prefill band

| T | llama-bench pp (tok/s) | cider prefill (sync) ms | cider tok/s | cider × 0.81 | cider / llama | **MLX warm forward (tok/s)** | cider / MLX |
|---|---|---|---|---|---|---|---|
| 150 | 11 774 ± 45 | 12.48 (12.47–12.51) | 12 015 | 9 732 | 0.83 | 16 546 (16 493–16 643) | 0.73 |
| 512 | 23 628 ± 35 | 35.02 (34.91–35.15) | 14 619 | 11 841 | 0.50 | 39 882 (39 596–39 963) | 0.37 |
| 2048 | 19 088 ± 103 | 143.3 (142.2–143.9) | 14 294 | 11 578 | 0.61 | 45 852 (45 844–45 894) | 0.31 |

llama-bench `-b/-ub 2048` (one micro-batch per prompt, as omniscient's
32 000 setting makes every batch). cider peak RSS ≈ 690 MiB at every T.
**cider plateaus at ~14.6 k tok/s from T = 512**; llama.cpp reaches
23.6 k and MLX 39.9 k on the same shapes. CLAUDE.md's "prefill gap vs mlx
is closed" was measured on an M4 Max at T ≈ 36 (and predates MLX's NAX
path, §6); it is machine- and T-specific and does not hold here.

### 2. Incumbent packing gain

llama-server batched (omniscient call shape): **87.3 texts/s**,
59 081 bytes/s, 17 276 GGUF tok/s (median 7.33 s per corpus pass,
7.26–7.67); single-text latency median 17.6 ms (13.8–27.8).
Single-sequence equivalent at the corpus mean of 198 tokens (interpolated
from pp150 / pp512): ~13 350 tok/s = 67.5 texts/s. **Packing gain: 1.29×**
(with the default 4 slots; a larger `-np` is untested, see Open levers).

### 3. Lever 3 ceiling (row packing)

64 × 150-token forwards: 799 ms. One 9600-token forward: 884 ms
(884.0–938.6; peak RSS 710 MiB; 10.9 k tok/s). **Single-long-forward
ratio: 0.90×** — causal attention over 9600 tokens dominates, so this
experiment cannot bound packing (as the plan warned). The usable bound is
the T plateau: cider reaches its compute-bound rate by T = 512
(14 619 vs 12 015 tok/s at T = 150; 1.22×), and removing the attention
share packed rows would not pay (3.0 % at 512, 7.3 % at 2048) puts the
attention-free plateau at ~15.1–15.4 k tok/s, so packing 64 × 150 rows
gains **≈ 1.25–1.28× (estimated)** on today's kernels — level with the
incumbent's 1.29×. qmm at M = 9600 was not measured directly.

### 4. Lever 4 ceiling (`eval_async` window)

| k | per-text ms (median of 5, min–max) |
|---|---|
| 1 | 12.64 (12.57–12.65) |
| 2 | 12.35 (12.34–12.38) |
| 3 | 12.33 (12.31–12.41) |
| 4 | 12.36 (12.32–12.41) |

**Ceiling: 1.03× measured; ≤ 1.07× analytic** (the window can only
hide CPU encode, which is 0.84 ms of the 11.6 ms GPU wait at
T = 150 in the profiled run). Argmax identical at every k; k = 1 agrees
with the `bench` prefill (sync) number within 1.2 %. Harness caveat: the
argmax check `eval`s synchronously inside the timed loop on cider's single
command queue, which can stall behind the in-flight forward; the analytic
bound is the one to trust.

### 5. Levers 1–2 GPU share (`[prefill]` op-kind table, `--features profiling`)

| T | slice_update % | sdpa % | quantized_matmul % |
|---|---|---|---|
| 150 | 0.5 | 2.1 | 91.8 |
| 512 | 0.3 | 3.0 | 90.5 |
| 2048 | 0.1 | 7.3 | 86.1 |

Lever 1 (no KV cache) and lever 2 (fused `bd128` attention) are
correctness/memory work with no measurable throughput upside.

### 6. The finding phase 0 was not looking for: NAX kernels

Prefill is **86–92 % quantized matmul**, and cider's qmm runs at an
effective ~10.5 TFLOPS (0.716 GFLOP/token × 14.6 k tok/s) while a direct
MLX micro-benchmark of `mx.quantized_matmul` at the embedding model's
prefill shapes (`scripts/measure_qmm_vs_gemm_mlx.py`, median of 5 rounds
of independent dispatches — an earlier accumulator-summing version of the
script under-read both by 1.2–3×, caught in review) reaches **47.9 TFLOPS
at M = 512** (41.1–48.9) and 51.5 (50.9–52.2) at M = 2048 (bf16
`mx.matmul`: 48.1 / 61.2). Against cider's ~11.6 TFLOPS qmm-only rate at
T = 512 (0.716 GFLOP/token × 512 ÷ 35.0 ms × 90.5 % GPU share) that is a
**~4× kernel gap**, larger than the 2.7× whole-forward ratio in §1. Per MLX's source (local checkout
7b7c124; the measured pip MLX is 0.32.3), on GPUs of generation ≥ 17
(≥ 18 for the `p` architecture) under macOS ≥ 26.2 (`is_nax_available`,
`device.cpp:828`), MLX dispatches the tensor-core **`qmm_t_nax` /
`steel_gemm_fused_nax`** kernels (`quantized.cpp:695`:
`is_nax_available() && transpose && K % 64 == 0`). That dispatch was not
confirmed with a `.gputrace` capture; the ~4× kernel-level gap is
measured, its attribution to NAX is read from source. A throwaway probe
also confirmed the runtime Metal compiler builds the flattened
`quantized_nax` source under language version 4.0 on this machine and
yields both `affine_qmm_t_nax_…_alN_{true,false}_batch_0` pipelines.
Those kernels (`quantized_nax.metal`, `steel/gemm/gemm_nax.h`, `nax.h`)
are **absent from `kernels-mlx/`** (vendored 8906447 and the in-flight
06eb748 sync alike) and from cider's dispatch, which only knows
`affine_qmm_t`. llama.cpp's 23.6 k tok/s sits between cider and MLX.

This supersedes the lever ranking in the spec: no scheduling or packing
work can close a 2.5× kernel gap, and once the gap is closed the same
packing/pipelining numbers above apply on top of a ~2.5× faster base.

### Decision

Spec rule: packing next if the lever-3 ceiling ≥ 1.5× **and** beats the
incumbent's packing gain; otherwise lever 4 lands in phase 3 and the
stretch goal is re-examined. Outcome: lever 3's estimated bound is
1.25–1.28× (below the 1.5× gate) and lever 4 is 1.03×, so **neither
packing nor the async window is the next lever**; phase 3 ships the
harness without either. Stretch goal ("significantly exceed llama.cpp"):
**not supported by levers 1–4.** Sequential cider at the corpus mean of
198 tokens: 12 015 + (48/362) × (14 619 − 12 015) ≈ 12 360 tok/s × 0.81
÷ 198 ≈ **50.6 texts/s = 0.58×** of llama-server's 87.3; with the
1.25× packing estimate ≈ 63 texts/s = **0.72×**. It **becomes plausible
only through the NAX qmm path (§6)**: MLX's warm forward on the same
proxy model, normalised (× 0.81 at 512, × 0.72 at 2048), is ≈ 1.37×
llama.cpp at T = 512 and ≈ 1.73× at T = 2048, and that is cider's
reachable ceiling once it dispatches the same kernels.

## Open levers

- **NAX `qmm_t` / steel gemm dispatch (new, largest):** bound 2.7× on
  the whole forward at T = 512 (MLX 39.9 k vs cider 14.6 k tok/s on the
  proxy model), ~4× on the qmm kernel itself. Requires
  vendoring `quantized_nax.*`, `gemm_nax.h`, `nax.h` (the sync script
  copies a fixed list), a runtime check mirroring `is_nax_available`
  (architecture gen ≥ 17, ≥ 18 for `p`; macOS ≥ 26.2), and a `qmm_t_nax` dispatch
  routine with its own parity test. Phase 1 of the embedding work should
  be re-planned around this; it benefits Qwen2.5 and Qwen3.5 prefill too.
- Row packing: ≈ 1.25–1.28× (estimated) on today's kernels; re-measure
  after NAX (the plateau moves).
- Incumbent headroom: llama-server packs 4 sequences by default; phase 3
  should also measure it with `-np 64` so the target is the incumbent at
  its best, not its default.
- `eval_async` window: 1.03×; closed.
- No-KV-cache forward and fused `bd128` attention: ship for memory and
  simplicity, no perf claim (slice_update ≤ 0.5 %, sdpa ≤ 7.3 %).
- Quantisation: 4-bit DWQ vs Q8_0 prefill time was **not measured** in
  phase 0 (hypothesis: a wash, prefill being compute-bound); drift vs a
  bf16 reference and the timing both land in phase 3.
