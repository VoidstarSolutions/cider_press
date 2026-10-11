# Qwen3-Embedding (0.6B / 4B / 8B)

Status: **phase 0 complete** (ceiling measurements, 2026-10-11); the `qwen3`
model family and the `Embedder` API are not written yet. Design:
`docs/superpowers/specs/2026-10-11-qwen3-embeddings-design.md` (local).

## Workload

omniscient (code-search MCP) sends serial batches of ≤ 64 texts / 32 000
bytes to a llama-server `/v1/embeddings` endpoint running
Qwen3-Embedding-0.6B Q8_0 with last-token pooling and `-c/-b/-ub 32000`,
so each batch is one physically packed forward. Corpus used here: 640
line-window chunks of the omniscient repo (`scripts/make_embed_corpus.py`,
20-line windows, ≤ 2000 bytes), mean 676 bytes, 197.8 GGUF tokens per text,
14 batches under omniscient's limits.

Checkpoint facts verified for phase 1: the mlx-community
`Qwen3-Embedding-{0.6B,4B}-4bit-DWQ` safetensors carry **BF16** scales,
biases and norms (the loader's requirement — unlike the mlx-community
Qwen2.5 snapshot, whose F16 scales it rejects), ship **no `lm_head`**, and
carry `model.layers.N.self_attn.{q,k}_norm.weight`.

## Phase 0 — measured ceilings

Machine: Apple M5 Max, 128 GiB, GPU architecture `applegpu_g17s`, macOS 26.
llama.cpp build b15ca938a (10050, Homebrew, backend BLAS+MTL, 6 threads,
`-ngl 99 -fa on`), GGUF sha256
`06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439`.
cider `df71713`, `--release`, `OPS_PER_COMMAND_BUFFER = 50`, proxy model
Qwen2.5-0.5B-Instruct-4bit (bf16-scale copy; 0.72 GFLOP/token vs 0.88 for
Qwen3-Embedding-0.6B, so cider tok/s × 0.82 = embedding-equivalent).
MLX 0.32.3 / mlx-lm on the same proxy checkpoint. Median of 5, spread in
parentheses; every cider number excludes the cold Metal JIT (31.8 s).

### 1. Single-sequence prefill band

| T | llama-bench pp (tok/s) | cider prefill (sync) ms | cider tok/s | cider × 0.82 | cider / llama | **MLX warm forward (tok/s)** | cider / MLX |
|---|---|---|---|---|---|---|---|
| 150 | 11 774 ± 45 | 12.49 (12.47–12.53) | 12 015 | 9 850 | 0.84 | 16 782 | 0.72 |
| 512 | 23 628 ± 35 | 34.96 (34.91–35.15) | 14 645 | 12 010 | 0.51 | 36 657 | 0.40 |
| 2048 | 19 088 ± 103 | 143.3 (142.2–143.9) | 14 293 | 11 720 | 0.61 | 45 409 | 0.31 |

llama-bench `-b/-ub 2048` (one micro-batch per prompt, as omniscient's
32 000 setting makes every batch). cider peak RSS ≈ 690 MiB at every T.
**cider plateaus at ~14.6 k tok/s from T = 512**; llama.cpp reaches
23.6 k and MLX 36.7 k on the same shapes.

### 2. Incumbent packing gain

llama-server batched (omniscient call shape): **87.3 texts/s**,
59 081 bytes/s, 17 276 GGUF tok/s (median 7.33 s per corpus pass,
7.26–7.67); single-text latency median 17.6 ms (13.8–27.8).
Single-sequence equivalent at the corpus mean of 198 tokens (interpolated
from pp150 / pp512): ~13 350 tok/s = 67.5 texts/s. **Packing gain: 1.29×.**

### 3. Lever 3 ceiling (row packing)

64 × 150-token forwards: 799 ms. One 9600-token forward: 884 ms
(884.0–938.6; peak RSS 710 MiB; 10.9 k tok/s). **Single-long-forward
ratio: 0.90×** — causal attention over 9600 tokens dominates, so this
experiment cannot bound packing (as the plan warned). The usable bound is
the T plateau: cider reaches its compute-bound rate by T = 512
(14 645 vs 12 015 tok/s at T = 150), so packing 64 × 150 rows into one
qmm-sized forward gains **at most 1.22×** on today's kernels
(97.6 vs 80.1 texts/s).

### 4. Lever 4 ceiling (`eval_async` window)

| k | per-text ms (median of 5) |
|---|---|
| 1 | 12.64 |
| 2 | 12.35 |
| 3 | 12.33 |
| 4 | 12.36 |

**Ceiling: 1.03×.** Argmax identical at every k. k = 1 agrees with the
`bench` prefill (sync) number within 1.2 %.

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
effective ~10.5 TFLOPS (0.72 GFLOP/token × 14.6 k tok/s) while a direct
MLX micro-benchmark of `mx.quantized_matmul` at the embedding model's
prefill shapes reaches **29.6 TFLOPS at M = 512** and 39.0 at M = 2048
(bf16 `mx.matmul`: 21.4 / 37.3). The difference is the kernel: on GPUs of
generation ≥ 17 under macOS 26.2 (`is_nax_available`, `device.cpp`), MLX
dispatches the tensor-core **`qmm_t_nax` / `steel_gemm_fused_nax`** kernels
(`quantized.cpp:695`: `is_nax_available() && transpose && K % 64 == 0`).
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
stretch goal is re-examined. Outcome: lever 3's usable bound is 1.22×
(below 1.5× and below the incumbent's 1.29×) and lever 4 is 1.03×, so
**neither packing nor the async window is the next lever**; phase 3 ships
the harness without either. Stretch goal ("significantly exceed
llama.cpp"): **not supported by levers 1–4** — sequential cider on this
workload would land at ~0.7× of llama-server batched (≈ 60 texts/s
embedding-equivalent vs 87.3) and at most ~0.85× with packing. It
**becomes plausible only through the NAX qmm path (§6)**: MLX's warm
forward on the same model is 1.55× llama.cpp at T = 512 and 2.4× at
T = 2048, and that is cider's reachable ceiling once it dispatches the
same kernels.

## Open levers

- **NAX `qmm_t` / steel gemm dispatch (new, largest):** bound 2.5× at
  T = 512 (MLX 36.7 k vs cider 14.6 k tok/s on the proxy model). Requires
  vendoring `quantized_nax.*`, `gemm_nax.h`, `nax.h` (the sync script
  copies a fixed list), a runtime check mirroring `is_nax_available`
  (architecture gen ≥ 17, macOS ≥ 26.2), and a `qmm_t_nax` dispatch
  routine with its own parity test. Phase 1 of the embedding work should
  be re-planned around this; it benefits Qwen2.5 and Qwen3.5 prefill too.
- Row packing: ≤ 1.22× on today's kernels; re-measure after NAX (the
  plateau moves).
- `eval_async` window: 1.03×; closed.
- No-KV-cache forward and fused `bd128` attention: ship for memory and
  simplicity, no perf claim (slice_update ≤ 0.5 %, sdpa ≤ 7.3 %).
- Quantisation: 4-bit DWQ vs Q8_0 is a wash for prefill time; drift vs a
  bf16 reference still to be reported in phase 3.
