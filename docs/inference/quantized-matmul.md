# Quantized matmul (qmv / qmm)

Every projection in the forward pass — the four attention projections, the
three FFN projections, and the LM head — is a multiply against a 4-bit
quantized weight matrix. This page is about the kernels that do that multiply:
how they dequantize on the fly, why decode and prefill route to two different
kernels, and why this op is the bandwidth-bound workhorse that sets the decode
clock. The *format* those weights are stored in — the packed lanes, the
per-group scales and biases, how they load — is the partner page,
[weights and quantization](./weights-and-quantization.md); this page assumes it
and picks up at the matmul.

## Concept

A linear layer computes `y = x @ Wᵀ`. When `W` is 4-bit affine-quantized, the
multiply has to reconstruct each weight `w ≈ scale · q + bias` from its packed
4-bit integer `q` and its per-group `(scale, bias)` before it can use it. The
key design choice is *where* that reconstruction happens: **inside the matmul
kernel**, on the fly, never materializing a dense weight tensor. A thread reads
the packed lanes and the group's scale/bias straight from memory, dequantizes
into registers, multiplies-accumulates, and moves on — the reconstructed `bf16`
weight exists only transiently in registers, never as a buffer. Materializing a
dense `[N, K]` `bf16` weight first would defeat the entire point of
quantization: it would stream the full-width bytes back out to memory and read
them again, paying the bandwidth cost quantization exists to avoid.

The op comes in two shapes, and which one runs is set by the number of input
rows `M` (the number of token positions being projected at once):

- **Matrix-vector (`qmv`), `M = 1` — decode.** One token's activation vector
  times the weight matrix. This is the steady-state generation path: every
  decode step projects a *single* `[K]` activation through each layer's
  weights, so `M = 1`. The kernel reads each weight exactly once and does a
  single multiply-accumulate against the one activation vector — almost no
  arithmetic reuse per byte read.
- **Matrix-matrix (`qmm`), `M > 1` — prefill.** The whole prompt's `[M, K]`
  activations times the weight matrix in one pass. Here each weight, once read,
  is reused across all `M` rows, so the kernel can tile and amortize.

That `M = 1` decode case is **the** reason decode is memory-bandwidth-bound (see
[weights and quantization](./weights-and-quantization.md)). With no arithmetic
reuse, the cost of a decode step is just *how fast the weight bytes arrive* —
the GPU finishes each multiply-accumulate long before the next weight lands. So
`qmv` is the workhorse of decode: 24 layers × 7 projections + 1 LM head =
**169 `qmv` dispatches per token**, every model weight read exactly once. The
faster those reads saturate memory bandwidth, the faster the model decodes.

## In cider-press

[`Tensor::quantized_matmul`](../../crates/cider-press-runtime/src/tensor.rs)
records the op and routes by `M`: rank-1 (or `M = 1`) activations take the
matvec path, `M > 1` the matmul path. At `eval()` the two paths dispatch two
different kernels over the same vendored
[`quantized.metal`](../../kernels-mlx/mlx/backend/metal/kernels/quantized.metal):

- decode → [`qmv.rs`](../../crates/cider-press-kernels/src/kernels/qmv.rs)
  (`affine_qmv` family),
- prefill → [`qmm.rs`](../../crates/cider-press-kernels/src/kernels/qmm.rs)
  (`affine_qmm_t` family).

Both take a
[`QuantizedWeight`](../../crates/cider-press-runtime/src/quantized.rs) — the
three-buffer newtype that carries the packed lanes, scales, and biases together
— so a dense tensor can't be handed to the quantized path by accident. The
constructor validates the activation is `bf16`, dense, and contiguous, that its
inner dim matches the weight's logical `K`. Any `N` is accepted: the `M > 1`
qmm path picks the aligned `affine_qmm_t` instantiation when `N % 32 == 0` and
the non-aligned one otherwise (see "Open levers" below).

### Tensor-core path (NAX)

On GPUs that have MLX's tensor-core kernels — generation ≥ 17 (≥ 18 for the
`p` class) under macOS ≥ 26.2, as reported by
[`Device::nax_available`](../../crates/cider-press-kernels/src/device.rs),
a port of MLX's `is_nax_available()` — the `M > 1` path routes every
quantized matmul with `K % 64 == 0` to `affine_qmm_t_nax` instead
(`affine_qmm_t_nax_bfloat16_t_gs_{gs}_b_{bits}_bm64_bn64_bk64_wm2_wn2_alN_{true|false}_batch_0`
from the vendored
[`quantized_nax.metal`](../../kernels-mlx/mlx/backend/metal/kernels/quantized_nax.metal),
flattened by `build.rs` into its own library and JIT-compiled on first NAX
dispatch — ~11 s cold for 540 kernels, sub-second from Metal's disk cache
after). The dispatcher
([`affine_qmm_t_nax_bf16`](../../crates/cider-press-kernels/src/kernels/qmm.rs))
mirrors `quantized.cpp::qmm_nax`: 64×64×64 tiles, threadgroup `(32, 2, 2)`,
grid `(⌈N/64⌉, ⌈M/64⌉, 1)`, `alN_true` when `N % 64 == 0`. The K-loop runs
whole tiles, so `K % 64 != 0` (a synthetic hidden size, say 96) silently stays
on the classic kernel. Everything else is unchanged: `M = 1` is still `qmv`,
non-NAX machines and `CIDER_PRESS_NO_NAX=1` (the bisecting escape hatch) take
the classic `affine_qmm_t`.

Parity is **bit-exact** against MLX on a NAX machine, where
`mx.quantized_matmul` runs the same kernel: the fixtures are generated on
demand (`dump_mlx_op qmm`) on aligned, unaligned-`N` (1000) and partial-row-tile
(`M = 150`) shapes. The NAX and classic kernels are *not* bit-identical to each
other (different accumulation order), so the model-level logits tests run under
their tolerance either way. One routing caveat when choosing parity shapes:
MLX sends transposed, unbatched `qmm` through `qmm_splitk` first, which picks a
classic split-K kernel (32×32 tiles, fp32 partials) whenever
`⌈N/32⌉·⌈M/32⌉ < 512` and `K` splits into whole 64-wide partitions — the
small-tile shapes (e.g. k/v at `N = 128`) where cider now dispatches NAX and MLX
does not. That is the one place this routing rule deviates from MLX's; see
"Open levers".

**The `qmv_fast` vs. generic variant.** `qmv.rs` mirrors MLX's dispatcher: it
selects the faster `affine_qmv_fast` kernel when `K` meets its bit-width
alignment (`K % 512 == 0` for 4-bit), using its partial-rows instantiation
(`_pr_1`) when `N % 8 != 0`, and the generic `affine_qmv` otherwise. The
selection is purely by shape alignment — same kernel source, a faster tiling
when `K` is a 512-multiple.
This matters because *no* native Qwen2.5-0.5B decode shape has a 512-multiple
`K` (they are `896` or `4864`), so absent intervention the fast path never
fires — which is exactly what the A1 padding below exploits.

**Bit-exact requirement.** The dispatch routines are derived from MLX and carry
its MIT header; the matvec/matmul kernels are the parity canary. cider-press
reproduces MLX's reconstruction `scale · q + bias` and accumulation
*bit-for-bit* — the `qmv`/`qmm` parity tests assert byte-identical output
against MLX fixtures, and the end-to-end greedy decode is byte-identical to the
MLX reference. Quantized matmul is held to the strict (bit-exact) end of the
[parity tolerances](../../CLAUDE.md), not the looser bound the
`exp`/`cos`/`sin` kernels get.

**The additive bias is a separate op — not fused.** Q, K, and V carry an
additive Linear `.bias` (distinct from the per-group quantization `.biases`; see
the [footgun](./weights-and-quantization.md)). That bias is added *after* the
matmul as a separate `x + bias` elementwise op, **not** folded into the kernel.
This matches MLX exactly — `mlx.nn.QuantizedLinear` issues the same separate add
(`quantized.py:276`, with no `mx.compile` fold in the model path), so both
runtimes pay it identically. (An earlier doc claim that "mlx fuses the bias" was
**wrong** and has been corrected.)

**Model variations.**

- **Group size / bit width** ride in the `QuantizedWeight`'s
  [`Quantization`](../../crates/cider-press-runtime/src/quantization.rs)
  descriptor and select the kernel instantiation by name
  (`..._gs_{group_size}_b_{bits}_...`). Qwen2.5-0.5B is `gs=64, b=4`; an 8-bit
  or `gs=32` checkpoint dispatches a different instantiation with no dispatch-code
  change, as long as the combination is one MLX's `instantiate_quantized_all`
  macro emits.
- **The LM head shares this path.** Qwen2.5 ties the embedding and head
  (`tie_word_embeddings = true`), so the final vocabulary projection is a
  `quantized_matmul` against the same `[vocab, D]` table used for the input
  embedding, transposed — `transpose=true`, the only direction wired. See
  [sampling](./sampling.md) for what happens to those logits, and
  [weights and quantization](./weights-and-quantization.md) for the tied-table
  loader.

## Performance

The bandwidth a `qmv` dispatch attains is strongly **N-dependent** (output
dimension), because the generic grid launches `(1, ceil(N/8), 1)` threadgroups —
small `N` simply doesn't launch enough threadgroups to fill the GPU. From the
`qmv audit` (GPU-counter effective bandwidth, M4 Max,
warm steady-state floor):

| shape (K×N) | linear       | cider GPU-counter eff BW |
|-------------|--------------|-------------------------:|
| 896×4864    | `gate/up_proj` | ~306 GB/s |
| 4864×896    | `down_proj`    | ~272 GB/s |
| 896×151936  | `lm_head`      | ~464 GB/s |
| 896×896     | `q/o_proj`     | ~85 GB/s  |
| 896×128     | `k/v_proj`     | ~13 GB/s  |

The large-N linears (the FFN projections and the LM head) run **near the
roofline** — ~464 GB/s at the LM head is the empirical kernel ceiling, against
the M4 Max ~410–546 GB/s spec roofline — so they have essentially no kernel-side
headroom. The small-N decode projections are the opposite: **occupancy-starved**
by the generic grid (q/o at ~85, k/v at ~13 GB/s), launching only a handful of
threadgroups.

**A1: fast-path padding** attacks that starvation the cheap, parity-preserving way. It pads the q/k/v/o
decode weights from `K = 896` to `K = 1024` so `K % 512 == 0` and the faster
`qmv_fast` variant becomes selectable — with **no edits to the vendored
`quantized.metal`**. The 128 pad columns carry **zero packed weights, zero
scale, and zero bias**, so each pad group dequantizes to exactly `0` regardless
of the activation tail: the matvec is bit-identical to the unpadded one for any
activation, proven by the `qmv_padded_parity` gate. `QuantizedWeight` carries a
logical `K` (896, what activations validate against) distinct from its physical
padded `K` (1024, what the kernel dispatches against) — that is the
`k_logical` field in
[`quantized.rs`](../../crates/cider-press-runtime/src/quantized.rs).

Kernel-level effect (GPU-counter eff BW, same harness):

| shape (K×N)    | linear   | unpadded (gen) | padded (fast) | speedup |
|----------------|----------|---------------:|--------------:|--------:|
| 896→1024 × 896 | q/o_proj | ~85 GB/s  | ~139 GB/s | ~1.37× |
| 896→1024 × 128 | k/v_proj | ~13 GB/s  | ~22 GB/s  | ~1.43× |

End-to-end (M4 Max, warm, `--max-tokens 128 --warmup 8`, depth-1, median of 5):
decode **~561 → ~599 tok/s**, while same-day `mlx_lm` on the same prompt sat at
**~568 tok/s** — so A1 moved cider-press to **~1.05× ahead** of `mlx_lm`. Greedy
output stays byte-identical. The cost is ~28 MiB of resident weight bytes (~3%
of peak RSS): q/k/v/o keep both an original (prefill, qmm) and a K-padded twin
(decode, qmv) resident, dispatched by a `PhasedLinear` that selects on sequence
length. Only q/k/v/o are padded — the LM head is already roofline-bound and
would *regress* on +14% weight bytes, and the FFN projections are already
near-roofline.

### Prefill: NAX vs classic `qmm`

Per-dispatch kernel time at `M = 512` (GPU-counter sampled, 64 dispatches per
command buffer; `perf_affine_qmm_t_nax_vs_classic_m512` in
[`qmm_dispatch_bench.rs`](../../crates/cider-press-kernels/tests/qmm_dispatch_bench.rs)),
M5 Max, macOS 27.0.1:

| shape (K×N)  | linear            | classic `affine_qmm_t` | `affine_qmm_t_nax` | speed-up |
|--------------|-------------------|-----------------------:|-------------------:|---------:|
| 1024×2048    | emb `q_proj`        | 639 µs / 3.4 TFLOPS    | 78 µs / 27.7 TFLOPS | 8.2× |
| 1024×1024    | emb k/v (gs64)    | 149 µs / 7.2           | 44 µs / 24.2        | 3.4× |
| 2048×1024    | emb `o_proj`        | 316 µs / 6.8           | 89 µs / 24.2        | 3.6× |
| 1024×3072    | emb gate/up       | 463 µs / 7.0           | 105 µs / 30.7       | 4.4× |
| 3072×1024    | emb down          | 472 µs / 6.8           | 147 µs / 21.9       | 3.2× |
| 896×896      | q25 `q_proj`        | 245 µs / 3.4           | 70 µs / 11.7        | 3.5× |
| 896×4864     | q25 gate/up       | 1124 µs / 4.0          | 228 µs / 19.6       | 4.9× |
| 4864×896     | q25 down          | 657 µs / 6.8           | 195 µs / 22.9       | 3.4× |

Whole-forward synchronous prefill on Qwen2.5-0.5B-Instruct-4bit (`cider-press
bench --prefill-iters 10`, medians of 5 runs; `classic` is the same binary with
`CIDER_PRESS_NO_NAX=1`; MLX 0.32.3 warm single forward on the same prompts for
reference, from the embedding phase-0 measurements):

| prompt tokens | classic   | NAX       | speed-up | MLX warm forward | cider / MLX |
|--------------:|----------:|----------:|---------:|-----------------:|------------:|
| 150           | 12.08 ms  | 7.14 ms   | 1.69×    | 9.07 ms          | 0.79×       |
| 512           | 33.84 ms  | 13.76 ms  | 2.46×    | 12.84 ms         | 1.07×       |
| 2048          | 140.5 ms  | 59.7 ms   | 2.35×    | 44.67 ms         | 1.34×       |

Decode is `M = 1` (`qmv`) and is untouched: 641–645 tok/s interleaved either
way. The residual at long prompts is attention and the dense gemm, which still
run the classic steel kernels — the next two levers below.

## Open levers

- **NAX gemm and attention (next).** With `qmm` on tensor cores the prefill
  gap to MLX closes at short and mid prompts (above) but reopens at `T = 2048`
  (1.34×), where MLX also runs `steel_gemm_fused_nax` and
  `steel_attention_nax`. Porting those dispatches is the planned follow-on
  (gemm, then attention).
- **MLX's split-K routing for small tiles.** MLX prefers a classic split-K
  `qmm` over NAX when the classic grid has fewer than 512 threadgroups (see
  the NAX subsection). cider dispatches NAX for every `K % 64 == 0` shape; on
  the k/v projections (`N = 128`) that means a 2×8-threadgroup NAX grid at
  `T = 512`. The whole-forward numbers above include that choice; whether a
  split-K port wins anything on those shapes is unmeasured.
- **NAX `bm = 32` variant for `M ≤ 32`.** MLX uses a 32-row `qmm_t_nax`
  instantiation when one block covers all of `M`; cider always dispatches
  `bm = 64`. Only matters for short chunks (`2 ≤ M ≤ 32`), where MLX itself
  mostly routes to `qmv` anyway.

- **A2 (a cider-owned small-N `qmv` kernel for k/v): CLOSED — no-go.** A2 was
  the deferred idea of writing a new MIT-attributed kernel that splits the
  output dim across more threadgroups to fix the k/v occupancy starvation (k/v
  still sits at ~22 GB/s even padded, ~7× below what the *same* `qmv_fast`
  reaches at N=1024). The **zero-cost-k/v ceiling experiment** settled it: an env-gated build
  that skips the k/v qmv dispatches entirely (substituting zeros, leaving the
  rest of the chain intact) measured a decode upper bound of
  **600.1 → 606.5 tok/s, just +1.07%** — *smaller* than the run-to-run variance
  (~±10 tok/s). The occupancy ratio was real but irrelevant: k/v's bytes are
  tiny and its qmv time is **hidden by the async decode pipeline** (off the
  critical path — see the [execution model](./execution-model.md)). A custom
  kernel would capture only a fraction of that ~1% ceiling at the cost of owning
  a Metal kernel — far below the ~3–4% threshold for breaking pure-vendoring.
  Tracing MLX's selector also confirmed it routes the same k/v shape to the same
  `qmv_fast`, so there was no free vendored win to claim either. There is no
  kernel-side decode headroom left worth owning.
- **`transpose=false` direction is unimplemented.** Only the standard
  `y = x @ Wᵀ` direction is wired; `quantized_matmul(transpose=false)` returns
  `Err` at construction
  ([`tensor.rs`](../../crates/cider-press-runtime/src/tensor.rs)). The tied head
  reuses the embedding table *transposed*, so the existing direction covers the
  whole Qwen2.5 forward pass — this is deferred to its first real consumer.
- **`affine_qmm` non-aligned (`_alN_false`) is now wired; non-transposed is not.**
  `qmm.rs` picks `_alN_true` when `N % 32 == 0` (the aligned hot path) and
  `_alN_false` otherwise — the Qwen3.6-27B's GDN `in_proj_a`/`in_proj_b` (N = 48)
  exercise the non-aligned variant in the M>1 prefill path (matches
  `mx.quantized_matmul` within the tested tolerance, `max_relative` 1e-2 /
  `epsilon` 5e-2; an exact-equality check differed by 2 bf16 ULP on one element).
  The non-transposed (`transpose=false`) variant lands when a consumer needs it.
- **Dense `matmul` is now production-unused** — every projection routes through
  the quantized path, so the dense `Tensor::matmul` is retained only as
  scaffolding plus its parity test, not on any live forward path.
