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
inner dim matches the weight's logical `K`, and (for the `M > 1` qmm path only)
that `N` is a multiple of 32, since the wired `affine_qmm_t` instantiation is
the aligned variant.

**The `qmv_fast` vs. generic variant.** `qmv.rs` mirrors MLX's dispatcher: it
selects the faster `affine_qmv_fast` kernel when `K % 512 == 0 && N % 8 == 0`,
and the generic `affine_qmv` otherwise. The selection is purely by shape
alignment — same kernel source, a faster tiling when `K` is a 512-multiple.
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
| 896×4864    | gate/up_proj | ~306 GB/s |
| 4864×896    | down_proj    | ~272 GB/s |
| 896×151936  | lm_head      | ~464 GB/s |
| 896×896     | q/o_proj     | ~85 GB/s  |
| 896×128     | k/v_proj     | ~13 GB/s  |

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

## Open levers

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
- **The generic non-aligned `affine_qmm` is unimplemented.** `qmm.rs` wires only
  the `transpose=true, aligned (N % 32 == 0), B=1` variant; the `_alN_false` and
  non-transposed variants land when a consumer needs them.
- **Dense `matmul` is now production-unused** — every projection routes through
  the quantized path, so the dense `Tensor::matmul` is retained only as
  scaffolding plus its parity test, not on any live forward path.
