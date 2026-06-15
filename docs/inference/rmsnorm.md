# RMSNorm

Every sub-layer in a transformer block sees its input through a
normalization first (the pre-norm placement — see
[transformer block](./transformer-block.md)). The normalization Qwen2.5 uses
is **RMSNorm**, the cheap descendant of LayerNorm that most modern
decoder-only models settled on. This page is about what it computes and why
that particular cheap form is enough.

## Concept

Activations flowing down a deep residual stack drift in scale — the residual
add is an accumulator, so without intervention the magnitudes grow layer over
layer until the matmuls and the softmax see numbers they were never tuned for.
Normalization rescales each activation vector back to a stable magnitude
before the sub-layer reads it, so every block sees inputs in roughly the same
range no matter how deep it sits.

The original transformer used **LayerNorm**, which standardizes each vector to
zero mean and unit variance, then applies a learned scale and shift:

```text
LayerNorm(x) = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta
```

**RMSNorm** drops two of those four pieces. It does *not* subtract the mean,
and it has *no* bias term — it only rescales by the root-mean-square of the
vector, then applies a learned per-channel scale:

```text
rms(x)     = sqrt(mean(x^2) + eps)
RMSNorm(x) = x / rms(x) * gamma
```

`mean(x^2)` is the average of the squared elements across the normalized axis
(the hidden dimension); `eps` is a small constant (`1e-6` for Qwen2.5) added
under the root so a near-zero vector cannot divide by zero. The `gamma` is a
learned `[hidden]` vector — one scale per channel — and that is the only
parameter RMSNorm carries.

The bet RMSNorm makes is that the *mean-subtraction* and the *bias* in
LayerNorm were not pulling their weight. Re-centering each vector to zero mean
costs a second reduction pass and, empirically, buys almost nothing once a
network is trained with it from the start; what actually stabilizes the stack
is fixing the *scale*, which the RMS term alone does. Dropping the mean also
makes the op cheaper and — because there is no subtraction — numerically
simpler: one sum of squares, one reciprocal square root, one scale. The result
is a normalization that costs about half of LayerNorm's reductions and matches
its quality on these architectures, which is why Llama, Qwen, and most of their
contemporaries use it.

## In cider-press

[`nn::rms_norm`](../../crates/cider-press-models/src/nn.rs) is the composed
entry point, and [`RmsNormLayer`](../../crates/cider-press-models/src/nn.rs)
is the stateful [`Module`](../../crates/cider-press-models/src/nn.rs) wrapper
that carries the `gamma` weight and `eps` and applies it on `forward`. But
`nn::rms_norm` does almost nothing itself — it delegates straight to the
**fused** [`Tensor::rms_norm`](../../crates/cider-press-runtime/src/tensor.rs)
op, which is one dispatch of MLX's
[`rms_single_row`](../../kernels-mlx/mlx/backend/metal/kernels/rms_norm.metal)
kernel, dispatched from
[`rms_norm.rs`](../../crates/cider-press-kernels/src/kernels/rms_norm.rs). That
kernel does the whole `square → mean → rsqrt → scale` in a single threadgroup
per row, reducing the sum of squares in fp32 regardless of the bf16 input:

```text,ignore
pub fn rms_norm(x: &Tensor, gamma: &Tensor, eps: f32) -> Result<Tensor> {
    Ok(x.rms_norm(Some(gamma), eps)?)   // → fused Tensor::rms_norm → rms_single_row
}
```

**Why fused, when the rest of `nn.rs` composes from primitives.** The house
rule in `cider-press` is *compose at the models layer when `mlx.nn` does — but
only when it actually does*. For SiLU and GELU it does (`mlx.nn` builds them in
Python from `sigmoid`/`erf`, so `nn::silu`/`nn::gelu` compose too). For RMSNorm
it does **not**: `mlx.nn.RMSNorm` calls `mx.fast.rms_norm`, the same fused
kernel. So composing RMSNorm out of `square`/`mean`/`rsqrt`/`mul` here would be
a **six-dispatch deviation *from* MLX**, not a step toward it — more dispatches,
a longer decode critical path, and a divergence from the reference path we
check parity against. Delegating to the fused kernel is the MLX-faithful choice
*and* the fast one; they happen to coincide.

That fused kernel carries `exp`-free but `rsqrt`-bearing arithmetic, so it
falls under the **bf16-ULP parity tolerance** rather than bit-exactness:
`metal::rsqrt` (like `exp`/`cos`/`sin`) drifts one to two bf16 ULPs across
Apple-Silicon GPU generations, so RMSNorm output is checked against MLX within
`~0.005 abs / 0.01 rel`, not bit-for-bit. (Pure data movers and the
qmv/qmm/dequantize kernels *are* bit-exact; the transcendental-bearing ones are
not — see [weights and quantization](./weights-and-quantization.md).)

`Tensor::rms_norm` takes the gamma as `Option<&Tensor>`. `Some(gamma)` is the
affine path checked above. `None` is the **weightless** path — mlx's
`mx.fast.rms_norm(x, weight=None)` — used by Qwen3.5's Gated-DeltaNet q/k norm,
which normalizes each head row without a learned scale. The forward kernel has
no separate weightless branch (it always multiplies by `w[w_stride * i]`; the
`has_w` function-constant only drives the unused VJP variants), so `None` binds
a device-cached one-element `[1.0]` scalar with `w_stride = 0` — every lane
reads `w[0]`, an identity gamma — instead of uploading a `[hidden]` ones-vector
per call. Same math, fewer bytes, mlx-faithful.

`Tensor::rms_norm` carries the precondition checks: `x` dense, contiguous, and
bf16 with the hidden dimension last; a `Some` `gamma` a rank-1 bf16 vector
matching that last axis; `eps` finite and non-negative. The single fused kernel handles
Qwen2.5-0.5B's `hidden = 896` comfortably — it is well within the single-row
regime (`rms_single_row` handles axes up to 4096; wider axes would need the
looped variant, which is not wired because no supported model needs it).

**Model variations.** RMSNorm is one point in a small design space:

- **RMSNorm vs. LayerNorm** — Qwen2.5 (and the Llama family) use RMSNorm: no
  mean-subtraction, no bias, just RMS-rescale times `gamma`. Older / encoder
  architectures (the 2017 transformer, BERT, GPT-2) use full LayerNorm with the
  mean-subtraction and a `beta` bias. The op signature here carries no `beta`
  because the target family has none.
- **Weight offset (`gamma` vs. `1 + gamma`)** — Qwen2.5 applies the learned
  scale as plain `gamma`. **Gemma** instead stores the weight centered at zero
  and applies `(1 + gamma)` — the same op, but the checkpoint's weight column
  is offset by one. That is a *loader* concern (add one to the stored weight at
  load, or fold it into the kernel) rather than a different normalization; the
  kernel here implements the plain-`gamma` form Qwen2.5 needs.

See [models/qwen2.5.md](./models/qwen2.5.md) for Qwen2.5's concrete norm
choices (pre-norm placement, `eps = 1e-6`, the two per-block norms plus the
final norm).

## Performance

RMSNorm was the **first** fusion lever in the decode-dispatch reduction work,
and it paid off out of proportion to the op's arithmetic cost — because the
decode bottleneck is the *length of the dependent-dispatch chain*, not raw
compute. Each RMSNorm fires `49` times per token (two per block × 24 blocks,
plus the one final norm), and the old composed form was **six dispatches**
each: roughly `294` dispatches per token spent on a sequence of cheap
element-wise passes, each one a separate link in the critical path the GPU has
to wait through.

Collapsing that composition into one `rms_single_row` dispatch per call cut
decode from **~975 → ~730 dispatches per token** and lifted throughput from
**~240 → ~302 tok/s (~1.26×)** (the decode-dispatch lever; see the
[execution model](./execution-model.md) page). The
win is almost entirely the shorter chain — five fewer dependent dispatches × 49
calls — not faster math; the arithmetic was always trivial relative to the
per-dispatch latency it sat behind. It also realigned the path with
`mlx.nn.RMSNorm`'s own `mx.fast.rms_norm`, so cider and the parity reference now
issue the same single op here.

## Open levers

None. RMSNorm is already the single fused `rms_single_row` dispatch — the exact
op `mlx.nn.RMSNorm` issues — so it is at MLX parity with nothing left to fuse or
elide. The remaining decode and prefill levers live on the projections and the
execution structure, not the norm (see [quantized matmul](./quantized-matmul.md)
and the [execution model](./execution-model.md)).
