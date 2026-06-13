# Feed-forward (SwiGLU)

The second sub-layer of every transformer block is the **feed-forward
network** — the *channel mixer* that reshapes each position's vector on its
own, with no information crossing between positions (that is attention's job;
see [transformer block](./transformer-block.md)). It is also where most of a
layer's weight bytes live. This page is about what that network computes, why
Qwen2.5 uses the *gated* SwiGLU form rather than a plain MLP, and the shape
that gating implies.

## Concept

The feed-forward network is **position-wise**: the same small network is
applied independently to every position's `[D]` vector. Its job is to give the
model non-linear capacity per position — attention moves information *between*
tokens, but it is (per head) a weighted average, a fundamentally linear mix;
the FFN is where each position gets transformed through a non-linearity.

The classic form is a two-layer MLP: project *up* into a wider hidden
dimension, apply a non-linear activation, project back *down*:

```text
MLP(x) = down(act(up(x)))           # up: [D] → [F], down: [F] → [D]
```

The widening (`F > D`) is the point — the intermediate dimension `F` is where
the non-linear work happens, and making it several times wider than `D` is
what gives the layer its capacity. The **expansion ratio** `F / D` is a
per-model knob; for Qwen2.5-0.5B it is `4864 / 896 ≈ 5.4×`.

**SwiGLU** replaces that single up-projection with a *gated* one. Instead of
one `up`, it runs **two** parallel projections of the same input — a `gate`
and an `up` — and multiplies them element-wise, with an activation on the gate
arm only, before the down-projection:

```text
SwiGLU(x) = down( silu(gate(x)) * up(x) )
```

`gate` and `up` are both `[D] → [F]`; their element-wise product is `[F]`;
`down` brings it back to `[D]`. So the network has **three** projections
(two in, one out) where the plain MLP had two. The intuition for the gate: the
`silu(gate(x))` arm acts as a learned, input-dependent multiplicative mask over
the `up(x)` features — the network can suppress or pass each of the `F`
intermediate channels conditionally on the input, rather than applying one
fixed activation shape. Gated linear units (GLUs) consistently outperform a
plain MLP at equal parameter count, which is why Llama, Qwen, and most modern
decoder-only families adopted SwiGLU. (To keep parameter count comparable to a
plain MLP despite the third matrix, gated models typically shrink `F` by ~⅔ —
Qwen2.5's `5.4×` already reflects that.)

The `silu` (a.k.a. Swish) activation is `silu(x) = x · sigmoid(x)` — a smooth,
non-monotonic curve that passes a little negative signal through near zero
rather than hard-clipping it like ReLU. The "Swi" in SwiGLU is the SiLU gate;
the "GLU" is the gating multiply.

## In cider-press

[`Mlp`](../../crates/cider-press-models/src/nn.rs) is the SwiGLU module. It
owns three bias-free [`Linear`](../../crates/cider-press-models/src/nn.rs)
projections — `gate`, `up`, `down` — and its `forward` is the recipe verbatim:

```text,ignore
let gated = silu(&self.gate.forward(x)?)?;   // silu(gate(x))
let up    = self.up.forward(x)?;             // up(x)
self.down.forward(&gated.mul(&up)?)          // down( silu(gate(x)) * up(x) )
```

Each of the three projections is a **quantized matmul** — `gate`/`up` are the
two `[D] → [F]` (`896 → 4864`) up-projections, `down` is the `[F] → [D]`
(`4864 → 896`) down-projection — so see [quantized matmul](./quantized-matmul.md)
for the qmv/qmm kernels they dispatch. All three are *bias-free*:
[`Mlp::new`](../../crates/cider-press-models/src/nn.rs) rejects any `Linear`
carrying an additive bias, because mixing a biased projection into a SwiGLU MLP
would silently compute a different op (Qwen2's MLP projections have no `.bias`;
only Q/K/V do — see the loader footgun in
[`CLAUDE.md`](../../CLAUDE.md)).

The activation, [`nn::silu`](../../crates/cider-press-models/src/nn.rs), is
**composed** from primitives — `silu(x) = x.sigmoid().mul(x)`, two dispatches —
rather than a single fused kernel. That is deliberate and follows the house
rule: *compose at the models layer when `mlx.nn` does*. MLX's `unary.metal` has
**no** `Silu` (or `Gelu`) kernel; `mlx.nn.silu` builds it in Python from
`sigmoid`, so `nn::silu` mirrors that exactly. (Contrast RMSNorm, which *is*
fused, because `mlx.nn.RMSNorm` calls a fused `mx.fast.rms_norm` — the rule is
"match what `mlx.nn` actually does," not a blanket preference either way; see
[RMSNorm](./rmsnorm.md).) The sibling
[`nn::gelu`](../../crates/cider-press-models/src/nn.rs) is composed for the same
reason (exact `erf`-based, matching `mlx.nn.gelu`), ready for a GeGLU model.

**Model variations.** The FFN is a busy point in the design space:

- **SwiGLU vs. GeGLU vs. plain MLP** — the gate arm's activation is the knob.
  Qwen2.5 (and Llama) use **SwiGLU** (SiLU gate). **GeGLU** swaps SiLU for GELU
  (`nn::gelu` is already in place for it). The pre-GLU families (GPT-2, the
  original transformer) use a **plain MLP** — `down(act(up(x)))`, two
  projections, no gate — typically with GELU or ReLU.
- **Mixture-of-Experts (MoE)** — the largest variation, and the major
  *unbuilt* feature. An MoE layer replaces the single shared FFN with `E`
  parallel expert FFNs plus a small **router**: per token, the router scores
  the experts and only the **top-`k`** run, so the layer has the parameter
  count of `E` MLPs but the compute of `k`. The projections become *batched*
  per-expert matmuls — MLX dispatches these through `gather_qmm` (a gathered
  quantized matmul that indexes the selected experts' weights). This is where
  the eventual MoE-target model will diverge from the dense path above; a
  future [`moe.md`](./moe.md) will cover the router, top-`k` selection, and the
  `gather_qmm` dispatch.

See [models/qwen2.5.md](./models/qwen2.5.md) for Qwen2.5's concrete FFN shape
(`D = 896`, `F = 4864`, SwiGLU, bias-free).

## Performance

The three FFN projections are a large share of per-layer compute — together
they move more weight bytes than the four attention projections (the `gate`,
`up`, and `down` matrices are each `896 × 4864`-ish, the widest weights in the
block bar the LM head). Unlike the small attention K/V projections, they are
**large-N linears that run near the bandwidth roofline**: the qmv audit
measures `gate`/`up` at **~306 GB/s** and `down` at **~272 GB/s** in decode,
approaching the ~464 GB/s empirical kernel ceiling (and the M4 Max
~410–546 GB/s spec roofline). That is the opposite end of the spectrum from the
occupancy-starved small-N projections (`k/v_proj` ~13 GB/s), so there is little
kernel-side headroom here — the FFN matmuls are essentially doing as well as the
hardware allows. The full bandwidth story, and why large-N saturates while
small-N starves, lives in [quantized matmul](./quantized-matmul.md); the
`silu` and `mul` between the projections are cheap element-wise passes over the
`[B, T, 4864]` intermediate, negligible against the matmuls.

## Open levers

- **MoE routing and expert dispatch** is the major unbuilt feature — the
  next-model lever, not a tuning of the dense path. It introduces the router,
  top-`k` expert selection, and `gather_qmm` (gathered quantized matmul), plus
  the **loop-or-fuse** decision: dispatch the selected experts as a loop of
  per-expert qmms, or fuse the gather into a single batched kernel. None of
  this exists yet; it lands with the first MoE target model (see the
  forthcoming [`moe.md`](./moe.md)).
- **SiLU stays composed** — `x · sigmoid(x)`, two dispatches — and that is
  **not** a lever. MLX ships no fused SiLU kernel, so fusing one here would be
  a *deviation from* the parity reference, not an alignment with it (same logic
  as the [RMSNorm](./rmsnorm.md) "compose when `mlx.nn` does" rule, applied the
  other way). The element-wise activation is also off every measured critical
  path — the FFN cost is the three matmuls, already near roofline.
