# RoPE

A transformer's attention is, by itself, **position-blind**: the `QKᵀ` dot
product treats the keys as an unordered set, so without help "the cat sat"
and "sat the cat" score identically. The model needs to know *where* each
token sits in the sequence. **Rotary position embedding (RoPE)** is how
Qwen2.5 (and most current decoder-only models) injects that information —
not as a vector added to the input, but as a position-dependent *rotation*
of the query and key features, applied right before attention scores them.
This page is about what that rotation computes and why it lands on `Q` and
`K` only.

## Concept

RoPE encodes position by rotating each feature *pair* of `Q` and `K` by an
angle proportional to the token's position. Split a head's `D`-dimensional
vector into `D/2` pairs; pair `i` is rotated by angle `position · θ_i`,
where the per-pair frequency `θ_i` falls off geometrically with `i`:

```text
θ_i = base^(-2i / D)     for i in 0 .. D/2
angle(pos, i) = pos · θ_i
```

`base` (often written `theta`, `10000` for the original formulation and
`1_000_000` for Qwen2.5) sets how fast the frequencies decay: low-index
pairs spin quickly (they encode fine, local position), high-index pairs
spin slowly (they encode coarse, long-range position). A larger base
stretches the slow end out, which is what lets a model address a longer
context without the high pairs wrapping around.

The reason this is worth doing as a *rotation* rather than an additive
position vector is what it does to the dot product. Rotating `Q` at
position `m` and `K` at position `n` by their respective angles makes the
resulting `QᵀK` depend only on the **difference** `m − n`, never on `m` and
`n` absolutely. So attention sees *relative* position for free: a query
three tokens back from its key scores the same whether that pair sits at
the start of the sequence or a thousand tokens in. That relative-distance
property is the whole point, and it falls straight out of the rotation
algebra.

Two consequences shape the implementation:

- **`Q` and `K` only, never `V`.** RoPE exists to shape the *scores* —
  the `QKᵀ` term — so it is applied to exactly the two tensors that feed
  that product. `V` carries the content that gets mixed *after* scoring;
  rotating it would corrupt the payload without affecting which positions
  attend to which. So `V` is left unrotated (see the [attention](./attention.md)
  page, where `rope` is applied to `Q`/`K` and the value path is untouched).
- **Per head.** Each attention head gets its own `D_h`-dimensional vector
  rotated independently — the rotation runs over the per-head feature axis,
  the same axis the head split produced.

## In cider-press

[`rope`](../../crates/cider-press-models/src/qwen2/attention.rs) is the
Qwen2 entry point: a thin wrapper that pulls `rope_theta` and `head_dim`
from the model config and calls `Tensor::rope` with Qwen2.5's defaults
(`traditional = false`, `scale = 1.0`, full-`head_dim` rotation). The
[`Attention`](../../crates/cider-press-models/src/qwen2/attention.rs)
forward calls it once for `Q` and once for `K`, after the per-head layout
and before the [KV-cache](./kv-cache.md) write. The runtime op dispatches
through
[`rope.rs`](../../crates/cider-press-kernels/src/kernels/rope.rs) onto
MLX's [`rope.metal`](../../kernels-mlx/mlx/backend/metal/kernels/rope.metal),
which computes the inverse frequencies on-GPU from `base` (no materialized
`freqs` array) — one threadgroup fan-out rotating the `D/2` pairs of every
(head, position) slot.

The kernel's behavior is **specialized by `[[function_constant]]`**, not by
kernel name — `rope.metal` has a single `rope_bfloat16` entry, and cider
binds three booleans at pipeline-build time to select the variant:
`forward = true`, `traditional = false` (Qwen2's `(x[:d/2], x[d/2:])`
non-interleaved split, as opposed to GPT-NeoX's interleaved layout), and
`hs_transpose = false` (the projection output is already in `[B, H, T, D]`
order). This is the **function-constant** dispatch mechanism, distinct from
the *kernel-name* selection some other ops use (softmax's `block_` vs.
`looped_`); RoPE picks its path purely through constant specialization.

RoPE carries `cos`/`sin` arithmetic, so it falls under the **bf16-ULP
parity tolerance** rather than bit-exactness: `metal::cos`/`sin` drift one
to two bf16 ULPs across Apple-Silicon GPU generations, so `rope` output is
checked against MLX within `~0.005 abs / 0.01 rel`, not bit-for-bit. (Pure
data movers and the qmv/qmm kernels *are* bit-exact; the
transcendental-bearing ones are not — see
[weights and quantization](./weights-and-quantization.md).)

**Model variations.** RoPE is the canonical example of *why the per-model
glue lives in the `qwen2/` submodule rather than the generic
[`nn`](../../crates/cider-press-models/src/nn.rs) surface*: the
bind-config-to-op idiom is genuinely per-model, even though the underlying
kernel is shared.

- **Frequency base / scaling map.** Qwen2.5 uses a single scalar
  `rope_theta` and nothing else — `rope` just rounds it to the kernel's
  `base` parameter. **Llama-3** layers an **NTK-aware** scaling map on top
  (it rescales the low-frequency pairs to extend usable context), and
  **Cohere** differs again in how much of the head dimension is rotated
  (rotary-dim coverage). Each is the *same* `rope.metal` kernel with a
  different config binding — which is exactly why the config-to-op wrapper
  is kept next to the model's `Config` struct instead of generalized into
  `nn`. (The kernel already exposes a `_freqs` explicit-frequencies variant
  for the YaRN / Llama-3 path; it is deferred until a model that needs it
  lands — see [`rope.rs`](../../crates/cider-press-kernels/src/kernels/rope.rs).)
- **Traditional (interleaved) vs. non-traditional.** GPT-NeoX-style models
  rotate *interleaved* feature pairs; Qwen2.5 uses the non-traditional
  `(x[:d/2], x[d/2:])` split. Selected by the `traditional` function
  constant, not a different kernel.
- **RoPE vs. ALiBi vs. learned/absolute.** RoPE is one of several ways to
  put position into attention. ALiBi instead adds a distance-based bias
  into the scores; older models used learned absolute position embeddings
  added to the input. Qwen2.5 is RoPE (see the [attention](./attention.md)
  page's positioning note).

See [models/qwen2.5.md](./models/qwen2.5.md) for Qwen2.5's concrete
`rope_theta` and head-dimension values.

## Performance

RoPE is small and **off the critical path** — two dispatches per layer
(one for `Q`, one for `K`), rotating `[1, 14, T, 64]` and `[1, 2, T, 64]`
tensors against the bandwidth-/compute-bound quantized matmuls that
dominate every block. Its arithmetic share of both decode and prefill is
negligible relative to the projections it sits between, and it has never
shown up as a lever in the dispatch-reduction or prefill work.

It is applied as a **strided** op: the per-head layout that feeds `rope` is
a view (`reshape`/`permute` over the projection output, head axis
unit-stride), and `rope` writes its output in the same layout — no
materialized contiguous copy before or after. This is part of the
**strided-RoPE write** that eliminated the per-layer Q/K project-head
copies (`gpu.copy` 146 → 26 across the prefill pass), landed alongside the
strided KV-cache write. That copy elimination **measured a no-op on
wall-clock** — the copies it removed were off the qmm-bound critical path
and overlapped the matmuls rather than serializing behind them — but the
change was **kept** because it is correct, moves less memory traffic, and
matches MLX's strided-input behavior (see the [attention](./attention.md)
page's copy-elimination note).

## Open levers

None material. RoPE is a single function-constant-specialized
`rope_bfloat16` dispatch per `Q`/`K`, applied strided in place, off the
critical path. The strided write already lands; there is no copy left to
elide and no fusion that would shorten a chain RoPE meaningfully sits on.
The remaining decode and prefill levers live on the projections and the
execution structure, not on RoPE (see [quantized matmul](./quantized-matmul.md)
and the [execution model](./execution-model.md)).
