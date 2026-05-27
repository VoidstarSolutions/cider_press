# CLAUDE.md

Guidance for Claude (and humans) working in this repo.

---

## Project context

`cider-press` is an inference-only tensor runtime and model loader for
Apple Silicon. Single-platform scope is deliberate: it lets us lean on
unified memory, ship hand-tuned Metal kernels, and skip the general-
purpose device abstraction that costs portable frameworks performance on
Mac.

The design hypothesis: most of the Apple-Silicon perf gap between
candle/burn and MLX is in the **graph/dispatch layer**, not the kernels.
MLX's `.metal` kernels are MIT-licensed and reusable. If we can load
them verbatim, dispatch them from Rust, and wrap them in a lazy graph
that fuses and batches command buffers, we capture most of MLX's perf
in idiomatic Rust.

That hypothesis is currently **unproven**. The entire project is gated
on the spike described below.

### Non-goals

- Training / autograd
- CUDA, ROCm, Vulkan, WebGPU
- Windows, Linux
- A general-purpose tensor library — this exists to run LLMs.

### Acknowledgments

Metal kernels and several design decisions derive from
[MLX](https://github.com/ml-explore/mlx) (MIT, © 2023 Apple Inc.). Any
file derived from MLX must retain the MIT copyright header.

---

## Repository layout

The crate is a Cargo workspace; each layer is its own member crate so
the dispatch surface, the runtime, and the model layer can be depended
on independently (and published independently down the road).

```
cider-press/
├── Cargo.toml                          # workspace manifest only
├── kernels-mlx/                        # vendored MLX kernel sources (MIT)
│   ├── COPYING                         # MLX MIT notice
│   ├── VENDORED.md                     # file list, upstream SHA, sync procedure
│   └── mlx/backend/metal/kernels/      # full upstream path preserved
├── crates/
│   ├── cider-press-kernels/            # vendored MLX dispatch surface
│   │   ├── build.rs                    # flattener: kernels-mlx/ → OUT_DIR/*_inlined.metal
│   │   ├── src/lib.rs                  # device/buffer/library + per-kernel dispatch (TBD)
│   │   └── tests/{stage*,fixtures}     # spike parity + perf tests
│   ├── cider-press-runtime/            # scaffold: lazy Tensor + graph + eval
│   ├── cider-press-models/             # scaffold: transformer architectures
│   ├── cider-press-test-utils/         # dev-only: shared MLX-parity test helpers
│   └── cider-press/                    # scaffold: facade + CLI
├── scripts/                            # one-shot uv scripts (fixtures, MLX sync, perf)
└── .github/workflows/{ci,mlx-sync}.yml
```

Dependencies flow strictly upward: `cider-press-models` depends on
`cider-press-runtime` depends on `cider-press-kernels`. The facade
crate re-exports from runtime + models. Workspace-level
`[workspace.lints]` and `[workspace.dependencies]` keep settings and
version pins in one place; each member uses `lints.workspace = true`
and `dep.workspace = true`.

## Current state: quantized matmul + layer-0 attention integration test landed

The development spike (Stages 0–5 below) passed every acceptance bar,
including bit-exact `qmv` parity with MLX's own `quantized_matmul`.
Branches landed since then (see `docs/QWEN_PATH.md` for the
branch-by-branch roadmap):

- **Branch 1 (`feat/runtime-data-primitives`).** `Device` handle with
  cached kernel libraries (`copy`, `quantized`); lazy `Tensor` with
  materialization-state-via-`OnceLock`; `Tensor::copy` for all five
  dtypes; `QuantizedWeight::matvec` for bf16 (runtime-level parity
  test against the Stage-4 fixture). `Tensor::eval` is the sync
  public boundary — single `Commands` per call, synchronous commit
  + wait. Internal pipelining stays deferred.
- **Branch 2 (`feat/runtime-views`).** `ViewSource` on `TensorInner`
  plus `Tensor::{reshape, permute, slice, broadcast_to}` + a strided
  `CpuIter` host accessor. Zero-copy views chase the backing leaf
  through a chain-flattening helper. `cpu_bytes` is now contract-
  bounded to "logical contiguous window or None"; non-contiguous
  views are read via `cpu_iter` / `cpu_to_vec`.
- **Branch 3 (`feat/weight-loading`).** `cider-press-models` scaffold
  with: a generic `safetensors_io` adapter (dense + MLX three-key
  quantized triples), a `Qwen2Config` serde-derived from
  `config.json` (HF revision pinned), and `Qwen2Weights` +
  `load_qwen2_weights` walking every key in HF's safetensors layout
  into typed runtime tensors. Loaded bytes round-trip the archive
  byte-for-byte (synthetic CI test + gated real-checkpoint test on
  `CIDER_QWEN_CHECKPOINT_PATH`). Includes the per-op MLX activation-
  dump harness `scripts/dump_mlx_op.py` that every later op branch
  adds cases to.
- **Branch 4 (`feat/elementwise-add`).** First binary op family
  end-to-end. Vendored MLX's `binary.metal` (+ `binary.h`,
  `binary_ops.h`); kernels-crate exposes 18 typed dispatch fns
  (`{vv,g1,g2,g3}_{add,mul}_{f32,f16,bf16}`). Runtime adds
  `BinaryOp { Add, Mul }`, `OpKind::Binary { op }`, and
  `Tensor::add` / `Tensor::mul` with NumPy broadcasting via the
  existing `broadcast_to` view. `dispatch_binary` picks `vv_*` when
  both inputs are non-view dense tensors with the same shape as the
  output and `g{1,2,3}_*` when broadcasting injected a view; rank
  > 3 in the strided case is deferred (`gn2_/gn4large_` not wired).
  `Device::binary_library()` caches the JIT'd `binary.metal`. Adds
  the `mul` case to `dump_mlx_op.py`. Validated bit-exactly vs MLX
  at three layers — kernels, runtime, and models (synthetic
  Qwen2-shape add+mul plus gated real-checkpoint test that loads
  `layer0.input_layernorm` and broadcast-adds it bit-exactly vs a
  CPU bf16 reference). CI now installs `uv` via
  `astral-sh/setup-uv@v6` so parity tests can shell out to
  `dump_mlx_op.py` at test time without checking in fixture bytes.
- **Branch 5 (`feat/rmsnorm`).** First reduction + unary op
  families, then `RMSNorm` composed from them. Vendored
  `unary.metal` + `reduce.metal` (and transitive headers). Kernels
  crate exposes `v_{square,rsqrt}_{f32,f16,bf16}` (6 fns) and
  `row_reduce_{sum,prod,min,max}_{f32,f16,bf16}` (12 fns); the
  reduce surface adopts a cleaner API than mirroring kernel host
  names — one `pub fn` per (op, dtype) that internally issues
  `init_reduce_*` + the right reducer variant. Variant scope is
  `row_reduce_looped` only; simple/all/col land when consumers
  appear. Axis scope is last-axis only; non-last reductions can
  be expressed via `permute` first. Runtime adds `OpKind::Unary`
  + `OpKind::Reduce` (`ReduceKind::Sum/Prod/Min/Max` — no `Mean`,
  composed) and `Tensor::{square, rsqrt, sum, prod, min, max,
  mean}`. The eps in RMSNorm flows through the graph as a
  host-side `[1]`-tensor broadcast through the existing binary
  path (deferred MLX scalar `sv_/vs_` kernels; perf delta vs `[1]`
  broadcast is essentially zero at this scale). Models crate ships
  `nn::rms_norm(x, gamma, eps)` composed as `x * rsqrt(mean(x²) +
  eps) * gamma`. Validated against `mx.fast.rms_norm` at bf16
  tolerance (composed path costs more bf16 round-trips than the
  fused kernel; well under 1 bf16 ULP) plus a gated real-checkpoint
  test that runs `nn::rms_norm` against the loaded layer-0
  `input_layernorm` weight at `config.rms_norm_eps`, vs a CPU f32
  reference.
- **Branch 6 (`feat/silu-and-gelu`).** Two activation functions
  composed from existing unary primitives. Branch surface ended up
  meaningfully different from the original roadmap: MLX's `unary.metal`
  doesn't ship `Silu` / `Gelu` as primitive ops — `mlx.nn.silu` and
  `mlx.nn.gelu` are themselves compositions on the Python side. So
  branch 6 mirrors that factoring: kernels crate adds `Sigmoid` and
  `Erf` (6 new fns; instantiations already present in the vendored
  `unary.metal`, no new sources), runtime adds `UnaryOp::{Sigmoid,
  Erf}` + `Tensor::{sigmoid, erf}`, and models crate composes
  `nn::silu(x) = x * sigmoid(x)` and exact `nn::gelu(x) = 0.5 * x *
  (1 + erf(x / sqrt(2)))`. Sigmoid (and therefore SiLU) uses a tight
  bf16-ULP tolerance (0.005 abs / 0.01 rel, ~2 ULPs) rather than
  bit-exact equality: `metal::exp` inside MLX's vendored Sigmoid kernel
  drifts 1–2 bf16 ULPs across Apple Silicon generations (M-series local
  vs macos-15 CI runners), even though the kernel source matches MLX
  byte-for-byte. The other unary kernels (square / rsqrt / erf) happen
  to stay bit-exact at the shapes we test. GELU uses bf16-compose
  tolerance (0.02 abs/rel) because MLX writes `x / sqrt(2)` (division
  by bf16-rounded `sqrt(2)`) where we reciprocal-multiply (no divide op
  yet); the bf16 constants differ in the last few bits. Adds `sigmoid`,
  `erf`, `silu`, `gelu` cases to `dump_mlx_op.py` (the silu/gelu cases
  drive against `mlx.nn`).
- **Branch 7 (`feat/gather`).** Embedding lookup end-to-end, against
  Qwen2's *quantized* embed_tokens. First branch that needed
  JIT-assembled kernels — MLX has no precompiled `gather.metal`;
  `indexing.cpp` assembles source per `(T, IdxT, NIDX, IDX_NDIM,
  LocT)` instantiation. Vendored `indexing/{indexing,gather}.h`.
  build.rs grows a `HEADER_BUNDLES` pass alongside `ENTRY_POINTS`
  that flattens header-only sources into prefix strings.
  Kernels crate: `kernels::gather::{Instantiation, make_source,
  dispatch}` with `bf16_u32_rank1` / `u32_u32_rank1` instantiations
  + the wrapper template copied verbatim from MLX's
  `backend/metal/jit/indexing.h`; reusable for softmax/RoPE/SDPA
  later (same JIT pattern). Also adds
  `affine_dequantize_bf16_gs64_b4` reusing the existing
  `quantized.metal` library — Qwen2 production config. Runtime:
  per-`(T, IdxT, NIDX, IDX_NDIM, LocT)` `Mutex<HashMap>` library
  cache on `Device` (the multi-instantiation case where `OnceLock`
  doesn't fit), `Tensor::gather` accepting BF16 + U32 src,
  `OpKind::Dequantize { group_size, bits }` + `Tensor::dequantize_affine`,
  `QuantizedWeight::components()` exposing the packed triple as
  three dense `Tensor`s sharing the underlying `MTLBuffer`s
  (refcount-bumped, zero copy). Models crate ships
  `nn::embed_tokens(input_ids, weight)` composing the lazy 4-op
  graph (gather × 3 + dequantize). All five parity tests are
  bit-exact (gather is a pure data-mover; per-row dequantize is
  per-row identical between MLX and us). Real-checkpoint test
  exercises the full path on Qwen2.5-0.5B's loaded `embed_tokens`,
  comparing against a CPU per-row dequantize from the loaded raw
  bytes. JIT cold-start is per-instantiation (one cache miss per
  unique tuple); cross-process Metal cache amortizes subsequent
  process invocations the same way it does for `.metal` libraries.

**Concrete target: Qwen2.5-0.5B-Instruct running interactively.**
Smallest published dense Qwen; same architecture family as the
eventual MoE target minus the router. `docs/QWEN_PATH.md` is the
branch-by-branch roadmap; `docs/RUNTIME_DESIGN.md` has the
framework-gap analysis for the two structural additions we knew
we'd need — both now landed (Views in branch 2; `KvCache` in branch 8).

- **Branch 8 (`feat/kv-cache`).** First runtime-side abstraction
  that isn't a lazy-graph op. `KvCache` is a separate type with
  pre-allocated `[max_tokens, n_kv_heads, head_dim]` slabs per K/V
  and a `position` counter; `update(k, v)` is eager — `Tensor::eval`s
  the inputs, then `memcpy`s their bytes into the slab via the
  unified-memory host pointer (Apple Silicon's shared storage makes
  this a host write, no Metal dispatch). `keys_view()`/`values_view()`
  return zero-copy `Tensor`s over the populated prefix via
  `Tensor::host_leaf` + a refcount-bumped `Buffer<u8>` clone; the
  aliasing contract (drop views before the next `update`) is
  documented on the type. Reset path for restarting a decode loop.
  Roundtrip test: multi-chunk `update`s into the cache and read back
  bit-exactly via `cpu_to_vec`; overflow / dtype / shape / device
  validation each rejected. Plan deferred: same-eval batching (folding
  the cache write into the SDPA command buffer) lands opportunistically
  in branch 11 if perf demands; true lazy KvCache stays deferred
  until branch-15 measurement justifies the mutable-leaf rework.
- **Branch 9 (`feat/rope`).** First op that uses Metal
  `[[function_constant]]` specialization. MLX ships
  `rope.metal` precompiled (229 LOC) with all instantiations baked
  in, so this is the `OnceLock<KernelLibrary>` pattern (same camp
  as binary/unary/reduce); the per-specialization pipelines live in
  the library's own pipeline-state cache. New
  `FunctionConstant::Bool { index, value }` enum +
  `KernelLibrary::pipeline_specialized(name, &[FunctionConstant])`
  in the kernels crate; the cache key derives from the index-sorted
  constants tuple so the same `(name, constants)` combination
  always hits the same entry and unspecialized `pipeline()` calls
  don't collide with any specialization. `Bool`-only surface;
  extend with `U32` / `F32` when softmax / SDPA need them. One
  dispatch wired (`kernels::rope::dispatch_rope_bf16`): Qwen2
  inference forward, `traditional=false, hs_transpose=false`,
  `with_freqs=false`, int32 indexing. The seven other
  instantiations (`_single`, `_freqs`, `_large` × dtype) stay
  dormant. `Tensor::rope(&self, offset, base, scale, rotary_dims)`
  carries `offset` as a graph-input edge (length-1 I32 tensor)
  rather than a host scalar — matches MLX's
  `set_input_array(offset, 2)`. The `Eq` derive on `OpKind` drops
  because `base_log2` / `scale` are `f32`; `PartialEq` still gives
  test ergonomics. Models-crate adds
  `qwen2::attention::rope(x, offset, &Qwen2Config)` — a thin
  config-binding wrapper that pulls `rope_theta` and `head_dim`,
  living in a new `qwen2::attention` submodule that future
  attention bits (Q/K/V projections, SDPA wiring) accumulate into.
  All three layers bit-exact vs `mx.fast.rope` at `offset=0`
  (identity rotation, cos=1/sin=0). At `offset=37` (decode), tight
  bf16-ULP tolerance (0.005 abs / 0.01 rel — same bar as sigmoid in
  branch 6) instead of bit-exact: `metal::cos` / `metal::sin` drift
  1–2 bf16 ULPs across Apple Silicon generations (M-series local
  vs macos-15 CI runners). Tested at `[1, 14, 4, 64]` (Q) and
  `[1, 2, 4, 64]` (K). No real-checkpoint test because rope
  applies to projection *outputs* and qmv-as-`Tensor`-op lands in
  branch 11b; the full layer-0 Q-projection → rope → cache → SDPA
  integration test is a branch-11 task.
- **Branch 10 (`feat/softmax`).** Numerically-stable last-axis
  softmax for attention scores. MLX ships `softmax.metal`
  precompiled (24 LOC entry-points + `softmax.h` 191 LOC) —
  another `OnceLock<KernelLibrary>` slot. **Correction to a
  branch-9 note: softmax does NOT use `[[function_constant]]`
  specialization** — variants are selected by kernel name
  (`block_softmax_<dtype>` vs `looped_softmax_<dtype>`, with
  `_precise_` mirrors that swap the half-precision accumulator
  for `float`). So `FunctionConstant` stays `Bool`-only; SDPA
  is the next candidate to need extension. Two entry points
  wired: `block_softmax_bfloat16` (bf16 accumulator,
  `precise=False`) and `block_softmax_precise_bfloat16` (float
  accumulator, `precise=True`). Looped + f32/f16 dtypes stay
  dormant. Threadgroup math ports MLX's `Softmax::eval_gpu`
  arithmetic verbatim (`SIMD_SIZE=32`, `N_READS=4`,
  `BLOCK_AXIS_LIMIT=4096`). `Tensor::softmax(&self, precise:
  bool)` is last-axis-only; the constructor hard-errors on
  axis sizes above `BLOCK_AXIS_LIMIT` rather than silently
  producing wrong output. Tolerance bars (not bit-exact) match
  the unary sigmoid story — `simd_max` / `simd_sum` reductions
  depend on lane-order summation and `metal::fast::exp` carries
  a relative-precision bound. The precise variant is ~2× tighter
  than the default because its float accumulator absorbs the
  lane-summation drift before the final bf16 cast.

- **Branch 11 (`feat/sdpa-split`).** Scaled dot-product attention
  as three composed ops: `qk_matmul`, `softmax(scale + mask)`,
  `attn_matmul`. **First non-MLX kernel in the tree** — MLX's
  precompiled GEMM is the templated `steel_gemm_*.metal` family
  (header-only, register-tiled), too much surface area for a
  branch whose goal is the *composition*, so we shipped our own
  naive bf16 batched matmul (`crates/cider-press-kernels/src/kernels/matmul.metal`,
  one thread per output element, float accumulator). The roadmap
  flags steel migration as branch-15 perf work. Build.rs gains
  no new entry — the kernel is `include_str!`'d directly from the
  crate (no MLX includes to resolve), parallel to the
  `KernelLibrary::matmul()` slot. Kernels crate: one dispatch
  fn (`kernels::matmul::dispatch_gemm_bf16`) taking
  `(A, B, C, M, N, K, batch, a_batch_stride, b_batch_stride, c_batch_stride)`
  in elements. Runtime: `OpKind::MatMul` + `Tensor::matmul(&self,
  other)` validating dense-contiguous-bf16-same-leading-dims, with
  the matmul-shape validation extracted to a `matmul_shapes()`
  helper that 11b's `qmm` can reuse. `dispatch_matmul` walks
  reshape-view chains via a new `matmul_input_bytes` helper —
  the SDPA composition produces `probs_3d.reshape([B, H_q, T,
  T_c])` view inputs that the dense matmul kernel can still read
  contiguously (asserts zero byte-offset; non-zero would need a
  `setBuffer:offset:` bind). Models crate ships
  `qwen2::attention::sdpa(q, k, v, mask, &Qwen2Config)`:
  GQA-broadcasts K/V via `reshape → broadcast_to → copy →
  reshape` (zero-copy through the view; one materializing copy
  to land contiguous bytes for matmul), runs `Q @ K^T` via
  `permute([0,1,3,2]).copy()` + matmul, then reshapes scores
  from rank-4 `[B, H_q, T, T_c]` to rank-3 `[B*H_q, T, T_c]` for
  the scale-mul / mask-add / softmax stretch (branch-4 only wired
  binary `g{1,2,3}` strided paths — rank-4 strided binary lands
  alongside its first consumer), reshapes back, then `probs @
  V`. Validated at all three layers vs `mx.matmul` (kernels +
  runtime) and `mx.fast.scaled_dot_product_attention` (models)
  at Qwen2.5-0.5B decode (`T=1, T_cache=8`) and prefill-causal
  (`T=T_cache=8`) shapes. Tolerance bar is bf16-compose
  precision; the models test uses the np.allclose-style
  combined bound `|a-b| ≤ atol + rtol*|b|` rather than the
  pure-AND form earlier branches use, because the composed-SDPA
  chain produces 1-ULP-near-zero drift that pure relative
  comparison flags spuriously. **Deferred: the layer-0
  real-checkpoint integration test** the roadmap calls for
  here. It requires Q/K/V/O projections of the loaded quantized
  weights, which means `Tensor::quantized_matmul` (qmv batched +
  qmm) — that's branch 11b's surface. Doing it here via
  `dequantize → matmul` would be throwaway scaffolding for 11b
  to rewrite. So branch 11 lands SDPA's three-op composition
  with synthetic-input parity; the end-to-end real-checkpoint
  attention integration test lands in 11b once
  `quantized_matmul` is the supported runtime op.

- **Branch 11b (`feat/quantized-matmul`).** Wired `qmv` (decode,
  M=1) and `qmm` (prefill, M>1) through the runtime as a single
  `Tensor::quantized_matmul` op. Kernels layer: lifted MLX's
  `affine_qmm_t` (aligned, transposed, batch=1) into a new
  `kernels::qmm::affine_qmm_t_bf16` dispatch fn in
  `crates/cider-press-kernels/src/kernels/qmm.rs`; binding contract
  and dispatch geometry transcribed from
  `mlx/backend/metal/quantized.cpp::Quantized::eval_gpu` (transpose=true,
  batched=0 path). Reuses `Device::quantized_library()` — no new
  library slot, no new build.rs entry. Parity fixture is
  `tests/fixtures/qmm_qwen2.safetensors` (M=8, K=N=896, Qwen2.5
  projection shape), generated by `scripts/gen_qmm_fixture.py`
  with uniform `(-0.5, 0.5)` distribution (matching the stage4
  convention; normal-distributed inputs at K=896 push output
  magnitudes outside the bf16 tolerance bar for M>1). Runtime:
  `OpKind::Qmv` renamed to `OpKind::QuantizedMatMul`; the rank-1-only
  `QuantizedWeight::matvec` is removed and replaced by
  `Tensor::quantized_matmul(&self, weight: &QuantizedWeight) ->
  Result<Tensor>` accepting `[..., M, K]` dense bf16 activation and
  `[N, K]` weight, producing `[..., M, N]` lazily. Eval-side
  dispatcher picks `affine_qmv` for M=1 and `affine_qmm_t` otherwise
  — both flow through the same library cache with no new
  infrastructure. Aligned-only this branch (`N % 32 == 0`); generic
  `affine_qmm` deferred to its first unaligned consumer. The
  transpose=false direction (needed for the tied LM head, which reads
  `embed_tokens` in the opposite orientation) is deferred to branch
  12c — adding a `transpose` flag to `quantized_matmul` is the right
  surface but not needed until the LM head is wired. Three rejection
  unit tests: wrong activation dtype, inner-dim mismatch, and a
  happy-path constructor asserting `OpKind::QuantizedMatMul`, lazy
  state, and output shape. Models crate adds
  `qwen2::attention::project_qkv` and `qwen2::attention::project_output`
  in `crates/cider-press-models/src/qwen2/attention.rs`: thin
  compositions over `quantized_matmul + add (bias) + reshape + permute
  + copy` that produce per-head `[B, H, T, D_h]` tensors; branch 12b
  folds them into `Attention::forward`. New `qmm` and `attention_layer0`
  cases in `scripts/dump_mlx_op.py`. The layer-0 attention
  real-checkpoint integration test (`crates/cider-press-models/tests/attention_layer0.rs`,
  gated on `CIDER_QWEN_CHECKPOINT_PATH`) exercises the full
  Q/K/V-project → rope → KvCache → SDPA → O-project chain on
  Qwen2.5-0.5B layer 0: prefill (T=8) drives `affine_qmm_t`;
  decode (T=1, offset=7) drives `affine_qmv`; the MLX reference
  primes its `KVCache` to offset=7 via `mlx_lm.models.cache.KVCache.state`
  setter. Tolerance bars: kernels/runtime qmm parity at
  `max_relative=1e-2, epsilon=5e-2` (consistent with stage4_qmv
  precedent; absorbs 1–2 bf16 ULP cross-generation drift); layer-0
  integration test at ATOL=5e-2, RTOL=5e-2 using the np.allclose-style
  combined `|a-b| ≤ atol + rtol*|b|` bound to absorb the ~9-op
  composed chain.

- **Branch 12a (`feat/models-linear-rmsnorm`).** First stateful
  model-layer abstractions. Introduces the forward-only `Module`
  trait (`forward(&self, x: &Tensor) -> Result<Tensor>`) that this
  crate's layers implement, plus the first two: `Linear` (quantized
  projection `y = x @ W^T (+ bias)` wrapping `Tensor::quantized_matmul`
  — `affine_qmv` for M=1, `affine_qmm_t` for M>1 — with an optional
  rank-1 dense bias broadcast over leading dims; owns its
  `QuantizedWeight` so a loaded checkpoint can be moved into a module
  tree, and rejects a mismatched-length bias at construction) and
  `RmsNormLayer` (carries gamma + eps, wraps the composed
  `nn::rms_norm`). The trait is single-input by design — auxiliary
  state (attention's RoPE offset, the `KvCache`) flows through a
  layer's own constructor/methods, not `forward`. Both live in
  `nn.rs` alongside the existing free functions. Parity validated
  against the existing `qmm` and `rms_norm` MLX dump fixtures (no new
  Python harness surface): `Linear` with/without bias vs
  `mx.quantized_matmul` at Qwen2.5 `[896, 896]` decode+prefill,
  `RmsNormLayer` vs `mx.fast.rms_norm` at `[1, 8, 896]`. `Linear` is
  quantized-only for now (Qwen2 has no dense Linears).

- **Branch 12b (`feat/models-attention-mlp`).** `Attention` and `Mlp`
  as the next model-layer abstractions built on the branch-12a
  `Module`/`Linear` base. `Mlp` is a generic `Module` in `nn.rs` —
  owns three bias-free `Linear`s (`gate`, `up`, `down`) and computes
  `down(silu(gate(x)) * up(x))` (MLX's `swiglu(a,b) = silu(a)*b`
  factoring). Constructed via `Mlp::new(gate, up, down)` rather than
  a qwen2-bound `from_weights`, deliberately keeping `nn.rs`
  model-agnostic. `Attention` lives in `qwen2::attention` and owns
  four `Linear`s (q/k/v with dense bias, o without) plus a cloned
  `Qwen2Config`; inherent `forward(&self, hidden, mask: Option<&Tensor>,
  offset: &Tensor, cache: &mut KvCache)` folds the full pipeline —
  Q/K/V project → per-head reshape → RoPE → `KvCache::update` →
  `keys_view`/`values_view` → SDPA → O-project — and deliberately
  does NOT implement `Module` (mirrors MLX's `Attention.__call__(x,
  mask, cache)`; auxiliary state through its own method signature,
  per the 12a design note). The former free functions `project_qkv`
  and `project_output` are absorbed into `Attention::forward` and
  removed; `rope` and `sdpa` are kept as reusable free helpers.
  Corrected a latent KV-cache view transpose bug: `keys_view()`/
  `values_view()` return `[T_cache, H_kv, D_h]`; conversion to
  SDPA's `[1, H_kv, T_cache, D_h]` now uses `permute([1,0,2]).copy()`
  before the reshape rather than a bare reshape — the old
  `attention_layer0.rs` reshape transposed axes silently whenever
  `H_kv > 1` and `T_cache > 1`. Tests: synthetic
  `attention_mlp_smoke.rs` (shape/dtype/lazy, no checkpoint);
  `attention_layer0.rs` rewritten to drive `Attention::forward`;
  `projections_smoke.rs` removed; new gated `mlp_layer0.rs` +
  `mlp_layer0` case in `scripts/dump_mlx_op.py`. **The gated
  `attention_layer0` and `mlp_layer0` parity tests were not run
  against a real checkpoint during the implementing session — run
  locally with `CIDER_QWEN_CHECKPOINT_PATH` before merge (the
  KV-view transpose fix is only exercised numerically there).**

**Next concrete step: branch 12c of the roadmap — `feat/models-qwen2`.**
Build `TransformerBlock` (residual connections wrapping
`input_layernorm → Attention` and `post_attention_layernorm → Mlp`)
and `Qwen2Model` (embed → N blocks → final norm), plus the tied LM
head via a `transpose` flag on `Tensor::quantized_matmul` (deferred
from branch 11b). Per-token logits parity vs MLX-LM on a fixed prompt
prefill.

Quantized-embedding decision resolved in branch 7: neither
quantized-row gather nor `gather → dequantize_row`. Instead,
`QuantizedWeight::components()` exposes the packed triple as three
dense tensors so plain `gather` runs on each, then one fused
`affine_dequantize` consumes the gathered triple. Tied LM head
(same weight, transpose direction) lands in branch 12c via a
`transpose` flag on `Tensor::quantized_matmul` rather than
dequantizing the full vocab table.

Loader-naming footgun worth remembering: `q_proj.bias` (singular,
dense bf16, additive Linear bias) is **distinct** from
`q_proj.biases` (plural, per-group quantization bias on the qmv ABI).
Only Q/K/V have the singular `.bias`; O has only `.biases`.
`Qwen2AttentionWeights` separates the two via `q_bias` (Tensor) and
the `QuantizedWeight` field's internal scales/biases.

### Spike goal

Prove that we can load an MLX `.metal` kernel source verbatim, compile
it at runtime, dispatch it from Rust against shared-memory buffers, and
get bit-identical results to the reference MLX Python implementation
on the same inputs.

If this works, the rest of the project is "just" scaling the pattern.
If it doesn't, we learn what the real obstacles are before investing in
framework design.

**Estimated effort:** 1–2 focused weekends.
**Out of scope for the spike:** the lazy graph, multiple ops, fusion,
the public Rust API, error types, anything ergonomic. Walking skeleton
only — target < 500 lines of Rust.

### Reference paths

- **Vendored MLX kernels (canonical):** `kernels-mlx/mlx/backend/metal/kernels/`
- **MLX C++ dispatch reference (your local MLX checkout):**
  `$MLX_DIR/mlx/backend/metal/{device,kernels,matmul,quantized}.cpp`

The vendored copy is the source of truth at build time, so no MLX
checkout is needed for `cargo build`/`test`. A local MLX checkout is
only needed when bumping the vendored sources, or when reading MLX's
C++ side as reference while porting new dispatch routines.

To bump vendored kernels:

```sh
git clone https://github.com/ml-explore/mlx.git ~/src/mlx     # one-time
MLX_DIR=~/src/mlx ./scripts/sync_mlx_kernels.sh
# then update the upstream-commit SHA in kernels-mlx/VENDORED.md
cargo test --workspace --release    # Stage 4 parity is the canary
```

The `.github/workflows/mlx-sync.yml` workflow does this automatically
on a weekly schedule and opens a PR when upstream drifts.

### Staged plan

Each stage isolates one risk. Combining stages blinds you to which
thing broke — resist the urge to skip ahead.

#### Stage 0 — environment + dependencies (≈30 min) — **done**

Currently pinned in the workspace manifest's `[workspace.dependencies]`:

```toml
half = "=2.7.1"
objc2 = "=0.6.4"
objc2-foundation = "=0.3.2"
objc2-metal = "=0.3.2"
approx = "=0.5.1"
safetensors = "=0.7.0"
```

Member crates pull what they need via `dep.workspace = true`.

Notes from doing it:

- The `objc2-*` ecosystem has moved past the original suggestions
  (`0.5` → `0.6`, `0.2` → `0.3`); feature names shifted too.
- Default features on `objc2-metal 0.3.x` activate 60+ items and cover
  everything the spike needs without manual curation. Worth revisiting
  feature minimization once dispatch is proven, if binary size matters.
- `MTLCreateSystemDefaultDevice` requires `CoreGraphics` at link time.
  Linking it directly via `#[link(name = "CoreGraphics", kind = "framework")]`
  in the test/example file is lighter than pulling in `objc2-core-graphics`.

**Acceptance:** `cargo build` succeeds on aarch64-apple-darwin. ✓

#### Stage 1 — device + queue + trivial inline kernel (≈2 hours) — **done**

See `crates/cider-press-kernels/tests/stage1_add_one.rs`. Round-trip after warm-up measured at
~237 µs on the development machine; acceptance was < 5 ms.

Before touching MLX, prove the dispatch loop with a kernel typed inline:

```metal
#include <metal_stdlib>
using namespace metal;
kernel void add_one(
    device const float* x [[buffer(0)]],
    device float* y       [[buffer(1)]],
    uint i [[thread_position_in_grid]]
) { y[i] = x[i] + 1.0; }
```

Write one Rust function (test or example) that:

1. Gets `MTLCreateSystemDefaultDevice`.
2. Creates a command queue.
3. Compiles the source string via `newLibraryWithSource:options:error:`.
4. Looks up `add_one`, builds an `MTLComputePipelineState`.
5. Allocates two `MTLBuffer`s with `MTLResourceStorageModeShared`,
   length 1024 × 4 bytes.
6. Writes `0..1024` as f32 into buffer 0 via the host pointer
   (`contents()` — this is the unified-memory payoff; no
   `did_modify_range` on Apple Silicon).
7. Encodes the dispatch with threadgroup size 64, grid size 1024.
8. Commits, waits, reads buffer 1 via host pointer.
9. Asserts `out[i] == i as f32 + 1.0`.

**Risks isolated:** `objc2-metal` API shape, runtime shader compilation,
buffer storage modes, threadgroup math, command-buffer lifecycle,
ownership under `objc2`'s `Retained<T>`.

**Acceptance:** `cargo test --release` passes; round-trip < 5 ms.

#### Stage 2 — load a real MLX `.metal` file (≈2 hours) — **done**

See `crates/cider-press-kernels/build.rs` and `crates/cider-press-kernels/tests/stage2_copy.rs`. Both `v_copyfloat32float32`
and `v_copyfloat16float16` produce bit-identical output on 1024
elements. Findings:

- Offline-inlining approach worked. `build.rs` flattens `copy.metal`
  plus 6 transitive headers (1418 lines total) into one MSL string in
  `OUT_DIR`. Recursion handles `#pragma once` via a canonicalized-path
  seen-set; system `#include <...>` directives are left for the Metal
  compiler.
- Zero MSL dialect drift. The MLX `bf16` shim and the
  `metal_stdlib`/`metal_math`/`metal_logging` system headers coexist
  on the current macOS / Metal toolchain without modification. The
  `logging.h` `__METAL_VERSION__ >= 320` fallback never fires (system
  Metal is new enough).
- `instantiate_kernel` macro: produces `[[host_name(N)]] kernel ...`,
  so `newFunctionWithName:` lookup name is literally the first macro
  argument — no mangling. Big plus for Stage 4.
- `constant T&` MSL parameters auto-bind to the next free buffer slot
  after explicit `[[buffer(N)]]` attributes. Use
  `setBytes:length:atIndex:` for small scalars rather than allocating
  an `MTLBuffer`.
- JIT cost data: first run of `cargo test --test stage2_copy --release`
  takes ~1.5 s, dominated by `newLibraryWithSource:` for the full
  flattened source. Repeat runs of the same binary are sub-50 ms,
  hinting at driver-level library caching across processes (partially
  addresses open question 2; Stage 4 cold-run will confirm).

#### Stage 2 — original plan

Pick `mlx/backend/metal/kernels/copy.metal`. Small, minimal includes,
trivial reference output (identity), exercises the include resolver
without GEMM templating.

1. Embed `copy.metal` plus transitive headers. MLX uses
   `#include "mlx/backend/metal/kernels/utils.h"`-style paths; Metal's
   runtime compiler has no filesystem unless given one. Two viable
   approaches:
   - **Offline inline (recommended for spike):** a small `build.rs`
     recursively inlines `#include "..."` into a single self-contained
     `.metal` string, embedded as `&'static str`. Mirrors MLX's JIT path.
     No Xcode tooling dependency at build time.
   - **`.metallib` precompile:** `xcrun -sdk macosx metal` at build time.
     Lower runtime cost, more build-system work.
2. Compile via `newLibraryWithSource:`. Expect errors; iterate.
3. Dispatch the simplest copy variant on a 1D f32 input.

**Risks isolated:** include flattening; MSL dialect drift between MLX's
assumed compiler and the system Metal compiler; MLX macros
(`instantiate_kernel`, `bfloat`, etc.) needing definition or stubs.

**Acceptance:** copy kernel produces bit-identical output for f32 and
f16 (use `half::f16`).

#### Stage 3 — reference parity harness (≈1 hour) — **done**

See `scripts/gen_stage3_fixtures.py` (one-shot PEP 723 `uv run` script
— no requirements.txt, no venv), `crates/cider-press-kernels/tests/fixtures/stage3_copy.safetensors`
(12.5 KB, deterministic via seed=0), and `crates/cider-press-kernels/tests/stage3_parity.rs`. Both
f32 and f16 `v_copy` outputs match the MLX-generated reference at
1e-6 / 1e-3 tolerance respectively. Findings:

- `safetensors 0.7` is friction-free: `SafeTensors::deserialize(&bytes)`
  → `.tensor(name)?.data()` returns raw `&[u8]`, cast manually. No
  `bytemuck` needed for spike scope.
- Python is treated as a one-shot fixture generator, **not** project
  infrastructure. There is no long-term Python support intended; the
  script exists so Stage 4 can compare against real MLX output.
- Test harness duplicates `MetalCtx` + `run_copy` from Stage 2. Stage 4
  will be the third copy — factor to `tests/common/mod.rs` then, not
  before.
- MSRV trap: `usize::is_multiple_of` is Rust 1.87+ and our
  `rust-version = "1.85"` declares older. Clippy catches it. Use
  `% N == 0` or bump `rust-version`.

#### Stage 3 — original plan

```python
import mlx.core as mx
import numpy as np
np.random.seed(0)
# Save inputs + MLX output to .npy / .safetensors for Rust to load.
```

Rust test helper loads `.npy` (via `npyz`) or `.safetensors` (via
`safetensors`), runs the cider-press kernel, compares with
`approx::assert_relative_eq!` at dtype-appropriate tolerance: `1e-6` for
f32, `1e-3` for f16, `1e-2` for bf16/quantized.

**Acceptance:** the Stage 2 copy kernel passes the parity harness.

#### Stage 4 — target: one quantized matmul (≈1 weekend) — **done**

See `crates/cider-press-kernels/tests/stage4_qmv.rs`, `scripts/gen_stage4_fixtures.py`,
`crates/cider-press-kernels/tests/fixtures/stage4_qmv.safetensors` (150 KB; K=N=512). Output
matches MLX's `mx.quantized_matmul` **bit-exactly**
(`max_relative=0, max_absolute=0`) on every element — far beyond the
acceptance bar of ~1e-2 relative. Findings:

- Spike hypothesis fully validated for one kernel. MLX's MSL sources
  compile and dispatch verbatim from Rust with no porting.
- Kernel name token gotcha: the `instantiate_quantized_batched` macro
  stringifies `#type` from the C++ template arg, so for bf16 the name
  is `..._bfloat16_t_...` (with the `_t_`), **not** `..._bfloat16_...`
  as MLX's `Dtype` label suggests. The C++ dispatcher uses
  `get_type_string()` which returns the matching MSL spelling.
- Used the *fast* variant since K=512 satisfies `K % 512 == 0 &&
  N % bn == 0`. MLX picks `affine_qmv` (generic) for smaller K.
- Buffer/dispatch transcription from `mlx/backend/metal/quantized.cpp`
  is the spike's main risk surface. For batch=1, no shape/stride
  bindings follow — `add_strides_and_shapes` returns early. Order:
  w(uint32), scales(bf16), biases(bf16), x(bf16), y(bf16), K(int32),
  N(int32). Dispatch is `dispatch_threadgroups` (grid in tg count,
  not threads) with grid `(M, N/bn, B)` and threadgroup `(bk, 2, 1)`.
- **Open question 2 (cross-process JIT cache) confirmed:** cold first
  run of `newLibraryWithSource:` on the 5865-line flattened source
  took ~29 s (debug). Subsequent test-binary runs (even fresh
  processes) drop to sub-100 ms total. Metal definitely caches
  compiled libraries to disk persistently; we don't need to manage
  `.metallib` ourselves for spike purposes (production may still
  want explicit precompilation to amortize the cold-start cost).
- Three Stage tests now duplicate `MetalCtx`-style boilerplate.
  Did not factor — the Stage 4 binding shape is materially different
  (7 buffers, mixed setBuffer/setBytes) from Stages 2/3 (3 buffers).
  A shared helper would have to be generic enough to be useless.

#### Stage 4 — original plan

Pick **`qmv`** (quantized matrix-vector, batch=1) from
`mlx/backend/metal/kernels/quantized.{h,metal}`.

- `qmv` is the inference hot path — token generation is dominated by
  matvec against the quantized weight matrix.
- Simpler than `qmm`: no tiling across the batch dim, one
  warp/threadgroup per output row.
- Exercises the full quantized weight layout (packed int4 + per-group
  scales + biases), which is what we ultimately need.

Specifics:

1. **One template instantiation.** Target **bf16 × int4,
   group_size=64** — the dominant production config for Llama/Qwen
   4-bit quants. Instantiated name will be ~`qmv_bfloat16_gs64_b4`;
   grep `quantized.metal` for the exact `instantiate_kernel(qmv...)`
   lines to confirm.
2. **Reference inputs in Python.** Tiny MLX script:
   - Random `[N, K] = [4096, 4096]` bf16 weight,
   - `mx.quantize(w, group_size=64, bits=4)` → `(w_q, scales, biases)`,
   - `[K]` bf16 vector,
   - `y = mx.quantized_matmul(x, w_q, scales, biases, ...)`,
   - Save all five arrays as safetensors.
3. **Match MLX's weight memory layout exactly.** Most bugs will live
   here. Packed int4 layout, scale/bias interleaving,
   row-major-vs-column-major must match byte-for-byte. Read
   `mlx/backend/metal/quantized.cpp` for how MLX binds these buffers
   and copy the binding order verbatim.
4. **Dispatch parameters from MLX.** Search the matching call site in
   `quantized.cpp`. Copy threadgroup size, grid size, and buffer-index
   assignments exactly. Do not improvise.
5. **Run, compare.** Tolerance ≈ `1e-2` relative for bf16 × int4.
   Expect tighter agreement on most elements, with outliers near zero
   where relative error is unstable — fall back to absolute tolerance
   there.

**Risks isolated:** weight packing layout, scale/bias ordering, kernel
constants, row/column conventions.

**Acceptance:** `qmv_bfloat16_gs64_b4` matches MLX's `quantized_matmul`
to bf16 tolerance on `[4096, 4096] @ [4096]`.

#### Stage 5 — measure (≈1 hour) — **done**

See `crates/cider-press-kernels/tests/stage5_perf.rs` and `scripts/measure_stage5_mlx.py`.

Measured on the development machine (release build, warmup=50,
timed=1000):

|             | cider-press | MLX Python | ratio |
|-------------|------------:|-----------:|------:|
| K=N=512     | ~170 µs/d   | ~118 µs/d  | 1.44× |
| K=N=4096    | ~180 µs/d   | ~121 µs/d  | 1.49× |

Findings:

- Per-dispatch latency is nearly flat across a 64× shape jump on
  both sides — dispatch overhead, not compute, dominates at qmv
  scale. Expected; qmv is memory-bound and the matrices fit in cache.
- We are **~1.5× slower than MLX**. That's between the spike's
  "validated" (≤10%) and "investigate" (≥2×) thresholds. Kernels are
  bit-exact (Stage 4), so the gap is in the dispatch layer — which is
  exactly what the lazy-graph design is supposed to address.
- The naive Rust path does `commit + waitUntilCompleted` per
  dispatch. MLX presumably pipelines command buffers within
  `mx.eval`. Closing the gap is a problem for the dispatch layer
  design, not the kernel layer.

#### Stage 5 — original plan

- Time 1000 dispatches of the same `qmv` call on the same buffers.
- Report tok/s-equivalent: `1 / (time per dispatch)`.
- Compare against MLX Python on identical shapes (`mx.eval` + timer).

Within 10% of MLX → kernel-reuse hypothesis validated; perf gap is in
graph/dispatch as predicted.

2× slower or worse → something wrong with dispatch (threadgroup size,
function constants, buffer storage). Investigate before moving on.

**Acceptance:** numbers documented in `docs/SPIKE_RESULTS.md` (create
it). Do not skip — "produces correct output" is half the answer; the
spike exists to learn whether the approach works.

### Open questions to answer during the spike

Non-blocking, but record findings:

1. How does offline include-flattening handle headers including
   `<metal_stdlib>` vs MLX's own `bfloat` shim? Conflicts with system
   MSL on current macOS?
2. Does `newLibraryWithSource:` reliably cache compiled libraries across
   process restarts, or do we need to persist `.metallib` manually?
3. Measured wall-clock cost of `newLibraryWithSource:` for the full
   `quantized.metal`? (Determines JIT-at-startup viability vs
   precompilation.)
4. `bfloat`-related MSL version requirements that constrain minimum
   macOS?

### What NOT to do during the spike

- Don't design the public Rust API. Every type can be `pub(crate)` or a
  free function in a test/example file.
- Don't add a `Tensor` abstraction. Raw `MTLBuffer` + shape tuples are
  fine.
- Don't add error types beyond `Box<dyn Error>` or `anyhow`.
- Don't try to load a real model. Validate one kernel, not a forward
  pass.
- Don't port MLX C++ files. Read them for reference only.
- Don't optimize. Correctness + an honest perf number, not a tuned
  implementation.

### Findings (post-spike)

**TL;DR:** spike succeeds. Every stage's acceptance criterion passed.
The central hypothesis — that MLX's Metal kernels can be reused
verbatim and the perf gap to MLX lives in the dispatch layer — is
validated by `qmv` matching MLX bit-exactly while running ~1.5×
slower under naive synchronous per-dispatch. **Recommendation:
proceed to lazy-graph design.**

#### Acceptance criteria

| Stage | Bar | Outcome |
|-------|-----|---------|
| 0     | `cargo build` succeeds | ✓ |
| 1     | inline add_one round-trip < 5 ms | ✓ (~240 µs) |
| 2     | MLX copy.metal bit-identical f32+f16 | ✓ |
| 3     | parity harness passes for copy | ✓ |
| 4     | qmv within bf16 tolerance of MLX | ✓ — **bit-exact** (max_rel=max_abs=0) |
| 5     | throughput vs MLX documented | ✓ — 1.44–1.49× slower |

#### Surprises worth carrying forward

- **`objc2-metal 0.3.x` API drift.** The spike doc's listed versions
  (0.5 / 0.2) are stale; current is 0.6.4 / 0.3.2. Default features
  are heavy but cover everything; minimization is post-spike work.
  Documented in [[Stage 0]].
- **`MTLCreateSystemDefaultDevice` needs `CoreGraphics` linked.**
  Resolved via `#[link(name = "CoreGraphics", kind = "framework")]`
  in the test file; avoids dragging in `objc2-core-graphics`.
- **Metal does cache JIT'd libraries cross-process.** Cold first
  `newLibraryWithSource:` on the 5865-line flattened `quantized.metal`:
  ~29 s. Subsequent runs (even fresh processes): sub-100 ms total
  test time. We can defer `.metallib` management; for production,
  shipping a precompiled `.metallib` would just amortize the
  cold-start cost. Answers open question 2.
- **Zero MSL dialect drift.** MLX's `bfloat` shim, `metal_stdlib`,
  `metal_math`, `metal_logging` all coexist on the current macOS
  Metal toolchain. `logging.h`'s `__METAL_VERSION__ >= 320` fallback
  never fires. Answers open questions 1 and 4.
- **`instantiate_kernel` macro stringifies the C++ type token
  literally.** For bf16 the kernel name is `..._bfloat16_t_...` (with
  `_t_`), not `..._bfloat16_...` as the `Dtype` label suggests. Easy
  to get wrong; the parity safetensors is the canary.
- **MLX `qmv` binding/dispatch transcription was the spike's only real
  risk surface,** and it worked. For batch=1, no shape/stride
  bindings follow K and N — `add_strides_and_shapes` returns early.
  Order: w(uint32), scales(bf16), biases(bf16), x(bf16), y(bf16),
  K(int32), N(int32). Dispatch via `dispatch_threadgroups` (grid in
  threadgroup count, not threads) with grid `(M, N/bn, B)` and
  threadgroup `(bk, 2, 1)`, bn=8, bk=32.
- **Build/test ergonomics.** PEP 723 + `uv run` made Python a
  zero-friction one-shot tool with no requirements.txt or venv to
  maintain. `safetensors 0.7` is friction-free from Rust;
  `bytemuck` is unnecessary at spike scale (manual `from_le_bytes`
  on a `chunks_exact` view is enough).

#### Measured throughput vs MLX

See [[Stage 5]] table. ~1.5× slower across both small (512²) and
production (4096²) dims; dispatch overhead dominates compute.

#### Recommendation

**Proceed to lazy-graph + eager-eval design.** The kernels are
known-good and the perf gap lives where we predicted, which is also
where this project intends to add value. Specifically:

1. Design the lazy `Tensor` + graph + `eval()` semantics in a separate
   planning doc. Reference: MLX's `array` + lazy eval model.
2. Sketch the buffer pool / allocator (see
   `mlx/backend/metal/allocator.cpp`).
3. Pick the second kernel: `scaled_dot_product_attention` — next
   inference hot path, very different dispatch pattern (per-head
   tiling, KV cache layout). Validate the same way with a parity
   safetensors.
4. Set up cross-process command-buffer batching (multiple ops per
   `MTLCommandBuffer`, multiple buffers in flight via fences /
   completion handlers) — this is the lever that should close the
   1.5× gap.
5. Decide on `.metallib` precompilation strategy for shipping. JIT
   caching is good enough for the dev loop; cold-start matters for
   first-token latency in production.

### After the spike

If acceptance criteria pass:

1. Design the lazy-tensor graph + `eval()` semantics (separate planning
   doc).
2. Sketch the buffer pool / allocator. See
   `mlx/backend/metal/allocator.cpp` for reference design.
3. Next kernel: `scaled_dot_product_attention` — next inference hot
   path, very different dispatch pattern (per-head tiling, KV cache
   layout). Validate the same way.

If acceptance criteria fail:

1. Write up where it failed and what the obstacle was.
2. Reassess: keep going, switch to wrapping MLX via FFI (`mlx-c`), or
   contribute Metal kernels upstream to candle. All three are
   reasonable; the spike exists to inform the choice.

---

## Conventions

- **macOS / Apple Silicon only.** Each member crate's `src/lib.rs` has
  a `compile_error!` for non-macOS targets. Don't add cross-platform
  shims.
- **No premature abstractions.** Until the spike concludes, prefer
  `pub(crate)` free functions and raw `MTLBuffer` over types. The cost
  of building the wrong abstraction first is high.
- **MLX attribution.** Any file containing MLX-derived content (kernel
  source, embedded headers, port of a dispatch routine) must retain the
  MLX MIT copyright header at the top.
- **Comments.** Default to none. Write one only when the *why* is
  non-obvious (a hidden invariant, a workaround for a specific MSL
  quirk, etc.). Identifier names should carry the *what*.

## Dev workflow

- Build: `cargo build`
- Test: `cargo test` (release mode for any perf-sensitive test:
  `cargo test --release`)
- Lints: `cargo clippy --all-targets --all-features -- -D warnings`
- Format: `cargo fmt`
- Docs: `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps`

Pre-commit hooks run fmt, clippy, tests, and docs. Install once with
`pre-commit install` (requires the `pre-commit` tool).
