# Path to Qwen2.5-0.5B inference

Target: **Qwen2.5-0.5B-Instruct** running interactively on Apple
Silicon via `cider-press`. Smallest published dense Qwen,
architecturally a Llama-family transformer with the Qwen-specific
RoPE/norm choices. Same shape as the eventual Qwen-MoE target
minus the router and expert dispatch, so every op we add carries
forward.

This doc is a branch-by-branch roadmap, not a spec. Each branch
should land as one focused PR. Cadence is side-quest — weeks
between sessions is fine.

## Reference points

- **MLX-LM** ships a verified Qwen2.5 port. It's our parity
  reference: dump activations from MLX at each layer, compare
  bit-or-bf16-tolerant against cider-press output.
- **HuggingFace `Qwen2.5-0.5B-Instruct`** is the weights source
  (~1 GB f16 / ~280 MB int4 via MLX quantization).
- **MLX's int4 quantization** matches our `qmv` already-validated
  layout (Q4, group_size=64). Reuse the path from Stage 4.

## Architecture summary

What Qwen2.5-0.5B is, mechanically (one forward pass):

```
tokens [B, T]
  → embed (gather)              [B, T, D]
  → for layer in 0..N:
        residual = x
        x = rmsnorm(x, ln1_gamma)
        q = quantized_matmul(x, W_q)
        k = quantized_matmul(x, W_k)
        v = quantized_matmul(x, W_v)
        reshape:  q → [B, T, H_q, D_h], k/v → [B, T, H_kv, D_h]
        apply RoPE to q, k
        transpose:  [B, T, H, D_h] → [B, H, T, D_h]
        kv_cache.update(k, v)
        cached_k, cached_v = kv_cache.views()    [B, H_kv, T_cache, D_h]
        # GQA: H_q heads attend over H_kv kv-heads via broadcasting
        attn = sdpa(q, cached_k, cached_v, mask, scale)
        attn = transpose + reshape back to [B, T, D]
        x = quantized_matmul(attn, W_o)
        x = residual + x

        residual = x
        x = rmsnorm(x, ln2_gamma)
        gate = silu(quantized_matmul(x, W_gate))
        up   =      quantized_matmul(x, W_up)
        x = quantized_matmul(gate * up, W_down)
        x = residual + x

  → x = rmsnorm(x, final_gamma)
  → logits = quantized_matmul(x, W_embed.T)     # tied embedding
  → sample → next token
```

Specifics for Qwen2.5-0.5B: `D=896`, `N=24 layers`, `H_q=14`,
`H_kv=2` (GQA ratio 7:1), `D_h=64`, `FFN_dim=4864`, `vocab≈151k`.
Subject to verification against `config.json` when we load it.

## Branch sequence

Each row is roughly one PR. Lengths are estimates; reality is
fuzzier and we'll inevitably collapse / split a few.

The big shift from the first draft of this doc: **weight loading moves to
branch 3, before any new ops land**. Loader needs nothing beyond the storage
primitives branch 1 already shipped (`from_bytes`, `quantized_from_bytes`),
and it costs nothing to validate (bytes-in-our-tensors == bytes-in-the-
safetensors). The payoff is that every later op branch validates against
real Qwen tensors — layer 0's actual `W_q`, layer 0's actual `gamma` — not
synthetic fixtures. Catches dtype/layout/shape-convention bugs at the op
that introduced them, instead of three branches later in some composition.

| # | Branch | Adds | Validated by |
|---|---|---|---|
| 1 | (done) `feat/runtime-data-primitives` | Tensor, Device, Layout, DType, OpKind, eval, Copy (all dtypes), Qmv (bf16/int4 gs=64) | Stage-4 fixture parity (already passing) |
| 2 | `feat/runtime-views` | `ViewSource` field on TensorInner; `Tensor::reshape` / `::transpose` / `::slice` / `::broadcast_to`; non-contiguous `cpu_*` accessors | Round-trip tests on reshape/transpose; broadcasting test against MLX shape rules |
| 3 | `feat/weight-loading` | safetensors integration; Qwen2.5 weight-key mapping; `config.json` parsing; HF revision SHA pinned. Includes the per-op MLX activation-dump harness (one `uv run` script that takes op name + shapes + dtypes and writes a parity safetensors). | Loaded tensor bytes == safetensors bytes (memcmp). Shape/dtype assertions match `config.json` predictions. No forward pass yet. |
| 4 | (done) `feat/elementwise-add` | First binary op family: vendored MLX's `binary.metal`, `OpKind::Binary { op: Add \| Mul }`, NumPy broadcasting via existing `broadcast_to`; `vv_*` fast path + `g{1,2,3}_*` rank-specialised strided paths for f32/f16/bf16 | MLX bit-exact parity at kernels, runtime, and models levels: synthetic `[1,8,896]` same-shape and broadcast `+[896]`; gated real-checkpoint test loads `layer0.input_layernorm` and broadcast-adds it bit-exactly vs a CPU bf16 reference |
| 5 | (done) `feat/rmsnorm` | Vendored MLX's `unary.metal` + `reduce.metal`. Kernels: `v_{square,rsqrt}_*` (6 fns) + `row_reduce_{sum,prod,min,max}_*` (12 fns, with `init_reduce` + `row_reduce_looped` bundled per op). Runtime: `OpKind::Unary` + `OpKind::Reduce`, `Tensor::{square, rsqrt, sum, prod, min, max, mean}`. Models: `nn::rms_norm(x, gamma, eps)` composed (no fused kernel). | MLX parity on `[1, 8, 896]` bf16 vs `mx.fast.rms_norm` (composed-path tolerance) plus gated real-checkpoint test against `layer0.input_layernorm` + CPU f32 reference |
| 6 | (done) `feat/silu-and-gelu` | Two activations composed from existing unary primitives. MLX's `unary.metal` doesn't ship `Silu`/`Gelu`; `mlx.nn` composes them on the Python side, so we mirror that factoring. Kernels: `v_{sigmoid,erf}_*` (6 fns; instantiations already present in the vendored `unary.metal`). Runtime: `UnaryOp::{Sigmoid, Erf}` + `Tensor::{sigmoid, erf}`. Models: `nn::silu(x) = x * sigmoid(x)`, exact `nn::gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))`. | SiLU bit-exact vs `mlx.nn.silu`. GELU bf16-compose tolerance vs `mlx.nn.gelu` (we reciprocal-multiply where MLX divides — runtime has no divide op yet — so the bf16-rounded constants differ in the last few bits). |
| 7 | (done) `feat/gather` | Embedding lookup end-to-end against Qwen2's quantized embed_tokens. First JIT-assembled kernel family — MLX has no precompiled `gather.metal`; source is generated per `(T, IdxT, NIDX, IDX_NDIM, LocT)` instantiation. Vendored `indexing/{indexing,gather}.h`; build.rs grows a `HEADER_BUNDLES` pass for header-only sources. Kernels: `gather::{Instantiation, make_source, dispatch}` (bf16+u32 instantiations) + `affine_dequantize_bf16_gs64_b4` (reuses existing `quantized.metal` library). Runtime: per-instantiation `Mutex<HashMap>` library cache on `Device`, `Tensor::gather` (BF16+U32 src), `OpKind::Dequantize` + `Tensor::dequantize_affine`, `QuantizedWeight::components()` exposing the packed triple as three dense Tensors sharing the underlying buffers. Models: `nn::embed_tokens` composes `gather × 3 → dequantize_affine`. The "decision" called out in the original entry resolved to "neither" — gather works on the components directly; dequantize is a single fused kernel over the gathered triple. | Bit-exact at every layer: gather is a pure data-mover, per-row dequantize is per-row identical between MLX and us. Real-checkpoint test exercises the full path on Qwen2.5-0.5B's loaded `embed_tokens` (6 token IDs across the vocab) vs CPU per-row dequantize from raw loaded bytes. |
| 8 | (done) `feat/kv-cache` | `KvCache` type (separate from `Tensor`) — first runtime-side abstraction that isn't a lazy-graph op. Pre-allocated `[max_tokens, n_kv_heads, head_dim]` slabs per K/V; `update(k, v)` is eager — materializes the source tensors then `memcpy`s their bytes into the slab via the unified-memory host pointer (Apple Silicon's shared storage makes this a host-side write, no Metal dispatch). `keys_view()`/`values_view()` return zero-copy `Tensor`s over the populated prefix via `Tensor::host_leaf` + a refcount-bumped `Buffer<u8>` clone. Aliasing contract documented: callers drop view tensors before the next `update`. Same-eval batching (folding the cache write into the SDPA command buffer) deferred to branch 11 if perf demands. | Roundtrip across multi-chunk `update`s reads back bit-exactly via `cpu_to_vec`; `reset()` rewinds; overflow / dtype / shape / device validation each rejected. No MLX parity needed (KvCache is a pure data-mover). |
| 9 | (done) `feat/rope` | MLX ships `rope.metal` precompiled — fused path, `OnceLock<KernelLibrary>` slot. First op needing Metal `[[function_constant]]` specialization, so kernels crate also gains `FunctionConstant::Bool { index, value }` + `KernelLibrary::pipeline_specialized(name, &[FunctionConstant])` with index-sorted cache key. One dispatch wired (`rope_bfloat16` with `forward=true, traditional=false, hs_transpose=false`, `with_freqs=false`, int32 indexing); seven other instantiations dormant. Runtime: `OpKind::Rope { base_log2, scale, rotary_dims }` with `(input, offset)` graph edges (offset is a length-1 I32 tensor, mirroring MLX's `set_input_array(offset, 2)`); `Tensor::rope(&self, offset, base, scale, rotary_dims)`. Drops `Eq` derive on `OpKind` (f32 params); `PartialEq` still gives test ergonomics. Models: `qwen2::attention::rope(x, offset, &Qwen2Config)` config-binding wrapper in a new `qwen2::attention` submodule that future attention bits accumulate into. | Bit-exact vs `mx.fast.rope` at every layer for `[1, 14, 4, 64]` (Q) and `[1, 2, 4, 64]` (K), `offset=0` and `offset=37`. No real-checkpoint case — rope applies to projection outputs; the full layer-0 Q-projection → rope → cache → SDPA flow is a branch-11 integration test. |
| 10 | (done) `feat/softmax` | MLX ships `softmax.metal` precompiled — fused path, `OnceLock<KernelLibrary>` slot. Surprise vs the branch-9 note: softmax does NOT use `[[function_constant]]` specialization (variants are by kernel name — `block_softmax_<dtype>` ≤ 4096-axis, `looped_softmax_<dtype>` > 4096-axis, `_precise_` mirrors using `float` accumulator). So `FunctionConstant` stays `Bool`-only. Two entry points wired: `block_softmax_bfloat16` (`precise=False`) and `block_softmax_precise_bfloat16` (`precise=True`, the inference choice). Looped + f32/f16 dtypes dormant; non-last-axis hard-errors. Threadgroup math ports MLX's `Softmax::eval_gpu` verbatim (`SIMD_SIZE=32`, `N_READS=4`, `BLOCK_AXIS_LIMIT=4096`). `Tensor::softmax(&self, precise: bool)`; `OpKind::Softmax { precise }`. | Tolerance bars (not bit-exact) at Qwen2 attention-score shape `[1, 14, T, T]` for T=4/32/256, both `precise=True` (~0.005 abs / 0.01 rel) and `precise=False` (~0.01 abs / 0.04 rel). Float accumulator buys ~2× tighter agreement by absorbing lane-summation drift before the final bf16 cast. |
| 11 | (done) `feat/sdpa-split` | SDPA as three composed ops. **First non-MLX kernel in the tree**: a naive bf16 batched matmul (`kernels/matmul.metal`, one thread per output element, float accumulator) instead of MLX's templated `steel_gemm_*` family (deferred to branch 15 per the "perf-driven steel migration" call). Kernels: `kernels::matmul::dispatch_gemm_bf16`. Runtime: `OpKind::MatMul` + `Tensor::matmul`; the matmul-shape validation lives in a `matmul_shapes()` helper that 11b's `qmm` will reuse. `dispatch_matmul` walks reshape-view chains (SDPA produces `probs_3d.reshape([B, H_q, T, T_c])` view inputs) via a new `matmul_input_bytes` helper; non-zero view byte-offsets hard-error. Models: `qwen2::attention::sdpa(q, k, v, mask, &Qwen2Config)` with GQA broadcast (`reshape → broadcast_to → copy → reshape`). Scores reshape from rank-4 to rank-3 for the scale-mul / mask-add / softmax stretch — branch 4 only wired `g{1,2,3}` strided binaries, so rank-4 broadcast binary stays deferred. The **layer-0 real-checkpoint integration test the roadmap originally placed here is moved to 11b** — it requires Q/K/V/O projections of loaded quantized weights, which needs `Tensor::quantized_matmul`. | Synthetic-input parity at all three layers vs `mx.matmul` (kernels + runtime) and `mx.fast.scaled_dot_product_attention` (models) at Qwen2.5-0.5B decode (`T=1, T_cache=8`) and prefill-causal (`T=T_cache=8`) shapes; bf16-compose tolerance using the np.allclose-style combined `atol + rtol*|b|` bound. |
| 11b | `feat/quantized-matmul` | `qmv` (decode, batch=1) + `qmm` (prefill) wired through the runtime as `Tensor::quantized_matmul` (the Stage-4 spike already validated `qmv` bit-exactly at the kernels layer). Needed before the tied LM head in branch 12c — the LM head reads the same quantized weight as `embed_tokens` but in the transpose direction, so dequantizing the full vocab table per step is the wrong move. | MLX parity on Qwen2's Q/K/V/O projection shapes + lm_head shape at bf16 tolerance |
| 12a | `feat/models-linear-rmsnorm` | `cider-press-models` scaffold; `Linear` and `RmsNormLayer` (thin wrappers around qmv / composed rmsnorm). Module trait/pattern lands here. | Unit tests against MLX equivalents |
| 12b | `feat/models-attention-mlp` | `Attention` (Q/K/V proj + RoPE + KV cache + SDPA + O proj) and `Mlp` (SwiGLU) | Per-module parity against MLX layer 0 |
| 12c | `feat/models-qwen2` | `TransformerBlock` + `Qwen2Model`; tied embedding handling; final norm + LM head | Logits parity vs MLX-LM on prefill of a fixed prompt |
| 13 | `feat/tokenizer` | BPE tokenizer (probably `tokenizers` crate from HF for Qwen's BPE vocab) | Round-trip encode/decode against HF tokenizer |
| 14 | `feat/cli + greedy decode` | CLI: load model, take prompt, prefill, decode loop, greedy argmax sampling, streamed detokenize | Generate a coherent completion |
| 15 | `feat/perf-measurement` | Tokens/sec bench against MLX-LM; memory profile; identify hot spots | Numbers documented in `docs/QWEN_PERF.md` |

## Things deferred past first-token

These don't block "Qwen2.5 generates coherent text"; they matter
for "it's fast and clean" but can wait until branch 15+:

- **Qmm (batched quantized matmul)**: needed for fast *prefill*.
  Without it, prefill happens one token at a time — correct but
  slow. Decode (the perf-dominant path) only needs Qmv. Land Qmm
  after the basics work.
- **Buffer pool / allocator**: per-`eval()` allocation isn't free
  but isn't catastrophic at our scale. Land when measurements
  point at it.
- **Cross-eval command-buffer batching**: the 1.5× spike gap.
  Closes after Qmm and pool give us cleaner numbers.
- **Top-k / top-p sampling**: greedy first; better sampling later.
- **Strided-aware kernels** (MLX's `g_copy` family): we'll
  materialize strided views via `Copy` to begin with. Strided
  kernels are a perf-only improvement.
- **`.metallib` precompilation**: matters for first-token latency
  in shipped binaries, not for dev-loop iteration.

## What to do next time we sit down

Branch 11b (`feat/quantized-matmul`). Specifically:

1. Promote `qmv` (the Stage-4 spike's already-bit-exact bf16 × int4
   path) into a runtime op. The current `QuantizedWeight::matvec`
   is rank-1-input-only (`[K] → [N]`); a real flow needs batched
   matvec for `T > 1` prefill at least. Decide whether to keep the
   `QuantizedWeight::matvec` ergonomic and add a separate
   `Tensor::quantized_matmul`, or replace matvec with the general
   form. Reuse the `matmul_shapes()` helper that branch 11
   factored out.
2. Add `qmm` (batched / prefill case). MLX has multiple `qmm_*`
   instantiations in `quantized.metal`; pick the bf16 × int4
   `group_size=64` family that matches Qwen2.5's pinned quant. The
   binding contract is more complex than `qmv` (shapes + strides
   bind when batch > 1). Read `mlx/backend/metal/quantized.cpp`'s
   `Quantized::eval_gpu` for the dispatch transcript.
3. **Layer-0 attention real-checkpoint integration test** (deferred
   from branch 11). Load layer-0 Q/K/V/O of Qwen2.5-0.5B (still
   gated on `CIDER_QWEN_CHECKPOINT_PATH`), project the hidden state
   via `Tensor::quantized_matmul`, apply rope to Q/K, push K/V into
   `KvCache`, run `qwen2::attention::sdpa`, compare against MLX's
   `mx.fast.scaled_dot_product_attention` over the same inputs.
   Tolerance: bf16-compose using the same combined `atol + rtol*|b|`
   bound branch 11's `sdpa_qwen2.rs` introduced.
4. Tied LM head dispatch decision: the LM head reuses the
   `embed_tokens` quantized weight but in the transpose direction
   (Qwen2's `tie_word_embeddings = true`). Decide whether to expose
   that via a `transpose` flag on `Tensor::quantized_matmul` (less
   API surface, matches MLX's `mx.quantized_matmul(transpose=True)`)
   or via a separate path. Defers a bit to branch 12c.

Branches still ahead of first end-to-end inference (br14): 11b
(quantized matmul), 12a/b/c (models layers + Qwen2 assembly), 13
(tokenizer), 14 (CLI + greedy decode). 5 branches between us and
the first runnable Qwen2.5-0.5B.

Open question carried forward from branch 7: the per-instantiation
JIT cache on `Device` now holds one library per unique kernel name.
For softmax / RoPE / SDPA the instantiation space is larger (more
template parameters), but the same `Mutex<HashMap>` pattern carries
forward. Worth noting that the cross-process Metal cache also
amortizes per-instantiation, so the wall-clock cost only matters on
first-ever invocation per instantiation per system.

Open question carried forward from branch 5 reduce-scope discussion:
whether the deferred kernel variants (`row_reduce_simple` for high
out-count cases, `all_reduce` for full-tensor collapse, `col_reduce`
for non-last reductions) should be wired alongside their first
consumer or in a dedicated reduction perf pass. Leaning the former
for `row_reduce_simple` (softmax on branch 10 wants it for prefill
perf) and the latter for the rest.
