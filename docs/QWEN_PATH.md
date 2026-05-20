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
| 7 | `feat/gather` | `OpKind::Gather` for embedding lookup; vendor MLX's gather kernel. **Decision needed here:** tied embeddings are quantized, so either (a) implement quantized-row gather, or (b) gather → `dequantize_row` for the embedding path while qmv handles the LM head. Leaning (b) — embedding gather is once per token at decode and tiny. | MLX parity on random indices against real `W_embed` |
| 8 | `feat/kv-cache` | `KvCache` type (separate from Tensor); `update` (eager `Copy` to a slab); view accessors | Round-trip: write N rows, read back via view, compare |
| 9 | `feat/rope` | `OpKind::Rope` (Qwen RoPE config — base θ, dim split); could be a fused kernel or composed | MLX parity on `[B, T, H, D_h]` Q/K |
| 10 | `feat/softmax` | `OpKind::Softmax` (reduction along last axis, reuses branch-5's reduction primitive); needed by SDPA and the router (later MoE) | MLX parity |
| 11 | `feat/sdpa-split` | SDPA as **three ops**: `qk_matmul`, `softmax(scale + mask)`, `attn_matmul`. Composes the previous primitives. Defer fused SDPA. | MLX parity on a single attention block driven from real layer 0 weights |
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

Branch 7 (`feat/gather`). Specifically:

1. Vendor MLX's gather kernel (`mlx/backend/metal/kernels/gather.metal`
   plus any transitive headers) and add a `KernelLibrary::gather`
   slot. Embedding gather is the only consumer for now — bf16 indexed
   by `u32`.
2. Decide quantized embedding strategy. `model.embed_tokens` is
   quantized (4-bit, group 64), and tied embeddings mean the LM head
   reads the same weight. Two paths:
   - (a) Implement a quantized-row gather kernel (gather + dequantize
     per row, fused).
   - (b) `gather → dequantize_row` as separate ops, reusing the qmv
     dequant path.
   Leaning (b) — embedding gather is once per token at decode and
   tiny; (a) is a perf-only win we can revisit if measurement asks.
   Either way, the LM head keeps using qmv against the same packed
   weight.
3. Runtime: `OpKind::Gather` + `Tensor::gather(indices)`. Output
   shape is `indices.shape() ++ source.shape()[1:]` for axis 0.
4. Parity: gather + dequant on random `u32` indices against MLX's
   reference, run against the real `embed_tokens` weight loaded by
   `load_qwen2_weights`. The gated real-checkpoint test pattern from
   branches 4+5 carries forward.

Open question carried forward from branch 5 reduce-scope discussion:
whether the deferred kernel variants (`row_reduce_simple` for high
out-count cases, `all_reduce` for full-tensor collapse, `col_reduce`
for non-last reductions) should be wired alongside their first
consumer or in a dedicated reduction perf pass. Leaning the former
for `row_reduce_simple` (softmax on branch 10 wants it for prefill
perf) and the latter for the rest.
