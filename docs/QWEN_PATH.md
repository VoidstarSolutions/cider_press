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

| # | Branch | Adds | Validated by |
|---|---|---|---|
| 1 | (closing now) `feat/runtime-data-primitives` | Tensor, Device, Layout, DType, OpKind, eval, Copy (all dtypes), Qmv (bf16/int4 gs=64) | Stage-4 fixture parity (already passing) |
| 2 | `feat/runtime-views` | `ViewSource` field on TensorInner; `Tensor::reshape` / `::transpose` / `::slice` / `::broadcast_to`; non-contiguous `cpu_*` accessors | Round-trip tests on reshape/transpose; broadcasting test against MLX shape rules |
| 3 | `feat/elementwise-add` | First binary op family: vendor MLX's `binary*.metal`, `OpKind::Add`, broadcasting support; while we're here, `OpKind::Mul` | MLX parity on `[B,T,D] + [D]` (broadcasting), `[X] * [X]` (elementwise) |
| 4 | `feat/rmsnorm` | `OpKind::RmsNorm` (kernel, not composed — perf-critical); vendor `rms_norm.metal` (or compose unary+reduction first, optimize later) | MLX parity on `[B, T, D]` with `[D]` gamma |
| 5 | `feat/silu-and-gelu` | `OpKind::SiLU` (Qwen uses SwiGLU = silu(gate) * up); maybe GELU for free since same `unary*.metal` | MLX parity per dtype |
| 6 | `feat/gather` | `OpKind::Gather` for embedding lookup; vendor MLX's gather kernel | MLX parity on random indices |
| 7 | `feat/kv-cache` | `KvCache` type (separate from Tensor); `update` (eager `Copy` to a slab); view accessors | Round-trip: write N rows, read back via view, compare |
| 8 | `feat/rope` | `OpKind::Rope` (Qwen RoPE config — base θ, dim split); could be a fused kernel or composed | MLX parity on `[B, T, H, D_h]` Q/K |
| 9 | `feat/softmax` | `OpKind::Softmax` (reduction along last axis); needed by SDPA and the router (later MoE) | MLX parity |
| 10 | `feat/sdpa-split` | SDPA as **three ops**: `qk_matmul`, `softmax(scale + mask)`, `attn_matmul`. Composes the previous primitives. Defer fused SDPA. | MLX parity on a single attention block |
| 11 | `feat/models-scaffold + qwen2` | Bring `cider-press-models` to life: layer modules (`Linear`, `RmsNormLayer`, `Attention`, `Mlp`, `TransformerBlock`, `Qwen2Model`); module pattern | Unit tests per module against MLX equivalents |
| 12 | `feat/weight-loading` | safetensors integration; Qwen2.5 weight-key mapping; config.json parsing | Load a checkpoint, run one forward pass, match MLX logits |
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

Branch 2 (`feat/runtime-views`). Specifically:

1. Read `cider_press_kernels::kernels::copy` and the MLX strided
   `g*_copy` macros to understand what strided dispatch looks like.
2. Decide on the `ViewSource` invariants (the design doc sketches
   them; lock them in with code).
3. Land `Tensor::reshape` (the trivial case — same byte layout,
   just relabels shape), then `transpose` and `slice` and
   `broadcast_to`.
4. Tests against MLX's broadcasting rules and the obvious
   reshape/transpose roundtrips.

The branch is self-contained: no new ops, no model code, no
weight loading. Just the view machinery.
