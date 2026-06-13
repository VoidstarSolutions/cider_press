# Qwen2.5-0.5B-Instruct (4-bit)

This is the model `cider-press` runs end-to-end today, and the worked
example threaded through the rest of this guide. It earns that spot by
being the smallest *published* dense Qwen2.5 checkpoint: 0.5B parameters,
24 layers, small enough that a full forward pass fits comfortably in
unified memory and fast enough that the prefill/decode split is easy to
measure. The pinned weights are the [mlx-community 4-bit
checkpoint](https://huggingface.co/mlx-community/Qwen2.5-0.5B-Instruct-4bit),
and `mlx_lm`'s Qwen2.5 port is the parity reference — every kernel and
composed op is checked bit-for-bit (or within a bf16-ULP tolerance) against
what MLX produces for the same input.

It is also a deliberate stepping stone. Qwen2.5-0.5B is a Llama-family
decoder-only transformer (RMSNorm + RoPE + SwiGLU — the Llama recipe) with
Qwen's particular norm/RoPE choices, and it
shares its whole op vocabulary with the eventual mixture-of-experts target
— the dense SwiGLU feed-forward here is exactly the per-expert MLP there,
minus the router and the expert gather. So every operation worked out on
this model (quantized matmul, GQA attention, RoPE, RMSNorm, the tied head)
carries forward unchanged; the MoE work becomes "add a router and a gather
in front of the FFN," not "re-derive the stack."

For the generic pipeline these stages compose into, read the
[spine](../README.md) first; this page only narrates what is *specific* to
Qwen2.5.

## Architecture at a glance

The shapes below are the pinned `config.json` for Qwen2.5-0.5B-Instruct,
deserialized into [`Qwen2Config`](../../../crates/cider-press-models/src/qwen2/config.rs).

| Field | Value | `config.json` key |
| --- | --- | --- |
| Hidden / residual dim `D` | 896 | `hidden_size` |
| Layers `N` | 24 | `num_hidden_layers` |
| Query heads `H_q` | 14 | `num_attention_heads` |
| KV heads `H_kv` | 2 | `num_key_value_heads` |
| GQA ratio | 7:1 (`H_q / H_kv`) | — |
| Head dim `D_h` | 64 (`D / H_q`) | — |
| FFN intermediate | 4864 | `intermediate_size` |
| Vocabulary | ≈151k (151936) | `vocab_size` |
| RoPE base `θ` | 1,000,000 | `rope_theta` |
| RMSNorm `ε` | 1e-6 | `rms_norm_eps` |
| Tied LM head | yes | `tie_word_embeddings` |
| Quantization | 4-bit affine, `group_size=64` | `quantization` |

A few of these are worth pinning in your head before reading on: the
residual stream is only 896 wide, GQA collapses 14 query heads onto 2 KV
heads (a 7:1 broadcast), and the embedding table is *tied* — the same
quantized `[vocab, D]` weight serves both the input lookup and the output
projection. The whole network — including the embedding/LM-head weight — is
stored 4-bit affine-quantized in groups of 64 elements, each group carrying
its own scale and bias.

## The forward pass

One forward pass, Qwen2.5-specific (adapted from `ARCHITECTURE.md`). Each
named operation links to its step doc:

```text
tokens [1, T]
  → embed (gather quantized embed_tokens → dequantize)        [1, T, 896]
  → for layer in 0..24:
        residual = x
        x = rmsnorm(x, ln1_gamma)                             # pre-attention norm
        q,k,v = quantized_matmul(x, W_{q,k,v}) + bias          # dense .bias add
        reshape: q → [1, 14, T, 64],  k,v → [1, 2, T, 64]
        apply RoPE to q, k                                    # θ = 1e6
        kv_cache.update(k, v); read cached_k, cached_v
        attn = sdpa(q, cached_k, cached_v, mask, scale)        # GQA 7:1 broadcast
        x = residual + quantized_matmul(attn, W_o)             # W_o: no bias
        residual = x
        x = rmsnorm(x, ln2_gamma)                             # pre-FFN norm
        x = residual + quantized_matmul(silu(gate(x)) * up(x), W_down)
  → x = rmsnorm(x, final_gamma)
  → logits = quantized_matmul(x, embed_tokens, transpose=true) # tied head: same [vocab,D] weight, read transposed
  → sample → next token
```

Links: [embedding](../embedding.md), [rmsnorm](../rmsnorm.md),
[quantized matmul](../quantized-matmul.md), [RoPE](../rope.md),
[KV-cache](../kv-cache.md), [attention](../attention.md),
[feed-forward](../feed-forward.md), [sampling](../sampling.md).

What is Qwen2.5-specific in that block, top to bottom: the embedding is
gathered from the *same* quantized table the LM head reuses; the norms are
**pre-norm** (applied to the block input, not the output) with a learned
`gamma` and no bias; only the Q/K/V projections carry an additive dense
bias (the output and FFN projections do not); RoPE uses a single large base
`θ = 1e6` over all of `D_h = 64`; attention is GQA with each of the 2 KV
heads broadcast across 7 query heads; and the head is the embedding table
transposed. The composition is run in
[`Qwen2Model`](../../../crates/cider-press-models/src/qwen2/model.rs)
(`embed → 24 blocks → final norm → tied head`), one
[`TransformerBlock`](../../../crates/cider-press-models/src/qwen2/block.rs)
per layer. In prefill the blocks run over all `T` prompt tokens but the
final norm + head run only on the last position (`forward_last`); in decode
they run on a single token (`T = 1`) — see the
[execution model](../execution-model.md).

## Where Qwen2.5 picks a variant

Each of these is a knob the transformer family leaves open; the bullet
names Qwen2.5's choice and what else exists. The detail lives in the
"Model variations" notes on the linked step doc.

- **GQA, 7:1** — 14 query heads share 2 KV heads, so the cache holds 2
  heads' worth of K/V instead of 14 (7× less cache traffic and memory).
  The alternatives are full multi-head attention (`H_kv = H_q`) and the
  multi-query extreme (`H_kv = 1`); Qwen2.5 sits between them. See
  [attention](../attention.md).
- **Qwen RoPE, single `θ`** — one rotary base `rope_theta = 1e6` applied
  across the whole head dim, no NTK/YaRN scaling and no partial-rotary
  split. Other families ship per-segment scaling or rotate only a fraction
  of `D_h`; Qwen2.5-0.5B does neither. See [RoPE](../rope.md).
- **Pre-norm RMSNorm** — each sub-layer normalizes its *input*
  (`x → sublayer(rmsnorm(x))`), using RMSNorm (no mean-subtraction, no
  bias) rather than LayerNorm, with `ε = 1e-6`. Post-norm and LayerNorm are
  the historical alternatives. See [rmsnorm](../rmsnorm.md).
- **Dense SwiGLU FFN** — a gated MLP, `down(silu(gate(x)) * up(x))`, with
  `intermediate_size = 4864`, run densely on every token. This is exactly
  the per-expert MLP a future MoE would gate behind a router; here there is
  no router. See [feed-forward](../feed-forward.md).
- **Two kinds of "bias" on the projections** — the loader footgun: Q, K,
  and V each have a `.bias` (singular: a dense bf16 *additive* Linear bias,
  added after the matmul) **and** the quant `.biases` (plural: the
  per-group zero-point on the 4-bit qmv ABI, consumed *inside*
  dequantization). They are unrelated despite the near-identical name. The
  output projection has only `.biases` (no additive bias). Mixing them up
  silently corrupts the projection. See
  [weights and quantization](../weights-and-quantization.md).
- **Tied LM head** — `tie_word_embeddings = true`, so there is no separate
  output weight: the head re-projects through the (quantized) embedding
  table with `quantized_matmul(.., transpose=true)`. Untied models carry a
  distinct `lm_head` weight; Qwen2.5-0.5B saves it. See
  [sampling](../sampling.md).
