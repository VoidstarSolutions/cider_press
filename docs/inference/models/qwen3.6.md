# Qwen3.6-27B (dense, Gated-DeltaNet hybrid)

This is the **next** model `cider-press` is being built toward — the eventual
target of the work tracked in [`ROADMAP.md`](../../../ROADMAP.md). It is a dense
27B that posts flagship agentic-coding scores (it beats the prior open flagship
Qwen3.5-397B-A17B on SWE-bench Verified), runs in **~15 GB** at 4-bit, and is the
best local coder that fits a 64 GB Mac. Crucially, its linear-attention mechanism
(**Gated DeltaNet**) is implemented in the incumbent (`mlx-lm`) as a *recurrent,
non-chunkwise* Metal kernel — so unlike Qwen2.5 (where we matched mature kernels),
there is real perf headroom here, concentrated in long-context prefill.

The development vehicle is **Qwen3.5-4B** — the smallest checkpoint sharing this
exact architecture (`model_type: qwen3_5`), 1/10th the download, structurally
identical to the 27B up to scale and two parameterized knobs. Everything worked
out against the 4B transfers to the 27B unchanged. The pinned references are the
[mlx-community 4-bit](https://huggingface.co/mlx-community/Qwen3.5-4B-4bit) and
[27B 4-bit](https://huggingface.co/mlx-community/Qwen3.6-27B-4bit) checkpoints;
`mlx-lm`'s `qwen3_5` port is the parity reference, validated bit-for-bit (bf16)
against fixtures from `scripts/dump_qwen35_fixtures.py`.

> Both checkpoints are `Qwen3_5ForConditionalGeneration` (vision-language). For
> text generation we **drop the vision tower entirely** (≈300 of the ~1200/2180
> tensors). `mlx-lm` loads and runs them text-only; `mlx-vlm` is not needed.

## Architecture at a glance

| Field | 4B (dev) | 27B (target) | key |
| --- | --- | --- | --- |
| Hidden `D` | 2560 | 5120 | `hidden_size` |
| Layers `N` | 32 | 64 | `num_hidden_layers` |
| Attn query / KV heads | 16 / 4 | 24 / 4 | `num_attention_heads` / `num_key_value_heads` |
| Attn head dim `D_h` | 256 | 256 | `head_dim` (note: **not** `D / H_q`) |
| Linear value / key heads | 32 / 16 | **48** / 16 | `linear_num_value_heads` / `linear_num_key_heads` |
| Linear key / value head dim | 128 / 128 | 128 / 128 | `linear_{key,value}_head_dim` |
| Conv kernel | 4 | 4 | `linear_conv_kernel_dim` |
| FFN intermediate | 9216 | 17408 | `intermediate_size` |
| Vocabulary | 248320 | 248320 | `vocab_size` |
| RoPE base `θ` / partial factor | 1e7 / 0.25 | 1e7 / 0.25 | `rope_theta` / `partial_rotary_factor` |
| mRoPE sections | [11,11,10] | [11,11,10] | `rope_parameters.mrope_section` |
| Full-attention interval | 4 (3 linear : 1 full) | 4 | `full_attention_interval` |
| Attention output gate | yes | yes | `attn_output_gate` |
| Tied LM head | **yes** | **no** (own `lm_head`) | `tie_word_embeddings` |
| SSM state dtype | float32 | float32 | `mamba_ssm_dtype` |
| Quantization | 4-bit affine, `group_size=64` | same | `quantization` |

The only 27B-specific loader work is the **untied `lm_head`** and the wider
**`value_dim = 48 × 128 = 6144`** (GQA ratio 3 instead of 2); both are
config-parameterized. Reference modeling code: `mlx_lm/models/qwen3_5.py` (the
dense module — reuses `Qwen3NextAttention`/`MLP`/`RMSNormGated` from
`qwen3_next.py`), `mlx_lm/models/gated_delta.py` (the recurrence + kernel).

## The hybrid stack

Each decoder layer is pre-norm `RMSNorm → mixer → +residual → RMSNorm → SwiGLU MLP
→ +residual`, identical in shape to Qwen2.5 *except* the mixer alternates by
position: with `full_attention_interval = 4`, layers `3, 7, 11, …` are **Gated
Attention** and the other three-in-four are **Gated DeltaNet** (linear). The two
mixers maintain different caches (a KV cache for full layers; a conv-window +
recurrent-state cache for linear layers).

## Gated DeltaNet (the linear mixer)

Per the reference (`qwen3_5.py:GatedDeltaNet`), on input `x` (post input-norm):

1. **Project** (four *separate* quantized projections — the checkpoint splits the
   reference's fused `in_proj_qkvz`/`in_proj_ba`):
   - `in_proj_qkv` → `q‖k‖v` (key_dim ‖ key_dim ‖ value_dim)
   - `in_proj_z` → `z` (the output gate), `in_proj_a`/`in_proj_b` → `a`, `b`
     (one scalar per value-head each)
2. **Depthwise causal Conv1d** (kernel 4, groups = channels) over `q‖k‖v`, then
   SiLU. Left-padded by `conv_kernel_size − 1 = 3` cached tokens (zeros at prefill
   start). *We compose this from 4 shifted scaled adds — no conv kernel to vendor.*
3. **Weightless normalize + scale** the post-conv q, k: `q = (1/d)·rmsnorm(q)`,
   `k = (1/√d)·rmsnorm(k)`, `d = 128`, no learned gamma.
4. **Recurrence** (`gated_delta_update`), per value-head, over a `[Dv, Dk]`
   fast-weight matrix `M` (128×128), for each timestep `t`:
   - decay: `M ← g · M`, where `g = exp(−exp(A_log) · softplus(a + dt_bias))`
     (Mamba2 forget gate; `A_log` upcast to fp32 in `compute_g`)
   - read: `kv = M · k` → `[Dv]`
   - delta rule: `δ = β · (v − kv)`, `β = sigmoid(b)` (write strength)
   - write: `M ← M + δ ⊗ k`
   - read out: `y = M · q` → `[Dv]`
5. **Gated output norm** `out = rmsnorm(out, norm.weight) · silu(z)` (in fp32),
   then `out_proj` back to `D`.

GQA: with `Hv = 32/48` value heads and `Hk = 16` key heads, q/k are repeated
`Hv/Hk` (2 or 3) inside the recurrence. The recurrent **state is fp32**, shaped
`[B, Hv, Dv, Dk]`; the conv state is `[B, 3, conv_dim]`.

## Gated Attention (the full mixer)

Standard GQA attention (`Qwen3NextAttention`) with three Qwen3 twists:

- **Output gate:** `q_proj` emits `2 × (H_q · D_h)`. The split is **per-head**,
  not flat: reshape to `[B, L, H_q, 2 · D_h]` then split the last axis into
  `queries ‖ gate` (so the layout interleaves `[head0: q×256, gate×256, head1: …]`
  — *not* a contiguous query block followed by a contiguous gate block). The
  attention output is multiplied by `sigmoid(gate)` before `o_proj`.
- **QK-norm:** per-head `RMSNorm(head_dim=256)` on queries and keys (`q_norm`,
  `k_norm`) before RoPE.
- **Partial rotary:** RoPE rotates only `head_dim × 0.25 = 64` of the 256 dims;
  the rest pass through. `θ = 1e7`. (mRoPE collapses to 1-D positions for text.)

`scale = 256^-0.5`, causal mask, no attention bias.

## Parity contract (load + compute gotchas)

These silently break parity if missed:

| Item | Contract |
| --- | --- |
| RMSNorm `+1` | The reference adds `1.0` to all standard norm gammas (`input/post_attention/final`, `q_norm`, `k_norm`) — **but only for unsanitized checkpoints**. The pre-quant mlx-community checkpoints are already sanitized, so **load gammas as-is**. The GDN gated `norm` is *not* in this set. Fixtures are the arbiter. |
| `A_log` dtype | Stored f32 (4B) or bf16 (27B); always **upcast to fp32** in `compute_g`. |
| State dtype | SSM recurrent state and the `g`/recurrence math are **fp32** (`mamba_ssm_dtype`). bf16 state corrupts output. |
| Embedding | plain lookup, **no √D scaling**; LM head tied (4B) / separate (27B). |
| q/k norm | **weightless** rmsnorm + `(1/d)` and `(1/√d)` scaling, `d = 128`. |
| Two gates | GDN uses `silu(z)`; attention uses `sigmoid(gate)` — different activations. |
| Conv weight | stored `[conv_dim, 4, 1]` (already axis-moved). |

## Loader layout (text model)

Keys live under `language_model.model.` (the `Model.sanitize` drops `vision_tower`
and remaps `model.language_model` → `language_model.model`). Per layer:

- linear: `linear_attn.{in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,out_proj}`
  (quantized: `.weight` U32 + `.scales`/`.biases` bf16), `linear_attn.conv1d.weight`,
  `linear_attn.A_log`, `linear_attn.dt_bias`, `linear_attn.norm.weight`
- full: `self_attn.{q_proj,k_proj,v_proj,o_proj}` (quantized), `self_attn.{q_norm,k_norm}.weight`
- both: `input_layernorm.weight`, `post_attention_layernorm.weight`,
  `mlp.{gate_proj,up_proj,down_proj}` (quantized)
- top: `embed_tokens` (quantized, tied head on 4B), `norm.weight`, (`lm_head` on 27B)

## Performance notes

The incumbent GDN kernel (`gated_delta.py`) is **one `metal_kernel` dispatch per
layer** with the timestep loop *inside* the kernel (state register-resident,
parallelized over `B·Hv·Dv` and the `Dk` reduction). It is **not** T separate
dispatches (that is only the CPU `gated_delta_ops` fallback), and it is **not**
the chunkwise-parallel formulation. Consequences:

- **Decode** (T=1): a single step — near-optimal already, so expect a Qwen2.5-like
  ~parity-to-small-lead regime.
- **Prefill / long context** (large T): the T-axis is serial, under-utilizing the
  GPU. The lever is the **chunkwise-parallel** GDN algorithm (intra-chunk matmuls +
  inter-chunk scan), which is matmul-bound — where cider's qmm is already at parity.

Incumbent baseline to beat (mlx-lm, this machine): 27B decode **~33 tok/s** at
4-bit (~15.4 GB peak); 4B **~172 tok/s** (~2.5 GB).

## Open items

- **Phase 1 (loader) landed:** `crates/cider-press-models/src/qwen3_5/` parses the
  nested `text_config` and maps every text tensor under `language_model.model.*`
  (vision skipped), shape-validated against config and byte-round-tripped vs the
  4B archive. No forward yet — Phases 2–3 add the mixers.
- **Phase 2 (gated-attention layer) landed:** `qwen3_5/attention.rs` + `block.rs` —
  per-head query‖gate split, weighted QK-norm, partial RoPE, GQA SDPA (composed
  head_dim-256 prefill / vector decode), sigmoid output gate; full decoder-layer
  parity vs mlx-lm on the 4B layer-3 fixture. Deferred follow-ups: a steel `bd256`
  prefill kernel (composed SDPA is correctness-first); a hermetic (regenerated)
  fixture flow (vs the current `/tmp` path); and fp32 score accumulation in the
  composed SDPA (it accumulates in bf16 today — within the combined bound, but a
  lever if parity is ever tightened).
- **Phase 3 (Gated-DeltaNet linear mixer) landed:** `qwen3_5/gated_deltanet.rs` +
  `block.rs` — the linear mixer composes a depthwise causal Conv1d (+SiLU) over
  the `q‖k‖v` projection, the per-(token, value-head) `compute_g`/`beta` gates
  (fp32), weightless per-head q/k RMSNorm + scalar scale, interleaved-GQA
  key/value broadcast, the fp32 per-token delta-rule recurrence over an
  `[Hv, Dv, Dk]` state, and a gated output RMSNorm before `out_proj`. Mixer and
  whole-linear-layer parity vs mlx-lm on the 4B layer-0 fixture (mixer max_abs
  ~7.8e-3, full layer ~1.6e-2). Enabled by 4 new runtime ops (`exp`, `log`,
  `cast`, `concat`). `Qwen35DecoderLayer` now generalizes over both mixers (a
  `Mixer::{Full, Linear}` enum) with two entry points — `forward` (full attn +
  KV cache) and `forward_linear` (GDN prefill, no cache). Deferred: the
  chunkwise-parallel GDN kernel (the Phase-7 perf-lead lever vs mlx-lm's naive
  per-token recurrence), and full-model decode-cache wiring — a unified per-layer
  cache enum threaded through one `forward` — which lands in Phase 4 (full-model
  assembly). Phase 3 complete.
- mRoPE interleave layout for text (sections `[11,11,10]` = 32 rotary pairs):
  **confirmed during Phase 2** — `rope_type=default` collapses mRoPE to plain 1-D
  partial RoPE for text, so no multi-section position handling is needed.
- See [`ROADMAP.md`](../../../ROADMAP.md) §4 for the phased build and §5 for the
  remaining gates.
