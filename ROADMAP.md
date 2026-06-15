# ROADMAP — Running Qwen3.6-27B on cider-press

> **Status:** planning (authored 2026-06-14). No code committed against this yet.
> This is the *strategic* roadmap. Each phase below gets its own detailed,
> TDD-style implementation plan (via the `writing-plans` skill) at the moment it
> is started — once the relevant mlx-lm reference source is in hand and the code
> steps can be written concretely rather than guessed.

**Goal:** run **Qwen3.6-27B** (dense, Gated-DeltaNet hybrid attention) end-to-end
on a 64 GB Apple-Silicon Mac via the `cider-press` CLI, parity-validated against
`mlx-lm`, and then *faster than the incumbent Mac option* — first at decode
parity, then with a genuine **prefill / long-context lead**.

**Why this model, why now:** Qwen3.6-27B (released 2026-04-22, Apache-2.0) is a
*dense* 27B that posts flagship agentic-coding scores — it beats the prior
open flagship Qwen3.5-397B-A17B on SWE-bench Verified (77.2 vs 76.2), SWE-bench
Pro, and Terminal-Bench 2.0. It is the **best local coder that comfortably fits
64 GB** (27B @ 4-bit ≈ ~15 GB), and it carries the architecture where the
incumbent kernel is weakest — so it is simultaneously the highest-value target
*and* the biggest perf-lead opportunity. The strategic bet: a real perf advantage
on a *popular* model is where cider-press could attract users.

---

## 1. Target facts (verified 2026-06-14)

| | Qwen3.6-27B (target) | Qwen3.5-4B (dev vehicle) |
|---|---|---|
| Class | dense FFN, GDN hybrid attention, VLM | same |
| `model_type` | `qwen3_5` (3.6 reuses the 3.5 text arch — **not a typo**) | `qwen3_5` |
| hidden / layers | 5120 / 64 | 2560 / 32 |
| attn heads / kv (head_dim) | 24 / 4 (**256**) | 16 / 4 (256) |
| linear-attn heads (head_dim) | **48** value / 16 key (128) | 32 value / 16 key (128) |
| conv kernel | 4 (depthwise causal) | 4 |
| full-attention interval | every 4th layer (3 GDN : 1 full) | every 4th |
| partial rotary factor | 0.25 | 0.25 |
| rope_theta | 1e7 | 1e7 |
| tie embeddings | **false** (separate lm_head) | true |
| intermediate / vocab | 17408 / 248320 | 9216 / 248320 |
| vision tower | ~27-layer ViT — **skipped (text-only)** | ~24-layer — skipped |

**Memory fit:** 27B @ 4-bit ≈ ~15 GB weights; 6–8-bit (~20–27 GB) also fits 64 GB
with room for the GDN state cache + the (sparse, 1-in-4) KV cache. Comfortable.

**Ruled out — Qwen3-Coder-Next (80B-A3B MoE):** MoE keeps all 80B weights resident
(~40 GB @ 4-bit) vs ~45–48 GB usable of 64 GB → edge-of-feasible before cache,
needs slow expert-offload to RAM. Not a local target on this machine.

---

## 2. The performance opportunity (the differentiator)

Unlike Qwen2.5 — where we fought *mature* MLX kernels (qmv/qmm) and parity-to-+5 %
was the ceiling — the GDN linear attention is **young and naive** in the incumbent:

- `mlx-lm`'s `mlx_lm/models/gated_delta.py` runs a **recurrent (non-chunkwise)**
  formulation: **one `mx.fast.metal_kernel` dispatch per GDN layer** with the
  `for t in 0..T` loop **inside** the kernel (state register-resident), parallelized
  over `B·Hv·Dv` + the `Dk` reduction — but the **T-axis is serial**. It does **not**
  use the chunkwise-parallel (matmul-heavy) algorithm. (Earlier "T separate
  dispatches" claim was wrong — that's only the CPU `gated_delta_ops` fallback.)
  So the lever is **GPU occupancy / arithmetic-intensity on the serial-T axis**
  (chunkwise-parallel → matmul throughput), NOT dispatch-count.
- Immaturity is visible in the wild: ollama #15865 (state written bf16 not fp32 →
  corrupted output) and mlx-lm #932 (2.7× decode slowdown pathology).

**Where the lead is, by regime:**
- **Decode** (T = 1/step → 1 dispatch/GDN layer): ~Qwen2.5 regime — modest gains
  from kernel polish + cross-layer command-buffer batching.
- **Prefill / long context** (large T): the prize — sequential T-dispatch + naive
  kernel vs a chunkwise-parallel, matmul-bound rewrite. This is the regime
  agentic coding actually lives in (262 K context).

**Levers, big → small:** (1) chunkwise-parallel GDN algorithm (structural,
matmul-bound → inherits cider's qmm parity; documented in NVlabs GatedDeltaNet /
flash-linear-attention — a real port, not research); (2) tiled kernel; (3)
command-buffer dispatch batching (proven lever from the Qwen2.5 work).

---

## 3. Architecture delta — reuse vs. new

**Reused from the Qwen2.5 port (no new work):** RMSNorm (fused), SiLU/SwiGLU MLP,
quantized matmul (qmv/qmm), embedding gather + dequantize, the standard
full-attention block, GQA broadcast, the safetensors loader pattern, the
3-layer parity-test discipline, the lazy `Tensor` graph + `eval()` boundary.

**New work (this roadmap):**
1. `qwen3_6` config + text-only weight loader (qkvz/ba projections, conv1d, `A_log`,
   `dt_bias`, gates; per-layer linear-vs-full selection; skip vision).
2. **head_dim-256 Gated-Attention SDPA** specialization (vendored steel attn is
   head_dim 128 only) — plus **QK-norm** (`q_norm`/`k_norm` `[256]`) and a sigmoid
   **output gate** (q_proj emits 8192 = 2×4096, split **per-head** into
   query‖gate — reshape `[…, H_q, 2·D_h]` then split last axis, not a flat
   block split). Not plain attention.
3. **Partial-rotary (0.25) RoPE** variant (+ confirm text-only mRoPE collapses to 1-D).
4. **Depthwise causal Conv1d** dispatch (kernel size 4).
5. **Gated-DeltaNet recurrence** — sequential form first (vendor mlx-lm's
   `gated_delta_step` MSL or compose), then **chunkwise-parallel** for the lead.
6. **Two-part recurrent cache**: conv sliding window (kernel-1 tokens) + the
   recurrent state matrix `[B, Hv, Dv, Dk]` — **state kept in fp32** (parity tripwire).

---

## 4. Phased plan

Dependency-ordered. Each phase produces working, testable software and exits on a
concrete **parity gate** (the Qwen2.5 discipline). Phases 0–5 reach *correctness*
(the model runs); 6–8 are the *perf lead*.

### Phase 0 — Verify, spec, and fixtures (no kernel code)
- Re-fetch the **authoritative Qwen3.6-27B `config.json`** + weight index; lock
  every hyperparameter and the exact safetensors key layout.
- **Confirm the dev vehicle**: does a smaller Qwen3.6 variant exist? If not,
  Qwen3.5-4B is the architectural proxy (same GDN hybrid).
- Pull the reference sources: `mlx_lm/models/qwen3_next.py` (composition,
  `layer_types`) and `gated_delta.py` (the kernel + `gated_delta_update`).
- Resolve the open questions in §5.
- Generate **golden fixtures** (per-op mlx-lm reference outputs) for every new op,
  via a one-shot `uv` PEP-723 script (Python = one-shot tool only).
- Write `docs/inference/models/qwen3.6.md` — the architecture spec (also rustdoc).
- **Exit:** locked config + golden vectors for each new op + written spec. No Rust yet.

### Phase 1 — Config + text-only loader (4B)
- `Qwen3_6Config`; safetensors loader mapping all per-layer weights; vision skipped;
  shape-validate against config.
- **Exit:** 4B checkpoint loads, all tensors mapped & validated; no forward yet.

### Phase 2 — Gated-attention layer (the 1-in-4)
- head_dim-256 SDPA + partial-rotary RoPE + standard KV cache, **plus QK-norm and
  the sigmoid output gate** (q_proj → query‖gate). GQA 16 q / 4 kv.
- **Exit:** single gated-attention layer parity vs mlx-lm (combined-bound tolerance).

### Phase 3 — Gated-DeltaNet layer, **sequential form** (correctness baseline)
- Depthwise causal Conv1d + SiLU; `compute_g` gate (exp/−exp/softplus), sigmoid β;
  q/k rms_norm + scale; two-part recurrent cache (fp32 state); per-token recurrence
  (vendor `gated_delta_step` MSL or compose).
- Deliberately match the incumbent's *naive* perf here — correctness first.
- **Exit:** single GDN layer parity vs mlx-lm's sequential path (fp32 state honored).

### Phase 4 — Assemble full model (4B), end-to-end parity
- 3:1 interleave, final norm, tied lm_head (4B), embedding (confirm any scaling),
  greedy decode through the `cider-press` CLI.
- **Exit:** logits parity vs mlx-lm on 4B across a prompt; identical greedy token stream.

### Phase 5 — Scale to 27B (functional milestone — *it runs*)
- Loader handles untied embeddings, 64 layers, the official quantization; memory
  profiling on the 64 GB machine; end-to-end correctness vs mlx-lm 27B.
- **Exit:** **Qwen3.6-27B runs end-to-end in the CLI, parity-validated, fits in memory.**

### Phase 6 — Perf baseline + dispatch levers
- Measure decode tok/s + prefill ms vs `mlx_lm` (27B and 4B). Apply command-buffer
  batching across the 3:1 layer pattern + fence chain (proven decode levers).
- **Exit:** measured baseline; decode at/near mlx parity (the ~5 % regime).

### Phase 7 — Chunkwise-parallel GDN (the lead)
- Implement chunkwise-parallel Gated DeltaNet (intra-chunk parallel via matmul +
  inter-chunk scan) — the algorithm mlx-lm skipped — plus a tiled kernel. Target
  the prefill / long-context lead.
- **Exit:** **measured prefill / long-context speedup vs `mlx_lm`** — the genuine
  perf-advantage milestone.

### Phase 8 — Stretch
- Tiled decode kernel; long-context cache memory work; `.metallib` precompile for
  cold-start; quantization-quality sweeps.

---

## 5. Open questions / verification gates

**Resolved in Phase 0 (2026-06-14, `scripts/dump_qwen35_arch.py` on Qwen3.5-4B):**
- ✓ **Dev vehicle** — Qwen3.5-4B loads + generates text-only; no smaller 3.6 needed.
- ✓ **Tooling** — `mlx-lm` loads this `Qwen3_5ForConditionalGeneration` **text-only**
  (172 tok/s decode on 4B); mlx-vlm not needed.
- ✓ **Quantization** — affine, group_size 64, 4-bit → matches cider's dequant ABI.
- ✓ **mRoPE** — `mrope_section [11,11,10]` (=32 rotary pairs), `partial_rotary_factor
  0.25`; text-only shares one 1-D position across sections.
- ✓ **Conv1d** — depthwise causal, 8192 channels (q‖k‖v), kernel 4.
- ✓ **Loader layout locked** — see §3; note GDN projections are **split** into
  separate quantized `in_proj_qkv/z/a/b` (not the reference's fused qkvz/ba), full
  layers carry **QK-norm** + an **output gate** in `q_proj` (8192 = query‖gate,
  split per-head — reshape `[…, H_q, 2·D_h]` then split last axis).

**Also resolved (2026-06-14):**
- ✓ **Embedding scaling** — none; plain `nn.Embedding`, no √hidden (`qwen3_5.py:263`).
- ✓ **27B architecture** — confirmed *now* (not deferred): identical to 4B up to
  scale, `linear_num_value_heads 48`, untied `lm_head`; loads + generates text-only
  in mlx-lm (~33 tok/s, ~15.4 GB).
- ✓ **Conv1d** — **compose, don't vendor**: a depthwise causal kernel-4 conv is 4
  shifted scaled adds from existing ops; mlx's general conv kernels aren't needed.

**Phase 0 is complete** (spec: `docs/inference/models/qwen3.6.md`). Remaining
fine-grained per-op fixtures (isolated conv / recurrence) are added per-phase as
each gate needs them; the backbone fixtures (embed, GDN layer, gated-attn layer,
logits) are generated by `scripts/dump_qwen35_fixtures.py`.

**Deferred to build time:**
- **mRoPE 1-D collapse** for text (sections `[11,11,10]`) — confirm in Phase 2.

---

## 6. Non-goals / scope guards

- Vision / multimodal — skipped entirely; text generation only.
- Training / autograd, MoE routing, non-Apple platforms (per CLAUDE.md non-goals).
- No premature perf work before Phase 5 correctness (correctness gate first, always).

---

## 7. References

- mlx-lm: `mlx_lm/models/qwen3_next.py`, `mlx_lm/models/gated_delta.py`
- Gated DeltaNet (NVlabs, ICLR 2025); flash-linear-attention (chunkwise algorithm)
- Incumbent-immaturity evidence: ollama #15865 (fp32 state), mlx-lm #932 (decode cliff)
- cider-press prior art: `docs/inference/` (attention, execution-model, qwen2.5)
- Memory: `project_qwen35_gdn.md`
