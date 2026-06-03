# Strided-aware copy elimination — design

Status: approved (design); implementation pending.
Branch: `perf/strided-copy-elim` (off `perf/async-pipelining`).
Date: 2026-06-03.

## Problem

Decode is ~1.35× slower than `mlx_lm` (~420 vs ~566 tok/s) and the gap
is now substantially **CPU-side**. A fresh `bench --gpu-profile` at the
current ~412 tok/s path shows:

- Per-token wall 2.42 ms = GPU wait 1.81 ms (75%) + CPU encode 0.43 ms
  (18%) + forward-build/overhead ~0.19 ms (8%).
- Our **GPU wait (1.81 ms) ≈ MLX's entire token (1.77 ms)** — our GPU
  execution is already near MLX parity; the visible gap is un-hidden CPU
  per-token work plus the dispatch chain length.
- Dispatch count ~680/token. `gpu.copy` is **146 dispatches, 14.4% of
  GPU** — pure layout materialization MLX never pays.

The 146 copies/token come from three sites in
`crates/cider-press-models/src/qwen2/attention.rs`, ×24 layers (+2 misc):

| site | op chain | per layer |
|---|---|---|
| `project_heads` (q,k,v) | `reshape → permute([0,2,1,3]) → copy` | 3 |
| KV-slab write (k,v) | `permute([2,1,3,0]) → copy → reshape` | 2 |
| attn-output collapse | `permute([0,2,1,3]) → copy → reshape` | 1 |

### Key finding: at decode (T=1) every one of these copies is degenerate

Working the strides at T=1, each `permute` only moves the **size-1 T (or
batch) axis**. The non-size-1 axes stay in row-major order, so the
permuted view is already contiguous *except* for a stride on a size-1
axis — which never affects iteration. The `copy()` materializes a buffer
**byte-identical** to its input.

They are not already elided because `Strides::is_contiguous`
(`strides.rs:64`) does an **exact** equality against canonical contiguous
strides; it does not ignore size-1 axes. MLX avoids these because its
`is_row_contiguous` ignores singleton dims, making `mx.copy` a no-op
there.

Consequence: the **decode** win needs *no new kernels* — only a
size-1-aware contiguity check. Strided-aware *kernels* are needed only
for **prefill** (T>1), where the permutes genuinely reorder memory.

## Roadmap rationale: why a general strided seam, not a T=1 special case

The architecture choice was made against the forward model roadmap, not
just Qwen2.5. GGUF metadata for the next targets:

- **Qwen3.6-27B** (`qwen35`): `ssm.conv_kernel = 4` (a conv1d /
  linear-attention component → channel↔sequence transposes every block,
  T≫1); `nextn_predict_layers = 1` (**MTP / speculative decode → T>1 in
  the decode hot path**, breaking the T=1 degeneracy); `key_length 256`
  with `rope.dimension_count 64` (**partial RoPE** — rotate a strided
  slice of head_dim); `rope.dimension_sections = [11,11,10,0]`
  (**mRoPE**).
- **Qwen3.6-35B-A3B** (`qwen35moe`): **256 experts, top-8 + shared
  expert** → MoE gather/scatter routing (indexed/grouped expert
  matmuls); same conv, partial/mRoPE, MTP.
- **Vision projector** (`clip`, both ship `mmproj`): 27-block ViT,
  `spatial_merge_size 2` (patch-merge reshape/transpose), conv patch
  stem, always T>1.
- **Parakeet v3** (NVIDIA FastConformer ASR — not local, from
  architecture knowledge): conv subsampling + conformer blocks
  alternating depthwise conv1d and MHSA → channel↔sequence transposes
  every block, always T≫1, no T=1 path.

A T=1-only elision (the narrow "Approach A") is inert for almost all of
this. The recurring cost across these models is not the new compute
kernels (conv, `gather_qmm` — separate work regardless) but the **layout
transforms surrounding them**. A general strided seam eliminates those
once instead of re-solving copy-vs-strided per new op. Hence: **build the
seam, roll it out incrementally, ship the Qwen2.5 decode win first.**

## Approach (the seam)

A tensor already carries its strides (`Layout::Dense { strides }` + view
`byte_offset`). The seam moves the materialize-or-not decision from the
model into each consumer op, behind one shared helper. The model stops
calling `.copy()` in the attention path.

### C1 — size-1-aware contiguity

`Strides::is_contiguous_ignoring_unit_dims(shape) -> bool`: canonical
row-major after dropping size-1 axes. Plus a helper to produce the
canonical (truly contiguous) strides for such a view. `is_contiguous`
itself is unchanged (callers that need exact contiguity keep it); the new
predicate is additive.

### C2 — `copy()` elision

In `Tensor::copy`: if `self` is a **view** whose strides are contiguous-
modulo-size-1, return `Self::view(source, byte_offset, shape,
canonical_strides)` — metadata only, no `OpKind::Copy`, no dispatch, no
allocation. The result reports canonical contiguous strides, so it
satisfies every existing consumer contiguity check (including
`is_dense_contiguous`'s exact match) and `reshape` stays zero-copy on it.

Constraints:
- Only elide when `self` is a non-canonical view (the case where the
  alternative is a real strided copy). An already-canonical input still
  materializes a fresh buffer — preserves the "give me a private buffer"
  intent some callers rely on.
- Byte-identical by construction (contiguous-mod-unit ⇒ logical row-major
  order equals buffer order), so parity is trivial.

### C3 — lowering helper (the reusable seam)

`ensure_readable(input: &Tensor, req: LayoutReq) -> Result<Tensor>` where
`LayoutReq` is an enum of what the consuming kernel can read. The full
variant set is declared up front to lock the seam's shape; only the
attention-path variants are implemented now (YAGNI on the rest), and
every unimplemented variant has a safe default so the seam never silently
narrows.

**Implemented now (attention path):**

- `Contiguous` — materialize unless already contiguous.
- `ContiguousModUnit` — accept contiguous-mod-size-1 (tighten via C2);
  else materialize.
- `Strided { max_rank }` — accept any Dense strides up to a rank the
  kernel handles; else materialize.
- `TransposedLastTwo` — accept a last-two-dims-swapped view (gemm reads
  it via a transpose flag); else materialize.
- `BroadcastBatch` — accept stride-0 batch axes (gemm batch stride 0);
  else materialize.

**Stub variants (declared, not implemented — lock the seam shape):**

- `Gathered { .. }` — indexed/gathered rows for MoE expert routing
  (`gather_qmm`). Driver: **Qwen3.6-35B-A3B** (256 experts, top-8).
- `ChannelTransposed` — channel↔sequence layout for conv1d (the
  `[B, C, T]` ↔ `[B, T, C]` swap a conformer/SSM block needs each pass).
  Drivers: **Qwen3.6** (`ssm.conv_kernel = 4`), **Parakeet v3**
  (conformer conv), **vision** conv stem.
- `PatchMerge { .. }` — windowed reshape for ViT patch merging
  (`spatial_merge_size`). Driver: **vision projector**.

Each stub variant's `ensure_readable` arm falls back to a materializing
`copy()` (the `Contiguous` behavior) until its kernel-side strided
support lands — correct, just not yet faster, and it compiles today so
the planned surface can't be dropped. A dedicated strided implementation
replaces the fallback when the driving model is built.

`ensure_readable` **never errors** — every arm, implemented or stub,
falls back to a materializing `copy()`. So removing a model `.copy()`
before its consumer learns the strided variant is still correct. Future
ops add a `LayoutReq` variant (or reuse one) without re-solving the
copy-vs-strided decision.

### C4 — per-consumer strided dispatch

Each attention consumer adopts `ensure_readable` and passes the input's
strides/offset to its kernel:

| consumer | `LayoutReq` | copy removed | kernel notes |
|---|---|---|---|
| `rope` (q,k) | `Strided{3}` | project_heads q/k | kernel already takes `strides[3]` / `out_strides[3]` (`rope.rs:74`) — dispatch-only |
| `slice_update` (k,v src) | `Strided` | k_upd / v_upd | read src via strides into the slab |
| `gemm` (Q@Kᵀ, probs@V, GQA) | `TransposedLastTwo` + `BroadcastBatch` | Kᵀ-transpose, GQA-broadcast, prefill k/v_view | vendored `steel` matmul supports transpose + leading-dim/batch strides; today `dispatch_gemm_bf16` assumes contiguous tiles |
| `qmv` (o_proj x) | `Strided` | attn-output collapse | strided activation read |

No edits to vendored `.metal`: the `g_copy`/`gg_copy`, `steel` matmul,
and strided `rope` variants are already vendored; the work is
dispatch-side selection + passing strides.

## Data flow (attention, post-change)

Model builds q/k/v with `reshape`+`permute` (zero-copy views) and **no
`.copy()`**. Consumers call `ensure_readable` and read strided where
supported; a materializing copy appears only when a kernel genuinely
can't read the layout.

## Staging (one concern per commit)

- **S1 = C1 + C2.** Removes all 144 decode copies with **zero model
  changes** (existing `.copy()` calls become free at T=1; the
  canonical-strided result still satisfies every consumer). Independently
  mergeable — the 1.35×-gap decode lever. Re-measure decode in
  `QWEN_PERF.md`.
- **S2.** C3 helper + RoPE adoption; drop q/k `project_heads` copy.
- **S3.** `slice_update` strided src; drop k_upd / v_upd copy.
- **S4.** `gemm` transpose + broadcast batch; drop Kᵀ + GQA-broadcast +
  prefill k/v_view copies. Largest piece.
- **S5.** `qmv` strided activation; drop attn-output collapse copy.

S2–S5 target prefill (and the MTP/T>1 decode hot path once that lands);
each carries a kernel-parity test and a prefill re-measure.

## Risks

- **Aliasing.** C2's elided result aliases its source buffer. The
  KvCache `src` views (k_upd/v_upd) now read the projection buffer in
  place; the source stays live via the view's `Arc`. Covered by the
  existing KvCache aliasing contract (drop K/V views before the next
  `update`); add an explicit elision-aliasing test.
- **`reshape` on the elided view.** Must stay zero-copy on a
  canonical-strided view — verify (it should: canonical strides are
  contiguous).
- **Non-attention `copy()` callers** (e.g. `generator.rs:334` logits
  slice). C2 only elides when byte-identical, so they are unaffected
  except when their view happens to be contiguous-mod-unit, where elision
  is correct.
- **gemm transpose parity (S4).** The transposed/broadcast path must
  match the materialized-copy result bit-exactly; guard with a prefill
  attention parity test in addition to the kernel test.

## Testing

- Kernel parity per consumer: bit-exact vs the materialized-copy result
  for data-movers / gemm / qmv; bf16-ULP for rope (per the repo
  tolerance rules).
- End-to-end `generator_parity_greedy_qwen2` gate on every stage.
- Decode re-measure after S1; prefill re-measure after S4 and S5;
  recorded in `QWEN_PERF.md`.
- New: C2 elision-aliasing test; `is_contiguous_ignoring_unit_dims` unit
  tests (incl. genuine-reorder negatives at T>1).

## Non-goals

- No new conv / `gather_qmm` / vision-merge kernels — the seam is built
  so they slot in later, but they are out of scope until those models
  land.
- No edits to vendored `.metal`.
- No prefill attention fusion (`sdpa_full` / Plan B) — a separate lever.
