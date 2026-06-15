# Task: optional-gamma (weightless) `rms_norm`

**Branch:** `feat/qwen35-rmsnorm-optional-gamma`
**Worktree:** `.claude/worktrees/feat+rmsnorm-optional-gamma`
**Origin:** phase-3 code-review finding #8 (the GDN review of `feat/qwen35-phase3`).
Branched off the phase-3 tip (commit `fedfab0`), so the consumer
(`qk_norm_scale`) and the review cleanups are present.

---

## TL;DR

`qk_norm_scale` (the Gated-DeltaNet q/k weightless RMSNorm) fakes "no weight"
by allocating a full `[D]` ones-vector every call and feeding it to the
weighted fused `rms_norm`. mlx instead runs the *same* kernel with a **stride-0
scalar weight** (a 1-element buffer, `w_stride = 0`) so every lane reads
`w[0]`. This task adds a weightless path to `Tensor::rms_norm` that mirrors mlx,
and switches `qk_norm_scale` onto it.

**Read this before you start — it changes the obvious mental model:** the
forward RMSNorm kernel **always multiplies by `w`**. There is *no* weightless
code path in the forward kernel; `has_w` is only consulted by the VJP
(backward) variants, which cider doesn't compile (inference-only). So "weightless"
does **not** mean "skip the multiply." It means "multiply by a stride-0 scalar
`1.0` instead of a `[D]` vector." The win is **buffer size + API fidelity, not
fewer FLOPs.** Size accordingly — this is a small, clean change, not an
architectural one. If your plan involves editing the vendored `.metal` to add an
`if (has_w)` branch to the *forward* kernel, stop: that diverges from upstream
and isn't what mlx does.

---

## Why this exists (the finding)

From the phase-3 review:

> Weightless RMSNorm is faked by constructing a fresh ones-vector gamma
> (`vec![bf16::ONE; dim]`) and feeding it to the fused weighted `rms_norm`,
> because the fused `Tensor::rms_norm` requires a gamma. mlx's fused path
> supports `weight=None` directly; cider only has the weighted form.

Accurate, with the caveat above (the multiply still runs in both). Net cost of
the status quo: a `D`-element host buffer built + uploaded on every
`qk_norm_scale` call (twice per GDN forward — q and k). Correct today, just not
mlx-faithful and slightly wasteful on buffer traffic.

mlx-lm reference for the consumer (`qwen3_5.py:GatedDeltaNet`):
```python
inv = k.shape[-1] ** -0.5
q = (inv**2) * mx.fast.rms_norm(q, None, 1e-6)   # weight=None
k =  inv     * mx.fast.rms_norm(k, None, 1e-6)   # weight=None
```

---

## How mlx does weight=None (the reference mechanism)

Forward dispatch — `../mlx/mlx/backend/metal/normalization.cpp`,
`RMSNorm::eval_gpu` (~lines 40–92):

- `bool has_w = w.ndim() != 0;` — None ⇒ `w` is a 0-dim array ⇒ `has_w = false`.
- `uint32_t w_stride = (w.ndim() == 1) ? w.strides()[0] : 0;` — None ⇒ **`w_stride = 0`**.
- `hash_name = op_name + (has_w ? "_w" : "_now")` and the func-const list binds
  `has_w` at slot 20. **But** — the forward kernels (`rms_single_row`,
  `rms_looped`) do **not** reference `has_w` in their bodies; they always emit
  `out[i] = w[w_stride * i] * ...` (`rms_norm.metal:71,76,144,150`). So binding
  `has_w` for the forward only changes the pipeline *name/hash* (a distinct
  cached variant); it does not change behavior. `has_w` is genuinely used only
  in the `vjp_rms_*` variants (`rms_norm.metal:248,258,360,373`).
- `w` is still bound to buffer slot 1 even when None. With `w_stride = 0`,
  `w += w_stride*lid*N_READS` is a no-op and `w[w_stride*i] == w[0]` for every
  element. mlx supplies a scalar `1.0` for `w` in the None case, so the per-lane
  multiply is `× 1.0`.

**Upshot:** weightless = bind a 1-element `[1.0]` BF16 buffer + pass
`w_stride = 0`. Identical math to the `[D]` ones-vector, fewer bytes.

---

## Current cider state (what you're changing)

### Consumer — the ONLY weightless caller
`crates/cider-press-models/src/qwen3_5/gated_deltanet.rs`, `qk_norm_scale`
(~lines 286–296):
```rust
let ones = Tensor::from_slice(device, &vec![bf16::ONE; dim], [dim])?;
let normed = x.copy()?.rms_norm(&ones, 1e-6)?;
```
Audit confirms every *other* `rms_norm` call passes a real loaded gamma and is
unaffected: `nn.rs:44` (`RmsNormLayer`), `nn.rs:299`, `attention.rs:313/314`
(gated-attn q/k norm — weighted, eps from config), `gated_deltanet.rs:623`
(GDN gated output norm — weighted). **Do not touch those.**

### Runtime API
`crates/cider-press-runtime/src/tensor.rs`, `Tensor::rms_norm(&self, weight: &Self, eps: f32)`
(~line 1820). Hard-requires `weight` rank-1, size `[hidden]`, BF16, contiguous,
non-view, same device. This is where the optional-gamma surface lands.

### eval dispatch
`crates/cider-press-runtime/src/eval.rs`, `dispatch_rms_norm` (~line 2180):
reads `inputs[0]=x`, `inputs[1]=w`, reinterprets both as `bf16`, computes
`axis_size`/`n_rows`, calls `rms_norm_bf16(...)`. The `OpKind::RmsNorm { eps }`
node carries only `eps` today; `w` is an op input.

### kernels-layer dispatch
`crates/cider-press-kernels/src/kernels/rms_norm.rs`, `rms_norm_bf16(...)`
(~line 36):
- Validates `w.len() == axis_size`.
- **Hardcodes `let w_stride: u32 = 1;`** (~line 88) with a comment citing mlx's
  `(w.ndim()==1) ? w.strides()[0] : 0`.
- Binds x@0, w@1, out@2, eps@3, axis_size@4, w_stride@5; pipeline name
  `"rmsbfloat16"` via plain `library.pipeline(...)` (no func consts).

The kernel ABI **already supports `w_stride`** as bound bytes — it's just pinned
to 1. That's the lever.

---

## DECISION (2026-06-15, after deeper survey)

**Chosen: A2 (`Option<&Self>` weight) + cache the scalar.** Rationale from the
survey:

- **Worth it? Fidelity, yes; perf, no.** A1/A2 do *not* reduce the per-call
  buffer-upload count (still one `from_slice` per call) — they only shrink the
  weight buffer 256 B → 2 B. The forward kernel multiplies either way. The
  *only* real (small) perf lever is **caching** the scalar so the weightless
  path stops uploading a buffer per call — hence we add it.
- **A2 churn is small (spec overcounted "~6 sites").** Direct `Tensor::rms_norm`
  production callers are only **3**: `nn.rs:44` (the `nn::rms_norm` wrapper —
  keeps `gamma: &Tensor`, passes `Some` internally), `gated_deltanet.rs:289`
  (→ `None`), `gated_deltanet.rs:623` (→ `Some`). `RmsNormLayer` (nn.rs:299) and
  gated-attn q/k (attention.rs:313/314) go through the wrapper → **zero** change.
- **Cache location:** a device-level lazily-built 1-element BF16 `[1.0]` leaf
  `Tensor`, reused as `inputs[1]` for every weightless `rms_norm`. `OpKind::RmsNorm`
  carries `w_stride` (1 weighted / 0 weightless); eval relaxes the
  `w.len()==axis_size` check for the stride-0 case.

GDN-decode hot-path context: ~24 GDN layers × 2 (q,k) = ~48 weightless calls per
token in the 4B; `linear_key_head_dim = 128` (the `[D]` being cached away).

---

## Recommended approach (Option A — mlx-faithful, minimal)

Plumb an optional weight through to a stride-0 scalar binding. No `.metal` edits.

1. **kernels layer** — let `rms_norm_bf16` take the weightless case:
   - Either add a `w_stride: u32` parameter (caller passes 1 for weighted, 0 for
     weightless), or an `Option<&Buffer<bf16>>` weight. Cleanest: accept
     `w: &Buffer<bf16>` + `w_stride: u32`, and relax the `w.len() == axis_size`
     check to allow `w.len() == 1 && w_stride == 0`.
   - Keep the pipeline as `"rmsbfloat16"`. (Optionally also bind `has_w=false`
     via `pipeline_specialized` + a `_now` name to mirror mlx's cache key, but
     since the forward kernel ignores `has_w`, this is cosmetic/parity-only and
     adds a second compiled variant — recommend **skipping** it unless you want
     byte-for-byte pipeline-name parity. Note the existing comment at
     `library.rs:~209` already says the forward references no func-const slots.)

2. **runtime API** — surface weightless. Two viable shapes:
   - **(A1)** Add `Tensor::rms_norm_weightless(&self, eps) -> Result<Self>` that
     builds a 1-element `[1.0]` BF16 leaf and schedules the op with `w_stride=0`.
   - **(A2)** Change `rms_norm` to take `weight: Option<&Self>` (more invasive —
     touches all ~6 call sites). A1 is less churn and reads clearly at the call
     site. **Prefer A1** unless you want the Option ergonomics everywhere.
   - The `OpKind::RmsNorm` node needs to carry whether it's weightless (so
     `dispatch_rms_norm` knows to pass `w_stride=0` and tolerate a 1-element w).
     Add a `weightless: bool` (or `w_stride: u32`) field to `OpKind::RmsNorm`.
   - Relax the rank-1/size-`[hidden]` weight validation for the weightless path
     (the scalar weight is `[1]`).

3. **eval** — `dispatch_rms_norm` passes `w_stride = if weightless { 0 } else { 1 }`
   and skips the `w.len()==axis_size` assumption for the scalar case.

4. **consumer** — `qk_norm_scale`: drop the `vec![bf16::ONE; dim]` + its
   `Tensor::from_slice`, call the new weightless path:
   ```rust
   let normed = x.copy()?.rms_norm_weightless(1e-6)?;
   ```
   (`x.copy()` is still needed — rms_norm rejects view inputs.)

### Option B (do NOT do unless scope changes)
Add `if (has_w)` to the *forward* `rms_single_row`/`rms_looped` to truly skip the
multiply. This **edits vendored MLX kernels** (must retain MIT header, re-run the
qmv/qmm parity canary, diverges from upstream which doesn't gate the forward),
and saves only `D` multiply-by-1.0 ops per row — not worth it. Listed only so the
next session doesn't rediscover it as a dead end.

---

## Acceptance criteria

- `qk_norm_scale` no longer allocates a `[D]` ones-vector; uses the weightless
  path with `w_stride = 0` and a 1-element (or stride-0) weight.
- **Parity unchanged.** These must stay green (run in the worktree):
  ```sh
  cargo test --release -p cider-press-models qwen3_5
  cargo test --release -p cider-press-models --test qwen35_gated_deltanet --test qwen35_gated_attention
  ```
  Especially `qk_norm_scale_matches_weightless_rms_reference`,
  `gated_deltanet_layer0_matches_mlx`, `decoder_layer0_matches_mlx`.
- A new runtime-level test: weightless `rms_norm` == weighted `rms_norm` with an
  explicit `[D]` ones gamma, bit-exact (it's the same kernel + same values).
- `cargo clippy --all-targets --all-features -- -D warnings` and `cargo fmt`
  clean.
- The qmv/qmm parity canary (`cargo test --workspace --release`) is unaffected —
  but you only need it if you touched anything under `kernels-mlx/` (you
  shouldn't, under Option A).

## Out of scope
- The looped (axis > 4096) kernel: cider only wires `rms_single_row`
  (`rms_norm_bf16` rejects `axis_size > LOOPED_LIMIT`). The GDN head dim is 128,
  so weightless only needs the single-row path. Don't wire looped unless a
  consumer appears.
- The weighted path / `RmsNormLayer` / gated-output norm — unchanged.

## Reality check on value
This is a **fidelity + minor-buffer-traffic** cleanup, not a perf lever (the
multiply runs either way; GDN head dim is 128, off the decode hot path). If, once
scoped, it looks like more churn than it's worth, "keep the ones-vector and close
#8 as won't-fix" is a legitimate outcome — the current code is correct. Decide
after step 1 sizing.

---

## Key file/line index (as of `fedfab0`)
| What | Path | Approx line |
|---|---|---|
| Consumer (only weightless caller) | `crates/cider-press-models/src/qwen3_5/gated_deltanet.rs` | 286–296 (`qk_norm_scale`) |
| Runtime API | `crates/cider-press-runtime/src/tensor.rs` | 1820 (`rms_norm`) |
| eval dispatch | `crates/cider-press-runtime/src/eval.rs` | 2180 (`dispatch_rms_norm`); `OpKind::RmsNorm` at 701 |
| kernels dispatch (w_stride pinned to 1) | `crates/cider-press-kernels/src/kernels/rms_norm.rs` | 36 (`rms_norm_bf16`), ~88 (`w_stride`) |
| Library compile note | `crates/cider-press-kernels/src/library.rs` | ~206 (`rms_norm`), `pipeline_specialized` referenced ~201 |
| Vendored kernel (don't edit) | `kernels-mlx/mlx/backend/metal/kernels/rms_norm.metal` | `rms_single_row` ~13; `has_w` decl ~3; unconditional `w[...]` 71,76 |
| mlx forward dispatch (reference) | `../mlx/mlx/backend/metal/normalization.cpp` | `RMSNorm::eval_gpu` ~40–92; `w_stride` ~83; `has_w` ~121 |
| mlx-lm consumer (reference) | `../mlx/.../qwen3_5.py` (or mlx-lm) | `GatedDeltaNet`: `rms_norm(q, None, 1e-6)` |
