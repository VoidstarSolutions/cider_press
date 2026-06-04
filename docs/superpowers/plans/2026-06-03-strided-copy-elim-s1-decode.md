# Strided Copy Elimination — Stage 1 (Decode) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the ~144 per-token materializing `copy()` dispatches on the decode path by recognizing that, at T=1, every attention permute is contiguous-modulo-size-1 and needs no copy.

**Architecture:** A `Tensor::copy()` on a view whose strides are contiguous after ignoring size-1 axes returns a zero-copy canonical-strided view over the same buffer instead of an `OpKind::Copy`. The only attention consumer that doesn't already accept a contiguous view is `rope`; this plan also teaches `rope` to accept one. No model-layer changes, no new kernels, no edits to vendored `.metal`. Prefill (T>1) is unaffected — its permutes are genuine reorders that fail the contiguous-mod-unit predicate and still materialize.

**Scope:** This is **Stage 1 (S1)** of the spec `docs/superpowers/specs/2026-06-03-strided-copy-elimination-design.md`. Stages S2–S5 (prefill strided kernels: rope strides, slice_update strided src, gemm transpose+broadcast, qmv strided) are deferred to follow-on plans written after S1 is measured.

**Tech Stack:** Rust, Apple Silicon Metal runtime (`cider-press-runtime`), vendored MLX kernels. Tests via `cargo test`; perf via `cargo run --release --features profiling -- bench`.

---

## File Structure

- `crates/cider-press-runtime/src/strides.rs` — add `Strides::is_contiguous_ignoring_unit_dims`. (C1)
- `crates/cider-press-runtime/src/tensor.rs` — `copy()` elision (C2); relax `rope()` construction guard to accept a contiguous view.
- `crates/cider-press-runtime/src/eval.rs` — `dispatch_rope` resolves a view via the shared contiguous-view resolver (`matmul_input_bytes`).
- `docs/QWEN_PERF.md`, `docs/ARCHITECTURE.md` — record the decode re-measure.

**Landing order (keeps every commit green):** C1 → rope-accepts-view → C2 elision → e2e gate → measure/docs. Rope must accept views *before* C2 starts producing them, or the end-to-end decode would break at the project_heads→rope boundary.

---

### Task 1: C1 — size-1-aware contiguity predicate

**Files:**
- Modify: `crates/cider-press-runtime/src/strides.rs` (add method after `is_contiguous`, ~line 67)
- Test: `crates/cider-press-runtime/src/strides.rs` (`#[cfg(test)] mod tests`)

- [ ] **Step 1: Write the failing tests**

Add to the `tests` module in `strides.rs`:

```rust
#[test]
fn contiguous_ignoring_unit_dims_accepts_exact_contiguous() {
    let shape = Shape::from([2, 3, 4]);
    let s = Strides::contiguous(&shape);
    assert!(s.is_contiguous_ignoring_unit_dims(&shape));
}

#[test]
fn contiguous_ignoring_unit_dims_accepts_degenerate_permute() {
    // [1, H, 1, D] produced by permuting [1, 1, H, D] ([0,2,1,3]).
    // Canonical would be [H*D, D, D, 1]; the permute leaves the size-1
    // axis-2 stride at H*D. Only a size-1 axis differs -> contiguous.
    let shape = Shape::from([1, 5, 1, 4]); // H=5, D=4
    let s = Strides::from([20isize, 4, 20, 1]);
    assert!(s.is_contiguous_ignoring_unit_dims(&shape));
}

#[test]
fn contiguous_ignoring_unit_dims_rejects_genuine_transpose() {
    // [2,3] transposed to [3,2]: strides [1,2]. Both dims > 1 -> reorder.
    let shape = Shape::from([3, 2]);
    let s = Strides::from([1isize, 2]);
    assert!(!s.is_contiguous_ignoring_unit_dims(&shape));
}

#[test]
fn contiguous_ignoring_unit_dims_rejects_broadcast() {
    // Broadcast row [1,4] -> [3,4]: leading stride 0 on a size-3 axis.
    let shape = Shape::from([3, 4]);
    let s = Strides::from([0isize, 1]);
    assert!(!s.is_contiguous_ignoring_unit_dims(&shape));
}

#[test]
fn contiguous_ignoring_unit_dims_scalar_is_true() {
    let shape = Shape::scalar();
    let s = Strides::contiguous(&shape);
    assert!(s.is_contiguous_ignoring_unit_dims(&shape));
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p cider-press-runtime --lib strides::tests::contiguous_ignoring_unit_dims 2>&1 | tail -20`
Expected: FAIL — `no method named is_contiguous_ignoring_unit_dims`.

- [ ] **Step 3: Implement the predicate**

Add after `is_contiguous` (after line 67) in `strides.rs`:

```rust
    /// Whether these strides describe a contiguous (row-major) layout
    /// of `shape` once size-1 axes are ignored.
    ///
    /// A size-1 axis contributes exactly one element, so its stride is
    /// never used during iteration — a view that is row-major over its
    /// non-unit axes (e.g. a `permute` that only moved a size-1 axis)
    /// enumerates the same buffer addresses in the same order as a truly
    /// contiguous tensor. `copy()` uses this to elide a materialization
    /// that would be byte-identical to its input. Differs from
    /// [`Strides::is_contiguous`], which requires *exact* equality.
    #[must_use]
    pub fn is_contiguous_ignoring_unit_dims(&self, shape: &Shape) -> bool {
        let dims = shape.dims();
        debug_assert_eq!(dims.len(), self.0.len());
        let mut expected: isize = 1;
        for (axis, &dim) in dims.iter().enumerate().rev() {
            if dim <= 1 {
                continue;
            }
            if self.0[axis] != expected {
                return false;
            }
            let dim_signed =
                isize::try_from(dim).expect("shape dimension does not fit in isize");
            expected = expected
                .checked_mul(dim_signed)
                .expect("contiguous stride computation overflowed isize");
        }
        true
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p cider-press-runtime --lib strides::tests::contiguous_ignoring_unit_dims 2>&1 | tail -20`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add crates/cider-press-runtime/src/strides.rs
git commit -m "feat(runtime): Strides::is_contiguous_ignoring_unit_dims

Row-major contiguity check that ignores size-1 axes (their stride is
never used). Enables eliding byte-identical copies of degenerate
permute views.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Teach `rope` to accept a contiguous view

**Files:**
- Modify: `crates/cider-press-runtime/src/tensor.rs` — `rope()` construction guard (the `if self.inner.view.is_some()` block).
- Modify: `crates/cider-press-runtime/src/eval.rs` — `dispatch_rope` input resolution (the `dense_input_buffer(input, ...)` call, ~line after `inputs.get(1)`).
- Test: `crates/cider-press-runtime/src/tensor.rs` (`#[cfg(test)] mod tests`).

Rationale: today `rope` requires a materialized dense leaf, which is why the model copies before it. C2 (next task) will turn that copy into a view, so `rope` must accept a *contiguous* view first. It must still reject a *strided* view (genuine reorder) — passing actual strides to the rope kernel is deferred to S2.

- [ ] **Step 1: Write the failing test**

Add to the `tests` module in `tensor.rs`:

```rust
#[test]
fn rope_accepts_canonical_view_matches_leaf() {
    // rope must accept a *canonical* view (view.is_some() AND
    // is_dense_contiguous) — that is exactly what copy()'s elision (Task 3)
    // produces for a degenerate decode permute. It must read identically to
    // the equivalent dense leaf. (The raw permute view is non-canonical and
    // is exercised in Task 3; here we test the canonical-view input rope
    // will actually receive.)
    let device = Device::shared().expect("system default device");
    let data: Vec<half::bf16> = (0..8).map(|i| half::bf16::from_f32(i as f32 + 1.0)).collect();
    let leaf = Tensor::from_slice(&device, &data, [1usize, 2, 1, 4]).expect("from_slice");

    // Identity permute -> a view that shares the leaf's canonical strides.
    let view = leaf.permute(&[0, 1, 2, 3]).expect("identity permute");
    assert!(view.inner.view.is_some(), "test input must be a view");
    assert!(
        view.layout().is_dense_contiguous(view.shape()),
        "test input must be canonical"
    );

    let offset = Tensor::from_slice(&device, &[0i32], [1usize]).expect("offset");
    let r_view = view.rope(&offset, 10000.0, 1.0, 4).expect("rope on canonical view");
    let r_leaf = leaf.rope(&offset, 10000.0, 1.0, 4).expect("rope on leaf");
    r_view.eval().expect("eval view");
    r_leaf.eval().expect("eval leaf");

    assert_eq!(
        r_view.cpu_slice::<half::bf16>().unwrap(),
        r_leaf.cpu_slice::<half::bf16>().unwrap()
    );
}

#[test]
fn rope_still_rejects_strided_view() {
    // A genuine transpose (both axes > 1) is NOT contiguous-mod-unit; rope
    // must still reject it until S2 wires actual strides into the kernel.
    let device = Device::shared().expect("system default device");
    let data: Vec<half::bf16> = (0..48).map(|i| half::bf16::from_f32(i as f32)).collect();
    // [1,2,3,4] -> permute([0,2,1,3]) -> [1,3,2,4] genuine reorder view.
    let base = Tensor::from_slice(&device, &data, [1usize, 2, 3, 4]).expect("from_slice");
    let view = base.permute(&[0, 2, 1, 3]).expect("permute");
    let offset = Tensor::from_slice(&device, &[0i32], [1usize]).expect("offset");
    let err = view.rope(&offset, 10000.0, 1.0, 4).unwrap_err();
    assert!(format!("{err}").contains("view"));
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p cider-press-runtime --lib rope_accepts_canonical_view 2>&1 | tail -20`
Expected: FAIL — `rope_accepts_canonical_view_matches_leaf` errors with `"rope: view inputs are not supported"` (the current guard rejects all views).

- [ ] **Step 3: Relax the construction guard**

In `tensor.rs`, replace the `rope` view guard:

```rust
        if self.inner.view.is_some() {
            return Err(Error::InvalidArgument(
                "rope: view inputs are not supported; copy() first to materialise \
                 a dense version (eval-side dispatch reads inputs as dense leaves \
                 and does not resolve view chains)"
                    .into(),
            ));
        }
```

with:

```rust
        // A contiguous view (canonical strides, e.g. a copy()-elided
        // degenerate permute) is byte-equivalent to a dense leaf, so the
        // eval-side resolver can read it from offset 0. A strided view
        // (genuine reorder) is not — its kernel strides aren't wired yet
        // (deferred to the prefill stage), so reject it.
        if self.inner.view.is_some() && !self.layout().is_dense_contiguous(self.shape()) {
            return Err(Error::InvalidArgument(
                "rope: strided view inputs are not supported; copy() first to \
                 materialise a dense version (the rope dispatch reads inputs as \
                 contiguous bytes and does not yet pass view strides)"
                    .into(),
            ));
        }
```

- [ ] **Step 4: Resolve the view in eval**

In `eval.rs` `dispatch_rope`, replace:

```rust
    let in_bytes = dense_input_buffer(input, outputs, index_of)?;
```

with:

```rust
    // `input` may be a contiguous view (a copy()-elided degenerate permute
    // from attention); resolve through the view chain and assert offset 0,
    // exactly like the qmv/slice_update activation paths. `Tensor::rope`
    // rejects non-contiguous views at construction, so the bytes read here
    // are contiguous-from-offset-0.
    let in_bytes = matmul_input_bytes("rope", input, outputs, index_of)?;
```

(`matmul_input_bytes` falls back to `dense_input_buffer` for non-view inputs, so prefill — where rope still receives materialized leaves — is unchanged.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test -p cider-press-runtime --lib rope_ 2>&1 | tail -20`
Expected: PASS (both new tests).

- [ ] **Step 6: Run the full runtime suite (no regressions)**

Run: `cargo test -p cider-press-runtime 2>&1 | tail -15`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add crates/cider-press-runtime/src/tensor.rs crates/cider-press-runtime/src/eval.rs
git commit -m "feat(runtime): rope accepts a contiguous view input

Relax the rope construction guard to allow contiguous (canonical-stride)
views and resolve them in dispatch_rope via the shared contiguous-view
resolver. Strided views are still rejected (kernel-stride wiring deferred
to the prefill stage). Enables copy() elision of the project_heads
permute to feed rope directly.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: C2 — `copy()` elision for degenerate views

**Files:**
- Modify: `crates/cider-press-runtime/src/tensor.rs` — `copy()` (lines ~695-713).
- Test: `crates/cider-press-runtime/src/tensor.rs` (`#[cfg(test)] mod tests`).

- [ ] **Step 1: Write the failing tests**

Add to the `tests` module in `tensor.rs`:

```rust
#[test]
fn copy_of_degenerate_permute_view_is_elided() {
    // [1,1,H,D] -> permute([0,2,1,3]) -> [1,H,1,D] contiguous-mod-unit view.
    // copy() must return a zero-copy view (no Copy op), with canonical
    // strides and the correct logical values.
    let device = Device::shared().expect("system default device");
    let data: Vec<f32> = (0..8).map(|i| i as f32).collect(); // H=2, D=4
    let base = Tensor::from_slice(&device, &data, [1usize, 1, 2, 4]).expect("from_slice");
    let view = base.permute(&[0, 2, 1, 3]).expect("permute"); // [1,2,1,4]

    let copied = view.copy().expect("copy");

    // Elided: it is a view, not a Copy op, and reports canonical strides.
    assert!(copied.op_kind().is_none(), "expected elision, got an op");
    assert_eq!(copied.shape().dims(), &[1, 2, 1, 4]);
    match copied.layout() {
        Layout::Dense { strides } => {
            assert!(strides.is_contiguous(copied.shape()), "strides not canonical");
        }
        Layout::Quantized(_) => panic!("unexpected quantized layout"),
    }

    // Values match the logical permute.
    copied.eval().expect("eval");
    // element [0,h,0,d] == base[0,0,h,d] == data[h*4+d]; canonical order
    // over [1,2,1,4] is just data in order.
    assert_eq!(copied.cpu_slice::<f32>().unwrap(), data.as_slice());
}

#[test]
fn copy_of_genuine_transpose_still_materializes() {
    // [2,3] -> [3,2] transpose is NOT contiguous-mod-unit; copy() must
    // still build a Copy op (existing behavior).
    let device = Device::shared().expect("system default device");
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let src = Tensor::from_slice(&device, &data, [2, 3]).expect("from_slice");
    let transposed = src.permute(&[1, 0]).expect("permute"); // 3x2 view
    let copied = transposed.copy().expect("copy");
    assert_eq!(copied.op_kind(), Some(OpKind::Copy));
}

#[test]
fn copy_of_contiguous_leaf_still_materializes() {
    // A non-view dense leaf: copy() must produce a fresh Copy op (callers
    // may rely on a private buffer).
    let device = Device::shared().expect("system default device");
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let leaf = Tensor::from_slice(&device, &data, [2, 2]).expect("from_slice");
    let copied = leaf.copy().expect("copy");
    assert_eq!(copied.op_kind(), Some(OpKind::Copy));
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p cider-press-runtime --lib copy_of_degenerate 2>&1 | tail -20`
Expected: FAIL — `copy_of_degenerate_permute_view_is_elided` sees `op_kind() == Some(Copy)`.

- [ ] **Step 3: Implement elision in `copy()`**

In `tensor.rs`, after the existing `if !matches!(self.inner.layout, Layout::Dense { .. })` guard and before `let layout = dense_layout(...)`, insert:

```rust
        // Elide a copy whose output would be byte-identical to its input:
        // a view that is contiguous once size-1 axes are ignored (e.g. a
        // decode-time permute that only moved the size-1 T axis). Return a
        // zero-copy view with canonical strides over the same storage
        // instead of an OpKind::Copy dispatch. Only views qualify — a
        // non-view leaf/op is already canonical and callers may rely on
        // copy() handing back a private buffer, so those still materialize.
        let strides = match &self.inner.layout {
            Layout::Dense { strides } => strides,
            Layout::Quantized(_) => unreachable!("guarded as Dense above"),
        };
        if self.inner.view.is_some()
            && !strides.is_contiguous(&self.inner.shape)
            && strides.is_contiguous_ignoring_unit_dims(&self.inner.shape)
        {
            let (source, byte_offset) = self.flatten_view();
            let canonical = Strides::contiguous(&self.inner.shape);
            return Ok(Self::view(
                source,
                byte_offset,
                self.inner.shape.clone(),
                canonical,
            ));
        }
```

(Ensure `Strides` is in scope; it is — `Layout::Dense { strides }` already references it. Add `use crate::strides::Strides;` to the imports if not already present.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p cider-press-runtime --lib copy_of_ 2>&1 | tail -20`
Expected: PASS (3 new tests + the existing `copy_of_*` tests stay green).

- [ ] **Step 5: Run the full runtime suite**

Run: `cargo test -p cider-press-runtime 2>&1 | tail -15`
Expected: PASS. (If any existing test asserted a Copy op on a degenerate view, reconcile it — none is expected.)

- [ ] **Step 6: Commit**

```bash
git add crates/cider-press-runtime/src/tensor.rs
git commit -m "feat(runtime): elide copy() of contiguous-mod-unit views

A view that is row-major over its non-unit axes (a decode permute that
only moved the size-1 T axis) is byte-identical to its source, so copy()
returns a zero-copy canonical-strided view instead of an OpKind::Copy.
Removes the ~144 decode-path copy dispatches. Genuine reorders and leaves
still materialize.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: End-to-end parity + lints

**Files:** none (verification only).

- [ ] **Step 1: Run the greedy-decode parity gate**

Run: `cargo test -p cider-press-models --release generator_parity_greedy_qwen2 2>&1 | tail -20`
Expected: PASS — greedy decode output is unchanged (the elision is byte-identical, so token-for-token parity with `mlx_lm` holds). This is the proof that C2 + the rope change preserve decode correctness end-to-end.

- [ ] **Step 2: Workspace test, clippy, fmt, docs**

Run:
```bash
cargo test --workspace --release 2>&1 | tail -15
cargo clippy --all-targets --all-features -- -D warnings 2>&1 | tail -15
cargo fmt --check
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps 2>&1 | tail -5
```
Expected: all clean.

- [ ] **Step 3: Commit (only if fmt/clippy required edits)**

```bash
git add -A
git commit -m "chore(runtime): fmt/clippy for copy elision"
```
(Skip if nothing changed.)

---

### Task 5: Re-measure decode + update perf docs

**Files:**
- Modify: `docs/QWEN_PERF.md`, `docs/ARCHITECTURE.md`.

- [ ] **Step 1: Capture the decode profile**

Run (warm; run twice, use the second to skip the JIT compile):
```bash
cargo run --release -p cider-press --features profiling -- bench \
  --checkpoint /Users/zacharyheylmun/dev/llm/qwen2.5-0.5b-instruct-4bit-bf16 \
  --max-tokens 128 --warmup 8 --gpu-profile 2>&1 | tail -40
```
Expected: `gpu.copy` dispatch count drops from 146 toward ~2 (only the non-attention misc copies remain); decode tok/s rises from ~412. Record the new decode tok/s (median of ≥3 warm runs), the new `gpu.copy` line, and total dispatches/token.

- [ ] **Step 2: Update `QWEN_PERF.md`**

In the Throughput table and headline, record the new decode tok/s and the narrowed gap to `mlx_lm` (~566). In the GPU-breakdown subsection, update the `gpu.copy` row (146 → new count) and the total dispatches/token (~680 → new). Add a "Decode copy elimination (S1)" note: decode permutes are contiguous-mod-size-1 at T=1, so `copy()` elides them to zero-copy views; cite the spec. Keep the historical pre-S1 numbers labeled as such.

- [ ] **Step 3: Update `ARCHITECTURE.md`**

In backlog item #9 (strided-aware kernels) and item #4a (fusion / dispatch-count), note that the **decode** copies are eliminated via copy-elision (S1) — distinct from the prefill strided-kernel work (S2–S5, still pending) — and update the critical-path summary decode tok/s + dispatch count to match `QWEN_PERF.md`.

- [ ] **Step 4: Commit**

```bash
git add docs/QWEN_PERF.md docs/ARCHITECTURE.md
git commit -m "docs(perf): record decode copy-elimination (S1) re-measure

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## After S1

Re-measure result decides S2–S5: if decode lands near `mlx_lm` the prefill strided-kernel investment can be reprioritized; either way each of S2 (rope strides), S3 (slice_update strided src), S4 (gemm transpose+broadcast), S5 (qmv strided) gets its own plan with kernel-level parity tests. Do **not** start them from this plan.
