# Fused Prefill Attention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the composed prefill (T>1) attention chain in Qwen2 with the single vendored MLX `steel_attention` (`sdpa_full`) kernel, mirroring MLX's single-primitive routing, to close most of the ~2.37 ms prefill gap.

**Architecture:** Vendor MLX's `steel/attn/` kernel subtree and JIT-compile it as a new `steel_attention` kernel library. Add a `dispatch_sdpa_full_bf16` routine in the kernels crate. Keep one `OpKind::Sdpa`; teach `eval.rs::dispatch_sdpa` to route by query length — `T_q==1` → existing `sdpa_vector` (decode, unchanged), `T_q>1` → steel-full. The Qwen2 model drops its `is_decode` branch and always calls `Tensor::sdpa`; `composed_sdpa` is deleted once parity is green.

**Tech Stack:** Rust (Cargo workspace), Apple Metal via `objc2-metal`, vendored MLX `.metal`/`.h` kernels (MIT), `half::bf16`, `uv`/PEP-723 one-shot Python for MLX fixtures.

**Spec:** `docs/superpowers/specs/2026-06-09-fused-prefill-attention-design.md` (local breadcrumb; the specs dir is gitignored).

**Target kernel (exact host name — verified against the instantiate macro in `steel_attention.metal`):**
```
steel_attention_bfloat16_bq32_bk32_bd64_wm4_wn1_maskbfloat16
```
Note: the `instantiate_attn` macro uses a separate `tname` token (`bfloat16`, **no `_t`**) for the host name, distinct from the `dtype` template token (`bfloat16_t`). This is a kernel-name gotcha — the name is `bfloat16`, not `bfloat16_t`. Function constants: `align_Q`(200), `align_K`(201), `has_mask`(300), `do_causal`(301), `has_sinks`(302), all bool.

---

## File Structure

**Created:**
- `kernels-mlx/mlx/backend/metal/kernels/steel/attn/params.h` (+ 6 more vendored files — see Task 1)
- `crates/cider-press-kernels/tests/steel_attention_parity.rs` — low-level kernel parity test
- `crates/cider-press-models/tests/sdpa_prefill_fused.rs` — fused-vs-MLX + fused-vs-composed oracle (Task 4)

**Modified:**
- `crates/cider-press-kernels/build.rs` — add `steel_attention` to `KERNEL_SOURCES`
- `crates/cider-press-kernels/src/library.rs` — `STEEL_ATTENTION_SOURCE` const + `KernelLibrary::steel_attention()`
- `crates/cider-press-kernels/src/kernels/sdpa.rs` — `AttnParams` + `dispatch_sdpa_full_bf16`
- `crates/cider-press-runtime/src/device.rs` — `steel_attention_library()` accessor + `OnceLock` field
- `crates/cider-press-runtime/src/tensor.rs` — relax `Tensor::sdpa` to accept `T_q>1`
- `crates/cider-press-runtime/src/eval.rs` — route `dispatch_sdpa` by `T_q`; add `dispatch_sdpa_full`
- `crates/cider-press-models/src/qwen2/attention.rs` — drop `is_decode` branch; delete `composed_sdpa` (Task 5)
- `kernels-mlx/VENDORED.md`, `scripts/sync_mlx_kernels.sh` — track the new subtree
- `docs/QWEN_PERF.md`, `docs/ARCHITECTURE.md`, `CLAUDE.md` — measured result (Task 6)

---

## Task 1: Vendor steel/attn kernel subtree + compile as a library

**Files:**
- Create: `kernels-mlx/mlx/backend/metal/kernels/steel/attn/{params.h,loader.h,mma.h,transforms.h,attn.h,kernels/steel_attention.h,kernels/steel_attention.metal}`
- Modify: `crates/cider-press-kernels/build.rs` (KERNEL_SOURCES array)
- Modify: `crates/cider-press-kernels/src/library.rs` (source const + ctor)
- Modify: `crates/cider-press-runtime/src/device.rs` (library accessor)
- Modify: `kernels-mlx/VENDORED.md`, `scripts/sync_mlx_kernels.sh`
- Test: `crates/cider-press-kernels/tests/steel_attention_parity.rs` (compile-smoke only in this task)

- [ ] **Step 1: Copy the 7 vendored files from the sibling MLX checkout**

```bash
cd /Users/zacharyheylmun/dev/llm/cider-press
SRC=../mlx/mlx/backend/metal/kernels/steel/attn
DST=kernels-mlx/mlx/backend/metal/kernels/steel/attn
mkdir -p "$DST/kernels"
cp "$SRC/params.h" "$SRC/loader.h" "$SRC/mma.h" "$SRC/transforms.h" "$SRC/attn.h" "$DST/"
cp "$SRC/kernels/steel_attention.h" "$SRC/kernels/steel_attention.metal" "$DST/kernels/"
ls -R "$DST"
```

Expected: the 5 top-level `.h` + `attn.h` and the `kernels/steel_attention.{h,metal}` listed. (Every other include — `utils.h`, `steel/defines.h`, `steel/utils.h`, `steel/gemm/params.h`, `steel/utils/integral_constant.h` — is already vendored.)

- [ ] **Step 2: Verify the MLX MIT header is present on every copied file**

Run:
```bash
for f in $(find kernels-mlx/mlx/backend/metal/kernels/steel/attn -type f); do echo "== $f =="; head -1 "$f"; done
```
Expected: each begins with `// Copyright © 2024[-25] Apple Inc.`. If any file lacks it, prepend the standard MLX MIT header (see a sibling file like `steel/gemm/gemm.h`). Per CLAUDE.md, MLX-derived files MUST retain the header.

- [ ] **Step 3: Add the kernel to the build flattener**

In `crates/cider-press-kernels/build.rs`, find the `KERNEL_SOURCES` array (the `sdpa_vector_only.metal` entry is the anchor) and add:

```rust
    (
        "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.metal",
        "steel_attention",
    ),
```

The flattener follows `#include "..."` automatically, so no other build change is needed.

- [ ] **Step 4: Add the source const + library constructor**

In `crates/cider-press-kernels/src/library.rs`, next to `SDPA_VECTOR_SOURCE`:

```rust
/// Pre-flattened MLX `steel/attn/kernels/steel_attention.metal` (fused
/// prefill / `sdpa_full` path), produced by `build.rs`.
const STEEL_ATTENTION_SOURCE: &str =
    include_str!(concat!(env!("OUT_DIR"), "/steel_attention_inlined.metal"));
```

And next to `sdpa_vector`:

```rust
    /// JIT-compile MLX's vendored fused attention kernel (`steel_attention`,
    /// the `sdpa_full` prefill path). All instantiations require
    /// `[[function_constant]]` specialization — use [`Self::pipeline_specialized`].
    pub fn steel_attention(device: &Device) -> Result<Self> {
        Self::from_source(device, STEEL_ATTENTION_SOURCE)
    }
```

- [ ] **Step 5: Add the cached device accessor**

In `crates/cider-press-runtime/src/device.rs`: add a field beside `sdpa_vector_library`:

```rust
    steel_attention_library: OnceLock<kernels::KernelLibrary>,
```

initialize it `OnceLock::new()` at BOTH constructor sites (there are two — search `sdpa_vector_library: OnceLock::new(),` and mirror each). Then add the accessor beside `softmax_library()`:

```rust
    pub(crate) fn steel_attention_library(&self) -> Result<&kernels::KernelLibrary> {
        if let Some(lib) = self.inner.steel_attention_library.get() {
            return Ok(lib);
        }
        let lib = kernels::KernelLibrary::steel_attention(&self.inner.kernels)?;
        Ok(self.inner.steel_attention_library.get_or_init(|| lib))
    }
```

- [ ] **Step 6: Write the compile-smoke test (specializes the pinned kernel name)**

Create `crates/cider-press-kernels/tests/steel_attention_parity.rs`:

```rust
//! Steel fused-attention (`sdpa_full`) kernel parity.
#![cfg(target_os = "macos")]

use cider_press_kernels::library::{FunctionConstant, KernelLibrary};
use cider_press_runtime::Device;

/// The kernel compiles and the pinned host name specializes (the
/// function-constant set MLX uses for causal bf16 prefill). Guards the
/// kernel-name gotcha: `bfloat16` (no `_t`) in the host name.
#[test]
fn steel_attention_kernel_specializes() {
    let device = Device::system_default().expect("device");
    let library = device.steel_attention_library().expect("compile steel_attention");
    let consts = [
        FunctionConstant::Bool { index: 200, value: false }, // align_Q (qL=39 % 32 != 0)
        FunctionConstant::Bool { index: 201, value: false }, // align_K
        FunctionConstant::Bool { index: 300, value: false }, // has_mask
        FunctionConstant::Bool { index: 301, value: true },  // do_causal
        FunctionConstant::Bool { index: 302, value: false }, // has_sinks
    ];
    library
        .pipeline_specialized("steel_attention_bfloat16_bq32_bk32_bd64_wm4_wn1_maskbfloat16", &consts)
        .expect("specialize pinned steel_attention kernel");
}
```

Note: `device.steel_attention_library()` is `pub(crate)`. If the test cannot reach it, expose a thin `#[doc(hidden)] pub` shim in the runtime crate's test-utils, OR move this assertion into a `#[cfg(test)] mod tests` inside `device.rs`. Prefer the in-crate unit test if the accessor stays `pub(crate)`.

- [ ] **Step 7: Build and run the smoke test**

Run:
```bash
cargo build -p cider-press-kernels && cargo test -p cider-press-runtime steel_attention_kernel_specializes -- --nocapture
```
Expected: PASS. First run pays the one-time Metal JIT compile of the flattened steel source (tens of seconds, disk-cached cross-process). If it fails with `KernelNotFound`, grep the flattened output for the real name and correct the pinned string:
```bash
grep -o 'steel_attention_bfloat16[a-z0-9_]*' "$(find target -name steel_attention_inlined.metal | head -1)" | sort -u
```

- [ ] **Step 8: Record the vendored files**

Update `kernels-mlx/VENDORED.md` — add the `steel/attn/` subtree under the file list, keeping the upstream commit SHA consistent with the rest. In `scripts/sync_mlx_kernels.sh`, add `steel/attn` to whatever directory/glob set the script copies so weekly drift-sync pulls it.

- [ ] **Step 9: Commit**

```bash
git add kernels-mlx/ crates/cider-press-kernels/build.rs crates/cider-press-kernels/src/library.rs \
  crates/cider-press-runtime/src/device.rs crates/cider-press-kernels/tests/steel_attention_parity.rs \
  scripts/sync_mlx_kernels.sh
git commit -m "feat(kernels): vendor MLX steel_attention (sdpa_full) + JIT library

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: `dispatch_sdpa_full_bf16` + low-level MLX parity

**Files:**
- Modify: `crates/cider-press-kernels/src/kernels/sdpa.rs` (add `AttnParams`, `dispatch_sdpa_full_bf16`)
- Test: `crates/cider-press-kernels/tests/steel_attention_parity.rs` (add the parity test)

- [ ] **Step 1: Write the failing parity test (fixture at T=39 causal)**

Append to `crates/cider-press-kernels/tests/steel_attention_parity.rs`. This dispatches the kernel directly and compares against an MLX causal-prefill fixture. Reuse the test-utils fixture helpers (`dump_mlx_op`, `read_bf16`, `tempdir`).

```rust
use cider_press_test_utils::{dump_mlx_op, read_bf16, tempdir};

#[test]
fn steel_attention_prefill_causal_matches_mlx() {
    // Qwen2.5-0.5B shape: H_q=14, H_kv=2 (gqa=7), D=64, T=39 causal.
    let (h_q, h_kv, d, t) = (14usize, 2usize, 64usize, 39usize);
    let tmp = tempdir();
    let path = tmp.join("steel-sdpa-causal-39.safetensors");
    dump_mlx_op(&path, &[
        "sdpa",
        "--h-q", "14", "--h-kv", "2",
        "--seq-q", "39", "--seq-kv", "39",
        "--head-dim", "64", "--dtype", "bf16", "--causal",
    ]);
    let q = read_bf16(&path, "q");   // [1, H_q, T, D] row-major
    let k = read_bf16(&path, "k");   // [1, H_kv, T, D]
    let v = read_bf16(&path, "v");
    let expected = read_bf16(&path, "out"); // [1, H_q, T, D]

    let device = cider_press_runtime::Device::system_default().expect("device");
    // ... upload q/k/v to bf16 Buffers, allocate out = H_q*T*D, build
    //     AttnParams from (h_q,h_kv,d,t), call dispatch_sdpa_full_bf16,
    //     commit_and_wait, read back, compare.
    let got = dispatch_and_read(&device, &q, &k, &v, h_q, h_kv, d, t);

    // softmax/exp drift → combined np.allclose bound (atol 0.03, rtol 0.08),
    // matching the existing prefill-causal SDPA tolerance.
    assert_combined_tolerance("steel sdpa_full prefill", &got, &expected, 0.03, 0.08);
}
```

(Implement `dispatch_and_read` + `assert_combined_tolerance` as local test helpers; `assert_combined_tolerance` is `|a-b| <= atol + rtol*|b|` element-wise, the same formula as `sdpa_qwen2.rs::assert_within_tolerance` — copy it.)

- [ ] **Step 2: Run it to confirm it fails (no dispatch fn yet)**

Run: `cargo test -p cider-press-kernels steel_attention_prefill_causal_matches_mlx`
Expected: FAIL to compile — `dispatch_sdpa_full_bf16` not found.

- [ ] **Step 3: Add the `AttnParams` struct (mirror `steel/attn/params.h` exactly)**

In `crates/cider-press-kernels/src/kernels/sdpa.rs`. Field order and types MUST match MLX's struct byte-for-byte (`setBytes` copies it raw). Layout: 6×i32 + 1×f32 + 7×i32 = 14×4 = 56 bytes (8-aligned), then 4×`[i64;3]` = 96 bytes; total 152, no interior padding.

```rust
/// Mirror of MLX `mlx::steel::AttnParams` (`steel/attn/params.h`),
/// `#[repr(C)]` so `setBytes` matches the MSL `constant AttnParams&`.
/// All strides are in *elements* (B, H, L); the trailing D stride is an
/// implicit 1 (the kernel does typed pointer math on the head dim).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct AttnParams {
    pub b: i32,
    pub h: i32,
    pub d: i32,
    pub q_l: i32,
    pub k_l: i32,
    pub gqa_factor: i32,
    pub scale: f32,
    pub n_q: i32,
    pub n_k: i32,
    pub n_q_aligned: i32,
    pub n_k_aligned: i32,
    pub q_l_rem: i32,
    pub k_l_rem: i32,
    pub q_l_off: i32,
    pub q_strides: [i64; 3],
    pub k_strides: [i64; 3],
    pub v_strides: [i64; 3],
    pub o_strides: [i64; 3],
}
```

- [ ] **Step 4: Add `dispatch_sdpa_full_bf16` (mirror MLX `sdpa_full_self_attention_metal`)**

In the same file. Constants `bq=32`, `bk=32` (since `d<128`), `wm=4`, `wn=1`. The caller supplies q/k/v/out buffers and strides; this fn computes the param block, specializes the pinned kernel, binds q0/k1/v2/o3 + params@4, and dispatches threadgroups `(NQ, H, B)` with group `(32, wm, wn)`.

```rust
/// Args for [`dispatch_sdpa_full_bf16`]. Strides are element strides for
/// the (B, H, L) axes; D is contiguous (stride 1). Layouts:
/// q/o `[B, H_q, qL, D]`, k/v `[B, H_kv, kL, D]`.
#[derive(Clone, Copy, Debug)]
pub struct SdpaFullArgs {
    pub batch: i32,
    pub h_q: i32,
    pub h_kv: i32,
    pub head_dim: i32,
    pub q_len: i32,
    pub k_len: i32,
    pub scale: f32,
    pub q_strides: [i64; 3],
    pub k_strides: [i64; 3],
    pub v_strides: [i64; 3],
    pub o_strides: [i64; 3],
    /// `do_causal` (func const 301). Prefill self-attention = true.
    pub causal: bool,
}

#[allow(clippy::too_many_lines)]
pub fn dispatch_sdpa_full_bf16(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    q: &Buffer<bf16>,
    k: &Buffer<bf16>,
    v: &Buffer<bf16>,
    out: &mut Buffer<bf16>,
    args: SdpaFullArgs,
) -> Result<()> {
    const BQ: i32 = 32;
    const BK: i32 = 32; // head_dim 64 < 128 → 32
    const WM: i32 = 4;
    const WN: i32 = 1;
    if args.head_dim != 64 {
        return Err(Error::InvalidArgument(format!(
            "sdpa_full: only head_dim 64 instantiated (got {})",
            args.head_dim
        )));
    }
    if args.gqa_factor() <= 0 || args.h_q % args.h_kv != 0 {
        return Err(Error::InvalidArgument(format!(
            "sdpa_full: bad GQA (H_q={}, H_kv={})",
            args.h_q, args.h_kv
        )));
    }
    let n_q = (args.q_len + BQ - 1) / BQ;
    let n_k = (args.k_len + BK - 1) / BK;
    let n_q_aligned = args.q_len / BQ;
    let n_k_aligned = args.k_len / BK;
    let params = AttnParams {
        b: args.batch,
        h: args.h_q,
        d: args.head_dim,
        q_l: args.q_len,
        k_l: args.k_len,
        gqa_factor: args.h_q / args.h_kv,
        scale: args.scale,
        n_q,
        n_k,
        n_q_aligned,
        n_k_aligned,
        q_l_rem: args.q_len - n_q_aligned * BQ,
        k_l_rem: args.k_len - n_k_aligned * BK,
        q_l_off: args.k_len - args.q_len,
        q_strides: args.q_strides,
        k_strides: args.k_strides,
        v_strides: args.v_strides,
        o_strides: args.o_strides,
    };

    let align_q = args.q_len % BQ == 0;
    let align_k = args.k_len % BK == 0;
    let pipeline = library.pipeline_specialized(
        "steel_attention_bfloat16_bq32_bk32_bd64_wm4_wn1_maskbfloat16",
        &[
            FunctionConstant::Bool { index: 200, value: align_q },
            FunctionConstant::Bool { index: 201, value: align_k },
            FunctionConstant::Bool { index: 300, value: false }, // has_mask
            FunctionConstant::Bool { index: 301, value: args.causal }, // do_causal
            FunctionConstant::Bool { index: 302, value: false }, // has_sinks
        ],
    )?;

    {
        let encoder = commands.encoder()?;
        encoder.setComputePipelineState(pipeline.metal_pipeline_state());
        // SAFETY: bindings match steel_attention's MSL signature
        // (steel_attention.h): q=0, k=1, v=2, o=3, AttnParams=4. The
        // mask params/buffer (5,6) and sinks (7) are gated behind
        // has_mask/has_sinks (pinned false), so absent from this
        // specialization — must not be bound.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(q.metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(k.metal_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(v.metal_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(out.metal_buffer()), 0, 3);
            let p: NonNull<c_void> = NonNull::from(&params).cast();
            encoder.setBytes_length_atIndex(p, std::mem::size_of::<AttnParams>(), 4);
        }
    }

    let grid = MTLSize { width: usize::try_from(n_q).unwrap(), height: usize::try_from(args.h_q).unwrap(), depth: usize::try_from(args.batch).unwrap() };
    let group = MTLSize { width: 32, height: usize::try_from(WM).unwrap(), depth: usize::try_from(WN).unwrap() };
    commands.dispatch_threadgroups(grid, group)
}
```

Add `impl SdpaFullArgs { fn gqa_factor(&self) -> i32 { if self.h_kv == 0 { 0 } else { self.h_q / self.h_kv } } }`. Add buffer-size bound checks mirroring `dispatch_sdpa_vector_bf16` (out len == H_q*qL*D; q/k/v large enough for max strided index) before dispatch — fail-loud, no on-GPU OOB.

- [ ] **Step 5: Run the parity test**

Run: `cargo test -p cider-press-kernels steel_attention_prefill_causal_matches_mlx -- --nocapture`
Expected: PASS within the combined bound (atol 0.03, rtol 0.08). If values are transposed/garbage, first suspect `AttnParams` field order or a stride passed in bytes instead of elements.

- [ ] **Step 6: Commit**

```bash
git add crates/cider-press-kernels/src/kernels/sdpa.rs crates/cider-press-kernels/tests/steel_attention_parity.rs
git commit -m "feat(kernels): dispatch_sdpa_full_bf16 (steel) + MLX prefill parity

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Runtime op routing by query length

**Files:**
- Modify: `crates/cider-press-runtime/src/tensor.rs` (relax `Tensor::sdpa` to `T_q>=1`)
- Modify: `crates/cider-press-runtime/src/eval.rs` (`dispatch_sdpa` routes by `T_q`; add `dispatch_sdpa_full`)
- Test: `crates/cider-press-runtime/tests/sdpa_full_routing.rs` (new)

- [ ] **Step 1: Write the failing routing test**

Create `crates/cider-press-runtime/tests/sdpa_full_routing.rs`. A T_q>1 causal `Tensor::sdpa` should build, eval, and produce the right output shape `[1, H_q, T, D]` (correctness vs MLX is covered at the model layer in Task 4; here we assert the op no longer rejects T_q>1 / causal).

```rust
#![cfg(target_os = "macos")]
use cider_press_runtime::{Device, Tensor};
use half::bf16;

#[test]
fn sdpa_accepts_prefill_causal_and_evals() {
    let (h_q, h_kv, t, d) = (14usize, 2usize, 39usize, 64usize);
    let device = Device::system_default().expect("device");
    let mk = |heads: usize| {
        let data = vec![bf16::from_f32(0.01); heads * t * d];
        Tensor::from_slice(&device, &data, [1, heads, t, d]).expect("t")
    };
    let (q, k, v) = (mk(h_q), mk(h_kv), mk(h_kv));
    let scale = 1.0f32 / (d as f32).sqrt();
    let out = Tensor::sdpa(&q, &k, &v, None, scale, h_q / h_kv, true).expect("build sdpa T>1");
    out.eval().expect("eval prefill sdpa");
    assert_eq!(out.shape().dims(), &[1, h_q, t, d]);
}
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `cargo test -p cider-press-runtime sdpa_accepts_prefill_causal_and_evals`
Expected: FAIL — `Tensor::sdpa` errors `"only decode T_q=1 is supported"`, or eval errors `"causal=true not yet supported (Plan B)"`.

- [ ] **Step 3: Relax the `Tensor::sdpa` shape guard**

In `crates/cider-press-runtime/src/tensor.rs`, replace the `qd[2] != 1` rejection (the block returning `"only decode T_q=1 is supported in the vector dispatch"`) with a positive-length check:

```rust
        if qd[2] == 0 {
            return Err(Error::InvalidArgument(
                "sdpa: query length T_q must be positive".into(),
            ));
        }
```

Leave every other guard (BF16, supported head dims, GQA, batch==1, k/v agreement) unchanged. The op kind, inputs, and output shape are already correct for T_q>1.

- [ ] **Step 4: Route `dispatch_sdpa` by query length**

In `crates/cider-press-runtime/src/eval.rs`, in `dispatch_sdpa`, REMOVE the two early `Plan B` rejects (`causal` and `inputs.len() > 3`). Read the query length in a SHORT-LIVED lock scope, then branch BEFORE the vector path re-locks — do **not** hold the `lock_inputs()` guard across the `dispatch_sdpa_full` call (it re-locks the same inputs → deadlock):

```rust
    // Scope the lock so it is released before either path re-locks.
    let t_q = {
        let inputs = op.lock_inputs();
        let q = inputs
            .first()
            .ok_or_else(|| Error::InvalidArgument("Sdpa: missing q (inputs[0])".into()))?;
        q.shape().dims()[2]
    };
    if t_q > 1 {
        return dispatch_sdpa_full(inner, op, commands, outputs, dst, index_of, scale, causal);
    }
    let inputs = op.lock_inputs(); // vector (decode) path re-locks
    // ---- existing vector (decode) path below, unchanged ----
```

(Keep the existing `k`/`v` binding and vector dispatch exactly as-is for the `t_q == 1` path. The early `head_dim`/`n_keys` reads in the existing code move below this re-lock.)

- [ ] **Step 5: Implement `dispatch_sdpa_full`**

Add to `eval.rs`, beside `dispatch_sdpa`. Resolve q/k/v storage + strides (mirror the vector path's `resolve_view_storage` + `layout_strides`), require D-contiguous and zero offset (fail-loud — this is the Stage-3 D-contiguity verification point), build dense output strides, and call the kernel.

```rust
#[allow(clippy::too_many_arguments)]
fn dispatch_sdpa_full(
    inner: &Arc<TensorInner>,
    op: &OpNode,
    commands: &mut Commands<'_>,
    outputs: &[Buffer<u8>],
    dst: &mut Buffer<u8>,
    index_of: &HashMap<*const TensorInner, usize>,
    scale: f32,
    causal: bool,
) -> Result<()> {
    let device = inner.device.as_ref().expect("op nodes carry a device");
    let inputs = op.lock_inputs();
    let q = inputs.first().expect("sdpa q");
    let k = inputs.get(1).expect("sdpa k");
    let v = inputs.get(2).expect("sdpa v");

    let qd = q.shape().dims();   // [1, H_q, qL, D]
    let kd = k.shape().dims();   // [1, H_kv, kL, D]
    let (h_q, q_len, head_dim) = (qd[1], qd[2], qd[3]);
    let (h_kv, k_len) = (kd[1], kd[2]);

    let (q_bytes, q_off) = resolve_view_storage(q, outputs, index_of)?;
    let (k_bytes, k_off) = resolve_view_storage(k, outputs, index_of)?;
    let (v_bytes, v_off) = resolve_view_storage(v, outputs, index_of)?;
    if q_off != 0 || k_off != 0 || v_off != 0 {
        return Err(Error::InvalidArgument(format!(
            "sdpa_full: non-zero view byte offset (q={q_off}, k={k_off}, v={v_off}) not supported"
        )));
    }
    let qs = layout_strides(q)?;
    let ks = layout_strides(k)?;
    let vs = layout_strides(v)?;
    if qs[3] != 1 || ks[3] != 1 || vs[3] != 1 {
        return Err(Error::InvalidArgument(format!(
            "sdpa_full: Q/K/V head dim must be contiguous (D stride 1); got q={}, k={}, v={}",
            qs[3], ks[3], vs[3]
        )));
    }
    let os = element_strides(&[1, h_q, q_len, head_dim]); // dense output

    let q_typed = unsafe { q_bytes.reinterpret_as::<half::bf16>() };
    let k_typed = unsafe { k_bytes.reinterpret_as::<half::bf16>() };
    let v_typed = unsafe { v_bytes.reinterpret_as::<half::bf16>() };
    let mut dst_typed = unsafe { dst.reinterpret_as::<half::bf16>() };

    let conv = |s: &[isize]| -> [i64; 3] { [s[0] as i64, s[1] as i64, s[2] as i64] };
    let library = device.steel_attention_library()?;
    cider_press_kernels::kernels::sdpa::dispatch_sdpa_full_bf16(
        commands,
        library,
        &q_typed,
        &k_typed,
        &v_typed,
        &mut dst_typed,
        cider_press_kernels::kernels::sdpa::SdpaFullArgs {
            batch: 1,
            h_q: i32::try_from(h_q).unwrap(),
            h_kv: i32::try_from(h_kv).unwrap(),
            head_dim: i32::try_from(head_dim).unwrap(),
            q_len: i32::try_from(q_len).unwrap(),
            k_len: i32::try_from(k_len).unwrap(),
            scale,
            q_strides: conv(&qs),
            k_strides: conv(&ks),
            v_strides: conv(&vs),
            o_strides: [os[0], os[1], os[2]],
            causal,
        },
    )
}
```

(If `os[i]` are `i64` already from `element_strides`, drop the cast. The `as i64` on `isize` strides is the existing pattern in this file — keep it consistent with `layout_strides`' return type.)

- [ ] **Step 6: Run the routing test + full kernels/runtime suites**

Run:
```bash
cargo test -p cider-press-runtime sdpa_accepts_prefill_causal_and_evals -- --nocapture
cargo test -p cider-press-runtime --release
```
Expected: PASS (the decode `sdpa` tests must still pass — the `t_q == 1` path is untouched).

- [ ] **Step 7: Commit**

```bash
git add crates/cider-press-runtime/src/tensor.rs crates/cider-press-runtime/src/eval.rs \
  crates/cider-press-runtime/tests/sdpa_full_routing.rs
git commit -m "feat(runtime): route Sdpa by T_q (vector decode / steel-full prefill)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Model integration (fused path live) + parity oracle

**Files:**
- Modify: `crates/cider-press-models/src/qwen2/attention.rs` (drop `is_decode` branch in `sdpa()`)
- Test: `crates/cider-press-models/tests/sdpa_prefill_fused.rs` (new — fused-vs-MLX + fused-vs-composed)

- [ ] **Step 1: Write the failing fused-path test (vs MLX, at the real T)**

Create `crates/cider-press-models/tests/sdpa_prefill_fused.rs`. Mirror `sdpa_qwen2.rs::qwen2_sdpa_prefill_causal_matches_mlx` but with no explicit mask passed to the model (the fused path uses `do_causal`), and a prefill length of 39. Compare the model's `attention::sdpa` output to the MLX causal fixture.

```rust
#![cfg(target_os = "macos")]
// Build q/k/v from a dump_mlx_op("sdpa", --causal) fixture at H_q=14,
// H_kv=2, D=64, T=39; call attention::sdpa(&q,&k,&v, None, &config) and
// compare to fixture "out" with the combined bound (atol 0.03, rtol 0.08).
```

Use the existing `sdpa_fixture` helper pattern from `sdpa_qwen2.rs` (copy `sdpa_fixture` + `assert_within_tolerance` or factor them into `cider-press-test-utils` if cleaner) with `seq_q = seq_kv = 39`, `causal = true`. Pass `None` as the mask to `attention::sdpa`.

- [ ] **Step 2: Run it to confirm it fails**

Run: `cargo test -p cider-press-models sdpa_prefill_fused -- --nocapture`
Expected: FAIL — the model still routes T>1 to `composed_sdpa` (which requires a mask for causal); with `None` mask it computes *non-causal* attention, so output mismatches the causal fixture.

- [ ] **Step 3: Drop the `is_decode` branch — always call `Tensor::sdpa`**

In `crates/cider-press-models/src/qwen2/attention.rs`, replace the body of `sdpa()` after the `t_q` computation. Remove the `is_decode` early-return AND the `composed_sdpa` tail; route everything through `Tensor::sdpa`, with `causal = t_q > 1`:

```rust
    let gqa_factor = h_q / h_kv;
    #[allow(clippy::cast_precision_loss)]
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    // Mirror MLX's single ScaledDotProductAttention primitive: eval routes
    // T_q==1 → sdpa_vector (decode), T_q>1 → steel sdpa_full (prefill,
    // causal). Self-attention prefill is causal via the kernel's do_causal
    // function constant — no materialized mask. An explicit mask arg is no
    // longer threaded (Qwen2 prefill is always causal self-attention).
    let _ = mask; // see Task 5: the mask param is removed once composed_sdpa is gone
    Ok(Tensor::sdpa(q, k, v, None, scale, gqa_factor, t_q > 1)?)
```

Keep `composed_sdpa` and `is_decode` in the file for now (dead-code-allow if the linter complains) — Task 5 deletes them after the oracle test below passes. If clippy `-D warnings` blocks on the now-unused `composed_sdpa`/`is_decode`, add `#[allow(dead_code)]` to both for this one commit.

- [ ] **Step 4: Add the fused-vs-composed oracle test (same inputs, both paths)**

In `sdpa_prefill_fused.rs`, add a test that calls BOTH the fused path (`Tensor::sdpa(..., causal=true)`) and the legacy `composed_sdpa` (via a thin `#[doc(hidden)]`/`pub(crate)`-exposed wrapper, or by building the explicit causal mask and calling the composed chain) on identical q/k/v, and asserts they agree within the combined bound. This is the known-good oracle that licenses deleting `composed_sdpa` in Task 5.

```rust
// fused = attention::sdpa(&q,&k,&v, None, &config)           // do_causal
// oracle = composed_sdpa(&q,&k,&v, Some(&causal_mask), &config) // additive -inf
// assert_combined_tolerance(&fused, &oracle, 0.03, 0.08);
```

If `composed_sdpa` is private, expose it for the test via `#[cfg(test)]`-gated `pub(crate)` re-export or a `#[doc(hidden)] pub fn` test shim in the qwen2 module.

- [ ] **Step 5: Run both tests**

Run: `cargo test -p cider-press-models sdpa_prefill_fused -- --nocapture`
Expected: PASS — fused matches MLX, and fused matches the composed oracle, both within (0.03, 0.08).

- [ ] **Step 6: Tighten the full-layer test tolerance**

In `crates/cider-press-models/tests/attention_layer0.rs`, the comment says to tighten when the fused SDPA lands. Lower `ATOL`/`RTOL` from the current loose `1e-1`/`5e-2` toward the SDPA combined bound (try `3e-2`/`8e-2` first). Run:
```bash
CIDER_QWEN_CHECKPOINT_PATH=<path> cargo test -p cider-press-models --release attention_layer0
```
Expected: PASS at the tighter bound. If it fails, back off to the loosest bound that passes and note the residual in a comment (the prefill sub-test exercises `affine_qmm_t` + the new fused kernel). If the checkpoint env var is unset the test self-skips — that is acceptable for this step; record that it was run locally with the checkpoint.

- [ ] **Step 7: Commit**

```bash
git add crates/cider-press-models/src/qwen2/attention.rs \
  crates/cider-press-models/tests/sdpa_prefill_fused.rs \
  crates/cider-press-models/tests/attention_layer0.rs
git commit -m "feat(models): route Qwen2 prefill attention through fused steel sdpa_full

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: Delete `composed_sdpa` (cleanup to MLX-mirroring shape)

**Files:**
- Modify: `crates/cider-press-models/src/qwen2/attention.rs`
- Modify: `crates/cider-press-models/tests/sdpa_qwen2.rs` (the prefill test now uses the fused path)

- [ ] **Step 1: Delete `composed_sdpa`, `is_decode`, and now-dead helpers**

Remove `composed_sdpa`, `is_decode`, and any helper used ONLY by the composed chain (e.g. `broadcast_kv_heads`) — grep first to confirm no other caller:

```bash
grep -rn "composed_sdpa\|is_decode\|broadcast_kv_heads" crates/cider-press-models/src
```

Simplify `sdpa()` to drop the `mask` parameter entirely (its only consumer was the composed chain). Update the single call site in `attention.rs::Attention::forward` (search `attention::sdpa(` / `self.sdpa(`) to drop the `mask`/`None` argument. Remove the now-unused `#[allow(dead_code)]` added in Task 4.

- [ ] **Step 2: Repoint the oracle test**

In `sdpa_prefill_fused.rs`, the fused-vs-composed oracle test (Task 4 Step 4) now references deleted code. Either delete that one test (the fused-vs-MLX test remains the parity guarantee) or convert its oracle to a tiny inline reference. Prefer deleting it — MLX is the authoritative oracle and the composed chain is gone. Keep the fused-vs-MLX test.

- [ ] **Step 3: Update `sdpa_qwen2.rs::qwen2_sdpa_prefill_causal_matches_mlx`**

That test passed an explicit mask to `attention::sdpa`. With the signature change, update it to pass no mask (fused `do_causal`). The fixture/tolerance stay the same. If the test is now redundant with `sdpa_prefill_fused.rs`, keep the shorter-T (8×8) version here as a fast smoke and the 39×39 one as the realistic case.

- [ ] **Step 4: Build clean (clippy -D warnings) + full model suite**

Run:
```bash
cargo clippy --all-targets --all-features -- -D warnings
cargo test -p cider-press-models --release
```
Expected: no warnings (no dead code), all tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/cider-press-models/src/qwen2/attention.rs crates/cider-press-models/tests/
git commit -m "refactor(models): delete composed_sdpa — fused steel path is the only prefill route

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: Measure + update perf docs

**Files:**
- Modify: `docs/QWEN_PERF.md`, `docs/ARCHITECTURE.md`, `CLAUDE.md`

- [ ] **Step 1: Verify end-to-end greedy decode parity (regression gate)**

Run the existing greedy parity gate (decode path is untouched; this confirms no collateral change):
```bash
cargo test --workspace --release
```
Expected: PASS, including the greedy-decode parity test.

- [ ] **Step 2: Measure synchronous prefill (warm, mean of 5)**

Run the synchronous prefill timer that produced the 10.19 ms baseline:
```bash
cargo run --release --features profiling -p cider-press -- bench --prefill-iters 5
```
(Confirm the exact flag name with `cargo run --release -p cider-press -- bench --help`; the baseline used `--prefill-iters` synchronous timing per `docs/QWEN_PERF.md`.) Record the warm mean and compare to baseline 10.19 ms and mlx 7.82 ms. Capture the GPU profile too if cheap:
```bash
cargo run --release --features profiling -p cider-press -- bench --gpu-profile-prefill
```
Expected: the `gpu.copy` dispatch count drops by ~9/layer and `gpu.matmul`/`gpu.softmax` attention-score shares collapse into one `steel_attention` dispatch.

- [ ] **Step 3: Update `docs/QWEN_PERF.md`**

In the "Prefill gap analysis" section, add a subsection recording: the post-fusion synchronous prefill time, the new gap vs mlx 7.82 ms, the before/after dispatch-count and GPU-share deltas (from Step 2), and the kernel used (`steel_attention_bfloat16_bq32_bk32_bd64_wm4_wn1`, non-nax, confirmed via the arch-gen-16 nax gate). Mark prefill lever #1 as DONE in the ranked backlog; note lever #2 (encode/dispatch residual) remains.

- [ ] **Step 4: Update `docs/ARCHITECTURE.md`**

In backlog item #9, mark the fused-prefill-attention sub-item as landed (date 2026-06-09) with the measured number; keep the encode/dispatch-residual item open.

- [ ] **Step 5: Update `CLAUDE.md`**

Update the "Current state" / perf summary lines: prefill now uses the fused steel `sdpa_full` kernel; record the new prefill figure and that the remaining prefill lever is the encode/dispatch residual (lever #2). Keep the decode figures unchanged.

- [ ] **Step 6: Commit**

```bash
git add docs/QWEN_PERF.md docs/ARCHITECTURE.md CLAUDE.md
git commit -m "docs(perf): fused prefill attention landed — measured prefill delta

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Notes for the implementer

- **Never push** without asking the user (project rule). Commits are fine; `git push` is a separate explicit authorization.
- **Tolerances:** bit-exact is NOT expected for SDPA (softmax/exp drift). Use the combined np.allclose bound `|a-b| <= atol + rtol*|b|` (atol 0.03, rtol 0.08), matching the existing prefill-causal SDPA test.
- **Kernel-name gotcha:** the steel host name uses `bfloat16` (no `_t`). If you hit `KernelNotFound`, grep the flattened `steel_attention_inlined.metal` in `target/**/OUT_DIR` for the real instantiated name.
- **MLX attribution:** every vendored file under `steel/attn/` MUST keep its MLX MIT header (CLAUDE.md hard rule).
- **D-contiguity (the one real risk):** Task 3 Step 5 fails loud if a Q/K/V view isn't D-contiguous. If a real Qwen2 prefill trips it, the fix is one contiguous-in-D copy of the offending view before `Tensor::sdpa` — still far cheaper than the deleted composed chain. Investigate the KvCache read-view layout before adding the copy.
- **Python is one-shot only** (uv + PEP 723) — `dump_mlx_op.py` is existing fixture tooling; don't add new long-lived Python infra.
</content>
</invoke>
