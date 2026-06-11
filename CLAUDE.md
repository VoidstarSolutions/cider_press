# CLAUDE.md

Guidance for Claude (and humans) working in this repo.

---

## Project context

`cider-press` is an inference-only tensor runtime and model loader for
Apple Silicon. Single-platform scope is deliberate: it lets us lean on
unified memory, ship hand-tuned Metal kernels, and skip the general-
purpose device abstraction that costs portable frameworks performance on
Mac.

The design hypothesis: most of the Apple-Silicon perf gap between
candle/burn and MLX is in the **graph/dispatch layer**, not the kernels.
MLX's `.metal` kernels are MIT-licensed and reusable — load them
verbatim, dispatch them from Rust, and wrap them in a lazy graph that
batches command buffers, and we capture most of MLX's perf in idiomatic
Rust.

Measurement backed the hypothesis, and the gap is now closed. The
vendored MLX kernels run bit-exact from Rust, and the dispatch-boundary
gap that started at ~10× has been worked down stage by stage — fused
`RMSNorm`, async decode pipelining, decode copy-elimination, and finally
the **encoder/sync-model port** (MLX's untracked buffers + fence chain +
concurrent encoders + graph-level barrier elision in Kahn-wave order).
Running Qwen2.5-0.5B-Instruct-4bit, decode now **edges past `mlx_lm` at
~599 vs ~568 tok/s, ~1.05×** after **qmv fast-path padding (A1)** — padding
the q/k/v/o decode projections to K=1024 so the vendored `qmv_fast` variant
fires (bit-exact via zero-scale/zero-bias pad groups; vendored kernels
untouched). The encoder/sync-model port before it had reached parity
(~561 vs ~564, ~1.00×). A2 (a cider-owned small-N `qmv` kernel for k/v) was
**measured and closed as a no-go**: a zero-cost-k/v ceiling experiment put the
hard upper bound at **+1.07%** (within run-to-run noise) — k/v qmv time is
hidden by the async decode pipeline, so it isn't worth owning a kernel. The
remaining levers are now prefill and memory, not kernel-side decode. On the
prefill side, the **fused steel `sdpa_full` attention** has now **landed**:
the composed prefill chain (per-layer GQA-broadcast/Kᵀ copies +
QKᵀ/scale/mask/softmax/·V) is deleted, replaced by one `sdpa` dispatch per
layer — synchronous prefill **10.19 → ~9.3 ms**, gap vs mlx **1.30× →
~1.19×** (decode unchanged). The realized ~0.86 ms win ≈ the eliminated
copy traffic (the attention compute moved *into* the kernel, not away).
The prefill **KV-cache-read copies** were then eliminated (strided cache
view fed straight to the kernel; `gpu.copy` 194 → 146; `is_decode` + the
`causal_mask` plumbing retired) but **measured a no-op** — those small
`[1,2,T,64]` copies are off the qmm-bound critical path, so the copy lever
looked closed on **compute share**. The **encode/dispatch trace (#2) then
ran** (profiling tooling: encode/wait split + `.gputrace` capture +
os_signpost, PR #38) and **reopened it**: the prefill gap (~9.32 vs mlx
~7.27 ms, ~1.28×) is **dispatch-count-induced GPU bubbles, not compute**.
cider issues **680 dispatches vs mlx's 507**, yet both replay to **~6.06 ms
of GPU compute** (qmm at parity) — so cider's GPU-wait (8.27 ms) − compute
(6.06) ≈ **~2.2 ms of bubbles** at command-buffer commit boundaries (~14
buffers vs ~10). The surplus is **146 copies + ~72 Q/K/V bias-adds** (46% of
dispatches, ~10% of compute) that mlx avoids via lazy strided views + fused
bias. So the **copy lever is REOPENED** — not for its compute, for its
dispatch count. Next prefill lever (follow-up branch): **eliminate the 146
copies** (stride-aware RoPE / cache-write / collapse-heads→o_proj) + **fold
Q/K/V bias into qmm** → ~460 dispatches, below mlx. (One direct check still
owed: confirm the GPU idle gaps land at the commit boundaries in the
timeline.) See `docs/QWEN_PERF.md` (§ "Prefill encode/dispatch trace —
findings") for the full numbers and `docs/ARCHITECTURE.md` for the backlog.

### Non-goals

- Training / autograd
- CUDA, ROCm, Vulkan, WebGPU
- Windows, Linux
- A general-purpose tensor library — this exists to run LLMs.

### Acknowledgments

Metal kernels and several design decisions derive from
[MLX](https://github.com/ml-explore/mlx) (MIT, © 2023 Apple Inc.). Any
file derived from MLX must retain the MIT copyright header.

---

## Repository layout

A Cargo workspace; each layer is its own member crate so the dispatch
surface, the runtime, and the model layer can be depended on (and
published) independently.

```text
cider-press/
├── kernels-mlx/                        # vendored MLX kernel sources (MIT)
│   └── mlx/backend/metal/kernels/      # full upstream path preserved
├── crates/
│   ├── cider-press-kernels/            # vendored MLX dispatch surface (build.rs flattens .metal)
│   ├── cider-press-runtime/            # lazy Tensor + graph + eval + KvCache + profiling
│   ├── cider-press-models/             # transformer architectures (Qwen2), tokenizer, generator
│   ├── cider-press-test-utils/         # dev-only: shared MLX-parity test helpers
│   └── cider-press/                    # facade + CLI (chat / bench)
├── scripts/                            # one-shot uv scripts (fixtures, MLX sync, perf)
└── docs/                               # ARCHITECTURE.md, QWEN_PERF.md, RUNTIME_DESIGN.md
```

Dependencies flow strictly upward: `cider-press-models` →
`cider-press-runtime` → `cider-press-kernels`. The facade crate
re-exports runtime + models. Workspace-level `[workspace.lints]` and
`[workspace.dependencies]` keep settings and version pins in one place;
each member uses `lints.workspace = true` and `dep.workspace = true`.

---

## Current state

Runs **Qwen2.5-0.5B-Instruct-4bit** end-to-end: load checkpoint →
tokenize → chat-template → greedy decode, streamed via the `cider-press`
CLI. The runtime is a lazy `Tensor` graph over vendored MLX Metal
kernels, with `eval()` as the synchronous dispatch boundary; the models
crate composes Qwen2 from runtime ops. Performance is measured and decode
now **leads `mlx_lm`** (~599 vs ~568 tok/s, ~1.05×) after qmv fast-path
padding (A1), built on the encoder/sync-model port that first reached
parity (~561 vs ~564, ~1.00×) (`docs/QWEN_PERF.md`). Prefill now runs the
**fused steel `sdpa_full` attention** kernel (the composed prefill chain is
deleted): synchronous prefill **~9.3 ms** (was 10.19), gap vs mlx **~1.19×**
(was 1.30×); decode unchanged.

**Remaining perf work is prefill- and memory-side**, not decode kernels: A2
(small-N `qmv` for k/v, backlog #4) was measured and **closed as a no-go**
(+1.07% ceiling, hidden by the decode pipeline — see `docs/QWEN_PERF.md`
§ "A2 measured — NO-GO"). The **encode/dispatch trace (#2) has now run**
(PR #38 tooling) and **localized the prefill gap to dispatch-count GPU
bubbles** (~2.2 ms; cider 680 dispatches vs mlx 507, both ~6.06 ms GPU
compute). This **reopened the prefill copy lever**: the next prefill work is
**eliminating the 146 copies** (stride-aware RoPE / cache-write /
collapse-heads→o_proj) + **folding Q/K/V bias into qmm** to drop ~218
dispatches below mlx's count (see `docs/QWEN_PERF.md` § "Prefill
encode/dispatch trace — findings"). Also still open: within-eval reuse for
RSS and `.metallib` precompile. `docs/ARCHITECTURE.md` holds the forward pass and
the prioritized backlog.

---

## Conventions

- **macOS / Apple Silicon only.** Each member crate's `src/lib.rs` has a
  `compile_error!` for non-macOS targets. Don't add cross-platform shims.
- **MLX attribution.** Any file with MLX-derived content (kernel source,
  embedded headers, a ported dispatch routine) must retain the MLX MIT
  copyright header.
- **Comments.** Default to none. Write one only when the *why* is
  non-obvious (a hidden invariant, an MSL quirk). Identifier names carry
  the *what*.

### Design notes (hard-won)

- **Compose at the models layer when `mlx.nn` does — but only when it
  actually does.** MLX's `unary.metal` has no `Silu`/`Gelu`; `mlx.nn`
  composes them in Python, so we mirror that: `nn::silu(x) = x *
  sigmoid(x)`, `nn::gelu` from `erf`. The test is what `mlx.nn` does, not
  a blanket "avoid fused kernels": `mlx.nn.RMSNorm` calls
  `mx.fast.rms_norm` (the fused `rms_norm.metal`), so `nn::rms_norm`
  delegates to the fused `Tensor::rms_norm` too — composing it would be a
  six-dispatch deviation *from* MLX, not alignment with it. Don't reach
  for a fused kernel MLX itself doesn't ship; do use the fused path where
  MLX has one.
- **Parity tolerances.** Bit-exact where achievable (pure data movers,
  qmv/qmm, per-row dequantize). For kernels carrying `exp`/`cos`/`sin`
  (sigmoid, softmax, RoPE), bf16-ULP tolerance (~0.005 abs / 0.01 rel) —
  `metal::exp`/`cos`/`sin` drift 1–2 bf16 ULPs across Apple-Silicon
  generations. For long composed chains (SDPA, full logits), the
  np.allclose combined bound `|a-b| ≤ atol + rtol*|b|`.
- **Kernel-name gotchas.** `instantiate_kernel` stringifies the C++ type
  token literally — bf16 kernels are named `..._bfloat16_t_...` (with
  `_t_`), not `..._bfloat16_...`. Some variants are selected by kernel
  *name* (softmax `block_` vs `looped_`, `_precise_`); others by
  `[[function_constant]]` specialization (RoPE). Don't assume one
  mechanism.
- **Metal JIT caching.** `newLibraryWithSource:` caches compiled
  libraries to disk cross-process: the first compile of a large flattened
  source is tens of seconds; subsequent processes are sub-100 ms. Cold
  first-token latency is a JIT artifact; a `.metallib` precompile is a
  shipping concern, not a dev-loop one.
- **KvCache** writes are eager host `memcpy`s into pre-allocated slabs
  (Apple Silicon shared storage — no Metal dispatch). Callers drop K/V
  views before the next `update` (aliasing contract, enforced at runtime).
- **Loader footgun.** `q_proj.bias` (singular, dense bf16, additive
  Linear bias) is distinct from `q_proj.biases` (plural, per-group
  quantization bias on the qmv ABI). Only Q/K/V have `.bias`; O has only
  `.biases`.
- **Metal dispatch model (the encoder/sync port, the decode lever that
  closed the `mlx_lm` gap).** Consecutive *serial* compute passes in one
  command buffer are **driver-merged** — splitting encoders alone is inert;
  you must commit separate command buffers (every ~50 ops, MLX's value) to
  pipeline the GPU. `updateFence` captures only commands encoded **before**
  it, so encoding it at encoder *open* is a silent no-op (the fence signals
  immediately, the next `waitForFence` waits on nothing) — encode it at
  *close*, before `endEncoding`. A global `memoryBarrier` **before every
  dispatch** on a concurrent encoder is *slower* than a serial encoder on
  real kernels (a sub-µs microbench inverts this — don't trust it); what
  pays is making barriers **rare** via graph-level hazard tracking (elide
  the barrier unless an op reads a buffer written since the last barrier)
  scheduled in **Kahn-wave** order so independent ops sit adjacent and
  overlap. With buffers hazard-**untracked** (no automatic seam analysis),
  cross-encoder/-token ordering rests entirely on the encoder **fence
  chain**. Escape hatches: `CIDER_PRESS_TRACKED_BUFFERS=1` (restore
  automatic tracking), `CIDER_PRESS_NO_BARRIER_ELISION=1` (barrier every
  dispatch) — both for bisecting sync bugs.

---

## Dev workflow

- Build: `cargo build`
- Test: `cargo test` (release for perf-sensitive tests: `cargo test --release`)
- Lints: `cargo clippy --all-targets --all-features -- -D warnings`
- Format: `cargo fmt`
- Docs: `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps`
- Profiling: build/run with `--features profiling` to enable the
  `cider_press_runtime::profile` spans (e.g. `cider-press bench`).

Pre-commit hooks run fmt, clippy, tests, and docs. Install once with
`pre-commit install`.

---

## Reference paths

- **Vendored MLX kernels (canonical, used at build time):**
  `kernels-mlx/mlx/backend/metal/kernels/`
- **Local MLX checkout (reference only):** sibling at `../mlx` — its
  `mlx/backend/metal/{device,kernels,matmul,quantized}.cpp` are the
  dispatch reference when porting new routines. Not needed for
  `cargo build`/`test`.

To bump vendored kernels:

```sh
MLX_DIR=../mlx ./scripts/sync_mlx_kernels.sh
# update the upstream-commit SHA in kernels-mlx/VENDORED.md, then:
cargo test --workspace --release    # qmv/qmm parity is the canary
```

`.github/workflows/mlx-sync.yml` does this weekly and opens a PR on drift.
