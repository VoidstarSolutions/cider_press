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

Measurement now backs the hypothesis. The vendored MLX kernels run
bit-exact from Rust, and the performance gap lives where predicted: at
the dispatch boundary. Running Qwen2.5-0.5B-Instruct-4bit, decode is
~10× slower than `mlx_lm`, with ~89% of each decode step in
`Tensor::eval` spread over ~49 synchronous `commit + wait` cycles per
token. Closing that gap (command-buffer batching) is the next work. See
`docs/QWEN_PERF.md` for the numbers and `docs/ARCHITECTURE.md` for the
forward backlog.

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
crate composes Qwen2 from runtime ops. The performance baseline is
measured (`docs/QWEN_PERF.md`).

**Next work is performance optimization** — the decode path is
dispatch-bound (see Project context). `docs/ARCHITECTURE.md` holds the
forward pass and the prioritized backlog.

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

- **Compose at the models layer when `mlx.nn` does.** MLX's `unary.metal`
  has no `Silu`/`Gelu`; `mlx.nn` composes them in Python, so we mirror
  that: `nn::silu(x) = x * sigmoid(x)`, `nn::gelu` from `erf`,
  `nn::rms_norm` from `mean(x²)`/`rsqrt`. Don't reach for a fused kernel
  MLX itself doesn't ship.
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
