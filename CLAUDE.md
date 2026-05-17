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
MLX's `.metal` kernels are MIT-licensed and reusable. If we can load
them verbatim, dispatch them from Rust, and wrap them in a lazy graph
that fuses and batches command buffers, we capture most of MLX's perf
in idiomatic Rust.

That hypothesis is currently **unproven**. The entire project is gated
on the spike described below.

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

## Current state: development spike

The repo is in **Stage 0** of a staged spike. The public API surface is
intentionally empty (`src/lib.rs` exports nothing yet) until the spike
informs the design. Do **not** add `Device`, `Tensor`, op modules, or
error types ahead of the spike's findings — the spike doc below
explicitly forbids it.

### Spike goal

Prove that we can load an MLX `.metal` kernel source verbatim, compile
it at runtime, dispatch it from Rust against shared-memory buffers, and
get bit-identical results to the reference MLX Python implementation
on the same inputs.

If this works, the rest of the project is "just" scaling the pattern.
If it doesn't, we learn what the real obstacles are before investing in
framework design.

**Estimated effort:** 1–2 focused weekends.
**Out of scope for the spike:** the lazy graph, multiple ops, fusion,
the public Rust API, error types, anything ergonomic. Walking skeleton
only — target < 500 lines of Rust.

### Reference paths

- MLX checkout: `/Users/zacharyheylmun/dev/llm/mlx/`
- MLX Metal kernels: `mlx/backend/metal/kernels/`
- MLX Metal C++ dispatch reference: `mlx/backend/metal/{device,kernels,matmul}.cpp`
- cider-press repo: `/Users/zacharyheylmun/dev/llm/cider-press/`

### Staged plan

Each stage isolates one risk. Combining stages blinds you to which
thing broke — resist the urge to skip ahead.

#### Stage 0 — environment + dependencies (≈30 min) — **done**

Currently pinned in `Cargo.toml`:

```toml
[dependencies]
half = "=2.7.1"

[target.'cfg(target_os = "macos")'.dependencies]
objc2 = "=0.6.4"
objc2-foundation = "=0.3.2"
objc2-metal = "=0.3.2"

[dev-dependencies]
approx = "=0.5.1"
```

Notes from doing it:

- The `objc2-*` ecosystem has moved past the original suggestions
  (`0.5` → `0.6`, `0.2` → `0.3`); feature names shifted too.
- Default features on `objc2-metal 0.3.x` activate 60+ items and cover
  everything the spike needs without manual curation. Worth revisiting
  feature minimization once dispatch is proven, if binary size matters.
- `MTLCreateSystemDefaultDevice` requires `CoreGraphics` at link time.
  Linking it directly via `#[link(name = "CoreGraphics", kind = "framework")]`
  in the test/example file is lighter than pulling in `objc2-core-graphics`.

**Acceptance:** `cargo build` succeeds on aarch64-apple-darwin. ✓

#### Stage 1 — device + queue + trivial inline kernel (≈2 hours) — **done**

See `tests/stage1_add_one.rs`. Round-trip after warm-up measured at
~237 µs on the development machine; acceptance was < 5 ms.

Before touching MLX, prove the dispatch loop with a kernel typed inline:

```metal
#include <metal_stdlib>
using namespace metal;
kernel void add_one(
    device const float* x [[buffer(0)]],
    device float* y       [[buffer(1)]],
    uint i [[thread_position_in_grid]]
) { y[i] = x[i] + 1.0; }
```

Write one Rust function (test or example) that:

1. Gets `MTLCreateSystemDefaultDevice`.
2. Creates a command queue.
3. Compiles the source string via `newLibraryWithSource:options:error:`.
4. Looks up `add_one`, builds an `MTLComputePipelineState`.
5. Allocates two `MTLBuffer`s with `MTLResourceStorageModeShared`,
   length 1024 × 4 bytes.
6. Writes `0..1024` as f32 into buffer 0 via the host pointer
   (`contents()` — this is the unified-memory payoff; no
   `did_modify_range` on Apple Silicon).
7. Encodes the dispatch with threadgroup size 64, grid size 1024.
8. Commits, waits, reads buffer 1 via host pointer.
9. Asserts `out[i] == i as f32 + 1.0`.

**Risks isolated:** `objc2-metal` API shape, runtime shader compilation,
buffer storage modes, threadgroup math, command-buffer lifecycle,
ownership under `objc2`'s `Retained<T>`.

**Acceptance:** `cargo test --release` passes; round-trip < 5 ms.

#### Stage 2 — load a real MLX `.metal` file (≈2 hours) — **done**

See `build.rs` and `tests/stage2_copy.rs`. Both `v_copyfloat32float32`
and `v_copyfloat16float16` produce bit-identical output on 1024
elements. Findings:

- Offline-inlining approach worked. `build.rs` flattens `copy.metal`
  plus 6 transitive headers (1418 lines total) into one MSL string in
  `OUT_DIR`. Recursion handles `#pragma once` via a canonicalized-path
  seen-set; system `#include <...>` directives are left for the Metal
  compiler.
- Zero MSL dialect drift. The MLX `bf16` shim and the
  `metal_stdlib`/`metal_math`/`metal_logging` system headers coexist
  on the current macOS / Metal toolchain without modification. The
  `logging.h` `__METAL_VERSION__ >= 320` fallback never fires (system
  Metal is new enough).
- `instantiate_kernel` macro: produces `[[host_name(N)]] kernel ...`,
  so `newFunctionWithName:` lookup name is literally the first macro
  argument — no mangling. Big plus for Stage 4.
- `constant T&` MSL parameters auto-bind to the next free buffer slot
  after explicit `[[buffer(N)]]` attributes. Use
  `setBytes:length:atIndex:` for small scalars rather than allocating
  an `MTLBuffer`.
- JIT cost data: first run of `cargo test --test stage2_copy --release`
  takes ~1.5 s, dominated by `newLibraryWithSource:` for the full
  flattened source. Repeat runs of the same binary are sub-50 ms,
  hinting at driver-level library caching across processes (partially
  addresses open question 2; Stage 4 cold-run will confirm).

#### Stage 2 — original plan

Pick `mlx/backend/metal/kernels/copy.metal`. Small, minimal includes,
trivial reference output (identity), exercises the include resolver
without GEMM templating.

1. Embed `copy.metal` plus transitive headers. MLX uses
   `#include "mlx/backend/metal/kernels/utils.h"`-style paths; Metal's
   runtime compiler has no filesystem unless given one. Two viable
   approaches:
   - **Offline inline (recommended for spike):** a small `build.rs`
     recursively inlines `#include "..."` into a single self-contained
     `.metal` string, embedded as `&'static str`. Mirrors MLX's JIT path.
     No Xcode tooling dependency at build time.
   - **`.metallib` precompile:** `xcrun -sdk macosx metal` at build time.
     Lower runtime cost, more build-system work.
2. Compile via `newLibraryWithSource:`. Expect errors; iterate.
3. Dispatch the simplest copy variant on a 1D f32 input.

**Risks isolated:** include flattening; MSL dialect drift between MLX's
assumed compiler and the system Metal compiler; MLX macros
(`instantiate_kernel`, `bfloat`, etc.) needing definition or stubs.

**Acceptance:** copy kernel produces bit-identical output for f32 and
f16 (use `half::f16`).

#### Stage 3 — reference parity harness (≈1 hour) — **done**

See `scripts/gen_stage3_fixtures.py` (one-shot PEP 723 `uv run` script
— no requirements.txt, no venv), `tests/fixtures/stage3_copy.safetensors`
(12.5 KB, deterministic via seed=0), and `tests/stage3_parity.rs`. Both
f32 and f16 `v_copy` outputs match the MLX-generated reference at
1e-6 / 1e-3 tolerance respectively. Findings:

- `safetensors 0.7` is friction-free: `SafeTensors::deserialize(&bytes)`
  → `.tensor(name)?.data()` returns raw `&[u8]`, cast manually. No
  `bytemuck` needed for spike scope.
- Python is treated as a one-shot fixture generator, **not** project
  infrastructure. There is no long-term Python support intended; the
  script exists so Stage 4 can compare against real MLX output.
- Test harness duplicates `MetalCtx` + `run_copy` from Stage 2. Stage 4
  will be the third copy — factor to `tests/common/mod.rs` then, not
  before.
- MSRV trap: `usize::is_multiple_of` is Rust 1.87+ and our
  `rust-version = "1.85"` declares older. Clippy catches it. Use
  `% N == 0` or bump `rust-version`.

#### Stage 3 — original plan

```python
import mlx.core as mx
import numpy as np
np.random.seed(0)
# Save inputs + MLX output to .npy / .safetensors for Rust to load.
```

Rust test helper loads `.npy` (via `npyz`) or `.safetensors` (via
`safetensors`), runs the cider-press kernel, compares with
`approx::assert_relative_eq!` at dtype-appropriate tolerance: `1e-6` for
f32, `1e-3` for f16, `1e-2` for bf16/quantized.

**Acceptance:** the Stage 2 copy kernel passes the parity harness.

#### Stage 4 — target: one quantized matmul (≈1 weekend)

Pick **`qmv`** (quantized matrix-vector, batch=1) from
`mlx/backend/metal/kernels/quantized.{h,metal}`.

- `qmv` is the inference hot path — token generation is dominated by
  matvec against the quantized weight matrix.
- Simpler than `qmm`: no tiling across the batch dim, one
  warp/threadgroup per output row.
- Exercises the full quantized weight layout (packed int4 + per-group
  scales + biases), which is what we ultimately need.

Specifics:

1. **One template instantiation.** Target **bf16 × int4,
   group_size=64** — the dominant production config for Llama/Qwen
   4-bit quants. Instantiated name will be ~`qmv_bfloat16_gs64_b4`;
   grep `quantized.metal` for the exact `instantiate_kernel(qmv...)`
   lines to confirm.
2. **Reference inputs in Python.** Tiny MLX script:
   - Random `[N, K] = [4096, 4096]` bf16 weight,
   - `mx.quantize(w, group_size=64, bits=4)` → `(w_q, scales, biases)`,
   - `[K]` bf16 vector,
   - `y = mx.quantized_matmul(x, w_q, scales, biases, ...)`,
   - Save all five arrays as safetensors.
3. **Match MLX's weight memory layout exactly.** Most bugs will live
   here. Packed int4 layout, scale/bias interleaving,
   row-major-vs-column-major must match byte-for-byte. Read
   `mlx/backend/metal/quantized.cpp` for how MLX binds these buffers
   and copy the binding order verbatim.
4. **Dispatch parameters from MLX.** Search the matching call site in
   `quantized.cpp`. Copy threadgroup size, grid size, and buffer-index
   assignments exactly. Do not improvise.
5. **Run, compare.** Tolerance ≈ `1e-2` relative for bf16 × int4.
   Expect tighter agreement on most elements, with outliers near zero
   where relative error is unstable — fall back to absolute tolerance
   there.

**Risks isolated:** weight packing layout, scale/bias ordering, kernel
constants, row/column conventions.

**Acceptance:** `qmv_bfloat16_gs64_b4` matches MLX's `quantized_matmul`
to bf16 tolerance on `[4096, 4096] @ [4096]`.

#### Stage 5 — measure (≈1 hour)

- Time 1000 dispatches of the same `qmv` call on the same buffers.
- Report tok/s-equivalent: `1 / (time per dispatch)`.
- Compare against MLX Python on identical shapes (`mx.eval` + timer).

Within 10% of MLX → kernel-reuse hypothesis validated; perf gap is in
graph/dispatch as predicted.

2× slower or worse → something wrong with dispatch (threadgroup size,
function constants, buffer storage). Investigate before moving on.

**Acceptance:** numbers documented in `docs/SPIKE_RESULTS.md` (create
it). Do not skip — "produces correct output" is half the answer; the
spike exists to learn whether the approach works.

### Open questions to answer during the spike

Non-blocking, but record findings:

1. How does offline include-flattening handle headers including
   `<metal_stdlib>` vs MLX's own `bfloat` shim? Conflicts with system
   MSL on current macOS?
2. Does `newLibraryWithSource:` reliably cache compiled libraries across
   process restarts, or do we need to persist `.metallib` manually?
3. Measured wall-clock cost of `newLibraryWithSource:` for the full
   `quantized.metal`? (Determines JIT-at-startup viability vs
   precompilation.)
4. `bfloat`-related MSL version requirements that constrain minimum
   macOS?

### What NOT to do during the spike

- Don't design the public Rust API. Every type can be `pub(crate)` or a
  free function in a test/example file.
- Don't add a `Tensor` abstraction. Raw `MTLBuffer` + shape tuples are
  fine.
- Don't add error types beyond `Box<dyn Error>` or `anyhow`.
- Don't try to load a real model. Validate one kernel, not a forward
  pass.
- Don't port MLX C++ files. Read them for reference only.
- Don't optimize. Correctness + an honest perf number, not a tuned
  implementation.

### Findings (post-spike)

Once acceptance criteria are evaluated, append a Findings section here
(or split to `docs/SPIKE_RESULTS.md`) covering:

1. Did each stage's acceptance criterion pass? If not, why?
2. Surprises in the `objc2-metal` API or MLX kernel sources.
3. Measured `qmv` throughput vs MLX Python.
4. Concrete recommendation: proceed to lazy-graph design / proceed to
   eager-only design / abandon and contribute to candle instead.

### After the spike

If acceptance criteria pass:

1. Design the lazy-tensor graph + `eval()` semantics (separate planning
   doc).
2. Sketch the buffer pool / allocator. See
   `mlx/backend/metal/allocator.cpp` for reference design.
3. Next kernel: `scaled_dot_product_attention` — next inference hot
   path, very different dispatch pattern (per-head tiling, KV cache
   layout). Validate the same way.

If acceptance criteria fail:

1. Write up where it failed and what the obstacle was.
2. Reassess: keep going, switch to wrapping MLX via FFI (`mlx-c`), or
   contribute Metal kernels upstream to candle. All three are
   reasonable; the spike exists to inform the choice.

---

## Conventions

- **macOS / Apple Silicon only.** `src/lib.rs` has a `compile_error!`
  for non-macOS targets. Don't add cross-platform shims.
- **No premature abstractions.** Until the spike concludes, prefer
  `pub(crate)` free functions and raw `MTLBuffer` over types. The cost
  of building the wrong abstraction first is high.
- **MLX attribution.** Any file containing MLX-derived content (kernel
  source, embedded headers, port of a dispatch routine) must retain the
  MLX MIT copyright header at the top.
- **Comments.** Default to none. Write one only when the *why* is
  non-obvious (a hidden invariant, a workaround for a specific MSL
  quirk, etc.). Identifier names should carry the *what*.

## Dev workflow

- Build: `cargo build`
- Test: `cargo test` (release mode for any perf-sensitive test:
  `cargo test --release`)
- Lints: `cargo clippy --all-targets --all-features -- -D warnings`
- Format: `cargo fmt`
- Docs: `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps`

Pre-commit hooks run fmt, clippy, tests, and docs. Install once with
`pre-commit install` (requires the `pre-commit` tool).
