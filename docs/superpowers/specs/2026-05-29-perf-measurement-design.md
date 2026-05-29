# Perf measurement + profiling facility — design

Date: 2026-05-29
Branch: `feat/perf-measurement`

## Purpose

Produce honest end-to-end performance numbers for Qwen2.5-0.5B-Instruct-4bit
running under `cider-press`, attribute decode-step time to its components,
and document the results so the *next* round of optimization work starts
from measurement rather than guesswork.

A general, reusable profiling facility is built as part of this work — it
outlives this effort and is documented as a first-class capability, not as
single-use instrumentation.

## Scope

**In scope — measure and document:**

- End-to-end tokens/sec on the real load → generate path, split into
  prefill and decode.
- Per-decode-step latency distribution attributed to forward-dispatch,
  argmax-round-trip, and KvCache-memcpy.
- Peak resident memory (RSS) sampled at pre-load, post-load, and
  post-decode marks.
- Side-by-side comparison against `mlx_lm.generate` (greedy) on the
  identical prompt and `max_tokens`.
- A general profiling facility (named spans, thread-local accumulation,
  cargo-feature-gated, zero-cost when disabled).
- Write-up in `docs/QWEN_PERF.md`.

**Explicitly NOT in scope** (deferred — these land *after* measurement
points at them, each as its own follow-up):

- Buffer pool / allocator.
- Cross-eval command-buffer batching (the spike's ~1.5× dispatch gap).
- GPU argmax.
- Batched quantized matmul (`qmm`) speedups for prefill.
- `.metallib` precompilation.

This branch produces the numbers that justify the above; it does not
begin them. Measure-only.

## Architecture

### A. Profiling facility (general, permanent)

A reusable instrumentation facility, **zero-cost when its cargo feature
is off**.

- **Feature gate:** a `profiling` cargo feature, off by default, declared
  on `cider-press-runtime` and propagated up through `cider-press-models`
  and the facade crate. With the feature off, the span API compiles to
  nothing — no branch, no allocation, no `Instant::now`. Compile-time
  no-op, not a runtime check.
- **Home:** lives in `cider-press-runtime` — the lowest crate that needs
  to instrument (`KvCache`, `Tensor::eval`) — and is re-exported upward.
- **Span API:** RAII guard, e.g. `let _span = profile::span("decode.forward");`
  records elapsed time on drop, plus an optional `profile::scope!(name, { … })`
  macro for block scoping. When the feature is off, `span()` returns a
  zero-sized guard with an empty drop.
- **Collector:** a **thread-local `Profiler`** accumulating per-name total
  duration and hit count. Resettable (`profile::reset()`) and drainable
  (`profile::drain()` → name → (total, count)). Thread-local rather than a
  threaded-through `&mut sink` because the forward path crosses crate
  boundaries deep into `Qwen2Model::forward → TransformerBlock → Attention
  → KvCache::update`; threading a collector would pollute every signature.
  The eval boundary is synchronous and single-threaded, so thread-local is
  sound.
- **Documentation:** module-level docs frame this as a general perf
  instrumentation tool. No references to roadmap branch numbering (the
  branch-plan references are being retired after this PR).

**Initial instrumented spans:** `prefill.forward`, `decode.forward`,
`decode.argmax`, `kvcache.update`. Adding more later is a one-line span
insertion — that is the point of the facility being general.

### B. `bench` subcommand on the CLI

`crates/cider-press/src/bin/cider-press.rs`. The current flat `Args`
becomes a clap subcommand enum:

- `chat` — the existing behavior (load, render chat template, greedy
  decode, streamed detokenization). Remains the primary/default path.
- `bench` — perf measurement. Args: `--checkpoint`, `--prompt` (with a
  built-in fixed default for the canonical run), `--max-tokens`,
  `--context-window`, `--warmup` (decode steps to discard before stats).

`bench` reuses the *exact* load → `Qwen2Model::from_weights` → `Generator`
path so the measurement reflects the real user-facing code. Output is a
human-readable report to stdout; those numbers are what get pasted into
`docs/QWEN_PERF.md`.

Bench numbers come from a `cargo run --features profiling -- bench …`
build. The shipped/default CLI stays free of timing code.

### C. Memory sampling

In-process via mach `task_info(TASK_VM_INFO)` reading `phys_footprint`.
Sampled at three marks: pre-load, post-load, post-decode. Self-contained
and reproducible from `cargo run`; needs a small mach FFI binding
(`libc` / `objc2`), no external tooling.

### D. MLX comparison harness

New `scripts/measure_qwen_mlx.py` (PEP 723 `uv run`, paralleling
`scripts/measure_stage5_mlx.py`). Runs `mlx_lm.load` +
`mlx_lm.generate(..., verbose=True)` greedy (temp=0.0) on the same prompt
and `max_tokens`; reports MLX prompt (prefill) tok/s, generation (decode)
tok/s, and `mx.get_peak_memory()`. Manual one-shot tool — the human runs
both sides and records the ratio. Not a regression gate (consistent with
the stage-5 manual-comparison precedent and the Python-is-one-shot rule).

## Data flow (bench run)

1. Sample RSS (pre-load). Load checkpoint — timed separately, reported,
   excluded from tok/s. Sample RSS (post-load).
2. Render the fixed prompt, encode → ids.
3. `profile::reset()`, run `generate(ids, max_tokens)`, drain the
   iterator fully.
4. Internally: one `prefill.forward` span, then N decode steps each
   emitting `decode.forward` / `decode.argmax` / `kvcache.update` spans.
5. Sample RSS (post-decode).
6. Report: prefill tok/s, decode tok/s, total wall time (load-excluded),
   per-span mean and share-of-step, decode-step p50/p99, three RSS marks.
   `--warmup` discards the first K decode steps before computing stats.

## `docs/QWEN_PERF.md` structure

- **Environment:** machine, macOS / Metal version, checkpoint + its
  pinned SHA, prompt, `max_tokens`, `warmup`.
- **Headline tok/s table:** cider-press prefill/decode vs MLX, with ratio
  (echoing the stage-5 table style).
- **Decode-step breakdown table:** forward / argmax / kvcache shares of a
  decode step.
- **Memory:** peak RSS load vs decode, vs MLX peak.
- **Hot spots & next levers:** prose tying observed shares back to the
  deferred optimizations (command-buffer batching, GPU argmax, buffer
  pool) so the next effort has a measurement-justified starting point.

## Testing

Measurement branch — the bar is "the harness runs and the facility is
sound," not numerical parity.

- **Profiling facility unit test (non-gated, feature on):** spans
  accumulate the expected names and counts; `reset` / `drain` semantics
  hold. Feature-off correctness is the compile-time no-op, covered by the
  default build compiling.
- **`bench` subcommand:** exercised manually against
  `CIDER_QWEN_CHECKPOINT_PATH`; no gated CI test (timings are not
  asserted). Reproduction steps recorded in `docs/QWEN_PERF.md`.
- **Lints:** `cargo clippy --all-targets --all-features` stays clean with
  the `profiling` feature on; the instrumented build is linted.

## Open items to settle in the implementation plan

- Exact span API surface: RAII `span()` only, or also a `scope!` macro.
- `Profiler` storage shape (e.g. `HashMap<&'static str, (Duration, u64)>`
  vs a small fixed vec) and drain ordering for stable report output.
- Whether the mach binding goes through `libc` or a tiny hand-rolled
  `extern "C"` block, and where it lives (runtime crate vs the CLI bin).
- Canonical fixed prompt + `max_tokens` defaults for the headline run.
