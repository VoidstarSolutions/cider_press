# Perf Measurement + Profiling Facility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce honest end-to-end tokens/sec, decode-step attribution, and peak-RSS numbers for Qwen2.5-0.5B-Instruct-4bit under `cider-press`, vs `mlx_lm`, documented in `docs/QWEN_PERF.md`.

**Architecture:** A general, cargo-feature-gated profiling facility (named RAII spans → thread-local accumulator, compile-time no-op when off) lives in `cider-press-runtime` and is re-exported upward. The runtime (`eval`, `KvCache`) and the model-layer `Generator` drop spans at the suspected hot spots. A new CLI `bench` subcommand reuses the real load→generate path, samples RSS in-process via mach `task_info`, and prints a report. A companion `uv` script measures the same prompt under `mlx_lm`.

**Tech Stack:** Rust (edition 2024, MSRV 1.85), `objc2-metal`, mach `task_info` FFI, clap subcommands, PEP 723 `uv` for the MLX side.

---

## File Structure

**Create:**
- `crates/cider-press-runtime/src/profile.rs` — the profiling facility (span API, thread-local `Profiler`, feature-gated impl).
- `crates/cider-press-runtime/tests/profiling.rs` — facility unit test (feature-gated).
- `crates/cider-press/src/sys.rs` — in-process RSS sampling via mach `task_info`.
- `scripts/measure_qwen_mlx.py` — `mlx_lm` greedy throughput + peak-memory companion.
- `docs/QWEN_PERF.md` — the deliverable write-up.

**Modify:**
- `crates/cider-press-runtime/Cargo.toml` — add `[features] profiling`.
- `crates/cider-press-runtime/src/lib.rs` — `pub mod profile;` + re-export.
- `crates/cider-press-runtime/src/eval.rs` — `tensor.eval` span.
- `crates/cider-press-runtime/src/kv_cache.rs` — `kvcache.update` + `kvcache.memcpy` spans.
- `crates/cider-press-models/Cargo.toml` — add `[features] profiling`.
- `crates/cider-press-models/src/generator.rs` — `argmax` span.
- `crates/cider-press/Cargo.toml` — add `[features] profiling`.
- `crates/cider-press/src/lib.rs` — `pub mod sys;`.
- `crates/cider-press/src/bin/cider-press.rs` — refactor to `chat`/`bench` subcommands.
- `docs/QWEN_PATH.md`, `CLAUDE.md` — mark branch 15 done (docs-alongside convention).

---

## Task 1: Profiling facility in the runtime crate

**Files:**
- Create: `crates/cider-press-runtime/src/profile.rs`
- Create: `crates/cider-press-runtime/tests/profiling.rs`
- Modify: `crates/cider-press-runtime/Cargo.toml`
- Modify: `crates/cider-press-runtime/src/lib.rs`

- [ ] **Step 1: Add the `profiling` feature to the runtime manifest**

In `crates/cider-press-runtime/Cargo.toml`, add a `[features]` section immediately after the `[lib]` line:

```toml
[lib]

[features]
# General perf-instrumentation. Off by default; when off, the `profile`
# span API compiles to a zero-sized no-op (no branch, no timer). Turn on
# to make `profile::span` accumulate into a thread-local profiler.
profiling = []
```

- [ ] **Step 2: Write the profiling module**

Create `crates/cider-press-runtime/src/profile.rs`:

```rust
//! General performance-instrumentation facility.
//!
//! Drop a named span at any point you want to time:
//!
//! ```ignore
//! let _span = cider_press_runtime::profile::span("decode.argmax");
//! // ... work ...
//! // span records elapsed time on drop, keyed by name.
//! ```
//!
//! When the `profiling` cargo feature is **off** (the default), [`span`]
//! returns a zero-sized guard whose drop is empty, so instrumentation
//! compiles away to nothing — no branch, no timer, no allocation. When
//! the feature is **on**, each guard records its lifetime into a
//! thread-local accumulator keyed by the span name; totals and hit
//! counts are read with [`drain`] and cleared with [`reset`].
//!
//! Accumulation is thread-local rather than threaded through call
//! signatures because instrumented work (graph eval, KV-cache writes,
//! sampling) crosses crate boundaries deep into the model layer;
//! threading a collector would pollute every signature. The eval
//! boundary is synchronous and single-threaded, so thread-local is
//! sound for the cases instrumented here. A caller that wants spans
//! recorded on a worker thread reads [`drain`] on that same thread.

#[cfg(feature = "profiling")]
mod imp {
    use std::cell::RefCell;
    use std::time::{Duration, Instant};

    thread_local! {
        static PROFILER: RefCell<Vec<(&'static str, Duration, u64)>> =
            const { RefCell::new(Vec::new()) };
    }

    /// RAII timing guard; records elapsed time on drop under its name.
    #[must_use = "binding the span to `_` drops it immediately, timing nothing"]
    pub struct Span {
        name: &'static str,
        start: Instant,
    }

    impl Drop for Span {
        fn drop(&mut self) {
            let elapsed = self.start.elapsed();
            PROFILER.with(|p| {
                let mut v = p.borrow_mut();
                if let Some(entry) = v.iter_mut().find(|(n, _, _)| *n == self.name) {
                    entry.1 += elapsed;
                    entry.2 += 1;
                } else {
                    v.push((self.name, elapsed, 1));
                }
            });
        }
    }

    /// Start timing a named span. The returned guard records on drop.
    #[must_use = "binding the span to `_` drops it immediately, timing nothing"]
    pub fn span(name: &'static str) -> Span {
        Span {
            name,
            start: Instant::now(),
        }
    }

    /// Clear all accumulated spans on the current thread.
    pub fn reset() {
        PROFILER.with(|p| p.borrow_mut().clear());
    }

    /// Take and clear the accumulated spans, sorted by name:
    /// `(name, total_elapsed, hit_count)`.
    #[must_use]
    pub fn drain() -> Vec<(&'static str, Duration, u64)> {
        PROFILER.with(|p| {
            let mut out = std::mem::take(&mut *p.borrow_mut());
            out.sort_by(|a, b| a.0.cmp(b.0));
            out
        })
    }

    /// Whether the `profiling` feature was compiled in.
    #[must_use]
    pub const fn is_enabled() -> bool {
        true
    }
}

#[cfg(not(feature = "profiling"))]
mod imp {
    use std::time::Duration;

    /// Zero-sized no-op guard (feature off).
    pub struct Span;

    #[inline(always)]
    #[must_use]
    pub fn span(_name: &'static str) -> Span {
        Span
    }

    #[inline(always)]
    pub fn reset() {}

    #[inline(always)]
    #[must_use]
    pub fn drain() -> Vec<(&'static str, Duration, u64)> {
        Vec::new()
    }

    /// Whether the `profiling` feature was compiled in.
    #[inline(always)]
    #[must_use]
    pub const fn is_enabled() -> bool {
        false
    }
}

pub use imp::{Span, drain, is_enabled, reset, span};
```

- [ ] **Step 3: Export the module from the runtime crate root**

In `crates/cider-press-runtime/src/lib.rs`, add `mod`/`pub` lines. After the existing `mod` block (the `mod device;` … `mod tensor;` list, around line 38-48), add:

```rust
pub mod profile;
```

Place it alphabetically among the `mod` declarations (after `mod layout;`, before `mod quantization;` is fine — it is `pub mod` so order is cosmetic). No `pub use` line is needed; callers reach it as `cider_press_runtime::profile::span`.

- [ ] **Step 4: Write the facility unit test**

Create `crates/cider-press-runtime/tests/profiling.rs`:

```rust
//! Unit coverage for the `profile` facility. Gated on the `profiling`
//! feature: with the feature off, `drain` always returns empty (the
//! whole point), so the assertions only make sense with it on. Run:
//! `cargo test -p cider-press-runtime --features profiling --test profiling`.
#![cfg(feature = "profiling")]

use cider_press_runtime::profile;

#[test]
fn accumulates_named_spans_with_hit_counts() {
    profile::reset();
    {
        let _a = profile::span("alpha");
        let _b = profile::span("beta");
    }
    {
        let _a = profile::span("alpha");
    }

    let stats = profile::drain();
    let alpha = stats
        .iter()
        .find(|(n, _, _)| *n == "alpha")
        .expect("alpha recorded");
    let beta = stats
        .iter()
        .find(|(n, _, _)| *n == "beta")
        .expect("beta recorded");
    assert_eq!(alpha.2, 2, "alpha hit twice");
    assert_eq!(beta.2, 1, "beta hit once");

    // drain cleared the accumulator.
    assert!(profile::drain().is_empty(), "drain leaves the profiler empty");
}

#[test]
fn drain_is_sorted_by_name() {
    profile::reset();
    let _z = profile::span("zeta");
    drop(_z);
    let _a = profile::span("alpha");
    drop(_a);

    let stats = profile::drain();
    let names: Vec<&str> = stats.iter().map(|(n, _, _)| *n).collect();
    assert_eq!(names, ["alpha", "zeta"]);
}
```

- [ ] **Step 5: Verify the feature-off build compiles (no-op path)**

Run: `cargo build -p cider-press-runtime`
Expected: builds clean. The `#[cfg(not(feature = "profiling"))]` impl is what compiles here.

- [ ] **Step 6: Verify the test passes with the feature on**

Run: `cargo test -p cider-press-runtime --features profiling --test profiling`
Expected: both tests PASS (`accumulates_named_spans_with_hit_counts`, `drain_is_sorted_by_name`).

- [ ] **Step 7: Verify clippy is clean both ways**

Run: `cargo clippy -p cider-press-runtime --all-targets -- -D warnings`
Then: `cargo clippy -p cider-press-runtime --all-targets --features profiling -- -D warnings`
Expected: no warnings either way.

- [ ] **Step 8: Commit**

```bash
git add crates/cider-press-runtime/Cargo.toml crates/cider-press-runtime/src/profile.rs crates/cider-press-runtime/src/lib.rs crates/cider-press-runtime/tests/profiling.rs
git commit -m "feat(runtime): feature-gated profiling facility

General named-span timing facility: RAII guard -> thread-local
accumulator, compiled to a zero-sized no-op when the profiling
feature is off. Drained as (name, total, count), sorted by name.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Instrument the runtime hot spots

**Files:**
- Modify: `crates/cider-press-runtime/src/eval.rs:28`
- Modify: `crates/cider-press-runtime/src/kv_cache.rs:183` (the `update` method)

- [ ] **Step 1: Add the `tensor.eval` span**

In `crates/cider-press-runtime/src/eval.rs`, at the very top of `pub(crate) fn eval(root: &Tensor) -> Result<()>` (currently line 28), insert the span as the first statement so it covers the whole batch dispatch including `commit_and_wait`:

```rust
pub(crate) fn eval(root: &Tensor) -> Result<()> {
    let _span = crate::profile::span("tensor.eval");
    // Step 1: topological order of unevaluated op nodes reachable from
```

(Leave the rest of the function unchanged. `eval` is not re-entrant — the per-op dispatcher never calls `Tensor::eval` — so this span never nests within itself.)

- [ ] **Step 2: Add the `kvcache.update` and `kvcache.memcpy` spans**

In `crates/cider-press-runtime/src/kv_cache.rs`, in `pub fn update(&mut self, k: &Tensor, v: &Tensor) -> Result<()>` (currently line 183), add an outer span as the first statement of the body, and wrap the `unsafe` host-write block in an inner span.

First statement of `update` (immediately after the function's opening brace, before the aliasing-contract check comment):

```rust
    pub fn update(&mut self, k: &Tensor, v: &Tensor) -> Result<()> {
        let _span = crate::profile::span("kvcache.update");
        // See the module-level "Aliasing contract" doc for why this
```

Then wrap the existing `unsafe { … }` block (currently lines 238-241) in a nested scope with its own span:

```rust
        {
            let _memcpy = crate::profile::span("kvcache.memcpy");
            // SAFETY: no GPU dispatch is reading the slab during update
            // (callers drop any prior keys_view/values_view tensors before
            // calling update — documented contract). Host writes into
            // shared-storage buffers on Apple Silicon are immediately
            // visible to subsequent GPU dispatches that bind the same
            // buffer.
            unsafe {
                self.keys.as_mut_slice()[offset..offset + write_bytes].copy_from_slice(k_bytes);
                self.values.as_mut_slice()[offset..offset + write_bytes].copy_from_slice(v_bytes);
            }
        }
        self.position += step_t;
        Ok(())
    }
```

(The two `k.eval()?` / `v.eval()?` calls earlier in `update` are intentionally left outside `kvcache.memcpy`; their cost lands in `tensor.eval`. So `kvcache.update` ⊇ 2×eval + `kvcache.memcpy`, an overlap the report explains.)

- [ ] **Step 3: Verify the default build still compiles (spans are no-ops)**

Run: `cargo build -p cider-press-runtime`
Expected: builds clean.

- [ ] **Step 4: Verify the runtime test suite still passes**

Run: `cargo test -p cider-press-runtime`
Expected: existing kv_cache / eval tests PASS (the spans are inert no-ops in this build).

- [ ] **Step 5: Verify clippy clean with the feature on**

Run: `cargo clippy -p cider-press-runtime --all-targets --features profiling -- -D warnings`
Expected: no warnings.

- [ ] **Step 6: Commit**

```bash
git add crates/cider-press-runtime/src/eval.rs crates/cider-press-runtime/src/kv_cache.rs
git commit -m "feat(runtime): instrument eval + KvCache with profiling spans

tensor.eval wraps the whole batch dispatch+wait; kvcache.update wraps
the KV append, with an inner kvcache.memcpy isolating the unified-memory
host write from the input evals it triggers.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Instrument the Generator argmax + propagate the feature to models

**Files:**
- Modify: `crates/cider-press-models/Cargo.toml`
- Modify: `crates/cider-press-models/src/generator.rs:214` (`argmax_last_position`)

- [ ] **Step 1: Add the `profiling` feature to the models manifest**

In `crates/cider-press-models/Cargo.toml`, add a `[features]` section immediately after the `[lib]` line:

```toml
[lib]

[features]
# Enables the runtime profiling facility through this crate so model-layer
# spans (e.g. argmax) record when the crate is built with --features profiling.
profiling = ["cider-press-runtime/profiling"]
```

- [ ] **Step 2: Add the `argmax` span around the round-trip**

In `crates/cider-press-models/src/generator.rs`, in `fn argmax_last_position(logits: &Tensor, vocab: usize) -> Result<u32>` (currently line 214), add a span covering the CPU round-trip — the `cpu_to_vec` copy plus the scan — but **after** `logits.eval()?` so the GPU/dispatch cost stays attributed to `tensor.eval` rather than to argmax:

```rust
fn argmax_last_position(logits: &Tensor, vocab: usize) -> Result<u32> {
    logits.eval()?;
    let _span = cider_press_runtime::profile::span("argmax");
    let elements: Vec<bf16> = logits.cpu_to_vec().ok_or_else(|| {
```

(Everything from `let elements …` onward is unchanged; the span guard drops at function return, covering the copy and the fold.)

- [ ] **Step 3: Verify the default build compiles**

Run: `cargo build -p cider-press-models`
Expected: builds clean (the span is a no-op; `cider_press_runtime::profile` exists regardless of feature).

- [ ] **Step 4: Verify models tests still pass**

Run: `cargo test -p cider-press-models --lib`
Expected: existing unit tests PASS.

- [ ] **Step 5: Verify the feature-on build + clippy**

Run: `cargo clippy -p cider-press-models --all-targets --features profiling -- -D warnings`
Expected: no warnings; confirms the feature chain to `cider-press-runtime/profiling` resolves.

- [ ] **Step 6: Commit**

```bash
git add crates/cider-press-models/Cargo.toml crates/cider-press-models/src/generator.rs
git commit -m "feat(models): argmax profiling span + profiling feature passthrough

Wraps the post-eval cpu_to_vec + scan in argmax_last_position with an
'argmax' span (eval cost stays in tensor.eval). Adds a profiling feature
forwarding to cider-press-runtime/profiling.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: In-process RSS sampling (facade `sys` module)

**Files:**
- Create: `crates/cider-press/src/sys.rs`
- Modify: `crates/cider-press/src/lib.rs`

- [ ] **Step 1: Write the mach `task_info` sampling module**

Create `crates/cider-press/src/sys.rs`:

```rust
//! In-process resident-memory sampling via mach `task_info`.
//!
//! Queries `MACH_TASK_BASIC_INFO` on the current task. `resident_size`
//! is the current RSS; `resident_size_max` is the process-lifetime peak
//! RSS (a high-water mark that never decreases). Both are in bytes.
//!
//! The mach symbols live in libSystem, which is always linked on macOS,
//! so no `#[link]` directive is needed. `MACH_TASK_BASIC_INFO` (flavor
//! 20) maps to a stable, fixed-layout struct — preferred here over
//! `TASK_VM_INFO`'s `phys_footprint`, whose field offset is brittle.

use std::mem;

const MACH_TASK_BASIC_INFO: u32 = 20;

/// Mirrors `mach_task_basic_info` (`<mach/task_info.h>`). `mach_vm_size_t`
/// is `u64`; `time_value_t` is two `integer_t` (`i32`); `policy_t` and
/// `integer_t` are `i32`. Size is 48 bytes = 12 `natural_t` (`u32`) words,
/// which is `MACH_TASK_BASIC_INFO_COUNT`.
#[repr(C)]
struct MachTaskBasicInfo {
    virtual_size: u64,
    resident_size: u64,
    resident_size_max: u64,
    user_time: [i32; 2],
    system_time: [i32; 2],
    policy: i32,
    suspend_count: i32,
}

unsafe extern "C" {
    // The `mach_task_self()` C macro expands to this global mach_port_t.
    static mach_task_self_: u32;
    fn task_info(
        target_task: u32,
        flavor: u32,
        task_info_out: *mut i32,
        task_info_out_cnt: *mut u32,
    ) -> i32;
}

fn query() -> Option<MachTaskBasicInfo> {
    let mut info = mem::MaybeUninit::<MachTaskBasicInfo>::uninit();
    #[allow(clippy::cast_possible_truncation)]
    let mut count = (mem::size_of::<MachTaskBasicInfo>() / mem::size_of::<u32>()) as u32;
    // SAFETY: task_info writes exactly `count` natural_t words into the
    // buffer; `count` is sized from the struct, and the struct layout
    // matches MACH_TASK_BASIC_INFO. On KERN_SUCCESS (0) the struct is
    // fully initialized.
    let kr = unsafe {
        task_info(
            mach_task_self_,
            MACH_TASK_BASIC_INFO,
            info.as_mut_ptr().cast::<i32>(),
            &mut count,
        )
    };
    if kr == 0 {
        // SAFETY: KERN_SUCCESS means task_info initialized the struct.
        Some(unsafe { info.assume_init() })
    } else {
        None
    }
}

/// Current resident set size in bytes, or `None` if the mach query fails.
#[must_use]
pub fn resident_bytes() -> Option<u64> {
    query().map(|i| i.resident_size)
}

/// Process-lifetime peak resident set size in bytes, or `None` on failure.
#[must_use]
pub fn peak_resident_bytes() -> Option<u64> {
    query().map(|i| i.resident_size_max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn samples_nonzero_and_peak_at_least_current() {
        let current = resident_bytes().expect("resident_bytes query");
        let peak = peak_resident_bytes().expect("peak_resident_bytes query");
        assert!(current > 0, "current RSS should be positive");
        assert!(peak >= current, "peak RSS should be >= current RSS");
    }
}
```

- [ ] **Step 2: Export the module from the facade crate root**

In `crates/cider-press/src/lib.rs`, after the closing of the `compile_error!` block (after line ~24), add:

```rust
pub mod sys;
```

- [ ] **Step 3: Verify build + unit test**

Run: `cargo test -p cider-press --lib sys`
Expected: `samples_nonzero_and_peak_at_least_current` PASS.

- [ ] **Step 4: Verify clippy clean**

Run: `cargo clippy -p cider-press --all-targets -- -D warnings`
Expected: no warnings.

- [ ] **Step 5: Commit**

```bash
git add crates/cider-press/src/sys.rs crates/cider-press/src/lib.rs
git commit -m "feat(cli): in-process RSS sampling via mach task_info

sys::resident_bytes / peak_resident_bytes read MACH_TASK_BASIC_INFO on
the current task. Stable fixed-layout struct, libSystem symbols, no
external tooling.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: `bench` subcommand on the CLI

**Files:**
- Modify: `crates/cider-press/Cargo.toml`
- Modify: `crates/cider-press/src/bin/cider-press.rs` (full rewrite of the arg surface + `main`/`run`)

- [ ] **Step 1: Add the `profiling` feature to the facade manifest**

In `crates/cider-press/Cargo.toml`, add a `[features]` section after the `[lints]` block (before `[[bin]]` is fine):

```toml
[features]
# Compiles the profiling facility into the CLI so `bench` can report a
# per-span breakdown. Forwards to both lower crates so their spans record.
profiling = ["cider-press-models/profiling", "cider-press-runtime/profiling"]
```

- [ ] **Step 2: Rewrite the CLI binary with `chat`/`bench` subcommands**

Replace the entire contents of `crates/cider-press/src/bin/cider-press.rs` with:

```rust
//! `cider-press` CLI.
//!
//! Two subcommands:
//! - `chat`  — load a Qwen2 checkpoint, render the prompt through its
//!   chat template, greedy-decode with streamed detokenization.
//! - `bench` — same load + greedy-decode path, but measure prefill and
//!   decode throughput, peak RSS, and (with `--features profiling`) a
//!   per-span time breakdown; print a report instead of the text.

use std::collections::HashSet;
use std::io::{self, Write};
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use clap::{Args, Parser, Subcommand};

use cider_press_models::chat_template::{ChatTemplate, Message};
use cider_press_models::generator::Generator;
use cider_press_models::qwen2::{Qwen2Config, Qwen2Model, load_qwen2_weights};
use cider_press_models::tokenizer::Tokenizer;
use cider_press_runtime::{Device, profile};
use safetensors::SafeTensors;

type BoxError = Box<dyn std::error::Error + Send + Sync>;

const DEFAULT_BENCH_PROMPT: &str = "Write a short paragraph about the city of Seattle.";

#[derive(Parser, Debug)]
#[command(name = "cider-press", version, about = "Cold-pressed LLM inference for Apple Silicon")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Greedy decode + streamed detokenization.
    Chat(ChatArgs),
    /// Benchmark prefill/decode throughput + peak RSS; print a report.
    Bench(BenchArgs),
}

#[derive(Args, Debug)]
struct ChatArgs {
    /// Checkpoint directory holding config.json, tokenizer.json,
    /// `tokenizer_config.json`, and a single-file model.safetensors.
    #[arg(long)]
    checkpoint: PathBuf,
    /// User message.
    #[arg(long)]
    prompt: String,
    /// Optional system message; omitted means no system turn.
    #[arg(long, conflicts_with = "no_chat_template")]
    system: Option<String>,
    /// Maximum new tokens to generate.
    #[arg(long, default_value_t = 256)]
    max_tokens: usize,
    /// Pre-allocated `KvCache` window in tokens.
    #[arg(long, default_value_t = 4096)]
    context_window: usize,
    /// Skip `ChatTemplate` rendering; encode the bare --prompt.
    /// `tokenizer_config.json` is still read for the EOS token ids.
    #[arg(long)]
    no_chat_template: bool,
}

#[derive(Args, Debug)]
struct BenchArgs {
    /// Checkpoint directory (same layout as `chat`).
    #[arg(long)]
    checkpoint: PathBuf,
    /// Prompt to benchmark; defaults to a fixed paragraph request.
    #[arg(long, default_value_t = DEFAULT_BENCH_PROMPT.to_owned())]
    prompt: String,
    /// Optional system message.
    #[arg(long, conflicts_with = "no_chat_template")]
    system: Option<String>,
    /// New tokens to generate (the decode workload).
    #[arg(long, default_value_t = 128)]
    max_tokens: usize,
    /// Pre-allocated `KvCache` window in tokens.
    #[arg(long, default_value_t = 4096)]
    context_window: usize,
    /// Decode steps to discard before timing (warm caches / JIT).
    #[arg(long, default_value_t = 8)]
    warmup: usize,
    /// Skip `ChatTemplate` rendering; encode the bare --prompt.
    #[arg(long)]
    no_chat_template: bool,
}

/// Shared load result for both subcommands.
struct Loaded {
    model: Qwen2Model,
    tokenizer: Tokenizer,
    eos_ids: HashSet<u32>,
    chat_template: ChatTemplate,
}

fn load(checkpoint: &PathBuf, device: &Device) -> Result<Loaded, BoxError> {
    let config_bytes = std::fs::read(checkpoint.join("config.json"))?;
    let config = Qwen2Config::from_json_bytes(&config_bytes)?;

    let archive_bytes = std::fs::read(checkpoint.join("model.safetensors"))?;
    let archive = SafeTensors::deserialize(&archive_bytes)?;
    let weights = load_qwen2_weights(&archive, &config, device)?;
    let model = Qwen2Model::from_weights(weights, config)?;

    let tokenizer = Tokenizer::from_file(&checkpoint.join("tokenizer.json"))?;
    let chat_template =
        ChatTemplate::from_file(&checkpoint.join("tokenizer_config.json"), &tokenizer)?;
    let eos_ids: HashSet<u32> = chat_template.eos_ids().collect();
    Ok(Loaded {
        model,
        tokenizer,
        eos_ids,
        chat_template,
    })
}

fn build_prompt(
    chat_template: &ChatTemplate,
    prompt: &str,
    system: Option<&str>,
    no_chat_template: bool,
) -> Result<String, BoxError> {
    if no_chat_template {
        return Ok(prompt.to_owned());
    }
    let mut messages = Vec::new();
    if let Some(sys) = system {
        messages.push(Message::system(sys));
    }
    messages.push(Message::user(prompt.to_owned()));
    Ok(chat_template.render(&messages)?)
}

fn run_chat(args: ChatArgs) -> Result<(), BoxError> {
    let device = Device::shared()?;
    let loaded = load(&args.checkpoint, &device)?;

    let prompt = build_prompt(
        &loaded.chat_template,
        &args.prompt,
        args.system.as_deref(),
        args.no_chat_template,
    )?;
    let ids = loaded.tokenizer.encode(&prompt)?;

    let mut generator = Generator::new(loaded.model, args.context_window, loaded.eos_ids)?;
    let mut stream = loaded.tokenizer.decode_stream(true);

    let stdout = io::stdout();
    let mut handle = stdout.lock();
    for id_result in generator.generate(&ids, args.max_tokens)? {
        let id = id_result?;
        if let Some(text) = stream.step(id)? {
            handle.write_all(text.as_bytes())?;
            handle.flush()?;
        }
    }
    writeln!(handle)?;
    Ok(())
}

#[allow(clippy::cast_precision_loss)]
fn run_bench(args: BenchArgs) -> Result<(), BoxError> {
    let device = Device::shared()?;

    let rss_pre_load = cider_press::sys::resident_bytes();
    let t_load = Instant::now();
    let loaded = load(&args.checkpoint, &device)?;
    let load_dur = t_load.elapsed();
    let rss_post_load = cider_press::sys::resident_bytes();

    let prompt = build_prompt(
        &loaded.chat_template,
        &args.prompt,
        args.system.as_deref(),
        args.no_chat_template,
    )?;
    let ids = loaded.tokenizer.encode(&prompt)?;
    let prompt_len = ids.len();

    let mut generator = Generator::new(loaded.model, args.context_window, loaded.eos_ids)?;

    // Prefill is performed inside generate() before it returns the iterator.
    let t_prefill = Instant::now();
    let iter = generator.generate(&ids, args.max_tokens)?;
    let prefill_dur = t_prefill.elapsed();

    // Decode: drain the iterator. Discard `warmup` tokens, then reset the
    // profiler and start the decode timer so both the tok/s figure and the
    // span breakdown reflect only the steady-state window.
    let mut produced = 0usize;
    let mut timed = 0usize;
    let mut decode_dur = std::time::Duration::ZERO;
    let mut decode_timer: Option<Instant> = None;
    for id_result in iter {
        let _ = id_result?;
        produced += 1;
        if produced == args.warmup {
            profile::reset();
            decode_timer = Some(Instant::now());
        } else if produced > args.warmup {
            timed += 1;
        }
    }
    if let Some(t) = decode_timer {
        decode_dur = t.elapsed();
    }

    let rss_post_decode = cider_press::sys::resident_bytes();
    let rss_peak = cider_press::sys::peak_resident_bytes();
    let spans = profile::drain();

    // ---- report ----
    let prefill_tok_s = prompt_len as f64 / prefill_dur.as_secs_f64();
    let decode_tok_s = if decode_dur.is_zero() {
        0.0
    } else {
        timed as f64 / decode_dur.as_secs_f64()
    };

    println!("cider-press bench");
    println!("  checkpoint     {}", args.checkpoint.display());
    println!("  prompt tokens  {prompt_len}");
    println!("  generated      {produced} (warmup {}, timed {timed})", args.warmup);
    println!();
    println!("  load           {:>8.3} s", load_dur.as_secs_f64());
    println!(
        "  prefill        {:>8.3} ms   {prefill_tok_s:>8.1} tok/s ({prompt_len} prompt tokens)",
        prefill_dur.as_secs_f64() * 1e3,
    );
    println!(
        "  decode         {:>8.3} ms   {decode_tok_s:>8.1} tok/s ({timed} tokens)",
        decode_dur.as_secs_f64() * 1e3,
    );
    println!();
    print_rss("  rss pre-load ", rss_pre_load);
    print_rss("  rss post-load", rss_post_load);
    print_rss("  rss post-dec ", rss_post_decode);
    print_rss("  rss peak     ", rss_peak);
    println!();
    if profile::is_enabled() {
        println!("  span breakdown (timed decode window):");
        println!("    {:<18} {:>10} {:>8} {:>12}", "span", "total/ms", "hits", "us/hit");
        for (name, total, hits) in &spans {
            let total_ms = total.as_secs_f64() * 1e3;
            let us_hit = if *hits == 0 {
                0.0
            } else {
                total.as_secs_f64() * 1e6 / *hits as f64
            };
            println!("    {name:<18} {total_ms:>10.3} {hits:>8} {us_hit:>12.2}");
        }
    } else {
        println!("  (span breakdown unavailable; rebuild with --features profiling)");
    }
    Ok(())
}

fn print_rss(label: &str, bytes: Option<u64>) {
    match bytes {
        Some(b) => println!("{label}  {:>8.1} MiB", b as f64 / (1024.0 * 1024.0)),
        None => println!("{label}  (unavailable)"),
    }
}

fn run() -> Result<(), BoxError> {
    match Cli::parse().command {
        Command::Chat(args) => run_chat(args),
        Command::Bench(args) => run_bench(args),
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
```

Note: `print_rss` uses `as f64` casts on `u64` — covered by the function-level `#[allow(clippy::cast_precision_loss)]` only if in the same fn. Since `print_rss` is a separate fn, add `#[allow(clippy::cast_precision_loss)]` directly above `fn print_rss` too.

- [ ] **Step 3: Add the clippy allow on `print_rss`**

Immediately above `fn print_rss(label: &str, bytes: Option<u64>) {`, add:

```rust
#[allow(clippy::cast_precision_loss)]
```

- [ ] **Step 4: Verify the default build (no profiling) compiles**

Run: `cargo build -p cider-press`
Expected: builds clean. `profile::is_enabled()` is `false`; bench prints the "rebuild with --features profiling" note.

- [ ] **Step 5: Verify the profiling build compiles + clippy clean**

Run: `cargo build -p cider-press --features profiling`
Then: `cargo clippy -p cider-press --all-targets --features profiling -- -D warnings`
Expected: builds and lints clean.

- [ ] **Step 6: Verify the help surface**

Run: `cargo run -p cider-press -- --help`
Expected: shows `chat` and `bench` subcommands.
Run: `cargo run -p cider-press -- bench --help`
Expected: shows `--checkpoint`, `--prompt`, `--max-tokens`, `--warmup`, etc.

- [ ] **Step 7: Commit**

```bash
git add crates/cider-press/Cargo.toml crates/cider-press/src/bin/cider-press.rs
git commit -m "feat(cli): bench subcommand + chat/bench split

Refactors the CLI into chat (existing behavior) and bench subcommands.
bench reuses the real load->generate path, reports prefill/decode tok/s
and peak RSS, and with --features profiling prints the per-span decode
breakdown. Adds a profiling feature forwarding to both lower crates.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: MLX comparison script

**Files:**
- Create: `scripts/measure_qwen_mlx.py`

- [ ] **Step 1: Write the `uv` script**

Create `scripts/measure_qwen_mlx.py`:

```python
#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "mlx-lm>=0.20",
#   "mlx>=0.18",
# ]
# ///
"""Measure mlx_lm greedy throughput + peak memory on a fixed prompt.

The cider-press `bench` subcommand reports prefill/decode tok/s and peak
RSS on the same prompt + max-tokens; this script produces the comparable
MLX numbers so docs/QWEN_PERF.md can table the ratio. Manual one-shot
tool (Python is one-shot infra here), mirroring measure_stage5_mlx.py.

Usage::

    uv run scripts/measure_qwen_mlx.py \\
        --checkpoint /path/to/qwen25-0.5b-instruct-4bit-mlx \\
        --prompt "Write a short paragraph about the city of Seattle." \\
        --max-tokens 128
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--prompt",
        default="Write a short paragraph about the city of Seattle.",
    )
    parser.add_argument("--system", default=None)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=8)
    args = parser.parse_args()

    model, tokenizer = load(args.checkpoint)

    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": args.prompt})
    prompt_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True
    )
    prompt = mx.array(prompt_ids)
    prompt_len = prompt.size

    mx.reset_peak_memory()

    # Prefill: time the first token (processes the whole prompt).
    gen = generate_step(prompt, model, temp=0.0)
    t_prefill = time.perf_counter()
    first_tok, _ = next(gen)
    mx.eval(first_tok)
    prefill_dur = time.perf_counter() - t_prefill

    # Decode: warmup then timed window.
    produced = 1  # the prefill token above
    timed = 0
    decode_dur = 0.0
    decode_start = None
    for tok, _ in gen:
        mx.eval(tok)
        produced += 1
        if produced == args.warmup:
            decode_start = time.perf_counter()
        elif produced > args.warmup:
            timed += 1
        if produced >= args.max_tokens:
            break
    if decode_start is not None:
        decode_dur = time.perf_counter() - decode_start

    peak_mib = mx.get_peak_memory() / (1024 * 1024)
    prefill_tok_s = prompt_len / prefill_dur if prefill_dur else 0.0
    decode_tok_s = timed / decode_dur if decode_dur else 0.0

    print("mlx_lm bench")
    print(f"  checkpoint     {args.checkpoint}")
    print(f"  prompt tokens  {prompt_len}")
    print(f"  generated      {produced} (warmup {args.warmup}, timed {timed})")
    print()
    print(f"  prefill        {prefill_dur * 1e3:8.3f} ms   {prefill_tok_s:8.1f} tok/s")
    print(f"  decode         {decode_dur * 1e3:8.3f} ms   {decode_tok_s:8.1f} tok/s")
    print(f"  peak memory    {peak_mib:8.1f} MiB")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Make it executable**

Run: `chmod +x scripts/measure_qwen_mlx.py`

- [ ] **Step 3: Sanity-check the script parses (no checkpoint needed)**

Run: `uv run scripts/measure_qwen_mlx.py --help`
Expected: prints usage with `--checkpoint`, `--prompt`, `--max-tokens`, `--warmup`. (If `generate_step`'s import path differs in the installed mlx_lm, adjust the import — `mlx_lm.generate.generate_step` is the 0.20+ location, matching the note in `scripts/dump_mlx_generate.py`.)

- [ ] **Step 4: Commit**

```bash
git add scripts/measure_qwen_mlx.py
git commit -m "feat(scripts): mlx_lm throughput + peak-memory companion

Measures prefill/decode tok/s and peak memory under mlx_lm greedy on the
same prompt the cider-press bench uses, so QWEN_PERF.md can table the
ratio. PEP 723 uv one-shot, paralleling measure_stage5_mlx.py.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 7: Run the measurement and write `docs/QWEN_PERF.md`

This task produces the deliverable. It requires a real checkpoint at the path you set in `$CIDER_QWEN_CHECKPOINT_PATH` (the same 4-bit MLX Qwen2.5-0.5B-Instruct used by the gated parity tests). The numbers in the doc come from these runs.

**Files:**
- Create: `docs/QWEN_PERF.md`
- Modify: `docs/QWEN_PATH.md`, `CLAUDE.md`

- [ ] **Step 1: Run the cider-press bench (release + profiling)**

Run:
```bash
cargo run --release -p cider-press --features profiling -- bench \
  --checkpoint "$CIDER_QWEN_CHECKPOINT_PATH" \
  --max-tokens 128 --warmup 8
```
Expected: a report with load time, prefill/decode tok/s, four RSS lines, and a span-breakdown table (`argmax`, `kvcache.memcpy`, `kvcache.update`, `tensor.eval`). Record the full output.

- [ ] **Step 2: Run the MLX comparison**

Run:
```bash
uv run scripts/measure_qwen_mlx.py \
  --checkpoint "$CIDER_QWEN_CHECKPOINT_PATH" \
  --max-tokens 128 --warmup 8
```
Expected: prefill/decode tok/s + peak memory. Record the output. (Use the same `--prompt` on both sides; both default to the Seattle paragraph.)

- [ ] **Step 3: Capture the environment**

Run:
```bash
sw_vers && sysctl -n machdep.cpu.brand_string && echo "checkpoint: $CIDER_QWEN_CHECKPOINT_PATH"
```
Record macOS version, chip, and checkpoint identity for the doc's environment section.

- [ ] **Step 4: Write `docs/QWEN_PERF.md`**

Create `docs/QWEN_PERF.md` and fill every `<…>` from the recorded outputs of Steps 1–3. This template has no optional sections — fill all cells:

```markdown
# Qwen2.5-0.5B perf

End-to-end performance of Qwen2.5-0.5B-Instruct-4bit under `cider-press`,
measured via `cider-press bench`, compared against `mlx_lm` on the same
prompt and token budget. Methodology mirrors the spike's Stage 5 (warm
up, then time a steady-state window); see `docs/QWEN_PATH.md` for the
op-by-op road that got here.

## Environment

- Machine: <chip from sysctl, e.g. Apple M2 Pro>
- OS: <macOS version from sw_vers>
- Checkpoint: <checkpoint dir / HF id>, 4-bit MLX quant (group_size=64)
- Prompt: "Write a short paragraph about the city of Seattle."
- max_tokens: 128, warmup: 8
- Build: `cargo run --release -p cider-press --features profiling`

## Throughput

| phase   | cider-press | mlx_lm | ratio (mlx / cp) |
|---------|------------:|-------:|-----------------:|
| prefill | <cp prefill tok/s>  | <mlx prefill tok/s> | <ratio> |
| decode  | <cp decode tok/s>   | <mlx decode tok/s>  | <ratio> |

Prefill processes <prompt_len> prompt tokens in one forward; decode is
the per-token steady state (the perf-dominant path). Load time
(excluded from tok/s): <load s> s.

## Decode-step breakdown

Per-span totals over the timed decode window (<timed> steps), from the
profiling facility. Spans overlap by construction — `kvcache.update`
includes the input evals counted in `tensor.eval`, and `tensor.eval`
covers all GPU dispatch+wait wherever triggered; `argmax` is the
post-eval CPU `cpu_to_vec` + scan only.

| span           | total (ms) | hits | µs/hit | share of decode |
|----------------|-----------:|-----:|-------:|----------------:|
| tensor.eval    | <…> | <…> | <…> | <…>% |
| argmax         | <…> | <…> | <…> | <…>% |
| kvcache.update | <…> | <…> | <…> | <…>% |
| kvcache.memcpy | <…> | <…> | <…> | <…>% |

(`share of decode` = span total / decode wall time; will not sum to
100% due to the documented overlap.)

## Memory

| mark        | cider-press RSS |
|-------------|----------------:|
| pre-load    | <…> MiB |
| post-load   | <…> MiB |
| post-decode | <…> MiB |
| peak        | <…> MiB |

mlx_lm peak (Metal allocations, `mx.get_peak_memory()`): <…> MiB. RSS
and MLX peak measure different things (process resident set vs Metal
buffer high-water), so compare trends, not absolute equality.

## Hot spots & next levers

<2-4 sentences keyed to the numbers: which span dominates the decode
step, and which deferred optimization it points at. Candidates, per the
roadmap's "deferred past first-token" list:>

- **Per-dispatch commit/wait** (the spike's ~1.5× gap): if `tensor.eval`
  dominates and µs/hit is large relative to the kernel work, cross-eval
  command-buffer batching is the lever.
- **bf16 argmax round-trip**: if `argmax` is a non-trivial share, GPU
  argmax removes the `[vocab]` `cpu_to_vec` + scan each step.
- **KvCache memcpy**: if `kvcache.memcpy` is material, folding the cache
  write into the SDPA command buffer (same-eval batching) is the lever.
- **Buffer pool**: per-`eval()` allocation cost shows up inside
  `tensor.eval`; a pool amortizes it.

These are measurement-justified starting points for follow-up work; none
are implemented here.

## Reproduce

```bash
cargo run --release -p cider-press --features profiling -- bench \
  --checkpoint "$CIDER_QWEN_CHECKPOINT_PATH" --max-tokens 128 --warmup 8
uv run scripts/measure_qwen_mlx.py \
  --checkpoint "$CIDER_QWEN_CHECKPOINT_PATH" --max-tokens 128 --warmup 8
```
```

- [ ] **Step 5: Mark branch 15 done in the roadmap + project state**

In `docs/QWEN_PATH.md`, change the branch-15 table row (line 97) status from the planned form to done, and update the "What to do next time" closing section (lines 130-156) to reflect that perf measurement landed and point at `docs/QWEN_PERF.md`. Specifically:
- Row 15: prefix the branch name with `(done)` and replace the "Validated by" cell with `Numbers in docs/QWEN_PERF.md`.
- Replace the "Branches still ahead of perf measurement: 15 (perf). One branch between us and docs/QWEN_PERF.md." sentence (lines 155-156) with a sentence stating perf measurement is complete and `docs/QWEN_PERF.md` holds the numbers; carry the deferred-optimization list forward as the next candidates.

In `CLAUDE.md`, add a branch-15 bullet to the "Current state" branch list (after the branch-12b/12c entries) summarizing: feature-gated `profile` facility (named spans → thread-local accumulator, zero-cost when off), `tensor.eval`/`kvcache.update`/`kvcache.memcpy`/`argmax` spans, the `bench` CLI subcommand, mach-`task_info` RSS sampling, `scripts/measure_qwen_mlx.py`, and `docs/QWEN_PERF.md` as the deliverable. Keep the prose framed around the capability (the branch-plan references are slated for cleanup in a later PR).

- [ ] **Step 6: Final workspace verification**

Run: `cargo test --workspace`
Then: `cargo test --workspace --features profiling` (workspace-level feature; turns on profiling where defined)
Then: `cargo clippy --all-targets --all-features -- -D warnings`
Then: `cargo fmt --check`
Then: `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps`
Expected: all green. (`cargo doc` covers the new module docs on `profile` and `sys`.)

- [ ] **Step 7: Commit**

```bash
git add docs/QWEN_PERF.md docs/QWEN_PATH.md CLAUDE.md
git commit -m "docs(perf): QWEN_PERF.md measurement results + mark branch 15 done

End-to-end Qwen2.5-0.5B tok/s + decode-step span breakdown + peak RSS,
vs mlx_lm on the same prompt. Documents the hot spots that justify the
deferred optimizations (command-buffer batching, GPU argmax, pool).

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review notes

- **Spec coverage:** profiling facility (Task 1), feature-gated zero-cost (Task 1 steps 1/5), thread-local spans at the named hot spots — `tensor.eval`/`kvcache.update`/`kvcache.memcpy` (Task 2), `argmax` (Task 3); `bench` subcommand reusing the real path (Task 5); in-process mach RSS (Task 4); MLX comparison script (Task 6); `docs/QWEN_PERF.md` with throughput / breakdown / memory / hot-spots sections (Task 7). All spec sections map to a task.
- **Deviation from spec, recorded:** RSS uses `MACH_TASK_BASIC_INFO.resident_size{,_max}` rather than `TASK_VM_INFO.phys_footprint` — stable fixed-layout struct over a brittle field offset; still in-process mach `task_info`. Documented in `sys.rs` and the memory section.
- **Deviation from spec, recorded:** spec listed `prefill.forward` / `decode.forward` spans; the plan instead derives prefill-vs-decode tok/s from bench wall-clock timers (more accurate for throughput) and uses `tensor.eval` for dispatch attribution, avoiding misleading overlap with the lazy-eval structure (forward builds the graph; compute happens at eval). Span set is `tensor.eval`, `kvcache.update`, `kvcache.memcpy`, `argmax`.
- **Type consistency:** `profile::{span, reset, drain, is_enabled, Span}` used identically across Tasks 1-5; `cider_press::sys::{resident_bytes, peak_resident_bytes}` defined in Task 4 and consumed in Task 5; `Loaded`/`load`/`build_prompt` defined and used within Task 5.
```
