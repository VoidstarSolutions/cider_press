//! Cold-pressed LLM inference for Apple Silicon.
//!
//! `cider-press` is the umbrella crate: a stable, re-exporting facade
//! over the workspace's lower layers. Typical users depend on this
//! crate; downstream tooling that only needs raw kernel dispatch or
//! the runtime without models can depend on `cider-press-kernels` or
//! `cider-press-runtime` directly. (Backtick-quoted here rather than
//! intra-doc-linked because the facade crate intentionally does not
//! depend on `cider-press-kernels` — that's the runtime's concern.)
//!
//! Status: scaffolded only. Re-exports will materialize as the runtime
//! and model layers grow public surface.
//!
//! See `CLAUDE.md` at the workspace root for project context, the
//! completed spike findings, and the next-step recommendation.

#![doc = include_str!("../../../docs/inference/README.md")]
#![cfg_attr(not(target_os = "macos"), allow(unused))]

#[cfg(not(target_os = "macos"))]
compile_error!(
    "cider-press currently targets macOS / Apple Silicon only. \
     Build on an aarch64-apple-darwin host."
);

pub mod sys;
