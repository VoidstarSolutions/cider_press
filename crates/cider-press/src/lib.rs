//! Cold-pressed LLM inference for Apple Silicon.
//!
//! `cider-press` is the umbrella crate: a stable, re-exporting facade
//! over the workspace's lower layers. Typical users depend on this
//! crate; downstream tooling that only needs raw kernel dispatch or
//! the runtime without models can depend on
//! [`cider_press_kernels`] or [`cider_press_runtime`] directly.
//!
//! Status: scaffolded only. Re-exports will materialize as the runtime
//! and model layers grow public surface.
//!
//! See `CLAUDE.md` at the workspace root for project context, the
//! completed spike findings, and the next-step recommendation.

#![cfg_attr(not(target_os = "macos"), allow(unused))]

#[cfg(not(target_os = "macos"))]
compile_error!(
    "cider-press currently targets macOS / Apple Silicon only. \
     Build on an aarch64-apple-darwin host."
);
