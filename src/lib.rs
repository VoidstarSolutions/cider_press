//! Cold-pressed LLM inference for Apple Silicon.
//!
//! `cider-press` is an inference-only tensor and model runtime targeting
//! Apple Silicon via Metal and the unified memory model.
//!
//! Status: pre-alpha. The project is currently in its initial development
//! spike — see `CLAUDE.md` for the staged plan. Public API surface is
//! intentionally empty until the spike informs the design.

#![cfg_attr(not(target_os = "macos"), allow(unused))]

#[cfg(not(target_os = "macos"))]
compile_error!(
    "cider-press currently targets macOS / Apple Silicon only. \
     Build on an aarch64-apple-darwin host."
);
