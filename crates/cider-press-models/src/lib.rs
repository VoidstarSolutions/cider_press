//! Transformer architectures + quantized weight loaders for `cider-press`.
//!
//! Sits on top of [`cider_press_runtime`]: model code consumes the
//! runtime's `Tensor` and ops, never reaches into kernel dispatch
//! directly. Per-architecture modules will be feature-gated once the
//! first architecture lands (see `CLAUDE.md`); the model-layout
//! decision is intentionally deferred until there's a second
//! architecture to compare against the first.
//!
//! Status: scaffolded only. The first architecture is TBD; SDPA is the
//! next required kernel in `cider-press-kernels`.

#![cfg_attr(not(target_os = "macos"), allow(unused))]

#[cfg(not(target_os = "macos"))]
compile_error!(
    "cider-press currently targets macOS / Apple Silicon only. \
     Build on an aarch64-apple-darwin host."
);
