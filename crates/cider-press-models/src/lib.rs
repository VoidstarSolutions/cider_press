//! Transformer architectures + quantized weight loaders for `cider-press`.
//!
//! Sits on top of [`cider_press_runtime`]: model code consumes the
//! runtime's `Tensor` and ops, never reaches into kernel dispatch
//! directly. Per-architecture modules will be gated behind submodules
//! (e.g. `qwen2`, landing in subsequent commits); the safetensors
//! plumbing in [`safetensors_io`] is shared across them.
//!
//! Status: weight-loader plumbing landing now. `safetensors_io` reads
//! dense and quantized tensors out of a parsed safetensors archive
//! into the runtime's `Tensor` / `QuantizedWeight` types. The Qwen2
//! config + weight-mapping layer is the next commit.

#![cfg_attr(not(target_os = "macos"), allow(unused))]

#[cfg(not(target_os = "macos"))]
compile_error!(
    "cider-press currently targets macOS / Apple Silicon only. \
     Build on an aarch64-apple-darwin host."
);

mod error;
pub mod safetensors_io;

pub use error::{Error, Result};
