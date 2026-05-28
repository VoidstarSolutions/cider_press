//! Transformer architectures + quantized weight loaders for `cider-press`.
//!
//! Sits on top of [`cider_press_runtime`]: model code consumes the
//! runtime's `Tensor` and ops, never reaches into kernel dispatch
//! directly. Per-architecture modules live in submodules (currently
//! [`qwen2`]); the safetensors plumbing in [`safetensors_io`] is
//! shared across them.
//!
//! Status: weight-loader plumbing in place. `safetensors_io` reads
//! dense and quantized tensors out of a parsed safetensors archive
//! into the runtime's `Tensor` / `QuantizedWeight` types, and
//! [`qwen2`] provides the Qwen2 config schema plus
//! `load_qwen2_weights` key mapping.

#![cfg_attr(not(target_os = "macos"), allow(unused))]

#[cfg(not(target_os = "macos"))]
compile_error!(
    "cider-press currently targets macOS / Apple Silicon only. \
     Build on an aarch64-apple-darwin host."
);

mod error;
pub mod chat_template;
pub mod nn;
pub mod qwen2;
pub mod safetensors_io;
pub mod tokenizer;

pub use error::{Error, Result};
pub use tokenizer::Tokenizer;
