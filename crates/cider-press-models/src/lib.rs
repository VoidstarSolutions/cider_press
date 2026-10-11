//! Transformer architectures + quantized weight loaders for `cider-press`.
//!
//! Sits on top of [`cider_press_runtime`]: model code consumes the
//! runtime's `Tensor` and ops, never reaches into kernel dispatch
//! directly. Per-architecture modules live in submodules ([`qwen2`],
//! [`qwen3_5`]); the safetensors plumbing in [`safetensors_io`] is
//! shared across them. [`generator`] drives any [`language_model::LanguageModel`]
//! with greedy decoding.

#![cfg_attr(not(target_os = "macos"), allow(unused))]

#[cfg(not(target_os = "macos"))]
compile_error!(
    "cider-press currently targets macOS / Apple Silicon only. \
     Build on an aarch64-apple-darwin host."
);

pub mod chat_template;
mod eos;
mod error;
pub mod generator;
pub mod language_model;
pub mod nn;
pub mod qwen2;
pub mod qwen3_5;
pub mod safetensors_io;
pub mod tokenizer;

pub use eos::config_eos_ids;
pub use error::{Error, Result};
pub use language_model::LanguageModel;
pub use tokenizer::Tokenizer;
