//! Lazy-tensor runtime for `cider-press`.
//!
//! Provides the `Tensor` / graph / `eval()` layer on top of
//! [`cider_press_kernels`]. The runtime owns the buffer pool, the
//! command-buffer batching strategy, and the public op API. Kernel
//! dispatch is delegated downward to `cider-press-kernels`; model
//! architectures are layered upward in `cider-press-models`.
//!
//! Status: data primitives in progress. This module currently lands
//! the runtime [`Error`] / [`Result`] surface; subsequent commits add
//! the dtype/shape/layout/tensor types.

#![cfg_attr(not(target_os = "macos"), allow(unused))]

#[cfg(not(target_os = "macos"))]
compile_error!(
    "cider-press currently targets macOS / Apple Silicon only. \
     Build on an aarch64-apple-darwin host."
);

mod dtype;
mod error;
mod layout;
mod quantization;
mod shape;
mod strides;

pub use dtype::{DType, Scalar};
pub use error::{Error, Result};
pub use layout::Layout;
pub use quantization::Quantization;
pub use shape::Shape;
pub use strides::Strides;
