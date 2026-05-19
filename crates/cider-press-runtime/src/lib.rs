//! Lazy-tensor runtime for `cider-press`.
//!
//! Provides the `Tensor` / graph / `eval()` layer on top of
//! [`cider_press_kernels`]. The runtime owns the buffer pool, the
//! command-buffer batching strategy, and the public op API. Kernel
//! dispatch is delegated downward to `cider-press-kernels`; model
//! architectures are layered upward in `cider-press-models`.
//!
//! Status: data primitives + storage + lazy graph. A [`Tensor`] can
//! be a metadata-only placeholder, a materialized leaf (real Metal
//! buffer), or a lazy op node referencing input tensors. The first
//! op constructor is [`Tensor::copy`]; op construction is purely
//! metadata (no GPU work, no output buffer). `eval()` follows in the
//! next commit and is what turns op nodes into leaves; see
//! `docs/RUNTIME_DESIGN.md` for the staged plan.
//!
//! Quantized layouts are first-class from day one so the API design
//! gets stress-tested by the awkward case before it accumulates
//! homogeneous-dense assumptions.

#![cfg_attr(not(target_os = "macos"), allow(unused))]

#[cfg(not(target_os = "macos"))]
compile_error!(
    "cider-press currently targets macOS / Apple Silicon only. \
     Build on an aarch64-apple-darwin host."
);

mod device;
mod dtype;
mod error;
mod layout;
mod quantization;
mod shape;
mod strides;
mod tensor;

pub use device::Device;
pub use dtype::{DType, Scalar};
pub use error::{Error, Result};
pub use layout::Layout;
pub use quantization::Quantization;
pub use shape::Shape;
pub use strides::Strides;
pub use tensor::{OpKind, Tensor};
