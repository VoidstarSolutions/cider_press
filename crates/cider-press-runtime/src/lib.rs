//! Lazy-tensor runtime for `cider-press`.
//!
//! Provides the `Tensor` / graph / `eval()` layer on top of
//! [`cider_press_kernels`]. The runtime owns the buffer pool, the
//! command-buffer batching strategy, and the public op API. Kernel
//! dispatch is delegated downward to `cider-press-kernels`; model
//! architectures are layered upward in `cider-press-models`.
//!
//! Status: the design-doc runtime slice is complete. A [`Tensor`]
//! can be a metadata-only placeholder, a host-constructed dense or
//! quantized leaf, or an op node referencing input tensors. Two ops
//! are wired through the runtime end-to-end:
//!
//! - [`Tensor::copy`] — element-wise identity (`f32`, `f16`, `bf16`,
//!   `i32`, `u32` — every dtype the runtime currently models).
//! - [`Tensor::quantized_matmul`] — affine-quantized matmul (bf16
//!   inputs, bf16 output; parity with MLX's `mx.quantized_matmul`;
//!   routes M=1 to `affine_qmv` and M>1 to `affine_qmm_t`).
//!
//! [`Tensor::eval`] is the synchronous boundary that walks the
//! graph, dispatches MLX kernels via [`cider_press_kernels`], blocks
//! on completion, and populates each op's result cache. See
//! `docs/RUNTIME_DESIGN.md` for the staged plan and remaining
//! follow-ups (buffer pool, attention, command-buffer pipelining).
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

mod buffer_pool;
mod device;
mod dtype;
mod error;
mod eval;
mod kv_cache;
mod pending_eval;
mod layout;
pub mod profile;
mod quantization;
mod quantized;
mod shape;
mod strides;
mod tensor;

pub use buffer_pool::PoolStats;
pub use device::Device;
pub use pending_eval::PendingEval;
pub use dtype::{DType, Scalar};
pub use error::{Error, Result};
pub use kv_cache::KvCache;
pub use layout::Layout;
pub use quantization::Quantization;
pub use quantized::QuantizedWeight;
pub use shape::Shape;
pub use strides::Strides;
pub use tensor::{ArgReduceKind, BinaryOp, CpuIter, OpKind, ReduceKind, Tensor, UnaryOp};
