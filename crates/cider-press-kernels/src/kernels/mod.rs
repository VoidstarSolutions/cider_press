//! Typed Rust dispatch functions over MLX-vendored Metal kernels.
//!
//! Each submodule here corresponds to an MLX dispatch routine
//! (`mlx/backend/metal/*.cpp`) and exposes one or more Rust functions
//! that encode the kernel against [`crate::Commands`]. The functions
//! never commit — see [`crate::Commands`] for the batching rationale.
//!
//! New kernels are added as the runtime layer needs them; the dispatch
//! transcription pattern is validated by the qmv parity test, so the
//! per-kernel work is mostly mechanical.

pub mod arg_reduce;
pub mod binary;
pub mod copy;
pub mod dequantize;
pub mod gather;
pub mod matmul;
pub mod qmm;
pub mod qmv;
pub mod reduce;
pub mod rope;
pub mod sdpa;
pub mod softmax;
pub mod unary;
