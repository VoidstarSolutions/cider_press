//! Typed Rust dispatch functions over MLX-vendored Metal kernels.
//!
//! Each submodule here corresponds to an MLX dispatch routine
//! (`mlx/backend/metal/*.cpp`) and exposes one or more Rust functions
//! that encode the kernel against [`crate::Commands`]. The functions
//! never commit — see [`crate::Commands`] for the batching rationale.
//!
//! New kernels are added as the runtime layer needs them; the spike
//! validated the dispatch transcription pattern (see Stage 4 findings
//! in `CLAUDE.md`) so the per-kernel work is mostly mechanical.

pub mod binary;
pub mod copy;
pub mod dequantize;
pub mod gather;
pub mod qmv;
pub mod reduce;
pub mod rope;
pub mod softmax;
pub mod unary;
