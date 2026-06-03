//! Low-level Metal kernel dispatch for `cider-press`.
//!
//! Wraps vendored MLX kernel sources (see `kernels-mlx/` at the workspace
//! root) in a Rust dispatch surface: device + queue plumbing, buffer
//! management, runtime kernel-library compilation, and one Rust function
//! per Metal kernel that handles bindings, threadgroup math, and
//! fast-vs-generic variant selection.
//!
//! This crate intentionally does not provide a `Tensor`, a graph, or a
//! buffer pool — those live in `cider-press-runtime`. If you only need
//! to dispatch MLX kernels from Rust without buying into the lazy-graph
//! runtime, this crate is the layer for you.
//!
//! Status: pre-alpha. Spike-validated (bit-exact parity with MLX's own
//! `quantized_matmul` on `qmv` bf16 × int4) but the public API is not
//! yet stable. See `CLAUDE.md` at the workspace root for the staged
//! plan and next-step recommendation.

#![cfg_attr(not(target_os = "macos"), allow(unused))]

#[cfg(not(target_os = "macos"))]
compile_error!(
    "cider-press currently targets macOS / Apple Silicon only. \
     Build on an aarch64-apple-darwin host."
);

mod buffer;
mod commands;
mod device;
mod error;
mod gpu_sampler;
pub mod kernels;
mod library;

pub use buffer::Buffer;
pub use commands::{Commands, CommandsInFlight};
pub use device::Device;
pub use error::{Error, Result};
pub use gpu_sampler::{GpuSampler, GpuSegment};
pub use library::{FunctionConstant, KernelLibrary, Pipeline};
