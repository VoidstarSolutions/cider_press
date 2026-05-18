//! Error type for `cider-press-kernels`.
//!
//! Distinct named variants for the failure modes a caller might want to
//! match on (missing device, kernel-not-found, MSL compile error). A
//! catch-all [`Error::AppleApi`] variant covers the long tail of "Apple
//! API returned nil from somewhere we don't expect" cases that shouldn't
//! happen but technically can.
//!
//! `#[non_exhaustive]` is deliberate: this is a library-crate boundary and
//! we will add variants as new failure modes appear.

use thiserror::Error;

/// Errors produced by `cider-press-kernels`.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
    /// `MTLCreateSystemDefaultDevice` returned nil. The host has no Metal
    /// device (or one we can talk to) — not survivable here.
    #[error("no Metal device available on this system")]
    NoDevice,

    /// `newLibraryWithSource:options:error:` failed to compile MSL.
    /// The string is the localized `NSError` description.
    #[error("Metal shader compile failed: {0}")]
    Compile(String),

    /// `newFunctionWithName:` returned nil — the requested kernel symbol
    /// is not present in the compiled library. Usually a name typo or a
    /// missing template instantiation.
    #[error("Metal kernel `{0}` not found in library")]
    KernelNotFound(String),

    /// `newComputePipelineStateWithFunction:error:` failed.
    #[error("failed to build compute pipeline for `{name}`: {message}")]
    Pipeline { name: String, message: String },

    /// `newBufferWithLength:options:` returned nil.
    #[error("failed to allocate Metal buffer of {bytes} bytes")]
    BufferAllocation { bytes: usize },

    /// Requested buffer length is too large to express in bytes
    /// (`len * size_of::<T>()` overflows `usize`). Hit before any
    /// Metal API call.
    #[error("buffer length {len} elements × {elem_size}-byte stride overflows usize")]
    BufferTooLarge { len: usize, elem_size: usize },

    /// An Apple framework call returned nil from somewhere we expected
    /// it not to (e.g. `commandBuffer`, `computeCommandEncoder`). The
    /// `context` is a short literal identifying the call site.
    #[error("Apple Metal API returned nil from {context}")]
    AppleApi { context: &'static str },
}

/// Convenience `Result` alias for the `cider-press-kernels` API.
pub type Result<T> = core::result::Result<T, Error>;
