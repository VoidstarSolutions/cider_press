//! Error type for `cider-press-runtime`.
//!
//! Distinct named variants for the failure modes a caller might want to
//! match on (shape/layout violations, quantization constraint failures).
//! `#[non_exhaustive]` is deliberate: this is a library-crate boundary
//! and we will add variants as the runtime grows.

use thiserror::Error;

/// Errors produced by `cider-press-runtime`.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
    /// An invariant on a runtime primitive was violated (e.g. a
    /// quantization group size that doesn't divide the row length, or a
    /// stride list with the wrong rank).
    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    /// Surfaced from `cider-press-kernels`.
    #[error(transparent)]
    Kernels(#[from] cider_press_kernels::Error),
}

/// Convenience `Result` alias for the `cider-press-runtime` API.
pub type Result<T> = core::result::Result<T, Error>;
