//! Crate-level error type. Kept separate from `cider_press_runtime`'s
//! error so that loader-specific failure modes (missing keys, dtype
//! mismatches in safetensors, JSON parse failures) don't leak into
//! the runtime's narrower contract.

use thiserror::Error;

/// Error type for everything in `cider-press-models`.
#[derive(Debug, Error)]
pub enum Error {
    /// A safetensors archive was missing a tensor key we expected.
    #[error("safetensors: missing tensor key {key:?}")]
    MissingKey {
        /// The key that was looked up.
        key: String,
    },
    /// A safetensors tensor had a dtype we don't know how to upload.
    #[error("safetensors: unsupported dtype {dtype:?} for key {key:?}")]
    UnsupportedDtype {
        /// The key whose dtype was rejected.
        key: String,
        /// The unsupported safetensors dtype, as a debug string.
        dtype: String,
    },
    /// A safetensors tensor's shape didn't match what the caller expected.
    #[error("safetensors: shape mismatch for {key:?}: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        /// The key whose shape was wrong.
        key: String,
        /// The shape the caller wanted.
        expected: Vec<usize>,
        /// The shape the archive actually carries.
        actual: Vec<usize>,
    },
    /// Failed to deserialize a safetensors archive header.
    #[error("safetensors: failed to deserialize archive: {0}")]
    Safetensors(#[from] safetensors::SafeTensorError),
    /// Failed to parse JSON (typically `config.json`).
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    /// Failed to load or run a tokenizer.
    #[error("tokenizer: {0}")]
    Tokenizer(#[from] tokenizers::Error),
    /// A chat template (minijinja) failed to load, compile, or render.
    #[error("chat template: {0}")]
    ChatTemplate(#[from] minijinja::Error),
    /// Runtime-layer error (allocation, upload, …) bubbling up
    /// through the loader path.
    #[error("runtime: {0}")]
    Runtime(#[from] cider_press_runtime::Error),
    /// Catch-all for argument-validation failures the loader raises
    /// itself (mismatched expectations, malformed config, …).
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
}

/// Convenience alias.
pub type Result<T> = std::result::Result<T, Error>;
