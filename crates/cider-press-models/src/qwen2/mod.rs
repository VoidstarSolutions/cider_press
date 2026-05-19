//! Qwen2.5 model loader.
//!
//! Currently covers configuration parsing only; the weight struct and
//! key-mapping function land in the next commit. This module is the
//! single entry point for loading a Qwen2-family checkpoint —
//! callers feed it the parsed `config.json` and the safetensors
//! archive and get back a typed `Qwen2Weights` (TBD) plus a
//! [`Qwen2Config`] describing the architecture.
//!
//! ## Pinned checkpoint
//!
//! The acceptance bar for this branch is "loaded bytes == safetensors
//! bytes via memcmp." The synthetic CI test exercises the loader
//! against an in-memory fixture; the optional `CIDER_QWEN_CHECKPOINT_PATH`
//! gated integration test (landing alongside the weight loader)
//! verifies against the real checkpoint at the revision pinned in
//! [`HF_REVISION`].

mod config;

pub use config::{HF_REPO, HF_REVISION, Qwen2Config, Qwen2QuantizationConfig};
