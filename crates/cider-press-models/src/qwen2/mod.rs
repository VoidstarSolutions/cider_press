//! Qwen2.5 model loader.
//!
//! Single entry point for loading a Qwen2-family checkpoint: callers
//! feed [`load_qwen2_weights`] the parsed `config.json` and the
//! safetensors archive and get back a typed [`Qwen2Weights`] plus a
//! [`Qwen2Config`] describing the architecture.
//!
//! ## Pinned checkpoint
//!
//! The acceptance bar for this branch is "loaded bytes == safetensors
//! bytes via memcmp." The synthetic CI test exercises the loader
//! against an in-memory fixture; the optional `CIDER_QWEN_CHECKPOINT_PATH`
//! gated integration test verifies against the real checkpoint at the
//! revision pinned in [`HF_REVISION`].

pub mod attention;
mod config;
mod weights;

pub use config::{HF_REPO, HF_REVISION, Qwen2Config, Qwen2QuantizationConfig};
pub use weights::{
    Qwen2AttentionWeights, Qwen2LayerWeights, Qwen2MlpWeights, Qwen2Weights, load_qwen2_weights,
};
