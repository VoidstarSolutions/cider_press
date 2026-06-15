//! Qwen3.5 / Qwen3.6 (dense, Gated-DeltaNet hybrid) text model.
//!
//! `model_type: qwen3_5`. Qwen3.6-27B reuses this exact text architecture, so
//! one module serves both; the 27B differs only by scale and two config knobs
//! (`linear_num_value_heads`, untied `lm_head`). Vision is skipped — text only.
//!
//! Phase 1 is the loader only (config + weight mapping + shape validation);
//! the forward pass lands in later phases. See `docs/inference/models/qwen3.6.md`
//! and `ROADMAP.md`.

mod config;
mod weights;

pub use config::{
    HF_REPO, HF_REVISION, LayerType, Qwen35Config, Qwen35QuantizationConfig, RopeParameters,
};
pub use weights::{
    Qwen35FullAttnWeights, Qwen35LayerWeights, Qwen35LinearAttnWeights, Qwen35MixerWeights,
    Qwen35MlpWeights, Qwen35Weights, load_qwen35_weights,
};
