//! Qwen3.5 / Qwen3.6 (dense, Gated-DeltaNet hybrid) text model.
//!
//! `model_type: qwen3_5`. Qwen3.6-27B reuses this exact text architecture, so
//! one module serves both; the 27B differs only by scale and two config knobs
//! (`linear_num_value_heads`, untied `lm_head`). Vision is skipped — text only.
//!
//! Provides the config and weight loader (single- or multi-shard, with shape
//! validation), the two token mixers (gated full attention and the
//! Gated-DeltaNet linear attention, [`GatedDeltaNet`]), the hybrid decoder
//! layer, and [`Qwen35Model`] with its per-layer cache state
//! ([`Qwen35LayerCache`], [`LinearState`]). See
//! `docs/inference/models/qwen3.6.md` and `ROADMAP.md`.

mod attention;
mod block;
mod cache;
mod config;
mod gated_deltanet;
mod model;
mod weights;

pub use attention::GatedAttention;
pub use block::Qwen35DecoderLayer;
pub use cache::{LinearState, Qwen35LayerCache};
pub use config::{
    HF_REPO, HF_REVISION, LayerType, Qwen35Config, Qwen35QuantizationConfig, RopeParameters,
};
pub use gated_deltanet::GatedDeltaNet;
pub use model::Qwen35Model;
#[cfg(test)]
pub(crate) use weights::test_support;
pub use weights::{
    Qwen35FullAttnWeights, Qwen35LayerWeights, Qwen35LinearAttnWeights, Qwen35MixerWeights,
    Qwen35MlpWeights, Qwen35Weights, load_qwen35_weights,
};
