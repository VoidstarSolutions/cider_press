//! Qwen2 `config.json` schema and HF revision pinning.
//!
//! Only the fields we actually consume during loading and inference
//! are deserialized; HF ships a number of fields (`attention_dropout`,
//! `initializer_range`, `transformers_version`, …) that the inference
//! runtime ignores. Unknown fields are silently allowed so newer
//! upstream additions don't break the loader.

use serde::Deserialize;

use crate::error::{Error, Result};

/// `HuggingFace` repo that hosts the canonical MLX-quantized checkpoint.
pub const HF_REPO: &str = "mlx-community/Qwen2.5-0.5B-Instruct-4bit";

/// Pinned HF revision SHA for [`HF_REPO`]. Community quantizations
/// can drift over time; pinning makes the gated real-checkpoint test
/// reproducible and gives us a clear bump point if we ever need to
/// adopt a newer quant.
///
/// Bumping procedure:
/// 1. Verify the new revision still produces tied-embedding `q4_gs64`
///    weights with matching architecture (`config.json` deltas).
/// 2. Run the gated integration test against the new revision.
/// 3. Update this constant and document the rationale in the commit
///    message.
pub const HF_REVISION: &str = "a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3";

/// Qwen2 architecture configuration, as deserialized from
/// `config.json`. Field names mirror HF's `snake_case`; field types
/// match the runtime's expectations.
#[derive(Clone, Debug, Deserialize)]
#[non_exhaustive]
pub struct Qwen2Config {
    /// Hidden / residual stream dimension (`D` in the `QWEN_PATH` doc).
    pub hidden_size: usize,
    /// Number of transformer blocks.
    pub num_hidden_layers: usize,
    /// Total attention heads (`H_q`).
    pub num_attention_heads: usize,
    /// KV heads (`H_kv`); GQA ratio is `num_attention_heads /
    /// num_key_value_heads`.
    pub num_key_value_heads: usize,
    /// `SwiGLU` intermediate dim.
    pub intermediate_size: usize,
    /// Token vocabulary size.
    pub vocab_size: usize,
    /// Maximum positional embedding length.
    pub max_position_embeddings: usize,
    /// `RoPE` base (`θ`).
    pub rope_theta: f64,
    /// Epsilon for `RMSNorm`.
    pub rms_norm_eps: f64,
    /// Whether the LM head shares weights with `embed_tokens`. For
    /// the pinned checkpoint this is `true` — the LM head reuses
    /// `model.embed_tokens` (and is therefore quantized too).
    #[serde(default)]
    pub tie_word_embeddings: bool,
    /// Quantization metadata, present iff the checkpoint is
    /// pre-quantized (MLX 4-bit). `None` for the bf16 HF originals.
    #[serde(default)]
    pub quantization: Option<Qwen2QuantizationConfig>,
}

/// MLX-style affine quantization block from `config.json`.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[non_exhaustive]
pub struct Qwen2QuantizationConfig {
    /// Elements per scale/bias group.
    pub group_size: u32,
    /// Bit-width per packed lane (e.g. 4 for int4).
    pub bits: u8,
}

impl Qwen2Config {
    /// Parse a `config.json` from raw bytes.
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self> {
        serde_json::from_slice(bytes).map_err(Error::from)
    }

    /// Per-head dimension: `hidden_size / num_attention_heads`.
    /// Returns [`Error::InvalidArgument`] if `hidden_size` doesn't
    /// divide evenly.
    pub fn head_dim(&self) -> Result<usize> {
        if self.num_attention_heads == 0 {
            return Err(Error::InvalidArgument(
                "Qwen2Config: num_attention_heads must be > 0".into(),
            ));
        }
        if self.hidden_size % self.num_attention_heads != 0 {
            return Err(Error::InvalidArgument(format!(
                "Qwen2Config: hidden_size {} not divisible by num_attention_heads {}",
                self.hidden_size, self.num_attention_heads,
            )));
        }
        Ok(self.hidden_size / self.num_attention_heads)
    }

    /// Grouped-query attention ratio:
    /// `num_attention_heads / num_key_value_heads`.
    pub fn gqa_ratio(&self) -> Result<usize> {
        if self.num_attention_heads == 0
            || self.num_key_value_heads == 0
            || self.num_attention_heads % self.num_key_value_heads != 0
        {
            return Err(Error::InvalidArgument(format!(
                "Qwen2Config: num_attention_heads {} not a multiple of num_key_value_heads {}",
                self.num_attention_heads, self.num_key_value_heads,
            )));
        }
        Ok(self.num_attention_heads / self.num_key_value_heads)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The canonical `config.json` from the pinned HF revision.
    /// Stored verbatim to verify the deserializer accepts the exact
    /// upstream payload without manual reshaping.
    const PINNED_CONFIG_JSON: &str = r#"{
        "architectures": ["Qwen2ForCausalLM"],
        "attention_dropout": 0.0,
        "bos_token_id": 151643,
        "eos_token_id": 151645,
        "hidden_act": "silu",
        "hidden_size": 896,
        "initializer_range": 0.02,
        "intermediate_size": 4864,
        "max_position_embeddings": 32768,
        "max_window_layers": 21,
        "model_type": "qwen2",
        "num_attention_heads": 14,
        "num_hidden_layers": 24,
        "num_key_value_heads": 2,
        "quantization": { "group_size": 64, "bits": 4 },
        "rms_norm_eps": 1e-06,
        "rope_theta": 1000000.0,
        "sliding_window": 32768,
        "tie_word_embeddings": true,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.43.1",
        "use_cache": true,
        "use_sliding_window": false,
        "vocab_size": 151936
    }"#;

    #[test]
    fn parses_pinned_config() {
        let cfg = Qwen2Config::from_json_bytes(PINNED_CONFIG_JSON.as_bytes()).expect("parse");
        assert_eq!(cfg.hidden_size, 896);
        assert_eq!(cfg.num_hidden_layers, 24);
        assert_eq!(cfg.num_attention_heads, 14);
        assert_eq!(cfg.num_key_value_heads, 2);
        assert_eq!(cfg.intermediate_size, 4864);
        assert_eq!(cfg.vocab_size, 151_936);
        assert_eq!(cfg.max_position_embeddings, 32_768);
        assert!((cfg.rope_theta - 1_000_000.0).abs() < f64::EPSILON);
        assert!(cfg.tie_word_embeddings);
        let q = cfg.quantization.expect("quantization present");
        assert_eq!(q.bits, 4);
        assert_eq!(q.group_size, 64);
    }

    #[test]
    fn computed_dims_match_qwen_path_doc() {
        let cfg = Qwen2Config::from_json_bytes(PINNED_CONFIG_JSON.as_bytes()).expect("parse");
        assert_eq!(cfg.head_dim().unwrap(), 64);
        assert_eq!(cfg.gqa_ratio().unwrap(), 7);
    }

    #[test]
    fn unknown_fields_are_ignored() {
        // Confirms serde's default behaviour on extra fields — newer
        // upstream additions shouldn't break the loader.
        let json = br#"{
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "intermediate_size": 16,
            "vocab_size": 32,
            "max_position_embeddings": 64,
            "rope_theta": 10000.0,
            "rms_norm_eps": 1e-6,
            "some_future_field": [1, 2, 3]
        }"#;
        let cfg = Qwen2Config::from_json_bytes(json).expect("parse with unknown field");
        assert_eq!(cfg.hidden_size, 8);
        assert!(cfg.quantization.is_none());
    }

    #[test]
    fn head_dim_rejects_indivisible() {
        let cfg = Qwen2Config {
            hidden_size: 100,
            num_hidden_layers: 1,
            num_attention_heads: 7,
            num_key_value_heads: 1,
            intermediate_size: 1,
            vocab_size: 1,
            max_position_embeddings: 1,
            rope_theta: 1.0,
            rms_norm_eps: 1.0,
            tie_word_embeddings: false,
            quantization: None,
        };
        assert!(cfg.head_dim().is_err());
    }

    #[test]
    fn head_dim_rejects_zero_attention_heads() {
        let cfg = Qwen2Config {
            hidden_size: 8,
            num_hidden_layers: 1,
            num_attention_heads: 0,
            num_key_value_heads: 1,
            intermediate_size: 1,
            vocab_size: 1,
            max_position_embeddings: 1,
            rope_theta: 1.0,
            rms_norm_eps: 1.0,
            tie_word_embeddings: false,
            quantization: None,
        };
        assert!(matches!(cfg.head_dim(), Err(Error::InvalidArgument(_))));
    }

    #[test]
    fn gqa_ratio_rejects_zero_kv_heads() {
        let cfg = Qwen2Config {
            hidden_size: 8,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 0,
            intermediate_size: 1,
            vocab_size: 1,
            max_position_embeddings: 1,
            rope_theta: 1.0,
            rms_norm_eps: 1.0,
            tie_word_embeddings: false,
            quantization: None,
        };
        assert!(matches!(cfg.gqa_ratio(), Err(Error::InvalidArgument(_))));
    }

    #[test]
    fn hf_revision_is_full_sha() {
        // Sanity: the pinned revision is a 40-hex-char commit SHA.
        // Catches accidental truncation when bumping.
        assert_eq!(HF_REVISION.len(), 40);
        assert!(HF_REVISION.chars().all(|c| c.is_ascii_hexdigit()));
    }
}
