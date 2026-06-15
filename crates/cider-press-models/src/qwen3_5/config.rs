//! Qwen3.5 / Qwen3.6 text-model configuration (parsed from `config.json`).
//!
//! Hyperparameters live under the nested `text_config`; `quantization` is at the
//! top level. We flatten both into one ergonomic struct via an internal raw
//! representation, mirroring `qwen2::config`.

use cider_press_runtime::Quantization;
use serde::Deserialize;

use crate::error::{Error, Result};

/// Default Hugging Face repo for the dev vehicle (4B). The 27B target lives at
/// `mlx-community/Qwen3.6-27B-4bit`.
pub const HF_REPO: &str = "mlx-community/Qwen3.5-4B-4bit";
/// Pinned revision used for reproducible loads.
pub const HF_REVISION: &str = "0e7ffd5c629ef7719d4cbc04069232580bfa9d9c";

/// Which mixer a decoder layer uses. The checkpoint lists one per layer in
/// `text_config.layer_types` (3 linear : 1 full, `full_attention_interval = 4`).
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LayerType {
    /// Gated-DeltaNet linear-attention mixer.
    LinearAttention,
    /// Full (Gated) softmax-attention mixer.
    FullAttention,
}

/// Affine quantization parameters (`group_size` 64, 4-bit). `mode` is ignored.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[non_exhaustive]
pub struct Qwen35QuantizationConfig {
    /// Elements per quantization group.
    pub group_size: u32,
    /// Bits per weight.
    pub bits: u8,
}

/// RoPE parameters (captured for later phases; unused by the loader).
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[non_exhaustive]
pub struct RopeParameters {
    /// Rotary base frequency θ.
    pub rope_theta: f64,
    /// Fraction of `head_dim` that is rotated (the rest passes through).
    pub partial_rotary_factor: f64,
    /// mRoPE section split (collapses to 1-D positions for text).
    pub mrope_section: Vec<usize>,
    /// Whether mRoPE sections are interleaved.
    #[serde(default)]
    pub mrope_interleaved: bool,
}

/// Flattened Qwen3.5/3.6 text-model config.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct Qwen35Config {
    /// Hidden / residual stream dimension.
    pub hidden_size: usize,
    /// Number of decoder layers.
    pub num_hidden_layers: usize,
    /// Full-attention query heads.
    pub num_attention_heads: usize,
    /// Full-attention key/value heads (GQA).
    pub num_key_value_heads: usize,
    /// Per-head dimension for full attention.
    pub head_dim: usize,
    /// `SwiGLU` intermediate dimension.
    pub intermediate_size: usize,
    /// Token vocabulary size.
    pub vocab_size: usize,
    /// Epsilon for `RMSNorm`.
    pub rms_norm_eps: f64,
    /// Whether the full-attention q/k/v projections carry an additive bias.
    pub attention_bias: bool,
    /// Whether `q_proj` emits a query‖gate pair (doubles its output width).
    pub attn_output_gate: bool,
    /// Layer stride between full-attention layers (3 linear : 1 full).
    pub full_attention_interval: usize,
    /// Depthwise causal-conv kernel width in the linear mixer.
    pub linear_conv_kernel_dim: usize,
    /// Per-head key dimension in the linear (Gated-DeltaNet) mixer.
    pub linear_key_head_dim: usize,
    /// Key heads in the linear mixer.
    pub linear_num_key_heads: usize,
    /// Value heads in the linear mixer.
    pub linear_num_value_heads: usize,
    /// Per-head value dimension in the linear mixer.
    pub linear_value_head_dim: usize,
    /// Whether the LM head shares weights with `embed_tokens`.
    pub tie_word_embeddings: bool,
    /// Per-layer mixer assignment.
    pub layer_types: Vec<LayerType>,
    /// RoPE parameters (used by later phases).
    pub rope: RopeParameters,
    /// Quantization metadata, present iff the checkpoint is pre-quantized.
    pub quantization: Option<Qwen35QuantizationConfig>,
}

#[derive(Deserialize)]
struct RawConfig {
    text_config: RawTextConfig,
    #[serde(default)]
    quantization: Option<Qwen35QuantizationConfig>,
}

#[derive(Deserialize)]
struct RawTextConfig {
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    intermediate_size: usize,
    vocab_size: usize,
    rms_norm_eps: f64,
    #[serde(default)]
    attention_bias: bool,
    #[serde(default)]
    attn_output_gate: bool,
    full_attention_interval: usize,
    linear_conv_kernel_dim: usize,
    linear_key_head_dim: usize,
    linear_num_key_heads: usize,
    linear_num_value_heads: usize,
    linear_value_head_dim: usize,
    #[serde(default)]
    tie_word_embeddings: bool,
    layer_types: Vec<LayerType>,
    rope_parameters: RopeParameters,
}

impl Qwen35Config {
    /// Parse a `config.json` byte slice (the full top-level object, including the
    /// nested `text_config` and top-level `quantization`).
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self> {
        let raw: RawConfig = serde_json::from_slice(bytes).map_err(Error::from)?;
        let t = raw.text_config;
        Ok(Self {
            hidden_size: t.hidden_size,
            num_hidden_layers: t.num_hidden_layers,
            num_attention_heads: t.num_attention_heads,
            num_key_value_heads: t.num_key_value_heads,
            head_dim: t.head_dim,
            intermediate_size: t.intermediate_size,
            vocab_size: t.vocab_size,
            rms_norm_eps: t.rms_norm_eps,
            attention_bias: t.attention_bias,
            attn_output_gate: t.attn_output_gate,
            full_attention_interval: t.full_attention_interval,
            linear_conv_kernel_dim: t.linear_conv_kernel_dim,
            linear_key_head_dim: t.linear_key_head_dim,
            linear_num_key_heads: t.linear_num_key_heads,
            linear_num_value_heads: t.linear_num_value_heads,
            linear_value_head_dim: t.linear_value_head_dim,
            tie_word_embeddings: t.tie_word_embeddings,
            layer_types: t.layer_types,
            rope: t.rope_parameters,
            quantization: raw.quantization,
        })
    }

    /// Linear-attention key projection width: `num_key_heads * key_head_dim`.
    #[must_use]
    pub fn key_dim(&self) -> usize {
        self.linear_num_key_heads * self.linear_key_head_dim
    }

    /// Linear-attention value projection width: `num_value_heads * value_head_dim`.
    #[must_use]
    pub fn value_dim(&self) -> usize {
        self.linear_num_value_heads * self.linear_value_head_dim
    }

    /// Depthwise-conv channel count over `q‖k‖v`: `2 * key_dim + value_dim`.
    #[must_use]
    pub fn conv_dim(&self) -> usize {
        2 * self.key_dim() + self.value_dim()
    }

    /// Full-attention query width: `num_attention_heads * head_dim`.
    #[must_use]
    pub fn attn_query_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    /// `q_proj` output width — doubled (query‖gate) when `attn_output_gate`.
    #[must_use]
    pub fn q_proj_out_dim(&self) -> usize {
        if self.attn_output_gate {
            2 * self.attn_query_dim()
        } else {
            self.attn_query_dim()
        }
    }

    /// Full-attention key/value width: `num_key_value_heads * head_dim`.
    #[must_use]
    pub fn kv_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }

    /// Whether layer `i` uses the linear (Gated-DeltaNet) mixer.
    #[must_use]
    pub fn is_linear(&self, i: usize) -> bool {
        self.layer_types[i] == LayerType::LinearAttention
    }

    /// Build the runtime [`Quantization`] from the config. Errors if the
    /// checkpoint is not pre-quantized (this loader requires it).
    pub fn quantization(&self) -> Result<Quantization> {
        let q = self.quantization.ok_or_else(|| {
            Error::InvalidArgument(
                "Qwen35Config.quantization is None — loader requires a pre-quantized checkpoint"
                    .into(),
            )
        })?;
        Quantization::new(q.bits, q.group_size).map_err(Error::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONFIG_JSON: &str = r#"{
        "model_type": "qwen3_5",
        "tie_word_embeddings": true,
        "quantization": { "group_size": 64, "bits": 4, "mode": "affine" },
        "text_config": {
            "hidden_size": 2560,
            "num_hidden_layers": 4,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "intermediate_size": 9216,
            "vocab_size": 248320,
            "rms_norm_eps": 1e-06,
            "attention_bias": false,
            "attn_output_gate": true,
            "full_attention_interval": 4,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_value_head_dim": 128,
            "tie_word_embeddings": true,
            "layer_types": [
                "linear_attention", "linear_attention",
                "linear_attention", "full_attention"
            ],
            "rope_parameters": {
                "rope_theta": 10000000,
                "partial_rotary_factor": 0.25,
                "mrope_section": [11, 11, 10],
                "mrope_interleaved": true
            }
        }
    }"#;

    #[test]
    fn parses_nested_text_config_and_layer_types() {
        let cfg = Qwen35Config::from_json_bytes(CONFIG_JSON.as_bytes()).unwrap();
        assert_eq!(cfg.hidden_size, 2560);
        assert_eq!(cfg.num_hidden_layers, 4);
        assert_eq!(cfg.head_dim, 256);
        assert_eq!(cfg.vocab_size, 248_320);
        assert!(cfg.tie_word_embeddings);
        assert!(!cfg.attention_bias);
        assert!(cfg.attn_output_gate);
        assert_eq!(cfg.layer_types.len(), 4);
        assert_eq!(cfg.layer_types[0], LayerType::LinearAttention);
        assert_eq!(cfg.layer_types[3], LayerType::FullAttention);
        assert!(cfg.is_linear(2));
        assert!(!cfg.is_linear(3));
        let q = cfg.quantization.unwrap();
        assert_eq!(q.group_size, 64);
        assert_eq!(q.bits, 4);
        assert!((cfg.rope.partial_rotary_factor - 0.25).abs() < f64::EPSILON);
        assert_eq!(cfg.rope.mrope_section, vec![11, 11, 10]);
    }

    #[test]
    fn derives_mixer_dims() {
        let cfg = Qwen35Config::from_json_bytes(CONFIG_JSON.as_bytes()).unwrap();
        assert_eq!(cfg.key_dim(), 2048);
        assert_eq!(cfg.value_dim(), 4096);
        assert_eq!(cfg.conv_dim(), 8192);
        assert_eq!(cfg.attn_query_dim(), 4096);
        assert_eq!(cfg.q_proj_out_dim(), 8192);
        assert_eq!(cfg.kv_dim(), 1024);
        assert_eq!(cfg.quantization().unwrap().group_size(), 64);
    }

    #[test]
    fn quantization_method_errors_when_absent() {
        // A config with no top-level `quantization` block — the loader requires
        // a pre-quantized checkpoint, so `quantization()` must error.
        const NO_QUANT: &str = r#"{
            "model_type": "qwen3_5",
            "text_config": {
                "hidden_size": 2560, "num_hidden_layers": 4, "num_attention_heads": 16,
                "num_key_value_heads": 4, "head_dim": 256, "intermediate_size": 9216,
                "vocab_size": 248320, "rms_norm_eps": 1e-06, "full_attention_interval": 4,
                "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128,
                "linear_num_key_heads": 16, "linear_num_value_heads": 32,
                "linear_value_head_dim": 128, "tie_word_embeddings": true,
                "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
                "rope_parameters": { "rope_theta": 10000000, "partial_rotary_factor": 0.25, "mrope_section": [11, 11, 10] }
            }
        }"#;
        let cfg = Qwen35Config::from_json_bytes(NO_QUANT.as_bytes()).unwrap();
        assert!(cfg.quantization.is_none());
        assert!(cfg.quantization().is_err());
    }
}
