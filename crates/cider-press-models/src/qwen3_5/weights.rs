//! Qwen3.5 / Qwen3.6 text-model weights: load + shape-validate every per-layer
//! tensor from a (single-shard) safetensors archive. Raw keys live under
//! `language_model.model.` (the `model.` remap is mlx-lm's in-memory `sanitize`,
//! not present on disk). Vision (`vision_tower.*`) is skipped. Loader only —
//! no forward.
//!
//! ```text
//! language_model.model.embed_tokens.{weight, scales, biases}      # quant, [V, D]
//! language_model.model.norm.weight                                # bf16,  [D]
//! language_model.model.layers.{i}.input_layernorm.weight          # bf16,  [D]
//! language_model.model.layers.{i}.post_attention_layernorm.weight # bf16,  [D]
//! ```
//!
//! Each layer's mixer is one of two variants, selected by
//! [`Qwen35Config::layer_types`]:
//!
//! - **Linear attention** (Gated-DeltaNet): split `in_proj_{qkv,z,a,b}`,
//!   an f32 `A_log`, a `dt_bias`, a depthwise `conv1d.weight`, a head-dim
//!   `norm`, and an `out_proj`.
//! - **Full (gated) attention**: `q_proj` (query‖gate), `k_proj`, `v_proj`,
//!   QK `RMSNorm` gammas, and `o_proj`. `attention_bias = false`, so there
//!   are no additive q/k/v biases on disk.

use cider_press_runtime::{DType, Device, Layout, Quantization, QuantizedWeight, Tensor};
use safetensors::SafeTensors;

use crate::error::{Error, Result};
use crate::qwen3_5::config::{LayerType, Qwen35Config};
use crate::safetensors_io::{read_dense_tensor, read_quantized_weight};

/// On-disk key prefix for every text-model tensor. The `model.`-only form is
/// mlx-lm's in-memory remap and is *not* what the checkpoint stores.
const TEXT_PREFIX: &str = "language_model.model";

/// All weights for one Qwen3.5/3.6 text model.
#[derive(Debug)]
#[non_exhaustive]
pub struct Qwen35Weights {
    /// Token embedding table, quantized. Logical shape `[vocab_size,
    /// hidden_size]`. Tied with the LM head (no separate `lm_head`).
    pub embed: QuantizedWeight,
    /// Final `RMSNorm` gamma. Dense bf16, shape `[hidden_size]`.
    pub norm: Tensor,
    /// Per-block weights. Length equals
    /// [`Qwen35Config::num_hidden_layers`].
    pub layers: Vec<Qwen35LayerWeights>,
}

/// Weights for one decoder block.
#[derive(Debug)]
#[non_exhaustive]
pub struct Qwen35LayerWeights {
    /// Pre-mixer `RMSNorm` gamma. Dense bf16, shape `[hidden_size]`.
    pub input_layernorm: Tensor,
    /// Pre-MLP `RMSNorm` gamma. Dense bf16, shape `[hidden_size]`.
    pub post_attention_layernorm: Tensor,
    /// `SwiGLU` MLP projections.
    pub mlp: Qwen35MlpWeights,
    /// Mixer weights, variant chosen by the layer's
    /// [`crate::qwen3_5::config::LayerType`].
    pub mixer: Qwen35MixerWeights,
}

/// A decoder layer's mixer weights — exactly one of the two architectures.
#[derive(Debug)]
pub enum Qwen35MixerWeights {
    /// Gated-DeltaNet linear-attention mixer.
    LinearAttn(Qwen35LinearAttnWeights),
    /// Full (gated) softmax-attention mixer.
    FullAttn(Qwen35FullAttnWeights),
}

/// `SwiGLU` MLP block weights.
#[derive(Debug)]
#[non_exhaustive]
pub struct Qwen35MlpWeights {
    /// Gate projection. Logical `[intermediate_size, hidden_size]`.
    pub gate_proj: QuantizedWeight,
    /// Up projection. Logical `[intermediate_size, hidden_size]`.
    pub up_proj: QuantizedWeight,
    /// Down projection. Logical `[hidden_size, intermediate_size]`.
    pub down_proj: QuantizedWeight,
}

/// Gated-DeltaNet (linear-attention) mixer weights.
#[derive(Debug)]
#[non_exhaustive]
pub struct Qwen35LinearAttnWeights {
    /// Fused q‖k‖v input projection. Logical `[conv_dim, hidden_size]`.
    pub in_proj_qkv: QuantizedWeight,
    /// Gate (`z`) input projection. Logical `[value_dim, hidden_size]`.
    pub in_proj_z: QuantizedWeight,
    /// `a` input projection. Logical `[linear_num_value_heads, hidden_size]`.
    pub in_proj_a: QuantizedWeight,
    /// `b` input projection. Logical `[linear_num_value_heads, hidden_size]`.
    pub in_proj_b: QuantizedWeight,
    /// Per-head decay log-rate. Dense **f32**, shape `[linear_num_value_heads]`.
    pub a_log: Tensor,
    /// Per-head Δt bias. Dense bf16, shape `[linear_num_value_heads]`.
    pub dt_bias: Tensor,
    /// Depthwise causal-conv kernel. Dense bf16, shape
    /// `[conv_dim, linear_conv_kernel_dim, 1]`.
    pub conv1d_weight: Tensor,
    /// Output `RMSNorm` gamma. Dense bf16, shape `[linear_value_head_dim]`.
    pub norm: Tensor,
    /// Output projection. Logical `[hidden_size, value_dim]`.
    pub out_proj: QuantizedWeight,
}

/// Full (gated) softmax-attention mixer weights.
#[derive(Debug)]
#[non_exhaustive]
pub struct Qwen35FullAttnWeights {
    /// Query‖gate projection. Logical `[q_proj_out_dim, hidden_size]`.
    pub q_proj: QuantizedWeight,
    /// Key projection. Logical `[kv_dim, hidden_size]`.
    pub k_proj: QuantizedWeight,
    /// Value projection. Logical `[kv_dim, hidden_size]`.
    pub v_proj: QuantizedWeight,
    /// Query `RMSNorm` gamma. Dense bf16, shape `[head_dim]`.
    pub q_norm: Tensor,
    /// Key `RMSNorm` gamma. Dense bf16, shape `[head_dim]`.
    pub k_norm: Tensor,
    /// Output projection. Logical `[hidden_size, attn_query_dim]`.
    pub o_proj: QuantizedWeight,
}

/// Load every weight referenced in a Qwen3.5/3.6 text checkpoint into typed
/// runtime tensors, validating each logical shape against `config`'s derived
/// dims. Mismatches return [`Error::ShapeMismatch`] or
/// [`Error::UnsupportedDtype`] with the offending key. Vision tensors are
/// never referenced.
pub fn load_qwen35_weights(
    archive: &SafeTensors<'_>,
    config: &Qwen35Config,
    device: &Device,
) -> Result<Qwen35Weights> {
    let quant = config.quantization()?;
    let hidden = config.hidden_size;

    let embed = read_quantized_weight(
        archive,
        &format!("{TEXT_PREFIX}.embed_tokens"),
        quant,
        device,
    )?;
    expect_q_shape(
        &format!("{TEXT_PREFIX}.embed_tokens"),
        &embed,
        config.vocab_size,
        hidden,
    )?;

    let norm = read_dense_tensor(archive, &format!("{TEXT_PREFIX}.norm.weight"), device)?;
    expect_dense(
        &format!("{TEXT_PREFIX}.norm.weight"),
        &norm,
        DType::BF16,
        &[hidden],
    )?;

    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    for i in 0..config.num_hidden_layers {
        layers.push(load_layer(archive, config, quant, device, i)?);
    }

    Ok(Qwen35Weights {
        embed,
        norm,
        layers,
    })
}

fn load_layer(
    archive: &SafeTensors<'_>,
    config: &Qwen35Config,
    quant: Quantization,
    device: &Device,
    i: usize,
) -> Result<Qwen35LayerWeights> {
    let prefix = format!("{TEXT_PREFIX}.layers.{i}");
    let hidden = config.hidden_size;

    let input_layernorm =
        read_dense_tensor(archive, &format!("{prefix}.input_layernorm.weight"), device)?;
    expect_dense(
        &format!("{prefix}.input_layernorm.weight"),
        &input_layernorm,
        DType::BF16,
        &[hidden],
    )?;

    let post_attention_layernorm = read_dense_tensor(
        archive,
        &format!("{prefix}.post_attention_layernorm.weight"),
        device,
    )?;
    expect_dense(
        &format!("{prefix}.post_attention_layernorm.weight"),
        &post_attention_layernorm,
        DType::BF16,
        &[hidden],
    )?;

    let mlp = load_mlp(archive, config, quant, device, &format!("{prefix}.mlp"))?;

    let mixer = match config.layer_types[i] {
        LayerType::LinearAttention => Qwen35MixerWeights::LinearAttn(load_linear_attn(
            archive,
            config,
            quant,
            device,
            &format!("{prefix}.linear_attn"),
        )?),
        LayerType::FullAttention => Qwen35MixerWeights::FullAttn(load_full_attn(
            archive,
            config,
            quant,
            device,
            &format!("{prefix}.self_attn"),
        )?),
    };

    Ok(Qwen35LayerWeights {
        input_layernorm,
        post_attention_layernorm,
        mlp,
        mixer,
    })
}

fn load_mlp(
    archive: &SafeTensors<'_>,
    config: &Qwen35Config,
    quant: Quantization,
    device: &Device,
    prefix: &str,
) -> Result<Qwen35MlpWeights> {
    let hidden = config.hidden_size;
    let intermediate = config.intermediate_size;

    let gate_proj = read_quantized_weight(archive, &format!("{prefix}.gate_proj"), quant, device)?;
    expect_q_shape(
        &format!("{prefix}.gate_proj"),
        &gate_proj,
        intermediate,
        hidden,
    )?;
    let up_proj = read_quantized_weight(archive, &format!("{prefix}.up_proj"), quant, device)?;
    expect_q_shape(&format!("{prefix}.up_proj"), &up_proj, intermediate, hidden)?;
    let down_proj = read_quantized_weight(archive, &format!("{prefix}.down_proj"), quant, device)?;
    expect_q_shape(
        &format!("{prefix}.down_proj"),
        &down_proj,
        hidden,
        intermediate,
    )?;

    Ok(Qwen35MlpWeights {
        gate_proj,
        up_proj,
        down_proj,
    })
}

fn load_linear_attn(
    archive: &SafeTensors<'_>,
    config: &Qwen35Config,
    quant: Quantization,
    device: &Device,
    prefix: &str,
) -> Result<Qwen35LinearAttnWeights> {
    let hidden = config.hidden_size;
    let value_heads = config.linear_num_value_heads;

    let in_proj_qkv =
        read_quantized_weight(archive, &format!("{prefix}.in_proj_qkv"), quant, device)?;
    expect_q_shape(
        &format!("{prefix}.in_proj_qkv"),
        &in_proj_qkv,
        config.conv_dim(),
        hidden,
    )?;
    let in_proj_z = read_quantized_weight(archive, &format!("{prefix}.in_proj_z"), quant, device)?;
    expect_q_shape(
        &format!("{prefix}.in_proj_z"),
        &in_proj_z,
        config.value_dim(),
        hidden,
    )?;
    let in_proj_a = read_quantized_weight(archive, &format!("{prefix}.in_proj_a"), quant, device)?;
    expect_q_shape(
        &format!("{prefix}.in_proj_a"),
        &in_proj_a,
        value_heads,
        hidden,
    )?;
    let in_proj_b = read_quantized_weight(archive, &format!("{prefix}.in_proj_b"), quant, device)?;
    expect_q_shape(
        &format!("{prefix}.in_proj_b"),
        &in_proj_b,
        value_heads,
        hidden,
    )?;

    let a_log = read_dense_tensor(archive, &format!("{prefix}.A_log"), device)?;
    expect_dense(
        &format!("{prefix}.A_log"),
        &a_log,
        DType::F32,
        &[value_heads],
    )?;
    let dt_bias = read_dense_tensor(archive, &format!("{prefix}.dt_bias"), device)?;
    expect_dense(
        &format!("{prefix}.dt_bias"),
        &dt_bias,
        DType::BF16,
        &[value_heads],
    )?;

    let conv1d_weight = read_dense_tensor(archive, &format!("{prefix}.conv1d.weight"), device)?;
    expect_dense(
        &format!("{prefix}.conv1d.weight"),
        &conv1d_weight,
        DType::BF16,
        &[config.conv_dim(), config.linear_conv_kernel_dim, 1],
    )?;

    let norm = read_dense_tensor(archive, &format!("{prefix}.norm.weight"), device)?;
    expect_dense(
        &format!("{prefix}.norm.weight"),
        &norm,
        DType::BF16,
        &[config.linear_value_head_dim],
    )?;

    let out_proj = read_quantized_weight(archive, &format!("{prefix}.out_proj"), quant, device)?;
    expect_q_shape(
        &format!("{prefix}.out_proj"),
        &out_proj,
        hidden,
        config.value_dim(),
    )?;

    Ok(Qwen35LinearAttnWeights {
        in_proj_qkv,
        in_proj_z,
        in_proj_a,
        in_proj_b,
        a_log,
        dt_bias,
        conv1d_weight,
        norm,
        out_proj,
    })
}

fn load_full_attn(
    archive: &SafeTensors<'_>,
    config: &Qwen35Config,
    quant: Quantization,
    device: &Device,
    prefix: &str,
) -> Result<Qwen35FullAttnWeights> {
    let hidden = config.hidden_size;

    let q_proj = read_quantized_weight(archive, &format!("{prefix}.q_proj"), quant, device)?;
    expect_q_shape(
        &format!("{prefix}.q_proj"),
        &q_proj,
        config.q_proj_out_dim(),
        hidden,
    )?;
    let k_proj = read_quantized_weight(archive, &format!("{prefix}.k_proj"), quant, device)?;
    expect_q_shape(
        &format!("{prefix}.k_proj"),
        &k_proj,
        config.kv_dim(),
        hidden,
    )?;
    let v_proj = read_quantized_weight(archive, &format!("{prefix}.v_proj"), quant, device)?;
    expect_q_shape(
        &format!("{prefix}.v_proj"),
        &v_proj,
        config.kv_dim(),
        hidden,
    )?;

    let q_norm = read_dense_tensor(archive, &format!("{prefix}.q_norm.weight"), device)?;
    expect_dense(
        &format!("{prefix}.q_norm.weight"),
        &q_norm,
        DType::BF16,
        &[config.head_dim],
    )?;
    let k_norm = read_dense_tensor(archive, &format!("{prefix}.k_norm.weight"), device)?;
    expect_dense(
        &format!("{prefix}.k_norm.weight"),
        &k_norm,
        DType::BF16,
        &[config.head_dim],
    )?;

    let o_proj = read_quantized_weight(archive, &format!("{prefix}.o_proj"), quant, device)?;
    expect_q_shape(
        &format!("{prefix}.o_proj"),
        &o_proj,
        hidden,
        config.attn_query_dim(),
    )?;

    Ok(Qwen35FullAttnWeights {
        q_proj,
        k_proj,
        v_proj,
        q_norm,
        k_norm,
        o_proj,
    })
}

fn expect_q_shape(key: &str, qw: &QuantizedWeight, n: usize, k: usize) -> Result<()> {
    let actual = qw.shape().dims();
    if actual != [n, k] {
        return Err(Error::ShapeMismatch {
            key: key.into(),
            expected: vec![n, k],
            actual: actual.to_vec(),
        });
    }
    Ok(())
}

fn expect_dense(key: &str, t: &Tensor, dtype: DType, shape: &[usize]) -> Result<()> {
    if !matches!(t.layout(), Layout::Dense { .. }) {
        return Err(Error::InvalidArgument(format!(
            "{key:?}: expected dense layout, got quantized"
        )));
    }
    if t.dtype() != dtype {
        return Err(Error::UnsupportedDtype {
            key: key.into(),
            dtype: format!("{:?}", t.dtype()),
        });
    }
    let actual = t.shape().dims();
    if actual != shape {
        return Err(Error::ShapeMismatch {
            key: key.into(),
            expected: shape.to_vec(),
            actual: actual.to_vec(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use cider_press_runtime::Device;
    use safetensors::{Dtype as StDtype, SafeTensors, tensor::TensorView};
    use std::{fs, path::PathBuf};

    use crate::qwen3_5::config::Qwen35Config;

    /// Tiny Qwen3.5 config that satisfies every quantization divisibility
    /// constraint (hidden=64 divisible by `group_size=64`; one head each),
    /// with a `[linear_attention, full_attention]` layer pair.
    const TINY_CONFIG_JSON: &str = r#"{
  "model_type": "qwen3_5",
  "quantization": { "group_size": 64, "bits": 4, "mode": "affine" },
  "text_config": {
    "hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 1,
    "num_key_value_heads": 1, "head_dim": 64, "intermediate_size": 128,
    "vocab_size": 256, "rms_norm_eps": 1e-06, "attention_bias": false,
    "attn_output_gate": true, "full_attention_interval": 2,
    "linear_conv_kernel_dim": 4, "linear_key_head_dim": 64,
    "linear_num_key_heads": 1, "linear_num_value_heads": 1, "linear_value_head_dim": 64,
    "tie_word_embeddings": true,
    "layer_types": ["linear_attention", "full_attention"],
    "rope_parameters": { "rope_theta": 10000000, "partial_rotary_factor": 0.25, "mrope_section": [11,11,10] }
  }
}"#;

    type Entry = (String, StDtype, Vec<usize>, Vec<u8>);

    fn dense_entry(key: &str, dtype: StDtype, shape: Vec<usize>) -> Entry {
        let count: usize = shape.iter().product();
        let elem = match dtype {
            StDtype::F32 => 4,
            StDtype::BF16 => 2,
            other => panic!("unexpected dense dtype {other:?}"),
        };
        (key.into(), dtype, shape, vec![0u8; count * elem])
    }

    /// Push the three component entries for a quantized linear at `base`
    /// with logical `[n, k]`, q4-gs64: `.weight` U32 `[n, k/8]`,
    /// `.scales`/`.biases` BF16 `[n, k/64]`.
    fn push_quantized(entries: &mut Vec<Entry>, base: &str, n: usize, k: usize) {
        let packed_k = k / 8;
        let groups = k / 64;
        entries.push((
            format!("{base}.weight"),
            StDtype::U32,
            vec![n, packed_k],
            vec![0u8; n * packed_k * 4],
        ));
        entries.push((
            format!("{base}.scales"),
            StDtype::BF16,
            vec![n, groups],
            vec![0u8; n * groups * 2],
        ));
        entries.push((
            format!("{base}.biases"),
            StDtype::BF16,
            vec![n, groups],
            vec![0u8; n * groups * 2],
        ));
    }

    /// Synthesize an in-memory safetensors archive holding every key the
    /// loader reads for `cfg`, under the `language_model.model.` prefix.
    fn synthesize_archive_bytes(cfg: &Qwen35Config) -> Vec<u8> {
        synthesize_archive_bytes_with(cfg, |_, _| None)
    }

    /// Same, but `override_shape(key, default)` may return a replacement
    /// logical shape for a dense tensor (used by the negative test).
    #[allow(clippy::too_many_lines)]
    fn synthesize_archive_bytes_with(
        cfg: &Qwen35Config,
        override_shape: impl Fn(&str, &[usize]) -> Option<Vec<usize>>,
    ) -> Vec<u8> {
        let hidden = cfg.hidden_size;
        let intermediate = cfg.intermediate_size;
        let value_heads = cfg.linear_num_value_heads;
        let p = TEXT_PREFIX;

        let mut entries: Vec<Entry> = Vec::new();
        let dense = |entries: &mut Vec<Entry>, key: &str, dtype: StDtype, shape: Vec<usize>| {
            let shape = override_shape(key, &shape).unwrap_or(shape);
            entries.push(dense_entry(key, dtype, shape));
        };

        push_quantized(
            &mut entries,
            &format!("{p}.embed_tokens"),
            cfg.vocab_size,
            hidden,
        );
        dense(
            &mut entries,
            &format!("{p}.norm.weight"),
            StDtype::BF16,
            vec![hidden],
        );

        for li in 0..cfg.num_hidden_layers {
            let lp = format!("{p}.layers.{li}");
            dense(
                &mut entries,
                &format!("{lp}.input_layernorm.weight"),
                StDtype::BF16,
                vec![hidden],
            );
            dense(
                &mut entries,
                &format!("{lp}.post_attention_layernorm.weight"),
                StDtype::BF16,
                vec![hidden],
            );
            push_quantized(
                &mut entries,
                &format!("{lp}.mlp.gate_proj"),
                intermediate,
                hidden,
            );
            push_quantized(
                &mut entries,
                &format!("{lp}.mlp.up_proj"),
                intermediate,
                hidden,
            );
            push_quantized(
                &mut entries,
                &format!("{lp}.mlp.down_proj"),
                hidden,
                intermediate,
            );

            match cfg.layer_types[li] {
                LayerType::LinearAttention => {
                    let ap = format!("{lp}.linear_attn");
                    push_quantized(
                        &mut entries,
                        &format!("{ap}.in_proj_qkv"),
                        cfg.conv_dim(),
                        hidden,
                    );
                    push_quantized(
                        &mut entries,
                        &format!("{ap}.in_proj_z"),
                        cfg.value_dim(),
                        hidden,
                    );
                    push_quantized(
                        &mut entries,
                        &format!("{ap}.in_proj_a"),
                        value_heads,
                        hidden,
                    );
                    push_quantized(
                        &mut entries,
                        &format!("{ap}.in_proj_b"),
                        value_heads,
                        hidden,
                    );
                    dense(
                        &mut entries,
                        &format!("{ap}.A_log"),
                        StDtype::F32,
                        vec![value_heads],
                    );
                    dense(
                        &mut entries,
                        &format!("{ap}.dt_bias"),
                        StDtype::BF16,
                        vec![value_heads],
                    );
                    dense(
                        &mut entries,
                        &format!("{ap}.conv1d.weight"),
                        StDtype::BF16,
                        vec![cfg.conv_dim(), cfg.linear_conv_kernel_dim, 1],
                    );
                    dense(
                        &mut entries,
                        &format!("{ap}.norm.weight"),
                        StDtype::BF16,
                        vec![cfg.linear_value_head_dim],
                    );
                    push_quantized(
                        &mut entries,
                        &format!("{ap}.out_proj"),
                        hidden,
                        cfg.value_dim(),
                    );
                }
                LayerType::FullAttention => {
                    let ap = format!("{lp}.self_attn");
                    push_quantized(
                        &mut entries,
                        &format!("{ap}.q_proj"),
                        cfg.q_proj_out_dim(),
                        hidden,
                    );
                    push_quantized(&mut entries, &format!("{ap}.k_proj"), cfg.kv_dim(), hidden);
                    push_quantized(&mut entries, &format!("{ap}.v_proj"), cfg.kv_dim(), hidden);
                    dense(
                        &mut entries,
                        &format!("{ap}.q_norm.weight"),
                        StDtype::BF16,
                        vec![cfg.head_dim],
                    );
                    dense(
                        &mut entries,
                        &format!("{ap}.k_norm.weight"),
                        StDtype::BF16,
                        vec![cfg.head_dim],
                    );
                    push_quantized(
                        &mut entries,
                        &format!("{ap}.o_proj"),
                        hidden,
                        cfg.attn_query_dim(),
                    );
                }
            }
        }

        let views: Vec<(String, TensorView<'_>)> = entries
            .iter()
            .map(|(name, dt, shape, bytes)| {
                let view = TensorView::new(*dt, shape.clone(), bytes).expect("valid TensorView");
                (name.clone(), view)
            })
            .collect();
        let pairs: Vec<(&String, &TensorView<'_>)> = views.iter().map(|(n, v)| (n, v)).collect();
        safetensors::serialize(pairs, None).expect("serialize")
    }

    #[test]
    fn loads_synthetic_archive() {
        let device = Device::shared().expect("device");
        let config = Qwen35Config::from_json_bytes(TINY_CONFIG_JSON.as_bytes()).expect("config");
        let archive_bytes = synthesize_archive_bytes(&config);
        let archive = SafeTensors::deserialize(&archive_bytes).expect("deserialize");

        let weights = load_qwen35_weights(&archive, &config, &device).expect("load");

        assert_eq!(weights.layers.len(), 2);
        assert_eq!(weights.embed.shape().dims(), &[256, 64]);

        let Qwen35MixerWeights::LinearAttn(lin) = &weights.layers[0].mixer else {
            panic!("layer 0 must be LinearAttn");
        };
        assert_eq!(lin.in_proj_qkv.shape().dims(), &[192, 64]);

        let Qwen35MixerWeights::FullAttn(full) = &weights.layers[1].mixer else {
            panic!("layer 1 must be FullAttn");
        };
        assert_eq!(full.q_proj.shape().dims(), &[128, 64]);
    }

    #[test]
    fn rejects_wrong_shape() {
        let device = Device::shared().expect("device");
        let config = Qwen35Config::from_json_bytes(TINY_CONFIG_JSON.as_bytes()).expect("config");
        // Synthesize the same archive but with `norm.weight` at [63] instead of [64].
        let bad_key = format!("{TEXT_PREFIX}.norm.weight");
        let archive_bytes = synthesize_archive_bytes_with(&config, |key, _default| {
            (key == bad_key).then(|| vec![63])
        });
        let archive = SafeTensors::deserialize(&archive_bytes).expect("deserialize");

        assert!(load_qwen35_weights(&archive, &config, &device).is_err());
    }

    fn checkpoint_path() -> Option<PathBuf> {
        std::env::var_os("CIDER_QWEN35_CHECKPOINT_PATH").map(PathBuf::from)
    }

    #[test]
    fn loads_4b_checkpoint_with_validated_shapes() {
        let Some(dir) = checkpoint_path() else {
            eprintln!("skipping: CIDER_QWEN35_CHECKPOINT_PATH unset");
            return;
        };
        let config =
            Qwen35Config::from_json_bytes(&fs::read(dir.join("config.json")).unwrap()).unwrap();
        let bytes = fs::read(dir.join("model.safetensors")).unwrap();
        let archive = SafeTensors::deserialize(&bytes).unwrap();
        let device = Device::shared().unwrap();

        let weights = load_qwen35_weights(&archive, &config, &device).unwrap();

        assert_eq!(weights.layers.len(), config.num_hidden_layers);
        assert!(matches!(
            weights.layers[0].mixer,
            Qwen35MixerWeights::LinearAttn(_)
        ));
        assert!(matches!(
            weights.layers[3].mixer,
            Qwen35MixerWeights::FullAttn(_)
        ));
        for layer in &weights.layers {
            assert_eq!(layer.input_layernorm.shape().dims(), &[config.hidden_size]);
        }
        assert_eq!(
            weights.embed.shape().dims(),
            &[config.vocab_size, config.hidden_size]
        );
    }
}
