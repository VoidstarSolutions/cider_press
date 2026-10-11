//! Qwen3.5 / Qwen3.6 text-model weights: load + shape-validate every per-layer
//! tensor from any [`TensorLookup`] (a single archive or a shard set). Raw keys live under
//! `language_model.model.` (the `model.` remap is mlx-lm's in-memory `sanitize`,
//! not present on disk). Vision (`vision_tower.*`) is skipped. Loader only —
//! no forward.
//!
//! ```text
//! language_model.model.embed_tokens.{weight, scales, biases}      # quant, [V, D]
//! language_model.lm_head.{weight, scales, biases}                 # quant, [V, D]  (untied 27B only)
//! language_model.model.norm.weight                                # bf16,  [D]
//! language_model.model.layers.{i}.input_layernorm.weight          # bf16,  [D]
//! language_model.model.layers.{i}.post_attention_layernorm.weight # bf16,  [D]
//! ```
//!
//! Each layer's mixer is one of two variants, selected by
//! [`Qwen35Config::layer_types`]:
//!
//! - **Linear attention** (Gated-DeltaNet): split `in_proj_{qkv,z,a,b}`,
//!   an `A_log` (f32 or bf16, upcast to f32), a `dt_bias`, a depthwise `conv1d.weight`, a head-dim
//!   `norm`, and an `out_proj`.
//! - **Full (gated) attention**: `q_proj` (query‖gate), `k_proj`, `v_proj`,
//!   QK `RMSNorm` gammas, and `o_proj`. `attention_bias = false`, so there
//!   are no additive q/k/v biases on disk.

use cider_press_runtime::{DType, Device, Layout, Quantization, QuantizedWeight, Tensor};

use crate::error::{Error, Result};
use crate::qwen3_5::config::{LayerType, Qwen35Config};
use crate::safetensors_io::{TensorLookup, read_dense_tensor, read_quantized_weight};

/// On-disk key prefix for every text-model tensor. The `model.`-only form is
/// mlx-lm's in-memory remap and is *not* what the checkpoint stores.
const TEXT_PREFIX: &str = "language_model.model";

/// All weights for one Qwen3.5/3.6 text model.
#[derive(Debug)]
#[non_exhaustive]
pub struct Qwen35Weights {
    /// Token embedding table, quantized. Logical shape `[vocab_size,
    /// hidden_size]`. On the 4B dev vehicle the LM head is tied to this table;
    /// the 27B target ships a separate [`Qwen35Weights::lm_head`].
    pub embed: QuantizedWeight,
    /// Separate (untied) LM head, quantized. Logical shape `[vocab_size,
    /// hidden_size]`. The loader decides tying from
    /// `config.tie_word_embeddings`: `None` means the head is tied to
    /// [`Qwen35Weights::embed`] (4B); `Some` is the separately loaded
    /// `language_model.lm_head` (27B). Consumers use it as-is.
    pub lm_head: Option<QuantizedWeight>,
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
    archive: &impl TensorLookup,
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

    // The tied 4B reuses `embed` as its head; the untied 27B ships a separate
    // `language_model.lm_head` (note: under `language_model.`, not the
    // `language_model.model.` text prefix).
    let lm_head = if config.tie_word_embeddings {
        None
    } else {
        let h = read_quantized_weight(archive, "language_model.lm_head", quant, device)?;
        expect_q_shape("lm_head", &h, config.vocab_size, hidden)?;
        Some(h)
    };

    Ok(Qwen35Weights {
        embed,
        lm_head,
        norm,
        layers,
    })
}

fn load_layer(
    archive: &impl TensorLookup,
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
    archive: &impl TensorLookup,
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
    archive: &impl TensorLookup,
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

    let a_log_key = format!("{prefix}.A_log");
    let a_log_raw = read_dense_tensor(archive, &a_log_key, device)?;
    // The 4B dev vehicle stores `A_log` as f32; the 27B target stores it bf16.
    // `compute_g` requires f32, so upcast bf16 -> f32 at load (exact widening;
    // identical to mlx-lm's `A_log.astype(float32)`).
    let a_log = match a_log_raw.dtype() {
        DType::F32 => a_log_raw,
        DType::BF16 => a_log_raw.cast(DType::F32)?,
        other => {
            return Err(Error::UnsupportedDtype {
                key: a_log_key,
                dtype: format!("{other:?}"),
            });
        }
    };
    expect_dense(&a_log_key, &a_log, DType::F32, &[value_heads])?;
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
    archive: &impl TensorLookup,
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
            "{key}: expected dense layout, got quantized"
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

/// Synthetic-archive builders shared by this module's loader tests and the
/// model-level tests that need a tiny, fully-loaded `Qwen35Model`.
#[cfg(test)]
pub(crate) mod test_support {
    use half::bf16;
    use safetensors::{Dtype as StDtype, tensor::TensorView};

    use crate::qwen3_5::config::{LayerType, Qwen35Config};

    use super::TEXT_PREFIX;

    /// Tiny Qwen3.5 config that satisfies every quantization divisibility
    /// constraint (hidden=64 divisible by `group_size=64`; one head each),
    /// with a `[linear_attention, full_attention]` layer pair.
    pub(crate) const TINY_CONFIG_JSON: &str = r#"{
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

    /// How synthesized tensors are filled: all-zero bytes (shape/loader tests)
    /// or deterministic pseudo-random values in numerically sane ranges (tests
    /// that run a forward and need non-degenerate activations).
    #[derive(Clone, Copy, Debug)]
    pub(crate) enum Fill {
        Zeros,
        Seeded(u64),
    }

    struct Filler {
        state: Option<u64>,
    }

    impl Filler {
        fn new(fill: Fill) -> Self {
            Self {
                state: match fill {
                    Fill::Zeros => None,
                    Fill::Seeded(seed) => Some(seed | 1),
                },
            }
        }

        /// xorshift64*; `None` when zero-filling.
        fn next_u64(&mut self) -> Option<u64> {
            let s = self.state.as_mut()?;
            *s ^= *s >> 12;
            *s ^= *s << 25;
            *s ^= *s >> 27;
            Some(s.wrapping_mul(0x2545_F491_4F6C_DD1D))
        }

        /// Uniform in `[lo, hi)`, or `0.0` when zero-filling.
        #[allow(clippy::cast_precision_loss)]
        fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
            self.next_u64().map_or(0.0, |x| {
                lo + (hi - lo) * ((x >> 40) as f32 / (1u64 << 24) as f32)
            })
        }

        fn bf16_bytes(&mut self, count: usize, lo: f32, hi: f32) -> Vec<u8> {
            bf16_le((0..count).map(|_| self.uniform(lo, hi)))
        }
    }

    fn bf16_le(vals: impl Iterator<Item = f32>) -> Vec<u8> {
        vals.flat_map(|v| bf16::from_f32(v).to_le_bytes()).collect()
    }

    /// A dense tensor. Seeded: RMSNorm gammas near 1, `A_log` (the only f32)
    /// in `[-1, 0.5)`, everything else (conv taps, `dt_bias`) in `[-0.5, 0.5)`.
    fn dense_entry(filler: &mut Filler, key: &str, dtype: StDtype, shape: Vec<usize>) -> Entry {
        let count: usize = shape.iter().product();
        let bytes = match dtype {
            StDtype::F32 => (0..count)
                .flat_map(|_| filler.uniform(-1.0, 0.5).to_le_bytes())
                .collect(),
            StDtype::BF16 if key.ends_with("norm.weight") => filler.bf16_bytes(count, 0.9, 1.1),
            StDtype::BF16 => filler.bf16_bytes(count, -0.5, 0.5),
            other => panic!("unexpected dense dtype {other:?}"),
        };
        (key.into(), dtype, shape, bytes)
    }

    /// Push the three component entries for a quantized linear at `base`
    /// with logical `[n, k]`, q4-gs64: `.weight` U32 `[n, k/8]`,
    /// `.scales`/`.biases` BF16 `[n, k/64]`. Seeded: random nibbles with
    /// `bias = -7.5 * scale`, so each dequantized group is zero-mean.
    fn push_quantized(
        filler: &mut Filler,
        entries: &mut Vec<Entry>,
        base: &str,
        n: usize,
        k: usize,
    ) {
        let packed_k = k / 8;
        let groups = k / 64;
        let weight: Vec<u8> = (0..n * packed_k)
            .flat_map(|_| {
                #[allow(clippy::cast_possible_truncation)]
                let word = filler.next_u64().map_or(0, |x| (x >> 32) as u32);
                word.to_le_bytes()
            })
            .collect();
        let scales: Vec<f32> = (0..n * groups)
            .map(|_| filler.uniform(0.01, 0.03))
            .collect();
        entries.push((
            format!("{base}.weight"),
            StDtype::U32,
            vec![n, packed_k],
            weight,
        ));
        entries.push((
            format!("{base}.scales"),
            StDtype::BF16,
            vec![n, groups],
            bf16_le(scales.iter().copied()),
        ));
        entries.push((
            format!("{base}.biases"),
            StDtype::BF16,
            vec![n, groups],
            bf16_le(scales.iter().map(|s| -7.5 * s)),
        ));
    }

    /// Synthesize an in-memory safetensors archive holding every key the
    /// loader reads for `cfg`, under the `language_model.model.` prefix, filled
    /// per `fill`. `with_lm_head` adds the untied `language_model.lm_head`
    /// quantized triple (`[vocab, hidden]`); `override_dense(key,
    /// default_shape)` may return a full replacement `(dtype, shape, bytes)`
    /// for a dense tensor (used by the wrong-shape negative test and the
    /// bf16-`A_log` test).
    #[allow(clippy::too_many_lines)]
    pub(crate) fn synthesize_archive_bytes_with(
        cfg: &Qwen35Config,
        with_lm_head: bool,
        fill: Fill,
        override_dense: impl Fn(&str, &[usize]) -> Option<(StDtype, Vec<usize>, Vec<u8>)>,
    ) -> Vec<u8> {
        let mut filler = Filler::new(fill);
        let filler = &mut filler;
        let hidden = cfg.hidden_size;
        let intermediate = cfg.intermediate_size;
        let value_heads = cfg.linear_num_value_heads;
        let p = TEXT_PREFIX;

        let mut entries: Vec<Entry> = Vec::new();
        let dense = |filler: &mut Filler,
                     entries: &mut Vec<Entry>,
                     key: &str,
                     dtype: StDtype,
                     shape: Vec<usize>| {
            match override_dense(key, &shape) {
                Some((dt, sh, bytes)) => entries.push((key.into(), dt, sh, bytes)),
                None => entries.push(dense_entry(filler, key, dtype, shape)),
            }
        };

        push_quantized(
            filler,
            &mut entries,
            &format!("{p}.embed_tokens"),
            cfg.vocab_size,
            hidden,
        );
        if with_lm_head {
            push_quantized(
                filler,
                &mut entries,
                "language_model.lm_head",
                cfg.vocab_size,
                hidden,
            );
        }
        dense(
            filler,
            &mut entries,
            &format!("{p}.norm.weight"),
            StDtype::BF16,
            vec![hidden],
        );

        for li in 0..cfg.num_hidden_layers {
            let lp = format!("{p}.layers.{li}");
            dense(
                filler,
                &mut entries,
                &format!("{lp}.input_layernorm.weight"),
                StDtype::BF16,
                vec![hidden],
            );
            dense(
                filler,
                &mut entries,
                &format!("{lp}.post_attention_layernorm.weight"),
                StDtype::BF16,
                vec![hidden],
            );
            push_quantized(
                filler,
                &mut entries,
                &format!("{lp}.mlp.gate_proj"),
                intermediate,
                hidden,
            );
            push_quantized(
                filler,
                &mut entries,
                &format!("{lp}.mlp.up_proj"),
                intermediate,
                hidden,
            );
            push_quantized(
                filler,
                &mut entries,
                &format!("{lp}.mlp.down_proj"),
                hidden,
                intermediate,
            );

            match cfg.layer_types[li] {
                LayerType::LinearAttention => {
                    let ap = format!("{lp}.linear_attn");
                    push_quantized(
                        filler,
                        &mut entries,
                        &format!("{ap}.in_proj_qkv"),
                        cfg.conv_dim(),
                        hidden,
                    );
                    push_quantized(
                        filler,
                        &mut entries,
                        &format!("{ap}.in_proj_z"),
                        cfg.value_dim(),
                        hidden,
                    );
                    push_quantized(
                        filler,
                        &mut entries,
                        &format!("{ap}.in_proj_a"),
                        value_heads,
                        hidden,
                    );
                    push_quantized(
                        filler,
                        &mut entries,
                        &format!("{ap}.in_proj_b"),
                        value_heads,
                        hidden,
                    );
                    dense(
                        filler,
                        &mut entries,
                        &format!("{ap}.A_log"),
                        StDtype::F32,
                        vec![value_heads],
                    );
                    dense(
                        filler,
                        &mut entries,
                        &format!("{ap}.dt_bias"),
                        StDtype::BF16,
                        vec![value_heads],
                    );
                    dense(
                        filler,
                        &mut entries,
                        &format!("{ap}.conv1d.weight"),
                        StDtype::BF16,
                        vec![cfg.conv_dim(), cfg.linear_conv_kernel_dim, 1],
                    );
                    dense(
                        filler,
                        &mut entries,
                        &format!("{ap}.norm.weight"),
                        StDtype::BF16,
                        vec![cfg.linear_value_head_dim],
                    );
                    push_quantized(
                        filler,
                        &mut entries,
                        &format!("{ap}.out_proj"),
                        hidden,
                        cfg.value_dim(),
                    );
                }
                LayerType::FullAttention => {
                    let ap = format!("{lp}.self_attn");
                    push_quantized(
                        filler,
                        &mut entries,
                        &format!("{ap}.q_proj"),
                        cfg.q_proj_out_dim(),
                        hidden,
                    );
                    push_quantized(
                        filler,
                        &mut entries,
                        &format!("{ap}.k_proj"),
                        cfg.kv_dim(),
                        hidden,
                    );
                    push_quantized(
                        filler,
                        &mut entries,
                        &format!("{ap}.v_proj"),
                        cfg.kv_dim(),
                        hidden,
                    );
                    dense(
                        filler,
                        &mut entries,
                        &format!("{ap}.q_norm.weight"),
                        StDtype::BF16,
                        vec![cfg.head_dim],
                    );
                    dense(
                        filler,
                        &mut entries,
                        &format!("{ap}.k_norm.weight"),
                        StDtype::BF16,
                        vec![cfg.head_dim],
                    );
                    push_quantized(
                        filler,
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
}

#[cfg(test)]
mod tests {
    use super::test_support::{Fill, TINY_CONFIG_JSON, synthesize_archive_bytes_with};
    use super::*;
    use crate::safetensors_io::ShardSet;
    use cider_press_runtime::Device;
    use safetensors::{Dtype as StDtype, SafeTensors};
    use std::fs;

    use crate::qwen3_5::config::Qwen35Config;

    /// A zero-filled tied-head archive for `cfg`.
    fn synthesize_archive_bytes(cfg: &Qwen35Config) -> Vec<u8> {
        synthesize_archive_bytes_with(cfg, false, Fill::Zeros, |_, _| None)
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
        // Tied config (default): the head reuses `embed`, no separate lm_head.
        assert!(weights.lm_head.is_none());

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
        let archive_bytes =
            synthesize_archive_bytes_with(&config, false, Fill::Zeros, |key, _default| {
                (key == bad_key).then(|| (StDtype::BF16, vec![63], vec![0u8; 63 * 2]))
            });
        let archive = SafeTensors::deserialize(&archive_bytes).expect("deserialize");

        assert!(load_qwen35_weights(&archive, &config, &device).is_err());
    }

    /// The tiny config with `tie_word_embeddings` flipped to `false` — the
    /// 27B's untied case, which makes the loader require a separate
    /// `language_model.lm_head`. Built from [`TINY_CONFIG_JSON`] so the two
    /// stay in sync.
    fn tiny_config_untied() -> Qwen35Config {
        let json = TINY_CONFIG_JSON.replace(
            "\"tie_word_embeddings\": true",
            "\"tie_word_embeddings\": false",
        );
        Qwen35Config::from_json_bytes(json.as_bytes()).expect("config")
    }

    #[test]
    fn loads_untied_lm_head() {
        let device = Device::shared().expect("device");
        let config = tiny_config_untied();
        assert!(!config.tie_word_embeddings);
        // Archive includes the separate `language_model.lm_head` triple.
        let archive_bytes = synthesize_archive_bytes_with(&config, true, Fill::Zeros, |_, _| None);
        let archive = SafeTensors::deserialize(&archive_bytes).expect("deserialize");

        let weights = load_qwen35_weights(&archive, &config, &device).expect("load");

        let lm_head = weights
            .lm_head
            .expect("untied config must load a separate lm_head");
        assert_eq!(lm_head.shape().dims(), &[256, 64]);
    }

    #[test]
    fn untied_missing_lm_head_errors() {
        let device = Device::shared().expect("device");
        let config = tiny_config_untied();
        // Untied config, but the archive LACKS `language_model.lm_head.*`.
        let archive_bytes = synthesize_archive_bytes_with(&config, false, Fill::Zeros, |_, _| None);
        let archive = SafeTensors::deserialize(&archive_bytes).expect("deserialize");

        let err = load_qwen35_weights(&archive, &config, &device)
            .expect_err("untied config without lm_head must error");
        assert!(matches!(err, Error::MissingKey { ref key } if key.contains("lm_head")));
    }

    /// Truncate an f32 to bf16 (drop the low 16 mantissa bits), returning the
    /// little-endian bf16 bytes and the value widened back to f32. The widened
    /// value is exactly the bf16 (low 16 bits zero), so an `A_log` built from
    /// these bytes must load bit-identically to `widened` after the upcast.
    fn f32_to_bf16_le(v: f32) -> ([u8; 2], f32) {
        let bits = v.to_bits();
        let hi = (bits >> 16) as u16;
        let widened = f32::from_bits(u32::from(hi) << 16);
        (hi.to_le_bytes(), widened)
    }

    /// The 27B stores `A_log` bf16; the loader must accept it and upcast to f32
    /// (`compute_g`'s invariant). Build an archive whose `A_log` is bf16 with
    /// known values and assert it loads as f32 with the exactly-widened values.
    #[test]
    fn bf16_a_log_upcasts_to_f32() {
        let device = Device::shared().expect("device");
        let config = Qwen35Config::from_json_bytes(TINY_CONFIG_JSON.as_bytes()).expect("config");
        let value_heads = config.linear_num_value_heads;
        // Layer 0 is LinearAttention in the tiny config.
        let a_log_key = format!("{TEXT_PREFIX}.layers.0.linear_attn.A_log");

        // Distinct, exactly-bf16-representable-after-truncation source values.
        let src: Vec<f32> = (0..u16::try_from(value_heads).unwrap())
            .map(|i| -1.5 - f32::from(i))
            .collect();
        let mut bf16_bytes = Vec::with_capacity(value_heads * 2);
        let mut widened = Vec::with_capacity(value_heads);
        for &v in &src {
            let (le, w) = f32_to_bf16_le(v);
            bf16_bytes.extend_from_slice(&le);
            widened.push(w);
        }

        let key = a_log_key.clone();
        let bytes = bf16_bytes.clone();
        let archive_bytes =
            synthesize_archive_bytes_with(&config, false, Fill::Zeros, |k, _default| {
                (k == key).then(|| (StDtype::BF16, vec![value_heads], bytes.clone()))
            });
        let archive = SafeTensors::deserialize(&archive_bytes).expect("deserialize");

        let weights = load_qwen35_weights(&archive, &config, &device).expect("load bf16 A_log");

        let Qwen35MixerWeights::LinearAttn(lin) = &weights.layers[0].mixer else {
            panic!("layer 0 must be LinearAttn");
        };
        assert_eq!(
            lin.a_log.dtype(),
            DType::F32,
            "bf16 A_log must be upcast to f32 at load"
        );
        assert_eq!(lin.a_log.shape().dims(), &[value_heads]);
        lin.a_log.eval().expect("eval a_log");
        let host = lin.a_log.cpu_to_vec::<f32>().expect("pull a_log to host");
        assert_eq!(
            host, widened,
            "upcast values must equal the bf16 source widened to f32 (exact)"
        );
    }

    /// The 4B path (f32 `A_log`) stays byte-identical: loaded as-is, no cast.
    #[test]
    fn f32_a_log_loads_as_f32() {
        let device = Device::shared().expect("device");
        let config = Qwen35Config::from_json_bytes(TINY_CONFIG_JSON.as_bytes()).expect("config");
        let archive_bytes = synthesize_archive_bytes(&config);
        let archive = SafeTensors::deserialize(&archive_bytes).expect("deserialize");

        let weights = load_qwen35_weights(&archive, &config, &device).expect("load");
        let Qwen35MixerWeights::LinearAttn(lin) = &weights.layers[0].mixer else {
            panic!("layer 0 must be LinearAttn");
        };
        assert_eq!(lin.a_log.dtype(), DType::F32);
    }

    #[test]
    #[ignore = "requires a real checkpoint via CIDER_QWEN35_CHECKPOINT_PATH"]
    fn loads_4b_checkpoint_with_validated_shapes() {
        let dir = cider_press_test_utils::qwen35_4b_checkpoint_dir();
        let config =
            Qwen35Config::from_json_bytes(&fs::read(dir.join("config.json")).unwrap()).unwrap();
        let shards = ShardSet::read(&dir).unwrap();
        let archive = shards.archive().unwrap();
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
