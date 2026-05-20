//! Qwen2 weight struct and key-mapping loader.
//!
//! HF's safetensors layout for the pinned MLX-quantized
//! Qwen2.5-0.5B-Instruct-4bit checkpoint:
//!
//! ```text
//! model.embed_tokens.{weight, scales, biases}     # quantized, [V, D]
//! model.norm.weight                               # bf16,      [D]
//! model.layers.{i}.input_layernorm.weight         # bf16,      [D]
//! model.layers.{i}.post_attention_layernorm.weight # bf16,     [D]
//! model.layers.{i}.self_attn.q_proj.{w, s, b}     # quantized, [H_q * D_h, D]
//! model.layers.{i}.self_attn.q_proj.bias          # bf16,      [H_q * D_h]
//! model.layers.{i}.self_attn.k_proj.{w, s, b}     # quantized, [H_kv * D_h, D]
//! model.layers.{i}.self_attn.k_proj.bias          # bf16,      [H_kv * D_h]
//! model.layers.{i}.self_attn.v_proj.{w, s, b}     # quantized, [H_kv * D_h, D]
//! model.layers.{i}.self_attn.v_proj.bias          # bf16,      [H_kv * D_h]
//! model.layers.{i}.self_attn.o_proj.{w, s, b}     # quantized, [D, H_q * D_h]
//! model.layers.{i}.mlp.gate_proj.{w, s, b}        # quantized, [I, D]
//! model.layers.{i}.mlp.up_proj.{w, s, b}          # quantized, [I, D]
//! model.layers.{i}.mlp.down_proj.{w, s, b}        # quantized, [D, I]
//! ```
//!
//! `H_q * D_h == D` for Qwen2 (no head-dim expansion) but the loader
//! keeps the dims separate for clarity and to catch unexpected
//! variants.
//!
//! ### `.bias` vs `.biases` (footgun)
//!
//! The naming collides in an unfortunate way. The QKV projections
//! each have an additive *Linear bias*, stored at `…q_proj.bias`
//! (singular, bf16, dense). Every quantized matrix *also* has
//! per-group *quantization biases*, stored at `…q_proj.biases`
//! (plural, bf16, dense, shape `[N, K / group_size]`). The output
//! projection has only `.biases`, no `.bias`. This module's field
//! names use `q_bias` / `k_bias` / `v_bias` for the Linear-additive
//! variant; the quantization biases ride along inside each
//! [`QuantizedWeight`].

use cider_press_runtime::{DType, Device, Layout, Quantization, QuantizedWeight, Tensor};
use safetensors::SafeTensors;

use crate::error::{Error, Result};
use crate::qwen2::config::Qwen2Config;
use crate::safetensors_io::{read_dense_tensor, read_quantized_weight};

/// All weights for one Qwen2-family model.
#[derive(Debug)]
pub struct Qwen2Weights {
    /// Token embedding table, quantized. Logical shape `[vocab_size,
    /// hidden_size]`. Tied with the LM head when
    /// [`Qwen2Config::tie_word_embeddings`] is `true`.
    pub embed: QuantizedWeight,
    /// Final `RMSNorm` gamma. Dense bf16, shape `[hidden_size]`.
    pub norm: Tensor,
    /// Per-block weights. Length equals
    /// [`Qwen2Config::num_hidden_layers`].
    pub layers: Vec<Qwen2LayerWeights>,
}

/// Weights for one transformer block.
#[derive(Debug)]
pub struct Qwen2LayerWeights {
    /// Pre-attention `RMSNorm` gamma. Shape `[hidden_size]`.
    pub input_layernorm: Tensor,
    /// Pre-MLP `RMSNorm` gamma. Shape `[hidden_size]`.
    pub post_attention_layernorm: Tensor,
    /// Attention projections + biases.
    pub attention: Qwen2AttentionWeights,
    /// MLP (`SwiGLU`) projections.
    pub mlp: Qwen2MlpWeights,
}

/// Attention block weights.
#[derive(Debug)]
pub struct Qwen2AttentionWeights {
    /// Q projection. Logical `[H_q * D_h, D]`.
    pub q_proj: QuantizedWeight,
    /// K projection. Logical `[H_kv * D_h, D]`.
    pub k_proj: QuantizedWeight,
    /// V projection. Logical `[H_kv * D_h, D]`.
    pub v_proj: QuantizedWeight,
    /// Output projection. Logical `[D, H_q * D_h]`. No additive bias.
    pub o_proj: QuantizedWeight,
    /// Q linear bias. Dense bf16, shape `[H_q * D_h]`.
    pub q_bias: Tensor,
    /// K linear bias. Shape `[H_kv * D_h]`.
    pub k_bias: Tensor,
    /// V linear bias. Shape `[H_kv * D_h]`.
    pub v_bias: Tensor,
}

/// `SwiGLU` MLP block weights.
#[derive(Debug)]
pub struct Qwen2MlpWeights {
    /// Gate projection. Logical `[I, D]` (`I` = `intermediate_size`).
    pub gate_proj: QuantizedWeight,
    /// Up projection. Logical `[I, D]`.
    pub up_proj: QuantizedWeight,
    /// Down projection. Logical `[D, I]`.
    pub down_proj: QuantizedWeight,
}

/// Load every weight referenced in a Qwen2 checkpoint into typed
/// runtime tensors.
///
/// Validates per-tensor dtypes, shapes, and quantization metadata
/// against `config`'s predictions; mismatches return
/// [`Error::ShapeMismatch`] or [`Error::InvalidArgument`] with the
/// offending key. This is the "loaded bytes == safetensors bytes"
/// contract from the `QWEN_PATH` doc: anything that passes here is
/// observably identical (byte-for-byte) to what's in the archive.
pub fn load_qwen2_weights(
    archive: &SafeTensors<'_>,
    config: &Qwen2Config,
    device: &Device,
) -> Result<Qwen2Weights> {
    if !config.tie_word_embeddings {
        return Err(Error::InvalidArgument(
            "Qwen2Config.tie_word_embeddings=false is not supported yet; \
             lm_head loading is missing"
                .into(),
        ));
    }

    let quant = derive_quantization(config)?;
    let hidden = config.hidden_size;
    let head_dim = config.head_dim()?;
    let q_proj_dim = config.num_attention_heads * head_dim;
    let kv_proj_dim = config.num_key_value_heads * head_dim;
    let intermediate = config.intermediate_size;

    let embed = read_quantized_weight(archive, "model.embed_tokens", quant, device)?;
    expect_shape(
        "model.embed_tokens",
        &embed_weight_shape(&embed),
        &[config.vocab_size, hidden],
    )?;

    let norm = read_dense_tensor(archive, "model.norm.weight", device)?;
    expect_dense_bf16("model.norm.weight", &norm, &[hidden])?;

    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    for i in 0..config.num_hidden_layers {
        layers.push(load_layer(
            archive,
            device,
            quant,
            i,
            hidden,
            q_proj_dim,
            kv_proj_dim,
            intermediate,
        )?);
    }

    Ok(Qwen2Weights {
        embed,
        norm,
        layers,
    })
}

#[allow(clippy::too_many_arguments)]
fn load_layer(
    archive: &SafeTensors<'_>,
    device: &Device,
    quant: Quantization,
    i: usize,
    hidden: usize,
    q_proj_dim: usize,
    kv_proj_dim: usize,
    intermediate: usize,
) -> Result<Qwen2LayerWeights> {
    let prefix = format!("model.layers.{i}");

    let input_layernorm =
        read_dense_tensor(archive, &format!("{prefix}.input_layernorm.weight"), device)?;
    expect_dense_bf16(
        &format!("{prefix}.input_layernorm.weight"),
        &input_layernorm,
        &[hidden],
    )?;

    let post_attention_layernorm = read_dense_tensor(
        archive,
        &format!("{prefix}.post_attention_layernorm.weight"),
        device,
    )?;
    expect_dense_bf16(
        &format!("{prefix}.post_attention_layernorm.weight"),
        &post_attention_layernorm,
        &[hidden],
    )?;

    let attention = load_attention(
        archive,
        device,
        quant,
        &format!("{prefix}.self_attn"),
        hidden,
        q_proj_dim,
        kv_proj_dim,
    )?;
    let mlp = load_mlp(
        archive,
        device,
        quant,
        &format!("{prefix}.mlp"),
        hidden,
        intermediate,
    )?;

    Ok(Qwen2LayerWeights {
        input_layernorm,
        post_attention_layernorm,
        attention,
        mlp,
    })
}

fn load_attention(
    archive: &SafeTensors<'_>,
    device: &Device,
    quant: Quantization,
    prefix: &str,
    hidden: usize,
    q_proj_dim: usize,
    kv_proj_dim: usize,
) -> Result<Qwen2AttentionWeights> {
    let q_proj = read_quantized_weight(archive, &format!("{prefix}.q_proj"), quant, device)?;
    expect_q_shape(&format!("{prefix}.q_proj"), &q_proj, q_proj_dim, hidden)?;
    let k_proj = read_quantized_weight(archive, &format!("{prefix}.k_proj"), quant, device)?;
    expect_q_shape(&format!("{prefix}.k_proj"), &k_proj, kv_proj_dim, hidden)?;
    let v_proj = read_quantized_weight(archive, &format!("{prefix}.v_proj"), quant, device)?;
    expect_q_shape(&format!("{prefix}.v_proj"), &v_proj, kv_proj_dim, hidden)?;
    let o_proj = read_quantized_weight(archive, &format!("{prefix}.o_proj"), quant, device)?;
    expect_q_shape(&format!("{prefix}.o_proj"), &o_proj, hidden, q_proj_dim)?;

    let q_bias = read_dense_tensor(archive, &format!("{prefix}.q_proj.bias"), device)?;
    expect_dense_bf16(&format!("{prefix}.q_proj.bias"), &q_bias, &[q_proj_dim])?;
    let k_bias = read_dense_tensor(archive, &format!("{prefix}.k_proj.bias"), device)?;
    expect_dense_bf16(&format!("{prefix}.k_proj.bias"), &k_bias, &[kv_proj_dim])?;
    let v_bias = read_dense_tensor(archive, &format!("{prefix}.v_proj.bias"), device)?;
    expect_dense_bf16(&format!("{prefix}.v_proj.bias"), &v_bias, &[kv_proj_dim])?;

    Ok(Qwen2AttentionWeights {
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        q_bias,
        k_bias,
        v_bias,
    })
}

fn load_mlp(
    archive: &SafeTensors<'_>,
    device: &Device,
    quant: Quantization,
    prefix: &str,
    hidden: usize,
    intermediate: usize,
) -> Result<Qwen2MlpWeights> {
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
    Ok(Qwen2MlpWeights {
        gate_proj,
        up_proj,
        down_proj,
    })
}

fn derive_quantization(config: &Qwen2Config) -> Result<Quantization> {
    let q = config.quantization.ok_or_else(|| {
        Error::InvalidArgument(
            "Qwen2Config.quantization is None — this loader requires a pre-quantized checkpoint"
                .into(),
        )
    })?;
    Quantization::new(q.bits, q.group_size).map_err(Error::from)
}

fn embed_weight_shape(qw: &QuantizedWeight) -> [usize; 2] {
    let dims = qw.shape().dims();
    debug_assert_eq!(dims.len(), 2);
    [dims[0], dims[1]]
}

fn expect_shape(key: &str, actual: &[usize], expected: &[usize]) -> Result<()> {
    if actual != expected {
        return Err(Error::ShapeMismatch {
            key: key.into(),
            expected: expected.to_vec(),
            actual: actual.to_vec(),
        });
    }
    Ok(())
}

fn expect_q_shape(key: &str, qw: &QuantizedWeight, n: usize, k: usize) -> Result<()> {
    expect_shape(key, qw.shape().dims(), &[n, k])
}

fn expect_dense_bf16(key: &str, t: &Tensor, shape: &[usize]) -> Result<()> {
    if !matches!(t.layout(), Layout::Dense { .. }) {
        return Err(Error::InvalidArgument(format!(
            "{key:?}: expected dense layout, got quantized"
        )));
    }
    if t.dtype() != DType::BF16 {
        return Err(Error::UnsupportedDtype {
            key: key.into(),
            dtype: format!("{:?}", t.dtype()),
        });
    }
    expect_shape(key, t.shape().dims(), shape)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen2::config::Qwen2QuantizationConfig;
    use half::bf16;
    use safetensors::{Dtype as StDtype, tensor::TensorView};

    /// Tiny synthetic Qwen2 config that exercises every divisibility
    /// the loader cares about: gs=64 divides every K, bits=4 divides
    /// every K*32.
    fn synthetic_config() -> Qwen2Config {
        Qwen2Config {
            hidden_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            intermediate_size: 128,
            vocab_size: 128,
            max_position_embeddings: 32,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-6,
            tie_word_embeddings: true,
            quantization: Some(Qwen2QuantizationConfig {
                group_size: 64,
                bits: 4,
            }),
        }
    }

    /// Deterministic byte filler keyed off the safetensors key so each
    /// tensor has unique, predictable content — makes the post-load
    /// memcmp meaningful.
    fn fill_bytes(key: &str, len: usize) -> Vec<u8> {
        // FNV-1a 64-bit so reasonably distinct per key without dragging
        // in a hash dep.
        const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
        const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
        let mut h = FNV_OFFSET;
        for &b in key.as_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(FNV_PRIME);
        }
        (0..len)
            .map(|i| {
                let v = h.wrapping_add(i as u64).wrapping_mul(FNV_PRIME);
                u8::try_from(v & 0xff).expect("masked")
            })
            .collect()
    }

    type Entry = (String, StDtype, Vec<usize>, Vec<u8>);

    fn dense_entry(key: &str, shape: Vec<usize>) -> Entry {
        let count: usize = shape.iter().product();
        let bytes = fill_bytes(key, count * std::mem::size_of::<bf16>());
        (key.into(), StDtype::BF16, shape, bytes)
    }

    /// Push the three component entries for a quantized linear at
    /// `base` with logical `[n, k]`, q4-gs64.
    fn push_quantized(entries: &mut Vec<Entry>, base: &str, n: usize, k: usize) {
        let packed_k = k / 8;
        let groups = k / 64;
        let w_key = format!("{base}.weight");
        let s_key = format!("{base}.scales");
        let b_key = format!("{base}.biases");
        entries.push((
            w_key.clone(),
            StDtype::U32,
            vec![n, packed_k],
            fill_bytes(&w_key, n * packed_k * 4),
        ));
        entries.push((
            s_key.clone(),
            StDtype::BF16,
            vec![n, groups],
            fill_bytes(&s_key, n * groups * 2),
        ));
        entries.push((
            b_key.clone(),
            StDtype::BF16,
            vec![n, groups],
            fill_bytes(&b_key, n * groups * 2),
        ));
    }

    fn synthesize_archive_bytes(cfg: &Qwen2Config) -> Vec<u8> {
        let hidden = cfg.hidden_size;
        let head_dim = cfg.head_dim().unwrap();
        let q_dim = cfg.num_attention_heads * head_dim;
        let kv_dim = cfg.num_key_value_heads * head_dim;
        let i_dim = cfg.intermediate_size;

        let mut entries: Vec<Entry> = Vec::new();
        push_quantized(&mut entries, "model.embed_tokens", cfg.vocab_size, hidden);
        entries.push(dense_entry("model.norm.weight", vec![hidden]));

        for li in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{li}");
            entries.push(dense_entry(
                &format!("{p}.input_layernorm.weight"),
                vec![hidden],
            ));
            entries.push(dense_entry(
                &format!("{p}.post_attention_layernorm.weight"),
                vec![hidden],
            ));
            push_quantized(
                &mut entries,
                &format!("{p}.self_attn.q_proj"),
                q_dim,
                hidden,
            );
            push_quantized(
                &mut entries,
                &format!("{p}.self_attn.k_proj"),
                kv_dim,
                hidden,
            );
            push_quantized(
                &mut entries,
                &format!("{p}.self_attn.v_proj"),
                kv_dim,
                hidden,
            );
            push_quantized(
                &mut entries,
                &format!("{p}.self_attn.o_proj"),
                hidden,
                q_dim,
            );
            entries.push(dense_entry(
                &format!("{p}.self_attn.q_proj.bias"),
                vec![q_dim],
            ));
            entries.push(dense_entry(
                &format!("{p}.self_attn.k_proj.bias"),
                vec![kv_dim],
            ));
            entries.push(dense_entry(
                &format!("{p}.self_attn.v_proj.bias"),
                vec![kv_dim],
            ));
            push_quantized(&mut entries, &format!("{p}.mlp.gate_proj"), i_dim, hidden);
            push_quantized(&mut entries, &format!("{p}.mlp.up_proj"), i_dim, hidden);
            push_quantized(&mut entries, &format!("{p}.mlp.down_proj"), hidden, i_dim);
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
    fn loads_synthetic_archive_with_full_byte_round_trip() {
        let device = Device::shared().expect("device");
        let cfg = synthetic_config();
        let archive_bytes = synthesize_archive_bytes(&cfg);
        let archive = SafeTensors::deserialize(&archive_bytes).expect("deserialize");

        let weights = load_qwen2_weights(&archive, &cfg, &device).expect("load");

        assert_eq!(weights.layers.len(), cfg.num_hidden_layers);

        // Memcmp every dense tensor.
        let dense_check = |key: &str, t: &Tensor| {
            let arc = archive.tensor(key).expect("present");
            assert_eq!(t.cpu_bytes().expect("dense bytes"), arc.data(), "{key}");
        };
        dense_check("model.norm.weight", &weights.norm);
        for (li, layer) in weights.layers.iter().enumerate() {
            let p = format!("model.layers.{li}");
            dense_check(
                &format!("{p}.input_layernorm.weight"),
                &layer.input_layernorm,
            );
            dense_check(
                &format!("{p}.post_attention_layernorm.weight"),
                &layer.post_attention_layernorm,
            );
            dense_check(
                &format!("{p}.self_attn.q_proj.bias"),
                &layer.attention.q_bias,
            );
            dense_check(
                &format!("{p}.self_attn.k_proj.bias"),
                &layer.attention.k_bias,
            );
            dense_check(
                &format!("{p}.self_attn.v_proj.bias"),
                &layer.attention.v_bias,
            );
        }

        // Memcmp every quantized triple (weights/scales/biases).
        let q_check = |base: &str, qw: &QuantizedWeight| {
            let (w, s, b) = qw.component_bytes();
            assert_eq!(
                w,
                archive.tensor(&format!("{base}.weight")).unwrap().data(),
                "{base}.weight"
            );
            assert_eq!(
                s,
                archive.tensor(&format!("{base}.scales")).unwrap().data(),
                "{base}.scales"
            );
            assert_eq!(
                b,
                archive.tensor(&format!("{base}.biases")).unwrap().data(),
                "{base}.biases"
            );
        };
        q_check("model.embed_tokens", &weights.embed);
        for (li, layer) in weights.layers.iter().enumerate() {
            let p = format!("model.layers.{li}");
            q_check(&format!("{p}.self_attn.q_proj"), &layer.attention.q_proj);
            q_check(&format!("{p}.self_attn.k_proj"), &layer.attention.k_proj);
            q_check(&format!("{p}.self_attn.v_proj"), &layer.attention.v_proj);
            q_check(&format!("{p}.self_attn.o_proj"), &layer.attention.o_proj);
            q_check(&format!("{p}.mlp.gate_proj"), &layer.mlp.gate_proj);
            q_check(&format!("{p}.mlp.up_proj"), &layer.mlp.up_proj);
            q_check(&format!("{p}.mlp.down_proj"), &layer.mlp.down_proj);
        }
    }

    #[test]
    fn missing_key_errors_with_offending_name() {
        let device = Device::shared().expect("device");
        let cfg = synthetic_config();
        let mut archive_bytes = synthesize_archive_bytes(&cfg);
        // Corrupt the archive by re-serializing without one key.
        let archive = SafeTensors::deserialize(&archive_bytes).expect("deserialize");
        // Rebuild without `model.norm.weight` to force a MissingKey.
        let mut kept: Vec<(String, TensorView<'_>)> = Vec::new();
        for name in archive.names() {
            if name == "model.norm.weight" {
                continue;
            }
            kept.push((name.to_string(), archive.tensor(name).unwrap()));
        }
        let pairs: Vec<(&String, &TensorView<'_>)> = kept.iter().map(|(n, v)| (n, v)).collect();
        archive_bytes = safetensors::serialize(pairs, None).expect("re-serialize");
        let archive = SafeTensors::deserialize(&archive_bytes).expect("deserialize");

        let err = load_qwen2_weights(&archive, &cfg, &device).unwrap_err();
        match err {
            Error::MissingKey { key } => assert_eq!(key, "model.norm.weight"),
            other => panic!("expected MissingKey, got {other:?}"),
        }
    }

    #[test]
    fn missing_quantization_config_errors() {
        let device = Device::shared().expect("device");
        let mut cfg = synthetic_config();
        cfg.quantization = None;
        // We don't even need to build a valid archive — the
        // quantization-derivation check fires first.
        let archive_bytes = synthesize_archive_bytes(&synthetic_config());
        let archive = SafeTensors::deserialize(&archive_bytes).expect("deserialize");
        let err = load_qwen2_weights(&archive, &cfg, &device).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)));
    }
}
