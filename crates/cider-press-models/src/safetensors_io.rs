//! Read tensor entries out of a parsed safetensors archive into the
//! runtime's [`Tensor`] / [`QuantizedWeight`] types.
//!
//! Architecture-agnostic plumbing — per-model key-mapping layers
//! (e.g. [`crate::qwen2`]) use these helpers.
//!
//! Two flavours:
//!
//! - [`read_dense_tensor`] for a single dense tensor under one key.
//! - [`read_quantized_weight`] for the MLX-style three-tensor
//!   `{base}.weight` (`u32` packed) / `{base}.scales` (`bf16`) /
//!   `{base}.biases` (`bf16`) triple.

use std::collections::HashMap;

use cider_press_runtime::{DType, Device, Quantization, QuantizedWeight, Tensor};
use safetensors::{Dtype as StDtype, SafeTensors, tensor::TensorView};

use crate::error::{Error, Result};

/// A source of named tensor views: either a single safetensors archive or a
/// multi-shard set resolved via `model.safetensors.index.json`.
///
/// The weight loaders read every tensor through this abstraction, so the same
/// loader code serves the single-archive checkpoints (Qwen2, the 4B dev
/// vehicle) and the 3-shard 27B checkpoint without branching.
pub trait TensorLookup {
    /// Return the [`TensorView`] for `key`, or [`Error::MissingKey`] if no
    /// source holds it.
    fn view(&self, key: &str) -> Result<TensorView<'_>>;
}

impl TensorLookup for SafeTensors<'_> {
    fn view(&self, key: &str) -> Result<TensorView<'_>> {
        self.tensor(key)
            .map_err(|_| Error::MissingKey { key: key.into() })
    }
}

/// A multi-shard safetensors set resolved via `model.safetensors.index.json`.
///
/// Borrows the shard byte buffers (`&'a [Vec<u8>]`) for the load's duration —
/// the caller owns the `Vec`s. A per-key index routes each lookup to the shard
/// that holds it (from `index.json`'s `weight_map`).
pub struct ShardedArchive<'a> {
    archives: Vec<SafeTensors<'a>>,
    key_to_shard: HashMap<String, usize>,
}

impl<'a> ShardedArchive<'a> {
    /// Build a sharded archive from the parsed `index.json`.
    ///
    /// `shard_names` are the distinct shard filenames (any order);
    /// `shard_bytes` are those shards' bytes in the **same order** as
    /// `shard_names`; `weight_map` is `index.json`'s `weight_map`
    /// (`tensor_key` → `shard_filename`).
    ///
    /// Errors if `shard_names` and `shard_bytes` differ in length, if a shard
    /// fails to deserialize, or if a `weight_map` filename isn't present in
    /// `shard_names`.
    pub fn new(
        shard_names: &[String],
        shard_bytes: &'a [Vec<u8>],
        weight_map: &HashMap<String, String>,
    ) -> Result<Self> {
        if shard_names.len() != shard_bytes.len() {
            return Err(Error::InvalidArgument(format!(
                "sharded archive: shard_names ({}) and shard_bytes ({}) length mismatch",
                shard_names.len(),
                shard_bytes.len(),
            )));
        }

        let name_to_index: HashMap<&str, usize> = shard_names
            .iter()
            .enumerate()
            .map(|(i, name)| (name.as_str(), i))
            .collect();

        let archives: Vec<SafeTensors<'a>> = shard_bytes
            .iter()
            .map(|bytes| SafeTensors::deserialize(bytes).map_err(Error::from))
            .collect::<Result<_>>()?;

        let mut key_to_shard = HashMap::with_capacity(weight_map.len());
        for (key, shard_name) in weight_map {
            let &index = name_to_index.get(shard_name.as_str()).ok_or_else(|| {
                Error::InvalidArgument(format!(
                    "sharded archive: weight_map references shard {shard_name:?} \
                     not present in shard_names",
                ))
            })?;
            key_to_shard.insert(key.clone(), index);
        }

        Ok(Self {
            archives,
            key_to_shard,
        })
    }
}

impl TensorLookup for ShardedArchive<'_> {
    fn view(&self, key: &str) -> Result<TensorView<'_>> {
        let &index = self
            .key_to_shard
            .get(key)
            .ok_or_else(|| Error::MissingKey { key: key.into() })?;
        self.archives[index]
            .tensor(key)
            .map_err(|_| Error::MissingKey { key: key.into() })
    }
}

/// Read a single dense tensor from `src` under `key`, upload its
/// bytes to `device`, and return a [`Tensor`] tagged with the matching
/// [`DType`].
///
/// The tensor's shape is taken verbatim from the safetensors entry's
/// `shape`; no reshape happens here.
pub fn read_dense_tensor(src: &impl TensorLookup, key: &str, device: &Device) -> Result<Tensor> {
    let view = src.view(key)?;
    let dtype = map_dtype(key, view.dtype())?;
    let shape: Vec<usize> = view.shape().to_vec();
    Tensor::from_bytes(device, view.data(), shape, dtype).map_err(Error::from)
}

/// Read a quantized weight matrix stored under three keys
/// `{base_name}.weight` (`u32` packed lanes), `{base_name}.scales`
/// (`bf16` per-group), and `{base_name}.biases` (`bf16` per-group),
/// and assemble them into a [`QuantizedWeight`].
///
/// The packed weight's safetensors shape is expected to be
/// `[N, K * bits / 32]`; the logical (pre-pack) shape `[N, K]` is
/// recovered from `bits`. Scales and biases must each have shape
/// `[N, K / group_size]`. Mismatches return
/// [`Error::ShapeMismatch`] with the offending key.
pub fn read_quantized_weight(
    src: &impl TensorLookup,
    base_name: &str,
    quantization: Quantization,
    device: &Device,
) -> Result<QuantizedWeight> {
    let w_key = format!("{base_name}.weight");
    let s_key = format!("{base_name}.scales");
    let b_key = format!("{base_name}.biases");

    let w_view = src.view(&w_key)?;
    let s_view = src.view(&s_key)?;
    let b_view = src.view(&b_key)?;

    if w_view.dtype() != StDtype::U32 {
        return Err(Error::UnsupportedDtype {
            key: w_key,
            dtype: format!("{:?}", w_view.dtype()),
        });
    }
    if s_view.dtype() != StDtype::BF16 {
        return Err(Error::UnsupportedDtype {
            key: s_key,
            dtype: format!("{:?}", s_view.dtype()),
        });
    }
    if b_view.dtype() != StDtype::BF16 {
        return Err(Error::UnsupportedDtype {
            key: b_key,
            dtype: format!("{:?}", b_view.dtype()),
        });
    }

    let w_shape = w_view.shape();
    if w_shape.len() != 2 {
        return Err(Error::InvalidArgument(format!(
            "quantized weight {w_key:?} must be rank-2 [N, packed_K]; got shape {w_shape:?}",
        )));
    }
    let n = w_shape[0];
    let packed_k = w_shape[1];
    let bits = quantization.bits() as usize;
    let bit_total = packed_k.checked_mul(32).ok_or_else(|| {
        Error::InvalidArgument(format!(
            "quantized weight {w_key:?}: packed_K * 32 overflows usize",
        ))
    })?;
    if bit_total % bits != 0 {
        return Err(Error::InvalidArgument(format!(
            "quantized weight {w_key:?}: packed_K ({packed_k}) × 32 not divisible by bits ({bits})",
        )));
    }
    let k = bit_total / bits;
    let group_size = quantization.group_size() as usize;
    if k % group_size != 0 {
        return Err(Error::InvalidArgument(format!(
            "quantized weight {w_key:?}: recovered K ({k}) not divisible by group_size \
             ({group_size})",
        )));
    }
    let expected_aux_shape = [n, k / group_size];
    if s_view.shape() != expected_aux_shape {
        return Err(Error::ShapeMismatch {
            key: s_key,
            expected: expected_aux_shape.to_vec(),
            actual: s_view.shape().to_vec(),
        });
    }
    if b_view.shape() != expected_aux_shape {
        return Err(Error::ShapeMismatch {
            key: b_key,
            expected: expected_aux_shape.to_vec(),
            actual: b_view.shape().to_vec(),
        });
    }

    QuantizedWeight::from_bytes(
        device,
        [n, k],
        quantization,
        w_view.data(),
        s_view.data(),
        b_view.data(),
    )
    .map_err(Error::from)
}

fn map_dtype(key: &str, dtype: StDtype) -> Result<DType> {
    match dtype {
        StDtype::F32 => Ok(DType::F32),
        StDtype::F16 => Ok(DType::F16),
        StDtype::BF16 => Ok(DType::BF16),
        StDtype::I32 => Ok(DType::I32),
        StDtype::U32 => Ok(DType::U32),
        other => Err(Error::UnsupportedDtype {
            key: key.into(),
            dtype: format!("{other:?}"),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cider_press_runtime::Quantization;
    use half::bf16;
    use safetensors::tensor::TensorView;

    /// Build a minimal in-memory safetensors archive for tests.
    fn build_archive(entries: &[(String, StDtype, Vec<usize>, Vec<u8>)]) -> Vec<u8> {
        let views: Vec<(String, TensorView<'_>)> = entries
            .iter()
            .map(|(name, dt, shape, bytes)| {
                let view =
                    TensorView::new(*dt, shape.clone(), bytes).expect("valid TensorView in test");
                (name.clone(), view)
            })
            .collect();
        let pairs: Vec<(&String, &TensorView<'_>)> = views.iter().map(|(n, v)| (n, v)).collect();
        safetensors::serialize(pairs, None).expect("serialize archive")
    }

    #[test]
    fn read_dense_tensor_round_trips_f32() {
        let device = Device::shared().expect("device");
        let payload: Vec<f32> = (0..16u8).map(|i| f32::from(i) * 0.5).collect();
        let bytes: Vec<u8> = payload.iter().flat_map(|x| x.to_le_bytes()).collect();

        let archive_bytes = build_archive(&[("w".into(), StDtype::F32, vec![4, 4], bytes)]);
        let archive = SafeTensors::deserialize(&archive_bytes).expect("deserialize");

        let t = read_dense_tensor(&archive, "w", &device).expect("load");
        assert_eq!(t.dtype(), DType::F32);
        assert_eq!(t.shape().dims(), &[4, 4]);
        assert_eq!(t.cpu_slice::<f32>().unwrap(), payload.as_slice());
    }

    #[test]
    fn read_dense_tensor_round_trips_bf16() {
        let device = Device::shared().expect("device");
        let payload: Vec<bf16> = (0..8u8).map(|i| bf16::from_f32(f32::from(i))).collect();
        let bytes: Vec<u8> = payload.iter().flat_map(|x| x.to_le_bytes()).collect();

        let archive_bytes = build_archive(&[("g".into(), StDtype::BF16, vec![8], bytes)]);
        let archive = SafeTensors::deserialize(&archive_bytes).expect("deserialize");

        let t = read_dense_tensor(&archive, "g", &device).expect("load");
        assert_eq!(t.dtype(), DType::BF16);
        assert_eq!(t.cpu_slice::<bf16>().unwrap(), payload.as_slice());
    }

    #[test]
    fn read_dense_tensor_missing_key_errors() {
        let device = Device::shared().expect("device");
        let archive_bytes = build_archive(&[(
            "present".into(),
            StDtype::F32,
            vec![1],
            0_f32.to_le_bytes().to_vec(),
        )]);
        let archive = SafeTensors::deserialize(&archive_bytes).expect("deserialize");
        let err = read_dense_tensor(&archive, "absent", &device).unwrap_err();
        assert!(matches!(err, Error::MissingKey { ref key } if key == "absent"));
    }

    #[test]
    fn read_dense_tensor_unsupported_dtype_errors() {
        let device = Device::shared().expect("device");
        let archive_bytes = build_archive(&[(
            "x".into(),
            StDtype::I64,
            vec![1],
            0_i64.to_le_bytes().to_vec(),
        )]);
        let archive = SafeTensors::deserialize(&archive_bytes).expect("deserialize");
        let err = read_dense_tensor(&archive, "x", &device).unwrap_err();
        assert!(matches!(err, Error::UnsupportedDtype { .. }));
    }

    /// Construct a synthetic q4-gs64 weight under three keys. Bytes are
    /// deterministic but not meaningful as quantization — we're
    /// exercising the loader's shape/dtype plumbing, not arithmetic.
    fn synthetic_q4_gs64_entries(
        base: &str,
        n: usize,
        k: usize,
    ) -> Vec<(String, StDtype, Vec<usize>, Vec<u8>)> {
        let packed_k = k / 8;
        let groups = k / 64;
        let w_bytes: Vec<u8> = (0..n * packed_k * 4)
            .map(|i| u8::try_from(i & 0xff).expect("masked"))
            .collect();
        let aux_words = n * groups;
        let s_bytes: Vec<u8> = (0..aux_words * 2)
            .map(|i| u8::try_from((i * 3) & 0xff).expect("masked"))
            .collect();
        let b_bytes: Vec<u8> = (0..aux_words * 2)
            .map(|i| u8::try_from((i * 7) & 0xff).expect("masked"))
            .collect();
        vec![
            (
                format!("{base}.weight"),
                StDtype::U32,
                vec![n, packed_k],
                w_bytes,
            ),
            (
                format!("{base}.scales"),
                StDtype::BF16,
                vec![n, groups],
                s_bytes,
            ),
            (
                format!("{base}.biases"),
                StDtype::BF16,
                vec![n, groups],
                b_bytes,
            ),
        ]
    }

    #[test]
    fn read_quantized_weight_round_trips_q4_gs64() {
        let device = Device::shared().expect("device");
        let n = 64;
        let k = 128;
        let entries = synthetic_q4_gs64_entries("ln", n, k);
        let archive_bytes = build_archive(&entries);
        let archive = SafeTensors::deserialize(&archive_bytes).expect("deserialize");

        let qw =
            read_quantized_weight(&archive, "ln", Quantization::Q4_GS64, &device).expect("load");
        assert_eq!(qw.shape().dims(), &[n, k]);
        assert_eq!(qw.quantization(), Quantization::Q4_GS64);
    }

    #[test]
    fn read_quantized_weight_missing_component_errors() {
        let device = Device::shared().expect("device");
        // Build only the weight + scales, omit biases.
        let mut entries = synthetic_q4_gs64_entries("ln", 64, 128);
        entries.retain(|(name, _, _, _)| !name.ends_with(".biases"));
        let archive_bytes = build_archive(&entries);
        let archive = SafeTensors::deserialize(&archive_bytes).expect("deserialize");

        let err =
            read_quantized_weight(&archive, "ln", Quantization::Q4_GS64, &device).unwrap_err();
        assert!(matches!(err, Error::MissingKey { ref key } if key == "ln.biases"));
    }

    #[test]
    fn read_quantized_weight_wrong_aux_shape_errors() {
        let device = Device::shared().expect("device");
        let mut entries = synthetic_q4_gs64_entries("ln", 64, 128);
        // Corrupt the scales shape but keep byte count consistent: pretend
        // it's [128, 1] instead of [64, 2].
        for (name, _, shape, _) in &mut entries {
            if name.ends_with(".scales") {
                *shape = vec![128, 1];
            }
        }
        let archive_bytes = build_archive(&entries);
        let archive = SafeTensors::deserialize(&archive_bytes).expect("deserialize");

        let err =
            read_quantized_weight(&archive, "ln", Quantization::Q4_GS64, &device).unwrap_err();
        assert!(matches!(err, Error::ShapeMismatch { .. }));
    }

    #[test]
    fn single_archive_view_round_trips() {
        let payload: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let bytes: Vec<u8> = payload.iter().flat_map(|x| x.to_le_bytes()).collect();
        let archive_bytes = build_archive(&[("solo".into(), StDtype::F32, vec![4], bytes.clone())]);
        let archive = SafeTensors::deserialize(&archive_bytes).expect("deserialize");

        let view = archive.view("solo").expect("view via trait");
        assert_eq!(view.dtype(), StDtype::F32);
        assert_eq!(view.shape(), &[4]);
        assert_eq!(view.data(), bytes.as_slice());

        let err = archive.view("nope").unwrap_err();
        assert!(matches!(err, Error::MissingKey { ref key } if key == "nope"));
    }

    #[test]
    fn sharded_archive_routes_keys_to_shards() {
        // Two single-key shards: A holds "alpha", B holds "beta".
        let alpha: Vec<f32> = vec![10.0, 11.0];
        let alpha_bytes: Vec<u8> = alpha.iter().flat_map(|x| x.to_le_bytes()).collect();
        let beta: Vec<bf16> = vec![
            bf16::from_f32(7.0),
            bf16::from_f32(8.0),
            bf16::from_f32(9.0),
        ];
        let beta_bytes: Vec<u8> = beta.iter().flat_map(|x| x.to_le_bytes()).collect();

        let shard_a =
            build_archive(&[("alpha".into(), StDtype::F32, vec![2], alpha_bytes.clone())]);
        let shard_b = build_archive(&[("beta".into(), StDtype::BF16, vec![3], beta_bytes.clone())]);

        let shard_names = vec!["sA.safetensors".to_string(), "sB.safetensors".to_string()];
        let shard_bytes = vec![shard_a, shard_b];

        let mut weight_map = HashMap::new();
        weight_map.insert("alpha".to_string(), "sA.safetensors".to_string());
        weight_map.insert("beta".to_string(), "sB.safetensors".to_string());

        let sharded =
            ShardedArchive::new(&shard_names, &shard_bytes, &weight_map).expect("build sharded");

        let av = sharded.view("alpha").expect("alpha");
        assert_eq!(av.dtype(), StDtype::F32);
        assert_eq!(av.shape(), &[2]);
        assert_eq!(av.data(), alpha_bytes.as_slice());

        let bv = sharded.view("beta").expect("beta");
        assert_eq!(bv.dtype(), StDtype::BF16);
        assert_eq!(bv.shape(), &[3]);
        assert_eq!(bv.data(), beta_bytes.as_slice());

        let err = sharded.view("missing").unwrap_err();
        assert!(matches!(err, Error::MissingKey { ref key } if key == "missing"));
    }

    #[test]
    fn sharded_archive_read_dense_via_generic_path() {
        let device = Device::shared().expect("device");
        let alpha: Vec<f32> = vec![1.5, 2.5, 3.5, 4.5];
        let alpha_bytes: Vec<u8> = alpha.iter().flat_map(|x| x.to_le_bytes()).collect();
        let shard_a = build_archive(&[("alpha".into(), StDtype::F32, vec![4], alpha_bytes)]);

        let shard_names = vec!["sA.safetensors".to_string()];
        let shard_bytes = vec![shard_a];
        let mut weight_map = HashMap::new();
        weight_map.insert("alpha".to_string(), "sA.safetensors".to_string());

        let sharded =
            ShardedArchive::new(&shard_names, &shard_bytes, &weight_map).expect("build sharded");

        let t = read_dense_tensor(&sharded, "alpha", &device).expect("read via sharded");
        assert_eq!(t.dtype(), DType::F32);
        assert_eq!(t.shape().dims(), &[4]);
        assert_eq!(t.cpu_slice::<f32>().unwrap(), alpha.as_slice());
    }

    #[test]
    fn sharded_archive_unknown_shard_name_errors() {
        let shard_a = build_archive(&[(
            "alpha".into(),
            StDtype::F32,
            vec![1],
            0_f32.to_le_bytes().to_vec(),
        )]);
        let shard_names = vec!["sA.safetensors".to_string()];
        let shard_bytes = vec![shard_a];
        let mut weight_map = HashMap::new();
        // weight_map points at a shard not in shard_names.
        weight_map.insert("alpha".to_string(), "sZ.safetensors".to_string());

        let result = ShardedArchive::new(&shard_names, &shard_bytes, &weight_map);
        assert!(matches!(result, Err(Error::InvalidArgument(_))));
    }
}
