//! Quantized-weight newtype and its op constructors.
//!
//! Quantized weights are a different beast than dense tensors: they
//! carry three buffers (packed lanes, per-group scales, per-group
//! biases), they require a [`Quantization`] descriptor to interpret,
//! and they only participate in a small set of ops (chiefly
//! [`Tensor::quantized_matmul`]). The design doc's recommendation:
//! encode the "this is a quantized weight" precondition in the type
//! system at the op boundary, so op signatures stay infallible at the
//! type level and end-users can't accidentally hand a dense tensor to
//! the quantized-matmul path.
//!
//! [`QuantizedWeight`] is a thin newtype around [`Tensor`]: the
//! underlying graph node carries [`Layout::Quantized`] and the
//! `LeafStorage::Quantized` variant. The escape hatch
//! [`QuantizedWeight::tensor`] returns the inner handle for callers
//! that need to mix it into generic-tensor APIs.

#![doc = include_str!("../../../docs/inference/quantized-matmul.md")]

use crate::Device;
use crate::Layout;
use crate::Quantization;
use crate::Shape;
use crate::Tensor;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::tensor::LeafStorage;

/// A quantized weight matrix, ready for [`Tensor::quantized_matmul`].
///
/// Construct via [`QuantizedWeight::from_bytes`]. Cheap to clone
/// (same `Arc<TensorInner>` underlying the wrapped [`Tensor`]).
#[derive(Clone)]
pub struct QuantizedWeight {
    tensor: Tensor,
    /// Logical inner dim. Equals the physical `shape()[1]` except for
    /// K-padded weights, where the trailing `k_physical - k_logical`
    /// columns are zero-scale/zero-bias pad groups that contribute
    /// exactly 0 to the matvec — activations validate against this,
    /// the kernel dispatches with the physical K.
    k_logical: usize,
}

impl QuantizedWeight {
    /// Construct a quantized weight from the three raw byte buffers
    /// MLX's `qmv` kernel expects.
    ///
    /// `shape` is the *logical* (unpacked) weight shape, conventionally
    /// `[N, K]` for an inference linear layer (output dim, input dim).
    /// Byte-length expectations:
    ///
    /// - `weights.len() == N * K * bits / 32 * 4`  (packed `u32` words)
    /// - `scales.len()  == N * K / group_size * 2` (bf16)
    /// - `biases.len()  == N * K / group_size * 2` (bf16)
    ///
    /// Validated before upload; mismatches return
    /// [`Error::InvalidArgument`].
    pub fn from_bytes(
        device: &Device,
        shape: impl Into<Shape>,
        quantization: Quantization,
        weights: &[u8],
        scales: &[u8],
        biases: &[u8],
    ) -> Result<Self> {
        let shape = shape.into();
        let k_physical = if shape.rank() == 2 {
            shape.dims()[1]
        } else {
            0
        };
        Self::from_bytes_inner(
            device,
            shape,
            k_physical,
            quantization,
            weights,
            scales,
            biases,
        )
    }

    /// [`Self::from_bytes`], but the weight is K-padded: `shape` is the
    /// physical `[N, k_physical]`; activations of width `k_logical` are
    /// accepted. The caller guarantees columns `k_logical..k_physical`
    /// decode to zero (zero-scale, zero-bias pad groups) — see
    /// `pad_quantized_k` in the models crate.
    pub fn from_bytes_k_padded(
        device: &Device,
        shape: impl Into<Shape>,
        k_logical: usize,
        quantization: Quantization,
        weights: &[u8],
        scales: &[u8],
        biases: &[u8],
    ) -> Result<Self> {
        let shape = shape.into();
        let k_physical = if shape.rank() == 2 {
            shape.dims()[1]
        } else {
            0
        };
        let group_size = quantization.group_size() as usize;
        if k_logical > k_physical {
            return Err(Error::InvalidArgument(format!(
                "QuantizedWeight: k_logical={k_logical} > k_physical={k_physical}",
            )));
        }
        if group_size > 0 && k_logical % group_size != 0 {
            return Err(Error::InvalidArgument(format!(
                "QuantizedWeight: k_logical={k_logical} must be a multiple of \
                 group_size={group_size}",
            )));
        }
        Self::from_bytes_inner(
            device,
            shape,
            k_logical,
            quantization,
            weights,
            scales,
            biases,
        )
    }

    fn from_bytes_inner(
        device: &Device,
        shape: Shape,
        k_logical: usize,
        quantization: Quantization,
        weights: &[u8],
        scales: &[u8],
        biases: &[u8],
    ) -> Result<Self> {
        quantization.validate()?;
        if shape.rank() != 2 {
            return Err(Error::InvalidArgument(format!(
                "QuantizedWeight: shape must be rank 2 ([N, K]); got rank {} ({:?})",
                shape.rank(),
                shape,
            )));
        }
        // K alignment is stricter than N*K alignment: components() exposes
        // per-row packed weights (`[N, K*bits/32]`) and per-row groups
        // (`[N, K/group_size]`), both of which need K itself to be aligned.
        // packed_word_count / group_count only check the total below.
        let k = shape.dims()[1];
        let group_size = quantization.group_size() as usize;
        if k % group_size != 0 {
            return Err(Error::InvalidArgument(format!(
                "QuantizedWeight: inner dim K={k} must be a multiple of \
                 group_size={group_size} (per-row groups must divide evenly)",
            )));
        }
        let bits = quantization.bits() as usize;
        if (k * bits) % 32 != 0 {
            return Err(Error::InvalidArgument(format!(
                "QuantizedWeight: K*bits={} must be a multiple of 32 \
                 (per-row packed u32 words must divide evenly)",
                k * bits,
            )));
        }
        let elem_count = shape.elem_count();
        let expected_w_words = quantization.packed_word_count(elem_count)?;
        let expected_w_bytes = expected_w_words * core::mem::size_of::<u32>();
        if weights.len() != expected_w_bytes {
            return Err(Error::InvalidArgument(format!(
                "QuantizedWeight: weights.len()={} != {expected_w_bytes} \
                 (N*K*bits/32 = {expected_w_words} u32 words for shape {:?} with {} bits)",
                weights.len(),
                shape,
                quantization.bits(),
            )));
        }
        let groups = quantization.group_count(elem_count)?;
        let expected_aux_bytes = groups * core::mem::size_of::<half::bf16>();
        if scales.len() != expected_aux_bytes {
            return Err(Error::InvalidArgument(format!(
                "QuantizedWeight: scales.len()={} != {expected_aux_bytes} \
                 (N*K/group_size = {groups} bf16 values)",
                scales.len(),
            )));
        }
        if biases.len() != expected_aux_bytes {
            return Err(Error::InvalidArgument(format!(
                "QuantizedWeight: biases.len()={} != {expected_aux_bytes} \
                 (N*K/group_size = {groups} bf16 values)",
                biases.len(),
            )));
        }

        let weights_buf = device.kernels().upload::<u8>(weights)?;
        let scales_buf = device.kernels().upload::<u8>(scales)?;
        let biases_buf = device.kernels().upload::<u8>(biases)?;

        let layout = Layout::Quantized(quantization);
        let tensor =
            Tensor::host_quantized_leaf(device, shape, layout, weights_buf, scales_buf, biases_buf);
        Ok(Self { tensor, k_logical })
    }

    /// The underlying [`Tensor`] handle. Use this when an API wants a
    /// generic tensor (e.g. the op graph stores `Tensor`s uniformly).
    #[must_use]
    pub fn tensor(&self) -> &Tensor {
        &self.tensor
    }

    /// Physical shape `[N, K]` of the quantized weight. For K-padded weights
    /// this is the padded (physical) K; activations must match `k_logical()`,
    /// not `shape()[1]`.
    #[must_use]
    pub fn shape(&self) -> &Shape {
        self.tensor.shape()
    }

    /// Logical inner dim (K). For unpadded weights this equals `shape()[1]`;
    /// for K-padded weights it is the original K that activations must match —
    /// the kernel reads the physical (padded) K from the weight tensor's shape.
    #[must_use]
    pub fn k_logical(&self) -> usize {
        self.k_logical
    }

    /// Host-readable byte views of the three component buffers
    /// (packed weights, scales, biases). Safe to call: the leaf is
    /// always materialized for [`QuantizedWeight::from_bytes`]
    /// constructions, and the cache invariant rules out concurrent
    /// GPU writes (same guarantee as [`Tensor::cpu_bytes`]).
    ///
    /// Primarily intended for round-trip / parity tests in higher
    /// layers — the runtime itself reads these buffers via typed
    /// reinterpret at dispatch time, not by hand.
    #[must_use]
    pub fn component_bytes(&self) -> (&[u8], &[u8], &[u8]) {
        let storage = self
            .tensor
            .inner
            .cache
            .get()
            .expect("QuantizedWeight always carries a materialized leaf by construction");
        match storage {
            LeafStorage::Quantized {
                weights,
                scales,
                biases,
            } => {
                // SAFETY: see method docs.
                let w = unsafe { weights.as_slice() };
                let s = unsafe { scales.as_slice() };
                let b = unsafe { biases.as_slice() };
                (w, s, b)
            }
            LeafStorage::Dense(_) => {
                unreachable!("QuantizedWeight always wraps LeafStorage::Quantized")
            }
        }
    }

    /// Expose the three packed buffers as standalone dense `Tensor`s
    /// with their natural dtypes and packed shapes:
    ///
    /// - `w_q`: `[N, K * bits / 32]` `U32`
    /// - `scales`, `biases`: `[N, K / group_size]` `BF16`
    ///
    /// Each returned tensor shares the same underlying `MTLBuffer`
    /// as this `QuantizedWeight` (refcount-bumped — no copy). Use
    /// these as inputs to [`Tensor::gather`] or
    /// [`Tensor::dequantize_affine`] when composing
    /// embedding-style lookups.
    #[must_use]
    pub fn components(&self) -> (Tensor, Tensor, Tensor) {
        let device = self
            .tensor
            .device()
            .expect("QuantizedWeight always carries a device");
        let storage = self
            .tensor
            .inner
            .cache
            .get()
            .expect("QuantizedWeight always carries a materialized leaf by construction");
        let (weights, scales, biases) = match storage {
            LeafStorage::Quantized {
                weights,
                scales,
                biases,
            } => (weights, scales, biases),
            LeafStorage::Dense(_) => {
                unreachable!("QuantizedWeight always wraps LeafStorage::Quantized")
            }
        };

        let logical = self.tensor.shape().dims();
        let n = logical[0];
        let k = logical[1];
        let q = self.quantization();
        let group_size = q.group_size() as usize;
        let bits = q.bits() as usize;

        let packed_cols = k * bits / 32;
        let groups_per_row = k / group_size;

        // SAFETY: reinterpret_as bumps the underlying MTLBuffer's
        // refcount but doesn't change the bytes. The Rust element
        // counts implied by Buffer<T>::len() come out right because
        // `from_bytes` validated the byte counts: weights = N*K*bits/8
        // bytes = packed_cols*N*4 bytes (U32 stride), scales/biases =
        // groups_per_row*N*2 bytes (BF16 stride).
        let w_buf = unsafe { weights.reinterpret_as::<u8>() };
        let s_buf = unsafe { scales.reinterpret_as::<u8>() };
        let b_buf = unsafe { biases.reinterpret_as::<u8>() };

        let w_shape = Shape::from([n, packed_cols]);
        let aux_shape = Shape::from([n, groups_per_row]);

        let dense = |shape: &Shape| Layout::Dense {
            strides: crate::Strides::contiguous(shape),
        };

        let w_tensor =
            Tensor::host_leaf(device, w_shape.clone(), DType::U32, dense(&w_shape), w_buf);
        let s_tensor = Tensor::host_leaf(
            device,
            aux_shape.clone(),
            DType::BF16,
            dense(&aux_shape),
            s_buf,
        );
        let b_tensor = Tensor::host_leaf(
            device,
            aux_shape.clone(),
            DType::BF16,
            dense(&aux_shape),
            b_buf,
        );
        (w_tensor, s_tensor, b_tensor)
    }

    /// The quantization descriptor (bits + group size).
    #[must_use]
    pub fn quantization(&self) -> Quantization {
        match self.tensor.layout() {
            Layout::Quantized(q) => *q,
            Layout::Dense { .. } => {
                unreachable!("QuantizedWeight always wraps a Layout::Quantized tensor")
            }
        }
    }
}

impl core::fmt::Debug for QuantizedWeight {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("QuantizedWeight")
            .field("shape", self.shape())
            .field("k_logical", &self.k_logical)
            .field("quantization", &self.quantization())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_sizes(n: usize, k: usize, q: Quantization) -> (usize, usize) {
        let weight_bytes = q.packed_word_count(n * k).unwrap() * core::mem::size_of::<u32>();
        let aux_bytes = q.group_count(n * k).unwrap() * core::mem::size_of::<half::bf16>();
        (weight_bytes, aux_bytes)
    }

    #[test]
    fn from_bytes_constructs_quantized_leaf() {
        let device = Device::shared().expect("device");
        let q = Quantization::Q4_GS64;
        let (w_bytes, aux_bytes) = fixture_sizes(64, 64, q);
        let w = vec![0u8; w_bytes];
        let s = vec![0u8; aux_bytes];
        let b = vec![0u8; aux_bytes];

        let qw = QuantizedWeight::from_bytes(&device, [64, 64], q, &w, &s, &b).expect("build");

        assert_eq!(qw.shape(), &Shape::from([64, 64]));
        assert_eq!(qw.quantization(), q);
        assert!(matches!(qw.tensor().layout(), Layout::Quantized(_)));
        assert!(qw.tensor().is_materialized());
        // Quantized tensors don't have a single cpu_bytes view.
        assert!(qw.tensor().cpu_bytes().is_none());
    }

    #[test]
    fn from_bytes_rejects_wrong_rank() {
        let device = Device::shared().expect("device");
        let q = Quantization::Q4_GS64;
        let (w_bytes, aux_bytes) = fixture_sizes(64, 64, q);
        let err = QuantizedWeight::from_bytes(
            &device,
            [64 * 64],
            q,
            &vec![0; w_bytes],
            &vec![0; aux_bytes],
            &vec![0; aux_bytes],
        )
        .unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)));
    }

    #[test]
    fn from_bytes_rejects_size_mismatch() {
        let device = Device::shared().expect("device");
        let q = Quantization::Q4_GS64;
        let (_w_bytes, aux_bytes) = fixture_sizes(64, 64, q);
        let err = QuantizedWeight::from_bytes(
            &device,
            [64, 64],
            q,
            &[0u8; 17], // wrong size
            &vec![0; aux_bytes],
            &vec![0; aux_bytes],
        )
        .unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)));
    }

    // from_bytes_k_padded tests: shape [8, 128], group_size=64, bits=4
    // w_bytes = 8 * 128 * 4 / 8 = 512; aux_bytes = 8 * (128/64) * 2 = 32
    const K_PAD_N: usize = 8;
    const K_PAD_K_PHYSICAL: usize = 128;

    #[test]
    fn from_bytes_k_padded_rejects_k_logical_gt_k_physical() {
        let device = Device::shared().expect("device");
        let q = Quantization::Q4_GS64;
        let (w_bytes, aux_bytes) = fixture_sizes(K_PAD_N, K_PAD_K_PHYSICAL, q);
        let err = QuantizedWeight::from_bytes_k_padded(
            &device,
            [K_PAD_N, K_PAD_K_PHYSICAL],
            K_PAD_K_PHYSICAL + 64, // k_logical > k_physical
            q,
            &vec![0u8; w_bytes],
            &vec![0u8; aux_bytes],
            &vec![0u8; aux_bytes],
        )
        .unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)));
    }

    #[test]
    fn from_bytes_k_padded_rejects_k_logical_not_multiple_of_group_size() {
        let device = Device::shared().expect("device");
        let q = Quantization::Q4_GS64;
        let (w_bytes, aux_bytes) = fixture_sizes(K_PAD_N, K_PAD_K_PHYSICAL, q);
        let err = QuantizedWeight::from_bytes_k_padded(
            &device,
            [K_PAD_N, K_PAD_K_PHYSICAL],
            33, // not a multiple of group_size=64
            q,
            &vec![0u8; w_bytes],
            &vec![0u8; aux_bytes],
            &vec![0u8; aux_bytes],
        )
        .unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)));
    }

    #[test]
    fn from_bytes_k_padded_happy_path() {
        let device = Device::shared().expect("device");
        let q = Quantization::Q4_GS64;
        let (w_bytes, aux_bytes) = fixture_sizes(K_PAD_N, K_PAD_K_PHYSICAL, q);
        let k_logical = 64; // half the physical K
        let qw = QuantizedWeight::from_bytes_k_padded(
            &device,
            [K_PAD_N, K_PAD_K_PHYSICAL],
            k_logical,
            q,
            &vec![0u8; w_bytes],
            &vec![0u8; aux_bytes],
            &vec![0u8; aux_bytes],
        )
        .expect("k-padded construction should succeed");
        assert_eq!(qw.k_logical(), k_logical);
        assert_eq!(qw.shape(), &Shape::from([K_PAD_N, K_PAD_K_PHYSICAL]));
        assert_eq!(qw.quantization(), q);
        assert!(matches!(qw.tensor().layout(), Layout::Quantized(_)));
    }
}
