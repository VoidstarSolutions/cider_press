//! Quantized-weight newtype and its op constructors.
//!
//! Quantized weights are a different beast than dense tensors: they
//! carry three buffers (packed lanes, per-group scales, per-group
//! biases), they require a [`Quantization`] descriptor to interpret,
//! and they only participate in a small set of ops (chiefly `qmv` and
//! eventually `qmm`). The design doc's recommendation: encode the
//! "this is a quantized weight" precondition in the type system at
//! the op boundary, so op signatures stay infallible at the type
//! level and end-users can't accidentally hand a dense tensor to
//! [`QuantizedWeight::matvec`].
//!
//! [`QuantizedWeight`] is a thin newtype around [`Tensor`]: the
//! underlying graph node carries [`Layout::Quantized`] and the
//! `LeafStorage::Quantized` variant. The escape hatch
//! [`QuantizedWeight::tensor`] returns the inner handle for callers
//! that need to mix it into generic-tensor APIs.

use crate::Device;
use crate::Layout;
use crate::Quantization;
use crate::Shape;
use crate::Tensor;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::tensor::OpKind;

/// A quantized weight matrix, ready for [`QuantizedWeight::matvec`].
///
/// Construct via [`QuantizedWeight::from_bytes`]. Cheap to clone
/// (same `Arc<TensorInner>` underlying the wrapped [`Tensor`]).
#[derive(Clone)]
pub struct QuantizedWeight {
    tensor: Tensor,
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
        quantization.validate()?;
        let shape = shape.into();
        if shape.rank() != 2 {
            return Err(Error::InvalidArgument(format!(
                "QuantizedWeight: shape must be rank 2 ([N, K]); got rank {} ({:?})",
                shape.rank(),
                shape,
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
        Ok(Self { tensor })
    }

    /// Schedule a quantized matrix-vector multiply: `y = x · dequant(w)ᵀ`.
    ///
    /// `x` must be a dense bf16 vector of length `K` (the weight's
    /// inner dim). The returned [`Tensor`] is a lazy op node
    /// representing the output bf16 vector of length `N`. No GPU
    /// work runs until [`Tensor::eval`].
    ///
    /// Currently only bf16 inputs are supported; the underlying
    /// kernel is MLX's `affine_qmv_bf16` family.
    pub fn matvec(&self, x: &Tensor) -> Result<Tensor> {
        let device = self.tensor.device().ok_or_else(|| {
            Error::InvalidArgument("QuantizedWeight::matvec: weight has no device".into())
        })?;
        let x_device = x.device().ok_or_else(|| {
            Error::InvalidArgument(
                "QuantizedWeight::matvec: activation x is a placeholder (no device)".into(),
            )
        })?;
        if !device.ptr_eq(x_device) {
            return Err(Error::InvalidArgument(
                "QuantizedWeight::matvec: weight and activation are on different devices".into(),
            ));
        }
        if x.dtype() != DType::BF16 {
            return Err(Error::InvalidArgument(format!(
                "QuantizedWeight::matvec: activation must be bf16; got {}",
                x.dtype()
            )));
        }
        if !matches!(x.layout(), Layout::Dense { .. }) {
            return Err(Error::InvalidArgument(
                "QuantizedWeight::matvec: activation must be dense".into(),
            ));
        }
        // Shape: weight is [N, K], activation must be [K], output is [N].
        let weight_shape = self.tensor.shape();
        let dims = weight_shape.dims();
        let (n, k) = (dims[0], dims[1]);
        let x_shape = x.shape();
        let x_dims = x_shape.dims();
        if x_dims.len() != 1 || x_dims[0] != k {
            return Err(Error::InvalidArgument(format!(
                "QuantizedWeight::matvec: activation shape must be [{k}]; got {x_shape:?}"
            )));
        }
        let q = self.quantization();
        let out_shape = Shape::from([n]);
        let out_layout = Layout::Dense {
            strides: crate::Strides::contiguous(&out_shape),
        };
        Ok(Tensor::op_tensor(
            device,
            out_shape,
            DType::BF16,
            out_layout,
            OpKind::Qmv {
                group_size: q.group_size(),
                bits: q.bits(),
            },
            vec![self.tensor.clone(), x.clone()],
        ))
    }

    /// The underlying [`Tensor`] handle. Use this when an API wants a
    /// generic tensor (e.g. the op graph stores `Tensor`s uniformly).
    #[must_use]
    pub fn tensor(&self) -> &Tensor {
        &self.tensor
    }

    /// Logical shape `[N, K]` of the quantized weight.
    #[must_use]
    pub fn shape(&self) -> &Shape {
        self.tensor.shape()
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
            .field("quantization", &self.quantization())
            .finish()
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

    #[test]
    fn matvec_builds_qmv_op() {
        let device = Device::shared().expect("device");
        let q = Quantization::Q4_GS64;
        let (w_bytes, aux_bytes) = fixture_sizes(512, 512, q);
        let qw = QuantizedWeight::from_bytes(
            &device,
            [512, 512],
            q,
            &vec![0; w_bytes],
            &vec![0; aux_bytes],
            &vec![0; aux_bytes],
        )
        .expect("build");

        let x_data = vec![half::bf16::ZERO; 512];
        let x = Tensor::from_slice(&device, &x_data, [512]).expect("x");
        let y = qw.matvec(&x).expect("matvec");

        assert_eq!(
            y.op_kind(),
            Some(OpKind::Qmv {
                group_size: 64,
                bits: 4,
            })
        );
        assert!(!y.is_materialized());
        assert_eq!(y.shape(), &Shape::from([512]));
        assert_eq!(y.dtype(), DType::BF16);
    }

    #[test]
    fn matvec_rejects_wrong_activation_dtype() {
        let device = Device::shared().expect("device");
        let q = Quantization::Q4_GS64;
        let (w_bytes, aux_bytes) = fixture_sizes(64, 64, q);
        let qw = QuantizedWeight::from_bytes(
            &device,
            [64, 64],
            q,
            &vec![0; w_bytes],
            &vec![0; aux_bytes],
            &vec![0; aux_bytes],
        )
        .expect("build");

        let x = Tensor::zeros(&device, [64], DType::F32).expect("x");
        let err = qw.matvec(&x).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)));
    }

    #[test]
    fn matvec_rejects_wrong_activation_shape() {
        let device = Device::shared().expect("device");
        let q = Quantization::Q4_GS64;
        let (w_bytes, aux_bytes) = fixture_sizes(64, 64, q);
        let qw = QuantizedWeight::from_bytes(
            &device,
            [64, 64],
            q,
            &vec![0; w_bytes],
            &vec![0; aux_bytes],
            &vec![0; aux_bytes],
        )
        .expect("build");

        let x_data = vec![half::bf16::ZERO; 32];
        let x = Tensor::from_slice(&device, &x_data, [32]).expect("x");
        let err = qw.matvec(&x).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)));
    }
}
