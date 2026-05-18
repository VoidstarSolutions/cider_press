//! Memory layout attribute on a [`Tensor`](crate::Tensor).
//!
//! A tensor is either:
//! - [`Layout::Dense`]: scalar elements laid out at the given strides
//!   in the underlying buffer. The element type is the tensor's
//!   [`DType`](crate::DType).
//! - [`Layout::Quantized`]: packed quantized scalars per
//!   [`Quantization`]. The tensor's [`DType`](crate::DType) is
//!   [`U32`](crate::DType::U32) (the packing word type) and the
//!   quantization descriptor lives here. Scales and biases are
//!   *separate* tensors, not bundled into this layout — that mirrors
//!   MLX's `quantized_matmul` ABI, where the three buffers are bound
//!   independently.
//!
//! Keeping quantization at the layout layer (rather than as a
//! [`DType`](crate::DType) variant) means dispatch code can match on a
//! small, exhaustive set of scalar dtypes and decide quantization in a
//! separate branch — which is exactly how MLX's `quantized.cpp`
//! dispatcher is structured.

use crate::{Quantization, Strides};

/// Memory layout of a tensor.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Layout {
    /// Standard strided dense storage of one scalar dtype.
    Dense {
        /// Per-axis steps in *elements*. Length must equal the
        /// tensor's rank.
        strides: Strides,
    },
    /// MLX-style affine-quantized layout. The tensor's dtype is
    /// [`U32`](crate::DType::U32) packed words.
    Quantized(Quantization),
}

impl Layout {
    /// True if this is a [`Layout::Dense`] variant with contiguous
    /// (row-major) strides for the given shape.
    #[must_use]
    pub fn is_dense_contiguous(&self, shape: &crate::Shape) -> bool {
        match self {
            Layout::Dense { strides } => strides.is_contiguous(shape),
            Layout::Quantized(_) => false,
        }
    }

    /// True if this layout carries a quantization descriptor.
    #[must_use]
    pub fn is_quantized(&self) -> bool {
        matches!(self, Layout::Quantized(_))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Shape;

    #[test]
    fn dense_contiguous_round_trip() {
        let shape = Shape::from([2, 3, 4]);
        let layout = Layout::Dense {
            strides: Strides::contiguous(&shape),
        };
        assert!(layout.is_dense_contiguous(&shape));
        assert!(!layout.is_quantized());
    }

    #[test]
    fn quantized_layout_flags() {
        let layout = Layout::Quantized(Quantization::Q4_GS64);
        assert!(layout.is_quantized());
        assert!(!layout.is_dense_contiguous(&Shape::from([4096, 4096])));
    }
}
