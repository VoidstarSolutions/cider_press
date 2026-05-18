//! Lazy tensor handle.
//!
//! [`Tensor`] is the user-facing handle: cheap to clone (it's an
//! [`Arc`]), carries [`Shape`] + [`DType`] + [`Layout`], and refers to
//! a node in the computation graph. The node is either a *leaf* (a
//! materialized buffer — once we wire storage in) or an *op* (an
//! operation plus its input tensors). Construction is always lazy: no
//! GPU work runs until `eval()` is called.
//!
//! This file is currently a **skeleton**. It defines the type surface
//! — shape, dtype, layout, lazy-source discriminant — but the only
//! [`Source`] variant is [`Source::Placeholder`], which carries
//! metadata but no buffer. Materialized leaves and op nodes land in
//! follow-up steps once the storage and op-graph designs are in place;
//! see `CLAUDE.md`'s post-spike Findings for the staged plan.

use std::sync::Arc;

use crate::dtype::Scalar;
use crate::{DType, Layout, Quantization, Shape, Strides};

/// Internal node a [`Tensor`] points to.
///
/// Behind an [`Arc`] so clones are cheap and graph nodes can be shared
/// by multiple consumers.
struct TensorInner {
    shape: Shape,
    dtype: DType,
    layout: Layout,
    source: Source,
}

/// What produces a tensor's values.
///
/// Grows as the runtime layer fills in: materialized leaves backed by
/// a buffer, op nodes referencing input tensors, etc. Today, only the
/// metadata-only `Placeholder` variant exists — enough to exercise the
/// type surface without committing to a storage representation.
enum Source {
    /// Metadata-only tensor with no backing buffer. Useful as a stand-in
    /// while the rest of the runtime is being designed.
    Placeholder,
}

/// User-facing lazy tensor handle.
///
/// Cloning a [`Tensor`] bumps a refcount — it does *not* copy data.
#[derive(Clone)]
pub struct Tensor {
    inner: Arc<TensorInner>,
}

impl Tensor {
    /// Construct a dense placeholder with row-major contiguous strides
    /// for `shape` and the matching [`DType`] for `T`.
    #[must_use]
    pub fn placeholder<T: Scalar>(shape: impl Into<Shape>) -> Self {
        let shape = shape.into();
        let strides = Strides::contiguous(&shape);
        Self {
            inner: Arc::new(TensorInner {
                shape,
                dtype: T::DTYPE,
                layout: Layout::Dense { strides },
                source: Source::Placeholder,
            }),
        }
    }

    /// Construct a dense placeholder with a dynamic [`DType`] tag and
    /// contiguous strides. Use this when the dtype is only known at
    /// runtime (e.g. model loading).
    #[must_use]
    pub fn dense_placeholder(shape: impl Into<Shape>, dtype: DType) -> Self {
        let shape = shape.into();
        let strides = Strides::contiguous(&shape);
        Self {
            inner: Arc::new(TensorInner {
                shape,
                dtype,
                layout: Layout::Dense { strides },
                source: Source::Placeholder,
            }),
        }
    }

    /// Construct a quantized-weight placeholder. The tensor's dtype is
    /// [`U32`](DType::U32) — the packing word type — and the
    /// quantization descriptor lives on [`Layout::Quantized`].
    ///
    /// `shape` is the *logical* (unpacked) shape; the packed buffer
    /// will have `quantization.packed_word_count(elem_count)` `u32`s.
    #[must_use]
    pub fn quantized_placeholder(shape: impl Into<Shape>, quantization: Quantization) -> Self {
        Self {
            inner: Arc::new(TensorInner {
                shape: shape.into(),
                dtype: DType::U32,
                layout: Layout::Quantized(quantization),
                source: Source::Placeholder,
            }),
        }
    }

    /// Logical shape of this tensor.
    #[must_use]
    pub fn shape(&self) -> &Shape {
        &self.inner.shape
    }

    /// Element dtype. For quantized tensors this is the packing-word
    /// type ([`U32`](DType::U32)), not the logical pre-quantization
    /// dtype.
    #[must_use]
    pub fn dtype(&self) -> DType {
        self.inner.dtype
    }

    /// Memory layout: dense (with strides) or quantized.
    #[must_use]
    pub fn layout(&self) -> &Layout {
        &self.inner.layout
    }

    /// Number of dimensions.
    #[must_use]
    pub fn rank(&self) -> usize {
        self.inner.shape.rank()
    }

    /// Total logical element count.
    #[must_use]
    pub fn elem_count(&self) -> usize {
        self.inner.shape.elem_count()
    }

    /// Whether this tensor is currently just metadata (no backing
    /// storage and no op to evaluate). Returns `true` for every tensor
    /// in the current skeleton; will return `false` once materialized
    /// and op-backed variants are wired in.
    #[must_use]
    pub fn is_placeholder(&self) -> bool {
        matches!(self.inner.source, Source::Placeholder)
    }
}

impl core::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Tensor")
            .field("shape", &self.inner.shape)
            .field("dtype", &self.inner.dtype)
            .field("layout", &self.inner.layout)
            .field("placeholder", &self.is_placeholder())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::bf16;

    #[test]
    fn dense_typed_placeholder() {
        let t = Tensor::placeholder::<f32>([4, 8]);
        assert_eq!(t.shape(), &Shape::from([4, 8]));
        assert_eq!(t.dtype(), DType::F32);
        assert!(matches!(t.layout(), Layout::Dense { .. }));
        assert!(t.layout().is_dense_contiguous(t.shape()));
        assert_eq!(t.rank(), 2);
        assert_eq!(t.elem_count(), 32);
        assert!(t.is_placeholder());
    }

    #[test]
    fn dense_dynamic_dtype() {
        let t = Tensor::dense_placeholder([16], DType::BF16);
        assert_eq!(t.dtype(), DType::BF16);
        assert!(t.layout().is_dense_contiguous(t.shape()));
    }

    #[test]
    fn clone_is_arc_shared() {
        let a = Tensor::placeholder::<bf16>([2, 2]);
        let b = a.clone();
        assert!(Arc::ptr_eq(&a.inner, &b.inner));
    }

    #[test]
    fn quantized_placeholder_carries_layout() {
        let t = Tensor::quantized_placeholder([4096, 4096], Quantization::Q4_GS64);
        assert_eq!(t.dtype(), DType::U32);
        match t.layout() {
            Layout::Quantized(q) => assert_eq!(*q, Quantization::Q4_GS64),
            Layout::Dense { .. } => panic!("expected quantized layout"),
        }
    }
}
