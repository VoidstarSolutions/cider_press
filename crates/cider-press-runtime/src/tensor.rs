//! Lazy tensor handle.
//!
//! [`Tensor`] is the user-facing handle: cheap to clone (it's an
//! [`Arc`]), carries [`Shape`] + [`DType`] + [`Layout`], and refers to
//! a node in the computation graph. The node is either a *leaf* (a
//! materialized buffer) or — in later commits — an *op* (an operation
//! plus its input tensors). Construction is always lazy: no GPU work
//! runs until `eval()` is called (op variant lands in a follow-up).
//!
//! This commit lands the **storage** layer: real Metal buffers behind
//! [`Source::Leaf`], allocated via [`Device`] and accessible host-side
//! through [`Tensor::cpu_bytes`] / [`Tensor::cpu_slice`]. The lazy op
//! graph + `eval()` follow in subsequent commits per the runtime
//! design doc (`docs/RUNTIME_DESIGN.md`).
//!
//! [`Source::Placeholder`] is retained for tests and pure type-surface
//! scaffolding that doesn't need a real device.

use std::sync::Arc;

use cider_press_kernels::Buffer;

use crate::dtype::Scalar;
use crate::error::{Error, Result};
use crate::{DType, Device, Layout, Quantization, Shape, Strides};

/// Internal node a [`Tensor`] points to.
///
/// Behind an [`Arc`] so clones are cheap and graph nodes can be shared
/// by multiple consumers.
struct TensorInner {
    shape: Shape,
    dtype: DType,
    layout: Layout,
    /// `None` for [`Source::Placeholder`] (pure metadata); `Some` for
    /// every materialized variant. Carried on the inner so op nodes
    /// (added in the next commit) can inherit a device from their
    /// inputs without each variant duplicating the field.
    device: Option<Device>,
    source: Source,
}

/// What produces a tensor's values.
///
/// Grows as the runtime layer fills in. Today: metadata-only
/// placeholders for scaffolding, and materialized leaves backed by a
/// real [`Buffer`]. Op nodes and pending (in-flight) variants land in
/// subsequent commits.
enum Source {
    /// Metadata-only tensor with no backing buffer. Useful as a
    /// stand-in for tests and for type-surface scaffolding that
    /// doesn't need a real Metal device.
    Placeholder,
    /// Materialized tensor backed by a real Metal buffer.
    ///
    /// The buffer is stored byte-erased (`Buffer<u8>`): the element
    /// dtype lives on [`TensorInner::dtype`]. Op-dispatch sites that
    /// need a typed view reinterpret the bytes at call time using the
    /// dtype tag; this keeps a `Vec<Tensor>` of mixed dtypes
    /// expressible without `Box<dyn ...>` while preserving the
    /// kernels-layer's typed-buffer dispatch contract.
    Leaf(Buffer<u8>),
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
    /// for `shape` and the matching [`DType`] for `T`. No backing
    /// buffer is allocated.
    #[must_use]
    pub fn placeholder<T: Scalar>(shape: impl Into<Shape>) -> Self {
        Self::placeholder_inner(shape.into(), T::DTYPE)
    }

    /// Construct a dense placeholder with a dynamic [`DType`] tag and
    /// contiguous strides. Use this when the dtype is only known at
    /// runtime (e.g. model loading).
    #[must_use]
    pub fn dense_placeholder(shape: impl Into<Shape>, dtype: DType) -> Self {
        Self::placeholder_inner(shape.into(), dtype)
    }

    fn placeholder_inner(shape: Shape, dtype: DType) -> Self {
        let strides = Strides::contiguous(&shape);
        Self {
            inner: Arc::new(TensorInner {
                shape,
                dtype,
                layout: Layout::Dense { strides },
                device: None,
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
                device: None,
                source: Source::Placeholder,
            }),
        }
    }

    /// Allocate a dense tensor of zeros with `shape` and `dtype` on
    /// `device`. The backing buffer is real shared-storage memory;
    /// host-side reads via [`Tensor::cpu_bytes`] are safe immediately.
    pub fn zeros(device: &Device, shape: impl Into<Shape>, dtype: DType) -> Result<Self> {
        let shape = shape.into();
        let byte_count = shape.elem_count() * dtype.size_bytes();
        let mut buffer = device.kernels().alloc_buffer::<u8>(byte_count)?;
        // SAFETY: just allocated, no GPU dispatch has referenced it.
        unsafe { buffer.as_mut_slice() }.fill(0);
        let layout = dense_layout(&shape);
        Ok(Self::leaf(device, shape, dtype, layout, buffer))
    }

    /// Allocate a dense tensor on `device` and upload `data` into it.
    /// The tensor's dtype is determined by `T`; `shape.elem_count()`
    /// must equal `data.len()`.
    pub fn from_slice<T: Scalar>(
        device: &Device,
        data: &[T],
        shape: impl Into<Shape>,
    ) -> Result<Self> {
        let shape = shape.into();
        if shape.elem_count() != data.len() {
            return Err(Error::InvalidArgument(format!(
                "from_slice: shape elem_count ({}) != data.len() ({})",
                shape.elem_count(),
                data.len()
            )));
        }
        // SAFETY: every `Scalar` implementor is plain-old-data with no
        // padding and well-defined as raw bytes; we use `&[u8]` purely
        // to feed the byte-erased kernels-layer upload path.
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), std::mem::size_of_val(data))
        };
        let buffer = device.kernels().upload::<u8>(bytes)?;
        let layout = dense_layout(&shape);
        Ok(Self::leaf(device, shape, T::DTYPE, layout, buffer))
    }

    /// Allocate a dense tensor on `device` and upload raw `bytes`,
    /// tagging it with the given `dtype`. Use this when the dtype is
    /// only known at runtime (model loading). `bytes.len()` must equal
    /// `shape.elem_count() * dtype.size_bytes()`.
    pub fn from_bytes(
        device: &Device,
        bytes: &[u8],
        shape: impl Into<Shape>,
        dtype: DType,
    ) -> Result<Self> {
        let shape = shape.into();
        let expected = shape.elem_count() * dtype.size_bytes();
        if bytes.len() != expected {
            return Err(Error::InvalidArgument(format!(
                "from_bytes: expected {expected} bytes for shape {:?} dtype {dtype}, got {}",
                shape,
                bytes.len()
            )));
        }
        let buffer = device.kernels().upload::<u8>(bytes)?;
        let layout = dense_layout(&shape);
        Ok(Self::leaf(device, shape, dtype, layout, buffer))
    }

    fn leaf(
        device: &Device,
        shape: Shape,
        dtype: DType,
        layout: Layout,
        buffer: Buffer<u8>,
    ) -> Self {
        Self {
            inner: Arc::new(TensorInner {
                shape,
                dtype,
                layout,
                device: Some(device.clone()),
                source: Source::Leaf(buffer),
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

    /// Device this tensor is associated with, or `None` for a
    /// placeholder.
    #[must_use]
    pub fn device(&self) -> Option<&Device> {
        self.inner.device.as_ref()
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

    /// Whether this tensor is currently a metadata-only placeholder
    /// with no backing storage.
    #[must_use]
    pub fn is_placeholder(&self) -> bool {
        matches!(self.inner.source, Source::Placeholder)
    }

    /// Whether this tensor has a materialized buffer that can be read
    /// host-side.
    #[must_use]
    pub fn is_materialized(&self) -> bool {
        matches!(self.inner.source, Source::Leaf(_))
    }

    /// Host-readable view of the tensor's raw bytes, or `None` if not
    /// materialized.
    ///
    /// Safe to call on a materialized tensor: leaves carry the
    /// invariant that no GPU dispatch is concurrently writing the
    /// buffer (today, trivially — leaves are only constructed from
    /// host-side data; once op dispatch lands, the eval boundary
    /// upholds the invariant by waiting on completion before
    /// rewriting an op node into a leaf).
    #[must_use]
    pub fn cpu_bytes(&self) -> Option<&[u8]> {
        match &self.inner.source {
            // SAFETY: see method docs — leaves uphold the no-concurrent-
            // GPU-write invariant.
            Source::Leaf(buffer) => Some(unsafe { buffer.as_slice() }),
            Source::Placeholder => None,
        }
    }

    /// Typed host-readable view of a dense leaf's contents. Returns
    /// `None` if the tensor isn't materialized, isn't dense, or has a
    /// dtype other than `T`.
    #[must_use]
    pub fn cpu_slice<T: Scalar>(&self) -> Option<&[T]> {
        if self.dtype() != T::DTYPE {
            return None;
        }
        if !matches!(self.layout(), Layout::Dense { .. }) {
            return None;
        }
        let bytes = self.cpu_bytes()?;
        let elems = self.elem_count();
        debug_assert_eq!(bytes.len(), elems * T::DTYPE.size_bytes());
        // SAFETY: dtype matches `T`, layout is dense, byte length matches
        // `elems * size_of::<T>()` by construction. Metal shared-storage
        // allocations are page-aligned (≥ 4 KiB), so any scalar `T` is
        // suitably aligned.
        Some(unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<T>(), elems) })
    }
}

fn dense_layout(shape: &Shape) -> Layout {
    Layout::Dense {
        strides: Strides::contiguous(shape),
    }
}

impl core::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let source = match &self.inner.source {
            Source::Placeholder => "Placeholder",
            Source::Leaf(_) => "Leaf",
        };
        f.debug_struct("Tensor")
            .field("shape", &self.inner.shape)
            .field("dtype", &self.inner.dtype)
            .field("layout", &self.inner.layout)
            .field("source", &source)
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
        assert!(!t.is_materialized());
        assert!(t.device().is_none());
        assert!(t.cpu_bytes().is_none());
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

    #[test]
    fn from_slice_roundtrips_f32() {
        let device = Device::shared().expect("system default device");
        #[allow(
            clippy::cast_precision_loss,
            reason = "values 0..32 fit exactly in f32"
        )]
        let data: Vec<f32> = (0..32).map(|i| i as f32 * 1.5).collect();
        let t = Tensor::from_slice(&device, &data, [4, 8]).expect("from_slice");

        assert_eq!(t.dtype(), DType::F32);
        assert!(t.is_materialized());
        assert!(!t.is_placeholder());
        assert!(t.device().is_some_and(|d| d.ptr_eq(&device)));

        let read_back = t.cpu_slice::<f32>().expect("cpu_slice f32");
        assert_eq!(read_back, data.as_slice());
    }

    #[test]
    fn from_slice_roundtrips_bf16() {
        let device = Device::shared().expect("system default device");
        #[allow(
            clippy::cast_precision_loss,
            reason = "values 0..16 fit exactly in f32"
        )]
        let data: Vec<bf16> = (0..16).map(|i| bf16::from_f32(i as f32)).collect();
        let t = Tensor::from_slice(&device, &data, [16]).expect("from_slice");

        assert_eq!(t.dtype(), DType::BF16);
        let read_back = t.cpu_slice::<bf16>().expect("cpu_slice bf16");
        assert_eq!(read_back, data.as_slice());
    }

    #[test]
    fn from_slice_rejects_shape_mismatch() {
        let device = Device::shared().expect("system default device");
        let data: Vec<f32> = vec![0.0; 10];
        let err = Tensor::from_slice(&device, &data, [4, 4]).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)));
    }

    #[test]
    fn zeros_produces_zeroed_buffer() {
        let device = Device::shared().expect("system default device");
        let t = Tensor::zeros(&device, [3, 5], DType::F32).expect("zeros");

        assert_eq!(t.elem_count(), 15);
        let bytes = t.cpu_bytes().expect("leaf bytes");
        assert_eq!(bytes.len(), 15 * 4);
        assert!(bytes.iter().all(|&b| b == 0));

        let typed = t.cpu_slice::<f32>().expect("cpu_slice f32");
        assert!(typed.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn from_bytes_runtime_dtype() {
        let device = Device::shared().expect("system default device");
        let raw: Vec<u8> = (0..32u8).collect();
        let t = Tensor::from_bytes(&device, &raw, [8], DType::F32).expect("from_bytes");

        assert_eq!(t.dtype(), DType::F32);
        assert_eq!(t.cpu_bytes().unwrap(), raw.as_slice());
    }

    #[test]
    fn from_bytes_rejects_byte_length_mismatch() {
        let device = Device::shared().expect("system default device");
        let raw: Vec<u8> = vec![0; 31];
        let err = Tensor::from_bytes(&device, &raw, [8], DType::F32).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)));
    }

    #[test]
    fn cpu_slice_rejects_dtype_mismatch() {
        let device = Device::shared().expect("system default device");
        let t = Tensor::zeros(&device, [4], DType::F32).expect("zeros");
        assert!(t.cpu_slice::<i32>().is_none());
        assert!(t.cpu_slice::<f32>().is_some());
    }
}
