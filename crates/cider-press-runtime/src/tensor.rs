//! Lazy tensor handle.
//!
//! [`Tensor`] is the user-facing handle: cheap to clone (it's an
//! [`Arc`]), carries [`Shape`] + [`DType`] + [`Layout`], and refers to
//! a node in the computation graph. A node is one of: a *placeholder*
//! (metadata only — test scaffolding), a *host-constructed leaf* (a
//! materialized buffer with no op lineage), or an *op* (an [`OpKind`]
//! plus its input tensors, possibly already evaluated).
//!
//! Construction is always lazy: building an op tensor does no GPU
//! work and allocates no output buffer. Both happen at [`Tensor::eval`]
//! time — the eval boundary topologically walks the unevaluated DAG
//! reachable from `self`, encodes one [`Commands`](cider_press_kernels::Commands)
//! batch, commits, waits, and populates each op's result cache.
//!
//! The op metadata (the `OpNode`) is immutable for the lifetime of
//! the tensor; the result cache ([`OnceLock<Buffer>`](std::sync::OnceLock))
//! transitions empty → populated exactly once, monotonically. This
//! is intentionally simpler than the `Mutex<Source>` alternative
//! sketched in `docs/RUNTIME_DESIGN.md`: no lock on the read path,
//! no race window between mutations, and [`Tensor::cpu_bytes`] can
//! return `&[u8]` directly.

use std::sync::{Arc, OnceLock};

use cider_press_kernels::Buffer;

use crate::dtype::Scalar;
use crate::error::{Error, Result};
use crate::{DType, Device, Layout, Quantization, Shape, Strides};

/// Internal node a [`Tensor`] points to.
///
/// Behind an [`Arc`] so clones are cheap and graph nodes can be shared
/// by multiple consumers.
pub(crate) struct TensorInner {
    pub(crate) shape: Shape,
    pub(crate) dtype: DType,
    pub(crate) layout: Layout,
    /// `None` only for [`Tensor::placeholder`] / [`Tensor::dense_placeholder`]
    /// / [`Tensor::quantized_placeholder`] — pure metadata, no buffer
    /// will ever exist. Every other tensor carries a real device.
    pub(crate) device: Option<Device>,
    /// Op lineage. `Some` for tensors produced by an op (whether
    /// evaluated or not); `None` for placeholders and host-constructed
    /// leaves. Set at construction and never mutated thereafter — the
    /// `OnceLock` cache below is what records evaluation.
    pub(crate) op: Option<OpNode>,
    /// Materialized storage. Empty for placeholders and unevaluated
    /// op nodes; populated either at construction (host-constructed
    /// leaves) or by [`Tensor::eval`] (op outputs). Once populated,
    /// the storage is immutable for the lifetime of the inner —
    /// that's what makes [`Tensor::cpu_bytes`] safe to expose as
    /// `&[u8]`.
    pub(crate) cache: OnceLock<LeafStorage>,
}

/// Materialized backing storage for a [`Tensor`].
///
/// Dense tensors carry a single byte-erased buffer; quantized
/// tensors carry the three buffers MLX's `qmv` family needs (packed
/// weights, per-group scales, per-group biases). Kept as an enum
/// rather than three optional fields because the variants are
/// disjoint by construction — a dense leaf never has scales/biases,
/// a quantized leaf never has a dense byte view.
pub(crate) enum LeafStorage {
    /// Byte-erased dense buffer. The dtype tag on
    /// [`TensorInner::dtype`] is the source of truth for what's in
    /// the bytes; op dispatch sites reinterpret at call time.
    Dense(Buffer<u8>),
    /// Three-buffer storage for affine-quantized weights. Layout
    /// matches what `cider_press_kernels::kernels::qmv` expects.
    Quantized {
        /// Packed lanes (e.g. int4 lanes packed eight-per-uint32 for
        /// `bits=4`). Stored byte-erased; reinterpret as
        /// `Buffer<u32>` at dispatch.
        weights: Buffer<u8>,
        /// Per-group scales, bf16. One scale per group.
        scales: Buffer<u8>,
        /// Per-group biases, bf16. One bias per group.
        biases: Buffer<u8>,
    },
}

/// A lazy operation node: the op kind plus its input tensors.
///
/// `inputs` are `Tensor` clones (refcount bumps), which is what makes
/// the graph a DAG rather than a tree — multiple consumers of the
/// same producer share an [`Arc<TensorInner>`].
pub(crate) struct OpNode {
    pub(crate) kind: OpKind,
    pub(crate) inputs: Vec<Tensor>,
}

/// Closed set of ops the runtime can dispatch.
///
/// Variant-per-op rather than `dyn Op`: we own every op in this crate
/// and the extensibility tax of a trait isn't worth it until we have
/// a reason to allow third-party ops. The dispatcher in `eval`
/// `match`es on this enum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum OpKind {
    /// Element-wise identity copy. Output is a fresh dense tensor with
    /// the same shape, dtype, and values as the input. Mirrors MLX's
    /// `mx.copy` and dispatches to the vendored `v_copy` kernel.
    Copy,
    /// Affine-quantized matrix-vector multiply: `y = x · dequant(w)ᵀ`.
    /// Inputs (in [`Tensor::op_inputs`] order) are `[w, x]` where `w`
    /// is a [`Layout::Quantized`] tensor and `x` is a dense bf16
    /// vector. `group_size` and `bits` are duplicated from `w`'s
    /// [`Quantization`](crate::Quantization) descriptor so the
    /// dispatcher doesn't have to introspect the input's layout.
    Qmv {
        /// Quantization group size (elements per scale/bias). Must
        /// match the input weight's [`Quantization::group_size`].
        group_size: u32,
        /// Quantization bit-width (e.g. 4 for int4 weights). Must
        /// match the input weight's [`Quantization::bits`].
        bits: u8,
    },
}

/// User-facing lazy tensor handle.
///
/// Cloning a [`Tensor`] bumps a refcount — it does *not* copy data.
#[derive(Clone)]
pub struct Tensor {
    pub(crate) inner: Arc<TensorInner>,
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
                op: None,
                cache: OnceLock::new(),
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
                op: None,
                cache: OnceLock::new(),
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
        Ok(Self::host_leaf(device, shape, dtype, layout, buffer))
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
        Ok(Self::host_leaf(device, shape, T::DTYPE, layout, buffer))
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
        Ok(Self::host_leaf(device, shape, dtype, layout, buffer))
    }

    fn host_leaf(
        device: &Device,
        shape: Shape,
        dtype: DType,
        layout: Layout,
        buffer: Buffer<u8>,
    ) -> Self {
        Self::host_leaf_storage(device, shape, dtype, layout, LeafStorage::Dense(buffer))
    }

    /// Crate-private constructor for quantized leaves (used by
    /// [`QuantizedWeight::from_bytes`](crate::QuantizedWeight::from_bytes)).
    pub(crate) fn host_quantized_leaf(
        device: &Device,
        shape: Shape,
        layout: Layout,
        weights: Buffer<u8>,
        scales: Buffer<u8>,
        biases: Buffer<u8>,
    ) -> Self {
        Self::host_leaf_storage(
            device,
            shape,
            DType::U32,
            layout,
            LeafStorage::Quantized {
                weights,
                scales,
                biases,
            },
        )
    }

    fn host_leaf_storage(
        device: &Device,
        shape: Shape,
        dtype: DType,
        layout: Layout,
        storage: LeafStorage,
    ) -> Self {
        let cache = OnceLock::new();
        cache
            .set(storage)
            .ok()
            .expect("freshly-constructed OnceLock");
        Self {
            inner: Arc::new(TensorInner {
                shape,
                dtype,
                layout,
                device: Some(device.clone()),
                op: None,
                cache,
            }),
        }
    }

    /// Crate-private constructor for op tensors (used by op builders
    /// like [`Tensor::copy`] and [`QuantizedWeight::matvec`](crate::QuantizedWeight::matvec)).
    pub(crate) fn op_tensor(
        device: &Device,
        shape: Shape,
        dtype: DType,
        layout: Layout,
        kind: OpKind,
        inputs: Vec<Tensor>,
    ) -> Self {
        Self {
            inner: Arc::new(TensorInner {
                shape,
                dtype,
                layout,
                device: Some(device.clone()),
                op: Some(OpNode { kind, inputs }),
                cache: OnceLock::new(),
            }),
        }
    }

    /// Schedule an identity copy of this tensor.
    ///
    /// Returns a new [`Tensor`] backed by an unevaluated
    /// [`OpKind::Copy`] node referencing `self` as its sole input. No
    /// GPU work runs and no output buffer is allocated; both happen
    /// at [`Tensor::eval`] time.
    ///
    /// The result inherits this tensor's [`Shape`], [`DType`], and
    /// [`Device`]; the output [`Layout`] is dense and contiguous,
    /// regardless of the input's stride pattern (`copy` materializes a
    /// dense version of the logical values, the same as MLX's
    /// `mx.copy`).
    pub fn copy(&self) -> Result<Self> {
        let device = self.inner.device.as_ref().ok_or_else(|| {
            Error::InvalidArgument("copy: cannot apply an op to a placeholder (no device)".into())
        })?;
        if !matches!(self.inner.layout, Layout::Dense { .. }) {
            return Err(Error::InvalidArgument(
                "copy: only dense tensors are supported".into(),
            ));
        }
        let layout = dense_layout(&self.inner.shape);
        Ok(Self::op_tensor(
            device,
            self.inner.shape.clone(),
            self.inner.dtype,
            layout,
            OpKind::Copy,
            vec![self.clone()],
        ))
    }

    /// Materialize this tensor and every unevaluated op node it
    /// transitively depends on.
    ///
    /// Topologically walks the lazy graph rooted at `self`, encodes
    /// one [`Commands`](cider_press_kernels::Commands) batch covering
    /// every unevaluated op (in dependency order), commits the batch
    /// to the GPU, blocks until completion, and populates each op's
    /// result cache with its output buffer.
    ///
    /// After `eval()` returns, this tensor and every reachable
    /// dependency are materialized: [`Tensor::cpu_bytes`] returns
    /// `Some` and [`Tensor::is_materialized`] returns `true`. Already-
    /// materialized leaves and placeholders are skipped; calling
    /// `eval()` on an already-evaluated graph is a no-op.
    ///
    /// Synchronous by design (sync public API; see
    /// `docs/RUNTIME_DESIGN.md`). Internal command-buffer pipelining
    /// to close the perf gap to MLX is a future implementation detail
    /// that does not change this signature.
    pub fn eval(&self) -> Result<()> {
        crate::eval::eval(self)
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

    /// Whether this tensor is a metadata-only placeholder with no
    /// device and no materialized buffer.
    #[must_use]
    pub fn is_placeholder(&self) -> bool {
        self.inner.device.is_none()
    }

    /// Whether this tensor's backing buffer has been materialized
    /// and can be read host-side. `true` for host-constructed leaves
    /// (always) and for op tensors after [`Tensor::eval`].
    #[must_use]
    pub fn is_materialized(&self) -> bool {
        self.inner.cache.get().is_some()
    }

    /// Whether this tensor was produced by an op (regardless of
    /// whether it has been evaluated yet). Use [`Tensor::is_materialized`]
    /// to ask "is the buffer ready?".
    #[must_use]
    pub fn is_op(&self) -> bool {
        self.inner.op.is_some()
    }

    /// [`OpKind`] of the op that produces this tensor, if any. `None`
    /// for placeholders and host-constructed leaves.
    #[must_use]
    pub fn op_kind(&self) -> Option<OpKind> {
        self.inner.op.as_ref().map(|n| n.kind)
    }

    /// Inputs to this tensor's op node, or `None` if it wasn't
    /// produced by an op. The slice elements are `Tensor` clones
    /// (refcount bumps); use [`Tensor::ptr_eq`] for identity.
    #[must_use]
    pub fn op_inputs(&self) -> Option<&[Tensor]> {
        self.inner.op.as_ref().map(|n| n.inputs.as_slice())
    }

    /// Whether two `Tensor` handles point at the same underlying graph
    /// node. The op layer uses this for graph-node identity (CSE,
    /// dedup); user code rarely needs it.
    #[must_use]
    pub fn ptr_eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    /// Host-readable view of the tensor's raw bytes, or `None` if the
    /// backing buffer hasn't been materialized yet *or* if the tensor
    /// is quantized (quantized tensors are composite — use the
    /// accessors on [`QuantizedWeight`](crate::QuantizedWeight) to
    /// reach the individual buffers).
    ///
    /// Safe to call: a populated `cache` carries the invariant that
    /// no GPU dispatch is concurrently writing the buffer.
    /// Host-constructed leaves uphold this trivially; op outputs
    /// uphold it because [`Tensor::eval`] populates the cache only
    /// after `waitUntilCompleted`.
    #[must_use]
    pub fn cpu_bytes(&self) -> Option<&[u8]> {
        match self.inner.cache.get()? {
            // SAFETY: see method docs — the cache invariant rules out
            // concurrent GPU writes.
            LeafStorage::Dense(buffer) => Some(unsafe { buffer.as_slice() }),
            LeafStorage::Quantized { .. } => None,
        }
    }

    /// Typed host-readable view of a dense materialized tensor.
    /// Returns `None` if the tensor isn't materialized, isn't dense,
    /// or has a dtype other than `T`.
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
        let mut s = f.debug_struct("Tensor");
        s.field("shape", &self.inner.shape)
            .field("dtype", &self.inner.dtype)
            .field("layout", &self.inner.layout)
            .field("op", &self.inner.op.as_ref().map(|n| n.kind))
            .field("materialized", &self.is_materialized());
        s.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::{bf16, f16};

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
        assert!(a.ptr_eq(&b));
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

    #[test]
    fn copy_builds_op_node_without_evaluating() {
        let device = Device::shared().expect("system default device");
        let leaf = Tensor::zeros(&device, [4, 8], DType::F32).expect("zeros");

        let op = leaf.copy().expect("copy");

        assert!(op.is_op());
        assert!(!op.is_materialized());
        assert!(!op.is_placeholder());
        assert_eq!(op.op_kind(), Some(OpKind::Copy));
        assert!(op.cpu_bytes().is_none());
    }

    #[test]
    fn copy_propagates_shape_dtype_and_device() {
        let device = Device::shared().expect("system default device");
        let leaf = Tensor::zeros(&device, [2, 3, 5], DType::BF16).expect("zeros");

        let op = leaf.copy().expect("copy");

        assert_eq!(op.shape(), leaf.shape());
        assert_eq!(op.dtype(), leaf.dtype());
        assert_eq!(op.rank(), 3);
        assert_eq!(op.elem_count(), 30);
        assert!(op.layout().is_dense_contiguous(op.shape()));
        assert!(op.device().is_some_and(|d| d.ptr_eq(&device)));
    }

    #[test]
    fn copy_input_arc_identity_preserved() {
        let device = Device::shared().expect("system default device");
        let leaf = Tensor::zeros(&device, [4], DType::F32).expect("zeros");

        let op = leaf.copy().expect("copy");
        let inputs = op.op_inputs().expect("op inputs");
        assert_eq!(inputs.len(), 1);
        assert!(inputs[0].ptr_eq(&leaf));
    }

    #[test]
    fn copy_chain_builds_nested_graph() {
        let device = Device::shared().expect("system default device");
        let leaf = Tensor::zeros(&device, [4], DType::F32).expect("zeros");

        let once = leaf.copy().expect("copy 1");
        let twice = once.copy().expect("copy 2");

        let outer_inputs = twice.op_inputs().expect("op inputs");
        assert_eq!(outer_inputs.len(), 1);
        assert!(outer_inputs[0].ptr_eq(&once));

        let inner_inputs = once.op_inputs().expect("op inputs");
        assert!(inner_inputs[0].ptr_eq(&leaf));
    }

    #[test]
    fn cloning_op_tensor_shares_node() {
        let device = Device::shared().expect("system default device");
        let leaf = Tensor::zeros(&device, [4], DType::F32).expect("zeros");
        let op = leaf.copy().expect("copy");
        let cloned = op.clone();
        assert!(op.ptr_eq(&cloned));
    }

    #[test]
    fn copy_rejects_placeholder() {
        let p = Tensor::placeholder::<f32>([4, 4]);
        let err = p.copy().unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)));
    }

    #[test]
    fn host_leaf_accessors_return_none_for_op_methods() {
        let device = Device::shared().expect("system default device");
        let leaf = Tensor::zeros(&device, [4], DType::F32).expect("zeros");
        assert!(leaf.op_kind().is_none());
        assert!(leaf.op_inputs().is_none());
        assert!(!leaf.is_op());
    }

    #[test]
    fn eval_on_host_leaf_is_noop() {
        let device = Device::shared().expect("system default device");
        let t = Tensor::zeros(&device, [4], DType::F32).expect("zeros");
        assert!(t.is_materialized());
        t.eval().expect("eval");
        assert!(t.is_materialized());
    }

    #[test]
    fn eval_on_placeholder_is_noop() {
        // Placeholders have no unevaluated ops underneath them, so
        // eval() is vacuously fine; it just walks an empty DAG.
        let p = Tensor::placeholder::<f32>([4]);
        p.eval().expect("eval on placeholder");
    }

    #[test]
    fn eval_materializes_copy_f32() {
        let device = Device::shared().expect("system default device");
        let data: Vec<f32> = (0..64)
            .map(|i| f32::from(i16::try_from(i).unwrap()))
            .collect();
        let src = Tensor::from_slice(&device, &data, [64]).expect("from_slice");
        let dst = src.copy().expect("copy");

        assert!(!dst.is_materialized());
        dst.eval().expect("eval");
        assert!(dst.is_materialized());

        let read_back = dst.cpu_slice::<f32>().expect("cpu_slice");
        assert_eq!(read_back, data.as_slice());
    }

    #[test]
    fn eval_materializes_copy_f16() {
        let device = Device::shared().expect("system default device");
        let data: Vec<f16> = (0..32)
            .map(|i| f16::from_f32(f32::from(i16::try_from(i).unwrap())))
            .collect();
        let src = Tensor::from_slice(&device, &data, [32]).expect("from_slice");
        let dst = src.copy().expect("copy");

        dst.eval().expect("eval");
        let read_back = dst.cpu_slice::<f16>().expect("cpu_slice");
        assert_eq!(read_back, data.as_slice());
    }

    #[test]
    fn eval_chain_materializes_intermediates() {
        let device = Device::shared().expect("system default device");
        let data: Vec<f32> = (0..16)
            .map(|i| f32::from(i16::try_from(i).unwrap()))
            .collect();
        let src = Tensor::from_slice(&device, &data, [16]).expect("from_slice");
        let once = src.copy().expect("copy 1");
        let twice = once.copy().expect("copy 2");

        twice.eval().expect("eval");

        assert!(once.is_materialized());
        assert!(twice.is_materialized());
        assert_eq!(once.cpu_slice::<f32>().unwrap(), data.as_slice());
        assert_eq!(twice.cpu_slice::<f32>().unwrap(), data.as_slice());
    }

    #[test]
    fn eval_shared_subgraph_dispatched_once() {
        let device = Device::shared().expect("system default device");
        let data: Vec<f32> = vec![1.0; 8];
        let src = Tensor::from_slice(&device, &data, [8]).expect("from_slice");
        let shared = src.copy().expect("copy shared");
        let a = shared.copy().expect("copy a");
        let b = shared.copy().expect("copy b");

        // Both a and b depend on `shared`; eval'ing a then b should
        // see `shared` materialized after the first call and skip it
        // on the second.
        a.eval().expect("eval a");
        assert!(shared.is_materialized());
        b.eval().expect("eval b");
        assert!(b.is_materialized());

        // All three carry the same values (identity copy chain).
        assert_eq!(shared.cpu_slice::<f32>().unwrap(), data.as_slice());
        assert_eq!(a.cpu_slice::<f32>().unwrap(), data.as_slice());
        assert_eq!(b.cpu_slice::<f32>().unwrap(), data.as_slice());
    }

    #[test]
    fn eval_materializes_copy_bf16() {
        let device = Device::shared().expect("system default device");
        let data: Vec<bf16> = (0..32)
            .map(|i| bf16::from_f32(f32::from(i16::try_from(i).unwrap())))
            .collect();
        let src = Tensor::from_slice(&device, &data, [32]).expect("from_slice");
        let dst = src.copy().expect("copy");

        dst.eval().expect("eval");
        let read_back = dst.cpu_slice::<bf16>().expect("cpu_slice");
        assert_eq!(read_back, data.as_slice());
    }

    #[test]
    fn eval_materializes_copy_i32() {
        let device = Device::shared().expect("system default device");
        let data: Vec<i32> = (0..32).map(|i| i * 7 - 100).collect();
        let src = Tensor::from_slice(&device, &data, [32]).expect("from_slice");
        let dst = src.copy().expect("copy");

        dst.eval().expect("eval");
        let read_back = dst.cpu_slice::<i32>().expect("cpu_slice");
        assert_eq!(read_back, data.as_slice());
    }

    #[test]
    fn eval_materializes_copy_u32() {
        let device = Device::shared().expect("system default device");
        let data: Vec<u32> = (0..32u32).map(|i| i.wrapping_mul(0x9E37_79B1)).collect();
        let src = Tensor::from_slice(&device, &data, [32]).expect("from_slice");
        let dst = src.copy().expect("copy");

        dst.eval().expect("eval");
        let read_back = dst.cpu_slice::<u32>().expect("cpu_slice");
        assert_eq!(read_back, data.as_slice());
    }
}
