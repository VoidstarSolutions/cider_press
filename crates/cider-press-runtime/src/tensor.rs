//! Lazy tensor handle.
//!
//! [`Tensor`] is the user-facing handle: cheap to clone (it's an
//! [`Arc`]), carries [`Shape`] + [`DType`] + [`Layout`], and refers to
//! a node in the computation graph. A node is one of: a *placeholder*
//! (metadata only — test scaffolding), a *host-constructed leaf* (a
//! materialized buffer with no op lineage), an *op* (an [`OpKind`]
//! plus its input tensors, possibly already evaluated), or a *view*
//! (a different shape / stride / byte-offset window onto another
//! tensor's storage; see [`ViewSource`]).
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
//!
//! **View invariants.** `view` and `op` are mutually exclusive: a
//! tensor either describes its own op lineage or it points at another
//! tensor's storage, never both. View tensors never populate their own
//! `cache`; their bytes come from chasing [`ViewSource::source`] to
//! the backing leaf. This means `cpu_bytes()` on a view follows the
//! chain rather than reading the view's own (always-empty) cache.

use std::marker::PhantomData;
use std::ops::Range;
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
    /// evaluated or not); `None` for placeholders, host-constructed
    /// leaves, and views. Set at construction and never mutated
    /// thereafter — the `OnceLock` cache below is what records
    /// evaluation.
    ///
    /// Mutually exclusive with `view`: a tensor either has its own op
    /// lineage or it's a view onto another tensor, never both.
    pub(crate) op: Option<OpNode>,
    /// View lineage. `Some` for tensors that share storage with
    /// another tensor (`reshape`, `permute`, `slice`, `broadcast_to`). View
    /// tensors never populate their own `cache` — they chase
    /// [`ViewSource::source`] to find the backing leaf at read time.
    pub(crate) view: Option<ViewSource>,
    /// Materialized storage. Empty for placeholders, unevaluated op
    /// nodes, and views; populated either at construction
    /// (host-constructed leaves) or by [`Tensor::eval`] (op outputs).
    /// Once populated, the storage is immutable for the lifetime of
    /// the inner — that's what makes [`Tensor::cpu_bytes`] safe to
    /// expose as `&[u8]`.
    pub(crate) cache: OnceLock<LeafStorage>,
}

/// A view onto another tensor's storage.
///
/// View tensors carry their own `shape` / `layout` (strides) /
/// `byte_offset` while sharing the backing buffer with `source`.
/// `source` may itself be a view (chains collapse at read time) or a
/// leaf / op tensor.
pub(crate) struct ViewSource {
    /// The tensor whose backing leaf this view ultimately reads from.
    pub(crate) source: Tensor,
    /// Byte offset into `source`'s element `[0, 0, …]` where this
    /// view's element `[0, 0, …]` lives. Defined *per hop* — i.e.
    /// relative to the immediate `source`, not the ultimate backing
    /// leaf. Chain walkers ([`Tensor::flatten_view`] and
    /// [`Tensor::resolve_leaf`]) sum these offsets across hops, so
    /// constructors must supply a hop-local offset; passing a leaf-
    /// relative offset for a multi-hop source double-counts the
    /// intermediate hops' offsets.
    ///
    /// Note: today most view-building APIs use
    /// [`Tensor::flatten_view`] before constructing the new view, so
    /// the chains they emit are one hop deep (source is always the
    /// storage owner). Multi-hop chains arise only when an external
    /// caller composes views manually without flattening.
    pub(crate) byte_offset: usize,
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
    Dense(crate::buffer_pool::PooledBuffer),
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
///
/// Only `PartialEq` (not `Eq`) is derived because [`OpKind::Rope`]
/// carries `f32` parameters (`base_log2`, `scale`) — the bit-exact
/// comparison `PartialEq` provides is what tests want anyway.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum OpKind {
    /// Element-wise identity copy. Output is a fresh dense tensor with
    /// the same shape, dtype, and values as the input. Mirrors MLX's
    /// `mx.copy` and dispatches to the vendored `v_copy` kernel.
    Copy,
    /// Affine-quantized matrix multiply: `y = x · dequant(w)ᵀ`.
    /// Inputs (in [`Tensor::op_inputs`] order) are `[w, x]` where `w`
    /// is a [`Layout::Quantized`] tensor and `x` is a dense bf16
    /// tensor of shape `[…, M, K]` (rank-1 `[K]` is the decode / M=1
    /// case). `group_size` and `bits` are duplicated from `w`'s
    /// [`Quantization`](crate::Quantization) descriptor so the
    /// dispatcher doesn't have to introspect the input's layout.
    /// The eval-side dispatcher picks `affine_qmv` for M=1 and
    /// `affine_qmm_t` for M>1.
    QuantizedMatMul {
        /// Quantization group size (elements per scale/bias). Must
        /// match the input weight's [`Quantization::group_size`].
        group_size: u32,
        /// Quantization bit-width (e.g. 4 for int4 weights). Must
        /// match the input weight's [`Quantization::bits`].
        bits: u8,
        /// `true` means `y = x @ W^T` (weight stored `[N, K]`, output dim
        /// `N`). This is the only direction currently implemented:
        /// `false` is rejected at op construction by
        /// [`Tensor::quantized_matmul`] with [`Error::InvalidArgument`].
        /// The tied LM head also uses `transpose=true` (same weight
        /// orientation as the standard `Linear`), so no consumer needs
        /// `false` yet.
        transpose: bool,
    },
    /// Element-wise binary op with `NumPy` broadcasting. Inputs (in
    /// [`Tensor::op_inputs`] order) are `[lhs, rhs]`, both dense
    /// tensors of the same [`DType`] on the same [`Device`]. The
    /// dispatcher picks between MLX's `vv_<Op>` (same-shape contiguous
    /// fast path) and `g{1,2,3}_<Op>` (rank-specialised strided path)
    /// based on whether either input is a view or has a shape
    /// shorter than the output.
    Binary {
        /// Which arithmetic operation to perform.
        op: BinaryOp,
    },
    /// Element-wise unary op on a dense contiguous input. Output has
    /// the same shape + dtype as the input; the dispatcher uses MLX's
    /// `v_<Op><tname><tname>` kernel (N=1 contiguous).
    Unary {
        /// Which unary operation to perform.
        op: UnaryOp,
    },
    /// Last-axis reduction on a dense contiguous input. Output drops
    /// the reduced axis (`keep_dim = false`) or keeps it as size 1
    /// (`keep_dim = true`). The runtime currently only supports
    /// reducing the *last* axis; non-last reductions can be expressed
    /// by permuting the input first.
    Reduce {
        /// Which reduction kind to perform. `Mean` is not represented
        /// here — at the runtime layer it's composed as `Sum / N`.
        kind: ReduceKind,
        /// `true` ⇒ output has the same rank as input with a size-1
        /// axis at the reduced position. `false` ⇒ output drops the
        /// reduced axis.
        keep_dim: bool,
    },
    /// Axis-0 gather (embedding-style lookup). Inputs (in
    /// [`Tensor::op_inputs`] order) are `[src, indices]` where `src`
    /// is a dense, contiguous tensor with rank ≥ 1 and `indices` is a
    /// dense rank-1 `U32` tensor (`[n]`). Output is dense
    /// `indices.shape() ++ src.shape()[1..]` with the same dtype as
    /// `src`. Wired source dtypes are `BF16` (the dense embedding
    /// case) and `U32` (the packed-quantized-weight case used by
    /// `nn::embed_tokens`); the underlying JIT machinery is
    /// dtype-agnostic and other dtypes / index ranks / axes land
    /// alongside their first consumer.
    Gather,
    /// Affine-dequantize: `out = w * scale + bias` per group. Inputs
    /// (in [`Tensor::op_inputs`] order) are `[w_q, scales, biases]`
    /// where `w_q` is dense `U32` (packed `bits`-wide values, `32 /
    /// bits` per element), `scales` and `biases` are dense `BF16`.
    /// Output is dense `BF16` of `w_q.shape()` with the last axis
    /// scaled to logical width (e.g. `w_q [[N, K * bits / 32]] ⇒
    /// out [N, K]`). The runtime composes `embed_tokens` as
    /// `gather(w_q, idx) + gather(scales, idx) + gather(biases, idx)
    /// → Dequantize`.
    Dequantize {
        /// Quantization group size (elements per scale/bias).
        group_size: u32,
        /// Quantization bit-width (e.g. 4 for int4).
        bits: u8,
    },
    /// Rotary positional embedding on a dense bf16 tensor. Inputs (in
    /// [`Tensor::op_inputs`] order) are `[input, offset]` where
    /// `input` is a dense, contiguous, BF16 tensor of rank 4
    /// (`[B, H, T, D]`) and `offset` is a dense, contiguous,
    /// length-1 [`DType::I32`] tensor giving the starting sequence
    /// position. Output has the same shape, dtype, and layout as
    /// `input`. Only `traditional=false, forward=true,
    /// hs_transpose=false, with_freqs=false` is wired today; see
    /// `kernels::rope` for the rationale.
    Rope {
        /// `log2(theta)` — the host pre-computes this so the MSL
        /// kernel can use `exp2(-d * base_log2)` directly. Mirrors
        /// `float base = std::log2(base_)` in MLX's `rope.cpp`.
        base_log2: f32,
        /// MLX's `scale_` — multiplies the position before computing
        /// the rotation angle. `1.0` for the standard formula.
        scale: f32,
        /// Number of leading per-head slots that get rotated (`dims_`
        /// in MLX). Must be even and ≤ the trailing axis of `input`.
        /// Equals `head_dim` for full-rotation models like Qwen2.
        rotary_dims: u32,
    },
    /// Numerically-stable softmax along the trailing axis. Inputs (in
    /// [`Tensor::op_inputs`] order) are `[input]` — a dense,
    /// contiguous, BF16 tensor with rank ≥ 1 whose trailing axis size
    /// is ≤ `kernels::softmax::BLOCK_AXIS_LIMIT` (4096). Output has
    /// the same shape and dtype. Non-last-axis and `looped_softmax_*`
    /// (axis > 4096) land alongside their first consumer.
    Softmax {
        /// `false` ⇒ `block_softmax_bfloat16` (bf16 accumulator;
        /// matches `mx.softmax(x, precise=False)`). `true` ⇒
        /// `block_softmax_precise_bfloat16` (float accumulator;
        /// matches `mx.softmax(x, precise=True)`, the standard
        /// choice for bf16 attention scores in inference).
        precise: bool,
    },
    /// Batched dense matrix multiply. Inputs (in [`Tensor::op_inputs`]
    /// order) are `[a, b]` — both dense, contiguous, [`DType::BF16`]
    /// tensors with matching leading dims and aligned matmul
    /// dimensions: `a.shape() = [..., M, K]`,
    /// `b.shape() = [..., K, N]`. Output is dense bf16 with shape
    /// `[..., M, N]`.
    ///
    /// Dispatches to cider-press's own naive `gemm_bfloat16` kernel
    /// (not MLX-derived); steel-tiled GEMM is deferred (see
    /// `docs/ARCHITECTURE.md`). Non-contiguous inputs (transposed
    /// views) materialize via [`Tensor::copy`] first — same
    /// contract as `softmax` / `rope` / reductions.
    MatMul,
    /// In-place write of `src`'s rows into a persistent,
    /// caller-owned slab buffer at row `offset_rows`. Inputs (in
    /// [`Tensor::op_inputs`] order) are `[slab, src]`. Unlike every
    /// other op, `eval` does NOT allocate a fresh output — it binds the
    /// slab buffer (resolved from `inputs[0]`) as the destination,
    /// encodes a copy of `src` into `slab[offset_rows..]`, and caches
    /// the slab buffer as this op's output. Output shape and dtype equal
    /// the slab's; the output layout is dense (the slab is dense).
    /// bf16 / rank-3 slab only (the KV cache is the only consumer; broaden when another arrives).
    SliceUpdate {
        /// Row index (along axis 0 of the slab) at which `src`'s first
        /// row lands. Byte offset is `offset_rows * (n_kv_heads * head_dim) * 2`.
        offset_rows: usize,
    },
    /// Fused scaled-dot-product attention. Inputs (in
    /// [`Tensor::op_inputs`] order) are `[q, k, v]` plus an optional
    /// `mask`. `q` is dense `[1, H_q, T_q, D]`; `k`/`v` are the
    /// [`KvCache`](crate::KvCache) views `[1, H_kv, T_cache, D]`, read
    /// strided in place. Output is dense `[1, H_q, T_q, D]`. The decode
    /// (vector) kernel handles GQA via `gqa_factor`.
    Sdpa {
        /// Softmax scale (typically `1/sqrt(head_dim)`).
        scale: f32,
        /// `H_q / H_kv` — query heads per kv head.
        gqa_factor: usize,
        /// Apply a causal mask (false for decode `T_q=1` full attention).
        causal: bool,
    },
    /// Index of the maximum element along `axis`. Input
    /// must be a dense, contiguous BF16 tensor. Output dtype is
    /// [`DType::U32`] with `axis` dropped. Only last-axis reductions
    /// are wired (same constraint as [`OpKind::Reduce`]).
    ArgReduce {
        /// Which index reduction to perform.
        kind: ArgReduceKind,
        /// The axis being reduced (resolved, non-negative).
        axis: usize,
    },
}

/// Element-wise binary operations supported by [`OpKind::Binary`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinaryOp {
    /// `c = a + b`.
    Add,
    /// `c = a * b`.
    Mul,
}

/// Element-wise unary operations supported by [`OpKind::Unary`].
///
/// Includes the two ops the `rms_norm` composition needs and the two
/// primitives MLX itself composes `silu` and `gelu` from in `mlx.nn`.
/// MLX's `unary.metal` has many more (Exp / Sin / Tanh / …) that land
/// the same way when their first consumer arrives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnaryOp {
    /// `out = x * x`.
    Square,
    /// `out = 1 / sqrt(x)`. Maps to the hardware `metal::rsqrt`.
    Rsqrt,
    /// `out = 1 / (1 + exp(-x))`. Primitive for `SiLU` composition.
    Sigmoid,
    /// `out = erf(x)`. Primitive for exact-`GELU` composition.
    Erf,
}

/// Reduction kinds supported by [`OpKind::Reduce`].
///
/// `Mean` is intentionally absent — it's composed at the runtime
/// layer as `Sum / N`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReduceKind {
    /// Sum of elements.
    Sum,
    /// Product of elements.
    Prod,
    /// Minimum element.
    Min,
    /// Maximum element.
    Max,
}

/// Index-reduction kinds for [`OpKind::ArgReduce`]. Only `ArgMax` is
/// wired (greedy decode).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArgReduceKind {
    ArgMax,
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
                view: None,
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
                view: None,
                cache: OnceLock::new(),
            }),
        }
    }

    /// Allocate a dense tensor of zeros with `shape` and `dtype` on
    /// `device`. The backing buffer is real shared-storage memory;
    /// host-side reads via [`Tensor::cpu_bytes`] are safe immediately.
    pub fn zeros(device: &Device, shape: impl Into<Shape>, dtype: DType) -> Result<Self> {
        let shape = shape.into();
        let byte_count = checked_byte_count(&shape, dtype)?;
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
        let expected = checked_byte_count(&shape, dtype)?;
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

    pub(crate) fn host_leaf(
        device: &Device,
        shape: Shape,
        dtype: DType,
        layout: Layout,
        buffer: Buffer<u8>,
    ) -> Self {
        Self::host_leaf_storage(
            device,
            shape,
            dtype,
            layout,
            LeafStorage::Dense(crate::buffer_pool::PooledBuffer::unpooled(buffer)),
        )
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
                view: None,
                cache,
            }),
        }
    }

    /// Crate-private constructor for view tensors. Used by the public
    /// view-building APIs (`reshape`, `permute`, `slice`,
    /// `broadcast_to`). The new tensor inherits `source`'s dtype and
    /// device; the caller chooses `shape` and `strides` and supplies a
    /// `byte_offset` relative to `source`'s backing leaf (i.e. the
    /// caller is expected to have already folded any
    /// `source`-is-also-a-view indirection via
    /// [`Tensor::flatten_view`]).
    pub(crate) fn view(source: Tensor, byte_offset: usize, shape: Shape, strides: Strides) -> Self {
        let device = source.inner.device.clone();
        let dtype = source.inner.dtype;
        let layout = Layout::Dense { strides };
        Self {
            inner: Arc::new(TensorInner {
                shape,
                dtype,
                layout,
                device,
                op: None,
                view: Some(ViewSource {
                    source,
                    byte_offset,
                }),
                cache: OnceLock::new(),
            }),
        }
    }

    /// Walk through any view-of-view chain to the storage-owning
    /// tensor (a host leaf or op node) and return it plus the
    /// accumulated byte offset. The returned tensor either has a
    /// populated cache (host leaf, evaluated op) or will get one when
    /// its op is evaluated.
    ///
    /// View-building APIs use this to keep new views one hop away
    /// from the storage owner instead of layering chain after chain.
    pub(crate) fn flatten_view(&self) -> (Tensor, usize) {
        let mut inner = &self.inner;
        let mut offset: usize = 0;
        while let Some(v) = inner.view.as_ref() {
            offset = offset
                .checked_add(v.byte_offset)
                .expect("byte offset overflowed usize during view-chain flatten");
            inner = &v.source.inner;
        }
        (
            Tensor {
                inner: inner.clone(),
            },
            offset,
        )
    }

    /// Crate-private constructor for op tensors (used by op builders
    /// like [`Tensor::copy`] and [`Tensor::quantized_matmul`]).
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
                view: None,
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

    /// Schedule an in-place write of `src` into this slab
    /// tensor at row `offset_rows`, returning an [`OpKind::SliceUpdate`]
    /// op tensor whose logical shape equals the slab's.
    ///
    /// `self` must be a dense bf16 rank-3 slab
    /// (`[max_rows, n_kv_heads, head_dim]`); `src` must be dense bf16
    /// rank-3 with the same trailing two dims. Preconditions (bf16,
    /// rank-3 slab and src, matching row shape `[n_kv_heads, head_dim]`,
    /// and an in-bounds `offset_rows`) are validated here at construction.
    pub fn slice_update(&self, src: &Tensor, offset_rows: usize) -> Result<Self> {
        let device = self.inner.device.as_ref().ok_or_else(|| {
            Error::InvalidArgument("slice_update: slab is a placeholder (no device)".into())
        })?;
        // The slab must be a caller-owned dense leaf. `eval` writes in
        // place through a refcount-bumped handle to the slab's buffer and
        // caches the SliceUpdate output as *unpooled* — so a pool-minted op
        // output as the slab would let the pool reclaim that buffer while
        // this unpooled alias is still live (corruption), and a view never
        // resolves to a dense buffer in `dense_input_buffer`. Only the slab
        // input is constrained; `src` may be an op or view (the dispatcher
        // chases its chain).
        if self.inner.op.is_some() || self.inner.view.is_some() {
            return Err(Error::InvalidArgument(
                "slice_update: slab must be a dense leaf, not an op output or a view".into(),
            ));
        }
        // Scope: bf16 / rank-3 slab. The KV cache is the only
        // consumer and is bf16; broaden when a second consumer needs it.
        if self.inner.dtype != DType::BF16 {
            return Err(Error::InvalidArgument(format!(
                "slice_update: only BF16 slabs are supported (got {:?})",
                self.inner.dtype
            )));
        }
        if src.inner.dtype != DType::BF16 {
            return Err(Error::InvalidArgument(format!(
                "slice_update: src must be BF16 (got {:?})",
                src.inner.dtype
            )));
        }
        let slab_dims = self.inner.shape.dims();
        let src_dims = src.inner.shape.dims();
        if slab_dims.len() != 3 || src_dims.len() != 3 {
            return Err(Error::InvalidArgument(format!(
                "slice_update: slab and src must be rank-3 \
                 [rows, n_kv_heads, head_dim] (slab {slab_dims:?}, src {src_dims:?})"
            )));
        }
        if slab_dims[1..] != src_dims[1..] {
            return Err(Error::InvalidArgument(format!(
                "slice_update: src row shape {:?} must match slab row shape {:?}",
                &src_dims[1..],
                &slab_dims[1..]
            )));
        }
        if !self.layout().is_dense_contiguous(self.shape()) {
            return Err(Error::InvalidArgument(
                "slice_update: slab must be dense and contiguous".into(),
            ));
        }
        if !src.layout().is_dense_contiguous(src.shape()) {
            return Err(Error::InvalidArgument(
                "slice_update: src must be dense and contiguous".into(),
            ));
        }
        let end = offset_rows.checked_add(src_dims[0]).ok_or_else(|| {
            Error::InvalidArgument("slice_update: offset_rows + src rows overflows usize".into())
        })?;
        if end > slab_dims[0] {
            return Err(Error::InvalidArgument(format!(
                "slice_update: write of {} rows at offset {offset_rows} is out of bounds \
                 for a slab with {} rows",
                src_dims[0], slab_dims[0]
            )));
        }
        Ok(Self::op_tensor(
            device,
            self.inner.shape.clone(),
            self.inner.dtype,
            dense_layout(&self.inner.shape),
            OpKind::SliceUpdate { offset_rows },
            vec![self.clone(), src.clone()],
        ))
    }

    /// Schedule `self + other` (element-wise, NumPy-broadcasting).
    /// See [`Tensor::binary`] for the broadcasting rules and shared
    /// preconditions.
    pub fn add(&self, other: &Tensor) -> Result<Self> {
        self.binary(other, BinaryOp::Add)
    }

    /// Schedule `self * other` (element-wise, NumPy-broadcasting).
    /// See [`Tensor::binary`] for the broadcasting rules and shared
    /// preconditions.
    pub fn mul(&self, other: &Tensor) -> Result<Self> {
        self.binary(other, BinaryOp::Mul)
    }

    /// Schedule an element-wise binary op between `self` and `other`
    /// with `NumPy` broadcasting.
    ///
    /// Both inputs must be dense (no quantized operand support yet),
    /// share a [`Device`], and share a [`DType`]. The output shape is
    /// the right-aligned broadcast of the two input shapes: shorter
    /// rank is padded on the left with size-1 axes, then each axis must
    /// be either equal between the two inputs or have one side equal
    /// to 1 (which the dispatcher will broadcast via zero strides).
    ///
    /// The returned tensor is dense and contiguous over the broadcast
    /// output shape; no GPU work runs until [`Tensor::eval`].
    ///
    /// Public so external code can pick the op without re-doing the
    /// shape/device/dtype validation in [`Tensor::add`] / [`Tensor::mul`].
    pub fn binary(&self, other: &Tensor, op: BinaryOp) -> Result<Self> {
        let lhs_device = self.inner.device.as_ref().ok_or_else(|| {
            Error::InvalidArgument(
                "binary: cannot apply an op to a placeholder (no device) on lhs".into(),
            )
        })?;
        let rhs_device = other.inner.device.as_ref().ok_or_else(|| {
            Error::InvalidArgument(
                "binary: cannot apply an op to a placeholder (no device) on rhs".into(),
            )
        })?;
        if !lhs_device.ptr_eq(rhs_device) {
            return Err(Error::InvalidArgument(
                "binary: lhs and rhs are on different devices".into(),
            ));
        }
        if self.inner.dtype != other.inner.dtype {
            return Err(Error::InvalidArgument(format!(
                "binary: dtype mismatch (lhs={:?}, rhs={:?}); \
                 mixed-dtype binary ops are not supported yet",
                self.inner.dtype, other.inner.dtype,
            )));
        }
        match (&self.inner.layout, &other.inner.layout) {
            (Layout::Dense { .. }, Layout::Dense { .. }) => {}
            _ => {
                return Err(Error::InvalidArgument(
                    "binary: only dense tensors are supported (quantized binary ops are not in scope)".into(),
                ));
            }
        }

        let out_shape = broadcast_shape(self.shape(), other.shape())?;

        // Align each operand to the output shape via `broadcast_to` if
        // its shape differs. `broadcast_to` is a zero-copy view that
        // pads leading axes with stride-0 and expands size-1 axes; if
        // the operand already matches, we keep the original (avoids an
        // unnecessary view node on the vv-fast path).
        let lhs = if self.shape().dims() == out_shape.dims() {
            self.clone()
        } else {
            self.broadcast_to(out_shape.clone())?
        };
        let rhs = if other.shape().dims() == out_shape.dims() {
            other.clone()
        } else {
            other.broadcast_to(out_shape.clone())?
        };

        let layout = dense_layout(&out_shape);
        Ok(Self::op_tensor(
            lhs_device,
            out_shape,
            self.inner.dtype,
            layout,
            OpKind::Binary { op },
            vec![lhs, rhs],
        ))
    }

    /// Schedule `out = x * x` element-wise.
    /// See [`Tensor::unary`] for shared preconditions.
    pub fn square(&self) -> Result<Self> {
        self.unary(UnaryOp::Square)
    }

    /// Schedule `out = 1 / sqrt(x)` element-wise.
    /// See [`Tensor::unary`] for shared preconditions.
    pub fn rsqrt(&self) -> Result<Self> {
        self.unary(UnaryOp::Rsqrt)
    }

    /// Schedule `out = 1 / (1 + exp(-x))` element-wise. Primitive for
    /// the `silu` composition. See [`Tensor::unary`] for shared
    /// preconditions.
    pub fn sigmoid(&self) -> Result<Self> {
        self.unary(UnaryOp::Sigmoid)
    }

    /// Schedule `out = erf(x)` element-wise. Primitive for the exact
    /// `gelu` composition. See [`Tensor::unary`] for shared
    /// preconditions.
    pub fn erf(&self) -> Result<Self> {
        self.unary(UnaryOp::Erf)
    }

    /// Schedule an element-wise unary op. Output inherits this
    /// tensor's [`Shape`], [`DType`], and [`Device`]; the output
    /// [`Layout`] is dense and contiguous.
    ///
    /// Input must be dense (quantized inputs are not supported) and
    /// contiguous in memory — the runtime currently only wires the
    /// MLX `v_` (contiguous) kernel family for unary ops. Apply
    /// [`Tensor::copy`] first to materialise a non-contiguous view.
    pub fn unary(&self, op: UnaryOp) -> Result<Self> {
        let device = self.inner.device.as_ref().ok_or_else(|| {
            Error::InvalidArgument("unary: cannot apply an op to a placeholder (no device)".into())
        })?;
        if !self.inner.dtype.is_float() {
            return Err(Error::InvalidArgument(format!(
                "unary: dtype {:?} is not supported (only F32/F16/BF16 are wired \
                 at the kernels layer)",
                self.inner.dtype,
            )));
        }
        let strides = match self.layout() {
            Layout::Dense { strides } => strides,
            Layout::Quantized(_) => {
                return Err(Error::InvalidArgument(
                    "unary: quantized tensors are not supported".into(),
                ));
            }
        };
        if self.inner.view.is_some() {
            return Err(Error::InvalidArgument(
                "unary: view inputs are not supported; copy() first to materialise \
                 a dense version (eval-side dispatch reads inputs as dense leaves \
                 and does not resolve view chains)"
                    .into(),
            ));
        }
        if !strides.is_contiguous(self.shape()) {
            return Err(Error::InvalidArgument(format!(
                "unary: input must be contiguous (got shape {:?} strides {:?}); \
                 copy() first to materialise a dense version",
                self.shape().dims(),
                strides.as_slice(),
            )));
        }
        let layout = dense_layout(self.shape());
        Ok(Self::op_tensor(
            device,
            self.shape().clone(),
            self.inner.dtype,
            layout,
            OpKind::Unary { op },
            vec![self.clone()],
        ))
    }

    /// Schedule a sum reduction along `axis`. See [`Tensor::reduce`]
    /// for the shared preconditions; negative axes count from the end
    /// (`-1` ⇒ last axis).
    pub fn sum(&self, axis: i64, keep_dim: bool) -> Result<Self> {
        self.reduce(ReduceKind::Sum, axis, keep_dim)
    }

    /// Schedule a product reduction along `axis`. See [`Tensor::reduce`].
    pub fn prod(&self, axis: i64, keep_dim: bool) -> Result<Self> {
        self.reduce(ReduceKind::Prod, axis, keep_dim)
    }

    /// Schedule a min reduction along `axis`. See [`Tensor::reduce`].
    pub fn min(&self, axis: i64, keep_dim: bool) -> Result<Self> {
        self.reduce(ReduceKind::Min, axis, keep_dim)
    }

    /// Schedule a max reduction along `axis`. See [`Tensor::reduce`].
    pub fn max(&self, axis: i64, keep_dim: bool) -> Result<Self> {
        self.reduce(ReduceKind::Max, axis, keep_dim)
    }

    /// Schedule a mean reduction along `axis`, composed as `sum / N`.
    /// `N` is the size of the reduced axis. The divisor is built as a
    /// 1-element tensor on the same device + dtype and multiplied
    /// against the sum via [`Tensor::mul`]; the broadcast view is
    /// dispatched through the existing strided binary path.
    pub fn mean(&self, axis: i64, keep_dim: bool) -> Result<Self> {
        let axis_resolved = resolve_axis(self.shape().rank(), axis)?;
        let n = self.shape().dims()[axis_resolved];
        if n == 0 {
            return Err(Error::InvalidArgument(
                "mean: cannot reduce a size-0 axis".into(),
            ));
        }
        let summed = self.sum(axis, keep_dim)?;
        let device = self
            .inner
            .device
            .as_ref()
            .expect("device verified by sum() above");

        // Build a [1] divisor tensor with bf16-rounded `1/N` (or f32
        // / f16 equivalent) on the same device + dtype. Done at
        // op-construction time so the value flows through the graph
        // alongside the sum.
        let inv_n = recip_scalar(n, self.inner.dtype)?;
        let divisor = host_scalar_tensor(device, self.inner.dtype, inv_n)?;
        summed.mul(&divisor)
    }

    /// Schedule a last-axis-only reduction. `axis` accepts negative
    /// indexing (`-1` ⇒ last axis), and must currently resolve to the
    /// last axis — the runtime hasn't wired non-last reductions yet
    /// (MLX's `col_reduce` family is deferred). Use [`Tensor::permute`]
    /// to move the target axis to the end first.
    ///
    /// `keep_dim = true` keeps the reduced axis as a size-1 dim
    /// (rank unchanged); `keep_dim = false` drops it.
    pub fn reduce(&self, kind: ReduceKind, axis: i64, keep_dim: bool) -> Result<Self> {
        let device = self.inner.device.as_ref().ok_or_else(|| {
            Error::InvalidArgument("reduce: cannot apply an op to a placeholder (no device)".into())
        })?;
        if !self.inner.dtype.is_float() {
            return Err(Error::InvalidArgument(format!(
                "reduce: dtype {:?} is not supported (only F32/F16/BF16 are wired \
                 at the kernels layer)",
                self.inner.dtype,
            )));
        }
        let strides = match self.layout() {
            Layout::Dense { strides } => strides,
            Layout::Quantized(_) => {
                return Err(Error::InvalidArgument(
                    "reduce: quantized tensors are not supported".into(),
                ));
            }
        };
        if self.inner.view.is_some() {
            return Err(Error::InvalidArgument(
                "reduce: view inputs are not supported; copy() first to materialise \
                 a dense version (eval-side dispatch reads inputs as dense leaves \
                 and does not resolve view chains)"
                    .into(),
            ));
        }
        if !strides.is_contiguous(self.shape()) {
            return Err(Error::InvalidArgument(format!(
                "reduce: input must be contiguous (got shape {:?} strides {:?}); \
                 copy() first to materialise a dense version",
                self.shape().dims(),
                strides.as_slice(),
            )));
        }
        let rank = self.shape().rank();
        let axis_resolved = resolve_axis(rank, axis)?;
        if axis_resolved != rank - 1 {
            return Err(Error::InvalidArgument(format!(
                "reduce: only last-axis reductions are wired (got axis {axis_resolved} of \
                 rank {rank}); permute the target axis to the end first",
            )));
        }
        let mut out_dims: Vec<usize> = self.shape().dims().to_vec();
        if keep_dim {
            out_dims[axis_resolved] = 1;
        } else {
            out_dims.remove(axis_resolved);
        }
        let out_shape = Shape::from(out_dims);
        let layout = dense_layout(&out_shape);
        Ok(Self::op_tensor(
            device,
            out_shape,
            self.inner.dtype,
            layout,
            OpKind::Reduce { kind, keep_dim },
            vec![self.clone()],
        ))
    }

    /// Index of the maximum along `axis` (must be the last axis), as a
    /// `U32` tensor with `axis` dropped. Input must be a dense,
    /// contiguous BF16 tensor (the only wired kernel instantiation).
    pub fn argmax(&self, axis: i64) -> Result<Self> {
        let device = self.inner.device.as_ref().ok_or_else(|| {
            Error::InvalidArgument("argmax: cannot apply an op to a placeholder (no device)".into())
        })?;
        if self.inner.dtype != DType::BF16 {
            return Err(Error::InvalidArgument(format!(
                "argmax: only BF16 input is wired (got {:?})",
                self.inner.dtype,
            )));
        }
        let strides = match self.layout() {
            Layout::Dense { strides } => strides,
            Layout::Quantized(_) => {
                return Err(Error::InvalidArgument(
                    "argmax: quantized tensors are not supported".into(),
                ));
            }
        };
        if self.inner.view.is_some() {
            return Err(Error::InvalidArgument(
                "argmax: view inputs are not supported; copy() first to materialise \
                 a dense version (eval-side dispatch reads inputs as dense leaves \
                 and does not resolve view chains)"
                    .into(),
            ));
        }
        if !strides.is_contiguous(self.shape()) {
            return Err(Error::InvalidArgument(format!(
                "argmax: input must be contiguous (got shape {:?} strides {:?}); \
                 copy() first to materialise a dense version",
                self.shape().dims(),
                strides.as_slice(),
            )));
        }
        let rank = self.shape().rank();
        let axis_resolved = resolve_axis(rank, axis)?;
        if axis_resolved != rank - 1 {
            return Err(Error::InvalidArgument(format!(
                "argmax: only last-axis is wired (got axis {axis_resolved} of rank {rank})",
            )));
        }
        let mut out_dims: Vec<usize> = self.shape().dims().to_vec();
        out_dims.remove(axis_resolved);
        let out_shape = Shape::from(out_dims);
        let layout = dense_layout(&out_shape);
        Ok(Self::op_tensor(
            device,
            out_shape,
            DType::U32,
            layout,
            OpKind::ArgReduce {
                kind: ArgReduceKind::ArgMax,
                axis: axis_resolved,
            },
            vec![self.clone()],
        ))
    }

    /// Schedule an axis-0 gather (embedding-style lookup):
    /// `out[t, ..] = self[indices[t], ..]`.
    ///
    /// Preconditions (mirroring the kernels-crate scope; broader
    /// instantiations land alongside their first consumer):
    ///
    /// - `self` (the source) is dense, contiguous, and rank ≥ 1.
    /// - `indices` is a dense, contiguous, rank-1 [`DType::U32`]
    ///   tensor on the same [`Device`] as `self`.
    /// - Source dtype is [`DType::BF16`] (the dense embedding case) or
    ///   [`DType::U32`] (the packed-quantized-weight case used by
    ///   `nn::embed_tokens`).
    ///
    /// Output shape is `indices.shape() ++ self.shape()[1..]` with
    /// the same dtype as `self`.
    pub fn gather(&self, indices: &Self) -> Result<Self> {
        let device = self.inner.device.as_ref().ok_or_else(|| {
            Error::InvalidArgument("gather: cannot apply an op to a placeholder (no device)".into())
        })?;
        let idx_device = indices.inner.device.as_ref().ok_or_else(|| {
            Error::InvalidArgument("gather: cannot use a placeholder as the indices input".into())
        })?;
        if !device.ptr_eq(idx_device) {
            return Err(Error::InvalidArgument(
                "gather: src and indices are on different devices".into(),
            ));
        }
        let src_strides = match self.layout() {
            Layout::Dense { strides } => strides,
            Layout::Quantized(_) => {
                return Err(Error::InvalidArgument(
                    "gather: quantized source tensors are not supported yet".into(),
                ));
            }
        };
        if !src_strides.is_contiguous(self.shape()) {
            return Err(Error::InvalidArgument(format!(
                "gather: src must be contiguous (got shape {:?} strides {:?}); \
                 copy() first to materialise a dense version",
                self.shape().dims(),
                src_strides.as_slice(),
            )));
        }
        if self.rank() == 0 {
            return Err(Error::InvalidArgument(
                "gather: src must have rank ≥ 1".into(),
            ));
        }
        // gather is dtype-agnostic at the Metal level (it's a pure
        // data-mover); the runtime wires the two source dtypes the
        // embedding-table integration needs: BF16 for the scales /
        // biases / fully-dense source, U32 for packed quantized
        // weights. Other dtypes land alongside their first consumer.
        if !matches!(self.inner.dtype, DType::BF16 | DType::U32) {
            return Err(Error::InvalidArgument(format!(
                "gather: source dtype {:?} is not wired (only BF16 and U32 today)",
                self.inner.dtype,
            )));
        }

        let idx_strides = match indices.layout() {
            Layout::Dense { strides } => strides,
            Layout::Quantized(_) => {
                return Err(Error::InvalidArgument(
                    "gather: quantized indices are not supported".into(),
                ));
            }
        };
        if !idx_strides.is_contiguous(indices.shape()) {
            return Err(Error::InvalidArgument(format!(
                "gather: indices must be contiguous (got shape {:?} strides {:?})",
                indices.shape().dims(),
                idx_strides.as_slice(),
            )));
        }
        if indices.rank() != 1 {
            return Err(Error::InvalidArgument(format!(
                "gather: only rank-1 indices are wired (got rank {})",
                indices.rank(),
            )));
        }
        if indices.inner.dtype != DType::U32 {
            return Err(Error::InvalidArgument(format!(
                "gather: indices must be U32 (got {:?})",
                indices.inner.dtype,
            )));
        }

        // out shape: [n_indices] ++ src.shape()[1..]
        let mut out_dims: Vec<usize> = Vec::with_capacity(self.rank());
        out_dims.extend_from_slice(indices.shape().dims());
        out_dims.extend_from_slice(&self.shape().dims()[1..]);
        let out_shape = Shape::from(out_dims);
        let layout = dense_layout(&out_shape);
        Ok(Self::op_tensor(
            device,
            out_shape,
            self.inner.dtype,
            layout,
            OpKind::Gather,
            vec![self.clone(), indices.clone()],
        ))
    }

    /// Schedule an affine dequantize: `out = w_q * scale + bias` per
    /// group, producing a dense `BF16` tensor from a quantized triple.
    ///
    /// `w_q`, `scales`, and `biases` are the same three tensors a
    /// [`QuantizedWeight`](crate::QuantizedWeight) carries internally;
    /// expose them via
    /// [`QuantizedWeight::components`](crate::QuantizedWeight::components)
    /// and pipe through `gather` first if you want only a per-row
    /// dequantize.
    ///
    /// Preconditions:
    ///
    /// - All three inputs share a [`Device`].
    /// - `w_q` is dense, contiguous, [`DType::U32`].
    /// - `scales` and `biases` are dense, contiguous, [`DType::BF16`].
    /// - `w_q.shape() == scales.shape() == biases.shape()` in all
    ///   leading axes; the trailing axis differs (packed
    ///   `[..., K * bits / 32]` for `w_q`, `[..., K / group_size]`
    ///   for `scales`/`biases`).
    /// - `group_size` and `bits` match the (`group_size=64`,
    ///   `bits=4`) wired by `kernels::dequantize::affine_dequantize_bf16_gs64_b4`.
    ///
    /// Output shape is `w_q.shape()` with the last axis expanded to
    /// `(packed_cols) * (32 / bits)` — i.e. the logical (pre-quant)
    /// width.
    pub fn dequantize_affine(
        w_q: &Self,
        scales: &Self,
        biases: &Self,
        group_size: u32,
        bits: u8,
    ) -> Result<Self> {
        let device = w_q.inner.device.as_ref().ok_or_else(|| {
            Error::InvalidArgument(
                "dequantize_affine: w_q cannot be a placeholder (no device)".into(),
            )
        })?;
        for (name, t) in [("scales", scales), ("biases", biases)] {
            let other_device = t.inner.device.as_ref().ok_or_else(|| {
                Error::InvalidArgument(format!(
                    "dequantize_affine: {name} cannot be a placeholder (no device)",
                ))
            })?;
            if !device.ptr_eq(other_device) {
                return Err(Error::InvalidArgument(format!(
                    "dequantize_affine: {name} is on a different device than w_q",
                )));
            }
        }
        // Only the (group_size=64, bits=4) instantiation is wired in
        // the kernels crate. Surface this here so callers fail fast
        // rather than at dispatch time.
        if !(group_size == 64 && bits == 4) {
            return Err(Error::InvalidArgument(format!(
                "dequantize_affine: only group_size=64 bits=4 is wired \
                 (got group_size={group_size}, bits={bits})",
            )));
        }
        if w_q.inner.dtype != DType::U32 {
            return Err(Error::InvalidArgument(format!(
                "dequantize_affine: w_q must be U32 (got {:?})",
                w_q.inner.dtype,
            )));
        }
        if scales.inner.dtype != DType::BF16 || biases.inner.dtype != DType::BF16 {
            return Err(Error::InvalidArgument(format!(
                "dequantize_affine: scales/biases must be BF16 (got {:?}/{:?})",
                scales.inner.dtype, biases.inner.dtype,
            )));
        }
        for (name, t) in [("w_q", w_q), ("scales", scales), ("biases", biases)] {
            let strides = match t.layout() {
                Layout::Dense { strides } => strides,
                Layout::Quantized(_) => {
                    return Err(Error::InvalidArgument(format!(
                        "dequantize_affine: {name} must be a dense tensor",
                    )));
                }
            };
            if !strides.is_contiguous(t.shape()) {
                return Err(Error::InvalidArgument(format!(
                    "dequantize_affine: {name} must be contiguous (got shape {:?} strides {:?})",
                    t.shape().dims(),
                    strides.as_slice(),
                )));
            }
        }
        // Logical width: each u32 in w_q holds 32/bits dequantized
        // values along the last axis.
        let w_dims = w_q.shape().dims();
        let last_packed = *w_dims.last().expect("w_q rank ≥ 1");
        let logical_last = last_packed * (32 / bits as usize);
        let mut out_dims: Vec<usize> = w_dims.to_vec();
        *out_dims.last_mut().expect("rank ≥ 1") = logical_last;
        let out_shape = Shape::from(out_dims);
        let layout = dense_layout(&out_shape);
        Ok(Self::op_tensor(
            device,
            out_shape,
            DType::BF16,
            layout,
            OpKind::Dequantize { group_size, bits },
            vec![w_q.clone(), scales.clone(), biases.clone()],
        ))
    }

    /// Schedule a rotary positional embedding on this tensor against
    /// the given starting `offset`.
    ///
    /// Preconditions (mirroring the kernels-crate scope; broader
    /// instantiations land alongside their first consumer):
    ///
    /// - `self` is dense, contiguous, [`DType::BF16`], rank 4
    ///   (`[B, H, T, D]`).
    /// - `offset` is dense, contiguous, [`DType::I32`], length 1.
    /// - Both tensors are on the same [`Device`].
    /// - `rotary_dims` is even and ≤ `D`. Equals `D` for full-rotation
    ///   models like Qwen2.
    /// - `D` (the last axis) is even.
    ///
    /// The host pre-computes `log2(base)` so the kernel can use a
    /// single `exp2` per thread; pass the unscaled `base` (i.e. `θ`
    /// = `10000` for the original `RoPE` paper, `1_000_000` for Qwen2.5).
    ///
    /// Output has the same shape, dtype, and (dense, contiguous)
    /// layout as `self`. The Qwen2 inference specialization
    /// (`forward=true, traditional=false, hs_transpose=false`,
    /// `with_freqs=false`) is the only one wired.
    #[allow(clippy::too_many_lines)]
    pub fn rope(&self, offset: &Self, base: f32, scale: f32, rotary_dims: usize) -> Result<Self> {
        let device = self.inner.device.as_ref().ok_or_else(|| {
            Error::InvalidArgument("rope: cannot apply an op to a placeholder (no device)".into())
        })?;
        let off_device = offset.inner.device.as_ref().ok_or_else(|| {
            Error::InvalidArgument("rope: cannot use a placeholder as the offset input".into())
        })?;
        if !device.ptr_eq(off_device) {
            return Err(Error::InvalidArgument(
                "rope: input and offset are on different devices".into(),
            ));
        }
        if self.inner.view.is_some() {
            return Err(Error::InvalidArgument(
                "rope: view inputs are not supported; copy() first to materialise \
                 a dense version (eval-side dispatch reads inputs as dense leaves \
                 and does not resolve view chains)"
                    .into(),
            ));
        }
        if offset.inner.view.is_some() {
            return Err(Error::InvalidArgument(
                "rope: view offset is not supported; copy() first to materialise \
                 a dense version"
                    .into(),
            ));
        }
        if !scale.is_finite() {
            return Err(Error::InvalidArgument(format!(
                "rope: scale must be finite (got {scale})"
            )));
        }
        if !base.is_finite() || base <= 0.0 {
            return Err(Error::InvalidArgument(format!(
                "rope: base must be finite and > 0 (got {base})"
            )));
        }
        if self.inner.dtype != DType::BF16 {
            return Err(Error::InvalidArgument(format!(
                "rope: only BF16 input is wired (got {:?})",
                self.inner.dtype,
            )));
        }
        if offset.inner.dtype != DType::I32 {
            return Err(Error::InvalidArgument(format!(
                "rope: offset must be I32 (got {:?})",
                offset.inner.dtype,
            )));
        }
        let strides = match self.layout() {
            Layout::Dense { strides } => strides,
            Layout::Quantized(_) => {
                return Err(Error::InvalidArgument(
                    "rope: quantized input is not supported".into(),
                ));
            }
        };
        if !strides.is_contiguous(self.shape()) {
            return Err(Error::InvalidArgument(format!(
                "rope: input must be contiguous (got shape {:?} strides {:?}); \
                 copy() first to materialise a dense version",
                self.shape().dims(),
                strides.as_slice(),
            )));
        }
        let dims = self.shape().dims();
        if dims.len() != 4 {
            return Err(Error::InvalidArgument(format!(
                "rope: input must be rank 4 [B, H, T, D] (got rank {})",
                dims.len(),
            )));
        }
        let head_dim = dims[3];
        if head_dim == 0 || head_dim % 2 != 0 {
            return Err(Error::InvalidArgument(format!(
                "rope: head_dim (last axis) must be a positive even number (got {head_dim})"
            )));
        }
        if rotary_dims == 0 || rotary_dims > head_dim || rotary_dims % 2 != 0 {
            return Err(Error::InvalidArgument(format!(
                "rope: rotary_dims must be even and in (0, head_dim={head_dim}] \
                 (got {rotary_dims})"
            )));
        }
        let off_strides = match offset.layout() {
            Layout::Dense { strides } => strides,
            Layout::Quantized(_) => {
                return Err(Error::InvalidArgument(
                    "rope: quantized offset is not supported".into(),
                ));
            }
        };
        if !off_strides.is_contiguous(offset.shape()) {
            return Err(Error::InvalidArgument(
                "rope: offset must be contiguous".into(),
            ));
        }
        if offset.shape().elem_count() != 1 {
            return Err(Error::InvalidArgument(format!(
                "rope: only a single scalar offset is wired (got shape {:?})",
                offset.shape().dims(),
            )));
        }

        let out_shape = self.shape().clone();
        let layout = dense_layout(&out_shape);
        let rotary_dims_u32 = u32::try_from(rotary_dims)
            .map_err(|_| Error::InvalidArgument("rope: rotary_dims overflows u32".into()))?;
        Ok(Self::op_tensor(
            device,
            out_shape,
            DType::BF16,
            layout,
            OpKind::Rope {
                base_log2: base.log2(),
                scale,
                rotary_dims: rotary_dims_u32,
            },
            vec![self.clone(), offset.clone()],
        ))
    }

    /// Schedule a numerically-stable softmax along the trailing axis.
    ///
    /// Preconditions (mirroring the kernels-crate scope; broader
    /// instantiations land alongside their first consumer):
    ///
    /// - `self` is dense, contiguous, [`DType::BF16`], rank ≥ 1.
    /// - The trailing axis size is ≤
    ///   [`cider_press_kernels::kernels::softmax::BLOCK_AXIS_LIMIT`]
    ///   (4096). Above this MLX swaps to a looped variant which isn't
    ///   wired yet; the op constructor rejects oversized inputs.
    /// - Non-last-axis softmax is not supported — permute the target
    ///   axis to the end first.
    ///
    /// `precise = false` matches `mx.softmax(x, precise=False)` (bf16
    /// accumulator). `precise = true` matches `mx.softmax(x,
    /// precise=True)` (float accumulator on the running `sum(exp)`)
    /// — the standard choice for bf16 attention scores in inference.
    ///
    /// Output has the same shape, dtype, and (dense, contiguous)
    /// layout as `self`.
    pub fn softmax(&self, precise: bool) -> Result<Self> {
        let device = self.inner.device.as_ref().ok_or_else(|| {
            Error::InvalidArgument(
                "softmax: cannot apply an op to a placeholder (no device)".into(),
            )
        })?;
        if self.inner.dtype != DType::BF16 {
            return Err(Error::InvalidArgument(format!(
                "softmax: only BF16 is wired (got {:?})",
                self.inner.dtype,
            )));
        }
        let strides = match self.layout() {
            Layout::Dense { strides } => strides,
            Layout::Quantized(_) => {
                return Err(Error::InvalidArgument(
                    "softmax: quantized input is not supported".into(),
                ));
            }
        };
        if self.inner.view.is_some() {
            return Err(Error::InvalidArgument(
                "softmax: view inputs are not supported; copy() first to materialise \
                 a dense version (eval-side dispatch reads inputs as dense leaves \
                 and does not resolve view chains)"
                    .into(),
            ));
        }
        if !strides.is_contiguous(self.shape()) {
            return Err(Error::InvalidArgument(format!(
                "softmax: input must be contiguous (got shape {:?} strides {:?}); \
                 copy() first to materialise a dense version",
                self.shape().dims(),
                strides.as_slice(),
            )));
        }
        let dims = self.shape().dims();
        let axis_size = *dims
            .last()
            .ok_or_else(|| Error::InvalidArgument("softmax: input must have rank ≥ 1".into()))?;
        if axis_size == 0 {
            return Err(Error::InvalidArgument(
                "softmax: trailing axis must be non-zero".into(),
            ));
        }
        if axis_size > cider_press_kernels::kernels::softmax::BLOCK_AXIS_LIMIT {
            return Err(Error::InvalidArgument(format!(
                "softmax: trailing axis size {axis_size} exceeds wired BLOCK_AXIS_LIMIT={}; \
                 looped variant is not wired yet",
                cider_press_kernels::kernels::softmax::BLOCK_AXIS_LIMIT,
            )));
        }

        let out_shape = self.shape().clone();
        let layout = dense_layout(&out_shape);
        Ok(Self::op_tensor(
            device,
            out_shape,
            DType::BF16,
            layout,
            OpKind::Softmax { precise },
            vec![self.clone()],
        ))
    }

    /// Schedule an affine-quantized matrix multiply: `y = self · dequant(weight)ᵀ`.
    ///
    /// `self` is a dense, contiguous BF16 activation of shape `[…, M, K]`
    /// (rank-1 `[K]` is the M=1 decode path). `weight` is rank-2 `[N, K]`.
    /// Output is dense BF16 `[…, M, N]`, or `[N]` for rank-1 input.
    ///
    /// The eval-side dispatcher routes M=1 to `affine_qmv` and M>1 to
    /// `affine_qmm_t`, both compiled from the vendored `quantized.metal`.
    ///
    /// Preconditions:
    /// - `self` has a device; `weight` has a device; same device.
    /// - `self.dtype() == BF16` and `self.layout()` is dense contiguous.
    /// - `weight.shape()` is rank-2.
    /// - Inner dim of `self` (last axis) matches `weight`'s K (second axis).
    /// - `transpose` must be `true` (i.e. `y = x @ W^T`, the standard
    ///   `Linear` orientation that the tied LM head also uses);
    ///   `transpose=false` is not yet implemented and returns `Err`
    ///   immediately.
    pub fn quantized_matmul(
        &self,
        weight: &crate::QuantizedWeight,
        transpose: bool,
    ) -> Result<Self> {
        if !transpose {
            return Err(Error::InvalidArgument(
                "quantized_matmul: transpose=false direction is not implemented yet \
                 (only the standard `y = x @ W^T` direction is wired)"
                    .into(),
            ));
        }
        let device = self.inner.device.as_ref().ok_or_else(|| {
            Error::InvalidArgument(
                "quantized_matmul: activation has no device (placeholder?)".into(),
            )
        })?;
        let w_device = weight.tensor().inner.device.as_ref().ok_or_else(|| {
            Error::InvalidArgument("quantized_matmul: weight has no device (placeholder?)".into())
        })?;
        if !device.ptr_eq(w_device) {
            return Err(Error::InvalidArgument(
                "quantized_matmul: activation and weight are on different devices".into(),
            ));
        }
        if self.inner.dtype != DType::BF16 {
            return Err(Error::InvalidArgument(format!(
                "quantized_matmul: activation must be BF16; got {:?}",
                self.inner.dtype
            )));
        }
        let strides = match self.layout() {
            Layout::Dense { strides } => strides,
            Layout::Quantized(_) => {
                return Err(Error::InvalidArgument(
                    "quantized_matmul: activation must be a dense tensor".into(),
                ));
            }
        };
        if !strides.is_contiguous(self.shape()) {
            return Err(Error::InvalidArgument(format!(
                "quantized_matmul: activation must be contiguous \
                 (shape {:?} strides {:?}); copy() first",
                self.shape().dims(),
                strides.as_slice(),
            )));
        }
        let w_dims = weight.tensor().shape().dims();
        if w_dims.len() != 2 {
            return Err(Error::InvalidArgument(format!(
                "quantized_matmul: weight must be rank 2; got rank {}",
                w_dims.len()
            )));
        }
        let (n, k) = (w_dims[0], w_dims[1]);
        let x_dims = self.shape().dims();
        if x_dims.is_empty() {
            return Err(Error::InvalidArgument(
                "quantized_matmul: activation must have rank >= 1".into(),
            ));
        }
        let x_k = *x_dims.last().expect("shape is non-empty");
        if x_k != k {
            return Err(Error::InvalidArgument(format!(
                "quantized_matmul: inner dim mismatch: activation last dim {x_k} != weight K {k}",
            )));
        }
        // The M>1 prefill path dispatches `affine_qmm_t`, whose wired
        // instantiation is aligned-only (N % 32 == 0). Reject the
        // unaligned M>1 case here rather than letting eval fail late;
        // M=1 decode routes to the generic `affine_qmv`, which has no
        // such constraint.
        let m: usize = if x_dims.len() == 1 {
            1
        } else {
            x_dims[..x_dims.len() - 1].iter().product::<usize>()
        };
        if m > 1 && n % 32 != 0 {
            return Err(Error::InvalidArgument(format!(
                "quantized_matmul: weight N ({n}) must be a multiple of 32 for the M>1 \
                 (prefill) path; only the aligned affine_qmm_t variant is wired",
            )));
        }
        let q = weight.quantization();
        // Build output shape: leading dims of self (all but last), then N.
        // For rank-1 activation [K] the leading dims are empty → output is [N].
        let mut out_dims: Vec<usize> = x_dims[..x_dims.len() - 1].to_vec();
        out_dims.push(n);
        let out_shape = Shape::from(out_dims);
        let out_layout = dense_layout(&out_shape);
        Ok(Self::op_tensor(
            device,
            out_shape,
            DType::BF16,
            out_layout,
            OpKind::QuantizedMatMul {
                group_size: q.group_size(),
                bits: q.bits(),
                transpose,
            },
            vec![weight.tensor().clone(), self.clone()],
        ))
    }

    /// Schedule a batched dense matrix multiply.
    ///
    /// `self.shape() = [..., M, K]`, `other.shape() = [..., K, N]`;
    /// the leading dims must match exactly (no broadcasting at the
    /// runtime layer — callers broadcast via [`Tensor::broadcast_to`]
    /// + [`Tensor::copy`] first, as GQA does). Output is `[..., M, N]` dense bf16.
    ///
    /// Preconditions (mirroring the kernels-crate scope):
    ///
    /// - Both inputs dense, contiguous, [`DType::BF16`], on the same
    ///   device, rank ≥ 2.
    /// - `self.shape()[-1] == other.shape()[-2]`.
    /// - Every dim except the last two is identical between `self`
    ///   and `other`.
    ///
    /// Transposed views (e.g. `K^T` in `Q @ K^T`) must be
    /// materialized via [`Tensor::copy`] before calling — the kernel
    /// does not currently take strides. This will be revisited when
    /// migrating to MLX's tiled `steel_gemm` family.
    pub fn matmul(&self, other: &Tensor) -> Result<Self> {
        let device = self.inner.device.as_ref().ok_or_else(|| {
            Error::InvalidArgument("matmul: cannot apply an op to a placeholder (no device)".into())
        })?;
        let other_device = other.inner.device.as_ref().ok_or_else(|| {
            Error::InvalidArgument("matmul: cannot apply an op to a placeholder (no device)".into())
        })?;
        if !device.ptr_eq(other_device) {
            return Err(Error::InvalidArgument(
                "matmul: both inputs must be on the same device".into(),
            ));
        }
        let lhs_dtype = self.inner.dtype;
        let rhs_dtype = other.inner.dtype;
        if lhs_dtype != DType::BF16 || rhs_dtype != DType::BF16 {
            return Err(Error::InvalidArgument(format!(
                "matmul: only BF16 is wired (got {lhs_dtype:?} × {rhs_dtype:?})",
            )));
        }
        let lhs_strides = match self.layout() {
            Layout::Dense { strides } => strides,
            Layout::Quantized(_) => {
                return Err(Error::InvalidArgument(
                    "matmul: quantized inputs are not supported (use Tensor::dequantize_affine \
                     or `Tensor::quantized_matmul`)"
                        .into(),
                ));
            }
        };
        let rhs_strides = match other.layout() {
            Layout::Dense { strides } => strides,
            Layout::Quantized(_) => {
                return Err(Error::InvalidArgument(
                    "matmul: quantized inputs are not supported".into(),
                ));
            }
        };
        if !lhs_strides.is_contiguous(self.shape()) {
            return Err(Error::InvalidArgument(format!(
                "matmul: LHS must be contiguous (got shape {:?} strides {:?}); \
                 copy() first to materialise",
                self.shape().dims(),
                lhs_strides.as_slice(),
            )));
        }
        if !rhs_strides.is_contiguous(other.shape()) {
            return Err(Error::InvalidArgument(format!(
                "matmul: RHS must be contiguous (got shape {:?} strides {:?}); \
                 copy() first to materialise",
                other.shape().dims(),
                rhs_strides.as_slice(),
            )));
        }

        let (m, n, _k_lhs, lead) = matmul_shapes(self.shape().dims(), other.shape().dims())?;
        let lhs_dims = self.shape().dims();

        let mut out_dims: Vec<usize> = lhs_dims[..lead].to_vec();
        out_dims.push(m);
        out_dims.push(n);
        let out_shape = Shape::from(out_dims);
        let layout = dense_layout(&out_shape);
        Ok(Self::op_tensor(
            device,
            out_shape,
            DType::BF16,
            layout,
            OpKind::MatMul,
            vec![self.clone(), other.clone()],
        ))
    }

    /// Fused scaled-dot-product attention (lazy). `q` `[1, H_q, 1, D]`,
    /// `k`/`v` `[1, H_kv, T_cache, D]` (typically [`KvCache`](crate::KvCache)
    /// views, read strided), optional additive `mask`. Returns
    /// `[1, H_q, 1, D]`.
    ///
    /// Decode-only: the vector dispatch handles a single query position
    /// (`T_q = 1`) and batch 1. Validates `head_dim` ∈ {64, 96, 128, 256}
    /// (the vector kernel's supported set), BF16 dtype, rank-4 shapes,
    /// `T_q`/batch, K/V cache-axis agreement, and GQA divisibility at
    /// construction.
    pub fn sdpa(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        scale: f32,
        gqa_factor: usize,
        causal: bool,
    ) -> Result<Tensor> {
        const SUPPORTED_HEAD_DIMS: [usize; 4] = [64, 96, 128, 256];
        let qd = q.shape().dims();
        let kd = k.shape().dims();
        let vd = v.shape().dims();
        if qd.len() != 4 || kd.len() != 4 || vd.len() != 4 {
            return Err(Error::InvalidArgument(format!(
                "sdpa: q/k/v must be rank-4 [1,H,T,D]; got {qd:?}, {kd:?}, {vd:?}"
            )));
        }
        let head_dim = qd[3];
        if !SUPPORTED_HEAD_DIMS.contains(&head_dim) {
            return Err(Error::InvalidArgument(format!(
                "sdpa: head_dim {head_dim} unsupported (vector kernel supports {SUPPORTED_HEAD_DIMS:?})"
            )));
        }
        if kd[3] != head_dim || vd[3] != head_dim {
            return Err(Error::InvalidArgument(format!(
                "sdpa: k/v head_dim must equal q's ({head_dim}); got k={}, v={}",
                kd[3], vd[3]
            )));
        }
        if q.dtype() != DType::BF16 || k.dtype() != DType::BF16 || v.dtype() != DType::BF16 {
            return Err(Error::InvalidArgument("sdpa: q/k/v must be BF16".into()));
        }
        if gqa_factor == 0 || qd[1] != kd[1] * gqa_factor {
            return Err(Error::InvalidArgument(format!(
                "sdpa: H_q ({}) must equal H_kv ({}) * gqa_factor ({gqa_factor})",
                qd[1], kd[1]
            )));
        }
        if qd[0] != 1 || kd[0] != 1 || vd[0] != 1 {
            return Err(Error::InvalidArgument(format!(
                "sdpa: batch must be 1 in the vector dispatch; got q/k/v = {}/{}/{}",
                qd[0], kd[0], vd[0]
            )));
        }
        if qd[2] != 1 {
            return Err(Error::InvalidArgument(format!(
                "sdpa: only decode T_q=1 is supported in the vector dispatch (got T_q={})",
                qd[2]
            )));
        }
        if kd[1] != vd[1] || kd[2] != vd[2] {
            return Err(Error::InvalidArgument(format!(
                "sdpa: k and v must agree on [H_kv, T_cache]; got k={kd:?}, v={vd:?}"
            )));
        }
        let device =
            q.inner.device.as_ref().ok_or_else(|| {
                Error::InvalidArgument("sdpa: q has no device (placeholder?)".into())
            })?;
        for (name, t) in [("k", k), ("v", v)] {
            let other = t.inner.device.as_ref().ok_or_else(|| {
                Error::InvalidArgument(format!("sdpa: {name} has no device (placeholder?)"))
            })?;
            if !device.ptr_eq(other) {
                return Err(Error::InvalidArgument(format!(
                    "sdpa: {name} is on a different device than q"
                )));
            }
        }
        if let Some(m) = mask {
            let md = m.inner.device.as_ref().ok_or_else(|| {
                Error::InvalidArgument("sdpa: mask has no device (placeholder?)".into())
            })?;
            if !device.ptr_eq(md) {
                return Err(Error::InvalidArgument(
                    "sdpa: mask is on a different device than q".into(),
                ));
            }
        }
        let out_shape = Shape::from(vec![qd[0], qd[1], qd[2], head_dim]);
        let layout = dense_layout(&out_shape);
        let mut inputs = vec![q.clone(), k.clone(), v.clone()];
        if let Some(m) = mask {
            inputs.push(m.clone());
        }
        Ok(Self::op_tensor(
            device,
            out_shape,
            DType::BF16,
            layout,
            OpKind::Sdpa {
                scale,
                gqa_factor,
                causal,
            },
            inputs,
        ))
    }

    /// Build a contiguous view of this tensor with a new shape.
    ///
    /// Zero-copy: the result shares the source's backing leaf. The
    /// new shape must have the same element count as the input, and
    /// the input must currently be contiguous. Non-contiguous inputs
    /// (transposed, broadcast, sliced along an inner dim) need to be
    /// materialized via [`Tensor::copy`] first; this restriction
    /// keeps the op cheap and avoids surprising hidden copies.
    ///
    /// The result is dense and contiguous with fresh row-major
    /// strides for the new shape, regardless of how the source's
    /// view chain is structured.
    pub fn reshape(&self, new_shape: impl Into<Shape>) -> Result<Self> {
        if self.inner.device.is_none() {
            return Err(Error::InvalidArgument(
                "reshape: cannot apply a view to a placeholder (no device)".into(),
            ));
        }
        let strides = match self.layout() {
            Layout::Dense { strides } => strides,
            Layout::Quantized(_) => {
                return Err(Error::InvalidArgument(
                    "reshape: quantized tensors are not supported".into(),
                ));
            }
        };
        if !strides.is_contiguous(self.shape()) {
            return Err(Error::InvalidArgument(format!(
                "reshape: input must be contiguous (got shape {:?} strides {:?})",
                self.shape().dims(),
                strides.as_slice(),
            )));
        }

        let new_shape = new_shape.into();
        if new_shape.elem_count() != self.elem_count() {
            return Err(Error::InvalidArgument(format!(
                "reshape: element count mismatch ({} -> {}); shapes {:?} -> {:?}",
                self.elem_count(),
                new_shape.elem_count(),
                self.shape().dims(),
                new_shape.dims(),
            )));
        }

        let new_strides = Strides::contiguous(&new_shape);
        let (source, byte_offset) = self.flatten_view();
        Ok(Self::view(source, byte_offset, new_shape, new_strides))
    }

    /// Build a permuted view of this tensor.
    ///
    /// `axes` must be a permutation of `0..self.rank()`. The result's
    /// `shape[i]` is `self.shape[axes[i]]` and the result's `strides[i]`
    /// is `self.strides[axes[i]]` — the same data, viewed in a
    /// different axis order. Zero-copy; shares the source's backing
    /// leaf at the same byte offset.
    ///
    /// Unlike [`Tensor::reshape`], `permute` does not require the
    /// input to be contiguous — it composes cleanly with prior view
    /// ops. (Subsequent ops that *do* require contiguity, like
    /// `reshape`, will need a materializing copy.)
    ///
    /// Matches MLX's `mx.transpose(x, axes)` semantics. Named
    /// `permute` rather than `transpose` because the latter carries
    /// linear-algebra "swap last two dims" baggage that doesn't fit
    /// a general axis reordering.
    pub fn permute(&self, axes: &[usize]) -> Result<Self> {
        if self.inner.device.is_none() {
            return Err(Error::InvalidArgument(
                "permute: cannot apply a view to a placeholder (no device)".into(),
            ));
        }
        let cur_strides = match self.layout() {
            Layout::Dense { strides } => strides,
            Layout::Quantized(_) => {
                return Err(Error::InvalidArgument(
                    "permute: quantized tensors are not supported".into(),
                ));
            }
        };

        let rank = self.rank();
        if axes.len() != rank {
            return Err(Error::InvalidArgument(format!(
                "permute: axes length {} does not match tensor rank {}",
                axes.len(),
                rank,
            )));
        }
        let mut seen = vec![false; rank];
        for &axis in axes {
            if axis >= rank {
                return Err(Error::InvalidArgument(format!(
                    "permute: axis {axis} is out of range for rank {rank}",
                )));
            }
            if seen[axis] {
                return Err(Error::InvalidArgument(format!(
                    "permute: axis {axis} appears more than once in {axes:?}",
                )));
            }
            seen[axis] = true;
        }

        let cur_dims = self.shape().dims();
        let cur_strides_slice = cur_strides.as_slice();
        let new_dims: Vec<usize> = axes.iter().map(|&a| cur_dims[a]).collect();
        let new_strides_vec: Vec<isize> = axes.iter().map(|&a| cur_strides_slice[a]).collect();

        let (source, byte_offset) = self.flatten_view();
        Ok(Self::view(
            source,
            byte_offset,
            Shape::from(new_dims),
            Strides::from(new_strides_vec),
        ))
    }

    /// Build a sliced view of this tensor.
    ///
    /// `ranges` provides one half-open `start..end` per axis; the
    /// result's shape is `[end - start, ...]` per axis and the strides
    /// are unchanged from the source. The view's `byte_offset` shifts
    /// by `Σ(start_i × stride_i × dtype_bytes)`.
    ///
    /// Zero-copy. Slicing along the leading axis preserves contiguity;
    /// slicing along an inner axis produces a non-contiguous view
    /// readable via [`Tensor::cpu_iter`].
    pub fn slice(&self, ranges: &[Range<usize>]) -> Result<Self> {
        if self.inner.device.is_none() {
            return Err(Error::InvalidArgument(
                "slice: cannot apply a view to a placeholder (no device)".into(),
            ));
        }
        let cur_strides = match self.layout() {
            Layout::Dense { strides } => strides,
            Layout::Quantized(_) => {
                return Err(Error::InvalidArgument(
                    "slice: quantized tensors are not supported".into(),
                ));
            }
        };

        let rank = self.rank();
        if ranges.len() != rank {
            return Err(Error::InvalidArgument(format!(
                "slice: expected {rank} ranges (one per axis), got {}",
                ranges.len(),
            )));
        }

        let cur_dims = self.shape().dims();
        let cur_strides_slice = cur_strides.as_slice();
        let dtype_bytes = self.inner.dtype.size_bytes();
        let elem_size_signed = isize::try_from(dtype_bytes).expect("dtype size fits in isize");

        let mut new_dims = Vec::with_capacity(rank);
        let mut byte_delta: isize = 0;
        for (axis, range) in ranges.iter().enumerate() {
            let dim = cur_dims[axis];
            if range.start > range.end {
                return Err(Error::InvalidArgument(format!(
                    "slice: axis {axis} range {}..{} has start > end",
                    range.start, range.end,
                )));
            }
            if range.end > dim {
                return Err(Error::InvalidArgument(format!(
                    "slice: axis {axis} range {}..{} exceeds dimension {dim}",
                    range.start, range.end,
                )));
            }
            new_dims.push(range.end - range.start);

            let start_signed = isize::try_from(range.start).expect("slice start fits in isize");
            let step = start_signed
                .checked_mul(cur_strides_slice[axis])
                .and_then(|s| s.checked_mul(elem_size_signed))
                .ok_or_else(|| {
                    Error::InvalidArgument(format!(
                        "slice: byte-offset arithmetic overflowed isize for axis {axis}",
                    ))
                })?;
            byte_delta = byte_delta.checked_add(step).ok_or_else(|| {
                Error::InvalidArgument("slice: cumulative byte offset overflowed isize".into())
            })?;
        }

        let (source, source_offset) = self.flatten_view();
        let source_offset_signed = isize::try_from(source_offset)
            .expect("source byte offset fits in isize (Metal allocations bounded)");
        let combined = source_offset_signed
            .checked_add(byte_delta)
            .ok_or_else(|| {
                Error::InvalidArgument("slice: combined byte offset overflowed isize".into())
            })?;
        let new_byte_offset = usize::try_from(combined).map_err(|_| {
            Error::InvalidArgument(
                "slice: resulting byte offset is negative (would underrun the backing leaf)".into(),
            )
        })?;

        let new_strides = Strides::from(cur_strides_slice.to_vec());
        Ok(Self::view(
            source,
            new_byte_offset,
            Shape::from(new_dims),
            new_strides,
        ))
    }

    /// Build a broadcast view of this tensor with the target shape.
    ///
    /// Follows MLX (and `NumPy`) right-aligned broadcasting:
    /// - the source shape is right-aligned against `target`;
    /// - any source axis of size 1 expands to the target's size at
    ///   that axis, with stride 0 (the underlying element is reread,
    ///   not copied);
    /// - any leading axes that `target` has but the source doesn't
    ///   are prepended with stride 0;
    /// - any non-1 source axis that doesn't match the target's size
    ///   at the aligned position is an error.
    ///
    /// Zero-copy. The result is non-contiguous whenever any expansion
    /// is involved, so reads go through [`Tensor::cpu_iter`].
    pub fn broadcast_to(&self, target: impl Into<Shape>) -> Result<Self> {
        if self.inner.device.is_none() {
            return Err(Error::InvalidArgument(
                "broadcast_to: cannot apply a view to a placeholder (no device)".into(),
            ));
        }
        let cur_strides = match self.layout() {
            Layout::Dense { strides } => strides,
            Layout::Quantized(_) => {
                return Err(Error::InvalidArgument(
                    "broadcast_to: quantized tensors are not supported".into(),
                ));
            }
        };

        let target = target.into();
        let src_dims = self.shape().dims();
        let src_strides_slice = cur_strides.as_slice();
        let target_dims = target.dims();

        if src_dims.len() > target_dims.len() {
            return Err(Error::InvalidArgument(format!(
                "broadcast_to: source rank {} exceeds target rank {}; \
                 broadcasting can add leading axes but not drop them",
                src_dims.len(),
                target_dims.len(),
            )));
        }

        let pad = target_dims.len() - src_dims.len();
        let mut new_strides_vec: Vec<isize> = Vec::with_capacity(target_dims.len());
        for (axis, &target_dim) in target_dims.iter().enumerate() {
            if axis < pad {
                new_strides_vec.push(0);
            } else {
                let src_axis = axis - pad;
                let src_dim = src_dims[src_axis];
                let src_stride = src_strides_slice[src_axis];
                if src_dim == target_dim {
                    new_strides_vec.push(src_stride);
                } else if src_dim == 1 {
                    new_strides_vec.push(0);
                } else {
                    return Err(Error::InvalidArgument(format!(
                        "broadcast_to: source axis {src_axis} of size {src_dim} cannot \
                         broadcast to target axis {axis} of size {target_dim}; \
                         only size-1 source axes can expand",
                    )));
                }
            }
        }

        let (source, byte_offset) = self.flatten_view();
        Ok(Self::view(
            source,
            byte_offset,
            target,
            Strides::from(new_strides_vec),
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

    /// Profiling-only eval: like [`Tensor::eval`], but times each
    /// dispatch's GPU execution via stage-boundary counter sampling and
    /// records per-[`OpKind`] GPU totals into [`crate::profile`]. One
    /// sampled encoder per dispatch perturbs the encode regime — read the
    /// GPU breakdown alongside, not instead of, the production eval spans.
    /// Errors on devices without stage-boundary counter sampling.
    pub fn profiled_eval(&self) -> Result<()> {
        crate::eval::profiled_eval(self)
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
    /// (always), for op tensors after [`Tensor::eval`], and for views
    /// whose source chain terminates at a materialized leaf.
    #[must_use]
    pub fn is_materialized(&self) -> bool {
        self.resolve_leaf().is_some()
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

    /// Host-readable view of the tensor's logical contiguous byte
    /// window, or `None` if it isn't observable as a contiguous slice.
    ///
    /// Returns `Some` iff every condition below holds:
    /// - the backing leaf is materialized (host-constructed leaf, or
    ///   an op output after [`Tensor::eval`]);
    /// - the layout is [`Layout::Dense`] with contiguous (row-major)
    ///   strides for the tensor's current shape;
    /// - the leaf storage is dense (quantized leaves carry three
    ///   byte-erased buffers — packed weights, scales, biases — that
    ///   don't collapse into a single contiguous `&[u8]` view; the
    ///   runtime reads them by-component at dispatch time via the
    ///   `LeafStorage::Quantized` variant).
    ///
    /// For non-contiguous views (transpose, broadcast, sliced along
    /// an inner dim) this returns `None`; use [`Tensor::cpu_iter`] for
    /// element-wise access regardless of stride pattern, or
    /// [`Tensor::copy`] to materialise the logical layout into a fresh
    /// contiguous tensor first.
    ///
    /// Safe to call: a populated `cache` on the backing leaf carries
    /// the invariant that no GPU dispatch is concurrently writing the
    /// buffer. Host-constructed leaves uphold this trivially; op
    /// outputs uphold it because [`Tensor::eval`] populates the cache
    /// only after `waitUntilCompleted`.
    #[must_use]
    pub fn cpu_bytes(&self) -> Option<&[u8]> {
        if !self.inner.layout.is_dense_contiguous(&self.inner.shape) {
            return None;
        }
        let (storage, byte_offset) = self.resolve_leaf()?;
        let buffer = match storage {
            LeafStorage::Dense(buffer) => buffer,
            LeafStorage::Quantized { .. } => return None,
        };
        // SAFETY: the cache invariant on the backing leaf rules out
        // concurrent GPU writes (see method docs).
        let bytes = unsafe { buffer.as_slice() };
        let elem_count = self.inner.shape.elem_count();
        let byte_count = elem_count.checked_mul(self.inner.dtype.size_bytes())?;
        let end = byte_offset.checked_add(byte_count)?;
        bytes.get(byte_offset..end)
    }

    /// Whether this tensor is a view onto another tensor's storage.
    /// Views share their backing leaf with their `source`; they never
    /// allocate their own buffer.
    #[must_use]
    pub fn is_view(&self) -> bool {
        self.inner.view.is_some()
    }

    /// Walk the view chain to the backing leaf, accumulating
    /// byte-offset. Returns `(leaf_storage, total_byte_offset)`. None
    /// if any link in the chain is unmaterialized (placeholder or an
    /// op that hasn't been evaluated yet).
    pub(crate) fn resolve_leaf(&self) -> Option<(&LeafStorage, usize)> {
        let mut current = &self.inner;
        let mut offset: usize = 0;
        loop {
            if let Some(storage) = current.cache.get() {
                return Some((storage, offset));
            }
            let view = current.view.as_ref()?;
            offset = offset.checked_add(view.byte_offset)?;
            current = &view.source.inner;
        }
    }

    /// Element-wise host iterator over the tensor's logical values.
    ///
    /// Works for *any* stride pattern (contiguous, transposed,
    /// broadcast, sliced). The iterator walks the logical shape in
    /// row-major order, computing each element's byte offset from
    /// the view's strides and reading the typed scalar from the
    /// backing leaf via `ptr::read_unaligned`.
    ///
    /// Returns `Some` iff the backing leaf is materialized, the
    /// tensor is dense (not quantized), and the dtype matches `T`.
    /// Test-grade performance only — production kernels do the same
    /// indexing on the GPU.
    #[must_use]
    pub fn cpu_iter<T: Scalar>(&self) -> Option<CpuIter<'_, T>> {
        if self.dtype() != T::DTYPE {
            return None;
        }
        let strides = match self.layout() {
            Layout::Dense { strides } => strides.as_slice(),
            Layout::Quantized(_) => return None,
        };
        let (storage, byte_offset) = self.resolve_leaf()?;
        let buffer = match storage {
            LeafStorage::Dense(buffer) => buffer,
            LeafStorage::Quantized { .. } => return None,
        };
        // SAFETY: the cache invariant on the backing leaf rules out
        // concurrent GPU writes; same as cpu_bytes.
        let bytes = unsafe { buffer.as_slice() };
        let dims = self.inner.shape.dims();
        let remaining = self.inner.shape.elem_count();
        Some(CpuIter {
            bytes,
            base_offset: byte_offset,
            shape: dims,
            strides,
            elem_size: T::DTYPE.size_bytes(),
            indices: vec![0; dims.len()],
            remaining,
            _t: PhantomData,
        })
    }

    /// Collect the tensor's logical values into a fresh `Vec<T>` in
    /// row-major order. Thin wrapper over [`Tensor::cpu_iter`] for
    /// tests where the vec is the assertion target. Returns `None`
    /// under the same conditions as `cpu_iter`.
    #[must_use]
    pub fn cpu_to_vec<T: Scalar>(&self) -> Option<Vec<T>> {
        Some(self.cpu_iter::<T>()?.collect())
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

/// Compute the byte count of a dense tensor with `shape` and `dtype`,
/// rejecting overflow with [`Error::InvalidArgument`] rather than
/// silently wrapping. `Shape::elem_count` already panics on its own
/// product overflowing — this catches the secondary overflow when
/// multiplying by `dtype.size_bytes()`.
pub(crate) fn checked_byte_count(shape: &Shape, dtype: DType) -> Result<usize> {
    shape
        .elem_count()
        .checked_mul(dtype.size_bytes())
        .ok_or_else(|| {
            Error::InvalidArgument(format!(
                "byte count overflows usize: shape {shape:?} × {} bytes per {dtype} element",
                dtype.size_bytes(),
            ))
        })
}

fn dense_layout(shape: &Shape) -> Layout {
    Layout::Dense {
        strides: Strides::contiguous(shape),
    }
}

/// Resolve a possibly-negative axis index against a rank. `-1`
/// resolves to `rank - 1`, `-rank` to `0`. Errors if out of range.
fn resolve_axis(rank: usize, axis: i64) -> Result<usize> {
    let rank_i64 = i64::try_from(rank)
        .map_err(|_| Error::InvalidArgument("axis: rank exceeds i64::MAX".into()))?;
    let resolved = if axis < 0 { axis + rank_i64 } else { axis };
    if !(0..rank_i64).contains(&resolved) {
        return Err(Error::InvalidArgument(format!(
            "axis {axis} is out of range for rank {rank}",
        )));
    }
    Ok(usize::try_from(resolved).expect("resolved is in [0, rank)"))
}

/// Build a `1/n` scalar in the requested dtype, in host f64 then
/// rounded down to the target dtype's precision.
fn recip_scalar(n: usize, dtype: DType) -> Result<[u8; 4]> {
    if n == 0 {
        return Err(Error::InvalidArgument("recip: divisor is zero".into()));
    }
    // f64 → target-dtype host bytes. Output is always padded to 4
    // bytes; the caller knows how many to write based on dtype.
    #[allow(clippy::cast_precision_loss)]
    let inv = 1.0_f64 / (n as f64);
    let mut buf = [0u8; 4];
    match dtype {
        DType::F32 => {
            #[allow(clippy::cast_possible_truncation)]
            let v = inv as f32;
            buf.copy_from_slice(&v.to_le_bytes());
        }
        DType::F16 => {
            #[allow(clippy::cast_possible_truncation)]
            let v = half::f16::from_f32(inv as f32);
            buf[..2].copy_from_slice(&v.to_le_bytes());
        }
        DType::BF16 => {
            #[allow(clippy::cast_possible_truncation)]
            let v = half::bf16::from_f32(inv as f32);
            buf[..2].copy_from_slice(&v.to_le_bytes());
        }
        DType::I32 | DType::U32 => {
            return Err(Error::InvalidArgument(format!(
                "recip: integer dtype {dtype:?} is not supported for mean()",
            )));
        }
    }
    Ok(buf)
}

/// Allocate a host-side scalar tensor of shape `[1]`, dtype `dtype`,
/// containing the value packed in `host_bytes` (interpreted per-dtype:
/// 4 bytes for f32, 2 bytes (the low half) for f16/bf16).
fn host_scalar_tensor(device: &Device, dtype: DType, host_bytes: [u8; 4]) -> Result<Tensor> {
    let shape = Shape::from([1usize]);
    let byte_count = checked_byte_count(&shape, dtype)?;
    let mut buffer = device.kernels().alloc_buffer::<u8>(byte_count)?;
    // SAFETY: just allocated, no GPU dispatch has referenced it.
    unsafe { buffer.as_mut_slice() }.copy_from_slice(&host_bytes[..byte_count]);
    let layout = dense_layout(&shape);
    Ok(Tensor::host_leaf(device, shape, dtype, layout, buffer))
}

/// NumPy-style right-aligned broadcast of two shapes.
///
/// Each output axis is the max of the corresponding (right-aligned)
/// input axes, with the constraint that mismatched axes are only legal
/// when one of them is 1. Inputs of different rank are aligned on the
/// right; the shorter shape is implicitly padded with size-1 leading
/// axes.
/// Validate a matmul's shape contract and return `(M, N, K, lead)`
/// where `lead = lhs_dims.len() - 2` is the count of leading
/// (batch) dims. Pulled out of [`Tensor::matmul`] to keep that
/// constructor under the `too_many_lines` lint and so future
/// matmul-shaped ops can reuse the same
/// validation.
fn matmul_shapes(lhs_dims: &[usize], rhs_dims: &[usize]) -> Result<(usize, usize, usize, usize)> {
    if lhs_dims.len() < 2 || rhs_dims.len() < 2 {
        return Err(Error::InvalidArgument(format!(
            "matmul: both inputs must have rank ≥ 2 (got {lhs_dims:?} × {rhs_dims:?})",
        )));
    }
    if lhs_dims.len() != rhs_dims.len() {
        return Err(Error::InvalidArgument(format!(
            "matmul: leading-rank mismatch (got {lhs_dims:?} × {rhs_dims:?}); \
             broadcast + copy before matmul if you need GQA-style fan-out",
        )));
    }
    let lead = lhs_dims.len() - 2;
    if lhs_dims[..lead] != rhs_dims[..lead] {
        return Err(Error::InvalidArgument(format!(
            "matmul: leading dims must match (got {lhs_dims:?} × {rhs_dims:?})",
        )));
    }
    let m = lhs_dims[lead];
    let k_lhs = lhs_dims[lead + 1];
    let k_rhs = rhs_dims[lead];
    let n = rhs_dims[lead + 1];
    if k_lhs != k_rhs {
        return Err(Error::InvalidArgument(format!(
            "matmul: inner dimension mismatch (LHS K={k_lhs} vs RHS K={k_rhs}); \
             shapes {lhs_dims:?} × {rhs_dims:?}",
        )));
    }
    if m == 0 || n == 0 || k_lhs == 0 {
        return Err(Error::InvalidArgument(format!(
            "matmul: zero-sized matmul dimensions (M={m}, N={n}, K={k_lhs})",
        )));
    }
    Ok((m, n, k_lhs, lead))
}

fn broadcast_shape(a: &Shape, b: &Shape) -> Result<Shape> {
    let a_dims = a.dims();
    let b_dims = b.dims();
    let out_rank = a_dims.len().max(b_dims.len());
    let mut out = vec![0usize; out_rank];
    for (i, slot) in out.iter_mut().enumerate() {
        let ai = a_dims.len().checked_sub(out_rank - i);
        let bi = b_dims.len().checked_sub(out_rank - i);
        let av = ai.map_or(1, |idx| a_dims[idx]);
        let bv = bi.map_or(1, |idx| b_dims[idx]);
        *slot = match (av, bv) {
            (x, y) if x == y => x,
            (1, y) => y,
            (x, 1) => x,
            (x, y) => {
                return Err(Error::InvalidArgument(format!(
                    "binary broadcast: incompatible axis sizes {x} and {y} at output axis {i} \
                     (shapes {a_dims:?} vs {b_dims:?})",
                )));
            }
        };
    }
    Ok(Shape::from(out))
}

/// Element-wise host iterator returned by [`Tensor::cpu_iter`].
///
/// Walks the logical shape in row-major order, reading each typed
/// scalar from the backing leaf via stride arithmetic. Yields `T` by
/// value (every [`Scalar`] is `Copy`).
///
/// The iterator borrows the backing leaf bytes; the leaf stays alive
/// for the iterator's lifetime via the source tensor's `Arc`.
pub struct CpuIter<'a, T: Scalar> {
    bytes: &'a [u8],
    base_offset: usize,
    shape: &'a [usize],
    strides: &'a [isize],
    elem_size: usize,
    indices: Vec<usize>,
    remaining: usize,
    _t: PhantomData<T>,
}

impl<T: Scalar> Iterator for CpuIter<'_, T> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        if self.remaining == 0 {
            return None;
        }

        let mut signed_off: isize = isize::try_from(self.base_offset)
            .expect("base_offset fits in isize (Metal allocations are well under isize::MAX)");
        for (idx, stride) in self.indices.iter().zip(self.strides.iter()) {
            let idx_signed = isize::try_from(*idx).expect("index fits in isize");
            let elem_signed = isize::try_from(self.elem_size).expect("dtype size fits in isize");
            let step = idx_signed
                .checked_mul(*stride)
                .and_then(|s| s.checked_mul(elem_signed))
                .expect("stride arithmetic overflowed isize");
            signed_off = signed_off
                .checked_add(step)
                .expect("byte offset overflowed isize");
        }
        let byte_off = usize::try_from(signed_off).expect("computed byte offset is non-negative");
        debug_assert!(byte_off + self.elem_size <= self.bytes.len());

        // SAFETY: byte_off + elem_size ≤ bytes.len() (asserted in debug;
        // the public view-building APIs guarantee this in release by
        // bounds-checking shape × strides at construction). Metal
        // shared-storage allocations are page-aligned, but we still use
        // read_unaligned so mid-buffer offsets are sound for any T.
        let val = unsafe {
            self.bytes
                .as_ptr()
                .add(byte_off)
                .cast::<T>()
                .read_unaligned()
        };

        // Advance odometer: increment last axis, carry up. Rank-0
        // tensors have an empty `indices` and bail out via the
        // `remaining` counter alone.
        for axis in (0..self.indices.len()).rev() {
            self.indices[axis] += 1;
            if self.indices[axis] < self.shape[axis] {
                break;
            }
            self.indices[axis] = 0;
        }
        self.remaining -= 1;
        Some(val)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<T: Scalar> ExactSizeIterator for CpuIter<'_, T> {
    fn len(&self) -> usize {
        self.remaining
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
    fn from_bytes_rejects_byte_count_overflow() {
        // shape.elem_count fits in usize, but elem_count * size_bytes
        // overflows. The checked byte_count helper should reject
        // before allocation is attempted, with no slice-length read.
        let device = Device::shared().expect("system default device");
        let huge = usize::MAX / 2; // F32 size_bytes = 4 → huge * 4 overflows.
        let err = Tensor::from_bytes(&device, &[], [huge], DType::F32).unwrap_err();
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
    fn eval_through_view_materializes_source_op() {
        // A view of an unevaluated op should, when eval()'d, dispatch
        // the underlying op so the view becomes materialised.
        let device = Device::shared().expect("system default device");
        let data: Vec<f32> = (0..12)
            .map(|i| f32::from(i16::try_from(i).unwrap()))
            .collect();
        let src = Tensor::from_slice(&device, &data, [3, 4]).expect("from_slice");
        let copied = src.copy().expect("copy"); // unevaluated op
        assert!(!copied.is_materialized());
        let view = copied.reshape([2, 6]).expect("reshape"); // view of op
        assert!(!view.is_materialized());

        view.eval().expect("eval view");

        assert!(copied.is_materialized());
        assert!(view.is_materialized());
        // View shares the source's bytes; reshape is contiguous so
        // cpu_slice works directly.
        assert_eq!(view.cpu_slice::<f32>().unwrap(), data.as_slice());
    }

    #[test]
    fn copy_of_transposed_view_materializes_logical_layout() {
        // A 2x3 row-major source, transposed to 3x2, then copied. The
        // copy should produce a contiguous 3x2 tensor whose values
        // match the transposed view element-for-element.
        let device = Device::shared().expect("system default device");
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
        let src = Tensor::from_slice(&device, &data, [2, 3]).expect("from_slice");
        let transposed = src.permute(&[1, 0]).expect("permute"); // 3x2 view

        let copied = transposed.copy().expect("copy");
        copied.eval().expect("eval");

        // Logical transpose: rows become columns.
        let expected: Vec<f32> = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        assert_eq!(copied.shape().dims(), &[3, 2]);
        assert_eq!(copied.cpu_slice::<f32>().unwrap(), expected.as_slice());
    }

    #[test]
    fn copy_of_broadcast_view_materializes_repeated_rows() {
        let device = Device::shared().expect("system default device");
        let data: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0]; // 1x4
        let src = Tensor::from_slice(&device, &data, [1, 4]).expect("from_slice");
        let broadcasted = src.broadcast_to([3, 4]).expect("broadcast_to");

        let copied = broadcasted.copy().expect("copy");
        copied.eval().expect("eval");

        let expected: Vec<f32> = vec![
            10.0, 20.0, 30.0, 40.0, 10.0, 20.0, 30.0, 40.0, 10.0, 20.0, 30.0, 40.0,
        ];
        assert_eq!(copied.cpu_slice::<f32>().unwrap(), expected.as_slice());
    }

    #[test]
    fn copy_of_sliced_view_materializes_window() {
        // Slice along axis 0 (preserves contiguity but introduces a
        // byte offset) and along axis 1 (introduces non-unit stride).
        let device = Device::shared().expect("system default device");
        let data: Vec<i32> = (0..24).collect(); // 4x6
        let src = Tensor::from_slice(&device, &data, [4, 6]).expect("from_slice");
        let window = src.slice(&[1..3, 0..6]).expect("slice axis 0"); // rows 1..3
        let inner = window.slice(&[0..2, 2..5]).expect("slice axis 1"); // cols 2..5

        let copied = inner.copy().expect("copy");
        copied.eval().expect("eval");

        // Rows 1,2 of `data`, columns 2..5:
        let expected: Vec<i32> = vec![8, 9, 10, 14, 15, 16];
        assert_eq!(copied.shape().dims(), &[2, 3]);
        assert_eq!(copied.cpu_slice::<i32>().unwrap(), expected.as_slice());
    }

    #[test]
    fn copy_of_view_over_unevaluated_op_dispatches_both() {
        // The view's source is another (unevaluated) Copy op. eval()
        // on the strided copy should walk: strided-copy → view →
        // source copy → leaf, dispatching the source copy in the same
        // batch and then the strided copy.
        let device = Device::shared().expect("system default device");
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let src = Tensor::from_slice(&device, &data, [2, 3]).expect("from_slice");
        let staged = src.copy().expect("first copy"); // unevaluated op
        let transposed = staged.permute(&[1, 0]).expect("permute"); // view of op
        let final_copy = transposed.copy().expect("strided copy");

        final_copy.eval().expect("eval");

        assert!(staged.is_materialized());
        assert!(final_copy.is_materialized());
        let expected: Vec<f32> = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        assert_eq!(final_copy.cpu_slice::<f32>().unwrap(), expected.as_slice());
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

    /// Test-only constructor for a view tensor pointing at `source`
    /// with an explicit `byte_offset`, `shape`, and (dense) `strides`,
    /// bypassing [`Tensor::flatten_view`]. The public view-building
    /// APIs (`reshape`, `permute`, `slice`, `broadcast_to`) flatten
    /// view chains on the way in; this helper exists so the chain-
    /// walking tests can build multi-hop chains directly.
    fn manual_view(source: &Tensor, byte_offset: usize, shape: Shape, strides: Strides) -> Tensor {
        let layout = Layout::Dense { strides };
        Tensor {
            inner: Arc::new(TensorInner {
                shape,
                dtype: source.dtype(),
                layout,
                device: source.device().cloned(),
                op: None,
                view: Some(ViewSource {
                    source: source.clone(),
                    byte_offset,
                }),
                cache: OnceLock::new(),
            }),
        }
    }

    #[test]
    fn view_chain_walks_to_materialized_leaf() {
        let device = Device::shared().expect("system default device");
        let data: Vec<f32> = (0..16)
            .map(|i| f32::from(i16::try_from(i).unwrap()))
            .collect();
        let leaf = Tensor::from_slice(&device, &data, [16]).expect("from_slice");

        // Full-window view: byte_offset = 0, same shape, contiguous strides.
        let v = manual_view(
            &leaf,
            0,
            Shape::from([16]),
            Strides::contiguous(&Shape::from([16])),
        );

        assert!(v.is_view());
        assert!(v.is_materialized());
        let bytes = v.cpu_bytes().expect("view cpu_bytes");
        assert_eq!(bytes.len(), 16 * 4);
        let typed = v.cpu_slice::<f32>().expect("view cpu_slice");
        assert_eq!(typed, data.as_slice());
    }

    #[test]
    fn view_with_byte_offset_reads_logical_window() {
        let device = Device::shared().expect("system default device");
        let data: Vec<f32> = (0..16)
            .map(|i| f32::from(i16::try_from(i).unwrap()))
            .collect();
        let leaf = Tensor::from_slice(&device, &data, [16]).expect("from_slice");

        // View the second half: skip 8 f32 (32 bytes), shape [8].
        let v = manual_view(
            &leaf,
            32,
            Shape::from([8]),
            Strides::contiguous(&Shape::from([8])),
        );

        let typed = v.cpu_slice::<f32>().expect("view cpu_slice");
        assert_eq!(typed, &data[8..]);
    }

    #[test]
    fn non_contiguous_view_returns_none_from_cpu_bytes() {
        let device = Device::shared().expect("system default device");
        let data: Vec<f32> = (0..6)
            .map(|i| f32::from(i16::try_from(i).unwrap()))
            .collect();
        let leaf = Tensor::from_slice(&device, &data, [2, 3]).expect("from_slice");

        // Transposed view: shape [3, 2], strides [1, 3] (non-contiguous).
        let v = manual_view(&leaf, 0, Shape::from([3, 2]), Strides::from([1isize, 3]));

        assert!(v.is_materialized());
        assert!(v.cpu_bytes().is_none());
        assert!(v.cpu_slice::<f32>().is_none());
    }

    #[test]
    fn view_of_view_accumulates_byte_offset() {
        let device = Device::shared().expect("system default device");
        let data: Vec<f32> = (0..16)
            .map(|i| f32::from(i16::try_from(i).unwrap()))
            .collect();
        let leaf = Tensor::from_slice(&device, &data, [16]).expect("from_slice");

        let outer = manual_view(
            &leaf,
            16,
            Shape::from([12]),
            Strides::contiguous(&Shape::from([12])),
        );
        // Take a further [8] sub-window starting 8 bytes in.
        let inner = manual_view(
            &outer,
            8,
            Shape::from([6]),
            Strides::contiguous(&Shape::from([6])),
        );

        // Total byte_offset = 16 + 8 = 24 → f32 index 6, length 6.
        let typed = inner.cpu_slice::<f32>().expect("view cpu_slice");
        assert_eq!(typed, &data[6..12]);
    }

    #[test]
    fn cpu_iter_matches_cpu_slice_for_contiguous_leaf() {
        let device = Device::shared().expect("system default device");
        let data: Vec<f32> = (0..24)
            .map(|i| f32::from(i16::try_from(i).unwrap()))
            .collect();
        let leaf = Tensor::from_slice(&device, &data, [2, 3, 4]).expect("from_slice");

        let iter_vec: Vec<f32> = leaf.cpu_iter::<f32>().expect("cpu_iter").collect();
        let slice = leaf.cpu_slice::<f32>().expect("cpu_slice");
        assert_eq!(iter_vec, slice);
        assert_eq!(iter_vec, data);
    }

    #[test]
    fn cpu_to_vec_round_trips_contiguous() {
        let device = Device::shared().expect("system default device");
        let data: Vec<i32> = (-8..8).collect();
        let leaf = Tensor::from_slice(&device, &data, [4, 4]).expect("from_slice");
        let v = leaf.cpu_to_vec::<i32>().expect("cpu_to_vec");
        assert_eq!(v, data);
    }

    #[test]
    fn cpu_iter_walks_transposed_view_in_logical_order() {
        // Build a [2, 3] f32 leaf and view it as [3, 2] with swapped
        // strides — the manual "transpose." cpu_iter should yield the
        // elements in row-major order of the transposed shape.
        let device = Device::shared().expect("system default device");
        let data: Vec<f32> = (0..6)
            .map(|i| f32::from(i16::try_from(i).unwrap()))
            .collect();
        let leaf = Tensor::from_slice(&device, &data, [2, 3]).expect("from_slice");

        // Original strides for [2,3] are [3, 1]; transpose swaps to [1, 3].
        let v = manual_view(&leaf, 0, Shape::from([3, 2]), Strides::from([1isize, 3]));

        let collected: Vec<f32> = v.cpu_iter::<f32>().expect("cpu_iter").collect();
        // Original layout: [[0,1,2],[3,4,5]]; transposed: [[0,3],[1,4],[2,5]].
        assert_eq!(collected, vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
    }

    #[test]
    fn cpu_iter_respects_byte_offset_window() {
        let device = Device::shared().expect("system default device");
        let data: Vec<f32> = (0..12)
            .map(|i| f32::from(i16::try_from(i).unwrap()))
            .collect();
        let leaf = Tensor::from_slice(&device, &data, [12]).expect("from_slice");

        // Skip the first 4 f32 (16 bytes), shape [8] contiguous.
        let v = manual_view(
            &leaf,
            16,
            Shape::from([8]),
            Strides::contiguous(&Shape::from([8])),
        );
        let collected: Vec<f32> = v.cpu_iter::<f32>().expect("cpu_iter").collect();
        assert_eq!(collected, data[4..].to_vec());
    }

    #[test]
    fn cpu_iter_handles_broadcast_zero_strides() {
        // Three-element vector broadcast to [2, 3] via stride [0, 1].
        let device = Device::shared().expect("system default device");
        let data: Vec<f32> = vec![10.0, 20.0, 30.0];
        let leaf = Tensor::from_slice(&device, &data, [3]).expect("from_slice");

        let v = manual_view(&leaf, 0, Shape::from([2, 3]), Strides::from([0isize, 1]));
        let collected: Vec<f32> = v.cpu_iter::<f32>().expect("cpu_iter").collect();
        assert_eq!(collected, vec![10.0, 20.0, 30.0, 10.0, 20.0, 30.0]);
    }

    #[test]
    fn cpu_iter_returns_none_for_dtype_mismatch() {
        let device = Device::shared().expect("system default device");
        let leaf = Tensor::zeros(&device, [4], DType::F32).expect("zeros");
        assert!(leaf.cpu_iter::<i32>().is_none());
        assert!(leaf.cpu_iter::<f32>().is_some());
    }

    #[test]
    fn cpu_iter_returns_none_for_unmaterialized_op() {
        let device = Device::shared().expect("system default device");
        let leaf = Tensor::zeros(&device, [4], DType::F32).expect("zeros");
        let op = leaf.copy().expect("copy");
        assert!(op.cpu_iter::<f32>().is_none());

        op.eval().expect("eval");
        assert!(op.cpu_iter::<f32>().is_some());
    }

    #[test]
    fn cpu_iter_size_hint_and_len_match_elem_count() {
        let device = Device::shared().expect("system default device");
        let leaf = Tensor::zeros(&device, [2, 3, 5], DType::F32).expect("zeros");
        let iter = leaf.cpu_iter::<f32>().expect("cpu_iter");
        assert_eq!(iter.len(), 30);
        assert_eq!(iter.size_hint(), (30, Some(30)));
    }

    #[test]
    fn reshape_identity_preserves_values() {
        let device = Device::shared().expect("system default device");
        let data: Vec<f32> = (0..24)
            .map(|i| f32::from(i16::try_from(i).unwrap()))
            .collect();
        let leaf = Tensor::from_slice(&device, &data, [2, 3, 4]).expect("from_slice");
        let v = leaf.reshape([2, 3, 4]).expect("reshape");

        assert!(v.is_view());
        assert_eq!(v.shape().dims(), &[2, 3, 4]);
        assert_eq!(v.cpu_to_vec::<f32>().unwrap(), data);
    }

    #[test]
    fn reshape_flatten_then_unflatten_round_trip() {
        let device = Device::shared().expect("system default device");
        let data: Vec<f32> = (0..24)
            .map(|i| f32::from(i16::try_from(i).unwrap()))
            .collect();
        let leaf = Tensor::from_slice(&device, &data, [2, 3, 4]).expect("from_slice");
        let flat = leaf.reshape([24]).expect("flatten");
        assert_eq!(flat.shape().dims(), &[24]);
        assert_eq!(flat.cpu_to_vec::<f32>().unwrap(), data);

        let cube = flat.reshape([4, 6]).expect("unflatten");
        assert_eq!(cube.shape().dims(), &[4, 6]);
        assert_eq!(cube.cpu_to_vec::<f32>().unwrap(), data);
        assert!(cube.layout().is_dense_contiguous(cube.shape()));
    }

    #[test]
    fn reshape_view_is_contiguous_via_cpu_bytes() {
        // Contiguous views must satisfy the cpu_bytes contract: same
        // logical bytes as cpu_iter would emit.
        let device = Device::shared().expect("system default device");
        let data: Vec<f32> = (0..12)
            .map(|i| f32::from(i16::try_from(i).unwrap()))
            .collect();
        let leaf = Tensor::from_slice(&device, &data, [12]).expect("from_slice");

        let v = leaf.reshape([3, 4]).expect("reshape");
        let bytes = v
            .cpu_bytes()
            .expect("contiguous view should expose cpu_bytes");
        assert_eq!(bytes.len(), 12 * 4);
        let typed = v.cpu_slice::<f32>().expect("cpu_slice");
        assert_eq!(typed, data.as_slice());
    }

    #[test]
    fn reshape_folds_view_chain() {
        // reshape-of-reshape should point directly at the leaf, not at
        // the intermediate view (one-hop chain instead of two).
        let device = Device::shared().expect("system default device");
        let data: Vec<f32> = (0..24)
            .map(|i| f32::from(i16::try_from(i).unwrap()))
            .collect();
        let leaf = Tensor::from_slice(&device, &data, [24]).expect("from_slice");

        let mid = leaf.reshape([2, 12]).expect("reshape 1");
        let outer = mid.reshape([4, 3, 2]).expect("reshape 2");

        // Inspect the chain: outer's ViewSource should point at the
        // leaf directly, not at `mid`.
        let outer_view = outer.inner.view.as_ref().expect("view");
        assert!(outer_view.source.ptr_eq(&leaf));
        assert_eq!(outer_view.byte_offset, 0);
    }

    #[test]
    fn reshape_rejects_elem_count_mismatch() {
        let device = Device::shared().expect("system default device");
        let leaf = Tensor::zeros(&device, [2, 3], DType::F32).expect("zeros");
        let err = leaf.reshape([2, 4]).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)));
    }

    #[test]
    fn reshape_rejects_non_contiguous_input() {
        let device = Device::shared().expect("system default device");
        let leaf = Tensor::zeros(&device, [2, 3], DType::F32).expect("zeros");
        // Hand-construct a transposed view (non-contiguous) and try to
        // reshape it — should error rather than silently producing
        // wrong data.
        let transposed = manual_view(&leaf, 0, Shape::from([3, 2]), Strides::from([1isize, 3]));
        let err = transposed.reshape([6]).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)));
    }

    #[test]
    fn reshape_rejects_quantized() {
        let t = Tensor::quantized_placeholder([4096, 4096], Quantization::Q4_GS64);
        let err = t.reshape([4096 * 4096]).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)));
    }

    #[test]
    fn reshape_rejects_placeholder() {
        let p = Tensor::placeholder::<f32>([2, 3]);
        let err = p.reshape([6]).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)));
    }

    #[test]
    fn reshape_of_evaluated_op_is_readable() {
        let device = Device::shared().expect("system default device");
        let data: Vec<f32> = (0..16)
            .map(|i| f32::from(i16::try_from(i).unwrap()))
            .collect();
        let src = Tensor::from_slice(&device, &data, [16]).expect("from_slice");
        let op = src.copy().expect("copy");

        // Reshape an unevaluated op output. The view is constructed
        // immediately; reading it requires the op to be evaluated.
        let v = op.reshape([4, 4]).expect("reshape");
        assert!(!v.is_materialized());
        op.eval().expect("eval");
        assert!(v.is_materialized());
        assert_eq!(v.cpu_to_vec::<f32>().unwrap(), data);
    }

    #[test]
    fn permute_identity_yields_same_values() {
        let device = Device::shared().expect("system default device");
        let data: Vec<f32> = (0..24)
            .map(|i| f32::from(i16::try_from(i).unwrap()))
            .collect();
        let leaf = Tensor::from_slice(&device, &data, [2, 3, 4]).expect("from_slice");

        let same = leaf.permute(&[0, 1, 2]).expect("permute identity");
        assert_eq!(same.shape().dims(), &[2, 3, 4]);
        assert_eq!(same.cpu_to_vec::<f32>().unwrap(), data);
    }

    #[test]
    fn permute_2d_transpose_matches_manual() {
        let device = Device::shared().expect("system default device");
        // [[0,1,2],[3,4,5]]
        let data: Vec<f32> = (0..6)
            .map(|i| f32::from(i16::try_from(i).unwrap()))
            .collect();
        let leaf = Tensor::from_slice(&device, &data, [2, 3]).expect("from_slice");

        let t = leaf.permute(&[1, 0]).expect("permute");
        assert_eq!(t.shape().dims(), &[3, 2]);
        // Transposed row-major: [[0,3],[1,4],[2,5]]
        assert_eq!(
            t.cpu_to_vec::<f32>().unwrap(),
            vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0],
        );
    }

    #[test]
    fn permute_4d_attention_pattern() {
        // [B=2, T=3, H=2, D_h=2] → [B, H, T, D_h] via axes [0, 2, 1, 3]
        // is the Q/K/V reshape every attention block does.
        let device = Device::shared().expect("system default device");
        let data: Vec<f32> = (0..24)
            .map(|i| f32::from(i16::try_from(i).unwrap()))
            .collect();
        let leaf = Tensor::from_slice(&device, &data, [2, 3, 2, 2]).expect("from_slice");

        let permuted = leaf.permute(&[0, 2, 1, 3]).expect("permute");
        assert_eq!(permuted.shape().dims(), &[2, 2, 3, 2]);

        // Reference: walk the original (b, t, h, d) layout with the
        // permuted index (b, h, t, d) and read the value.
        let mut expected: Vec<f32> = Vec::with_capacity(24);
        for b in 0..2 {
            for h in 0..2 {
                for t in 0..3 {
                    for d in 0..2 {
                        let src_idx = ((b * 3 + t) * 2 + h) * 2 + d;
                        expected.push(data[src_idx]);
                    }
                }
            }
        }
        assert_eq!(permuted.cpu_to_vec::<f32>().unwrap(), expected);
    }

    #[test]
    fn permute_of_permute_composes() {
        // Swapping back returns the original logical order.
        let device = Device::shared().expect("system default device");
        let data: Vec<f32> = (0..12)
            .map(|i| f32::from(i16::try_from(i).unwrap()))
            .collect();
        let leaf = Tensor::from_slice(&device, &data, [3, 4]).expect("from_slice");

        let once = leaf.permute(&[1, 0]).expect("permute 1");
        let twice = once.permute(&[1, 0]).expect("permute 2");
        assert_eq!(twice.shape().dims(), &[3, 4]);
        assert_eq!(twice.cpu_to_vec::<f32>().unwrap(), data);
    }

    #[test]
    fn permute_folds_through_reshape_chain() {
        // reshape then permute should keep the view one hop from the
        // backing leaf, not stack two view nodes.
        let device = Device::shared().expect("system default device");
        let data: Vec<f32> = (0..12)
            .map(|i| f32::from(i16::try_from(i).unwrap()))
            .collect();
        let leaf = Tensor::from_slice(&device, &data, [12]).expect("from_slice");
        let v = leaf
            .reshape([3, 4])
            .expect("reshape")
            .permute(&[1, 0])
            .expect("permute");
        let view_node = v.inner.view.as_ref().expect("view");
        assert!(view_node.source.ptr_eq(&leaf));
    }

    #[test]
    fn permute_rejects_wrong_length() {
        let device = Device::shared().expect("system default device");
        let leaf = Tensor::zeros(&device, [2, 3], DType::F32).expect("zeros");
        let err = leaf.permute(&[0]).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)));
    }

    #[test]
    fn permute_rejects_out_of_range_axis() {
        let device = Device::shared().expect("system default device");
        let leaf = Tensor::zeros(&device, [2, 3], DType::F32).expect("zeros");
        let err = leaf.permute(&[0, 5]).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)));
    }

    #[test]
    fn permute_rejects_duplicate_axis() {
        let device = Device::shared().expect("system default device");
        let leaf = Tensor::zeros(&device, [2, 3], DType::F32).expect("zeros");
        let err = leaf.permute(&[0, 0]).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)));
    }

    #[test]
    fn permute_rejects_quantized() {
        let t = Tensor::quantized_placeholder([4096, 4096], Quantization::Q4_GS64);
        let err = t.permute(&[1, 0]).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)));
    }

    #[test]
    fn permute_rejects_placeholder() {
        let p = Tensor::placeholder::<f32>([2, 3]);
        let err = p.permute(&[1, 0]).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)));
    }

    #[test]
    fn permuted_view_reports_non_contiguous_cpu_bytes() {
        let device = Device::shared().expect("system default device");
        let leaf = Tensor::zeros(&device, [2, 3], DType::F32).expect("zeros");
        let t = leaf.permute(&[1, 0]).expect("permute");
        // Transposed strides aren't row-major-contiguous for [3,2].
        assert!(t.cpu_bytes().is_none());
        // But cpu_iter still works.
        assert!(t.cpu_iter::<f32>().is_some());
    }

    #[test]
    fn slice_leading_axis_is_contiguous() {
        let device = Device::shared().expect("system default device");
        let data: Vec<f32> = (0..12)
            .map(|i| f32::from(i16::try_from(i).unwrap()))
            .collect();
        let leaf = Tensor::from_slice(&device, &data, [4, 3]).expect("from_slice");

        let v = leaf.slice(&[1..3, 0..3]).expect("slice leading");
        assert_eq!(v.shape().dims(), &[2, 3]);
        // Leading slice [1..3] is contiguous in row-major.
        let bytes = v.cpu_bytes().expect("contiguous via leading slice");
        assert_eq!(bytes.len(), 6 * 4);
        assert_eq!(v.cpu_to_vec::<f32>().unwrap(), data[3..9].to_vec());
    }

    #[test]
    fn slice_inner_axis_is_non_contiguous() {
        let device = Device::shared().expect("system default device");
        let data: Vec<f32> = (0..12)
            .map(|i| f32::from(i16::try_from(i).unwrap()))
            .collect();
        let leaf = Tensor::from_slice(&device, &data, [4, 3]).expect("from_slice");

        // Take columns 1..3 of every row.
        let v = leaf.slice(&[0..4, 1..3]).expect("slice inner");
        assert_eq!(v.shape().dims(), &[4, 2]);
        // Inner-axis slice is non-contiguous: stride is still 3, but
        // shape's row-major contiguous would require stride 2.
        assert!(v.cpu_bytes().is_none());

        let mut expected: Vec<f32> = Vec::with_capacity(8);
        for row in 0..4 {
            for col in 1..3 {
                expected.push(data[row * 3 + col]);
            }
        }
        assert_eq!(v.cpu_iter::<f32>().unwrap().collect::<Vec<_>>(), expected);
    }

    #[test]
    fn slice_chain_accumulates_byte_offset() {
        // Slice twice; the second slice should fold into a single view
        // pointing directly at the leaf with combined offset.
        let device = Device::shared().expect("system default device");
        let data: Vec<f32> = (0..16)
            .map(|i| f32::from(i16::try_from(i).unwrap()))
            .collect();
        let leaf = Tensor::from_slice(&device, &data, [16]).expect("from_slice");

        let a = leaf.slice(std::slice::from_ref(&(4..12))).expect("slice 1");
        let b = a.slice(std::slice::from_ref(&(2..6))).expect("slice 2");
        assert_eq!(b.shape().dims(), &[4]);
        assert_eq!(b.cpu_to_vec::<f32>().unwrap(), data[6..10].to_vec());

        let view_node = b.inner.view.as_ref().expect("view");
        assert!(view_node.source.ptr_eq(&leaf));
        // 4 + 2 = 6 f32s, × 4 bytes = 24.
        assert_eq!(view_node.byte_offset, 24);
    }

    #[test]
    fn slice_of_permute_reads_logical_values() {
        let device = Device::shared().expect("system default device");
        let data: Vec<f32> = (0..12)
            .map(|i| f32::from(i16::try_from(i).unwrap()))
            .collect();
        let leaf = Tensor::from_slice(&device, &data, [3, 4]).expect("from_slice");

        // Transpose to [4, 3], then take the middle two rows.
        let t = leaf.permute(&[1, 0]).expect("permute");
        let v = t.slice(&[1..3, 0..3]).expect("slice");
        assert_eq!(v.shape().dims(), &[2, 3]);

        // Permuted logical layout columns (col 1, col 2 of original):
        // col 1: [1, 5, 9]; col 2: [2, 6, 10].
        let expected = vec![1.0, 5.0, 9.0, 2.0, 6.0, 10.0];
        assert_eq!(v.cpu_iter::<f32>().unwrap().collect::<Vec<_>>(), expected);
    }

    #[test]
    fn slice_empty_range_yields_zero_count() {
        let device = Device::shared().expect("system default device");
        let leaf = Tensor::zeros(&device, [4, 3], DType::F32).expect("zeros");
        let v = leaf.slice(&[2..2, 0..3]).expect("empty slice");
        assert_eq!(v.shape().dims(), &[0, 3]);
        assert_eq!(v.elem_count(), 0);
        assert_eq!(v.cpu_to_vec::<f32>().unwrap(), Vec::<f32>::new());
    }

    #[test]
    fn slice_rejects_wrong_range_count() {
        let device = Device::shared().expect("system default device");
        let leaf = Tensor::zeros(&device, [2, 3], DType::F32).expect("zeros");
        let err = leaf.slice(std::slice::from_ref(&(0..2))).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)));
    }

    #[test]
    fn slice_rejects_out_of_bounds() {
        let device = Device::shared().expect("system default device");
        let leaf = Tensor::zeros(&device, [2, 3], DType::F32).expect("zeros");
        let err = leaf.slice(&[0..2, 0..5]).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)));
    }

    #[test]
    fn slice_rejects_inverted_range() {
        let device = Device::shared().expect("system default device");
        let leaf = Tensor::zeros(&device, [2, 3], DType::F32).expect("zeros");
        // Construct an inverted range via the struct literal so clippy
        // doesn't reject `1..0` — the whole point of this test is to
        // assert slice itself rejects start > end.
        let inverted = Range { start: 1, end: 0 };
        let err = leaf.slice(&[inverted, 0..3]).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)));
    }

    #[test]
    fn slice_rejects_quantized() {
        let t = Tensor::quantized_placeholder([4096, 4096], Quantization::Q4_GS64);
        let err = t.slice(&[0..1024, 0..1024]).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)));
    }

    #[test]
    fn slice_rejects_placeholder() {
        let p = Tensor::placeholder::<f32>([2, 3]);
        let err = p.slice(&[0..2, 0..3]).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)));
    }

    #[test]
    fn broadcast_to_pads_leading_axes() {
        // [D] -> [B, T, D]: the rmsnorm-style gamma broadcast.
        let device = Device::shared().expect("system default device");
        let gamma: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let leaf = Tensor::from_slice(&device, &gamma, [4]).expect("from_slice");

        let v = leaf.broadcast_to([2, 3, 4]).expect("broadcast");
        assert_eq!(v.shape().dims(), &[2, 3, 4]);

        // The full broadcast is gamma repeated B*T times.
        let expected: Vec<f32> = (0..6).flat_map(|_| gamma.iter().copied()).collect();
        assert_eq!(v.cpu_iter::<f32>().unwrap().collect::<Vec<_>>(), expected);
    }

    #[test]
    fn broadcast_to_expands_size_one_axis() {
        // [1, 4] -> [3, 4]: same-rank broadcast.
        let device = Device::shared().expect("system default device");
        let row: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0];
        let leaf = Tensor::from_slice(&device, &row, [1, 4]).expect("from_slice");

        let v = leaf.broadcast_to([3, 4]).expect("broadcast");
        let expected: Vec<f32> = (0..3).flat_map(|_| row.iter().copied()).collect();
        assert_eq!(v.cpu_iter::<f32>().unwrap().collect::<Vec<_>>(), expected);
    }

    #[test]
    fn broadcast_to_identity_keeps_strides() {
        // Same shape broadcast is a no-op contents-wise but still
        // builds a view (cheap and uniform with the other cases).
        let device = Device::shared().expect("system default device");
        let data: Vec<f32> = (0..6)
            .map(|i| f32::from(i16::try_from(i).unwrap()))
            .collect();
        let leaf = Tensor::from_slice(&device, &data, [2, 3]).expect("from_slice");
        let v = leaf.broadcast_to([2, 3]).expect("broadcast");
        assert_eq!(v.shape().dims(), &[2, 3]);
        assert_eq!(v.cpu_iter::<f32>().unwrap().collect::<Vec<_>>(), data);
    }

    #[test]
    fn broadcast_to_supports_zero_strides_via_iter() {
        // Scalar-ish source: [1] broadcast to [5] yields constant view.
        let device = Device::shared().expect("system default device");
        let leaf = Tensor::from_slice(&device, &[42.0_f32], [1]).expect("from_slice");
        let v = leaf.broadcast_to([5]).expect("broadcast");
        assert_eq!(
            v.cpu_iter::<f32>().unwrap().collect::<Vec<_>>(),
            vec![42.0; 5]
        );
    }

    #[test]
    fn broadcast_to_rejects_non_one_mismatch() {
        let device = Device::shared().expect("system default device");
        let leaf = Tensor::zeros(&device, [3, 4], DType::F32).expect("zeros");
        let err = leaf.broadcast_to([3, 5]).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)));
    }

    #[test]
    fn broadcast_to_rejects_target_with_lower_rank() {
        let device = Device::shared().expect("system default device");
        let leaf = Tensor::zeros(&device, [2, 3], DType::F32).expect("zeros");
        let err = leaf.broadcast_to([3]).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)));
    }

    #[test]
    fn broadcast_to_rejects_quantized() {
        let t = Tensor::quantized_placeholder([4096, 4096], Quantization::Q4_GS64);
        let err = t.broadcast_to([4096, 4096]).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)));
    }

    #[test]
    fn broadcast_to_rejects_placeholder() {
        let p = Tensor::placeholder::<f32>([4]);
        let err = p.broadcast_to([2, 4]).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)));
    }

    #[test]
    fn broadcast_view_reports_non_contiguous_cpu_bytes() {
        let device = Device::shared().expect("system default device");
        let leaf = Tensor::zeros(&device, [4], DType::F32).expect("zeros");
        let v = leaf.broadcast_to([3, 4]).expect("broadcast");
        // Stride pattern [0, 1] isn't row-major-contiguous for shape [3, 4].
        assert!(v.cpu_bytes().is_none());
        assert!(v.cpu_iter::<f32>().is_some());
    }

    #[test]
    fn view_of_unevaluated_op_is_not_materialized() {
        let device = Device::shared().expect("system default device");
        let data: Vec<f32> = vec![0.0; 8];
        let leaf = Tensor::from_slice(&device, &data, [8]).expect("from_slice");
        let op = leaf.copy().expect("copy");

        let v = manual_view(
            &op,
            0,
            Shape::from([8]),
            Strides::contiguous(&Shape::from([8])),
        );
        assert!(!v.is_materialized());
        assert!(v.cpu_bytes().is_none());

        // After evaluating the op, the view becomes readable.
        op.eval().expect("eval");
        assert!(v.is_materialized());
        let typed = v.cpu_slice::<f32>().expect("view cpu_slice");
        assert_eq!(typed, data.as_slice());
    }

    #[test]
    fn broadcast_shape_pads_leading_axes() {
        let a = Shape::from([1, 8, 896]);
        let b = Shape::from([896]);
        let out = broadcast_shape(&a, &b).expect("broadcast");
        assert_eq!(out.dims(), &[1, 8, 896]);
    }

    #[test]
    fn broadcast_shape_expands_size_one() {
        let a = Shape::from([3, 1, 5]);
        let b = Shape::from([4, 5]);
        let out = broadcast_shape(&a, &b).expect("broadcast");
        assert_eq!(out.dims(), &[3, 4, 5]);
    }

    #[test]
    fn broadcast_shape_rejects_incompatible_axes() {
        let a = Shape::from([3, 4]);
        let b = Shape::from([5, 4]);
        let err = broadcast_shape(&a, &b).unwrap_err();
        assert!(
            matches!(err, Error::InvalidArgument(_)),
            "expected InvalidArgument, got {err:?}"
        );
    }

    #[test]
    fn add_picks_vv_path_for_same_shape() {
        let device = Device::shared().expect("device");
        let lhs = Tensor::from_slice(&device, &[1.0f32; 16], [4, 4]).expect("lhs");
        let rhs = Tensor::from_slice(&device, &[2.0f32; 16], [4, 4]).expect("rhs");
        let sum = lhs.add(&rhs).expect("add");
        // Neither input was broadcast, so neither op input should be a
        // view — the dispatcher will pick `vv_Addfloat32`.
        let inputs = sum.op_inputs().expect("op inputs");
        assert_eq!(inputs.len(), 2);
        assert!(inputs[0].inner.view.is_none());
        assert!(inputs[1].inner.view.is_none());
        sum.eval().expect("eval");
        let out = sum.cpu_to_vec::<f32>().expect("out");
        assert_eq!(out, [3.0f32; 16]);
    }

    #[test]
    fn add_inserts_broadcast_view_when_shapes_differ() {
        let device = Device::shared().expect("device");
        let lhs = Tensor::from_slice(&device, &[1.0f32; 8], [2, 4]).expect("lhs");
        let rhs = Tensor::from_slice(&device, &[10.0f32; 4], [4]).expect("rhs");
        let sum = lhs.add(&rhs).expect("add");
        let inputs = sum.op_inputs().expect("op inputs");
        // lhs already matches output shape, so it stays a non-view leaf;
        // rhs gets a broadcast_to view to rank 2.
        assert!(inputs[0].inner.view.is_none());
        assert!(inputs[1].inner.view.is_some());
        sum.eval().expect("eval");
        let out = sum.cpu_to_vec::<f32>().expect("out");
        assert_eq!(out, vec![11.0; 8]);
    }

    #[test]
    fn add_rejects_dtype_mismatch() {
        let device = Device::shared().expect("device");
        let lhs = Tensor::from_slice(&device, &[1.0f32], [1]).expect("lhs");
        let rhs = Tensor::from_slice(&device, &[bf16::from_f32(1.0)], [1]).expect("rhs");
        let err = lhs.add(&rhs).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)));
    }

    #[test]
    fn mul_works_through_vv_path() {
        let device = Device::shared().expect("device");
        let lhs = Tensor::from_slice(&device, &[2.0f32, 3.0, 4.0, 5.0], [4]).expect("lhs");
        let rhs = Tensor::from_slice(&device, &[10.0f32, 10.0, 10.0, 10.0], [4]).expect("rhs");
        let prod = lhs.mul(&rhs).expect("mul");
        prod.eval().expect("eval");
        assert_eq!(
            prod.cpu_to_vec::<f32>().unwrap(),
            vec![20.0, 30.0, 40.0, 50.0]
        );
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

    // ── quantized_matmul constructor tests ──────────────────────────────────

    // N=4, K=64, Q4_GS64:
    //   w_bytes   = N * K * bits / 32 * sizeof(u32) = 4 * 64 * 4 / 32 * 4 = 128
    //   aux_bytes = N * K / group_size * sizeof(bf16) = 4 * 64 / 64 * 2    =   8
    const QMM_N: usize = 4;
    const QMM_K: usize = 64;
    const QMM_W_BYTES: usize = QMM_N * QMM_K * 4 / 32 * 4; // 128
    const QMM_AUX_BYTES: usize = QMM_N * QMM_K / 64 * 2; // 8

    fn make_test_qw(device: &crate::Device) -> crate::QuantizedWeight {
        crate::QuantizedWeight::from_bytes(
            device,
            [QMM_N, QMM_K],
            crate::Quantization::Q4_GS64,
            &[0u8; QMM_W_BYTES],
            &[0u8; QMM_AUX_BYTES],
            &[0u8; QMM_AUX_BYTES],
        )
        .expect("build synthetic QuantizedWeight")
    }

    #[test]
    fn quantized_matmul_builds_op() {
        let device = Device::shared().expect("device");
        let qw = make_test_qw(&device);

        let x_bytes = QMM_K * 2; // bf16 = 2 bytes/elem
        let x = Tensor::from_bytes(&device, &vec![0u8; x_bytes], [QMM_K], DType::BF16)
            .expect("build activation");

        let y = x.quantized_matmul(&qw, true).expect("quantized_matmul");

        assert!(!y.is_materialized(), "output must be lazy");
        assert_eq!(
            y.op_kind(),
            Some(OpKind::QuantizedMatMul {
                group_size: crate::Quantization::Q4_GS64.group_size(),
                bits: crate::Quantization::Q4_GS64.bits(),
                transpose: true,
            })
        );
        assert_eq!(y.shape().dims(), &[QMM_N]);
        assert_eq!(y.dtype(), DType::BF16);
    }

    #[test]
    fn quantized_matmul_rejects_wrong_activation_dtype() {
        let device = Device::shared().expect("device");
        let qw = make_test_qw(&device);

        // F32 activation — wrong dtype.
        let x_bytes = QMM_K * 4; // f32 = 4 bytes/elem
        let x = Tensor::from_bytes(&device, &vec![0u8; x_bytes], [QMM_K], DType::F32)
            .expect("build f32 activation");

        let err = x.quantized_matmul(&qw, true).unwrap_err();
        assert!(
            matches!(err, Error::InvalidArgument(_)),
            "expected InvalidArgument, got {err:?}"
        );
    }

    #[test]
    fn quantized_matmul_rejects_inner_dim_mismatch() {
        let device = Device::shared().expect("device");
        let qw = make_test_qw(&device);

        // Activation with K+1 — inner dim mismatch.
        let wrong_k = QMM_K + 1;
        let x_bytes = wrong_k * 2; // bf16
        let x = Tensor::from_bytes(&device, &vec![0u8; x_bytes], [wrong_k], DType::BF16)
            .expect("build mismatched activation");

        let err = x.quantized_matmul(&qw, true).unwrap_err();
        assert!(
            matches!(err, Error::InvalidArgument(_)),
            "expected InvalidArgument, got {err:?}"
        );
    }

    #[test]
    fn quantized_matmul_rejects_rank_0_activation() {
        let device = Device::shared().expect("device");
        let qw = make_test_qw(&device);

        // Rank-0 (scalar) activation — has no inner K axis.
        let x = Tensor::from_bytes(&device, &[0u8; 2], [], DType::BF16)
            .expect("build rank-0 activation");

        let err = x.quantized_matmul(&qw, true).unwrap_err();
        assert!(
            matches!(&err, Error::InvalidArgument(msg) if msg.contains("rank >= 1")),
            "expected rank >= 1 rejection, got {err:?}"
        );
    }

    #[test]
    fn quantized_matmul_rejects_unaligned_n_for_prefill() {
        let device = Device::shared().expect("device");
        // make_test_qw has N=4, which is not a multiple of 32.
        let qw = make_test_qw(&device);

        // M=2 (>1) prefill path, K matches — only N alignment is wrong.
        let m = 2;
        let x_bytes = m * QMM_K * 2; // bf16
        let x = Tensor::from_bytes(&device, &vec![0u8; x_bytes], [m, QMM_K], DType::BF16)
            .expect("build prefill activation");

        let err = x.quantized_matmul(&qw, true).unwrap_err();
        assert!(
            matches!(&err, Error::InvalidArgument(msg) if msg.contains("multiple of 32")),
            "expected aligned-N rejection, got {err:?}"
        );
    }

    #[test]
    fn quantized_matmul_rejects_transpose_false() {
        let device = Device::shared().expect("device");
        let qw = make_test_qw(&device);
        let x_bytes = QMM_K * 2;
        let x = Tensor::from_bytes(&device, &vec![0u8; x_bytes], [QMM_K], DType::BF16)
            .expect("build activation");

        let err = x.quantized_matmul(&qw, false).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("transpose=false") && msg.contains("not implemented"),
            "expected Unimplemented for transpose=false; got: {msg}"
        );
    }

    #[test]
    fn slice_update_writes_rows_at_offset_in_place() {
        let device = Device::shared().expect("system default device");
        // 4-row slab [max_tokens=4, n_kv_heads=2, head_dim=3], zero-filled.
        let slab = Tensor::zeros(&device, [4usize, 2, 3], DType::BF16).expect("slab");
        // One row of known values to write at row offset 2.
        #[allow(
            clippy::cast_precision_loss,
            reason = "values 1..=6 fit exactly in f32"
        )]
        let src_vals: Vec<bf16> = (1..=6).map(|i| bf16::from_f32(i as f32)).collect();
        let src = Tensor::from_slice(&device, &src_vals, [1usize, 2, 3]).expect("src");

        let updated = slab.slice_update(&src, 2).expect("slice_update");
        updated.eval().expect("eval");

        let out = updated.cpu_to_vec::<bf16>().expect("out bytes");
        assert_eq!(out.len(), 4 * 2 * 3);
        // Row 2 (elements 12..18) holds src; all other rows stay zero.
        let zero = bf16::from_f32(0.0);
        for (i, &x) in out.iter().enumerate() {
            if (12..18).contains(&i) {
                assert_eq!(x, src_vals[i - 12], "row 2 element {i} should be written");
            } else {
                assert_eq!(
                    x, zero,
                    "element {i} outside the written row must stay zero"
                );
            }
        }
    }

    #[test]
    fn slice_update_accepts_reshape_view_src() {
        // The real consumer (attention's `k_upd`/`v_upd`) passes a
        // contiguous reshape *view* of a copy op as `src`, not a dense
        // leaf. Exercise that path so the dispatcher resolves the view
        // chain rather than rejecting it.
        let device = Device::shared().expect("system default device");
        let slab = Tensor::zeros(&device, [4usize, 2, 3], DType::BF16).expect("slab");
        #[allow(
            clippy::cast_precision_loss,
            reason = "values 1..=6 fit exactly in f32"
        )]
        let src_vals: Vec<bf16> = (1..=6).map(|i| bf16::from_f32(i as f32)).collect();
        // copy() -> op node; reshape() -> contiguous view of that op.
        let src = Tensor::from_slice(&device, &src_vals, [2usize, 3])
            .expect("base")
            .copy()
            .expect("copy")
            .reshape([1usize, 2, 3])
            .expect("reshape view");
        assert!(src.inner.view.is_some(), "src must be a view for this test");

        let updated = slab.slice_update(&src, 1).expect("slice_update");
        updated.eval().expect("eval");

        let out = updated.cpu_to_vec::<bf16>().expect("out bytes");
        let zero = bf16::from_f32(0.0);
        for (i, &x) in out.iter().enumerate() {
            if (6..12).contains(&i) {
                assert_eq!(x, src_vals[i - 6], "row 1 element {i} should be written");
            } else {
                assert_eq!(
                    x, zero,
                    "element {i} outside the written row must stay zero"
                );
            }
        }
    }

    #[test]
    fn slice_update_rejects_offset_past_end() {
        let device = Device::shared().expect("device");
        let slab = Tensor::zeros(&device, [4usize, 2, 3], DType::BF16).expect("slab");
        let src = Tensor::zeros(&device, [2usize, 2, 3], DType::BF16).expect("src");
        // offset 3 + 2 rows = 5 > 4 max rows.
        let err = slab.slice_update(&src, 3).unwrap_err();
        assert!(
            format!("{err}").contains("out of bounds"),
            "expected an out-of-bounds error, got: {err}"
        );
    }

    #[test]
    fn slice_update_rejects_mismatched_row_shape() {
        let device = Device::shared().expect("device");
        let slab = Tensor::zeros(&device, [4usize, 2, 3], DType::BF16).expect("slab");
        // src row shape [2, 4] != slab row shape [2, 3].
        let src = Tensor::zeros(&device, [1usize, 2, 4], DType::BF16).expect("src");
        let err = slab.slice_update(&src, 0).unwrap_err();
        assert!(
            format!("{err}").contains("row shape"),
            "expected a row-shape mismatch error, got: {err}"
        );
    }

    #[test]
    fn slice_update_rejects_non_bf16() {
        let device = Device::shared().expect("device");
        let slab = Tensor::zeros(&device, [4usize, 2, 3], DType::F32).expect("slab");
        let src = Tensor::zeros(&device, [1usize, 2, 3], DType::F32).expect("src");
        let err = slab.slice_update(&src, 0).unwrap_err();
        assert!(
            format!("{err}").contains("BF16"),
            "expected a BF16-only error, got: {err}"
        );
    }

    #[test]
    fn slice_update_rejects_op_or_view_slab() {
        // The slab must be a dense leaf: an op-output slab would be
        // pool-minted, so writing in place and caching the result as
        // unpooled would alias a buffer the pool can reclaim; a view slab
        // never resolves to a dense buffer. `src` is fine as either.
        let device = Device::shared().expect("device");
        let src = Tensor::zeros(&device, [1usize, 2, 3], DType::BF16).expect("src");

        let op_slab = {
            let leaf = Tensor::zeros(&device, [4usize, 2, 3], DType::BF16).expect("leaf");
            leaf.add(&leaf).expect("op slab")
        };
        assert!(op_slab.inner.op.is_some(), "op_slab must be an op node");
        let err = op_slab.slice_update(&src, 0).unwrap_err();
        assert!(
            format!("{err}").contains("dense leaf"),
            "expected a dense-leaf error for an op slab, got: {err}"
        );

        let view_slab = Tensor::zeros(&device, [4usize, 2, 3], DType::BF16)
            .expect("leaf")
            .reshape([2usize, 4, 3])
            .expect("view slab");
        assert!(
            view_slab.inner.view.is_some(),
            "view_slab must be a view node"
        );
        let err = view_slab.slice_update(&src, 0).unwrap_err();
        assert!(
            format!("{err}").contains("dense leaf"),
            "expected a dense-leaf error for a view slab, got: {err}"
        );
    }

    #[test]
    fn argmax_returns_u32_index_dropping_last_axis() {
        let device = Device::shared().expect("device");
        let vals: Vec<bf16> = [0.1f32, 0.5, 0.9, 0.2]
            .iter()
            .map(|&x| bf16::from_f32(x))
            .collect();
        let t = Tensor::from_slice(&device, &vals, [1usize, 4]).expect("leaf");
        let idx = t.argmax(1).expect("argmax");
        assert_eq!(idx.shape().dims(), &[1]);
        assert_eq!(idx.dtype(), DType::U32);
        idx.eval().expect("eval");
        assert_eq!(idx.cpu_slice::<u32>().expect("u32 slice"), &[2]);
    }

    #[test]
    fn argmax_decode_shape_1_1_vocab() {
        let device = Device::shared().expect("device");
        // [1, 1, 5] — the decode logits shape; max at vocab index 3.
        let vals: Vec<bf16> = [0.1f32, 0.4, 0.2, 0.9, 0.3]
            .iter()
            .map(|&x| bf16::from_f32(x))
            .collect();
        let logits = Tensor::from_slice(&device, &vals, [1usize, 1, 5]).expect("leaf");
        let idx = logits.argmax(2).expect("argmax");
        assert_eq!(idx.shape().dims(), &[1, 1]);
        idx.eval().expect("eval");
        assert_eq!(idx.cpu_slice::<u32>().expect("u32"), &[3]);
    }

    #[test]
    fn argmax_rejects_non_bf16() {
        let device = Device::shared().expect("device");
        let t = Tensor::zeros(&device, [1usize, 4], DType::F32).expect("leaf");
        let err = t.argmax(1).unwrap_err();
        assert!(format!("{err}").contains("BF16"), "got: {err}");
    }
}
