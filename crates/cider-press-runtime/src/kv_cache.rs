//! Key/value cache for transformer decode loops.
//!
//! [`KvCache`] is the first runtime-side abstraction that isn't a
//! lazy-graph op. The decode loop appends one (or, for prefill, `T`)
//! K/V rows per step into a pre-allocated slab and immediately reads
//! the populated prefix back through SDPA — semantics that don't fit
//! the `OnceLock<LeafStorage>` "set once" model used by [`Tensor`].
//! See `docs/RUNTIME_DESIGN.md`'s framework-gap analysis for the
//! motivation behind separating this from [`Tensor`].
//!
//! ## Shape convention
//!
//! Slabs are laid out `[max_tokens, n_kv_heads, head_dim]` (T-outer)
//! to match the design-doc shape and keep `update` a contiguous byte
//! append. The eventual SDPA dispatcher will permute to its preferred
//! layout through the existing zero-copy view machinery.
//!
//! ## Eager update
//!
//! [`KvCache::update`] is eager: it materializes the source K/V
//! tensors and `memcpy`s their bytes into the slab via the unified-
//! memory host pointer before returning. Apple Silicon's shared
//! storage makes this a single host write — no Metal dispatch, no
//! `did_modify_range`. The trade-off vs. issuing a `Copy` kernel into
//! the slab is documented in `docs/QWEN_PATH.md`'s branch-8 entry:
//! same-eval batching (folding the cache write into the SDPA command
//! buffer) is deferred until branch 11 if perf measurement demands
//! it.
//!
//! ## Aliasing contract
//!
//! [`KvCache::keys_view`] / [`KvCache::values_view`] return [`Tensor`]
//! handles that share the slab's underlying `MTLBuffer`. A subsequent
//! [`KvCache::update`] mutates the slab, which would invalidate any
//! still-live view. Callers must drop view tensors before calling
//! `update` again — in practice the decode loop consumes views
//! through `eval()` within the same iteration that produced them, so
//! the lifetime falls out naturally.

use std::fmt;

use cider_press_kernels::Buffer;

use crate::Device;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::layout::Layout;
use crate::shape::Shape;
use crate::strides::Strides;
use crate::tensor::Tensor;

/// Per-layer key/value cache backing a transformer decode loop.
///
/// One instance per decoder layer. Holds a pair of pre-allocated
/// `[max_tokens, n_kv_heads, head_dim]` slabs in shared storage; the
/// populated prefix grows by `step_t` rows on each [`KvCache::update`]
/// and is exposed as zero-copy [`Tensor`] views.
pub struct KvCache {
    device: Device,
    keys: Buffer<u8>,
    values: Buffer<u8>,
    dtype: DType,
    max_tokens: usize,
    n_kv_heads: usize,
    head_dim: usize,
    position: usize,
}

impl KvCache {
    /// Allocate a fresh cache with capacity for `max_tokens` rows.
    ///
    /// The two slabs are allocated uninitialized — no zero-fill,
    /// because the `keys_view` / `values_view` shapes only ever
    /// expose the populated prefix `[0..position]`.
    ///
    /// `dtype` must be a float type (`f32`, `f16`, or `bf16`);
    /// quantized K/V caches are out of scope.
    pub fn new(
        device: &Device,
        max_tokens: usize,
        n_kv_heads: usize,
        head_dim: usize,
        dtype: DType,
    ) -> Result<Self> {
        if max_tokens == 0 || n_kv_heads == 0 || head_dim == 0 {
            return Err(Error::InvalidArgument(format!(
                "KvCache::new: dims must be non-zero (max_tokens={max_tokens}, \
                 n_kv_heads={n_kv_heads}, head_dim={head_dim})"
            )));
        }
        match dtype {
            DType::F32 | DType::F16 | DType::BF16 => {}
            DType::I32 | DType::U32 => {
                return Err(Error::InvalidArgument(format!(
                    "KvCache::new: dtype {dtype:?} is not supported (float dtypes only)"
                )));
            }
        }
        let elem_count = max_tokens
            .checked_mul(n_kv_heads)
            .and_then(|x| x.checked_mul(head_dim))
            .ok_or_else(|| {
                Error::InvalidArgument(format!(
                    "KvCache::new: slab element count overflows usize \
                     (max_tokens={max_tokens}, n_kv_heads={n_kv_heads}, head_dim={head_dim})"
                ))
            })?;
        let byte_count = elem_count.checked_mul(dtype.size_bytes()).ok_or_else(|| {
            Error::InvalidArgument(format!(
                "KvCache::new: slab byte count overflows usize \
                 (elem_count={elem_count}, dtype={dtype})"
            ))
        })?;
        let keys = device.kernels().alloc_buffer::<u8>(byte_count)?;
        let values = device.kernels().alloc_buffer::<u8>(byte_count)?;
        Ok(Self {
            device: device.clone(),
            keys,
            values,
            dtype,
            max_tokens,
            n_kv_heads,
            head_dim,
            position: 0,
        })
    }

    /// Append `step_t` rows of K and V into the slabs at the current
    /// position. `k` and `v` must each be dense, contiguous, shaped
    /// `[step_t, n_kv_heads, head_dim]` on the same device and dtype
    /// as this cache. After this call returns, `position()` advances
    /// by `step_t`.
    ///
    /// `k` and `v` are evaluated synchronously (the inputs may be op
    /// nodes whose materialization the decode loop needs before
    /// SDPA reads the cache anyway), then their bytes are `memcpy`'d
    /// into the slabs via the unified-memory host pointer.
    pub fn update(&mut self, k: &Tensor, v: &Tensor) -> Result<()> {
        let step_t = self.validate_update_input(k, "k")?;
        let step_t_v = self.validate_update_input(v, "v")?;
        if step_t != step_t_v {
            return Err(Error::InvalidArgument(format!(
                "KvCache::update: k and v must have matching step_t \
                 (k.shape[0]={step_t}, v.shape[0]={step_t_v})"
            )));
        }
        if self.position.saturating_add(step_t) > self.max_tokens {
            return Err(Error::InvalidArgument(format!(
                "KvCache::update: writing {step_t} rows at position {} would exceed max_tokens={}",
                self.position, self.max_tokens
            )));
        }

        k.eval()?;
        v.eval()?;
        let k_bytes = k.cpu_bytes().ok_or_else(|| {
            Error::InvalidArgument(
                "KvCache::update: k did not materialize as dense contiguous bytes".into(),
            )
        })?;
        let v_bytes = v.cpu_bytes().ok_or_else(|| {
            Error::InvalidArgument(
                "KvCache::update: v did not materialize as dense contiguous bytes".into(),
            )
        })?;

        let row_bytes = self.n_kv_heads * self.head_dim * self.dtype.size_bytes();
        let offset = self.position * row_bytes;
        let write_bytes = step_t * row_bytes;
        debug_assert_eq!(k_bytes.len(), write_bytes);
        debug_assert_eq!(v_bytes.len(), write_bytes);

        // SAFETY: no GPU dispatch is reading the slab during update
        // (callers drop any prior keys_view/values_view tensors before
        // calling update — documented contract). Host writes into
        // shared-storage buffers on Apple Silicon are immediately
        // visible to subsequent GPU dispatches that bind the same
        // buffer.
        unsafe {
            self.keys.as_mut_slice()[offset..offset + write_bytes].copy_from_slice(k_bytes);
            self.values.as_mut_slice()[offset..offset + write_bytes].copy_from_slice(v_bytes);
        }
        self.position += step_t;
        Ok(())
    }

    /// View of the populated K prefix as a dense
    /// `[position, n_kv_heads, head_dim]` [`Tensor`] sharing the
    /// slab's underlying `MTLBuffer`. Reads `position == 0` as a
    /// zero-row view (still valid; SDPA over a zero-length cache is
    /// the prefill-start case).
    #[must_use]
    pub fn keys_view(&self) -> Tensor {
        self.prefix_view(&self.keys)
    }

    /// View of the populated V prefix; see [`KvCache::keys_view`].
    #[must_use]
    pub fn values_view(&self) -> Tensor {
        self.prefix_view(&self.values)
    }

    /// Number of rows currently populated (`0..=max_tokens`).
    #[must_use]
    pub fn position(&self) -> usize {
        self.position
    }

    /// Capacity of the cache in rows.
    #[must_use]
    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    /// Number of KV heads per row (Qwen2.5-0.5B has 2; GQA factor 7
    /// against 14 query heads).
    #[must_use]
    pub fn n_kv_heads(&self) -> usize {
        self.n_kv_heads
    }

    /// Per-head dimensionality.
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Element type of K and V.
    #[must_use]
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// [`Device`] the slabs are allocated on.
    #[must_use]
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Reset to position 0 without freeing the slabs — cheap path
    /// for restarting a decode loop on a new prompt with the same
    /// cache instance. The slab bytes are left untouched (no
    /// zero-fill); the next [`KvCache::update`] writes over them.
    pub fn reset(&mut self) {
        self.position = 0;
    }

    fn validate_update_input(&self, t: &Tensor, name: &str) -> Result<usize> {
        let t_device = t.device().ok_or_else(|| {
            Error::InvalidArgument(format!(
                "KvCache::update: {name} is a placeholder (no device)"
            ))
        })?;
        if !self.device.ptr_eq(t_device) {
            return Err(Error::InvalidArgument(format!(
                "KvCache::update: {name} is on a different device than the cache"
            )));
        }
        if t.dtype() != self.dtype {
            return Err(Error::InvalidArgument(format!(
                "KvCache::update: {name} dtype {:?} does not match cache dtype {:?}",
                t.dtype(),
                self.dtype,
            )));
        }
        let dims = t.shape().dims();
        if dims.len() != 3 || dims[1] != self.n_kv_heads || dims[2] != self.head_dim {
            return Err(Error::InvalidArgument(format!(
                "KvCache::update: {name} shape {:?} does not match \
                 [step_t, n_kv_heads={}, head_dim={}]",
                dims, self.n_kv_heads, self.head_dim,
            )));
        }
        Ok(dims[0])
    }

    fn prefix_view(&self, slab: &Buffer<u8>) -> Tensor {
        let shape = Shape::new([self.position, self.n_kv_heads, self.head_dim]);
        let layout = Layout::Dense {
            strides: Strides::contiguous(&shape),
        };
        // Refcount-bump clone: the view aliases the slab's MTLBuffer.
        // The aliasing contract (see module docs) makes this view live
        // only until the caller drops it before the next update().
        let buf = slab.clone_handle();
        Tensor::host_leaf(&self.device, shape, self.dtype, layout, buf)
    }
}

impl fmt::Debug for KvCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KvCache")
            .field("dtype", &self.dtype)
            .field("max_tokens", &self.max_tokens)
            .field("n_kv_heads", &self.n_kv_heads)
            .field("head_dim", &self.head_dim)
            .field("position", &self.position)
            .finish_non_exhaustive()
    }
}
