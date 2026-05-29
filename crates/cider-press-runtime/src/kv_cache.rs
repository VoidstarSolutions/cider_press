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
//! ## Lazy update
//!
//! [`KvCache::update`] is lazy: it builds an in-graph `slice_update` op
//! per slab (writing this step's K/V rows into the slab buffer at the
//! current row offset) and returns without evaluating. No host `memcpy`
//! and no forced `eval` — the write lands in the same command buffer as
//! the SDPA read that consumes it, which is what collapses a decode step
//! to a single GPU submission. Each `update` chains its op off the
//! previous one (or the slab leaf on the first call); `keys_view` /
//! `values_view` expose the populated prefix of the latest op.
//!
//! ## Aliasing contract (append-only)
//!
//! `update` performs no host-side mutation: the write is an in-graph
//! `SliceUpdate` GPU op, ordered against the SDPA read by Metal's hazard
//! tracking within a command buffer and by `commit_and_wait` across
//! buffers. The cache is **append-only** — `position` only grows and each
//! `update` writes a fresh, disjoint row range — so a [`Tensor`] returned
//! by [`KvCache::keys_view`]/[`KvCache::values_view`] reads a prefix that
//! prior dispatches have fully written and remains valid until a later
//! [`KvCache::update`] overwrites the rows it spans — the earliest such
//! overwrite being the first `update` after a [`KvCache::reset`] rewinds
//! `position`. (The eager design's `Arc::strong_count` guard, which
//! prevented host-side overlapping `&[u8]`/`&mut [u8]` UB, is obsolete
//! now that there is no host write.)

use std::fmt;
use std::sync::Arc;

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
    // Full-shape `[max_tokens, n_kv_heads, head_dim]` host_leaf tensors.
    // Views handed out by `keys_view` / `values_view` are slices of these,
    // so the view's logical length matches its shape even when only a prefix
    // is populated.
    keys_slab: Tensor,
    values_slab: Tensor,
    dtype: DType,
    max_tokens: usize,
    n_kv_heads: usize,
    head_dim: usize,
    position: usize,
    // Most recent SliceUpdate op-tensor per slab. `update` builds these
    // lazily (no host memcpy); `keys_view`/`values_view` slice them so
    // SDPA's cache read is graph-downstream of the write. `None` before
    // the first update (or after `reset`).
    keys_latest: Option<Tensor>,
    values_latest: Option<Tensor>,
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
        let slab_shape = Shape::new([max_tokens, n_kv_heads, head_dim]);
        let slab_layout = Layout::Dense {
            strides: Strides::contiguous(&slab_shape),
        };
        let keys_slab = Tensor::host_leaf(
            device,
            slab_shape.clone(),
            dtype,
            slab_layout.clone(),
            keys.clone_handle(),
        );
        let values_slab = Tensor::host_leaf(
            device,
            slab_shape,
            dtype,
            slab_layout,
            values.clone_handle(),
        );
        // Freshly built and unshared here — a construction tripwire only.
        debug_assert_eq!(Arc::strong_count(&keys_slab.inner), 1);
        debug_assert_eq!(Arc::strong_count(&values_slab.inner), 1);
        Ok(Self {
            device: device.clone(),
            keys_slab,
            values_slab,
            dtype,
            max_tokens,
            n_kv_heads,
            head_dim,
            position: 0,
            keys_latest: None,
            values_latest: None,
        })
    }

    /// Append `step_t` rows of K and V into the slabs at the current
    /// position. `k` and `v` must each be dense, contiguous, shaped
    /// `[step_t, n_kv_heads, head_dim]` on the same device and dtype
    /// as this cache. After this call returns, `position()` advances
    /// by `step_t`.
    ///
    /// Builds a lazy [`SliceUpdate`] op per slab (no eval, no host
    /// `memcpy`). The write lands in the same command buffer as the
    /// downstream SDPA read that consumes [`KvCache::keys_view`] /
    /// [`KvCache::values_view`].
    pub fn update(&mut self, k: &Tensor, v: &Tensor) -> Result<()> {
        let _span = crate::profile::span("kvcache.update");
        // No aliasing guard: `update` does no host write (the write is an
        // in-graph SliceUpdate op), and the cache is append-only, so a
        // live view reads a still-valid prefix. See the module-level
        // append-only aliasing contract.
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

        // Lazy in-graph write: chain each SliceUpdate off the previous
        // op-tensor (or the slab leaf on first use) so that successive
        // writes compose in graph order. No eval, no host memcpy — the
        // writes land in the same command buffer as the SDPA read that
        // follows.
        let k_base = self.keys_latest.as_ref().unwrap_or(&self.keys_slab);
        let v_base = self.values_latest.as_ref().unwrap_or(&self.values_slab);
        self.keys_latest = Some(k_base.slice_update(k, self.position)?);
        self.values_latest = Some(v_base.slice_update(v, self.position)?);
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
        self.prefix_view(self.keys_latest.as_ref().unwrap_or(&self.keys_slab))
    }

    /// View of the populated V prefix; see [`KvCache::keys_view`].
    #[must_use]
    pub fn values_view(&self) -> Tensor {
        self.prefix_view(self.values_latest.as_ref().unwrap_or(&self.values_slab))
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
        self.keys_latest = None;
        self.values_latest = None;
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

    fn prefix_view(&self, slab: &Tensor) -> Tensor {
        // Slice the full-shape slab tensor along axis 0 down to the
        // populated prefix. The resulting view aliases the slab's
        // MTLBuffer (zero-copy) and its logical length matches its shape,
        // so vector-kernel dispatch sees a consistent
        // `buffer.len() == shape.elem_count() * dtype.size_bytes()`.
        slab.slice(&[0..self.position, 0..self.n_kv_heads, 0..self.head_dim])
            .expect("prefix slice is in-bounds by construction")
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

#[cfg(test)]
mod tests {
    use half::bf16;

    use super::*;
    use crate::Device;
    use crate::dtype::DType;

    #[test]
    fn lazy_update_then_view_reads_written_rows() {
        let device = Device::shared().expect("device");
        let mut cache = KvCache::new(&device, 4, 2, 3, DType::BF16).expect("cache");

        let k0: Vec<bf16> = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]
            .iter()
            .map(|&x| bf16::from_f32(x))
            .collect();
        let v0: Vec<bf16> = [7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0]
            .iter()
            .map(|&x| bf16::from_f32(x))
            .collect();
        let kt = Tensor::from_slice(&device, &k0, [1usize, 2, 3]).expect("k0");
        let vt = Tensor::from_slice(&device, &v0, [1usize, 2, 3]).expect("v0");
        cache.update(&kt, &vt).expect("update 0");
        assert_eq!(cache.position(), 1);

        {
            let kview = cache.keys_view();
            kview.eval().expect("eval kview");
            assert_eq!(kview.cpu_to_vec::<bf16>().expect("k bytes"), k0);
            let vview = cache.values_view();
            vview.eval().expect("eval vview");
            assert_eq!(vview.cpu_to_vec::<bf16>().expect("v bytes"), v0);
        } // drop views before the next update (aliasing)

        // Second row at offset 1; the prefix view must show both rows.
        let k1: Vec<bf16> = [13.0f32, 14.0, 15.0, 16.0, 17.0, 18.0]
            .iter()
            .map(|&x| bf16::from_f32(x))
            .collect();
        let kt1 = Tensor::from_slice(&device, &k1, [1usize, 2, 3]).expect("k1");
        cache.update(&kt1, &kt1).expect("update 1");
        assert_eq!(cache.position(), 2);

        let kview = cache.keys_view();
        kview.eval().expect("eval kview 2");
        let mut expected = k0.clone();
        expected.extend_from_slice(&k1);
        assert_eq!(kview.cpu_to_vec::<bf16>().expect("k bytes 2"), expected);
    }

    #[test]
    fn view_stays_valid_across_later_append() {
        let device = Device::shared().expect("device");
        let mut cache = KvCache::new(&device, 4, 2, 3, DType::BF16).expect("cache");

        let row = |base: i16| -> Vec<bf16> {
            (0..6i16)
                .map(|i| bf16::from_f32(f32::from(base + i)))
                .collect()
        };
        let r0 = row(1);
        let t0 = Tensor::from_slice(&device, &r0, [1usize, 2, 3]).expect("t0");
        cache.update(&t0, &t0).expect("update 0");

        // Capture a view of the row-0 prefix, then append row 1.
        let kview0 = cache.keys_view();
        let r1 = row(100);
        let t1 = Tensor::from_slice(&device, &r1, [1usize, 2, 3]).expect("t1");
        cache.update(&t1, &t1).expect("update 1");

        // The earlier view still reads its (still-valid) prefix: append only
        // wrote the disjoint row 1, so row 0's bytes are unchanged.
        kview0.eval().expect("eval kview0");
        assert_eq!(kview0.cpu_to_vec::<bf16>().expect("kview0"), r0);
    }
}
