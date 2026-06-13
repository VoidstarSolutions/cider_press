//! Key/value cache for transformer decode loops.
//!
//! [`KvCache`] is the first runtime-side abstraction that isn't a
//! lazy-graph op. The decode loop appends one (or, for prefill, `T`)
//! K/V rows per step into a pre-allocated slab and immediately reads
//! the populated prefix back through SDPA — semantics that don't fit
//! the `OnceLock<LeafStorage>` "set once" model used by [`Tensor`].
//! See [`the execution model`](../../../docs/inference/execution-model.md)
//! for why this is separate from [`Tensor`].
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
//! to a single GPU submission. Each `update` builds its op directly on the
//! slab leaf (not chained off the prior update), so per-step graphs are not
//! retained across decode steps and the buffer pool can recycle per-step
//! scratch; `keys_view` / `values_view` expose the populated prefix of the
//! latest op.
//!
//! ## Aliasing contract (append-only)
//!
//! `update` performs no host-side mutation: the write is an in-graph
//! `SliceUpdate` GPU op, ordered against the SDPA read by the encoder fence
//! chain and in-encoder barriers within a command buffer and by
//! `commit_and_wait` across buffers. The cache is **append-only** — `position` only grows and each
//! `update` writes a fresh, disjoint row range — so a [`Tensor`] returned
//! by [`KvCache::keys_view`]/[`KvCache::values_view`] reads a prefix that
//! prior dispatches have fully written and remains valid until a later
//! [`KvCache::update`] overwrites the rows it spans — the earliest such
//! overwrite being the first `update` after a [`KvCache::reset`] rewinds
//! `position`. (The eager design's `Arc::strong_count` guard, which
//! prevented host-side overlapping `&[u8]`/`&mut [u8]` UB, is obsolete
//! now that there is no host write.)
//!
//! Because both [`KvCache::update`] and [`KvCache::reset`] reuse the slab,
//! each fails loud (`Error::InvalidArgument`) if the previous `update` has
//! not been evaluated: that guarantees every `SliceUpdate` the cache ever
//! produced is materialized, so any outstanding view short-circuits on
//! `eval` (a materialized node is never re-dispatched) and cannot replay a
//! stale write into the reused slab.
#![doc = include_str!("../../../docs/inference/kv-cache.md")]

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
    // Most recent SliceUpdate op-tensor per slab, built directly on the
    // slab leaf (not chained off the prior update, so per-step graphs free
    // for the buffer pool). `keys_view`/`values_view` slice them so SDPA's
    // cache read is graph-downstream of the write. `None` before the first
    // update (or after `reset`).
    keys_latest: Option<Tensor>,
    values_latest: Option<Tensor>,
}

impl KvCache {
    /// Allocate a fresh cache with capacity for `max_tokens` rows.
    ///
    /// The two slabs are zero-filled at construction. `keys_view` /
    /// `values_view` only ever expose the populated prefix `[0..position]`,
    /// so under serial dispatch the fill is invisible — but concurrent
    /// dispatch loses the automatic write-before-read ordering, and a
    /// zeroed slab keeps a premature read of an unwritten row benign rather
    /// than garbage (see the module-level aliasing-contract note).
    ///
    /// `dtype` must be [`DType::BF16`] — the only dtype the slab write path
    /// ([`Tensor::slice_update`]) supports. Other floats and quantized K/V
    /// caches are out of scope.
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
        // BF16-only: the slab write path (`Tensor::slice_update`) rejects
        // every other dtype, so accepting F16/F32 here would construct a
        // cache that fails on its first `update()`.
        if dtype != DType::BF16 {
            return Err(Error::InvalidArgument(format!(
                "KvCache::new: only BF16 is supported (got {dtype:?}); \
                 the slab write path is BF16-only"
            )));
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
        let mut keys = device.kernels().alloc_buffer::<u8>(byte_count)?;
        let mut values = device.kernels().alloc_buffer::<u8>(byte_count)?;
        // Zero the full slabs. Reads only ever target the populated prefix
        // `[0..position]`. Under concurrent dispatch with untracked buffers,
        // an unwritten row can be read before its write lands if the encoder
        // fence chain has a gap (e.g. the first step before any write has
        // committed); a deterministically-zero slab makes that read benign
        // instead of garbage. Cheap (one host memset at construction; shared
        // storage).
        // SAFETY: freshly allocated; no dispatch has referenced these
        // buffers yet, so the host pointer is exclusively ours to write.
        unsafe {
            keys.as_mut_slice().fill(0);
            values.as_mut_slice().fill(0);
        }
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
    /// Builds a lazy `SliceUpdate` op per slab (no eval, no host
    /// `memcpy`). The write lands in the same command buffer as the
    /// downstream SDPA read that consumes [`KvCache::keys_view`] /
    /// [`KvCache::values_view`].
    ///
    /// Successive `update`s on the same cache must be separated by an
    /// `eval` (or `eval_async` commit) of the populated prefix
    /// (`keys_view`/`values_view`): each `update` keeps only the latest
    /// `slice_update` op, so an un-committed prior write would be dropped.
    /// Returns `Error::InvalidArgument` if called again before the previous
    /// update is committed. (Decode commits each token; prefill is a single
    /// update. A still-in-flight commit qualifies — the write is encoded
    /// and submitted, so it can neither be dropped nor replayed.)
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

        // Eager slab-base contract: each `update` retains only the latest
        // SliceUpdate op; the next one bases on the raw slab. If a prior
        // update has not been committed, basing on the slab would drop its
        // (unreferenced, never-executed) write. Fail loud rather than
        // silently lose it — callers must `eval`/`eval_async` between
        // successive updates (decode commits each token; prefill is a
        // single update + eval).
        if !self.prior_update_committed() {
            return Err(Error::InvalidArgument(
                "KvCache::update: the previous update has not been committed; \
                 successive updates must be separated by an eval or eval_async \
                 (the slab-based slice_update would otherwise drop the prior, \
                 unsubmitted write)"
                    .into(),
            ));
        }

        // Lazy in-graph write: build each SliceUpdate directly on the slab
        // leaf (not chained off the previous op-tensor). The slab already
        // holds the prior committed rows, and this step's disjoint row write
        // is ordered before the SDPA read by the encoder fence chain. Basing
        // on the slab
        // — rather than retaining the prior `keys_latest`/`values_latest`
        // op — means each step's projection graph frees once the next
        // `update` reassigns these fields, so the buffer pool can recycle
        // per-step scratch. No eval, no host memcpy.
        self.keys_latest = Some(self.keys_slab.slice_update(k, self.position)?);
        self.values_latest = Some(self.values_slab.slice_update(v, self.position)?);
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
    ///
    /// Requires the last update to have been committed, for the same
    /// reason as [`KvCache::update`]: clearing `keys_latest`/`values_latest`
    /// here drops the cache's only strong ref to the latest `SliceUpdate`,
    /// so an *un-committed* one outstanding via [`KvCache::keys_view`] could
    /// later replay its stale write into the (now reused) slab. With the
    /// last update committed, every `SliceUpdate` the cache produced has a
    /// populated cache (the update guard covers superseded ones), so any
    /// outstanding view short-circuits on `eval` and cannot replay. A
    /// still-in-flight commit qualifies, but the *caller* must ensure the
    /// GPU work completes before the slab bytes are rewritten (the
    /// generator does: dropped `PendingEval`s block on completion). Returns
    /// `Error::InvalidArgument` otherwise.
    pub fn reset(&mut self) -> Result<()> {
        if !self.prior_update_committed() {
            return Err(Error::InvalidArgument(
                "KvCache::reset: the last update has not been committed; resetting now \
                 would let an outstanding, un-committed K/V view replay its stale write \
                 into the reused slab (eval the view before reset)"
                    .into(),
            ));
        }
        self.position = 0;
        self.keys_latest = None;
        self.values_latest = None;
        Ok(())
    }

    /// Whether the latest K/V `SliceUpdate` (if any) has been committed to
    /// the GPU — by a completed `eval` or a possibly-still-in-flight
    /// `eval_async`. `true` when there is no pending update (both `None`).
    /// Shared by [`KvCache::update`] and [`KvCache::reset`], which both
    /// reuse the slab and so must not strand an un-committed prior write.
    /// Deliberately `Tensor::is_committed`, not `Tensor::is_materialized`:
    /// the async decode pipeline calls `update` while the prior token's
    /// command buffer is still running, and committed is what the
    /// no-drop/no-replay argument needs.
    fn prior_update_committed(&self) -> bool {
        self.keys_latest.as_ref().is_none_or(Tensor::is_committed)
            && self.values_latest.as_ref().is_none_or(Tensor::is_committed)
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
        let core_ok = (dims.len() == 3 || (dims.len() == 4 && dims[3] == 1))
            && dims[1] == self.n_kv_heads
            && dims[2] == self.head_dim;
        if !core_ok {
            return Err(Error::InvalidArgument(format!(
                "KvCache::update: {name} shape {:?} does not match \
                 [step_t, n_kv_heads={}, head_dim={}] (optional trailing 1)",
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
    fn new_rejects_non_bf16_dtype() {
        // The slab write path is BF16-only; F16/F32 caches would construct
        // here and then fail on their first update(), so reject up front.
        let device = Device::shared().expect("device");
        for dtype in [DType::F32, DType::F16, DType::I32, DType::U32] {
            let err =
                KvCache::new(&device, 4, 2, 3, dtype).expect_err("non-BF16 cache must be rejected");
            assert!(
                err.to_string().contains("only BF16"),
                "expected a BF16-only error for {dtype:?}, got: {err}"
            );
        }
    }

    #[test]
    fn new_cache_zeroes_full_slabs() {
        let device = Device::shared().expect("device");
        let cache = KvCache::new(&device, 4, 2, 3, DType::BF16).expect("cache");

        // Read the *full* slab memory — every row, not just the populated
        // prefix `keys_view` exposes. Under concurrent dispatch an unwritten
        // row can be read before its `SliceUpdate` lands, so the slab must be
        // deterministically zero, never left uninitialized.
        for (name, slab) in [("keys", &cache.keys_slab), ("values", &cache.values_slab)] {
            let bytes = match slab.inner.cache.get() {
                Some(crate::tensor::LeafStorage::Dense(buf)) => unsafe { buf.as_slice() },
                _ => panic!("{name} slab is not a dense leaf"),
            };
            assert!(
                bytes.iter().all(|&b| b == 0),
                "{name} slab not zero-initialized: first nonzero byte at {:?}",
                bytes.iter().position(|&b| b != 0),
            );
        }
    }

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

        // Capture a view of the row-0 prefix; eval it (satisfying the
        // between-update eval contract), then append row 1.
        let kview0 = cache.keys_view();
        kview0.eval().expect("eval kview0 before update 1");
        cache
            .values_view()
            .eval()
            .expect("eval vview0 before update 1");
        let r1 = row(100);
        let t1 = Tensor::from_slice(&device, &r1, [1usize, 2, 3]).expect("t1");
        cache.update(&t1, &t1).expect("update 1");

        // The earlier view still reads its (still-valid) prefix: append only
        // wrote the disjoint row 1, so row 0's bytes are unchanged.
        kview0.eval().expect("eval kview0");
        assert_eq!(kview0.cpu_to_vec::<bf16>().expect("kview0"), r0);
    }

    /// Regression guard for the buffer-pool retention bug: `update` must
    /// not pin each step's projection graph across the whole decode. We
    /// drive a decode-like loop where the K/V inputs are pooled op
    /// outputs (`leaf.add`), and assert the pool registers reuse hits —
    /// which only happens if per-step graphs free between steps. With the
    /// old chaining `update` (base = previous op-tensor) every step's
    /// `add` output stays pinned by `keys_latest`/`values_latest` for the
    /// whole loop, so hits stay at 0 and this fails.
    #[test]
    fn update_does_not_retain_per_step_graphs() {
        let device = Device::system_default().expect("device");
        let mut cache = KvCache::new(&device, 16, 2, 3, DType::BF16).expect("cache");

        let row: Vec<bf16> = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]
            .iter()
            .map(|&x| bf16::from_f32(x))
            .collect();

        for _ in 0..8 {
            let leaf = Tensor::from_slice(&device, &row, [1usize, 2, 3]).expect("leaf");
            let k = leaf.add(&leaf).expect("k");
            let v = leaf.add(&leaf).expect("v");
            cache.update(&k, &v).expect("update");
            cache.keys_view().eval().expect("eval keys");
            cache.values_view().eval().expect("eval values");
        }

        assert!(
            device.pool_stats().hits > 0,
            "per-step buffers must free and recycle (pool hits {} == 0 means \
             update is retaining each step's graph)",
            device.pool_stats().hits
        );
    }

    /// Two updates without an intervening eval must fail loud (the eager
    /// slab-base form would otherwise silently drop the first write).
    #[test]
    fn update_without_intervening_eval_errors() {
        let device = Device::system_default().expect("device");
        let mut cache = KvCache::new(&device, 4, 2, 3, DType::BF16).expect("cache");

        let row: Vec<bf16> = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]
            .iter()
            .map(|&x| bf16::from_f32(x))
            .collect();
        let k = Tensor::from_slice(&device, &row, [1usize, 2, 3]).expect("k");
        let v = Tensor::from_slice(&device, &row, [1usize, 2, 3]).expect("v");

        cache.update(&k, &v).expect("first update ok");
        // No eval here → second update must error rather than lose row 0.
        let err = cache.update(&k, &v).unwrap_err();
        assert!(
            format!("{err}").contains("must be separated by an eval"),
            "expected eval-between-updates error, got: {err}"
        );

        // After evaluating, a further update is accepted again.
        cache.keys_view().eval().expect("eval keys");
        cache.values_view().eval().expect("eval values");
        cache.update(&k, &v).expect("update after eval ok");
    }

    /// `reset` reuses the slab, so an un-committed last update must fail
    /// loud — otherwise an outstanding view could replay a stale write.
    #[test]
    fn reset_without_intervening_eval_errors() {
        let device = Device::system_default().expect("device");
        let mut cache = KvCache::new(&device, 4, 2, 3, DType::BF16).expect("cache");

        let row: Vec<bf16> = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]
            .iter()
            .map(|&x| bf16::from_f32(x))
            .collect();
        let k = Tensor::from_slice(&device, &row, [1usize, 2, 3]).expect("k");
        let v = Tensor::from_slice(&device, &row, [1usize, 2, 3]).expect("v");

        cache.update(&k, &v).expect("update ok");
        let err = cache.reset().unwrap_err();
        assert!(
            format!("{err}").contains("has not been committed"),
            "expected a commit-before-reset error, got: {err}"
        );

        // After evaluating the last update, reset is accepted.
        cache.keys_view().eval().expect("eval keys");
        cache.values_view().eval().expect("eval values");
        cache.reset().expect("reset after eval ok");
        assert_eq!(cache.position(), 0);
    }

    /// The async decode pipeline calls `update` while the prior update's
    /// command buffer is still in flight: a committed-but-unwaited
    /// `eval_async` must satisfy the slab-reuse guard (committed writes can
    /// neither be dropped nor replayed), even though the in-flight tensor
    /// is not yet host-readable.
    #[test]
    fn update_accepts_prior_update_committed_but_in_flight() {
        let device = Device::system_default().expect("device");
        let mut cache = KvCache::new(&device, 4, 2, 3, DType::BF16).expect("cache");

        let row: Vec<bf16> = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]
            .iter()
            .map(|&x| bf16::from_f32(x))
            .collect();
        let k = Tensor::from_slice(&device, &row, [1usize, 2, 3]).expect("k");
        let v = Tensor::from_slice(&device, &row, [1usize, 2, 3]).expect("v");

        cache.update(&k, &v).expect("update 1");
        let pending_k = cache.keys_view().eval_async().expect("commit keys");
        let pending_v = cache.values_view().eval_async().expect("commit values");
        cache
            .update(&k, &v)
            .expect("update while prior commit is in flight");
        pending_k.wait().expect("wait k");
        pending_v.wait().expect("wait v");
    }

    /// A fresh cache has no pending update, so `reset` is a no-op success.
    #[test]
    fn reset_on_empty_cache_is_ok() {
        let device = Device::system_default().expect("device");
        let mut cache = KvCache::new(&device, 4, 2, 3, DType::BF16).expect("cache");
        cache.reset().expect("reset on empty cache ok");
        assert_eq!(cache.position(), 0);
    }
}
