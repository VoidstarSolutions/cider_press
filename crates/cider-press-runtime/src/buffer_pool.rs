//! Cross-token buffer recycling.
//!
//! Decode allocates one fresh `MTLBuffer` per op output, ~1000 per
//! token, on the serialized command-buffer encode path. [`BufferPool`]
//! is a size-keyed free-list: an op output's buffer is returned here
//! when its tensor node drops (between tokens) and handed back out to
//! the next token's eval, so steady-state decode stops calling
//! `newBufferWithLength:`. A byte cap bounds retained memory.
//!
//! See `docs/superpowers/specs/2026-05-31-buffer-pool-design.md`.

use std::collections::HashMap;
use std::ops::Deref;
use std::sync::{Mutex, Weak};

use cider_press_kernels::Buffer;

/// Default ceiling on bytes held in the free-list. Sized to hold roughly
/// one decode token's simultaneously-live scratch (the +32% peak-RSS
/// regression added ~290 MiB) with headroom. Tunable via
/// [`crate::Device::set_pool_cap`]; refined by measurement (see plan
/// Task 6).
pub(crate) const DEFAULT_POOL_CAP_BYTES: usize = 384 << 20;

/// Size-keyed free-list of reusable shared-storage buffers.
///
/// Keyed by **requested** byte length (the count cider-press asked Metal
/// for), not the page-rounded allocation size: a recycled buffer of the
/// same requested length has an identical rounded allocation, so exact
/// keying makes reuse a drop-in replacement.
// Not yet wired into eval; the next commit threads these through LeafStorage.
#[allow(dead_code)]
pub(crate) struct BufferPool {
    free: HashMap<usize, Vec<Buffer<u8>>>,
    pooled_bytes: usize,
    cap_bytes: usize,
    hits: u64,
    misses: u64,
    high_water_bytes: usize,
}

/// Snapshot of pool counters, for the bench harness and reuse tests.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PoolStats {
    pub hits: u64,
    pub misses: u64,
    pub pooled_bytes: usize,
    pub high_water_bytes: usize,
}

impl BufferPool {
    pub(crate) fn new(cap_bytes: usize) -> Self {
        Self {
            free: HashMap::new(),
            pooled_bytes: 0,
            cap_bytes,
            hits: 0,
            misses: 0,
            high_water_bytes: 0,
        }
    }

    /// Pop a recycled buffer of exactly `bytes`, or `None` on a miss.
    pub(crate) fn take(&mut self, bytes: usize) -> Option<Buffer<u8>> {
        if let Some(slot) = self.free.get_mut(&bytes) {
            if let Some(buf) = slot.pop() {
                self.pooled_bytes -= bytes;
                self.hits += 1;
                return Some(buf);
            }
        }
        self.misses += 1;
        None
    }

    /// Return `buf` (whose requested length is `bytes`) to the free-list,
    /// unless doing so would exceed the cap — in which case `buf` is
    /// dropped here, releasing the `MTLBuffer`.
    pub(crate) fn give(&mut self, buf: Buffer<u8>, bytes: usize) {
        if self.pooled_bytes + bytes > self.cap_bytes {
            return; // over cap: let `buf` drop and release.
        }
        self.free.entry(bytes).or_default().push(buf);
        self.pooled_bytes += bytes;
        self.high_water_bytes = self.high_water_bytes.max(self.pooled_bytes);
    }

    pub(crate) fn set_cap(&mut self, cap_bytes: usize) {
        self.cap_bytes = cap_bytes;
    }

    pub(crate) fn stats(&self) -> PoolStats {
        PoolStats {
            hits: self.hits,
            misses: self.misses,
            pooled_bytes: self.pooled_bytes,
            high_water_bytes: self.high_water_bytes,
        }
    }
}

/// A `Buffer<u8>` that, when pool-minted, returns itself to its
/// [`BufferPool`] on drop instead of releasing the `MTLBuffer`.
///
/// Sole-ownership invariant: a pool-minted `PooledBuffer` is the unique
/// long-lived owner of its `MTLBuffer`. Transient typed views from
/// `Buffer::reinterpret_as` / `clone_handle` (made during dispatch
/// encoding) are dropped before the owning tensor node drops, and the
/// drop happens after `commit_and_wait`, so the GPU is done and no view
/// is outstanding when the buffer returns to the pool. The `SliceUpdate`
/// slab and host/constant leaves are minted [`PooledBuffer::unpooled`]
/// (`pool: None`) and therefore never return — they are not pool-owned
/// (the slab is co-owned by the [`KvCache`]; recycling it would corrupt
/// cached K/V).
// Not yet wired into eval; the next commit threads these through LeafStorage.
#[allow(dead_code)]
pub(crate) struct PooledBuffer {
    /// `Some` until `Drop` takes it. `Deref` unwraps it.
    buffer: Option<Buffer<u8>>,
    /// Requested byte length this buffer was minted at — its free-list key.
    bytes: usize,
    /// `Some` ⇒ return to this pool on drop; `None` ⇒ release normally.
    pool: Option<Weak<Mutex<BufferPool>>>,
}

#[allow(dead_code)]
impl PooledBuffer {
    /// A pool-minted buffer that returns to `pool` on drop.
    pub(crate) fn pooled(buffer: Buffer<u8>, bytes: usize, pool: Weak<Mutex<BufferPool>>) -> Self {
        Self {
            buffer: Some(buffer),
            bytes,
            pool: Some(pool),
        }
    }

    /// A buffer that is *not* pool-owned: releases normally on drop.
    /// Used for the `SliceUpdate` slab clone and host/constant leaves.
    pub(crate) fn unpooled(buffer: Buffer<u8>) -> Self {
        let bytes = buffer.byte_len();
        Self {
            buffer: Some(buffer),
            bytes,
            pool: None,
        }
    }
}

impl Deref for PooledBuffer {
    type Target = Buffer<u8>;

    fn deref(&self) -> &Buffer<u8> {
        self.buffer
            .as_ref()
            .expect("PooledBuffer::buffer is Some until Drop")
    }
}

impl Drop for PooledBuffer {
    fn drop(&mut self) {
        let Some(buffer) = self.buffer.take() else {
            return;
        };
        let Some(weak) = self.pool.as_ref() else {
            return; // unpooled: `buffer` drops here, releasing the MTLBuffer.
        };
        if let Some(pool) = weak.upgrade() {
            pool.lock()
                .expect("buffer pool mutex poisoned")
                .give(buffer, self.bytes);
        }
        // else: pool already torn down; `buffer` drops here.
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::Device;

    fn buf(device: &Device, bytes: usize) -> Buffer<u8> {
        device
            .kernels_for_test()
            .alloc_buffer::<u8>(bytes)
            .expect("alloc")
    }

    #[test]
    fn take_on_empty_pool_is_miss() {
        let mut pool = BufferPool::new(DEFAULT_POOL_CAP_BYTES);
        assert!(pool.take(1024).is_none());
        assert_eq!(pool.stats().misses, 1);
        assert_eq!(pool.stats().hits, 0);
    }

    #[test]
    fn give_then_take_same_size_hits() {
        let device = Device::system_default().expect("device");
        let mut pool = BufferPool::new(DEFAULT_POOL_CAP_BYTES);
        pool.give(buf(&device, 1024), 1024);
        assert_eq!(pool.stats().pooled_bytes, 1024);
        assert!(pool.take(1024).is_some());
        assert_eq!(pool.stats().hits, 1);
        assert_eq!(pool.stats().pooled_bytes, 0);
    }

    #[test]
    fn take_on_present_but_empty_slot_is_miss() {
        // After draining a size's only buffer, the slot Vec lingers empty;
        // a further take of that size must still count as a miss.
        let device = Device::system_default().expect("device");
        let mut pool = BufferPool::new(DEFAULT_POOL_CAP_BYTES);
        pool.give(buf(&device, 1024), 1024);
        assert!(pool.take(1024).is_some());
        assert!(pool.take(1024).is_none());
        assert_eq!(pool.stats().misses, 1);
    }

    #[test]
    fn high_water_holds_after_take_drains_pool() {
        let device = Device::system_default().expect("device");
        let mut pool = BufferPool::new(DEFAULT_POOL_CAP_BYTES);
        pool.give(buf(&device, 1024), 1024);
        assert!(pool.take(1024).is_some());
        assert_eq!(pool.stats().pooled_bytes, 0);
        assert_eq!(pool.stats().high_water_bytes, 1024);
    }

    #[test]
    fn keying_is_exact_size() {
        let device = Device::system_default().expect("device");
        let mut pool = BufferPool::new(DEFAULT_POOL_CAP_BYTES);
        pool.give(buf(&device, 1024), 1024);
        assert!(pool.take(2048).is_none());
        assert_eq!(pool.stats().pooled_bytes, 1024);
    }

    #[test]
    fn give_over_cap_drops_buffer() {
        let device = Device::system_default().expect("device");
        let mut pool = BufferPool::new(1024);
        pool.give(buf(&device, 1024), 1024);
        pool.give(buf(&device, 1024), 1024);
        assert_eq!(pool.stats().pooled_bytes, 1024);
        assert_eq!(pool.stats().high_water_bytes, 1024);
    }

    #[test]
    fn unpooled_drop_does_not_return_to_pool() {
        let device = Device::system_default().expect("device");
        let pool = Arc::new(Mutex::new(BufferPool::new(DEFAULT_POOL_CAP_BYTES)));
        let pb = PooledBuffer::unpooled(buf(&device, 512));
        drop(pb);
        assert_eq!(pool.lock().unwrap().stats().pooled_bytes, 0);
    }

    #[test]
    fn pooled_drop_returns_to_pool() {
        let device = Device::system_default().expect("device");
        let pool = Arc::new(Mutex::new(BufferPool::new(DEFAULT_POOL_CAP_BYTES)));
        let pb = PooledBuffer::pooled(buf(&device, 512), 512, Arc::downgrade(&pool));
        drop(pb);
        assert_eq!(pool.lock().unwrap().stats().pooled_bytes, 512);
    }

    #[test]
    fn pooled_drop_with_dead_pool_does_not_panic() {
        let device = Device::system_default().expect("device");
        let pool = Arc::new(Mutex::new(BufferPool::new(DEFAULT_POOL_CAP_BYTES)));
        let pb = PooledBuffer::pooled(buf(&device, 512), 512, Arc::downgrade(&pool));
        drop(pool);
        drop(pb);
    }

    #[test]
    fn deref_exposes_inner_buffer_len() {
        let device = Device::system_default().expect("device");
        let pb = PooledBuffer::unpooled(buf(&device, 256));
        assert_eq!(pb.byte_len(), 256);
    }
}
