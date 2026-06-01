//! GPU device handle.
//!
//! [`Device`] is the runtime-layer wrapper around
//! [`cider_press_kernels::Device`]. It exists so that tensors and ops
//! can carry a cheap-to-clone, identity-comparable device reference
//! without exposing the kernels-layer types directly. Cloning a
//! [`Device`] bumps an [`Arc`]; two clones of the same handle compare
//! equal via [`Arc::ptr_eq`], which the op layer uses to assert all
//! inputs share a device.
//!
//! The handle also owns lazily-initialized JIT'd kernel libraries
//! (e.g. the vendored MLX `copy.metal`) that the dispatcher needs at
//! eval time. JIT cost is amortized across every `eval()` on this
//! device, not paid per-dispatch.
//!
//! [`Device::shared`] returns a lazily-initialized process-global
//! default device for the common single-device case. Construct an
//! explicit handle via [`Device::system_default`] when you need
//! control over initialization (e.g. surfacing the [`Result`] in a
//! test, or to fail fast at startup).

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use cider_press_kernels as kernels;
use cider_press_kernels::Buffer;

use crate::error::Result;

struct DeviceInner {
    kernels: kernels::Device,
    /// JIT'd MLX `copy.metal` library. Populated on first `eval()`
    /// that needs it; subsequent calls reuse the cached library.
    copy_library: OnceLock<kernels::KernelLibrary>,
    /// JIT'd MLX `quantized.metal` library (cold-start ~29 s, warm
    /// ~100 ms thanks to Metal's cross-process compiled-library
    /// cache). Populated on first qmv/qmm dispatch.
    quantized_library: OnceLock<kernels::KernelLibrary>,
    /// JIT'd MLX `binary.metal` library. Populated on first binary
    /// op dispatch (Add / Mul today; the rest of the family lands as
    /// the runtime asks for it).
    binary_library: OnceLock<kernels::KernelLibrary>,
    /// JIT'd MLX `unary.metal` library. Populated on first unary op
    /// dispatch (Square / Rsqrt today).
    unary_library: OnceLock<kernels::KernelLibrary>,
    /// JIT'd MLX `reduce.metal` library. Cold-start cost is sizeable
    /// (the flattened source is the largest after quantized.metal),
    /// so callers see a one-time pause on first reduce dispatch.
    /// Subsequent calls return the cached library.
    reduce_library: OnceLock<kernels::KernelLibrary>,
    /// JIT'd MLX `rope.metal` library. Populated on first
    /// [`crate::Tensor::rope`] eval; subsequent ropes hit the cached
    /// library and look up the per-specialization pipeline from the
    /// library's own pipeline-state cache.
    rope_library: OnceLock<kernels::KernelLibrary>,
    /// JIT'd MLX `softmax.metal` library. Populated on first
    /// [`crate::Tensor::softmax`] eval. Variant selection (default vs
    /// `_precise_`, block vs looped) is by kernel name, so the
    /// library's pipeline-state cache picks up each consumer's
    /// specialization automatically.
    softmax_library: OnceLock<kernels::KernelLibrary>,
    /// JIT-assembled gather libraries, one entry per
    /// [`kernels::kernels::gather::Instantiation`]. Keyed by the
    /// instantiation's kernel name (which uniquely identifies the
    /// `(T, IdxT, NIDX, IDX_NDIM, LocT)` tuple). Unlike the other
    /// libraries this is per-instantiation because MLX has no
    /// precompiled gather.metal — every `(T, IdxT, ...)` combination
    /// is a separate compiled library, matching MLX's own
    /// `d.get_library(lib_name, builder)` pattern.
    gather_libraries: Mutex<HashMap<String, Arc<kernels::KernelLibrary>>>,
    /// JIT'd cider-press matmul library (our own kernel, not
    /// MLX-derived — see `crates/cider-press-kernels/src/kernels/matmul.metal`).
    /// Populated on first [`crate::Tensor::matmul`] eval.
    matmul_library: OnceLock<kernels::KernelLibrary>,
    /// JIT'd MLX `sdpa_vector` library. Populated on first
    /// [`crate::Tensor::sdpa`] eval; subsequent decodes reuse the
    /// cached library and its per-`head_dim` pipeline-state cache.
    sdpa_vector_library: OnceLock<kernels::KernelLibrary>,
    /// JIT'd MLX `arg_reduce.metal` library. Populated on first
    /// [`crate::Tensor::argmax`] eval; subsequent calls reuse the
    /// cached library.
    arg_reduce_library: OnceLock<kernels::KernelLibrary>,
    /// Cross-token scratch recycling. See [`crate::buffer_pool`].
    pool: Arc<Mutex<crate::buffer_pool::BufferPool>>,
}

/// Refcounted handle to the system's default Metal device.
///
/// Cheap to clone (bumps an [`Arc`]). Identity is by pointer: two
/// [`Device`]s compare equal iff their inner [`Arc`]s point at the
/// same kernels-layer device.
#[derive(Clone)]
pub struct Device {
    inner: Arc<DeviceInner>,
}

impl Device {
    /// Construct a new handle by initializing
    /// [`kernels::Device::system_default`]. Each call allocates a new
    /// underlying device and command queue; use [`Device::shared`] for
    /// the common case of a single process-wide device.
    pub fn system_default() -> Result<Self> {
        Ok(Self {
            inner: Arc::new(DeviceInner {
                kernels: kernels::Device::system_default()?,
                copy_library: OnceLock::new(),
                quantized_library: OnceLock::new(),
                binary_library: OnceLock::new(),
                unary_library: OnceLock::new(),
                reduce_library: OnceLock::new(),
                rope_library: OnceLock::new(),
                softmax_library: OnceLock::new(),
                gather_libraries: Mutex::new(HashMap::new()),
                matmul_library: OnceLock::new(),
                sdpa_vector_library: OnceLock::new(),
                arg_reduce_library: OnceLock::new(),
                pool: Arc::new(Mutex::new(crate::buffer_pool::BufferPool::new(
                    crate::buffer_pool::DEFAULT_POOL_CAP_BYTES,
                ))),
            }),
        })
    }

    /// Lazily-initialized process-global default device.
    ///
    /// The first successful caller initializes the device; subsequent
    /// callers receive cheap clones of the same handle. If
    /// initialization failed previously, each call re-attempts — the
    /// failure modes for `MTLCreateSystemDefaultDevice` are not
    /// expected to flap, but caching a `Result` would require `Error:
    /// Clone` which we don't want to commit to crate-wide.
    pub fn shared() -> Result<Self> {
        static SHARED: OnceLock<Arc<DeviceInner>> = OnceLock::new();
        if let Some(inner) = SHARED.get() {
            return Ok(Self {
                inner: inner.clone(),
            });
        }
        let fresh = Arc::new(DeviceInner {
            kernels: kernels::Device::system_default()?,
            copy_library: OnceLock::new(),
            quantized_library: OnceLock::new(),
            binary_library: OnceLock::new(),
            unary_library: OnceLock::new(),
            reduce_library: OnceLock::new(),
            rope_library: OnceLock::new(),
            softmax_library: OnceLock::new(),
            gather_libraries: Mutex::new(HashMap::new()),
            matmul_library: OnceLock::new(),
            sdpa_vector_library: OnceLock::new(),
            arg_reduce_library: OnceLock::new(),
            pool: Arc::new(Mutex::new(crate::buffer_pool::BufferPool::new(
                crate::buffer_pool::DEFAULT_POOL_CAP_BYTES,
            ))),
        });
        let stored = SHARED.get_or_init(|| fresh);
        Ok(Self {
            inner: stored.clone(),
        })
    }

    /// Whether two [`Device`] handles point at the same underlying
    /// kernels-layer device. Cheap pointer comparison.
    #[must_use]
    pub fn ptr_eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    /// Access the underlying kernels-layer device. Crate-private: the
    /// runtime uses this to thread allocation and dispatch through;
    /// callers outside the crate should stay at the [`Device`] level.
    pub(crate) fn kernels(&self) -> &kernels::Device {
        &self.inner.kernels
    }

    /// Allocate a scratch buffer of `bytes`, drawing from the recycling
    /// pool when a freed buffer of the same requested size is available
    /// and falling back to a fresh Metal allocation otherwise.
    pub(crate) fn alloc_pooled(&self, bytes: usize) -> Result<Buffer<u8>> {
        if let Some(buf) = self
            .inner
            .pool
            .lock()
            .expect("buffer pool mutex poisoned")
            .take(bytes)
        {
            return Ok(buf);
        }
        Ok(self.inner.kernels.alloc_buffer::<u8>(bytes)?)
    }

    /// Wrap an op-output buffer as a pool-owned [`PooledBuffer`] that
    /// returns to this device's pool on drop. `bytes` is the requested
    /// length it was allocated at (its free-list key).
    pub(crate) fn pooled(
        &self,
        buffer: Buffer<u8>,
        bytes: usize,
    ) -> crate::buffer_pool::PooledBuffer {
        crate::buffer_pool::PooledBuffer::pooled(buffer, bytes, Arc::downgrade(&self.inner.pool))
    }

    /// Snapshot of the pool's hit/miss/byte counters.
    #[must_use]
    pub fn pool_stats(&self) -> crate::buffer_pool::PoolStats {
        self.inner
            .pool
            .lock()
            .expect("buffer pool mutex poisoned")
            .stats()
    }

    /// Override the pool's retained-byte ceiling. Lowering it below the
    /// currently-pooled bytes evicts free-list entries until the pool is
    /// back under the new ceiling, so a shrink frees memory immediately.
    pub fn set_pool_cap(&self, cap_bytes: usize) {
        self.inner
            .pool
            .lock()
            .expect("buffer pool mutex poisoned")
            .set_cap(cap_bytes);
    }

    /// Test-only accessor for the underlying kernels device (so
    /// `buffer_pool` unit tests can allocate raw buffers).
    #[cfg(test)]
    pub(crate) fn kernels_for_test(&self) -> &kernels::Device {
        &self.inner.kernels
    }

    /// Whether this device supports GPU stage-boundary counter sampling,
    /// the prerequisite for [`Tensor::profiled_eval`](crate::Tensor::profiled_eval).
    #[must_use]
    pub fn supports_stage_boundary_sampling(&self) -> bool {
        self.kernels().supports_stage_boundary_sampling()
    }

    /// Lazily JIT-compile and cache MLX's `copy.metal` library.
    ///
    /// First call pays the JIT cost (sub-100 ms warm — Metal caches
    /// the compiled library to disk across processes). Subsequent
    /// calls return the cached library.
    pub(crate) fn copy_library(&self) -> Result<&kernels::KernelLibrary> {
        if let Some(lib) = self.inner.copy_library.get() {
            return Ok(lib);
        }
        let lib = kernels::KernelLibrary::copy(&self.inner.kernels)?;
        // Race-safe: if another thread already populated the slot, the
        // value we just built is dropped and the stored one is returned.
        Ok(self.inner.copy_library.get_or_init(|| lib))
    }

    /// Lazily JIT-compile and cache MLX's `quantized.metal` library.
    ///
    /// Cold-start cost is significant (~29 s on first compile); Metal
    /// caches the result to disk across processes, so subsequent runs are
    /// sub-100 ms. Cached in-process so a long-running session pays the
    /// warm cost only once.
    pub(crate) fn quantized_library(&self) -> Result<&kernels::KernelLibrary> {
        if let Some(lib) = self.inner.quantized_library.get() {
            return Ok(lib);
        }
        let lib = kernels::KernelLibrary::quantized(&self.inner.kernels)?;
        Ok(self.inner.quantized_library.get_or_init(|| lib))
    }

    /// Lazily JIT-compile and cache MLX's `binary.metal` library.
    /// First call pays the JIT cost; subsequent calls return the cached
    /// library. See [`Device::copy_library`] for the cache semantics.
    pub(crate) fn binary_library(&self) -> Result<&kernels::KernelLibrary> {
        if let Some(lib) = self.inner.binary_library.get() {
            return Ok(lib);
        }
        let lib = kernels::KernelLibrary::binary(&self.inner.kernels)?;
        Ok(self.inner.binary_library.get_or_init(|| lib))
    }

    /// Lazily JIT-compile and cache MLX's `unary.metal` library.
    pub(crate) fn unary_library(&self) -> Result<&kernels::KernelLibrary> {
        if let Some(lib) = self.inner.unary_library.get() {
            return Ok(lib);
        }
        let lib = kernels::KernelLibrary::unary(&self.inner.kernels)?;
        Ok(self.inner.unary_library.get_or_init(|| lib))
    }

    /// Lazily JIT-compile and cache MLX's `reduce.metal` library.
    /// First-call cost is sizeable (~several seconds cold on the dev
    /// machine, sub-100 ms warm thanks to Metal's cross-process JIT
    /// cache).
    pub(crate) fn reduce_library(&self) -> Result<&kernels::KernelLibrary> {
        if let Some(lib) = self.inner.reduce_library.get() {
            return Ok(lib);
        }
        let lib = kernels::KernelLibrary::reduce(&self.inner.kernels)?;
        Ok(self.inner.reduce_library.get_or_init(|| lib))
    }

    /// Lazily JIT-compile and cache MLX's `rope.metal` library. Hosts
    /// every rope instantiation; the per-specialization
    /// `[[function_constant]]` pipelines live in the library's
    /// internal pipeline cache.
    pub(crate) fn rope_library(&self) -> Result<&kernels::KernelLibrary> {
        if let Some(lib) = self.inner.rope_library.get() {
            return Ok(lib);
        }
        let lib = kernels::KernelLibrary::rope(&self.inner.kernels)?;
        Ok(self.inner.rope_library.get_or_init(|| lib))
    }

    /// Lazily JIT-compile and cache MLX's `softmax.metal` library.
    /// Hosts both block / looped × default / precise variants; the
    /// per-variant pipelines live in the library's internal cache.
    pub(crate) fn softmax_library(&self) -> Result<&kernels::KernelLibrary> {
        if let Some(lib) = self.inner.softmax_library.get() {
            return Ok(lib);
        }
        let lib = kernels::KernelLibrary::softmax(&self.inner.kernels)?;
        Ok(self.inner.softmax_library.get_or_init(|| lib))
    }

    /// Lazily JIT-compile and cache a gather library for one
    /// instantiation tuple. The kernel name is the cache key, so each
    /// unique `(T, IdxT, NIDX, IDX_NDIM, LocT)` combination is a
    /// separate compiled library — mirroring MLX's
    /// `d.get_library(lib_name, builder)` pattern in
    /// `backend/metal/indexing.cpp`.
    pub(crate) fn gather_library(
        &self,
        inst: &kernels::kernels::gather::Instantiation,
    ) -> Result<Arc<kernels::KernelLibrary>> {
        let kernel_name = inst.kernel_name();
        let mut libs = self
            .inner
            .gather_libraries
            .lock()
            .expect("gather_libraries mutex poisoned");
        if let Some(lib) = libs.get(&kernel_name) {
            return Ok(lib.clone());
        }
        let source = kernels::kernels::gather::make_source(inst);
        let lib = Arc::new(kernels::KernelLibrary::from_source(
            &self.inner.kernels,
            &source,
        )?);
        libs.insert(kernel_name, lib.clone());
        Ok(lib)
    }

    /// Lazily JIT-compile and cache cider-press's own bf16 matmul
    /// library (`kernels/matmul.metal`). See [`Device::copy_library`]
    /// for cache semantics.
    pub(crate) fn matmul_library(&self) -> Result<&kernels::KernelLibrary> {
        if let Some(lib) = self.inner.matmul_library.get() {
            return Ok(lib);
        }
        let lib = kernels::KernelLibrary::matmul(&self.inner.kernels)?;
        Ok(self.inner.matmul_library.get_or_init(|| lib))
    }

    /// Lazily JIT-compile and cache MLX's `sdpa_vector` library. See
    /// [`Device::copy_library`] for cache semantics.
    pub(crate) fn sdpa_vector_library(&self) -> Result<&kernels::KernelLibrary> {
        if let Some(lib) = self.inner.sdpa_vector_library.get() {
            return Ok(lib);
        }
        let lib = kernels::KernelLibrary::sdpa_vector(&self.inner.kernels)?;
        Ok(self.inner.sdpa_vector_library.get_or_init(|| lib))
    }

    /// Lazily JIT-compile and cache MLX's `arg_reduce.metal` library.
    pub(crate) fn arg_reduce_library(&self) -> Result<&kernels::KernelLibrary> {
        if let Some(lib) = self.inner.arg_reduce_library.get() {
            return Ok(lib);
        }
        let lib = kernels::KernelLibrary::arg_reduce(&self.inner.kernels)?;
        Ok(self.inner.arg_reduce_library.get_or_init(|| lib))
    }
}

impl core::fmt::Debug for Device {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Device")
            .field("addr", &Arc::as_ptr(&self.inner))
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_returns_same_handle() {
        let a = Device::shared().expect("system default device");
        let b = Device::shared().expect("system default device");
        assert!(a.ptr_eq(&b));
    }

    #[test]
    fn system_default_is_independent_of_shared() {
        let s = Device::shared().expect("shared device");
        let d = Device::system_default().expect("explicit device");
        assert!(!s.ptr_eq(&d));
    }

    #[test]
    fn clone_preserves_identity() {
        let a = Device::shared().expect("shared device");
        let b = a.clone();
        assert!(a.ptr_eq(&b));
    }

    #[test]
    fn copy_library_is_cached() {
        let d = Device::shared().expect("shared device");
        let a = std::ptr::from_ref(d.copy_library().expect("copy library"));
        let b = std::ptr::from_ref(d.copy_library().expect("copy library"));
        assert_eq!(a, b);
    }
}
