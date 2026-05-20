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

use std::sync::{Arc, OnceLock};

use cider_press_kernels as kernels;

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
    /// Cold-start cost is significant (~29 s on first compile, per
    /// Stage-4 spike findings); Metal caches the result to disk
    /// across processes, so subsequent runs are sub-100 ms. Cached
    /// in-process so a long-running session pays the warm cost only
    /// once.
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
