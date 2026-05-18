//! Metal device wrapper.
//!
//! Owns the system's default [`MTLDevice`] plus a single shared
//! [`MTLCommandQueue`]. This is the analogue of MLX's
//! `mlx::core::metal::Device` minus the buffer allocator and stream
//! machinery — those belong to the runtime layer.
//!
//! Buffer allocation lives here (not in [`crate::buffer`]) because every
//! allocation needs the underlying [`MTLDevice`]; keeping it on
//! [`Device`] avoids leaking the raw handle for routine work.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice, MTLResourceOptions};

use crate::buffer::Buffer;
use crate::error::{Error, Result};

// `MTLCreateSystemDefaultDevice` requires CoreGraphics at link time.
// Linking once here propagates to every downstream consumer of the crate.
#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {}

/// Owns the system default Metal device and a single command queue.
pub struct Device {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
}

impl Device {
    /// Initialize from `MTLCreateSystemDefaultDevice` and create one
    /// shared command queue.
    pub fn system_default() -> Result<Self> {
        let device = MTLCreateSystemDefaultDevice().ok_or(Error::NoDevice)?;
        let queue = device.newCommandQueue().ok_or(Error::AppleApi {
            context: "MTLDevice::newCommandQueue",
        })?;
        Ok(Self { device, queue })
    }

    /// Allocate a typed shared-storage buffer with `len` elements of `T`.
    ///
    /// Contents are not initialized. Use [`Device::upload`] when the
    /// initial contents matter.
    pub fn alloc_buffer<T>(&self, len: usize) -> Result<Buffer<T>> {
        // ZSTs don't have a sensible Metal-buffer representation; reject
        // at compile time so the generic only monomorphizes for real types.
        const {
            assert!(
                size_of::<T>() > 0,
                "Buffer<T>: zero-sized element types are not supported"
            );
        }
        let elem_size = size_of::<T>();
        let bytes = len
            .checked_mul(elem_size)
            .ok_or(Error::BufferTooLarge { len, elem_size })?;
        let raw = self
            .device
            .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
            .ok_or(Error::BufferAllocation { bytes })?;
        // SAFETY: shared-storage allocation succeeded; `len` matches the byte
        // length we requested, divided by `size_of::<T>()`.
        Ok(unsafe { Buffer::from_raw_parts(raw, len) })
    }

    /// Allocate a buffer and copy `data` into it via the unified-memory
    /// host pointer. On Apple Silicon this is a single host-side write —
    /// no flush or `did_modify_range` call is needed.
    pub fn upload<T: Copy>(&self, data: &[T]) -> Result<Buffer<T>> {
        let mut buf = self.alloc_buffer::<T>(data.len())?;
        // SAFETY: we just allocated this buffer; no dispatch has referenced
        // it yet, so it is safe to write to the host pointer.
        unsafe { buf.as_mut_slice() }.copy_from_slice(data);
        Ok(buf)
    }

    /// Escape hatch to the underlying [`MTLDevice`]. Use this if you need
    /// to reach a Metal API not wrapped here (texture allocation, custom
    /// resource options, etc.); routine work should prefer the typed
    /// methods on [`Device`].
    #[must_use]
    pub fn metal_device(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.device
    }

    /// Escape hatch to the underlying [`MTLCommandQueue`]. See
    /// [`Device::metal_device`].
    #[must_use]
    pub fn metal_queue(&self) -> &ProtocolObject<dyn MTLCommandQueue> {
        &self.queue
    }
}
