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

use std::ptr::NonNull;
use std::sync::Mutex;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandQueue, MTLCounterSamplingPoint, MTLCreateSystemDefaultDevice,
    MTLDevice, MTLFence, MTLResourceOptions, MTLTimestamp,
};

use crate::buffer::Buffer;
use crate::error::{Error, Result};

/// Escape hatch for bisecting sync bugs: `CIDER_PRESS_TRACKED_BUFFERS=1`
/// restores Metal's automatic hazard tracking (correct but slower —
/// the explicit fence/barrier sync stays on regardless).
fn tracked_buffers_requested() -> bool {
    static TRACKED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *TRACKED
        .get_or_init(|| std::env::var_os("CIDER_PRESS_TRACKED_BUFFERS").is_some_and(|v| v == "1"))
}

// `MTLCreateSystemDefaultDevice` requires CoreGraphics at link time.
// Linking once here propagates to every downstream consumer of the crate.
#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {}

/// Owns the system default Metal device and a single command queue.
#[allow(clippy::struct_field_names)]
pub struct Device {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    /// Fence signaled by the most recently closed compute encoder.
    /// Each new encoder waits on it, forming a strict execution chain
    /// across encoders, command buffers, and eval calls — the explicit
    /// replacement for Metal's automatic hazard tracking once buffers
    /// go untracked (redundant until then — buffers are still tracked).
    /// Mutex only for `Sync`: the chain's take/set pairing assumes ONE
    /// thread drives dispatch — concurrent sessions would interleave
    /// take/set and silently corrupt the ordering, lock notwithstanding.
    last_fence: Mutex<Option<Retained<ProtocolObject<dyn MTLFence>>>>,
}

impl Device {
    /// Initialize from `MTLCreateSystemDefaultDevice` and create one
    /// shared command queue.
    pub fn system_default() -> Result<Self> {
        let device = MTLCreateSystemDefaultDevice().ok_or(Error::NoDevice)?;
        let queue = device.newCommandQueue().ok_or(Error::AppleApi {
            context: "MTLDevice::newCommandQueue",
        })?;
        Ok(Self {
            device,
            queue,
            last_fence: Mutex::new(None),
        })
    }

    pub(crate) fn take_last_fence(&self) -> Option<Retained<ProtocolObject<dyn MTLFence>>> {
        self.last_fence.lock().expect("fence mutex poisoned").take()
    }

    pub(crate) fn set_last_fence(&self, fence: Retained<ProtocolObject<dyn MTLFence>>) {
        *self.last_fence.lock().expect("fence mutex poisoned") = Some(fence);
    }

    pub(crate) fn new_fence(&self) -> Result<Retained<ProtocolObject<dyn MTLFence>>> {
        self.device.newFence().ok_or(Error::AppleApi {
            context: "MTLDevice::newFence",
        })
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
        let options = if tracked_buffers_requested() {
            MTLResourceOptions::StorageModeShared
        } else {
            MTLResourceOptions::StorageModeShared | MTLResourceOptions::HazardTrackingModeUntracked
        };
        let raw = self
            .device
            .newBufferWithLength_options(bytes, options)
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

    /// Whether this device can sample GPU counters at compute
    /// stage (encoder) boundaries. Apple Silicon supports this and only
    /// this — never `atDispatchBoundary` (TBDR architecture). The
    /// profiled eval requires it; production eval does not.
    #[must_use]
    pub fn supports_stage_boundary_sampling(&self) -> bool {
        self.device
            .supportsCounterSampling(MTLCounterSamplingPoint::AtStageBoundary)
    }

    /// Nanoseconds per GPU timestamp tick, from a CPU/GPU correlation
    /// sample. GPU counter timestamps are in GPU ticks; multiply a
    /// tick delta by this to get nanoseconds. Sampled once per call —
    /// the ratio is stable, so callers cache it per eval.
    #[must_use]
    pub fn gpu_timestamp_period_ns(&self) -> f64 {
        // Two correlated CPU(ns)/GPU(tick) anchors a few ms apart give
        // ns-per-tick = Δcpu_ns / Δgpu_tick. A single pair can't yield a
        // ratio, so we take two and divide the deltas. CPU timestamps
        // from this API are already in nanoseconds.
        let mut cpu0: MTLTimestamp = 0;
        let mut gpu0: MTLTimestamp = 0;
        let mut cpu1: MTLTimestamp = 0;
        let mut gpu1: MTLTimestamp = 0;
        // SAFETY: all four are valid stack pointers to u64 (MTLTimestamp).
        unsafe {
            self.device
                .sampleTimestamps_gpuTimestamp(NonNull::from(&mut cpu0), NonNull::from(&mut gpu0));
        }
        // Busy a touch so the second GPU tick differs from the first.
        // A no-op command buffer commit gives a clean GPU-clock advance.
        let cmd = self
            .queue
            .commandBuffer()
            .expect("commandBuffer for timestamp calibration");
        cmd.commit();
        cmd.waitUntilCompleted();
        // SAFETY: cpu1/gpu1 are live stack locals; same invariant as the first call.
        unsafe {
            self.device
                .sampleTimestamps_gpuTimestamp(NonNull::from(&mut cpu1), NonNull::from(&mut gpu1));
        }
        // u64 → f64: timestamps are monotonic wall-clock nanoseconds / GPU
        // ticks; the values fit within f64's ~53-bit mantissa for any
        // reasonable uptime, so the precision loss is acceptable.
        #[allow(clippy::cast_precision_loss)]
        let cpu_delta = cpu1.saturating_sub(cpu0) as f64;
        #[allow(clippy::cast_precision_loss)]
        let gpu_ticks = gpu1.saturating_sub(gpu0) as f64;
        if cpu_delta <= 0.0 || gpu_ticks <= 0.0 {
            // Degenerate clock read; fall back to 1.0 (report raw ticks).
            return 1.0;
        }
        cpu_delta / gpu_ticks
    }
}
