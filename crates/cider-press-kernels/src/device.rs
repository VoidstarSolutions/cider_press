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

use std::collections::{HashMap, HashSet};
use std::ptr::NonNull;
use std::sync::Mutex;

use objc2::rc::Retained;
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2_foundation::{NSString, NSURL};
use objc2_metal::{
    MTLBuffer, MTLCaptureDescriptor, MTLCaptureDestination, MTLCaptureManager, MTLCommandBuffer,
    MTLCommandQueue, MTLCounterSamplingPoint, MTLCreateSystemDefaultDevice, MTLDevice, MTLFence,
    MTLResourceOptions, MTLTimestamp,
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

/// A fence map entry displaced by a publish: `(key, prior_fence)`, where
/// `None` means the key was freshly added. Returned in publish order so a
/// session can undo it in reverse.
type DisplacedOutput = (usize, Option<Retained<ProtocolObject<dyn MTLFence>>>);

/// Owns the system default Metal device and a single command queue.
#[allow(clippy::struct_field_names)]
pub struct Device {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    /// Per-output-buffer fence map (the MLX `prev_ce_outputs_` port): maps a
    /// written buffer's identity key to the fence of the encoder that last
    /// wrote it. A new encoder waits only the fences of buffers it reads OR
    /// overwrites (outputs are waited as inputs — the pool-recycling WAW
    /// guard). The map is the sole cross-encoder ordering mechanism. Mutex only
    /// for `Sync`: the publish/wait pairing assumes ONE thread drives dispatch.
    ///
    /// Unlike MLX (which retires entries from a command-buffer completion
    /// handler), entries are removed only on overwrite or abandon-restore —
    /// never on completion. For a hot pool whose size classes are stable this
    /// self-bounds: every live key is rewritten in submission order. It is NOT
    /// strictly bounded, though: an `MTLBuffer` that is written once and then
    /// dropped (a one-off size class, or a free-list entry evicted by
    /// `BufferPool::set_cap`) leaves a stale entry forever. If Metal later
    /// reuses that pointer for an unrelated allocation, the key collides (ABA)
    /// and `fence_waits` returns the stale — already-signaled — fence,
    /// effectively no wait. Harmless today because keys are buffer pointers held
    /// live by the pool across a generation and eviction is rare; a workload
    /// that churns size classes mid-flight would want MLX-style retirement.
    prev_outputs: Mutex<HashMap<usize, Retained<ProtocolObject<dyn MTLFence>>>>,
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
            prev_outputs: Mutex::new(HashMap::new()),
        })
    }

    /// Fences a closing encoder must wait on: the prior-writer fence of every
    /// `key` it reads or overwrites that is still live in the map, deduped by
    /// fence identity. When `unresolved` is set (an input could not be resolved
    /// to a key), wait the whole current frontier instead — every distinct
    /// fence in the map — conservatively ordering against all still-live prior
    /// writers.
    pub(crate) fn fence_waits(
        &self,
        keys: &[usize],
        unresolved: bool,
    ) -> Vec<Retained<ProtocolObject<dyn MTLFence>>> {
        let map = self.prev_outputs.lock().expect("prev_outputs poisoned");
        let mut seen: HashSet<usize> = HashSet::new();
        let mut out: Vec<Retained<ProtocolObject<dyn MTLFence>>> = Vec::new();
        let push_unique = |fence: &Retained<ProtocolObject<dyn MTLFence>>,
                           out: &mut Vec<Retained<ProtocolObject<dyn MTLFence>>>,
                           seen: &mut HashSet<usize>| {
            let id = Retained::as_ptr(fence).cast::<()>() as usize;
            if seen.insert(id) {
                out.push(fence.clone());
            }
        };
        if unresolved {
            for fence in map.values() {
                push_unique(fence, &mut out, &mut seen);
            }
            return out;
        }
        for &k in keys {
            if let Some(fence) = map.get(&k) {
                push_unique(fence, &mut out, &mut seen);
            }
        }
        out
    }

    /// Record `fence` as the last writer of each `key`. Returns the displaced
    /// prior values (`(key, old)`) in publish order so an abandoned session can
    /// undo them via [`Device::restore_outputs`].
    pub(crate) fn publish_outputs(
        &self,
        keys: &[usize],
        fence: &Retained<ProtocolObject<dyn MTLFence>>,
    ) -> Vec<DisplacedOutput> {
        let mut map = self.prev_outputs.lock().expect("prev_outputs poisoned");
        keys.iter()
            .map(|&k| (k, map.insert(k, fence.clone())))
            .collect()
    }

    /// Undo a [`Device::publish_outputs`] (abandoned, uncommitted session whose
    /// fence will never signal). Restores in REVERSE publish order so the
    /// earliest displaced value wins when a key was published twice in one
    /// session (the profiling path); `None` removes a freshly-added key.
    pub(crate) fn restore_outputs(&self, displaced: Vec<DisplacedOutput>) {
        let mut map = self.prev_outputs.lock().expect("prev_outputs poisoned");
        for (k, old) in displaced.into_iter().rev() {
            match old {
                Some(f) => {
                    map.insert(k, f);
                }
                None => {
                    map.remove(&k);
                }
            }
        }
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
        // Round the Metal allocation to 512 B without changing the Rust-logical
        // length: K-padded qmv reads `in_vec_size` elements from an activation
        // whose logical row is shorter (896 bf16 = 1792 B read as 1024 = 2048 B),
        // and the read must stay within MTLBuffer.length — rounding gives every
        // buffer that capacity. The slack is zeroed below (tail values DO enter
        // the kernel math — see the three-part contract in kernels/qmv.rs).
        // Element counts everywhere derive from Buffer::len, which stays at
        // the requested size.
        // Rounding up can itself overflow (`next_multiple_of` panics in debug,
        // wraps in release) — a wrap would yield `alloc_bytes < bytes`, breaking
        // the slack-zeroing and `MTLBuffer.length >= bytes` contracts below.
        let alloc_bytes = bytes
            .checked_next_multiple_of(512)
            .ok_or(Error::BufferTooLarge { len, elem_size })?;
        let options = if tracked_buffers_requested() {
            MTLResourceOptions::StorageModeShared
        } else {
            MTLResourceOptions::StorageModeShared | MTLResourceOptions::HazardTrackingModeUntracked
        };
        let raw = self
            .device
            .newBufferWithLength_options(alloc_bytes, options)
            .ok_or(Error::BufferAllocation { bytes: alloc_bytes })?;
        // Zero the slack [bytes..alloc_bytes): K-padded qmv reads past the
        // logical row into this region, and the kernel's accum += x*q is
        // NaN-poisonable even against zero weight nibbles (0 * NaN = NaN).
        // Zeroing once at allocation makes the tail 0.0 for the buffer's
        // lifetime: no dispatch or host write ever covers more than the
        // logical byte length, and the pool recycles within exact-size
        // buckets, so the slack is never rewritten. That "no write exceeds
        // the logical byte length" is a codebase-wide write-discipline
        // CONTRACT this invariant depends on, not a mechanism enforced here:
        // a future write past Buffer::len would silently dirty the slack.
        // Storage is always Shared
        // here (the only two option sets above are Shared / Shared+Untracked),
        // so the host `contents()` pointer is valid and a write_bytes suffices
        // — no GPU dispatch has referenced this fresh allocation yet.
        if alloc_bytes > bytes {
            let contents = raw.contents().as_ptr().cast::<u8>();
            // SAFETY: `contents` is the base of a freshly-allocated shared
            // MTLBuffer of `alloc_bytes`; `[bytes..alloc_bytes)` is in bounds
            // (alloc_bytes > bytes), the slack is ≤511 B, and no other handle
            // or dispatch aliases this buffer yet.
            unsafe { contents.add(bytes).write_bytes(0u8, alloc_bytes - bytes) };
        }
        // SAFETY: shared-storage allocation succeeded; the MTLBuffer holds at
        // least `alloc_bytes` >= `bytes`; `len` is the requested element count.
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

    /// Record one GPU frame produced by `f` to a `.gputrace` document at
    /// `path`, for inspection in Xcode / Instruments. The capture targets
    /// this device's command queue.
    ///
    /// Requires `MTL_CAPTURE_ENABLED=1` in the environment — Apple gates
    /// programmatic capture behind it; without it
    /// [`MTLCaptureManager::supportsDestination`] returns `false` and this
    /// returns an error naming the missing variable. `f`'s return value is
    /// passed back unchanged on success, so a fallible `f` rides through as
    /// `Ok(Result<…>)` (callers `??`). `stopCapture` runs even if `f` panics
    /// (an RAII guard), so a panicking closure cannot leave Metal capture
    /// enabled for the rest of the process.
    pub fn capture_gpu_trace<T>(&self, path: &std::path::Path, f: impl FnOnce() -> T) -> Result<T> {
        // RAII so `stopCapture` runs on unwind too — a panicking `f` must not
        // leave capture armed for the rest of the process.
        struct StopGuard<'a>(&'a MTLCaptureManager);
        impl Drop for StopGuard<'_> {
            fn drop(&mut self) {
                self.0.stopCapture();
            }
        }
        // SAFETY: sharedCaptureManager has no preconditions; the returned
        // singleton lives for the process.
        let manager = unsafe { MTLCaptureManager::sharedCaptureManager() };
        if !manager.supportsDestination(MTLCaptureDestination::GPUTraceDocument) {
            // `supportsDestination` is false both when MTL_CAPTURE_ENABLED is
            // unset (the common, fixable case) and when the OS/hardware policy
            // forbids capture (unfixable). Only tell the user to set the env
            // var when it actually is unset, so we don't send them in circles.
            let msg = if std::env::var_os("MTL_CAPTURE_ENABLED").is_none() {
                "capture_gpu_trace: GPUTraceDocument capture unavailable — set \
                 MTL_CAPTURE_ENABLED=1 in the environment before running"
            } else {
                "capture_gpu_trace: GPUTraceDocument capture unsupported despite \
                 MTL_CAPTURE_ENABLED being set (OS/hardware capture policy)"
            };
            return Err(Error::InvalidArgument(msg.into()));
        }
        let url = NSURL::fileURLWithPath(&NSString::from_str(&path.to_string_lossy()));
        let desc = MTLCaptureDescriptor::new();
        // SAFETY: the command queue outlives the descriptor (owned by `self`).
        unsafe {
            let obj: &AnyObject = self.queue.as_ref();
            desc.setCaptureObject(Some(obj));
        }
        desc.setDestination(MTLCaptureDestination::GPUTraceDocument);
        desc.setOutputURL(Some(&url));

        manager
            .startCaptureWithDescriptor_error(&desc)
            .map_err(|e| {
                Error::InvalidArgument(format!("capture_gpu_trace: startCapture failed: {e:?}"))
            })?;
        let _stop = StopGuard(&manager);
        Ok(f())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_gpu_trace_writes_bundle() {
        // Programmatic capture only works with MTL_CAPTURE_ENABLED=1 in the
        // env (Apple gates it). Skip cleanly otherwise — same posture as the
        // counter-sampling skips.
        if std::env::var_os("MTL_CAPTURE_ENABLED").is_none() {
            eprintln!("skipping: MTL_CAPTURE_ENABLED not set");
            return;
        }
        let device = Device::system_default().expect("device");
        // Unique per (pid, atomic seq) so parallel test threads don't race on
        // the directory — see cider_press_test_utils::tempdir.
        let dir = cider_press_test_utils::tempdir("capture-smoke");
        let path = dir.join("prefill.gputrace");

        let ran = device
            .capture_gpu_trace(&path, || {
                let buf = device.upload(&[1.0f32, 2.0, 3.0]).expect("upload");
                let commands = device.commands().expect("commands");
                commands.commit_and_wait().expect("commit");
                drop(buf);
                42u32
            })
            .expect("capture_gpu_trace");

        assert_eq!(ran, 42);
        assert!(path.exists(), "gputrace bundle not written: {path:?}");
    }

    #[test]
    fn fence_map_publish_and_wait_dedup() {
        let device = Device::system_default().expect("device");
        let f1 = device.new_fence().expect("fence");
        // Publish one fence under two keys; a wait over both must dedup to one.
        let displaced = device.publish_outputs(&[10, 20], &f1);
        assert_eq!(displaced, vec![(10usize, None), (20usize, None)]);
        let waits = device.fence_waits(&[20, 10], false);
        assert_eq!(waits.len(), 1, "same fence under two keys must dedup");
        assert_eq!(Retained::as_ptr(&waits[0]), Retained::as_ptr(&f1));
        // A key never published yields no wait.
        assert!(device.fence_waits(&[99], false).is_empty());
    }

    #[test]
    fn fence_map_unresolved_waits_whole_frontier() {
        let device = Device::system_default().expect("device");
        let f1 = device.new_fence().expect("f1");
        let f2 = device.new_fence().expect("f2");
        device.publish_outputs(&[1], &f1);
        device.publish_outputs(&[2], &f2);
        // Unresolved input => wait every distinct fence in the map, regardless of keys.
        let waits = device.fence_waits(&[], true);
        assert_eq!(waits.len(), 2, "unresolved must wait the whole frontier");
    }

    #[test]
    fn fence_map_restore_roundtrip() {
        let device = Device::system_default().expect("device");
        let f1 = device.new_fence().expect("f1");
        let f2 = device.new_fence().expect("f2");
        device.publish_outputs(&[7], &f1);
        // Overwrite key 7 (f1 -> f2) and add a fresh key 8, then undo both.
        let d1 = device.publish_outputs(&[7], &f2);
        let d2 = device.publish_outputs(&[8], &f1);
        assert_eq!(d1, vec![(7usize, Some(f1.clone()))]);
        assert_eq!(d2, vec![(8usize, None)]);
        device.restore_outputs(d2);
        device.restore_outputs(d1);
        // Key 7 is back to f1; key 8 is gone.
        let w7 = device.fence_waits(&[7], false);
        assert_eq!(w7.len(), 1);
        assert_eq!(Retained::as_ptr(&w7[0]), Retained::as_ptr(&f1));
        assert!(device.fence_waits(&[8], false).is_empty());
    }
}
