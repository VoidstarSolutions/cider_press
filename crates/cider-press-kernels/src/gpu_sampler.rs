//! GPU counter-sampling state for stage-boundary timing.
//!
//! Apple Silicon GPUs sample counters only at compute stage (encoder)
//! boundaries, so per-dispatch GPU timing requires one sampled encoder
//! per dispatch. [`GpuSampler`] owns the [`MTLCounterSampleBuffer`] all
//! those encoders write into, hands out (start, end) sample-index pairs,
//! records a `(label, start, end)` segment per op, and resolves the
//! buffer into per-segment GPU-tick intervals after the command buffer
//! completes. This type carries no policy — the runtime layer maps op
//! kinds to labels and buckets the resolved intervals.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSRange;
use objc2_metal::{
    MTLCommonCounterSetTimestamp, MTLCounterResultTimestamp, MTLCounterSampleBuffer,
    MTLCounterSampleBufferDescriptor, MTLCounterSet, MTLDevice, MTLStorageMode,
};

use crate::device::Device;
use crate::error::{Error, Result};

/// One resolved op interval: a label plus its GPU execution window in
/// ticks. Convert to ns with [`Device::gpu_timestamp_period_ns`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuSegment {
    /// Caller-supplied static label (e.g. the op-kind name).
    pub label: &'static str,
    /// GPU tick at encoder start.
    pub start_tick: u64,
    /// GPU tick at encoder end.
    pub end_tick: u64,
}

impl GpuSegment {
    /// Tick span (`end - start`), saturating at 0 for any inverted or
    /// error-valued pair.
    #[must_use]
    pub fn ticks(&self) -> u64 {
        self.end_tick.saturating_sub(self.start_tick)
    }
}

/// Owns the counter sample buffer and the per-op segment records.
pub struct GpuSampler {
    buffer: Retained<ProtocolObject<dyn MTLCounterSampleBuffer>>,
    capacity_samples: usize,
    cursor: usize,
    segments: Vec<(&'static str, usize, usize)>,
}

impl GpuSampler {
    /// Allocate a sample buffer sized for `op_capacity` ops (two
    /// timestamps each). Errors if the device lacks a timestamp counter
    /// set or rejects the descriptor.
    pub fn new(device: &Device, op_capacity: usize) -> Result<Self> {
        let sample_count = op_capacity.checked_mul(2).ok_or(Error::AppleApi {
            context: "GpuSampler: op_capacity * 2 overflow",
        })?;
        let counter_set = timestamp_counter_set(device)?;

        let descriptor = MTLCounterSampleBufferDescriptor::new();
        descriptor.setCounterSet(Some(&counter_set));
        descriptor.setStorageMode(MTLStorageMode::Shared);
        // SAFETY: `setSampleCount` is a plain ObjC setter; any usize is a
        // valid argument. An out-of-range count is caught by the device
        // below (newCounterSampleBufferWithDescriptor returns NSError),
        // not by this call.
        unsafe { descriptor.setSampleCount(sample_count) };

        let buffer = device
            .metal_device()
            .newCounterSampleBufferWithDescriptor_error(&descriptor)
            .map_err(|_| Error::AppleApi {
                context: "MTLDevice::newCounterSampleBufferWithDescriptor",
            })?;

        Ok(Self {
            buffer,
            capacity_samples: sample_count,
            cursor: 0,
            segments: Vec::with_capacity(op_capacity),
        })
    }

    /// Total timestamp slots (2 × op capacity).
    #[must_use]
    pub fn sample_capacity(&self) -> usize {
        self.capacity_samples
    }

    /// Segments recorded so far.
    #[must_use]
    pub fn recorded_segments(&self) -> &[(&'static str, usize, usize)] {
        &self.segments
    }

    /// The underlying sample buffer, for attaching to a compute pass
    /// descriptor. Consumed by the profiled-eval path in a later task.
    #[allow(dead_code)]
    pub(crate) fn buffer(&self) -> &ProtocolObject<dyn MTLCounterSampleBuffer> {
        &self.buffer
    }

    /// Reserve the next (start, end) index pair for an op labelled
    /// `label`, recording the segment. Returns `(start_idx, end_idx)` to
    /// bind into a compute pass descriptor. Errors if capacity is
    /// exhausted. Consumed by the profiled-eval path in a later task.
    #[allow(dead_code)]
    pub(crate) fn reserve(&mut self, label: &'static str) -> Result<(usize, usize)> {
        let start = self.cursor;
        let end = start + 1;
        // `end` is an index; `capacity_samples` is the exclusive bound.
        if end >= self.capacity_samples {
            return Err(Error::AppleApi {
                context: "GpuSampler: sample capacity exhausted",
            });
        }
        self.cursor = end + 1;
        self.segments.push((label, start, end));
        Ok((start, end))
    }

    /// Resolve the recorded range into per-segment tick intervals. Call
    /// after the command buffer has completed. Reads `2 * segments.len()`
    /// timestamps and pairs them back to their labels.
    pub fn resolve(&self) -> Result<Vec<GpuSegment>> {
        if self.segments.is_empty() {
            return Ok(Vec::new());
        }
        let range = NSRange::new(0, self.cursor);
        // SAFETY: `buffer` is a valid `Retained<>` ObjC object; the range
        // [0, cursor) covers only indices we reserved (cursor <= capacity_samples).
        let data = unsafe { self.buffer.resolveCounterRange(range) }.ok_or(Error::AppleApi {
            context: "MTLCounterSampleBuffer::resolveCounterRange returned nil",
        })?;
        let byte_len = data.len();
        let want = self.cursor * std::mem::size_of::<MTLCounterResultTimestamp>();
        if byte_len < want {
            return Err(Error::AppleApi {
                context: "GpuSampler: resolved counter data shorter than reserved range",
            });
        }
        // NSData bytes are `cursor` contiguous MTLCounterResultTimestamp
        // values (verified length above); each is `#[repr(C)] { u64 }`.
        // We read each as native-endian bytes rather than reinterpreting
        // the pointer, since NSData only guarantees byte alignment.
        // SAFETY: `data` (an `NSData`) outlives this borrow, and `want <=
        // byte_len` guarantees `cursor` 8-byte chunks are in bounds.
        let bytes = unsafe { data.as_bytes_unchecked() };
        let ticks: Vec<u64> = bytes[..want]
            .chunks_exact(std::mem::size_of::<MTLCounterResultTimestamp>())
            .map(|c| u64::from_ne_bytes(c.try_into().expect("8-byte timestamp chunk")))
            .collect();
        Ok(self
            .segments
            .iter()
            .map(|&(label, s, e)| GpuSegment {
                label,
                start_tick: ticks[s],
                end_tick: ticks[e],
            })
            .collect())
    }
}

/// Find the device's timestamp counter set by its common name.
fn timestamp_counter_set(device: &Device) -> Result<Retained<ProtocolObject<dyn MTLCounterSet>>> {
    let sets = device.metal_device().counterSets().ok_or(Error::AppleApi {
        context: "MTLDevice::counterSets returned nil",
    })?;
    // SAFETY: reading a framework-provided static NSString constant.
    let want = unsafe { MTLCommonCounterSetTimestamp };
    for set in &sets {
        if *set.name() == *want {
            return Ok(set);
        }
    }
    Err(Error::AppleApi {
        context: "GpuSampler: device has no timestamp counter set",
    })
}
