//! Command-buffer handle for batchable kernel dispatch.
//!
//! [`Commands`] owns one `MTLCommandBuffer` plus a lazily-opened compute
//! encoder. Kernel dispatch functions (in [`crate::kernels`]) take a
//! `&mut Commands`, encode their work, and return — they never commit
//! on their own. The caller controls when the batch flushes to the GPU
//! via [`Commands::commit_and_wait`] (blocking) or [`Commands::commit`]
//! (non-blocking, returning a [`CommandsInFlight`] handle to wait on later).
//!
//! This is the design lever that the qmv dispatch-latency benchmark
//! identified: the ~1.5× latency gap vs. MLX comes from
//! committing+waiting per dispatch. The runtime layer will batch many
//! dispatches into one `Commands` to close that gap. The kernel layer
//! merely needs to make that batching trivial.
//!
//! If a `Commands` is dropped without being committed, its work is
//! discarded — the GPU never sees the encoded commands. This is the
//! intended "give up on this batch" behavior.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePassDescriptor,
};

use crate::device::Device;
use crate::error::{Error, Result};
use crate::gpu_sampler::{GpuSampler, GpuSegment};

/// In-progress command-buffer session.
///
/// Hold onto this across multiple kernel dispatches to batch them into
/// one GPU submission. The lifetime parameter ties it to the originating
/// [`Device`] so the device cannot be dropped while encoding is open.
pub struct Commands<'d> {
    _device: &'d Device,
    cmd: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    encoder: Option<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>>,
    /// `Some` only in the profiling regime — one sampled encoder per
    /// dispatch. `None` is the production single-encoder path.
    sampler: Option<GpuSampler>,
    /// When set, the next `encoder()` opens a sampled pass for this
    /// (start, end) sample-index pair, then clears.
    armed: Option<(usize, usize)>,
}

impl<'d> Commands<'d> {
    /// Open a new command-buffer session against `device`'s queue.
    ///
    /// Use [`Device::commands`] in normal code.
    pub fn new(device: &'d Device) -> Result<Self> {
        let cmd = device
            .metal_queue()
            .commandBuffer()
            .ok_or(Error::AppleApi {
                context: "MTLCommandQueue::commandBuffer",
            })?;
        Ok(Self {
            _device: device,
            cmd,
            encoder: None,
            sampler: None,
            armed: None,
        })
    }

    /// Open a profiled session sized for `op_capacity` dispatches. Each
    /// later `begin_profiled_op` opens a fresh sampled encoder.
    pub fn new_profiled(device: &'d Device, op_capacity: usize) -> Result<Self> {
        let mut commands = Self::new(device)?;
        commands.sampler = Some(GpuSampler::new(device, op_capacity)?);
        Ok(commands)
    }

    /// Close the current encoder (writing its end timestamp) and arm the
    /// next `encoder()` to open a sampled pass for `label`. No-op when
    /// not in the profiling regime.
    pub fn begin_profiled_op(&mut self, label: &'static str) {
        let Some(sampler) = self.sampler.as_mut() else {
            return;
        };
        // Close any open encoder so its stage-end timestamp is written.
        if let Some(enc) = self.encoder.take() {
            enc.endEncoding();
        }
        match sampler.reserve(label) {
            Ok(pair) => self.armed = Some(pair),
            // Capacity exhausted: stop sampling further ops rather than
            // erroring mid-eval (the report shows fewer segments). Calling
            // this twice with no dispatch between also just opens and
            // closes an empty sampled encoder — a harmless ~0-tick segment.
            Err(_) => self.armed = None,
        }
    }

    /// Borrow the current compute encoder, opening one on first use.
    ///
    /// Kernel dispatch functions in [`crate::kernels`] call this to get
    /// a target for `setBuffer:` / `dispatchThreadgroups:`. They must
    /// not call `endEncoding` on it — the encoder lives until
    /// [`Commands::commit_and_wait`] closes it.
    pub(crate) fn encoder(&mut self) -> Result<&ProtocolObject<dyn MTLComputeCommandEncoder>> {
        if self.encoder.is_none() {
            let enc = if let Some((start, end)) = self.armed.take() {
                let sampler = self
                    .sampler
                    .as_ref()
                    .expect("armed implies a sampler is present");
                let descriptor = MTLComputePassDescriptor::computePassDescriptor();
                let attachment = {
                    // SAFETY: index 0 is always valid on the attachment
                    // array — it is allocated with at least one slot.
                    unsafe {
                        descriptor
                            .sampleBufferAttachments()
                            .objectAtIndexedSubscript(0)
                    }
                };
                attachment.setSampleBuffer(Some(sampler.buffer()));
                // SAFETY: start/end are reserved indices < the buffer's
                // sampleCount (GpuSampler::reserve enforces the bound).
                unsafe {
                    attachment.setStartOfEncoderSampleIndex(start);
                    attachment.setEndOfEncoderSampleIndex(end);
                }
                self.cmd
                    .computeCommandEncoderWithDescriptor(&descriptor)
                    .ok_or(Error::AppleApi {
                        context: "MTLCommandBuffer::computeCommandEncoderWithDescriptor",
                    })?
            } else {
                self.cmd.computeCommandEncoder().ok_or(Error::AppleApi {
                    context: "MTLCommandBuffer::computeCommandEncoder",
                })?
            };
            self.encoder = Some(enc);
        }
        Ok(self.encoder.as_ref().expect("just inserted"))
    }

    /// Close any open encoder, commit the command buffer, and block
    /// until the GPU finishes. Consumes the handle so the same batch
    /// can't be re-committed.
    pub fn commit_and_wait(mut self) -> Result<()> {
        if let Some(enc) = self.encoder.take() {
            enc.endEncoding();
        }
        self.cmd.commit();
        self.cmd.waitUntilCompleted();
        Ok(())
    }

    /// Close any open encoder and commit the command buffer **without
    /// waiting**. Returns a [`CommandsInFlight`] handle whose GPU work may
    /// still be running; block on it via [`CommandsInFlight::wait`] (or by
    /// dropping it). Consumes the handle so the same batch can't be
    /// re-committed.
    pub fn commit(mut self) -> Result<CommandsInFlight> {
        if let Some(enc) = self.encoder.take() {
            enc.endEncoding();
        }
        self.cmd.commit();
        // Refcount bump: the `Commands` Drop runs at end of this fn (encoder
        // already taken, so it is a no-op), while the handle keeps the
        // command buffer alive independently.
        Ok(CommandsInFlight {
            cmd: self.cmd.clone(),
        })
    }

    /// Profiling-only sibling of [`Commands::commit_and_wait`]: closes the
    /// encoder, commits, waits, then resolves per-op GPU-tick segments.
    /// Returns an empty vec if this session has no sampler.
    pub fn commit_wait_resolve(mut self) -> Result<Vec<GpuSegment>> {
        if let Some(enc) = self.encoder.take() {
            enc.endEncoding();
        }
        self.cmd.commit();
        self.cmd.waitUntilCompleted();
        match self.sampler.as_ref() {
            Some(sampler) => sampler.resolve(),
            None => Ok(Vec::new()),
        }
    }
}

/// A committed command buffer whose GPU work may still be in flight.
///
/// Hold it to keep the submission alive; [`CommandsInFlight::wait`] blocks
/// until the GPU finishes. `Drop` also blocks on completion, so dropping a
/// handle without an explicit `wait` is safe — it never lets a caller reuse
/// buffers the GPU is still reading or writing.
pub struct CommandsInFlight {
    cmd: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
}

impl CommandsInFlight {
    /// Block until the GPU finishes this batch. Consumes the handle.
    pub fn wait(self) -> Result<()> {
        self.cmd.waitUntilCompleted();
        Ok(())
    }
}

impl Drop for CommandsInFlight {
    fn drop(&mut self) {
        // `wait` consumes `self`, so `Drop` only fires when the caller
        // discards the handle without an explicit wait. `waitUntilCompleted`
        // on an already-completed buffer returns immediately.
        self.cmd.waitUntilCompleted();
    }
}

impl Drop for Commands<'_> {
    fn drop(&mut self) {
        // If the user dropped without committing, end encoding cleanly
        // but don't commit — they explicitly chose not to send this
        // batch to the GPU. The command buffer reference dies with us.
        if let Some(enc) = self.encoder.take() {
            enc.endEncoding();
        }
    }
}

// `Device::commands` lives in this module so it can hand back a
// `Commands<'_>` constructed via the module-private fields above.
impl Device {
    /// Begin a new [`Commands`] session on this device's queue.
    pub fn commands(&self) -> Result<Commands<'_>> {
        Commands::new(self)
    }

    /// Begin a profiled [`Commands`] session sized for `op_capacity`
    /// dispatches. Returns [`Error::AppleApi`] if the device lacks a
    /// timestamp counter set (see [`Device::supports_stage_boundary_sampling`]).
    pub fn commands_profiled(&self, op_capacity: usize) -> Result<Commands<'_>> {
        Commands::new_profiled(self, op_capacity)
    }
}

#[cfg(test)]
mod tests {
    use half::bf16;

    use crate::kernels::arg_reduce::argmax_bf16;
    use crate::{Buffer, Device, KernelLibrary};

    /// `commit` (no wait) then waiting the returned handle must produce the
    /// same result as `commit_and_wait` — the GPU work runs, we just defer
    /// the block. Uses argmax as a trivial dispatch with a known answer.
    #[test]
    fn commit_then_wait_handle_completes_work() {
        let device = Device::system_default().expect("device");
        let library = KernelLibrary::arg_reduce(&device).expect("arg_reduce lib");
        let host: Vec<bf16> = [0.1f32, 0.5, 0.3, 0.9, 0.2]
            .iter()
            .map(|&x| bf16::from_f32(x))
            .collect();
        let src: Buffer<bf16> = device.upload(&host).expect("upload");
        let mut dst: Buffer<u32> = device.alloc_buffer(1).expect("alloc dst");

        let mut cmds = device.commands().expect("commands");
        argmax_bf16(&mut cmds, &library, &src, &mut dst, host.len()).expect("dispatch");
        let inflight = cmds.commit().expect("commit without wait");
        inflight.wait().expect("wait");

        // SAFETY: dst is a [1] u32 buffer, valid after the handle is waited.
        assert_eq!(unsafe { dst.as_slice()[0] }, 3);
    }

    /// Dropping the handle without calling `wait` must still block on
    /// completion (RAII safety) — the result is valid after the drop.
    #[test]
    fn dropping_handle_blocks_until_complete() {
        let device = Device::system_default().expect("device");
        let library = KernelLibrary::arg_reduce(&device).expect("arg_reduce lib");
        let host: Vec<bf16> = [0.1f32, 0.9, 0.3]
            .iter()
            .map(|&x| bf16::from_f32(x))
            .collect();
        let src: Buffer<bf16> = device.upload(&host).expect("upload");
        let mut dst: Buffer<u32> = device.alloc_buffer(1).expect("alloc dst");

        let mut cmds = device.commands().expect("commands");
        argmax_bf16(&mut cmds, &library, &src, &mut dst, host.len()).expect("dispatch");
        drop(cmds.commit().expect("commit")); // Drop must waitUntilCompleted

        assert_eq!(unsafe { dst.as_slice()[0] }, 1);
    }
}
