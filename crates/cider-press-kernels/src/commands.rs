// Dispatch wrappers derived from MLX (https://github.com/ml-explore/mlx).
// Copyright © 2023-2024 Apple Inc. Released under the MIT license — see
// the full upstream notice at `kernels-mlx/COPYING`.

//! Command-buffer handle for batchable kernel dispatch.
//!
//! [`Commands`] owns one `MTLCommandBuffer` plus a lazily-opened compute
//! encoder. Kernel dispatch functions (in [`crate::kernels`]) take a
//! `&mut Commands`, encode their work, and return — they never commit
//! on their own. The caller controls when the batch flushes to the GPU
//! via [`Commands::commit_and_wait`] (blocking) or [`Commands::commit`]
//! (non-blocking, returning a [`CommandsInFlight`] handle to wait on later).
//!
//! The runtime layer batches many dispatches into one `Commands` (the
//! eval loop commits a fresh session every ~50 ops). Encoders open with
//! concurrent dispatch; ordering is explicit — a buffer-scope
//! `memoryBarrier` between dispatches (elidable via
//! [`Commands::elide_next_barrier`]) inside an encoder, and an
//! `MTLFence` chain across encoders (wait at open, update at close).
//!
//! If a `Commands` is dropped without being committed, its work is
//! discarded — the GPU never sees the encoded commands. This is the
//! intended "give up on this batch" behavior; the drop also restores
//! the fence-chain tail the session displaced (see `chain_restore`),
//! since fences updated by an uncommitted buffer never signal.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBarrierScope, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLComputePassDescriptor, MTLDispatchType, MTLFence, MTLSize,
};

use crate::device::Device;
use crate::error::{Error, Result};
use crate::gpu_sampler::{GpuSampler, GpuSegment};

/// In-progress command-buffer session.
///
/// Hold onto this across multiple kernel dispatches to batch them into
/// one GPU submission. The lifetime parameter ties it to the originating
/// [`Device`] so the device cannot be dropped while encoding is open.
///
/// # Concurrency contract
///
/// **One open session per device, one thread driving dispatch.** The
/// cross-encoder fence chain lives on the [`Device`], and concurrent
/// sessions interleave its take/set pairing — ordering edges between
/// encoders are silently lost (the per-call mutex on the chain tail
/// exists only for `Sync`; it cannot make overlapping sessions sound).
/// This is a documented contract, not a type-enforced one: callers that
/// need parallel dispatch must use separate [`Device`]s, each with its
/// own queue and fence chain.
pub struct Commands<'d> {
    device: &'d Device,
    cmd: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    encoder: Option<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>>,
    /// Fence the current open encoder will signal; published to `device`
    /// when the encoder is closed.
    fence: Option<Retained<ProtocolObject<dyn MTLFence>>>,
    /// `Some` only in the profiling regime — one sampled encoder per
    /// dispatch. `None` is the production single-encoder path.
    sampler: Option<GpuSampler>,
    /// When set, the next `encoder()` opens a sampled pass for this
    /// (start, end) sample-index pair, then clears.
    armed: Option<(usize, usize)>,
    /// What Drop should do to the device fence-chain tail if this
    /// session is abandoned uncommitted (work discarded). Fences
    /// published by this session's encoders would never signal — their
    /// updates die with the uncommitted buffer — so the tail the first
    /// encoder displaced must be put back, even when it was "no tail".
    chain_restore: ChainRestore,
    /// Number of dispatches issued in the current open encoder. Used by
    /// `dispatch_threads`/`dispatch_threadgroups` to insert a barrier
    /// before every dispatch after the first (concurrent encoder model).
    dispatched_in_encoder: usize,
    /// One-shot: when set, the next `dispatch_*` skips its hazard barrier
    /// and clears the flag. Set by [`Commands::elide_next_barrier`] after
    /// the caller has proven the next dispatch reads nothing written since
    /// the last barrier in this encoder.
    elide_next_barrier: bool,
}

/// Snapshot of the device fence-chain tail a [`Commands`] session
/// displaced, restored by Drop when the session is abandoned.
enum ChainRestore {
    /// No encoder opened yet, or the session committed — leave the
    /// device tail alone.
    Untouched,
    /// The first encoder displaced this tail (`None` on a device with
    /// no tail yet). Drop puts it back, clearing any never-signaling
    /// fence this session published.
    Tail(Option<Retained<ProtocolObject<dyn MTLFence>>>),
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
            device,
            cmd,
            encoder: None,
            fence: None,
            sampler: None,
            armed: None,
            chain_restore: ChainRestore::Untouched,
            dispatched_in_encoder: 0,
            elide_next_barrier: false,
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
        if self.sampler.is_none() {
            return;
        }
        // Close any open encoder so its stage-end timestamp is written.
        self.close_encoder();
        let sampler = self.sampler.as_mut().expect("checked above");
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
    ///
    /// **Warning:** do not encode dispatches directly on the returned encoder.
    /// Use [`Commands::dispatch_threads`] / [`Commands::dispatch_threadgroups`]
    /// so the concurrent-encoder barrier is inserted. A raw dispatch silently
    /// skips the barrier and corrupts ordering under untracked buffers.
    pub(crate) fn encoder(&mut self) -> Result<&ProtocolObject<dyn MTLComputeCommandEncoder>> {
        if self.encoder.is_none() {
            let enc = if let Some((start, end)) = self.armed.take() {
                let sampler = self
                    .sampler
                    .as_ref()
                    .expect("armed implies a sampler is present");
                let descriptor = MTLComputePassDescriptor::computePassDescriptor();
                descriptor.setDispatchType(MTLDispatchType::Concurrent);
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
                let descriptor = MTLComputePassDescriptor::computePassDescriptor();
                descriptor.setDispatchType(MTLDispatchType::Concurrent);
                self.cmd
                    .computeCommandEncoderWithDescriptor(&descriptor)
                    .ok_or(Error::AppleApi {
                        context: "MTLCommandBuffer::computeCommandEncoderWithDescriptor",
                    })?
            };
            self.dispatched_in_encoder = 0;
            // A freshly-opened encoder's first dispatch carries no barrier
            // anyway; a stale elide flag carried across the open would be a
            // no-op there but could mask a real first-dispatch barrier in a
            // future regime — clear it so the flag never crosses encoders.
            self.elide_next_barrier = false;
            let prev = self.device.take_last_fence();
            if let Some(prev) = &prev {
                enc.waitForFence(prev);
            }
            if matches!(self.chain_restore, ChainRestore::Untouched) {
                self.chain_restore = ChainRestore::Tail(prev);
            }
            // `chain_restore` is already set: if this `?` aborts the open, Drop
            // still restores the displaced tail instead of losing the chain.
            let own = self.device.new_fence()?;
            // NOTE: updateFence is NOT called here. At encoder open it would
            // capture no commands and signal immediately — the cross-encoder
            // chain would be a no-op. It is encoded in close_encoder(), after
            // all dispatches, matching MLX's end_encoding placement.
            self.fence = Some(own);
            self.encoder = Some(enc);
        }
        Ok(self.encoder.as_ref().expect("just inserted"))
    }

    /// Close any open encoder and publish its fence as the chain tail.
    fn close_encoder(&mut self) {
        if let Some(enc) = self.encoder.take() {
            // updateFence must be encoded after all dispatches — it signals
            // once all commands encoded before it complete. At encoder open
            // it would capture nothing and fire immediately.
            if let Some(fence) = self.fence.take() {
                enc.updateFence(&fence);
                self.device.set_last_fence(Some(fence));
            }
            enc.endEncoding();
            self.dispatched_in_encoder = 0;
            self.elide_next_barrier = false;
        }
    }

    /// Skip the hazard barrier before the NEXT dispatch only (one-shot).
    /// Callers must prove the next dispatch reads nothing written since
    /// the last barrier in this encoder — see eval's graph-level hazard
    /// tracking. Wrongly eliding corrupts ordering silently.
    pub fn elide_next_barrier(&mut self) {
        self.elide_next_barrier = true;
    }

    /// Encode a `dispatchThreads` call, fencing it behind every prior
    /// dispatch in this encoder with a buffer-scope memory barrier (the
    /// encoder is concurrent; ordering is explicit). Derived from MLX's
    /// `CommandEncoder::dispatch_threads` (MIT, © 2023-2024 Apple Inc.).
    ///
    /// The barrier is elided when [`Commands::elide_next_barrier`] was
    /// called before this dispatch (one-shot): the graph-level hazard
    /// tracker in eval proves the dispatch is independent of everything
    /// encoded since the last barrier, so it may overlap on the GPU.
    pub fn dispatch_threads(&mut self, grid: MTLSize, threadgroup: MTLSize) -> Result<()> {
        let elide = std::mem::take(&mut self.elide_next_barrier);
        let n = self.dispatched_in_encoder;
        let encoder = self.encoder()?;
        if n > 0 && !elide {
            encoder.memoryBarrierWithScope(MTLBarrierScope::Buffers);
        }
        encoder.dispatchThreads_threadsPerThreadgroup(grid, threadgroup);
        self.dispatched_in_encoder += 1;
        Ok(())
    }

    /// `dispatchThreadgroups` analogue of [`Commands::dispatch_threads`].
    pub fn dispatch_threadgroups(&mut self, grid: MTLSize, threadgroup: MTLSize) -> Result<()> {
        let elide = std::mem::take(&mut self.elide_next_barrier);
        let n = self.dispatched_in_encoder;
        let encoder = self.encoder()?;
        if n > 0 && !elide {
            encoder.memoryBarrierWithScope(MTLBarrierScope::Buffers);
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, threadgroup);
        self.dispatched_in_encoder += 1;
        Ok(())
    }

    /// Close any open encoder, commit the command buffer, and block
    /// until the GPU finishes. Consumes the handle so the same batch
    /// can't be re-committed. Returns [`Error::CommandBufferFailed`] if
    /// the GPU reports a failure status for the batch.
    pub fn commit_and_wait(mut self) -> Result<()> {
        self.close_encoder();
        self.chain_restore = ChainRestore::Untouched;
        self.cmd.commit();
        self.cmd.waitUntilCompleted();
        check_completed(&self.cmd)
    }

    /// Close any open encoder and commit the command buffer **without
    /// waiting**. Returns a [`CommandsInFlight`] handle whose GPU work may
    /// still be running; block on it via [`CommandsInFlight::wait`] (or by
    /// dropping it). Consumes the handle so the same batch can't be
    /// re-committed.
    pub fn commit(mut self) -> Result<CommandsInFlight> {
        self.close_encoder();
        self.chain_restore = ChainRestore::Untouched;
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
        self.close_encoder();
        self.chain_restore = ChainRestore::Untouched;
        self.cmd.commit();
        self.cmd.waitUntilCompleted();
        check_completed(&self.cmd)?;
        match self.sampler.as_ref() {
            Some(sampler) => sampler.resolve(),
            None => Ok(Vec::new()),
        }
    }
}

/// Map a waited command buffer's final status to a `Result`.
/// `waitUntilCompleted` returns only once the buffer is `Completed` or
/// `Error`; any other status here is a Metal contract violation and is
/// reported as a failure too.
fn completion_result(
    status: MTLCommandBufferStatus,
    error_description: Option<String>,
) -> Result<()> {
    if status == MTLCommandBufferStatus::Completed {
        return Ok(());
    }
    Err(Error::CommandBufferFailed(
        error_description.unwrap_or_else(|| format!("MTLCommandBufferStatus({})", status.0)),
    ))
}

/// Check a just-waited command buffer's status, pulling the `NSError`
/// localized description on failure.
fn check_completed(cmd: &ProtocolObject<dyn MTLCommandBuffer>) -> Result<()> {
    completion_result(
        cmd.status(),
        cmd.error().map(|e| e.localizedDescription().to_string()),
    )
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
    /// Returns [`Error::CommandBufferFailed`] if the GPU reports a
    /// failure status for the batch.
    pub fn wait(self) -> Result<()> {
        self.cmd.waitUntilCompleted();
        check_completed(&self.cmd)
    }
}

impl Drop for CommandsInFlight {
    fn drop(&mut self) {
        // `wait` consumes `self`, so `Drop` only fires when the caller
        // discards the handle without an explicit wait. `waitUntilCompleted`
        // on an already-completed buffer returns immediately. A GPU failure
        // status is unobservable here (Drop can't return); callers who care
        // must `wait()` explicitly.
        self.cmd.waitUntilCompleted();
    }
}

impl Drop for Commands<'_> {
    fn drop(&mut self) {
        // Abandoned (uncommitted) session: the encoded fence updates die
        // with the buffer, so end the encoder WITHOUT publishing its fence
        // and restore the chain tail this session displaced. Committed
        // sessions cleared `chain_restore` and closed their encoder already.
        if let Some(enc) = self.encoder.take() {
            // Abandoned path: end the encoder WITHOUT encoding updateFence and
            // without publishing to the device chain. The buffer is never
            // committed, so any fence update would never signal — callers
            // waiting on it would hang. chain_restore (below) puts the old
            // tail back so the next session's waitForFence doesn't stall.
            enc.endEncoding();
        }
        let restore = std::mem::replace(&mut self.chain_restore, ChainRestore::Untouched);
        if let ChainRestore::Tail(prev) = restore {
            // Restores a `None` tail too: on a fresh device, a fence this
            // session published (close_encoder during profiling) must be
            // CLEARED, not left as a never-signaling tail.
            self.device.set_last_fence(prev);
        }
    }
}

// `Device::commands` lives in this module so it can hand back a
// `Commands<'_>` constructed via the module-private fields above.
impl Device {
    /// Begin a new [`Commands`] session on this device's queue.
    ///
    /// At most one session may be open per device at a time — see the
    /// concurrency contract on [`Commands`]. Overlapping sessions are
    /// not detected; they silently corrupt cross-encoder ordering.
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
    use objc2_metal::MTLCommandBufferStatus;

    use crate::kernels::arg_reduce::argmax_bf16;
    use crate::{Buffer, Device, KernelLibrary};

    #[test]
    fn completion_result_ok_for_completed_status() {
        assert!(super::completion_result(MTLCommandBufferStatus::Completed, None).is_ok());
    }

    #[test]
    fn completion_result_maps_error_status_with_description() {
        let err = super::completion_result(
            MTLCommandBufferStatus::Error,
            Some("IOGPUCommandQueue page fault".into()),
        )
        .expect_err("Error status must map to Err");
        assert!(err.to_string().contains("page fault"), "{err}");
    }

    #[test]
    fn completion_result_errs_for_error_status_without_description() {
        assert!(super::completion_result(MTLCommandBufferStatus::Error, None).is_err());
    }

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

    /// An abandoned (uncommitted) Commands session must restore the fence
    /// chain tail it displaced, so the next session's encoder does not wait
    /// on a fence whose `updateFence` died with the uncommitted buffer.
    ///
    /// Failure mode without `chain_restore`: session C's `encoder()` calls
    /// `waitForFence` on B's never-signaled fence and hangs forever. CI
    /// surfaces this as a test timeout.
    #[test]
    fn abandoned_session_restores_fence_chain() {
        let device = Device::system_default().expect("device");
        let library = KernelLibrary::arg_reduce(&device).expect("arg_reduce lib");

        let host: Vec<bf16> = [0.1f32, 0.5, 0.3, 0.9, 0.2]
            .iter()
            .map(|&x| bf16::from_f32(x))
            .collect();
        let src: Buffer<bf16> = device.upload(&host).expect("upload src");
        let mut dst_a: Buffer<u32> = device.alloc_buffer(1).expect("alloc dst_a");
        let mut dst_c: Buffer<u32> = device.alloc_buffer(1).expect("alloc dst_c");

        // Session A: committed — publishes a real fence tail.
        let mut cmds_a = device.commands().expect("commands A");
        argmax_bf16(&mut cmds_a, &library, &src, &mut dst_a, host.len()).expect("dispatch A");
        cmds_a.commit_and_wait().expect("commit A");

        // Session B: abandoned (dropped uncommitted). Without chain_restore
        // this would leave B's never-signaling fence as the chain tail.
        let mut cmds_b = device.commands().expect("commands B");
        cmds_b.encoder().expect("open encoder B");
        drop(cmds_b);

        // Session C: must complete — would hang if B's fence is the tail.
        let mut cmds_c = device.commands().expect("commands C");
        argmax_bf16(&mut cmds_c, &library, &src, &mut dst_c, host.len()).expect("dispatch C");
        cmds_c.commit_and_wait().expect("commit C");

        // SAFETY: commit_and_wait blocked; GPU is done writing dst_a / dst_c.
        assert_eq!(unsafe { dst_a.as_slice()[0] }, 3, "session A result");
        assert_eq!(unsafe { dst_c.as_slice()[0] }, 3, "session C result");
    }

    /// An abandoned session on a FRESH device (no prior fence tail) must
    /// clear any fence it published, not just restore a displaced one.
    /// Closing an encoder publishes the session's own fence as the
    /// device tail; if the session is then dropped uncommitted, that
    /// fence never signals. Calls the private `close_encoder` directly —
    /// the public route is `begin_profiled_op`, but GPU counter sampling
    /// is unavailable on virtualized CI runners. Asserts the tail state
    /// directly — the hang-based variant of this failure would stall the
    /// suite instead of failing.
    #[test]
    fn abandoned_session_on_fresh_device_clears_fence_tail() {
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
        // Closes the open encoder, publishing this session's fence.
        cmds.close_encoder();
        drop(cmds); // abandoned uncommitted — the published fence never signals

        assert!(
            device.take_last_fence().is_none(),
            "abandoned session left its never-signaling fence as the chain tail"
        );
    }
}
