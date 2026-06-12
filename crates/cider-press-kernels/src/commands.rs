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
//! [`Commands::elide_next_barrier`]) inside an encoder, and a per-output
//! fence map on the [`Device`] across encoders: a closing encoder waits
//! only the prior-writer fences of the buffers it reads or overwrites,
//! then updates its own fence (wait + update at close).
//!
//! If a `Commands` is dropped without being committed, its work is
//! discarded — the GPU never sees the encoded commands. This is the
//! intended "give up on this batch" behavior; the drop also restores
//! the device ordering state the session displaced (see `restore`),
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
/// per-output fence map lives on the [`Device`], and concurrent sessions
/// interleave its publish/wait pairing — ordering edges between encoders
/// are silently lost (the mutex on the map exists only for `Sync`; it
/// cannot make overlapping sessions sound). This is a documented
/// contract, not a type-enforced one: callers that need parallel
/// dispatch must use separate [`Device`]s, each with its own queue and
/// fence map.
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
    /// What Drop should undo if this session is abandoned uncommitted (work
    /// discarded). Fences published by this session's encoders would never
    /// signal — their updates die with the uncommitted buffer — so the
    /// device ordering state the first encoder displaced must be put back.
    restore: AbandonRestore,
    /// Number of dispatches issued in the current open encoder. Used by
    /// `dispatch_threads`/`dispatch_threadgroups` to insert a barrier
    /// before every dispatch after the first (concurrent encoder model).
    dispatched_in_encoder: usize,
    /// One-shot: when set, the next `dispatch_*` skips its hazard barrier
    /// and clears the flag. Set by [`Commands::elide_next_barrier`] after
    /// the caller has proven the next dispatch reads nothing written since
    /// the last barrier in this encoder.
    elide_next_barrier: bool,
    /// Buffer keys read by ops in the current open encoder. Fed by
    /// eval's resolver; consumed at `close_encoder`.
    input_keys: Vec<usize>,
    /// Buffer keys written by ops in the current open encoder. Waited
    /// as inputs (WAW / pool-recycling guard) and published at `close_encoder`.
    output_keys: Vec<usize>,
    /// Set when an op input could not be resolved to a key — forces a
    /// whole-frontier wait at close (the conservative fallback).
    unresolved_input: bool,
}

/// A fence-map entry displaced by a publish: `(key, prior_fence)`, where
/// `None` means the key was freshly added. Mirrors `device::DisplacedOutput`
/// (the boundary between the two modules is `pub(crate)`-thin, so the alias is
/// repeated rather than re-exported).
type DisplacedOutput = (usize, Option<Retained<ProtocolObject<dyn MTLFence>>>);

/// What `Drop` must undo if a session is abandoned (work discarded,
/// uncommitted) after it displaced device-level ordering state. Set once on
/// the first encoder open; cleared by the commit paths.
enum AbandonRestore {
    /// No encoder opened yet, or the session committed — leave device state.
    Untouched,
    /// Closed encoders in this session published these outputs (and displaced
    /// these prior values). Drop restores them, clearing the never-signaling
    /// fences this uncommitted session published.
    Map(Vec<DisplacedOutput>),
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
            restore: AbandonRestore::Untouched,
            dispatched_in_encoder: 0,
            elide_next_barrier: false,
            input_keys: Vec::new(),
            output_keys: Vec::new(),
            unresolved_input: false,
        })
    }

    /// Record a buffer key written by the next dispatch.
    pub fn record_output(&mut self, key: usize) {
        self.output_keys.push(key);
    }

    /// Record a buffer key read by the next dispatch.
    pub fn record_input(&mut self, key: usize) {
        self.input_keys.push(key);
    }

    /// Mark that the next dispatch reads a buffer that could not be resolved to
    /// a key — forces a whole-frontier wait when this encoder closes.
    pub fn record_unresolved_input(&mut self) {
        self.unresolved_input = true;
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
            // No open-time wait — the wait set is known only at close, from the
            // accumulated input/output keys.
            if matches!(self.restore, AbandonRestore::Untouched) {
                self.restore = AbandonRestore::Map(Vec::new());
            }
            // `restore` is already set: if this `?` aborts the open, Drop still
            // restores the displaced ordering state.
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

    /// Close any open encoder, publishing its fence under each output key in
    /// the device fence map (after the encoder waits its read/overwrite prior
    /// writers).
    ///
    /// `updateFence` must be encoded after all dispatches — it signals once
    /// all commands encoded before it complete. At encoder open it would
    /// capture nothing and fire immediately.
    fn close_encoder(&mut self) {
        if let Some(enc) = self.encoder.take() {
            // Wait only the prior-writer fences of buffers this encoder reads or
            // overwrites (outputs-as-inputs = the pool-recycling guard). Read the
            // map BEFORE publishing our own outputs.
            let mut wait_keys = self.input_keys.clone();
            wait_keys.extend(self.output_keys.iter().copied());
            for fence in self.device.fence_waits(&wait_keys, self.unresolved_input) {
                enc.waitForFence(&fence);
            }
            if let Some(fence) = self.fence.take() {
                enc.updateFence(&fence);
                let displaced = self.device.publish_outputs(&self.output_keys, &fence);
                if let AbandonRestore::Map(v) = &mut self.restore {
                    v.extend(displaced);
                }
            }
            enc.endEncoding();
            self.dispatched_in_encoder = 0;
            self.elide_next_barrier = false;
            self.input_keys.clear();
            self.output_keys.clear();
            self.unresolved_input = false;
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
        self.restore = AbandonRestore::Untouched;
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
        self.restore = AbandonRestore::Untouched;
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
        self.restore = AbandonRestore::Untouched;
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
        // Abandoned (uncommitted) session: the encoded fence updates die with
        // the buffer, so the device ordering state this session displaced must
        // be put back. Committed sessions cleared `restore` and closed their
        // encoder already.
        if let Some(enc) = self.encoder.take() {
            // Abandoned: end WITHOUT publishing this encoder's fence (the buffer
            // is never committed, so the fence would never signal).
            enc.endEncoding();
        }
        match std::mem::replace(&mut self.restore, AbandonRestore::Untouched) {
            // Undo every output this session published — those fences die with
            // the uncommitted buffer.
            AbandonRestore::Map(displaced) => self.device.restore_outputs(displaced),
            AbandonRestore::Untouched => {}
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

    /// An abandoned map-mode session that already closed an encoder (publishing
    /// its outputs) must remove those entries on Drop — their fence will never
    /// signal, and a later reader of the same buffer key would hang on it.
    #[test]
    fn abandoned_map_session_restores_published_outputs() {
        let device = Device::system_default().expect("device");
        let key = 0xABCD_usize;

        let mut cmds = super::Commands::new(&device).expect("map commands");
        cmds.encoder().expect("open encoder"); // creates own fence, restore = Map([])
        cmds.record_output(key);
        cmds.close_encoder(); // publishes map[key] = own fence (never committed)
        // Sanity: the key is live before the abandon.
        assert_eq!(device.fence_waits(&[key], false).len(), 1);

        drop(cmds); // abandoned uncommitted -> restore must remove the stale entry

        assert!(
            device.fence_waits(&[key], false).is_empty(),
            "abandoned map session left a never-signaling fence under its output key"
        );
    }
}
