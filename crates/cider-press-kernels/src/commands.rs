//! Command-buffer handle for batchable kernel dispatch.
//!
//! [`Commands`] owns one `MTLCommandBuffer` plus a lazily-opened compute
//! encoder. Kernel dispatch functions (in [`crate::kernels`]) take a
//! `&mut Commands`, encode their work, and return — they never commit
//! on their own. The caller controls when the batch flushes to the GPU
//! via [`Commands::commit_and_wait`].
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
use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder};

use crate::device::Device;
use crate::error::{Error, Result};

/// In-progress command-buffer session.
///
/// Hold onto this across multiple kernel dispatches to batch them into
/// one GPU submission. The lifetime parameter ties it to the originating
/// [`Device`] so the device cannot be dropped while encoding is open.
pub struct Commands<'d> {
    _device: &'d Device,
    cmd: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    encoder: Option<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>>,
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
        })
    }

    /// Borrow the current compute encoder, opening one on first use.
    ///
    /// Kernel dispatch functions in [`crate::kernels`] call this to get
    /// a target for `setBuffer:` / `dispatchThreadgroups:`. They must
    /// not call `endEncoding` on it — the encoder lives until
    /// [`Commands::commit_and_wait`] closes it.
    pub(crate) fn encoder(&mut self) -> Result<&ProtocolObject<dyn MTLComputeCommandEncoder>> {
        if self.encoder.is_none() {
            let enc = self.cmd.computeCommandEncoder().ok_or(Error::AppleApi {
                context: "MTLCommandBuffer::computeCommandEncoder",
            })?;
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
}
