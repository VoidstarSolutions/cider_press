//! Handle for a committed-but-unwaited [`crate::Tensor::eval_async`] batch.
//!
//! `eval_async` populates result caches at commit time (buffer handles
//! valid immediately; bytes valid only after completion) and hands back a
//! `PendingEval`. The handle owns the in-flight command buffer plus the
//! topo `order` of dispatched nodes — keeping their `Arc<TensorInner>`s, and
//! thus their pooled output buffers, alive until the GPU finishes.
//!
//! Contract: do not `cpu_slice` / `cpu_to_vec` any tensor dispatched by the
//! originating `eval_async` until this handle has been waited. GPU→GPU reads
//! across the next command buffer are safe (the serial queue's hazard
//! tracking orders them); only host reads of in-flight bytes are not.

use std::sync::Arc;

use cider_press_kernels::CommandsInFlight;

use crate::error::Result;
use crate::tensor::TensorInner;

/// A committed `eval_async` batch whose GPU work may still be running.
///
/// `in_flight` is declared before `order` so that on drop the command-buffer
/// wait (in `CommandsInFlight::drop`) runs *before* the pooled buffers held
/// by `order` are released — they must never return to the pool mid-flight.
pub struct PendingEval {
    in_flight: Option<CommandsInFlight>,
    #[allow(dead_code)] // held purely to pin in-flight buffers until wait/drop.
    order: Vec<Arc<TensorInner>>,
}

impl PendingEval {
    /// Construct a handle owning the in-flight command buffer and the topo
    /// order that pins its buffers. Crate-internal: only `eval_async` builds
    /// these.
    pub(crate) fn new(in_flight: CommandsInFlight, order: Vec<Arc<TensorInner>>) -> Self {
        Self {
            in_flight: Some(in_flight),
            order,
        }
    }

    /// An already-complete handle for the case where the graph was fully
    /// materialized (nothing to dispatch).
    pub(crate) fn empty() -> Self {
        Self {
            in_flight: None,
            order: Vec::new(),
        }
    }

    /// Block until this batch's GPU work completes. Consumes the handle; the
    /// pinned buffers are released only after the wait returns.
    pub fn wait(mut self) -> Result<()> {
        let _span = crate::profile::span("tensor.eval_async.wait");
        if let Some(in_flight) = self.in_flight.take() {
            in_flight.wait()?;
        }
        Ok(())
    }
}

impl std::fmt::Debug for PendingEval {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PendingEval")
            .field("in_flight", &self.in_flight.is_some())
            .field("pinned_nodes", &self.order.len())
            .finish()
    }
}
