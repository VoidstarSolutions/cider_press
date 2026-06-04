//! Handle for a committed-but-unwaited [`crate::Tensor::eval_async`] batch.
//!
//! `eval_async` populates result caches at commit time (buffer handles
//! valid immediately; bytes valid only after completion) and hands back a
//! `PendingEval`. The handle owns the in-flight command buffer plus the
//! topo `order` of dispatched nodes — keeping their `Arc<TensorInner>`s, and
//! thus their pooled output buffers, alive until the GPU finishes.
//!
//! Host reads of dispatched tensors are gated while the batch is in
//! flight: each node carries an `in_flight` flag (set at commit) that makes
//! `cpu_slice` / `cpu_to_vec` / `is_materialized` report the tensor as
//! unmaterialized until this handle observes GPU completion (an explicit
//! [`PendingEval::wait`] or the blocking drop). GPU→GPU reads across the
//! next command buffer are unaffected (the serial queue's hazard tracking
//! orders them) — dependent graphs can chain off in-flight tensors freely.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use cider_press_kernels::CommandsInFlight;

use crate::error::Result;
use crate::tensor::TensorInner;

/// A committed `eval_async` batch whose GPU work may still be running.
///
/// Dropping the handle blocks on completion (via `CommandsInFlight::drop`)
/// before the pooled buffers held by `order` are released — they must never
/// return to the pool mid-flight — then lifts the host-read gate on every
/// dispatched node. A GPU failure status is only observable through an
/// explicit [`PendingEval::wait`].
pub struct PendingEval {
    in_flight: Option<CommandsInFlight>,
    /// Pins in-flight buffers until wait/drop; also the set of nodes whose
    /// `in_flight` host-read gate this handle must lift on completion.
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

    /// Block until this batch's GPU work completes. Consumes the handle;
    /// the pinned buffers are released and the host-read gate lifted (in
    /// `Drop`, which runs even on error — once the wait returns the GPU is
    /// done touching the buffers, failed or not) only after the wait
    /// returns. Propagates a GPU failure status as
    /// [`CommandBufferFailed`](cider_press_kernels::Error::CommandBufferFailed).
    pub fn wait(mut self) -> Result<()> {
        let _span = crate::profile::span("tensor.eval_async.wait");
        match self.in_flight.take() {
            Some(in_flight) => Ok(in_flight.wait()?),
            None => Ok(()),
        }
    }
}

impl Drop for PendingEval {
    fn drop(&mut self) {
        // Block on completion (no-op if `wait` already took the handle)
        // *before* lifting the gate or releasing `order`'s pinned buffers.
        drop(self.in_flight.take());
        // GPU work is complete (successfully or not) — bytes are stable, so
        // host reads are race-free from here.
        for inner in &self.order {
            inner.in_flight.store(false, Ordering::Release);
        }
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
