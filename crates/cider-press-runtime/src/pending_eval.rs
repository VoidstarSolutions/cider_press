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
/// One batch spans several command buffers (`encode_ops` commits every
/// ~50 ops); the handles are held in submission order. Dropping the
/// handle blocks on completion (via each `CommandsInFlight::drop`)
/// before the pooled buffers held by `order` are released — they must never
/// return to the pool mid-flight — then lifts the host-read gate on every
/// dispatched node. A GPU failure status is only observable through an
/// explicit [`PendingEval::wait`].
pub struct PendingEval {
    /// Committed chunks in submission order — the last completes last.
    in_flight: Vec<CommandsInFlight>,
    /// Pins in-flight buffers until wait/drop; also the set of nodes whose
    /// `in_flight` host-read gate this handle must lift on completion.
    order: Vec<Arc<TensorInner>>,
}

impl PendingEval {
    /// Construct a handle owning the in-flight command buffers and the topo
    /// order that pins their buffers. Crate-internal: only `eval_async`
    /// builds these.
    pub(crate) fn new(in_flight: Vec<CommandsInFlight>, order: Vec<Arc<TensorInner>>) -> Self {
        Self { in_flight, order }
    }

    /// An already-complete handle for the case where the graph was fully
    /// materialized (nothing to dispatch).
    pub(crate) fn empty() -> Self {
        Self {
            in_flight: Vec::new(),
            order: Vec::new(),
        }
    }

    /// Block until this batch's GPU work completes. Consumes the handle;
    /// the pinned buffers are released and the host-read gate lifted (in
    /// `Drop`, which runs even on error — once every chunk's wait returns
    /// the GPU is done touching the buffers, failed or not) only after the
    /// wait returns. Propagates the first GPU failure status as
    /// [`CommandBufferFailed`](cider_press_kernels::Error::CommandBufferFailed)
    /// (later chunks are still waited, in `Drop`).
    pub fn wait(mut self) -> Result<()> {
        let _span = crate::profile::span("tensor.eval_async.wait");
        for chunk in self.in_flight.drain(..) {
            chunk.wait()?;
        }
        Ok(())
    }
}

impl Drop for PendingEval {
    fn drop(&mut self) {
        // Block on completion (no-op if `wait` already drained the handles)
        // *before* lifting the gate or releasing `order`'s pinned buffers.
        self.in_flight.clear();
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
            .field("in_flight_chunks", &self.in_flight.len())
            .field("pinned_nodes", &self.order.len())
            .finish()
    }
}
