//! Eval: walk the lazy graph, encode its dispatches into command-buffer
//! chunks (committed every [`OPS_PER_COMMAND_BUFFER`] ops), populate
//! result caches.
//!
//! Public entry points are [`Tensor::eval`](crate::Tensor::eval)
//! (synchronous: waits every chunk before returning) and
//! [`Tensor::eval_async`](crate::Tensor::eval_async) (commits without
//! waiting, returning a [`PendingEval`](crate::PendingEval) the caller
//! waits later — the decode pipeline runs the next token's encode under the
//! prior token's GPU work). Both share the topo walk, per-op dispatcher,
//! and cache-population pass, and both run a [`detach_order`] pass after
//! populating caches so an evaluated node drops its op inputs (MLX's
//! detach-on-eval) and stops retaining its upstream graph.
//!
//! **Untracked-buffer safety rests on two invariants.** Cross-encoder and
//! cross-token ordering — the case where a buffer written by one command
//! buffer is read by a later one — is guaranteed by the encoder fence chain
//! in [`cider_press_kernels::Commands`] (see also `PooledBuffer`'s doc in
//! `buffer_pool.rs` for the pool-reuse safety argument). Intra-encoder
//! ordering — the case where two dispatches in the *same* encoder touch the
//! same buffer — is guaranteed by `memoryBarrier` calls inserted by
//! `encode_ops`; those barriers are elidable only when `encode_ops`'s
//! window-reset hazard tracker (`unsynced`) confirms no overlap since the
//! last barrier.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, OnceLock};

use cider_press_kernels::{
    Buffer, Commands, CommandsInFlight, KernelLibrary,
    kernels::{arg_reduce, binary, copy, qmm, qmv, reduce, unary},
};

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::layout::Layout;
use crate::tensor::{
    ArgReduceKind, BinaryOp, LeafStorage, OpKind, OpNode, ReduceKind, Tensor, TensorInner, UnaryOp,
    checked_byte_count,
};

/// Static label for an [`OpKind`], used as the GPU-segment bucket key in
/// the profiled eval. One label per kind; `QuantizedMatMul` is not split
/// into qmv/qmm here because the kind carries no M — the dispatcher picks
/// the kernel from the activation shape (at decode, M=1, so every segment
/// is a qmv).
pub(crate) fn op_kind_label(kind: &OpKind) -> &'static str {
    match kind {
        OpKind::Copy => "copy",
        OpKind::QuantizedMatMul { .. } => "quantized_matmul",
        OpKind::Binary { .. } => "binary",
        OpKind::Unary { .. } => "unary",
        OpKind::Reduce { .. } => "reduce",
        OpKind::Gather => "gather",
        OpKind::Dequantize { .. } => "dequantize",
        OpKind::Rope { .. } => "rope",
        OpKind::RmsNorm { .. } => "rms_norm",
        OpKind::Softmax { .. } => "softmax",
        OpKind::MatMul => "matmul",
        OpKind::SliceUpdate { .. } => "slice_update",
        OpKind::Sdpa { .. } => "sdpa",
        OpKind::ArgReduce { .. } => "arg_reduce",
    }
}

/// Step 1: schedule the unevaluated op nodes reachable from `root` in
/// Kahn level order (a valid topo order in which every op appears after
/// all of its producers, and mutually independent ops are adjacent).
///
/// Why level order rather than DFS post-order: `encode_ops` elides a
/// dispatch's memory barrier when none of its inputs were written since
/// the last barrier. DFS emits each dependency *chain* back-to-back, so
/// almost every op hazards on its immediate predecessor and the barrier
/// stays. In a Kahn wave every op depends only on earlier waves, so the
/// FIRST op of a wave pays the barrier (which collapses the hazard
/// window) and the REST elide — their producers are no longer in the
/// window. Barriers drop toward the graph's critical-path length, and the
/// independent ops a wave groups together (q/k/v projections, the two
/// rope halves, gate‖up matmuls) land adjacent so they overlap on the
/// concurrent encoder.
///
/// Within a wave, nodes keep reachability-discovery order (a `Vec`
/// worklist, never `HashMap` iteration) so the schedule is deterministic
/// across runs.
fn build_order(root: &Tensor) -> Vec<Arc<TensorInner>> {
    // Reachability pass. Mirrors `visit`'s skip logic exactly: dedup by
    // `Arc` identity; a node whose cache is already populated is terminal
    // (skip its whole subtree, even if it carries an op); op nodes are the
    // only nodes scheduled; views own no storage, so an input reached
    // through a view chain contributes the view's *source* op as the
    // dependency. Iterative (graph depth ~600) to avoid blowing the stack.
    let mut discovered: HashSet<*const TensorInner> = HashSet::new();
    let mut nodes: Vec<Arc<TensorInner>> = Vec::new();
    // Producer Arcs per node, parallel to `nodes`; translated to indices
    // once every node has a slot (a producer may be discovered after its
    // consumer).
    let mut producer_arcs: Vec<Vec<Arc<TensorInner>>> = Vec::new();
    let mut index_of: HashMap<*const TensorInner, usize> = HashMap::new();
    let mut stack: Vec<Arc<TensorInner>> = Vec::new();

    if let Some(node) = resolve_to_op(&root.inner) {
        stack.push(node);
    }
    while let Some(inner) = stack.pop() {
        let ptr = Arc::as_ptr(&inner);
        if !discovered.insert(ptr) {
            continue;
        }
        index_of.insert(ptr, nodes.len());
        let op = inner
            .op
            .as_ref()
            .expect("resolve_to_op only yields op nodes");
        // Direct op-node dependencies, deduped per unique producer (an op
        // reading the same producer twice must not double-count in-degree).
        let mut node_deps: Vec<Arc<TensorInner>> = Vec::new();
        let mut seen_dep: HashSet<*const TensorInner> = HashSet::new();
        for input in op.inputs() {
            if let Some(dep) = resolve_to_op(&input.inner) {
                if seen_dep.insert(Arc::as_ptr(&dep)) {
                    stack.push(dep.clone());
                    node_deps.push(dep);
                }
            }
        }
        nodes.push(inner);
        producer_arcs.push(node_deps);
    }

    // In-degrees and consumer lists over the discovered subgraph.
    let mut indegree: Vec<usize> = vec![0; nodes.len()];
    let mut consumers: Vec<Vec<usize>> = vec![Vec::new(); nodes.len()];
    for (i, producers) in producer_arcs.iter().enumerate() {
        for p in producers {
            let pi = index_of[&Arc::as_ptr(p)];
            indegree[i] += 1;
            consumers[pi].push(i);
        }
    }

    // Kahn levels. Wave 0 = in-degree-0 nodes in discovery order; emit a
    // wave, decrement its consumers, the newly-zeroed ones form the next.
    let mut order: Vec<Arc<TensorInner>> = Vec::with_capacity(nodes.len());
    let mut wave: Vec<usize> = (0..nodes.len()).filter(|&i| indegree[i] == 0).collect();
    while !wave.is_empty() {
        let mut next: Vec<usize> = Vec::new();
        for &i in &wave {
            order.push(nodes[i].clone());
            for &c in &consumers[i] {
                indegree[c] -= 1;
                if indegree[c] == 0 {
                    next.push(c);
                }
            }
        }
        wave = next;
    }

    debug_assert_eq!(
        order.len(),
        nodes.len(),
        "Kahn schedule dropped nodes (cycle in a supposedly-acyclic graph?)"
    );
    order
}

/// Resolve a node to the op node that must be scheduled before any
/// consumer reading it, mirroring `visit`'s descent:
/// - an already-cached node is terminal (no dispatch, no dependency),
///   even if it carries an op — `visit` skips it identically;
/// - an op node (uncached) is itself the dependency;
/// - a view (uncached, no op) owns no storage, so chase its source;
/// - a placeholder (no cache, no op, no view) is terminal.
fn resolve_to_op(inner: &Arc<TensorInner>) -> Option<Arc<TensorInner>> {
    let mut current = inner.clone();
    loop {
        if current.cache.get().is_some() {
            return None;
        }
        if current.op.is_some() {
            return Some(current);
        }
        match current.view.as_ref() {
            Some(view) => current = view.source.inner.clone(),
            None => return None,
        }
    }
}

/// Ops encoded per command buffer before committing and opening the next.
/// One monolithic per-token command buffer runs its dependent dispatch
/// chain measurably slower than the same chain split across several
/// buffers (the queue pipelines buffer seams; consecutive encoders within
/// one buffer are driver-merged, so splitting encoders alone does
/// nothing). MLX commits every ~50 ops on M-class for the same reason
/// (`get_max_ops_mb_per_buffer`); a sweep of {25, 50, 100} here is flat
/// within run-to-run variance, so we hold MLX's 50. Cross-chunk ordering is
/// carried by the encoder fence chain (`cider_press_kernels::Commands`): the
/// first encoder of each new chunk waits the last fence published by the
/// prior chunk. This is load-bearing — buffers are allocated
/// hazard-untracked (no automatic command-buffer seam analysis), so the
/// fence chain is the *only* thing ordering one chunk against the next and
/// against the prior token's still-in-flight tail. Chunk size therefore
/// trades CPU encode granularity against how finely the GPU can pipeline
/// seams across tokens.
const OPS_PER_COMMAND_BUFFER: usize = 50;

/// Steps 3–4: index the nodes, then allocate each op's output and encode
/// its dispatch, committing a command buffer every
/// [`OPS_PER_COMMAND_BUFFER`] ops. Returns the per-node output buffers,
/// their pool tags (`Some(bytes)` ⇒ pool-minted; `None` ⇒ `SliceUpdate`
/// slab clone, never pooled), and the in-flight handles of every committed
/// chunk in submission order — none of them waited.
#[allow(clippy::type_complexity)]
fn encode_ops(
    order: &[Arc<TensorInner>],
    device: &crate::Device,
) -> Result<(Vec<Buffer<u8>>, Vec<Option<usize>>, Vec<CommandsInFlight>)> {
    let mut index_of: HashMap<*const TensorInner, usize> = HashMap::new();
    for (i, inner) in order.iter().enumerate() {
        index_of.insert(Arc::as_ptr(inner), i);
    }
    let mut outputs: Vec<Buffer<u8>> = Vec::with_capacity(order.len());
    let mut tags: Vec<Option<usize>> = Vec::with_capacity(order.len());
    let mut in_flight: Vec<CommandsInFlight> =
        Vec::with_capacity(order.len() / OPS_PER_COMMAND_BUFFER + 1);
    let mut commands = device.kernels().commands()?;
    let elide = barrier_elision_enabled();
    // Graph-level barrier elision (MLX's window-reset model). A
    // `memoryBarrier` on the concurrent encoder orders ALL dispatches before
    // it against ALL after it, so at any encode point only buffers written
    // since the LAST barrier can race a new dispatch — that working set is
    // `unsynced`. An op whose reads (and own dst) miss `unsynced` is
    // independent of everything since the last barrier ⇒ its barrier is
    // elided and its dst joins `unsynced`. A hazard keeps the barrier, which
    // syncs everything before it ⇒ `unsynced` collapses to just this op's
    // dst. The set is cleared at each chunk boundary: a fresh encoder's first
    // dispatch has no barrier and the fence chain orders prior chunks.
    let mut unsynced: HashSet<usize> = HashSet::new();
    for (i, inner) in order.iter().enumerate() {
        if i > 0 && i % OPS_PER_COMMAND_BUFFER == 0 {
            in_flight.push(commands.commit()?);
            commands = device.kernels().commands()?;
            unsynced.clear();
        }
        let op = inner
            .op
            .as_ref()
            .expect("topo invariant: only op nodes are in `order`");
        let (mut dst, tag) = if let OpKind::SliceUpdate { .. } = op.kind {
            let op_inputs = op.lock_inputs();
            let slab = op_inputs.first().ok_or_else(|| {
                Error::InvalidArgument("SliceUpdate: missing slab input (inputs[0])".into())
            })?;
            (dense_input_buffer(slab, &outputs, &index_of)?, None)
        } else {
            let byte_count = checked_byte_count(&inner.shape, inner.dtype)?;
            let (buf, pool_key_bytes) = device.alloc_pooled(byte_count)?;
            (buf, Some(pool_key_bytes))
        };
        if elide {
            let dst_key = buffer_key(&dst);
            let mut hazard = unsynced.contains(&dst_key);
            if !hazard {
                for input in op.lock_inputs().iter() {
                    if let Some(key) = input_buffer_key(input, &outputs, &index_of) {
                        // HAZARD_KEY is an unresolvable input: force a barrier
                        // (it is never a real, elidable buffer key). A resolved
                        // key is a hazard only if it was written since the last
                        // barrier (in `unsynced`).
                        if key == HAZARD_KEY || unsynced.contains(&key) {
                            hazard = true;
                            break;
                        }
                    }
                }
            }
            if hazard {
                // Barrier stays. It syncs everything encoded before it; only
                // this op's own (about-to-be-written) output is unsynced after.
                unsynced.clear();
                unsynced.insert(dst_key);
            } else {
                commands.elide_next_barrier();
                unsynced.insert(dst_key);
            }
        }
        dispatch(inner, &mut commands, &outputs, &mut dst, &index_of)?;
        outputs.push(dst);
        tags.push(tag);
    }
    in_flight.push(commands.commit()?);
    Ok((outputs, tags, in_flight))
}

/// Step 6: populate each node's result cache from its output buffer.
/// The buffer handle is valid immediately after allocation; its bytes are only
/// valid after `commit_and_wait` (or equivalent GPU completion). The sync
/// path orders this after the wait; the async path sets each node's
/// `in_flight` flag first, so host reads stay gated until `PendingEval`
/// observes completion.
fn populate_caches(
    order: &[Arc<TensorInner>],
    outputs: Vec<Buffer<u8>>,
    tags: Vec<Option<usize>>,
    device: &crate::Device,
) {
    for ((inner, buffer), tag) in order.iter().zip(outputs).zip(tags) {
        let storage = match tag {
            Some(bytes) => LeafStorage::Dense(device.pooled(buffer, bytes)),
            None => LeafStorage::Dense(crate::buffer_pool::PooledBuffer::unpooled(buffer)),
        };
        let _ = inner.cache.set(storage);
    }
}

/// Detach every just-evaluated node from its producers (MLX's
/// detach-on-eval). Clears each op node's inputs so an evaluated tensor no
/// longer retains its upstream graph; its own cache/buffer is untouched.
/// Call only after the nodes' caches are populated.
///
/// In the async path (`eval_async`) this runs before GPU completion; in-flight
/// buffer safety is handled by `PendingEval` pinning `order` and by the encoder
/// fence chain for cross-eval inputs (see `PooledBuffer` doc).
fn detach_order(order: &[Arc<TensorInner>]) {
    for inner in order {
        let op = inner
            .op
            .as_ref()
            .expect("topo invariant: only op nodes are in `order`");
        op.detach();
    }
}

/// Commit `root`'s graph without waiting. Populates result caches at commit
/// time (buffer handles valid now; bytes valid after the returned handle is
/// waited) and returns a [`PendingEval`](crate::PendingEval) owning the
/// in-flight command buffers plus the topo order (pinning the in-flight
/// pooled buffers). See `pending_eval` for the host-read contract.
pub(crate) fn eval_async(root: &Tensor) -> Result<crate::PendingEval> {
    let _span = crate::profile::span("tensor.eval_async");
    let order = build_order(root);
    if order.is_empty() {
        return Ok(crate::PendingEval::empty());
    }
    let device = root.inner.device.as_ref().ok_or_else(|| {
        Error::InvalidArgument("eval_async: placeholder has no device to dispatch on".into())
    })?;
    let (outputs, tags, in_flight) = encode_ops(&order, device)?;
    // Flag before populating: a populated cache with the flag set means
    // "buffer exists, bytes not host-readable" (resolve_leaf returns None).
    // GPU-side chaining reads the cache directly and is unaffected.
    for inner in &order {
        inner
            .in_flight
            .store(true, std::sync::atomic::Ordering::Release);
    }
    populate_caches(&order, outputs, tags, device);
    // Detach before the GPU finishes. A cross-eval input whose last Arc drops
    // here returns its buffer to the pool mid-flight; reuse by the next eval is
    // safe via the encoder fence chain. See the `PooledBuffer` doc for the full
    // safety argument.
    detach_order(&order);
    Ok(crate::PendingEval::new(in_flight, order))
}

pub(crate) fn eval(root: &Tensor) -> Result<()> {
    let _span = crate::profile::span("tensor.eval");
    let sp_encode =
        crate::profile::signpost::interval_begin(crate::profile::signpost::Region::Encode);
    let encode = crate::profile::span("tensor.eval.encode");
    let order = build_order(root);
    if order.is_empty() {
        crate::profile::signpost::interval_end(sp_encode);
        return Ok(()); // every reachable node is already terminal.
    }
    let device = root.inner.device.as_ref().ok_or_else(|| {
        Error::InvalidArgument("eval: placeholder has no device to dispatch on".into())
    })?;
    let (outputs, tags, in_flight) = encode_ops(&order, device)?;
    drop(encode);
    crate::profile::signpost::interval_end(sp_encode);

    let sp_wait = crate::profile::signpost::interval_begin(crate::profile::signpost::Region::Wait);
    let wait = crate::profile::span("tensor.eval.wait");
    for chunk in in_flight {
        chunk.wait()?;
    }
    drop(wait);
    crate::profile::signpost::interval_end(sp_wait);

    populate_caches(&order, outputs, tags, device);
    detach_order(&order);
    Ok(())
}

/// Profiling-only eval: one sampled compute encoder per dispatch, so GPU
/// time can be attributed per [`OpKind`]. Perturbs the encode regime
/// (extra encoders, lost cross-dispatch overlap) — read its GPU
/// breakdown alongside the production `tensor.eval.encode`/`.wait` spans,
/// never as a replacement. Errors if the device lacks stage-boundary
/// counter sampling.
pub(crate) fn profiled_eval(root: &Tensor) -> Result<()> {
    let order = build_order(root);
    if order.is_empty() {
        return Ok(());
    }
    let device =
        root.inner.device.as_ref().ok_or_else(|| {
            Error::InvalidArgument("profiled_eval: placeholder has no device".into())
        })?;
    if !device.kernels().supports_stage_boundary_sampling() {
        return Err(Error::InvalidArgument(
            "profiled_eval: device does not support stage-boundary counter sampling".into(),
        ));
    }
    let mut commands = device.kernels().commands_profiled(order.len())?;
    let mut index_of: HashMap<*const TensorInner, usize> = HashMap::new();
    for (i, inner) in order.iter().enumerate() {
        index_of.insert(Arc::as_ptr(inner), i);
    }
    let mut outputs: Vec<Buffer<u8>> = Vec::with_capacity(order.len());
    // Per-output pool tag: `Some(bytes)` ⇒ pool-minted, return to pool on
    // drop; `None` ⇒ SliceUpdate slab clone (caller-owned), never pooled.
    let mut tags: Vec<Option<usize>> = Vec::with_capacity(order.len());
    for inner in &order {
        let op = inner
            .op
            .as_ref()
            .expect("topo invariant: only op nodes are in `order`");
        commands.begin_profiled_op(op_kind_label(&op.kind));
        let (mut dst, tag) = if let OpKind::SliceUpdate { .. } = op.kind {
            let op_inputs = op.lock_inputs();
            let slab = op_inputs.first().ok_or_else(|| {
                Error::InvalidArgument("SliceUpdate: missing slab input (inputs[0])".into())
            })?;
            (dense_input_buffer(slab, &outputs, &index_of)?, None)
        } else {
            let byte_count = checked_byte_count(&inner.shape, inner.dtype)?;
            let (buf, pool_key_bytes) = device.alloc_pooled(byte_count)?;
            (buf, Some(pool_key_bytes))
        };
        dispatch(inner, &mut commands, &outputs, &mut dst, &index_of)?;
        outputs.push(dst);
        tags.push(tag);
    }

    let segments = commands.commit_wait_resolve()?;
    let period_ns = device.kernels().gpu_timestamp_period_ns();
    for seg in &segments {
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss
        )]
        let ns = (seg.ticks() as f64 * period_ns) as u64;
        crate::profile::record_gpu(gpu_key(seg.label), ns);
    }

    populate_caches(&order, outputs, tags, device);
    detach_order(&order);
    Ok(())
}

// Arms must match op_kind_label's output. A new OpKind variant compiles
// without touching this (the compiler only enforces op_kind_label), so
// it silently buckets to "gpu.other" until an arm is added here.
/// Map an op-kind label to its `gpu.<label>` profile key. Static strings
/// so the profile accumulator keys stay `&'static str`.
fn gpu_key(label: &'static str) -> &'static str {
    match label {
        "copy" => "gpu.copy",
        "quantized_matmul" => "gpu.quantized_matmul",
        "binary" => "gpu.binary",
        "unary" => "gpu.unary",
        "reduce" => "gpu.reduce",
        "gather" => "gpu.gather",
        "dequantize" => "gpu.dequantize",
        "rope" => "gpu.rope",
        "rms_norm" => "gpu.rms_norm",
        "softmax" => "gpu.softmax",
        "matmul" => "gpu.matmul",
        "slice_update" => "gpu.slice_update",
        "sdpa" => "gpu.sdpa",
        "arg_reduce" => "gpu.arg_reduce",
        _ => "gpu.other",
    }
}

/// Whether graph-level barrier elision is active. ON unless the
/// environment variable `CIDER_PRESS_NO_BARRIER_ELISION` is exactly
/// `"1"` (a bisection kill switch that restores barrier-per-dispatch).
/// Read once and cached.
fn barrier_elision_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("CIDER_PRESS_NO_BARRIER_ELISION").as_deref() != Ok("1"))
}

/// Metal-buffer identity for hazard tracking: the address of the
/// `MTLBuffer` protocol object, stable for the buffer's life.
fn buffer_key(buf: &Buffer<u8>) -> usize {
    std::ptr::from_ref(buf.metal_buffer()).cast::<()>() as usize
}

/// Sentinel key meaning "input could not be resolved to a buffer
/// identity" — the encode loop treats it as a forced hazard (barrier
/// stays). `0` is never a valid objc object pointer, so it can never
/// collide with a real buffer key.
const HAZARD_KEY: usize = 0;

/// Resolve an op input to the Metal-buffer identity hazard tracking keys
/// it by, mirroring exactly how `dense_input_buffer` / `resolve_view_storage`
/// resolve the *same* input for dispatch:
///
/// - op node in this eval ⇒ its allocated `outputs[i]` buffer;
/// - view chain ⇒ the underlying storage owner's buffer (offset ignored —
///   we key the whole buffer, the coarsest-correct granularity);
/// - cached dense leaf / prior-eval op ⇒ its dense buffer (prior-eval
///   writes are fence-ordered, so the key simply won't be in `unsynced`);
/// - quantized leaf ⇒ `None`: qmv/qmm/dequantize weight slabs are uploaded
///   host-side and never written by any dispatch, so they cannot race.
///
/// Returns `Some(HAZARD_KEY)` for anything it cannot confidently resolve
/// (the conservative fallback: barrier stays).
fn input_buffer_key(
    input: &Tensor,
    outputs: &[Buffer<u8>],
    index_of: &HashMap<*const TensorInner, usize>,
) -> Option<usize> {
    let mut current = &input.inner;
    // Walk any view chain to the storage owner (offset irrelevant — whole
    // buffer is keyed).
    while let Some(view) = current.view.as_ref() {
        current = &view.source.inner;
    }
    if let Some(&i) = index_of.get(&Arc::as_ptr(current)) {
        return match outputs.get(i) {
            Some(buf) => Some(buffer_key(buf)),
            // Topo order guarantees this is allocated; if not, be safe.
            None => Some(HAZARD_KEY),
        };
    }
    match current.cache.get() {
        // SAFETY: `Buffer<u8>` → `Buffer<u8>` reinterpret is trivially valid
        // (any bit pattern is a valid `u8`, lengths match). This is a refcount
        // bump over the same underlying `MTLBuffer`, used only to read the
        // buffer pointer for hazard-tracking key derivation; the handle is
        // dropped immediately after `buffer_key` returns.
        Some(LeafStorage::Dense(buf)) => Some(buffer_key(&unsafe { buf.reinterpret_as::<u8>() })),
        // Quantized weight slabs are never written by a dispatch ⇒ skip.
        Some(LeafStorage::Quantized { .. }) => None,
        // Placeholder mid-graph: unresolvable ⇒ hazard.
        None => Some(HAZARD_KEY),
    }
}

/// Resolve a dense op input's backing buffer.
///
/// Returns a fresh `Buffer<u8>` handle (refcount bump via
/// `reinterpret_as::<u8>`) so the caller doesn't have to juggle
/// borrows from two different lifetime sources (the in-progress
/// `outputs` vec and the input's `OnceLock` cache). The handle
/// references the same underlying `MTLBuffer`; the original storage
/// stays alive via the input's `Arc<TensorInner>`.
fn dense_input_buffer(
    input: &Tensor,
    outputs: &[Buffer<u8>],
    index_of: &HashMap<*const TensorInner, usize>,
) -> Result<Buffer<u8>> {
    let src_ref: &Buffer<u8> = if let Some(&i) = index_of.get(&Arc::as_ptr(&input.inner)) {
        // Topo order guarantees `i < current_index`, so `outputs[i]`
        // is already populated. If it isn't, the walk is wrong.
        outputs.get(i).ok_or_else(|| {
            Error::InvalidArgument(
                "eval: dispatcher saw an input whose output hadn't been allocated yet \
                 (topo order bug?)"
                    .into(),
            )
        })?
    } else {
        // Not in this eval's dispatch set ⇒ must already be cached
        // (host-constructed leaf or previously-evaluated op).
        match input.inner.cache.get() {
            Some(LeafStorage::Dense(buf)) => buf,
            Some(LeafStorage::Quantized { .. }) => {
                return Err(Error::InvalidArgument(
                    "eval: dense input expected, got quantized leaf".into(),
                ));
            }
            None => {
                return Err(Error::InvalidArgument(
                    "eval: op input is neither cached nor being dispatched in this eval \
                     (placeholder used as input?)"
                        .into(),
                ));
            }
        }
    };
    // SAFETY: `Buffer<u8>` → `Buffer<u8>` reinterpret is trivially
    // valid (any bit pattern is a valid `u8`, lengths match). This
    // just bumps the Metal-buffer refcount so the caller can own a
    // typed handle independent of where the source borrow came from.
    Ok(unsafe { src_ref.reinterpret_as::<u8>() })
}

/// Borrow the three buffers of a quantized leaf, with refcount-bumped
/// handles so the caller doesn't deal with `OnceLock` lifetimes.
fn quantized_input_buffers(input: &Tensor) -> Result<(Buffer<u8>, Buffer<u8>, Buffer<u8>)> {
    match input.inner.cache.get() {
        Some(LeafStorage::Quantized {
            weights,
            scales,
            biases,
        }) => {
            // SAFETY: u8 → u8 refcount bumps, as in `dense_input_buffer`.
            let w = unsafe { weights.reinterpret_as::<u8>() };
            let s = unsafe { scales.reinterpret_as::<u8>() };
            let b = unsafe { biases.reinterpret_as::<u8>() };
            Ok((w, s, b))
        }
        Some(LeafStorage::Dense(_)) => Err(Error::InvalidArgument(
            "eval: QuantizedMatMul expected a quantized weight leaf, found dense".into(),
        )),
        None => Err(Error::InvalidArgument(
            "eval: QuantizedMatMul weight input is not materialized (placeholder or unevaluated op?)".into(),
        )),
    }
}

fn dispatch(
    inner: &Arc<TensorInner>,
    commands: &mut Commands<'_>,
    outputs: &[Buffer<u8>],
    dst: &mut Buffer<u8>,
    index_of: &HashMap<*const TensorInner, usize>,
) -> Result<()> {
    let op = inner
        .op
        .as_ref()
        .expect("topo invariant: only op nodes here");
    match op.kind {
        OpKind::Copy => dispatch_copy(inner, op, commands, outputs, dst, index_of),
        OpKind::QuantizedMatMul {
            group_size,
            bits,
            transpose,
        } => dispatch_quantized_matmul(
            inner, op, commands, outputs, dst, index_of, group_size, bits, transpose,
        ),
        OpKind::Binary { op: binary_op } => {
            dispatch_binary(inner, op, commands, outputs, dst, index_of, binary_op)
        }
        OpKind::Unary { op: unary_op } => {
            dispatch_unary(inner, op, commands, outputs, dst, index_of, unary_op)
        }
        OpKind::Reduce { kind, keep_dim } => {
            dispatch_reduce(inner, op, commands, outputs, dst, index_of, kind, keep_dim)
        }
        OpKind::Gather => dispatch_gather(inner, op, commands, outputs, dst, index_of),
        OpKind::Dequantize { group_size, bits } => dispatch_dequantize(
            inner, op, commands, outputs, dst, index_of, group_size, bits,
        ),
        OpKind::Rope {
            base_log2,
            scale,
            rotary_dims,
        } => dispatch_rope(
            inner,
            op,
            commands,
            outputs,
            dst,
            index_of,
            base_log2,
            scale,
            rotary_dims,
        ),
        OpKind::RmsNorm { eps } => {
            dispatch_rms_norm(inner, op, commands, outputs, dst, index_of, eps)
        }
        OpKind::Softmax { precise } => {
            dispatch_softmax(inner, op, commands, outputs, dst, index_of, precise)
        }
        OpKind::MatMul => dispatch_matmul(inner, op, commands, outputs, dst, index_of),
        OpKind::SliceUpdate { offset_rows } => {
            dispatch_slice_update(inner, op, commands, outputs, dst, index_of, offset_rows)
        }
        OpKind::Sdpa {
            scale,
            gqa_factor,
            causal,
        } => dispatch_sdpa(
            inner, op, commands, outputs, dst, index_of, scale, gqa_factor, causal,
        ),
        OpKind::ArgReduce { kind, axis } => {
            dispatch_arg_reduce(inner, op, commands, outputs, dst, index_of, kind, axis)
        }
    }
}

fn dispatch_copy(
    inner: &Arc<TensorInner>,
    op: &OpNode,
    commands: &mut Commands<'_>,
    outputs: &[Buffer<u8>],
    dst: &mut Buffer<u8>,
    index_of: &HashMap<*const TensorInner, usize>,
) -> Result<()> {
    let device = inner
        .device
        .as_ref()
        .expect("op nodes are always constructed with a device");
    let inputs = op.lock_inputs();
    let input = inputs
        .first()
        .expect("Copy has exactly one input by construction");
    let library = device.copy_library()?;

    // Views own no storage; resolving them yields the storage owner
    // plus the cumulative byte offset to the view's element [0,..]. A
    // view always takes the strided path because its element order
    // (transpose, broadcast, sliced inner dim) is not generally
    // byte-equivalent to its leaf, and a non-zero byte offset can't be
    // expressed through the vector kernel anyway.
    if input.inner.view.is_some() {
        return dispatch_copy_strided(input, inner, commands, outputs, dst, index_of, library);
    }

    let src = dense_input_buffer(input, outputs, index_of)?;

    // SAFETY (every arm below): the runtime's dtype tag is the source
    // of truth for the byte contents; reinterpreting `Buffer<u8>` as
    // the typed view matches what the kernel expects. Byte length
    // divides evenly because the buffers were sized as
    // `elem_count * size_bytes` at allocation.
    match inner.dtype {
        DType::F32 => {
            let src_typed = unsafe { src.reinterpret_as::<f32>() };
            let mut dst_typed = unsafe { dst.reinterpret_as::<f32>() };
            copy::copy_v_f32(commands, library, &src_typed, &mut dst_typed)?;
        }
        DType::F16 => {
            let src_typed = unsafe { src.reinterpret_as::<half::f16>() };
            let mut dst_typed = unsafe { dst.reinterpret_as::<half::f16>() };
            copy::copy_v_f16(commands, library, &src_typed, &mut dst_typed)?;
        }
        DType::BF16 => {
            let src_typed = unsafe { src.reinterpret_as::<half::bf16>() };
            let mut dst_typed = unsafe { dst.reinterpret_as::<half::bf16>() };
            copy::copy_v_bf16(commands, library, &src_typed, &mut dst_typed)?;
        }
        DType::I32 => {
            let src_typed = unsafe { src.reinterpret_as::<i32>() };
            let mut dst_typed = unsafe { dst.reinterpret_as::<i32>() };
            copy::copy_v_i32(commands, library, &src_typed, &mut dst_typed)?;
        }
        DType::U32 => {
            let src_typed = unsafe { src.reinterpret_as::<u32>() };
            let mut dst_typed = unsafe { dst.reinterpret_as::<u32>() };
            copy::copy_v_u32(commands, library, &src_typed, &mut dst_typed)?;
        }
    }

    Ok(())
}

/// Dispatch the spike `SliceUpdate`: write `src` (dense bf16 rank-3,
/// contiguous) into the slab buffer bound as `dst` at byte offset
/// `offset_rows * rows_stride`. Reuses MLX's `copy_gg` family via
/// `copy_g_bf16`, which takes a destination byte offset. The slab's
/// other rows are untouched (the kernel writes only `src.elem_count()`
/// elements starting at the offset).
#[allow(clippy::too_many_arguments)]
fn dispatch_slice_update(
    inner: &Arc<TensorInner>,
    op: &OpNode,
    commands: &mut Commands<'_>,
    outputs: &[Buffer<u8>],
    dst: &mut Buffer<u8>,
    index_of: &HashMap<*const TensorInner, usize>,
    offset_rows: usize,
) -> Result<()> {
    let device = inner
        .device
        .as_ref()
        .expect("op nodes are always constructed with a device");
    // Preconditions (bf16, rank-3, matching row shape, in-bounds offset)
    // are enforced by `Tensor::slice_update` at construction; re-assert
    // the dtype in debug builds as a tripwire.
    debug_assert_eq!(
        inner.dtype,
        DType::BF16,
        "slice_update dispatch: non-bf16 reached eval"
    );
    let inputs = op.lock_inputs();
    let src = inputs.get(1).ok_or_else(|| {
        Error::InvalidArgument("SliceUpdate: missing src input (inputs[1])".into())
    })?;
    let src_dims = src.shape().dims();
    let row_elems = src_dims[1] * src_dims[2];
    let dst_byte_offset = offset_rows * row_elems * DType::BF16.size_bytes();

    // `src` is typically a contiguous reshape *view* of a copy op (the
    // `k_upd`/`v_upd` produced by attention), so resolve through the view
    // chain like the matmul path does — `dense_input_buffer` only handles
    // direct op outputs / cached leaves and would reject a view. The
    // contiguous-offset-0 assertion in `matmul_input_bytes` holds because
    // the reshape just reinterprets a dense buffer's shape.
    let src_bytes = matmul_input_bytes("slice_update", src, outputs, index_of)?;
    // SAFETY: the dtype guard above pinned this op to BF16; the buffers
    // were sized as `elem_count * size_bytes`, so the byte length divides
    // evenly into the bf16 element count. dst is the slab buffer; the
    // copy writes only src.elem_count() elements starting at the offset.
    let src_typed = unsafe { src_bytes.reinterpret_as::<half::bf16>() };
    let mut dst_typed = unsafe { dst.reinterpret_as::<half::bf16>() };

    let shape_i32 = shape_to_i32(src_dims)?;
    let strides = contiguous_strides_i64(&shape_i32);

    let library = device.copy_library()?;
    copy::copy_g_bf16(
        commands,
        library,
        &src_typed,
        0,
        &strides,
        &mut dst_typed,
        dst_byte_offset,
        &strides,
        &shape_i32,
    )?;
    Ok(())
}

/// Strided same-dtype copy: input is a view onto some storage owner;
/// gather it element-wise into a fresh contiguous dst via MLX's
/// `copy_gg*` family. Mirrors what MLX's `copy_gpu` does for
/// `CopyType::General` reshapes of non-contiguous arrays.
#[allow(clippy::too_many_arguments)]
fn dispatch_copy_strided(
    input: &Tensor,
    inner: &Arc<TensorInner>,
    commands: &mut Commands<'_>,
    outputs: &[Buffer<u8>],
    dst: &mut Buffer<u8>,
    index_of: &HashMap<*const TensorInner, usize>,
    library: &KernelLibrary,
) -> Result<()> {
    let (src_bytes, src_byte_offset) = resolve_view_storage(input, outputs, index_of)?;
    let src_strides = match input.layout() {
        Layout::Dense { strides } => strides_to_i64(strides.as_slice())?,
        Layout::Quantized(_) => {
            return Err(Error::InvalidArgument(
                "eval: strided copy from a quantized view is not supported".into(),
            ));
        }
    };
    let shape_i32 = shape_to_i32(input.shape().dims())?;
    let dst_strides = contiguous_strides_i64(&shape_i32);

    match inner.dtype {
        DType::F32 => {
            let src = unsafe { src_bytes.reinterpret_as::<f32>() };
            let mut dst_typed = unsafe { dst.reinterpret_as::<f32>() };
            copy::copy_g_f32(
                commands,
                library,
                &src,
                src_byte_offset,
                &src_strides,
                &mut dst_typed,
                0,
                &dst_strides,
                &shape_i32,
            )?;
        }
        DType::F16 => {
            let src = unsafe { src_bytes.reinterpret_as::<half::f16>() };
            let mut dst_typed = unsafe { dst.reinterpret_as::<half::f16>() };
            copy::copy_g_f16(
                commands,
                library,
                &src,
                src_byte_offset,
                &src_strides,
                &mut dst_typed,
                0,
                &dst_strides,
                &shape_i32,
            )?;
        }
        DType::BF16 => {
            let src = unsafe { src_bytes.reinterpret_as::<half::bf16>() };
            let mut dst_typed = unsafe { dst.reinterpret_as::<half::bf16>() };
            copy::copy_g_bf16(
                commands,
                library,
                &src,
                src_byte_offset,
                &src_strides,
                &mut dst_typed,
                0,
                &dst_strides,
                &shape_i32,
            )?;
        }
        DType::I32 => {
            let src = unsafe { src_bytes.reinterpret_as::<i32>() };
            let mut dst_typed = unsafe { dst.reinterpret_as::<i32>() };
            copy::copy_g_i32(
                commands,
                library,
                &src,
                src_byte_offset,
                &src_strides,
                &mut dst_typed,
                0,
                &dst_strides,
                &shape_i32,
            )?;
        }
        DType::U32 => {
            let src = unsafe { src_bytes.reinterpret_as::<u32>() };
            let mut dst_typed = unsafe { dst.reinterpret_as::<u32>() };
            copy::copy_g_u32(
                commands,
                library,
                &src,
                src_byte_offset,
                &src_strides,
                &mut dst_typed,
                0,
                &dst_strides,
                &shape_i32,
            )?;
        }
    }
    Ok(())
}

/// Walk a view chain to the storage owner; return a refcount-bumped
/// `Buffer<u8>` plus the cumulative byte offset to the view's
/// element `[0, 0, …]`.
///
/// The storage owner is either (a) an op output that was just
/// allocated earlier in this eval call — looked up via `index_of` —
/// or (b) a previously-cached leaf / op (host upload or earlier
/// `eval`). Hitting a placeholder mid-chain means eval was called on
/// a graph that wasn't fully constructed; we surface that as an error.
fn resolve_view_storage(
    input: &Tensor,
    outputs: &[Buffer<u8>],
    index_of: &HashMap<*const TensorInner, usize>,
) -> Result<(Buffer<u8>, usize)> {
    let mut current = &input.inner;
    let mut offset: usize = 0;
    loop {
        if let Some(view) = current.view.as_ref() {
            offset = offset.checked_add(view.byte_offset).ok_or_else(|| {
                Error::InvalidArgument("eval: view-chain byte offset overflowed usize".into())
            })?;
            current = &view.source.inner;
            continue;
        }
        if let Some(&i) = index_of.get(&Arc::as_ptr(current)) {
            let buf = outputs.get(i).ok_or_else(|| {
                Error::InvalidArgument(
                    "eval: strided-copy input's storage owner has no allocated output yet \
                     (topo order bug?)"
                        .into(),
                )
            })?;
            // SAFETY: u8→u8 reinterpret is trivially valid; this is a
            // refcount bump so the caller owns a handle independent of
            // the borrow on `outputs`.
            return Ok((unsafe { buf.reinterpret_as::<u8>() }, offset));
        }
        match current.cache.get() {
            Some(LeafStorage::Dense(buf)) => {
                // SAFETY: as above.
                return Ok((unsafe { buf.reinterpret_as::<u8>() }, offset));
            }
            Some(LeafStorage::Quantized { .. }) => {
                return Err(Error::InvalidArgument(
                    "eval: strided copy from a quantized leaf is not supported".into(),
                ));
            }
            None => {
                return Err(Error::InvalidArgument(
                    "eval: view chain terminates at a placeholder (no storage to copy from)".into(),
                ));
            }
        }
    }
}

fn strides_to_i64(strides: &[isize]) -> Result<Vec<i64>> {
    strides
        .iter()
        .map(|&s| {
            i64::try_from(s).map_err(|_| {
                Error::InvalidArgument(format!("eval: stride {s} does not fit in i64"))
            })
        })
        .collect()
}

fn shape_to_i32(dims: &[usize]) -> Result<Vec<i32>> {
    dims.iter()
        .map(|&d| {
            i32::try_from(d).map_err(|_| {
                Error::InvalidArgument(format!("eval: shape dim {d} does not fit in i32"))
            })
        })
        .collect()
}

fn contiguous_strides_i64(shape: &[i32]) -> Vec<i64> {
    let mut strides = vec![1i64; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        strides[d] = strides[d + 1] * i64::from(shape[d + 1]);
    }
    strides
}

#[allow(clippy::too_many_arguments, clippy::many_single_char_names)]
fn dispatch_quantized_matmul(
    inner: &Arc<TensorInner>,
    op: &OpNode,
    commands: &mut Commands<'_>,
    outputs: &[Buffer<u8>],
    dst: &mut Buffer<u8>,
    index_of: &HashMap<*const TensorInner, usize>,
    group_size: u32,
    bits: u8,
    transpose: bool,
) -> Result<()> {
    if !transpose {
        // `Tensor::quantized_matmul` rejects `transpose=false` at construction;
        // a runtime guard here keeps release builds honest if the construction
        // check ever drifts.
        return Err(Error::InvalidArgument(
            "QuantizedMatMul: transpose=false is not yet implemented in the dispatcher".into(),
        ));
    }
    let device = inner
        .device
        .as_ref()
        .expect("op nodes are always constructed with a device");
    let inputs = op.lock_inputs();
    let weight = inputs.first().ok_or_else(|| {
        Error::InvalidArgument("QuantizedMatMul: missing weight input (inputs[0])".into())
    })?;
    let x_tensor = inputs.get(1).ok_or_else(|| {
        Error::InvalidArgument("QuantizedMatMul: missing activation input (inputs[1])".into())
    })?;

    let (w_bytes, scales_bytes, biases_bytes) = quantized_input_buffers(weight)?;
    // The activation may be a contiguous *view* (e.g. a `reshape` of an
    // op output) — `Tensor::quantized_matmul` accepts those. Resolve
    // through the view chain like the dense matmul path does; the qmv/qmm
    // kernels read contiguous bytes from offset 0, which `matmul_input_bytes`
    // enforces.
    let x_bytes = matmul_input_bytes("quantized_matmul", x_tensor, outputs, index_of)?;

    let library = device.quantized_library()?;

    // SAFETY: dtype tags are the source of truth for what's in each
    // byte buffer. `Quantization::from_bytes` validated the byte
    // counts at construction; the kernel wraps further validation
    // around the typed lengths.
    let w_typed = unsafe { w_bytes.reinterpret_as::<u32>() };
    let scales_typed = unsafe { scales_bytes.reinterpret_as::<half::bf16>() };
    let biases_typed = unsafe { biases_bytes.reinterpret_as::<half::bf16>() };
    let x_typed = unsafe { x_bytes.reinterpret_as::<half::bf16>() };
    let mut y_typed = unsafe { dst.reinterpret_as::<half::bf16>() };

    // Compute M from the activation's shape. Rank-1 activation is the
    // M=1 decode case; for rank ≥ 2 collapse all dims except the last
    // two (K) into the batch/leading dims and multiply by the second-
    // to-last dim to get M.
    let x_dims = x_tensor.shape().dims();
    let m: usize = if x_dims.len() == 1 {
        1
    } else {
        let lead: usize = x_dims[..x_dims.len() - 2].iter().product::<usize>().max(1);
        lead * x_dims[x_dims.len() - 2]
    };
    // N = weight[0], K = weight[1]; validated at op construction.
    let w_dims = weight.shape().dims();
    let n = w_dims[0];
    let k = w_dims[1];

    if m == 1 {
        qmv::affine_qmv_bf16(
            commands,
            library,
            &w_typed,
            &scales_typed,
            &biases_typed,
            &x_typed,
            &mut y_typed,
            k,
            group_size,
            u32::from(bits),
        )?;
    } else {
        // Padded weights are decode-only (M=1); an M>1 activation against a
        // padded weight self-rejects in `validate_qmm_buffer_lens` because
        // x.len() == M * k_logical != M * k_physical.
        qmm::affine_qmm_t_bf16(
            commands,
            library,
            &w_typed,
            &scales_typed,
            &biases_typed,
            &x_typed,
            &mut y_typed,
            m,
            n,
            k,
            group_size,
            u32::from(bits),
        )?;
    }

    Ok(())
}

/// Dispatch an element-wise binary op (Add / Mul).
///
/// Picks one of MLX's two `binary.metal` families:
///
/// - `vv_<Op><dtype>` when both inputs are non-view dense tensors with
///   the same shape as the output (typical residual add / Hadamard
///   product). One contiguous dispatch, no stride bookkeeping.
/// - `g{1,2,3}_<Op><dtype>` when at least one input is a view, which
///   is exactly the broadcast case after `Tensor::binary` runs
///   `broadcast_to` to align operand ranks. Strides come from each
///   input's [`Layout::Dense`] entries (zero-stride axes encode the
///   broadcast); the output is dense row-major.
///
/// Rank > 3 in the strided case is rejected for now (`gn2_` /
/// `gn4large_` not wired). The runtime hasn't needed a binary op with
/// rank-4 broadcast yet — Qwen2 attention/MLP operate at rank 3.
fn dispatch_binary(
    inner: &Arc<TensorInner>,
    op: &OpNode,
    commands: &mut Commands<'_>,
    outputs: &[Buffer<u8>],
    dst: &mut Buffer<u8>,
    index_of: &HashMap<*const TensorInner, usize>,
    binary_op: BinaryOp,
) -> Result<()> {
    let device = inner
        .device
        .as_ref()
        .expect("op nodes are always constructed with a device");
    let inputs = op.lock_inputs();
    let lhs = inputs
        .first()
        .ok_or_else(|| Error::InvalidArgument("Binary: missing lhs input (inputs[0])".into()))?;
    let rhs = inputs
        .get(1)
        .ok_or_else(|| Error::InvalidArgument("Binary: missing rhs input (inputs[1])".into()))?;
    let library = device.binary_library()?;

    let is_strided = lhs.inner.view.is_some() || rhs.inner.view.is_some();

    if !is_strided {
        return dispatch_binary_vv(
            inner, lhs, rhs, commands, outputs, dst, index_of, library, binary_op,
        );
    }
    dispatch_binary_strided(
        inner, lhs, rhs, commands, outputs, dst, index_of, library, binary_op,
    )
}

#[allow(clippy::too_many_arguments)]
fn dispatch_binary_vv(
    inner: &Arc<TensorInner>,
    lhs: &Tensor,
    rhs: &Tensor,
    commands: &mut Commands<'_>,
    outputs: &[Buffer<u8>],
    dst: &mut Buffer<u8>,
    index_of: &HashMap<*const TensorInner, usize>,
    library: &KernelLibrary,
    binary_op: BinaryOp,
) -> Result<()> {
    let a = dense_input_buffer(lhs, outputs, index_of)?;
    let b = dense_input_buffer(rhs, outputs, index_of)?;

    // SAFETY: dtype tag drives the reinterpret. Allocation sized the
    // buffers as `elem_count * size_bytes`, so byte lengths divide
    // evenly into the typed element count.
    match inner.dtype {
        DType::F32 => {
            let a_t = unsafe { a.reinterpret_as::<f32>() };
            let b_t = unsafe { b.reinterpret_as::<f32>() };
            let mut c_t = unsafe { dst.reinterpret_as::<f32>() };
            match binary_op {
                BinaryOp::Add => binary::vv_add_f32(commands, library, &a_t, &b_t, &mut c_t)?,
                BinaryOp::Mul => binary::vv_mul_f32(commands, library, &a_t, &b_t, &mut c_t)?,
            }
        }
        DType::F16 => {
            let a_t = unsafe { a.reinterpret_as::<half::f16>() };
            let b_t = unsafe { b.reinterpret_as::<half::f16>() };
            let mut c_t = unsafe { dst.reinterpret_as::<half::f16>() };
            match binary_op {
                BinaryOp::Add => binary::vv_add_f16(commands, library, &a_t, &b_t, &mut c_t)?,
                BinaryOp::Mul => binary::vv_mul_f16(commands, library, &a_t, &b_t, &mut c_t)?,
            }
        }
        DType::BF16 => {
            let a_t = unsafe { a.reinterpret_as::<half::bf16>() };
            let b_t = unsafe { b.reinterpret_as::<half::bf16>() };
            let mut c_t = unsafe { dst.reinterpret_as::<half::bf16>() };
            match binary_op {
                BinaryOp::Add => binary::vv_add_bf16(commands, library, &a_t, &b_t, &mut c_t)?,
                BinaryOp::Mul => binary::vv_mul_bf16(commands, library, &a_t, &b_t, &mut c_t)?,
            }
        }
        dtype @ (DType::I32 | DType::U32) => {
            return Err(Error::InvalidArgument(format!(
                "binary: dtype {dtype:?} not wired for Add/Mul yet (only float dtypes are \
                 instantiated; integer instantiations exist in binary.metal but the typed \
                 dispatch fns aren't exposed)",
            )));
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn dispatch_binary_strided(
    inner: &Arc<TensorInner>,
    lhs: &Tensor,
    rhs: &Tensor,
    commands: &mut Commands<'_>,
    outputs: &[Buffer<u8>],
    dst: &mut Buffer<u8>,
    index_of: &HashMap<*const TensorInner, usize>,
    library: &KernelLibrary,
    binary_op: BinaryOp,
) -> Result<()> {
    let rank = inner.shape.dims().len();
    if rank == 0 || rank > 3 {
        return Err(Error::InvalidArgument(format!(
            "binary: strided dispatch for rank {rank} is not wired (only g1/g2/g3 instantiated; \
             gn2_/gn4large_ are deferred)",
        )));
    }

    let (a_bytes, a_byte_offset) = resolve_view_storage(lhs, outputs, index_of)?;
    let (b_bytes, b_byte_offset) = resolve_view_storage(rhs, outputs, index_of)?;
    if a_byte_offset != 0 || b_byte_offset != 0 {
        // MLX's g_nd1/2/3 take no byte offset — they assume each input
        // buffer is bound at the broadcast view's element [0,..]. We'd
        // need to bind with `setBuffer:offset:` to support this, but
        // none of the current consumers (residual + bias adds) produce
        // non-zero offsets, so surface the gap explicitly.
        return Err(Error::InvalidArgument(format!(
            "binary: non-zero view byte offset not yet supported in strided dispatch \
             (lhs={a_byte_offset}, rhs={b_byte_offset}); add a setBuffer:offset bind when \
             the first slice-broadcast case appears",
        )));
    }

    let a_strides = strides_to_i64(layout_strides(lhs)?)?;
    let b_strides = strides_to_i64(layout_strides(rhs)?)?;
    let shape_i32 = shape_to_i32(inner.shape.dims())?;

    if a_strides.len() != rank || b_strides.len() != rank {
        return Err(Error::InvalidArgument(format!(
            "binary: input strides rank mismatch (out rank {rank}, lhs strides {}, rhs strides {})",
            a_strides.len(),
            b_strides.len(),
        )));
    }

    macro_rules! dispatch_typed {
        ($ty:ty, $add1:path, $mul1:path, $add2:path, $mul2:path, $add3:path, $mul3:path) => {{
            let a_t = unsafe { a_bytes.reinterpret_as::<$ty>() };
            let b_t = unsafe { b_bytes.reinterpret_as::<$ty>() };
            let mut c_t = unsafe { dst.reinterpret_as::<$ty>() };
            match (rank, binary_op) {
                (1, BinaryOp::Add) => $add1(
                    commands,
                    library,
                    &a_t,
                    &b_t,
                    &mut c_t,
                    a_strides[0],
                    b_strides[0],
                    shape_i32[0],
                )?,
                (1, BinaryOp::Mul) => $mul1(
                    commands,
                    library,
                    &a_t,
                    &b_t,
                    &mut c_t,
                    a_strides[0],
                    b_strides[0],
                    shape_i32[0],
                )?,
                (2, BinaryOp::Add) => $add2(
                    commands,
                    library,
                    &a_t,
                    &b_t,
                    &mut c_t,
                    [a_strides[0], a_strides[1]],
                    [b_strides[0], b_strides[1]],
                    [shape_i32[0], shape_i32[1]],
                )?,
                (2, BinaryOp::Mul) => $mul2(
                    commands,
                    library,
                    &a_t,
                    &b_t,
                    &mut c_t,
                    [a_strides[0], a_strides[1]],
                    [b_strides[0], b_strides[1]],
                    [shape_i32[0], shape_i32[1]],
                )?,
                (3, BinaryOp::Add) => $add3(
                    commands,
                    library,
                    &a_t,
                    &b_t,
                    &mut c_t,
                    [a_strides[0], a_strides[1], a_strides[2]],
                    [b_strides[0], b_strides[1], b_strides[2]],
                    [shape_i32[0], shape_i32[1], shape_i32[2]],
                )?,
                (3, BinaryOp::Mul) => $mul3(
                    commands,
                    library,
                    &a_t,
                    &b_t,
                    &mut c_t,
                    [a_strides[0], a_strides[1], a_strides[2]],
                    [b_strides[0], b_strides[1], b_strides[2]],
                    [shape_i32[0], shape_i32[1], shape_i32[2]],
                )?,
                _ => unreachable!("rank bounded above"),
            }
        }};
    }

    match inner.dtype {
        DType::F32 => dispatch_typed!(
            f32,
            binary::g1_add_f32,
            binary::g1_mul_f32,
            binary::g2_add_f32,
            binary::g2_mul_f32,
            binary::g3_add_f32,
            binary::g3_mul_f32
        ),
        DType::F16 => dispatch_typed!(
            half::f16,
            binary::g1_add_f16,
            binary::g1_mul_f16,
            binary::g2_add_f16,
            binary::g2_mul_f16,
            binary::g3_add_f16,
            binary::g3_mul_f16
        ),
        DType::BF16 => dispatch_typed!(
            half::bf16,
            binary::g1_add_bf16,
            binary::g1_mul_bf16,
            binary::g2_add_bf16,
            binary::g2_mul_bf16,
            binary::g3_add_bf16,
            binary::g3_mul_bf16
        ),
        dtype @ (DType::I32 | DType::U32) => {
            return Err(Error::InvalidArgument(format!(
                "binary: dtype {dtype:?} not wired for Add/Mul yet",
            )));
        }
    }
    Ok(())
}

fn layout_strides(t: &Tensor) -> Result<&[isize]> {
    match t.layout() {
        Layout::Dense { strides } => Ok(strides.as_slice()),
        Layout::Quantized(_) => Err(Error::InvalidArgument(
            "binary: quantized input is not supported".into(),
        )),
    }
}

/// Dispatch an element-wise unary op (Square / Rsqrt / Sigmoid /
/// Erf). Inputs must be dense and contiguous (the runtime's
/// `Tensor::unary` enforces this at construction time); the
/// kernels-crate `v_*` family is the only variant currently wired.
fn dispatch_unary(
    inner: &Arc<TensorInner>,
    op: &OpNode,
    commands: &mut Commands<'_>,
    outputs: &[Buffer<u8>],
    dst: &mut Buffer<u8>,
    index_of: &HashMap<*const TensorInner, usize>,
    unary_op: UnaryOp,
) -> Result<()> {
    let device = inner
        .device
        .as_ref()
        .expect("op nodes are always constructed with a device");
    let inputs = op.lock_inputs();
    let input = inputs
        .first()
        .ok_or_else(|| Error::InvalidArgument("Unary: missing input (inputs[0])".into()))?;
    let library = device.unary_library()?;

    let src = dense_input_buffer(input, outputs, index_of)?;

    // SAFETY: dtype tag drives the reinterpret. Allocation sized the
    // buffers as `elem_count * size_bytes`, so byte lengths divide
    // evenly into the typed element count.
    match inner.dtype {
        DType::F32 => {
            let s = unsafe { src.reinterpret_as::<f32>() };
            let mut d = unsafe { dst.reinterpret_as::<f32>() };
            match unary_op {
                UnaryOp::Square => unary::v_square_f32(commands, library, &s, &mut d)?,
                UnaryOp::Rsqrt => unary::v_rsqrt_f32(commands, library, &s, &mut d)?,
                UnaryOp::Sigmoid => unary::v_sigmoid_f32(commands, library, &s, &mut d)?,
                UnaryOp::Erf => unary::v_erf_f32(commands, library, &s, &mut d)?,
            }
        }
        DType::F16 => {
            let s = unsafe { src.reinterpret_as::<half::f16>() };
            let mut d = unsafe { dst.reinterpret_as::<half::f16>() };
            match unary_op {
                UnaryOp::Square => unary::v_square_f16(commands, library, &s, &mut d)?,
                UnaryOp::Rsqrt => unary::v_rsqrt_f16(commands, library, &s, &mut d)?,
                UnaryOp::Sigmoid => unary::v_sigmoid_f16(commands, library, &s, &mut d)?,
                UnaryOp::Erf => unary::v_erf_f16(commands, library, &s, &mut d)?,
            }
        }
        DType::BF16 => {
            let s = unsafe { src.reinterpret_as::<half::bf16>() };
            let mut d = unsafe { dst.reinterpret_as::<half::bf16>() };
            match unary_op {
                UnaryOp::Square => unary::v_square_bf16(commands, library, &s, &mut d)?,
                UnaryOp::Rsqrt => unary::v_rsqrt_bf16(commands, library, &s, &mut d)?,
                UnaryOp::Sigmoid => unary::v_sigmoid_bf16(commands, library, &s, &mut d)?,
                UnaryOp::Erf => unary::v_erf_bf16(commands, library, &s, &mut d)?,
            }
        }
        dtype @ (DType::I32 | DType::U32) => {
            return Err(Error::InvalidArgument(format!(
                "unary: dtype {dtype:?} not wired for float unary ops yet \
                 (only float dtypes are instantiated at the kernels layer)",
            )));
        }
    }
    Ok(())
}

/// Dispatch a last-axis reduction (Sum / Prod / Min / Max). The
/// runtime's `Tensor::reduce` enforces last-axis-only and contiguous
/// input at construction time; here we just thread the input shape
/// (with the reduced axis re-inserted if the op tensor's shape
/// dropped it) into the kernels-crate dispatch fn.
#[allow(clippy::too_many_arguments)]
fn dispatch_reduce(
    inner: &Arc<TensorInner>,
    op: &OpNode,
    commands: &mut Commands<'_>,
    outputs: &[Buffer<u8>],
    dst: &mut Buffer<u8>,
    index_of: &HashMap<*const TensorInner, usize>,
    kind: ReduceKind,
    keep_dim: bool,
) -> Result<()> {
    let device = inner
        .device
        .as_ref()
        .expect("op nodes are always constructed with a device");
    let inputs = op.lock_inputs();
    let input = inputs
        .first()
        .ok_or_else(|| Error::InvalidArgument("Reduce: missing input (inputs[0])".into()))?;
    let library = device.reduce_library()?;

    // The kernels-crate dispatch needs the *input* shape (it reads
    // the last axis as the row size and the rest as the row count).
    // Reconstructing it: if keep_dim, the input shape is identical to
    // the output minus the last-axis size-1 (we don't have access to
    // the original row size on this side); instead read it off the
    // input tensor directly, which the OpNode already retains.
    let src_dims: Vec<usize> = input.shape().dims().to_vec();
    let _ = keep_dim; // shape inference uses the input's dims, not output's.

    let src = dense_input_buffer(input, outputs, index_of)?;

    // SAFETY: dtype tag drives the reinterpret; allocation byte
    // lengths divide evenly into the typed element count.
    match inner.dtype {
        DType::F32 => {
            let s = unsafe { src.reinterpret_as::<f32>() };
            let mut d = unsafe { dst.reinterpret_as::<f32>() };
            match kind {
                ReduceKind::Sum => {
                    reduce::row_reduce_sum_f32(commands, library, &s, &src_dims, &mut d)?;
                }
                ReduceKind::Prod => {
                    reduce::row_reduce_prod_f32(commands, library, &s, &src_dims, &mut d)?;
                }
                ReduceKind::Min => {
                    reduce::row_reduce_min_f32(commands, library, &s, &src_dims, &mut d)?;
                }
                ReduceKind::Max => {
                    reduce::row_reduce_max_f32(commands, library, &s, &src_dims, &mut d)?;
                }
            }
        }
        DType::F16 => {
            let s = unsafe { src.reinterpret_as::<half::f16>() };
            let mut d = unsafe { dst.reinterpret_as::<half::f16>() };
            match kind {
                ReduceKind::Sum => {
                    reduce::row_reduce_sum_f16(commands, library, &s, &src_dims, &mut d)?;
                }
                ReduceKind::Prod => {
                    reduce::row_reduce_prod_f16(commands, library, &s, &src_dims, &mut d)?;
                }
                ReduceKind::Min => {
                    reduce::row_reduce_min_f16(commands, library, &s, &src_dims, &mut d)?;
                }
                ReduceKind::Max => {
                    reduce::row_reduce_max_f16(commands, library, &s, &src_dims, &mut d)?;
                }
            }
        }
        DType::BF16 => {
            let s = unsafe { src.reinterpret_as::<half::bf16>() };
            let mut d = unsafe { dst.reinterpret_as::<half::bf16>() };
            match kind {
                ReduceKind::Sum => {
                    reduce::row_reduce_sum_bf16(commands, library, &s, &src_dims, &mut d)?;
                }
                ReduceKind::Prod => {
                    reduce::row_reduce_prod_bf16(commands, library, &s, &src_dims, &mut d)?;
                }
                ReduceKind::Min => {
                    reduce::row_reduce_min_bf16(commands, library, &s, &src_dims, &mut d)?;
                }
                ReduceKind::Max => {
                    reduce::row_reduce_max_bf16(commands, library, &s, &src_dims, &mut d)?;
                }
            }
        }
        dtype @ (DType::I32 | DType::U32) => {
            return Err(Error::InvalidArgument(format!(
                "reduce: dtype {dtype:?} not wired for Sum/Prod/Min/Max yet \
                 (only float dtypes are wired at the kernels layer)",
            )));
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn dispatch_arg_reduce(
    inner: &Arc<TensorInner>,
    op: &OpNode,
    commands: &mut Commands<'_>,
    outputs: &[Buffer<u8>],
    dst: &mut Buffer<u8>,
    index_of: &HashMap<*const TensorInner, usize>,
    kind: ArgReduceKind,
    axis: usize,
) -> Result<()> {
    let device = inner.device.as_ref().expect("op nodes carry a device");
    let inputs = op.lock_inputs();
    let input = inputs
        .first()
        .ok_or_else(|| Error::InvalidArgument("ArgReduce: missing input (inputs[0])".into()))?;
    let axis_size = input.shape().dims()[axis];
    let library = device.arg_reduce_library()?;
    let src = dense_input_buffer(input, outputs, index_of)?;
    // SAFETY: src is bf16 (validated at Tensor::argmax); dst is the
    // op's U32 output buffer (sized by checked_byte_count from the op
    // dtype). u8 reinterprets preserve byte length.
    let s = unsafe { src.reinterpret_as::<half::bf16>() };
    let mut d = unsafe { dst.reinterpret_as::<u32>() };
    match kind {
        ArgReduceKind::ArgMax => {
            arg_reduce::argmax_bf16(commands, library, &s, &mut d, axis_size)?;
        }
    }
    Ok(())
}

/// Dispatch an axis-0 gather. Inputs (per `Tensor::gather`): a dense
/// contiguous source tensor (currently `BF16` for the dense-embedding
/// case or `U32` for the packed-quantized-weight case) of rank ≥ 1,
/// plus a dense contiguous rank-1 `U32` indices tensor. The runtime
/// layer enforces these preconditions at construction time; here we
/// just transcribe MLX's `Gather::eval_gpu` binding.
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
fn dispatch_gather(
    inner: &Arc<TensorInner>,
    op: &OpNode,
    commands: &mut Commands<'_>,
    outputs: &[Buffer<u8>],
    dst: &mut Buffer<u8>,
    index_of: &HashMap<*const TensorInner, usize>,
) -> Result<()> {
    let device = inner
        .device
        .as_ref()
        .expect("op nodes are always constructed with a device");
    let inputs = op.lock_inputs();
    let src_tensor = inputs
        .first()
        .ok_or_else(|| Error::InvalidArgument("Gather: missing src (inputs[0])".into()))?;
    let idx_tensor = inputs
        .get(1)
        .ok_or_else(|| Error::InvalidArgument("Gather: missing indices (inputs[1])".into()))?;

    let inst = match inner.dtype {
        DType::BF16 => cider_press_kernels::kernels::gather::Instantiation::bf16_u32_rank1(),
        DType::U32 => cider_press_kernels::kernels::gather::Instantiation::u32_u32_rank1(),
        dtype => {
            return Err(Error::InvalidArgument(format!(
                "gather: dispatch not wired for src dtype {dtype:?}"
            )));
        }
    };
    let library = device.gather_library(&inst)?;

    let src_bytes = dense_input_buffer(src_tensor, outputs, index_of)?;
    let idx_bytes = dense_input_buffer(idx_tensor, outputs, index_of)?;
    let idx_typed = unsafe { idx_bytes.reinterpret_as::<u32>() };

    let src_shape_dims = src_tensor.shape().dims();
    let n_indices = idx_tensor.shape().dims()[0];
    // src is rank-1+ — slice axes are everything past axis 0.
    let slice_size: usize = src_shape_dims.iter().skip(1).product();

    // Element strides (MLX convention): row-major across src.
    let src_strides_elem: Vec<i64> = element_strides(src_shape_dims);

    let kernels_dev = device.kernels();
    let src_shape_i32: Vec<i32> = src_shape_dims.iter().map(|&d| d as i32).collect();
    let src_shape_buf = kernels_dev.upload(&src_shape_i32)?;
    let src_strides_buf = kernels_dev.upload(&src_strides_elem)?;
    // axes[0] = 0 (gathered axis), slice_sizes mirrors src.shape() with
    // gathered axes set to 1.
    let mut slice_sizes_i32: Vec<i32> = src_shape_i32.clone();
    slice_sizes_i32[0] = 1;
    let slice_sizes_buf = kernels_dev.upload(&slice_sizes_i32)?;
    let axes_buf = kernels_dev.upload(&[0i32])?;
    let idx_shapes_buf = kernels_dev.upload(&[n_indices as i32])?;
    let idx_strides_buf = kernels_dev.upload(&[1i64])?;
    let idx_contigs_buf = kernels_dev.upload(&[1u8])?;

    // Build a small closure for the metadata bundle so we don't repeat
    // it per dtype branch.
    macro_rules! call {
        ($T:ty) => {{
            let src_typed = unsafe { src_bytes.reinterpret_as::<$T>() };
            let mut dst_typed = unsafe { dst.reinterpret_as::<$T>() };
            let indices_refs = [&idx_typed];
            cider_press_kernels::kernels::gather::dispatch(
                commands,
                &library,
                &inst,
                cider_press_kernels::kernels::gather::GatherDispatch {
                    src: &src_typed,
                    out: &mut dst_typed,
                    indices: &indices_refs,
                    src_shape: &src_shape_buf,
                    src_strides: &src_strides_buf,
                    src_ndim: src_shape_dims.len() as u64,
                    slice_sizes: &slice_sizes_buf,
                    axes: &axes_buf,
                    idx_shapes: &idx_shapes_buf,
                    idx_strides: &idx_strides_buf,
                    idx_contigs: &idx_contigs_buf,
                    idx_ndim: 1,
                    grid: (n_indices, 1, slice_size),
                },
            )?;
        }};
    }
    match inner.dtype {
        DType::BF16 => call!(half::bf16),
        DType::U32 => call!(u32),
        // Already filtered above; reaching here would be a bug.
        dtype => unreachable!("gather: dtype guard let {dtype:?} through"),
    }
    Ok(())
}

/// Dispatch an affine dequantize on three dense input tensors
/// `(w_q, scales, biases)`. The runtime's `Tensor::dequantize_affine`
/// enforces preconditions (dtypes, contiguity, `group_size`/`bits`
/// matching the wired kernel) at construction; here we just route
/// the three buffers through the single wired instantiation in the
/// kernels crate.
#[allow(clippy::too_many_arguments)]
fn dispatch_dequantize(
    inner: &Arc<TensorInner>,
    op: &OpNode,
    commands: &mut Commands<'_>,
    outputs: &[Buffer<u8>],
    dst: &mut Buffer<u8>,
    index_of: &HashMap<*const TensorInner, usize>,
    group_size: u32,
    bits: u8,
) -> Result<()> {
    let device = inner
        .device
        .as_ref()
        .expect("op nodes are always constructed with a device");
    if !(group_size == 64 && bits == 4) {
        return Err(Error::InvalidArgument(format!(
            "dequantize: only group_size=64 bits=4 is wired \
             (got group_size={group_size}, bits={bits})",
        )));
    }
    let inputs = op.lock_inputs();
    let w_q = inputs
        .first()
        .ok_or_else(|| Error::InvalidArgument("Dequantize: missing w_q (inputs[0])".into()))?;
    let scales = inputs
        .get(1)
        .ok_or_else(|| Error::InvalidArgument("Dequantize: missing scales (inputs[1])".into()))?;
    let biases = inputs
        .get(2)
        .ok_or_else(|| Error::InvalidArgument("Dequantize: missing biases (inputs[2])".into()))?;

    let library = device.quantized_library()?;
    let w_bytes = dense_input_buffer(w_q, outputs, index_of)?;
    let s_bytes = dense_input_buffer(scales, outputs, index_of)?;
    let b_bytes = dense_input_buffer(biases, outputs, index_of)?;
    // SAFETY: dtype guards on Tensor::dequantize_affine enforce
    // w_q=U32, scales=BF16, biases=BF16. Output is BF16.
    let w_typed = unsafe { w_bytes.reinterpret_as::<u32>() };
    let s_typed = unsafe { s_bytes.reinterpret_as::<half::bf16>() };
    let b_typed = unsafe { b_bytes.reinterpret_as::<half::bf16>() };
    let mut dst_typed = unsafe { dst.reinterpret_as::<half::bf16>() };

    cider_press_kernels::kernels::dequantize::affine_dequantize_bf16_gs64_b4(
        commands,
        library,
        &w_typed,
        &s_typed,
        &b_typed,
        &mut dst_typed,
    )?;
    Ok(())
}

/// Dispatch a rotary positional embedding. Preconditions enforced by
/// [`Tensor::rope`]: input is rank-4 dense contiguous BF16, offset is
/// length-1 contiguous I32, both on the same device. We pass the strides
/// directly from the input's row-major layout — `Tensor::rope` rejects
/// non-contiguous inputs precisely so this is sound.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
fn dispatch_rope(
    inner: &Arc<TensorInner>,
    op: &OpNode,
    commands: &mut Commands<'_>,
    outputs: &[Buffer<u8>],
    dst: &mut Buffer<u8>,
    index_of: &HashMap<*const TensorInner, usize>,
    base_log2: f32,
    scale: f32,
    rotary_dims: u32,
) -> Result<()> {
    let device = inner
        .device
        .as_ref()
        .expect("op nodes are always constructed with a device");

    let inputs = op.lock_inputs();
    let input = inputs
        .first()
        .ok_or_else(|| Error::InvalidArgument("Rope: missing input (inputs[0])".into()))?;
    let offset = inputs
        .get(1)
        .ok_or_else(|| Error::InvalidArgument("Rope: missing offset (inputs[1])".into()))?;

    // `input` may be a contiguous view (a copy()-elided degenerate permute
    // from attention); resolve through the view chain and assert offset 0,
    // exactly like the qmv/slice_update activation paths. `Tensor::rope`
    // rejects non-contiguous views at construction, so the bytes read here
    // are contiguous-from-offset-0.
    let in_bytes = matmul_input_bytes("rope", input, outputs, index_of)?;
    let off_bytes = dense_input_buffer(offset, outputs, index_of)?;

    let in_typed = unsafe { in_bytes.reinterpret_as::<half::bf16>() };
    let off_typed = unsafe { off_bytes.reinterpret_as::<i32>() };
    let mut dst_typed = unsafe { dst.reinterpret_as::<half::bf16>() };

    let dims = input.shape().dims();
    let batch = dims[0] as i32;
    let n_head = dims[1] as i32;
    let seq_len = dims[2] as i32;
    let head_dim = dims[3] as i32;
    let mat = i64::from(seq_len) * i64::from(head_dim);

    let args = cider_press_kernels::kernels::rope::RopeArgs {
        batch,
        n_head,
        seq_len,
        head_dim,
        rotary_dims: rotary_dims as i32,
        scale,
        base_log2,
        // Element strides for row-contiguous [B, H, T, D] over the
        // (head, sequence, feature) axes; offset_stride=0 because we
        // only accept a 1-element scalar offset today.
        strides: [mat, i64::from(head_dim), 1],
        out_strides: [mat, i64::from(head_dim), 1],
        offset_stride: 0,
    };

    let library = device.rope_library()?;
    cider_press_kernels::kernels::rope::dispatch_rope_bf16(
        commands,
        library,
        &in_typed,
        &mut dst_typed,
        &off_typed,
        args,
    )?;
    Ok(())
}

fn dispatch_rms_norm(
    inner: &Arc<TensorInner>,
    op: &OpNode,
    commands: &mut Commands<'_>,
    outputs: &[Buffer<u8>],
    dst: &mut Buffer<u8>,
    index_of: &HashMap<*const TensorInner, usize>,
    eps: f32,
) -> Result<()> {
    let device = inner
        .device
        .as_ref()
        .expect("op nodes are always constructed with a device");

    let inputs = op.lock_inputs();
    let x = inputs
        .first()
        .ok_or_else(|| Error::InvalidArgument("RmsNorm: missing x (inputs[0])".into()))?;
    let w = inputs
        .get(1)
        .ok_or_else(|| Error::InvalidArgument("RmsNorm: missing weight (inputs[1])".into()))?;

    let x_bytes = dense_input_buffer(x, outputs, index_of)?;
    let w_bytes = dense_input_buffer(w, outputs, index_of)?;
    let x_typed = unsafe { x_bytes.reinterpret_as::<half::bf16>() };
    let w_typed = unsafe { w_bytes.reinterpret_as::<half::bf16>() };
    let mut dst_typed = unsafe { dst.reinterpret_as::<half::bf16>() };

    // x is [..., axis_size] row-major; the kernel treats it as
    // [n_rows, axis_size] with one threadgroup per row.
    let axis_size = *x
        .shape()
        .dims()
        .last()
        .expect("rms_norm input has rank ≥ 1");
    let n_rows = x.shape().elem_count() / axis_size;

    let library = device.rms_norm_library()?;
    cider_press_kernels::kernels::rms_norm::rms_norm_bf16(
        commands,
        library,
        &x_typed,
        &w_typed,
        &mut dst_typed,
        axis_size,
        n_rows,
        eps,
    )?;
    Ok(())
}

/// Dispatch a last-axis softmax. `Tensor::softmax` already enforces
/// contiguity + BF16 + bounded axis size, so here we just compute
/// `n_rows = elem_count / axis_size` and route to the kernels-layer
/// dispatch fn (default vs precise selected by the `precise` flag).
#[allow(clippy::too_many_arguments)]
fn dispatch_softmax(
    inner: &Arc<TensorInner>,
    op: &OpNode,
    commands: &mut Commands<'_>,
    outputs: &[Buffer<u8>],
    dst: &mut Buffer<u8>,
    index_of: &HashMap<*const TensorInner, usize>,
    precise: bool,
) -> Result<()> {
    let device = inner
        .device
        .as_ref()
        .expect("op nodes are always constructed with a device");

    let inputs = op.lock_inputs();
    let input = inputs
        .first()
        .ok_or_else(|| Error::InvalidArgument("Softmax: missing input (inputs[0])".into()))?;
    let in_bytes = dense_input_buffer(input, outputs, index_of)?;
    let in_typed = unsafe { in_bytes.reinterpret_as::<half::bf16>() };
    let mut dst_typed = unsafe { dst.reinterpret_as::<half::bf16>() };

    let dims = input.shape().dims();
    let axis_size = *dims.last().expect("Tensor::softmax enforces rank ≥ 1");
    let n_rows = input.shape().elem_count() / axis_size;

    let library = device.softmax_library()?;
    if precise {
        cider_press_kernels::kernels::softmax::dispatch_block_softmax_precise_bf16(
            commands,
            library,
            &in_typed,
            &mut dst_typed,
            axis_size,
            n_rows,
        )?;
    } else {
        cider_press_kernels::kernels::softmax::dispatch_block_softmax_bf16(
            commands,
            library,
            &in_typed,
            &mut dst_typed,
            axis_size,
            n_rows,
        )?;
    }
    Ok(())
}

/// Resolve a matmul input's backing bytes. `Tensor::matmul` enforces
/// contiguous strides over the input shape; this function additionally
/// asserts the resolved view chain lands at byte offset 0 — non-zero
/// offsets would require a `setBuffer:offset:` bind that the kernel
/// doesn't currently take.
fn matmul_input_bytes(
    op: &str,
    input: &Tensor,
    outputs: &[Buffer<u8>],
    index_of: &HashMap<*const TensorInner, usize>,
) -> Result<Buffer<u8>> {
    if input.inner.view.is_some() {
        let (buf, byte_offset) = resolve_view_storage(input, outputs, index_of)?;
        if byte_offset != 0 {
            return Err(Error::InvalidArgument(format!(
                "{op}: non-zero view byte offset {byte_offset} is not supported by the \
                 contiguous matmul kernel; add a `setBuffer:offset:` bind when the first \
                 slice-into-matmul case appears",
            )));
        }
        return Ok(buf);
    }
    dense_input_buffer(input, outputs, index_of)
}

/// Dispatch a batched bf16 matmul. `Tensor::matmul` already
/// enforced contiguity + matching leading dims + aligned inner
/// dimension, so here we just extract `(batch, M, N, K)` and
/// route to the kernels-layer dispatch fn.
fn dispatch_matmul(
    inner: &Arc<TensorInner>,
    op: &OpNode,
    commands: &mut Commands<'_>,
    outputs: &[Buffer<u8>],
    dst: &mut Buffer<u8>,
    index_of: &HashMap<*const TensorInner, usize>,
) -> Result<()> {
    let device = inner
        .device
        .as_ref()
        .expect("op nodes are always constructed with a device");

    let inputs = op.lock_inputs();
    let lhs = inputs
        .first()
        .ok_or_else(|| Error::InvalidArgument("MatMul: missing LHS (inputs[0])".into()))?;
    let rhs = inputs
        .get(1)
        .ok_or_else(|| Error::InvalidArgument("MatMul: missing RHS (inputs[1])".into()))?;

    // `Tensor::matmul` enforces contiguous strides on each input, but
    // a reshape-of-op still leaves a view in the chain. Walk the chain to
    // the storage owner; the contract guarantees a byte-offset of zero and
    // contiguous bytes thereafter.
    let lhs_bytes = matmul_input_bytes("matmul", lhs, outputs, index_of)?;
    let rhs_bytes = matmul_input_bytes("matmul", rhs, outputs, index_of)?;
    let lhs_typed = unsafe { lhs_bytes.reinterpret_as::<half::bf16>() };
    let rhs_typed = unsafe { rhs_bytes.reinterpret_as::<half::bf16>() };
    let mut dst_typed = unsafe { dst.reinterpret_as::<half::bf16>() };

    let lhs_dims = lhs.shape().dims();
    let rhs_dims = rhs.shape().dims();
    let lead = lhs_dims.len() - 2;
    let m = lhs_dims[lead];
    let k = lhs_dims[lead + 1];
    let n = rhs_dims[lead + 1];
    let batch: usize = lhs_dims[..lead].iter().product::<usize>().max(1);

    let library = device.matmul_library()?;
    cider_press_kernels::kernels::matmul::dispatch_gemm_bf16(
        commands,
        library,
        &lhs_typed,
        &rhs_typed,
        &mut dst_typed,
        m,
        n,
        k,
        batch,
        m * k,
        k * n,
        m * n,
    )?;
    Ok(())
}

/// Dispatch fused SDPA, routing on query length `T_q`: `T_q==1` (decode)
/// takes the vector path here, `T_q>1` (prefill) delegates to
/// `dispatch_sdpa_full` (steel `sdpa_full`, causal via the kernel's
/// `do_causal`). Explicit masks are rejected at construction, so neither
/// path handles one. Vector path: Q is dense `[1,H_q,1,D]`; K/V are
/// `[1,H_kv,T_cache,D]`, read strided in place via `layout_strides` — no
/// GQA-broadcast/transpose copy.
#[allow(clippy::too_many_arguments)]
fn dispatch_sdpa(
    inner: &Arc<TensorInner>,
    op: &OpNode,
    commands: &mut Commands<'_>,
    outputs: &[Buffer<u8>],
    dst: &mut Buffer<u8>,
    index_of: &HashMap<*const TensorInner, usize>,
    scale: f32,
    gqa_factor: usize,
    causal: bool,
) -> Result<()> {
    // Route by query length: T_q>1 (prefill) takes the steel fused-full
    // path, T_q==1 (decode) the vector path. The lock is scoped so it is
    // released before either path re-locks the same inputs (re-locking
    // under a held guard would deadlock).
    let t_q = {
        let inputs = op.lock_inputs();
        let q = inputs
            .first()
            .ok_or_else(|| Error::InvalidArgument("Sdpa: missing q (inputs[0])".into()))?;
        q.shape().dims()[2]
    };
    if t_q > 1 {
        return dispatch_sdpa_full(inner, op, commands, outputs, dst, index_of, scale, causal);
    }

    let inputs = op.lock_inputs(); // vector (decode) path re-locks
    let device = inner.device.as_ref().expect("op nodes carry a device");

    let q = inputs
        .first()
        .ok_or_else(|| Error::InvalidArgument("Sdpa: missing q (inputs[0])".into()))?;
    let k = inputs
        .get(1)
        .ok_or_else(|| Error::InvalidArgument("Sdpa: missing k (inputs[1])".into()))?;
    let v = inputs
        .get(2)
        .ok_or_else(|| Error::InvalidArgument("Sdpa: missing v (inputs[2])".into()))?;

    let head_dim = *q.shape().dims().last().expect("sdpa q is rank-4");
    let n_keys = k.shape().dims()[2];

    let q_bytes = matmul_input_bytes("sdpa", q, outputs, index_of)?;
    let (k_bytes, k_off) = resolve_view_storage(k, outputs, index_of)?;
    let (v_bytes, v_off) = resolve_view_storage(v, outputs, index_of)?;
    if k_off != 0 || v_off != 0 {
        return Err(Error::InvalidArgument(format!(
            "sdpa: non-zero K/V view byte offset (k={k_off}, v={v_off}) not yet supported"
        )));
    }

    let k_strides = layout_strides(k)?; // &[isize], len 4: [s0, s_head, s_seq, s_d]
    let v_strides = layout_strides(v)?;
    // The kernel reads the head dimension with contiguous pointer indexing
    // (`keys[j]`/`values[j]`), so only the head/seq strides are passed; a
    // non-unit trailing (D) stride would silently attend over wrong elements.
    if k_strides[3] != 1 || v_strides[3] != 1 {
        return Err(Error::InvalidArgument(format!(
            "sdpa: K/V head dimension must be contiguous (D stride 1); got k={}, v={}",
            k_strides[3], v_strides[3]
        )));
    }

    // SAFETY: `Tensor::sdpa` pinned BF16 at construction, so every buffer
    // here is bf16; each was sized as `elem_count * size_bytes`, so the
    // typed reinterpret divides evenly and the strides describe valid
    // in-bounds reads.
    let q_typed = unsafe { q_bytes.reinterpret_as::<half::bf16>() };
    let k_typed = unsafe { k_bytes.reinterpret_as::<half::bf16>() };
    let v_typed = unsafe { v_bytes.reinterpret_as::<half::bf16>() };
    let mut dst_typed = unsafe { dst.reinterpret_as::<half::bf16>() };

    let library = device.sdpa_vector_library()?;
    cider_press_kernels::kernels::sdpa::dispatch_sdpa_vector_bf16(
        commands,
        library,
        &q_typed,
        &k_typed,
        &v_typed,
        &mut dst_typed,
        cider_press_kernels::kernels::sdpa::SdpaVectorArgs {
            gqa_factor: i32::try_from(gqa_factor)
                .map_err(|_| Error::InvalidArgument("sdpa: gqa_factor too large".into()))?,
            n_keys: i32::try_from(n_keys)
                .map_err(|_| Error::InvalidArgument("sdpa: n_keys too large".into()))?,
            k_head_stride: u64::try_from(k_strides[1]).map_err(|_| {
                Error::InvalidArgument(format!(
                    "sdpa: k_head_stride {} is negative or overflows u64",
                    k_strides[1]
                ))
            })?,
            k_seq_stride: u64::try_from(k_strides[2]).map_err(|_| {
                Error::InvalidArgument(format!(
                    "sdpa: k_seq_stride {} is negative or overflows u64",
                    k_strides[2]
                ))
            })?,
            v_head_stride: u64::try_from(v_strides[1]).map_err(|_| {
                Error::InvalidArgument(format!(
                    "sdpa: v_head_stride {} is negative or overflows u64",
                    v_strides[1]
                ))
            })?,
            v_seq_stride: u64::try_from(v_strides[2]).map_err(|_| {
                Error::InvalidArgument(format!(
                    "sdpa: v_seq_stride {} is negative or overflows u64",
                    v_strides[2]
                ))
            })?,
            scale,
            head_dim,
        },
    )?;
    Ok(())
}

/// Fused full-attention (prefill, `T_q>1`) via MLX's vendored steel
/// `sdpa_full` kernel. Mirrors the vector path's storage/stride
/// resolution but requires a contiguous head dimension and zero view
/// offset on Q/K/V. Does not guard `head_dim` locally: `head_dim != 64`
/// is rejected downstream by `dispatch_sdpa_full_bf16` (the only
/// instantiated steel shape).
#[allow(clippy::too_many_arguments)]
fn dispatch_sdpa_full(
    inner: &Arc<TensorInner>,
    op: &OpNode,
    commands: &mut Commands<'_>,
    outputs: &[Buffer<u8>],
    dst: &mut Buffer<u8>,
    index_of: &HashMap<*const TensorInner, usize>,
    scale: f32,
    causal: bool,
) -> Result<()> {
    let device = inner.device.as_ref().expect("op nodes carry a device");
    let inputs = op.lock_inputs();
    let q = inputs
        .first()
        .ok_or_else(|| Error::InvalidArgument("Sdpa: missing q (inputs[0])".into()))?;
    let k = inputs
        .get(1)
        .ok_or_else(|| Error::InvalidArgument("Sdpa: missing k (inputs[1])".into()))?;
    let v = inputs
        .get(2)
        .ok_or_else(|| Error::InvalidArgument("Sdpa: missing v (inputs[2])".into()))?;

    let qd = q.shape().dims().to_vec(); // [1, H_q, qL, D]
    let kd = k.shape().dims().to_vec(); // [1, H_kv, kL, D]
    let (h_q, q_len, head_dim) = (qd[1], qd[2], qd[3]);
    let (h_kv, k_len) = (kd[1], kd[2]);

    let (q_bytes, q_off) = resolve_view_storage(q, outputs, index_of)?;
    let (k_bytes, k_off) = resolve_view_storage(k, outputs, index_of)?;
    let (v_bytes, v_off) = resolve_view_storage(v, outputs, index_of)?;
    if q_off != 0 || k_off != 0 || v_off != 0 {
        return Err(Error::InvalidArgument(format!(
            "sdpa_full: non-zero view byte offset (q={q_off}, k={k_off}, v={v_off}) not supported"
        )));
    }

    let qs = layout_strides(q)?;
    let ks = layout_strides(k)?;
    let vs = layout_strides(v)?;
    // The steel `sdpa_full` kernel reads each head's D elements with a
    // unit-stride pointer walk, so it assumes the head dimension is
    // contiguous; a non-unit D stride would silently read the wrong
    // elements. Fail loud rather than dispatch a garbage attention.
    if qs[3] != 1 || ks[3] != 1 || vs[3] != 1 {
        return Err(Error::InvalidArgument(format!(
            "sdpa_full: Q/K/V head dim must be contiguous (D stride 1); got q={}, k={}, v={}",
            qs[3], ks[3], vs[3]
        )));
    }
    let os = element_strides(&[1, h_q, q_len, head_dim]); // dense output (B,H,L,D)

    // SAFETY: `Tensor::sdpa` pinned BF16 at construction; each buffer was
    // sized as `elem_count * size_bytes`, so the typed reinterpret divides
    // evenly and the strides describe valid in-bounds reads.
    let q_typed = unsafe { q_bytes.reinterpret_as::<half::bf16>() };
    let k_typed = unsafe { k_bytes.reinterpret_as::<half::bf16>() };
    let v_typed = unsafe { v_bytes.reinterpret_as::<half::bf16>() };
    let mut dst_typed = unsafe { dst.reinterpret_as::<half::bf16>() };

    let conv = |s: &[isize]| -> [i64; 3] { [s[0] as i64, s[1] as i64, s[2] as i64] };
    let library = device.steel_attention_library()?;
    cider_press_kernels::kernels::sdpa::dispatch_sdpa_full_bf16(
        commands,
        library,
        &q_typed,
        &k_typed,
        &v_typed,
        &mut dst_typed,
        cider_press_kernels::kernels::sdpa::SdpaFullArgs {
            batch: 1,
            h_q: i32::try_from(h_q)
                .map_err(|_| Error::InvalidArgument("sdpa_full: h_q too large".into()))?,
            h_kv: i32::try_from(h_kv)
                .map_err(|_| Error::InvalidArgument("sdpa_full: h_kv too large".into()))?,
            head_dim: i32::try_from(head_dim)
                .map_err(|_| Error::InvalidArgument("sdpa_full: head_dim too large".into()))?,
            q_len: i32::try_from(q_len)
                .map_err(|_| Error::InvalidArgument("sdpa_full: q_len too large".into()))?,
            k_len: i32::try_from(k_len)
                .map_err(|_| Error::InvalidArgument("sdpa_full: k_len too large".into()))?,
            scale,
            q_strides: conv(qs),
            k_strides: conv(ks),
            v_strides: conv(vs),
            o_strides: [os[0], os[1], os[2]],
            causal,
        },
    )?;
    Ok(())
}

/// Row-major element strides for a shape: for `[d0, d1, d2]`,
/// `[d1*d2, d2, 1]`.
#[allow(clippy::cast_possible_wrap)]
fn element_strides(shape: &[usize]) -> Vec<i64> {
    let mut strides = vec![1i64; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1] as i64;
    }
    strides
}

#[cfg(test)]
mod pool_tests {
    use crate::{Device, Tensor};

    /// A second eval of the same-shaped graph on a fresh device reuses
    /// the first eval's freed op-output buffer: the pool registers a hit.
    #[test]
    fn second_eval_reuses_freed_output_buffer() {
        let device = Device::system_default().expect("device");

        {
            let a = Tensor::from_slice(&device, &[1.0f32, 2.0, 3.0, 4.0], [4]).unwrap();
            let b = Tensor::from_slice(&device, &[10.0f32, 20.0, 30.0, 40.0], [4]).unwrap();
            let r = a.add(&b).unwrap();
            r.eval().unwrap();
            assert_eq!(r.cpu_slice::<f32>().unwrap(), &[11.0, 22.0, 33.0, 44.0]);
        } // r drops here → its output buffer returns to the pool.

        let before = device.pool_stats().hits;

        {
            let a = Tensor::from_slice(&device, &[5.0f32, 6.0, 7.0, 8.0], [4]).unwrap();
            let b = Tensor::from_slice(&device, &[1.0f32, 1.0, 1.0, 1.0], [4]).unwrap();
            let r = a.add(&b).unwrap();
            r.eval().unwrap();
            assert_eq!(r.cpu_slice::<f32>().unwrap(), &[6.0, 7.0, 8.0, 9.0]);
        }

        assert!(
            device.pool_stats().hits > before,
            "second eval should reuse the freed output buffer"
        );
    }
}

#[cfg(test)]
mod label_tests {
    use super::op_kind_label;
    use crate::tensor::{BinaryOp, OpKind};

    #[test]
    fn labels_cover_each_kind() {
        assert_eq!(op_kind_label(&OpKind::Copy), "copy");
        assert_eq!(op_kind_label(&OpKind::MatMul), "matmul");
        assert_eq!(
            op_kind_label(&OpKind::Binary { op: BinaryOp::Add }),
            "binary"
        );
        assert_eq!(
            op_kind_label(&OpKind::QuantizedMatMul {
                group_size: 64,
                bits: 4,
                transpose: true
            }),
            "quantized_matmul"
        );
        assert_eq!(
            op_kind_label(&OpKind::SliceUpdate { offset_rows: 0 }),
            "slice_update"
        );
    }
}

#[cfg(test)]
mod async_tests {
    use half::bf16;

    use crate::{Device, Tensor};

    fn bf16_vec(xs: &[f32]) -> Vec<bf16> {
        xs.iter().map(|&x| bf16::from_f32(x)).collect()
    }

    /// A committed-but-unwaited graph yields correct bytes after `wait()`.
    #[test]
    fn eval_async_then_wait_materializes() {
        let device = Device::system_default().expect("device");
        let a = Tensor::from_slice(&device, &bf16_vec(&[1.0, 2.0, 3.0, 4.0]), [4]).expect("a");
        let b = a.add(&a).expect("b = a + a");
        let pending = b.eval_async().expect("eval_async");
        pending.wait().expect("wait");
        assert_eq!(
            b.cpu_to_vec::<bf16>().expect("b bytes"),
            bf16_vec(&[2.0, 4.0, 6.0, 8.0])
        );
    }

    /// Host reads are gated while a committed batch is in flight: between
    /// `eval_async` and `wait`, the tensor has storage (dependent graphs can
    /// chain off it) but must not look host-readable.
    #[test]
    fn eval_async_gates_host_reads_until_wait() {
        let device = Device::system_default().expect("device");
        let a = Tensor::from_slice(&device, &bf16_vec(&[1.0, 2.0]), [2]).expect("a");
        let b = a.add(&a).expect("b = a + a");
        let pending = b.eval_async().expect("eval_async");
        assert!(
            !b.is_materialized(),
            "in-flight tensor must not report host-readable"
        );
        assert!(b.cpu_bytes().is_none());
        assert!(b.cpu_to_vec::<bf16>().is_none());
        pending.wait().expect("wait");
        assert!(b.is_materialized());
        assert_eq!(b.cpu_to_vec::<bf16>().expect("b"), bf16_vec(&[2.0, 4.0]));
    }

    /// Two chained async evals — the second consumes the first's output —
    /// equal two synchronous evals. Proves cross-command-buffer ordering.
    #[test]
    fn chained_async_evals_order_correctly() {
        let device = Device::system_default().expect("device");
        let a = Tensor::from_slice(&device, &bf16_vec(&[1.0, 2.0, 3.0, 4.0]), [4]).expect("a");
        let b = a.add(&a).expect("b");
        let p1 = b.eval_async().expect("p1"); // b committed, not waited
        let c = b.add(&b).expect("c = b + b"); // reads b's in-flight buffer
        let p2 = c.eval_async().expect("p2");
        p1.wait().expect("wait p1");
        p2.wait().expect("wait p2");
        assert_eq!(
            c.cpu_to_vec::<bf16>().expect("c bytes"),
            bf16_vec(&[4.0, 8.0, 12.0, 16.0])
        );
    }

    /// Dropping a `PendingEval` without calling `wait()` must still block on GPU
    /// completion (the in-flight handle's Drop), leaving valid bytes. Guards
    /// the `in_flight`-before-`order` field drop ordering.
    #[test]
    fn dropping_pending_eval_blocks_until_complete() {
        let device = Device::system_default().expect("device");
        let a = Tensor::from_slice(&device, &bf16_vec(&[3.0, 4.0]), [2]).expect("a");
        let b = a.add(&a).expect("b");
        drop(b.eval_async().expect("eval_async")); // dropped, not waited
        assert_eq!(b.cpu_to_vec::<bf16>().expect("b"), bf16_vec(&[6.0, 8.0]));
    }

    /// Holding many `PendingEval`s (never waiting until the end) keeps every
    /// in-flight buffer pinned, so a long dependent chain stays correct.
    ///
    /// Pool-competition note: ideally we would assert that a same-sized
    /// competing alloc (done while chain handles are held) recycles a freed
    /// buffer and still produces the wrong result if pinning is broken. In
    /// practice the chain holds *all* same-sized (2-byte bf16) free-list
    /// slots while its handles are unwaited, so a competing alloc always
    /// misses the pool — deterministic freelist-slot contention is not
    /// achievable here without bypassing the pool's encapsulation. Instead,
    /// we document the pool's miss count (which must equal 1 for the
    /// competing eval, proving that all slots were in fact held), and assert
    /// the chain result is still correct after waiting — the correctness
    /// assertion is the primary guard; the pool-stats assertion documents
    /// exactly what "pinning was exercised" means here.
    #[test]
    fn held_pending_evals_pin_buffers_through_a_chain() {
        let device = Device::system_default().expect("device");
        let mut t = Tensor::from_slice(&device, &bf16_vec(&[1.0]), [1]).expect("t0");
        let mut handles = Vec::new();
        for _ in 0..16 {
            t = t.add(&t).expect("double");
            handles.push(t.eval_async().expect("eval_async"));
        }

        // While all 16 chain PendingEvals are unwaited, run a competing
        // same-sized (1-element bf16) synchronous eval. Because all
        // same-sized free-list slots are pinned by the chain, the
        // competitor must allocate fresh (miss). The miss count increasing
        // documents that the freelist was fully occupied — i.e., buffers
        // are not leaking back to the pool mid-flight.
        let misses_before = device.pool_stats().misses;
        let competitor = Tensor::from_slice(&device, &bf16_vec(&[1.0]), [1]).expect("comp");
        let comp_r = competitor.add(&competitor).expect("comp_r");
        comp_r.eval().expect("comp eval");
        let misses_after = device.pool_stats().misses;
        assert!(
            misses_after > misses_before,
            "competitor eval unexpectedly hit the pool while chain buffers should be pinned \
             (misses before={misses_before}, after={misses_after}); this is surprising — \
             check whether the chain is actually holding its handles"
        );
        assert_eq!(comp_r.cpu_to_vec::<bf16>().expect("comp"), bf16_vec(&[2.0]));

        for h in handles {
            h.wait().expect("wait");
        }
        // 1 doubled 16 times = 2^16 = 65536.
        assert_eq!(t.cpu_to_vec::<bf16>().expect("final"), bf16_vec(&[65536.0]));
    }

    /// `eval_async` on an already-materialized tensor returns an empty
    /// handle that waits trivially.
    #[test]
    fn eval_async_on_materialized_is_empty() {
        let device = Device::system_default().expect("device");
        let a = Tensor::from_slice(&device, &bf16_vec(&[5.0]), [1]).expect("a");
        a.eval().expect("eval"); // materialize
        let pending = a.eval_async().expect("eval_async on materialized");
        pending.wait().expect("wait");
        assert_eq!(a.cpu_to_vec::<bf16>().expect("a"), bf16_vec(&[5.0]));
    }
}

#[cfg(test)]
mod detach_tests {
    use half::bf16;

    use crate::{Device, Tensor};

    fn bf16_vec(xs: &[f32]) -> Vec<bf16> {
        xs.iter().map(|&x| bf16::from_f32(x)).collect()
    }

    /// After eval, every evaluated node has dropped its inputs (detached),
    /// while its value is still correct and materialized.
    #[test]
    fn eval_detaches_inputs() {
        let device = Device::system_default().expect("device");
        let a = Tensor::from_slice(&device, &bf16_vec(&[1.0, 2.0, 3.0, 4.0]), [4]).expect("a");
        let b = a.add(&a).expect("b = a + a"); // intermediate, held
        let c = b.add(&b).expect("c = b + b"); // root, held
        c.eval().expect("eval");

        assert!(
            c.op_inputs().expect("c is an op").is_empty(),
            "root must be detached after eval"
        );
        assert!(
            b.op_inputs().expect("b is an op").is_empty(),
            "intermediate must be detached after eval"
        );
        assert!(c.is_materialized() && b.is_materialized());
        assert_eq!(
            c.cpu_to_vec::<bf16>().expect("c"),
            bf16_vec(&[4.0, 8.0, 12.0, 16.0])
        );
    }

    /// `eval_async` detaches at commit time (before wait), while the value is
    /// still produced correctly once waited.
    #[test]
    fn eval_async_detaches_at_commit() {
        let device = Device::system_default().expect("device");
        let a = Tensor::from_slice(&device, &bf16_vec(&[1.0, 2.0]), [2]).expect("a");
        let b = a.add(&a).expect("b");
        let pending = b.eval_async().expect("eval_async");
        assert!(
            b.op_inputs().expect("b is an op").is_empty(),
            "detach happens at commit, before wait"
        );
        pending.wait().expect("wait");
        assert_eq!(b.cpu_to_vec::<bf16>().expect("b"), bf16_vec(&[2.0, 4.0]));
    }

    /// Detach is idempotent: a second eval (short-circuited by the cache)
    /// does not panic and leaves inputs empty.
    #[test]
    fn detach_is_idempotent() {
        let device = Device::system_default().expect("device");
        let a = Tensor::from_slice(&device, &bf16_vec(&[5.0]), [1]).expect("a");
        let b = a.add(&a).expect("b");
        b.eval().expect("eval 1");
        b.eval().expect("eval 2 (cached, no-op)");
        assert!(b.op_inputs().expect("b is an op").is_empty());
        assert_eq!(b.cpu_to_vec::<bf16>().expect("b"), bf16_vec(&[10.0]));
    }

    /// A view onto a detached node still resolves through its cache.
    ///
    /// `b` is already cached before `v` is created, so `v.eval()` short-circuits
    /// immediately (the topo walk skips cached nodes). This test confirms that
    /// detaching `b` does not break later view *resolution* through `b`'s cache;
    /// it does not itself exercise the detach pass (that is covered by
    /// `eval_detaches_inputs` and `eval_async_detaches_at_commit`).
    #[test]
    fn view_of_detached_node_still_resolves() {
        let device = Device::system_default().expect("device");
        let a = Tensor::from_slice(&device, &bf16_vec(&[1.0, 2.0, 3.0, 4.0]), [4]).expect("a");
        let b = a.add(&a).expect("b"); // [2,4,6,8]
        b.eval().expect("eval");
        let v = b.reshape([2usize, 2]).expect("reshape view");
        v.eval().expect("eval view");
        assert_eq!(
            v.cpu_to_vec::<bf16>().expect("v"),
            bf16_vec(&[2.0, 4.0, 6.0, 8.0])
        );
    }

    /// The regression guard: chaining an evaluated op-tensor across
    /// iterations must let the buffer pool recycle scratch. Without detach
    /// the chain retains every iteration's graph and pool hits stay flat.
    #[test]
    fn chaining_evaluated_tensor_recycles_pool() {
        let device = Device::system_default().expect("device");
        let mut t = Tensor::from_slice(&device, &bf16_vec(&[1.0, 1.0, 1.0, 1.0]), [4]).expect("t0");
        let before = device.pool_stats().hits;
        for _ in 0..8 {
            let next = t.add(&t).expect("double");
            next.eval().expect("eval");
            t = next; // hold the evaluated op-tensor as the next input
        }
        assert!(
            device.pool_stats().hits > before,
            "detach must free per-iteration scratch for pool reuse (hits {} did not grow from {})",
            device.pool_stats().hits,
            before
        );
    }
}

#[cfg(test)]
mod order_tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use super::build_order;
    use crate::tensor::TensorInner;
    use crate::{Device, Tensor};

    /// `build_order` emits every producer before its consumer (the defining
    /// topo property). Tested on a diamond graph:
    ///
    /// ```text
    ///   a (leaf)
    ///   ├─ b = a + a
    ///   └─ c = a + a
    ///       d = b + c
    /// ```
    ///
    /// Both `b` and `c` are producers of `d`, and `a` is a producer of both.
    /// The Kahn schedule must place `b` and `c` before `d`.
    #[test]
    fn build_order_producers_before_consumers() {
        let device = Device::system_default().expect("device");
        let a = Tensor::from_slice(&device, &[1.0f32, 2.0], [2]).expect("a");
        let b = a.add(&a).expect("b = a + a");
        let c = a.add(&a).expect("c = a + a");
        let d = b.add(&c).expect("d = b + c");

        let order = build_order(&d);

        // Build a position map: Arc ptr → index in order.
        let pos: HashMap<*const TensorInner, usize> = order
            .iter()
            .enumerate()
            .map(|(i, inner)| (Arc::as_ptr(inner), i))
            .collect();

        // For every node in the order, every op-input that is itself scheduled
        // must appear at an earlier position.
        for (consumer_pos, inner) in order.iter().enumerate() {
            let op = inner.op.as_ref().expect("order contains only op nodes");
            for input in op.inputs() {
                if let Some(&producer_pos) = pos.get(&Arc::as_ptr(&input.inner)) {
                    assert!(
                        producer_pos < consumer_pos,
                        "producer at position {producer_pos} must precede consumer at \
                         position {consumer_pos} in build_order output"
                    );
                }
            }
        }

        // Sanity: diamond has exactly 3 op nodes (b, c, d); `a` is a leaf.
        assert_eq!(order.len(), 3, "diamond graph should produce 3 op nodes");
    }

    /// A producer consumed only through a view: `d = v + v` where
    /// `v = reshape(b)` and `b` is an unevaluated op. Views own no
    /// storage, so `resolve_to_op` must chase the view chain to its
    /// source op — `b` must be scheduled, and before `d`.
    #[test]
    fn build_order_chases_view_inputs_to_source_op() {
        let device = Device::system_default().expect("device");
        let a = Tensor::from_slice(&device, &[1.0f32, 2.0], [2]).expect("a");
        let b = a.add(&a).expect("b = a + a");
        let v = b.reshape([1, 2]).expect("v = reshape(b)");
        let d = v.add(&v).expect("d = v + v");

        let order = build_order(&d);

        assert_eq!(
            order.len(),
            2,
            "b and d are the only op nodes (v is a view, a is a leaf)"
        );
        let position = |t: &Tensor| {
            order
                .iter()
                .position(|inner| Arc::as_ptr(inner) == Arc::as_ptr(&t.inner))
        };
        let pos_b = position(&b).expect("view source op `b` must be scheduled");
        let pos_d = position(&d).expect("root op `d` must be scheduled");
        assert!(
            pos_b < pos_d,
            "view source op `b` (pos {pos_b}) must precede its consumer `d` (pos {pos_d})"
        );
    }
}
