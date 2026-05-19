//! Eval: walk the lazy graph, encode one command-buffer batch, commit,
//! populate result caches.
//!
//! Public entry point is [`Tensor::eval`](crate::Tensor::eval); this
//! module owns the topological walk, the per-op dispatcher, and the
//! cache-population pass. It is intentionally a *single* `Commands`
//! per `eval()` call with a synchronous `commit_and_wait` — the
//! design doc's "first cut" eval. Internal pipelining (multiple
//! command buffers in flight, completion handlers) is a future
//! addition that does not change the public surface.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use cider_press_kernels::{
    Buffer, Commands,
    kernels::{copy, qmv},
};

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::tensor::{LeafStorage, OpKind, OpNode, Tensor, TensorInner};

pub(crate) fn eval(root: &Tensor) -> Result<()> {
    // Step 1: topological order of unevaluated op nodes reachable from
    // `root`. Reverse-postorder DFS, deduped by `Arc` pointer identity.
    // Skips placeholders, already-materialized leaves, and op nodes
    // whose cache is already populated.
    let mut visited: HashSet<*const TensorInner> = HashSet::new();
    let mut order: Vec<Arc<TensorInner>> = Vec::new();
    visit(&root.inner, &mut visited, &mut order);

    if order.is_empty() {
        return Ok(()); // every reachable node is already terminal.
    }

    // Step 2: open one `Commands` for the whole batch.
    let device = root.inner.device.as_ref().ok_or_else(|| {
        Error::InvalidArgument("eval: placeholder has no device to dispatch on".into())
    })?;
    let mut commands = device.kernels().commands()?;

    // Step 3: index every node we're about to dispatch, so a later
    // node's dispatcher can look up an earlier node's just-allocated
    // output without the cache being populated yet (cache writes are
    // deferred until step 5 — bytes aren't valid until commit + wait).
    let mut index_of: HashMap<*const TensorInner, usize> = HashMap::new();
    for (i, inner) in order.iter().enumerate() {
        index_of.insert(Arc::as_ptr(inner), i);
    }

    // Step 4: for each op in topo order, allocate output and encode.
    // `outputs` grows as we go; `&outputs` is a valid shared-borrow
    // for input lookup on subsequent iterations while we hold an
    // exclusive `&mut dst` to the fresh local.
    let mut outputs: Vec<Buffer<u8>> = Vec::with_capacity(order.len());
    for inner in &order {
        let byte_count = inner.shape.elem_count() * inner.dtype.size_bytes();
        let mut dst = device.kernels().alloc_buffer::<u8>(byte_count)?;
        dispatch(inner, &mut commands, &outputs, &mut dst, &index_of)?;
        outputs.push(dst);
    }

    // Step 5: synchronous commit + wait. After this returns, every
    // encoded dispatch has completed and the output bytes are valid.
    commands.commit_and_wait()?;

    // Step 6: populate caches. Race-safe: if another thread also
    // evaluated this node (no concurrent-eval story yet, but the
    // structure already supports it), one set wins and the other's
    // buffer is dropped. Op outputs are always dense — no op
    // currently produces a quantized tensor (no `mx.quantize`
    // analogue yet).
    for (inner, buffer) in order.into_iter().zip(outputs) {
        let _ = inner.cache.set(LeafStorage::Dense(buffer));
    }

    Ok(())
}

fn visit(
    inner: &Arc<TensorInner>,
    visited: &mut HashSet<*const TensorInner>,
    order: &mut Vec<Arc<TensorInner>>,
) {
    let ptr = Arc::as_ptr(inner);
    if !visited.insert(ptr) {
        return;
    }
    if inner.cache.get().is_some() {
        return; // already materialized; skip the whole subtree below.
    }
    if let Some(op) = inner.op.as_ref() {
        for input in &op.inputs {
            visit(&input.inner, visited, order);
        }
        order.push(inner.clone());
    }
    // Placeholders (no op, no cache) are silently skipped — eval()
    // catches the broken case at step 2 if the root needs dispatching.
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
            "eval: Qmv expected a quantized weight leaf, found dense".into(),
        )),
        None => Err(Error::InvalidArgument(
            "eval: Qmv weight input is not materialized (placeholder or unevaluated op?)".into(),
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
        OpKind::Qmv { group_size, bits } => dispatch_qmv(
            inner, op, commands, outputs, dst, index_of, group_size, bits,
        ),
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
    let input = op
        .inputs
        .first()
        .expect("Copy has exactly one input by construction");
    let src = dense_input_buffer(input, outputs, index_of)?;

    let library = device.copy_library()?;

    match inner.dtype {
        DType::F32 => {
            // SAFETY: the runtime's dtype tag is the source of truth
            // for the byte contents; reinterpreting `Buffer<u8>` as
            // `Buffer<f32>` matches what the kernel expects. Byte
            // length divides evenly because the buffers were sized
            // as `elem_count * size_bytes` at allocation.
            let src_typed = unsafe { src.reinterpret_as::<f32>() };
            let mut dst_typed = unsafe { dst.reinterpret_as::<f32>() };
            copy::copy_v_f32(commands, library, &src_typed, &mut dst_typed)?;
        }
        DType::F16 => {
            // SAFETY: same justification as the F32 arm.
            let src_typed = unsafe { src.reinterpret_as::<half::f16>() };
            let mut dst_typed = unsafe { dst.reinterpret_as::<half::f16>() };
            copy::copy_v_f16(commands, library, &src_typed, &mut dst_typed)?;
        }
        other => {
            return Err(Error::InvalidArgument(format!(
                "Copy: dtype {other} is not yet wired through the dispatcher \
                 (kernels crate currently exposes only f32 and f16)"
            )));
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn dispatch_qmv(
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
    let weight = op
        .inputs
        .first()
        .ok_or_else(|| Error::InvalidArgument("Qmv: missing weight input (inputs[0])".into()))?;
    let x_tensor = op.inputs.get(1).ok_or_else(|| {
        Error::InvalidArgument("Qmv: missing activation input (inputs[1])".into())
    })?;

    let (w_bytes, scales_bytes, biases_bytes) = quantized_input_buffers(weight)?;
    let x_bytes = dense_input_buffer(x_tensor, outputs, index_of)?;

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

    qmv::affine_qmv_bf16(
        commands,
        library,
        &w_typed,
        &scales_typed,
        &biases_typed,
        &x_typed,
        &mut y_typed,
        group_size,
        u32::from(bits),
    )?;

    Ok(())
}
