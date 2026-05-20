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
    Buffer, Commands, KernelLibrary,
    kernels::{copy, qmv},
};

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::layout::Layout;
use crate::tensor::{LeafStorage, OpKind, OpNode, Tensor, TensorInner, checked_byte_count};

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
        let byte_count = checked_byte_count(&inner.shape, inner.dtype)?;
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
    } else if let Some(view) = inner.view.as_ref() {
        // Views own no storage; eval needs to chase the source so the
        // op that backs the view (if any) gets dispatched. The view
        // itself never enters `order` — it has nothing to dispatch.
        visit(&view.source.inner, visited, order);
    }
    // Placeholders (no op, no view, no cache) are silently skipped —
    // eval() catches the broken case at step 2 if the root needs
    // dispatching.
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
