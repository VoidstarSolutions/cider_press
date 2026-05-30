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
    kernels::{binary, copy, qmm, qmv, reduce, unary},
};

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::layout::Layout;
use crate::tensor::{
    BinaryOp, LeafStorage, OpKind, OpNode, ReduceKind, Tensor, TensorInner, UnaryOp,
    checked_byte_count,
};

/// Static label for an [`OpKind`], used as the GPU-segment bucket key in
/// the profiled eval. One label per kind; `QuantizedMatMul` is not split
/// into qmv/qmm here because the kind carries no M — the dispatcher picks
/// the kernel from the activation shape (at decode, M=1, so every segment
/// is a qmv).
#[allow(dead_code)]
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
        OpKind::Softmax { .. } => "softmax",
        OpKind::MatMul => "matmul",
        OpKind::SliceUpdate { .. } => "slice_update",
    }
}

pub(crate) fn eval(root: &Tensor) -> Result<()> {
    let _span = crate::profile::span("tensor.eval");
    // `tensor.eval.encode` isolates the CPU-side cost of *building* the
    // command buffer — the topo walk, per-op output allocation, and
    // dispatch encoding (steps 1–4) — from `tensor.eval.wait`, which is
    // the GPU execution time (the synchronous commit + wait, step 5).
    // Their sum is the whole eval minus the cheap cache-population pass
    // (step 6). Splitting them tells us whether decode is bound by
    // CPU encode/alloc overhead or by GPU kernel time. See
    // `docs/QWEN_PERF.md`.
    let encode = crate::profile::span("tensor.eval.encode");
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
        let op = inner
            .op
            .as_ref()
            .expect("topo invariant: only op nodes are in `order`");
        // SliceUpdate writes into a persistent caller-owned buffer
        // (inputs[0], the slab) rather than a fresh output. Bind that
        // buffer as `dst` so the encoded copy mutates it in place; its
        // contents from earlier (already-committed) evals stay valid.
        let mut dst = if let OpKind::SliceUpdate { .. } = op.kind {
            let slab = op.inputs.first().ok_or_else(|| {
                Error::InvalidArgument("SliceUpdate: missing slab input (inputs[0])".into())
            })?;
            dense_input_buffer(slab, &outputs, &index_of)?
        } else {
            let byte_count = checked_byte_count(&inner.shape, inner.dtype)?;
            device.kernels().alloc_buffer::<u8>(byte_count)?
        };
        dispatch(inner, &mut commands, &outputs, &mut dst, &index_of)?;
        outputs.push(dst);
    }

    // Step 5: synchronous commit + wait. After this returns, every
    // encoded dispatch has completed and the output bytes are valid.
    // Close the encode span here so it excludes GPU time, and time the
    // commit + wait under its own span.
    drop(encode);
    let wait = crate::profile::span("tensor.eval.wait");
    commands.commit_and_wait()?;
    drop(wait);

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
        OpKind::Softmax { precise } => {
            dispatch_softmax(inner, op, commands, outputs, dst, index_of, precise)
        }
        OpKind::MatMul => dispatch_matmul(inner, op, commands, outputs, dst, index_of),
        OpKind::SliceUpdate { offset_rows } => {
            dispatch_slice_update(inner, op, commands, outputs, dst, index_of, offset_rows)
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
    let src = op.inputs.get(1).ok_or_else(|| {
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
    let weight = op.inputs.first().ok_or_else(|| {
        Error::InvalidArgument("QuantizedMatMul: missing weight input (inputs[0])".into())
    })?;
    let x_tensor = op.inputs.get(1).ok_or_else(|| {
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
            group_size,
            u32::from(bits),
        )?;
    } else {
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
    let lhs = op
        .inputs
        .first()
        .ok_or_else(|| Error::InvalidArgument("Binary: missing lhs input (inputs[0])".into()))?;
    let rhs = op
        .inputs
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
    let input = op
        .inputs
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
    let input = op
        .inputs
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
    let src_tensor = op
        .inputs
        .first()
        .ok_or_else(|| Error::InvalidArgument("Gather: missing src (inputs[0])".into()))?;
    let idx_tensor = op
        .inputs
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
    let w_q = op
        .inputs
        .first()
        .ok_or_else(|| Error::InvalidArgument("Dequantize: missing w_q (inputs[0])".into()))?;
    let scales = op
        .inputs
        .get(1)
        .ok_or_else(|| Error::InvalidArgument("Dequantize: missing scales (inputs[1])".into()))?;
    let biases = op
        .inputs
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

    let input = op
        .inputs
        .first()
        .ok_or_else(|| Error::InvalidArgument("Rope: missing input (inputs[0])".into()))?;
    let offset = op
        .inputs
        .get(1)
        .ok_or_else(|| Error::InvalidArgument("Rope: missing offset (inputs[1])".into()))?;

    let in_bytes = dense_input_buffer(input, outputs, index_of)?;
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

    let input = op
        .inputs
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

    let lhs = op
        .inputs
        .first()
        .ok_or_else(|| Error::InvalidArgument("MatMul: missing LHS (inputs[0])".into()))?;
    let rhs = op
        .inputs
        .get(1)
        .ok_or_else(|| Error::InvalidArgument("MatMul: missing RHS (inputs[1])".into()))?;

    // `Tensor::matmul` enforces contiguous strides on each input, but
    // a reshape-of-op still leaves a view in the chain (e.g. SDPA's
    // `probs_3d.reshape([B, H_q, T, T_c])` view of the softmax result).
    // Walk the chain to the storage owner; the contract guarantees a
    // byte-offset of zero and contiguous bytes thereafter.
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
