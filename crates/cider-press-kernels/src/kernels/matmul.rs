//! Batched bf16 matrix multiply over our own `matmul.metal`.
//!
//! This is the **first kernel that is not MLX-derived**: a naive
//! one-thread-per-output GEMM, float accumulator, contiguous bf16
//! inputs. The MLX kernels (`steel_gemm_*.metal`) are tiled and
//! register-blocked; we deliberately skip that complexity until
//! perf-branch 15. SDPA shapes (Q @ K^T and attn @ V at Qwen2.5-0.5B
//! head sizes) keep this kernel inside the budget even unoptimized.
//!
//! Inputs: `A [Batch, M, K]`, `B [Batch, K, N]`, both contiguous bf16.
//! Output: `C [Batch, M, N]`, contiguous bf16. Leading-dim batching
//! is flattened by the host (the caller reshapes higher-rank tensors
//! to rank-3 first); per-batch element strides are passed as
//! `a_batch_stride = M * K`, etc., leaving room to broadcast a
//! single-batch B across the batch dim by passing
//! `b_batch_stride = 0` if a future caller wants it (no current
//! consumer does — SDPA's GQA broadcast materializes via
//! `Tensor::copy`).

#![allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]

use std::ffi::c_void;
use std::ptr::NonNull;

use half::bf16;
use objc2_metal::{MTLComputeCommandEncoder, MTLSize};

use crate::buffer::Buffer;
use crate::commands::Commands;
use crate::error::{Error, Result};
use crate::library::KernelLibrary;

/// Threadgroup tile: one thread per output element. 8 × 8 is a safe
/// default well under Apple Silicon's max-threads-per-threadgroup
/// (1024 on every M-series we target) and aligns naturally to the
/// 32-wide SIMD width. Tuning is a branch-15 concern.
const TG_M: usize = 8;
const TG_N: usize = 8;

/// Dispatch the naive bf16 matmul. `batch` is the product of all
/// leading dims; `m`, `n`, `k` are the matmul dimensions.
/// `a_batch_stride` / `b_batch_stride` / `c_batch_stride` are in
/// **elements**, not bytes — typically `m * k`, `k * n`, `m * n`
/// respectively for the unbroadcast case.
///
/// Preconditions:
/// - `src_a` has at least `batch * a_batch_stride` elements visible
///   (and at least `m * k` per batch tile).
/// - `src_b` has at least `batch * b_batch_stride` elements visible
///   (and at least `k * n` per batch tile).
/// - `dst` has at least `batch * c_batch_stride` elements writable
///   (and at least `m * n` per batch tile).
#[allow(clippy::too_many_arguments)]
pub fn dispatch_gemm_bf16(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    src_a: &Buffer<bf16>,
    src_b: &Buffer<bf16>,
    dst: &mut Buffer<bf16>,
    m: usize,
    n: usize,
    k: usize,
    batch: usize,
    a_batch_stride: usize,
    b_batch_stride: usize,
    c_batch_stride: usize,
) -> Result<()> {
    if m == 0 || n == 0 || k == 0 || batch == 0 {
        return Ok(());
    }

    let m_u32 = u32::try_from(m)
        .map_err(|_| Error::InvalidArgument(format!("matmul: M={m} does not fit in u32")))?;
    let n_u32 = u32::try_from(n)
        .map_err(|_| Error::InvalidArgument(format!("matmul: N={n} does not fit in u32")))?;
    let k_u32 = u32::try_from(k)
        .map_err(|_| Error::InvalidArgument(format!("matmul: K={k} does not fit in u32")))?;
    let a_stride_u32 = u32::try_from(a_batch_stride).map_err(|_| {
        Error::InvalidArgument(format!(
            "matmul: a_batch_stride={a_batch_stride} does not fit in u32",
        ))
    })?;
    let b_stride_u32 = u32::try_from(b_batch_stride).map_err(|_| {
        Error::InvalidArgument(format!(
            "matmul: b_batch_stride={b_batch_stride} does not fit in u32",
        ))
    })?;
    let c_stride_u32 = u32::try_from(c_batch_stride).map_err(|_| {
        Error::InvalidArgument(format!(
            "matmul: c_batch_stride={c_batch_stride} does not fit in u32",
        ))
    })?;

    let need_a = batch
        .checked_mul(a_batch_stride)
        .ok_or_else(|| Error::InvalidArgument("matmul: A footprint overflows usize".into()))?;
    let need_b = batch
        .checked_mul(b_batch_stride)
        .ok_or_else(|| Error::InvalidArgument("matmul: B footprint overflows usize".into()))?;
    let need_c = batch
        .checked_mul(c_batch_stride)
        .ok_or_else(|| Error::InvalidArgument("matmul: C footprint overflows usize".into()))?;
    if src_a.len() < need_a || src_b.len() < need_b || dst.len() < need_c {
        return Err(Error::InvalidArgument(format!(
            "matmul: buffer too small (need A={need_a}, B={need_b}, C={need_c}; \
             got A={}, B={}, C={})",
            src_a.len(),
            src_b.len(),
            dst.len(),
        )));
    }

    let pipeline = library.pipeline("gemm_bfloat16")?;
    let encoder = commands.encoder()?;
    encoder.setComputePipelineState(pipeline.metal_pipeline_state());

    // SAFETY: bindings match the kernel signature in matmul.metal —
    //   0 A (bf16, batch * a_batch_stride)
    //   1 B (bf16, batch * b_batch_stride)
    //   2 C (bf16, batch * c_batch_stride)
    //   3 M, 4 N, 5 K (uint)
    //   6 a_batch_stride, 7 b_batch_stride, 8 c_batch_stride (uint)
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(src_a.metal_buffer()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(src_b.metal_buffer()), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(dst.metal_buffer()), 0, 2);
        let m_ptr: NonNull<c_void> = NonNull::from(&m_u32).cast();
        encoder.setBytes_length_atIndex(m_ptr, std::mem::size_of::<u32>(), 3);
        let n_ptr: NonNull<c_void> = NonNull::from(&n_u32).cast();
        encoder.setBytes_length_atIndex(n_ptr, std::mem::size_of::<u32>(), 4);
        let k_ptr: NonNull<c_void> = NonNull::from(&k_u32).cast();
        encoder.setBytes_length_atIndex(k_ptr, std::mem::size_of::<u32>(), 5);
        let a_ptr: NonNull<c_void> = NonNull::from(&a_stride_u32).cast();
        encoder.setBytes_length_atIndex(a_ptr, std::mem::size_of::<u32>(), 6);
        let b_ptr: NonNull<c_void> = NonNull::from(&b_stride_u32).cast();
        encoder.setBytes_length_atIndex(b_ptr, std::mem::size_of::<u32>(), 7);
        let c_ptr: NonNull<c_void> = NonNull::from(&c_stride_u32).cast();
        encoder.setBytes_length_atIndex(c_ptr, std::mem::size_of::<u32>(), 8);
    }

    let grid = MTLSize {
        width: m,
        height: n,
        depth: batch,
    };
    let tg = MTLSize {
        width: TG_M.min(m),
        height: TG_N.min(n),
        depth: 1,
    };
    encoder.dispatchThreads_threadsPerThreadgroup(grid, tg);
    Ok(())
}
