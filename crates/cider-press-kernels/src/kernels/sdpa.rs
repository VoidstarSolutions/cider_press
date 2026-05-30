// Dispatch routine derived from MLX (https://github.com/ml-explore/mlx).
// Copyright © 2023-2024 Apple Inc. Released under the MIT license — see
// the full upstream notice at `kernels-mlx/COPYING`.

//! Fused scaled-dot-product attention dispatch (vector / decode path).
//!
//! Dispatches MLX's vendored `sdpa_vector`: one threadgroup per
//! (batch·query-head), reading K/V strided in place — no GQA-broadcast
//! or transpose copies. GQA in-kernel via `gqa_factor`. Mirrors MLX's
//! `sdpa_vector()` (`backend/metal/scaled_dot_product_attention.cpp`).
//! Decode-only: no mask/causal/sinks, query not transposed (those
//! function constants are pinned false; masked / 2-pass paths land
//! later).

use std::ffi::c_void;
use std::ptr::NonNull;

use half::bf16;
use objc2_metal::{MTLComputeCommandEncoder, MTLSize};

use crate::buffer::Buffer;
use crate::commands::Commands;
use crate::error::{Error, Result};
use crate::library::{FunctionConstant, KernelLibrary};

/// Scalar params for [`dispatch_sdpa_vector_bf16`]. Strides are in
/// *elements* — the MSL kernel does typed (`device T*`) pointer math, so
/// they mirror MLX's `array.strides()` (element strides). For a
/// row-contiguous K/V of shape `[1, H_kv, T, D]`: head stride `T*D`, seq
/// stride `D`.
#[derive(Clone, Copy, Debug)]
pub struct SdpaVectorArgs {
    /// `H_q / H_kv` — the in-kernel GQA broadcast factor.
    pub gqa_factor: i32,
    /// Number of keys / KV sequence length (`N` in the kernel).
    pub n_keys: i32,
    /// Element stride between K heads.
    pub k_head_stride: u64,
    /// Element stride between K sequence positions.
    pub k_seq_stride: u64,
    /// Element stride between V heads.
    pub v_head_stride: u64,
    /// Element stride between V sequence positions.
    pub v_seq_stride: u64,
    /// Attention scale, typically `1/sqrt(head_dim)`.
    pub scale: f32,
    /// Per-head feature dimension (`D`). Selects the kernel
    /// specialization `sdpa_vector_bfloat16_t_{D}_{D}`.
    pub head_dim: usize,
}

/// `sdpa_vector_bfloat16_t_{D}_{D}` with all six bool function constants
/// (indices 20-25) pinned false: no mask, query not transposed, no
/// causal, no bool/float mask, no sinks — the single-query decode case.
///
/// `q` is the single query row(s) `[1, H_q, 1, D]` (`H_q*D` elements);
/// `k`/`v` are the KV cache `[1, H_kv, N, D]`. `out` must hold `H_q*D`
/// elements. The number of query heads is derived as
/// `out.len() / head_dim`. One threadgroup is dispatched per query head
/// (MLX's `grid = (B*H_q, T_q, 1)` threadgroups, `T_q = 1` here).
#[allow(clippy::too_many_lines)]
pub fn dispatch_sdpa_vector_bf16(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    q: &Buffer<bf16>,
    k: &Buffer<bf16>,
    v: &Buffer<bf16>,
    out: &mut Buffer<bf16>,
    args: SdpaVectorArgs,
) -> Result<()> {
    if args.head_dim == 0 {
        return Err(Error::InvalidArgument(
            "sdpa_vector: head_dim must be positive".into(),
        ));
    }
    if out.len() % args.head_dim != 0 {
        return Err(Error::InvalidArgument(format!(
            "sdpa_vector: out length {} is not a multiple of head_dim {}",
            out.len(),
            args.head_dim,
        )));
    }
    let n_q_heads = out.len() / args.head_dim;
    if n_q_heads == 0 {
        return Err(Error::InvalidArgument(
            "sdpa_vector: out buffer is empty (zero query heads)".into(),
        ));
    }
    if q.len() < n_q_heads * args.head_dim {
        return Err(Error::InvalidArgument(format!(
            "sdpa_vector: q too small (need {}, got {})",
            n_q_heads * args.head_dim,
            q.len(),
        )));
    }
    if args.n_keys <= 0 {
        return Err(Error::InvalidArgument(format!(
            "sdpa_vector: n_keys must be positive (got {})",
            args.n_keys
        )));
    }
    if args.gqa_factor <= 0 {
        return Err(Error::InvalidArgument(format!(
            "sdpa_vector: gqa_factor must be positive (got {})",
            args.gqa_factor
        )));
    }

    let kname = format!("sdpa_vector_bfloat16_t_{}_{}", args.head_dim, args.head_dim);
    let pipeline = library.pipeline_specialized(
        &kname,
        &[
            FunctionConstant::Bool {
                index: 20,
                value: false,
            }, // has_mask
            FunctionConstant::Bool {
                index: 21,
                value: false,
            }, // query_transposed
            FunctionConstant::Bool {
                index: 22,
                value: false,
            }, // do_causal
            FunctionConstant::Bool {
                index: 23,
                value: false,
            }, // bool_mask
            FunctionConstant::Bool {
                index: 24,
                value: false,
            }, // float_mask
            FunctionConstant::Bool {
                index: 25,
                value: false,
            }, // has_sinks
               // Index 26 (`blocks`, int) is the 2-pass split-K partition count —
               // unused by the single-pass vector kernel, so it is left unbound.
        ],
    )?;

    let encoder = commands.encoder()?;
    encoder.setComputePipelineState(pipeline.metal_pipeline_state());

    // SAFETY: buffer / byte indices match the `sdpa_vector` MSL signature
    // (sdpa_vector.h): q=0, k=1, v=2, out=3; gqa_factor(int)=4, N(int)=5,
    // k_head_stride(size_t)=6, k_seq_stride=7, v_head_stride=8,
    // v_seq_stride=9, scale(float)=10. The mask/sinks slots (11-17) are
    // gated behind the function constants we pinned false above, so they
    // are absent from this specialization and must not be bound. Caller
    // sized the buffers to match (checked above for q/out).
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(q.metal_buffer()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(k.metal_buffer()), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(v.metal_buffer()), 0, 2);
        encoder.setBuffer_offset_atIndex(Some(out.metal_buffer()), 0, 3);

        let gqa: NonNull<c_void> = NonNull::from(&args.gqa_factor).cast();
        encoder.setBytes_length_atIndex(gqa, std::mem::size_of::<i32>(), 4);
        let n: NonNull<c_void> = NonNull::from(&args.n_keys).cast();
        encoder.setBytes_length_atIndex(n, std::mem::size_of::<i32>(), 5);
        let khs: NonNull<c_void> = NonNull::from(&args.k_head_stride).cast();
        encoder.setBytes_length_atIndex(khs, std::mem::size_of::<u64>(), 6);
        let kss: NonNull<c_void> = NonNull::from(&args.k_seq_stride).cast();
        encoder.setBytes_length_atIndex(kss, std::mem::size_of::<u64>(), 7);
        let vhs: NonNull<c_void> = NonNull::from(&args.v_head_stride).cast();
        encoder.setBytes_length_atIndex(vhs, std::mem::size_of::<u64>(), 8);
        let vss: NonNull<c_void> = NonNull::from(&args.v_seq_stride).cast();
        encoder.setBytes_length_atIndex(vss, std::mem::size_of::<u64>(), 9);
        let scale: NonNull<c_void> = NonNull::from(&args.scale).cast();
        encoder.setBytes_length_atIndex(scale, std::mem::size_of::<f32>(), 10);
    }

    // Grid: one THREADGROUP per (batch·query-head). MLX dispatches
    // `grid = (B*H_q, T_q, 1)` *threadgroups* (not threads) with a
    // `(1024, 1, 1)` group — see `sdpa_vector()`'s
    // `dispatch_threadgroups`. T_q = 1 for the decode path.
    let grid = MTLSize {
        width: n_q_heads,
        height: 1,
        depth: 1,
    };
    let group = MTLSize {
        width: 1024,
        height: 1,
        depth: 1,
    };
    encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, group);
    Ok(())
}
