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
//!
//! Explained in the inference guide: [`docs/inference/attention.md`](../../../../docs/inference/attention.md).

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
    // GQA maps each query head to `q_head / gqa_factor`; an uneven split
    // would index a KV head past the cache. The strided K/V reads run
    // `keys[(n_kv_heads-1)*head_stride + (n_keys-1)*seq_stride + d]`, so
    // bound both buffers against that maximum element before dispatching —
    // a short K/V (or oversized strides) would otherwise read OOB on-GPU.
    let gqa = usize::try_from(args.gqa_factor).expect("gqa_factor > 0 checked above");
    if n_q_heads % gqa != 0 {
        return Err(Error::InvalidArgument(format!(
            "sdpa_vector: n_q_heads ({n_q_heads}) must be divisible by gqa_factor ({gqa})"
        )));
    }
    let n_kv_heads = n_q_heads / gqa;
    let n_keys = usize::try_from(args.n_keys).expect("n_keys > 0 checked above");
    let khs = usize::try_from(args.k_head_stride)
        .map_err(|_| Error::InvalidArgument("sdpa_vector: k_head_stride too large".into()))?;
    let kss = usize::try_from(args.k_seq_stride)
        .map_err(|_| Error::InvalidArgument("sdpa_vector: k_seq_stride too large".into()))?;
    let vhs = usize::try_from(args.v_head_stride)
        .map_err(|_| Error::InvalidArgument("sdpa_vector: v_head_stride too large".into()))?;
    let vss = usize::try_from(args.v_seq_stride)
        .map_err(|_| Error::InvalidArgument("sdpa_vector: v_seq_stride too large".into()))?;
    let max_index = |head_stride: usize, seq_stride: usize| {
        (n_kv_heads - 1)
            .saturating_mul(head_stride)
            .saturating_add((n_keys - 1).saturating_mul(seq_stride))
            .saturating_add(args.head_dim)
    };
    let required_k = max_index(khs, kss);
    if k.len() < required_k {
        return Err(Error::InvalidArgument(format!(
            "sdpa_vector: k too small (need {required_k}, got {})",
            k.len()
        )));
    }
    let required_v = max_index(vhs, vss);
    if v.len() < required_v {
        return Err(Error::InvalidArgument(format!(
            "sdpa_vector: v too small (need {required_v}, got {})",
            v.len()
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

    {
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
    commands.dispatch_threadgroups(grid, group)
}

/// Mirrors `mlx::steel::AttnParams` (`steel/attn/params.h`) byte-for-byte
/// — copied raw into the kernel via `setBytes`, so field order and types
/// must match the C++ struct exactly. Strides are *element* strides for
/// the `(B, H, L)` axes; the head-dim axis `D` is implicitly contiguous
/// (stride 1) in the kernel's typed pointer math.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct AttnParams {
    pub b: i32,
    pub h: i32,
    pub d: i32,
    pub q_l: i32,
    pub k_l: i32,
    pub gqa_factor: i32,
    pub scale: f32,
    pub n_q: i32,
    pub n_k: i32,
    pub n_q_aligned: i32,
    pub n_k_aligned: i32,
    pub q_l_rem: i32,
    pub k_l_rem: i32,
    pub q_l_off: i32,
    pub q_strides: [i64; 3],
    pub k_strides: [i64; 3],
    pub v_strides: [i64; 3],
    pub o_strides: [i64; 3],
}

/// Constants for the pinned `bd64` steel attention instantiation
/// (`head_dim < 128` ⇒ `bk = 32`). MLX picks `bq = 32`, `wm = 4`,
/// `wn = 1` for this shape (`sdpa_full_self_attention_metal`).
const BQ: i32 = 32;
const BK: i32 = 32;
const WM: usize = 4;
const WN: usize = 1;

/// Scalar params for [`dispatch_sdpa_full_bf16`]. Strides are in
/// *elements* over the `(B, H, L)` axes (head-dim contiguous). For a
/// row-contiguous Q/O of shape `[B, H_q, T, D]`: batch stride `H_q*T*D`,
/// head stride `T*D`, seq stride `D`. Same for K/V with `H_kv`.
#[derive(Clone, Copy, Debug)]
pub struct SdpaFullArgs {
    pub batch: i32,
    pub h_q: i32,
    pub h_kv: i32,
    pub head_dim: i32,
    pub q_len: i32,
    pub k_len: i32,
    pub scale: f32,
    pub q_strides: [i64; 3],
    pub k_strides: [i64; 3],
    pub v_strides: [i64; 3],
    pub o_strides: [i64; 3],
    pub causal: bool,
}

/// Fused prefill attention via MLX's vendored `steel_attention`
/// (`sdpa_full`). Pinned to the `bd64` bf16 instantiation
/// (`steel_attention_bfloat16_bq32_bk32_bd64_wm4_wn1_maskbfloat16`);
/// `head_dim` must be 64. Mirrors `sdpa_full_self_attention_metal`
/// (`backend/metal/scaled_dot_product_attention.cpp`) for the
/// no-mask / no-sinks case: function constants 200/201 (align Q/K),
/// 300 = `has_mask` (false), 301 = `do_causal` (`args.causal`), 302 =
/// `has_sinks` (false). The mask/sinks buffers (5/6/7) are gated behind
/// those false constants, so they are absent from this specialization
/// and must not be bound.
///
/// `out` must hold `h_q * q_len * head_dim` elements. Buffers are
/// bounds-checked against the maximum strided element index before
/// dispatch (fail-loud, no on-GPU OOB).
// `n_q`/`n_k`, `n_q_aligned`/`n_k_aligned`, `q_l_rem`/`k_l_rem` mirror the
// `mlx::steel::AttnParams` field names one-for-one — renaming to dodge the
// near-name lint would obscure the byte-for-byte correspondence.
#[allow(clippy::too_many_lines, clippy::similar_names)]
pub fn dispatch_sdpa_full_bf16(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    q: &Buffer<bf16>,
    k: &Buffer<bf16>,
    v: &Buffer<bf16>,
    out: &mut Buffer<bf16>,
    args: SdpaFullArgs,
) -> Result<()> {
    if args.head_dim != 64 {
        return Err(Error::InvalidArgument(format!(
            "sdpa_full: only head_dim 64 is instantiated (got {})",
            args.head_dim
        )));
    }
    if args.batch <= 0 || args.h_q <= 0 || args.h_kv <= 0 {
        return Err(Error::InvalidArgument(format!(
            "sdpa_full: batch/h_q/h_kv must be positive (got {}, {}, {})",
            args.batch, args.h_q, args.h_kv
        )));
    }
    if args.q_len <= 0 || args.k_len <= 0 {
        return Err(Error::InvalidArgument(format!(
            "sdpa_full: q_len/k_len must be positive (got {}, {})",
            args.q_len, args.k_len
        )));
    }
    if args.h_q % args.h_kv != 0 {
        return Err(Error::InvalidArgument(format!(
            "sdpa_full: h_q ({}) must be divisible by h_kv ({})",
            args.h_q, args.h_kv
        )));
    }
    // `q_l_off = k_len - q_len` is the kernel's causal-window start offset; a
    // negative value (k_len < q_len) would corrupt the causal masking and the
    // strided reads. Self-attention prefill always has k_len >= q_len.
    if args.k_len < args.q_len {
        return Err(Error::InvalidArgument(format!(
            "sdpa_full: k_len ({}) must be >= q_len ({})",
            args.k_len, args.q_len
        )));
    }
    let gqa_factor = args.h_q / args.h_kv;

    let n_q = (args.q_len + BQ - 1) / BQ;
    let n_k = (args.k_len + BK - 1) / BK;
    let n_q_aligned = args.q_len / BQ;
    let n_k_aligned = args.k_len / BK;
    let q_l_rem = args.q_len - n_q_aligned * BQ;
    let k_l_rem = args.k_len - n_k_aligned * BK;
    let q_l_off = args.k_len - args.q_len;

    let align_q = args.q_len % BQ == 0;
    let align_k = args.k_len % BK == 0;

    let params = AttnParams {
        b: args.batch,
        h: args.h_q,
        d: args.head_dim,
        q_l: args.q_len,
        k_l: args.k_len,
        gqa_factor,
        scale: args.scale,
        n_q,
        n_k,
        n_q_aligned,
        n_k_aligned,
        q_l_rem,
        k_l_rem,
        q_l_off,
        q_strides: args.q_strides,
        k_strides: args.k_strides,
        v_strides: args.v_strides,
        o_strides: args.o_strides,
    };

    // Bound-check buffers against the maximum strided element index. The
    // kernel's typed pointer math reaches batch·head·seq via the passed
    // (B, H, L) element strides, with D contiguous — fail loud here rather
    // than OOB on-GPU.
    let head_dim = usize::try_from(args.head_dim).expect("head_dim 64 checked above");
    let stride_bound = |strides: &[i64; 3], heads: i32, len: i32| -> Result<usize> {
        let conv = |s: i64| {
            usize::try_from(s)
                .map_err(|_| Error::InvalidArgument("sdpa_full: stride out of range".into()))
        };
        let batch_stride = conv(strides[0])?;
        let head_stride = conv(strides[1])?;
        let seq_stride = conv(strides[2])?;
        let n_batch = usize::try_from(args.batch).expect("batch > 0 checked above");
        let n_heads = usize::try_from(heads).expect("heads > 0 checked above");
        let n_seq = usize::try_from(len).expect("len > 0 checked above");
        Ok((n_batch - 1)
            .saturating_mul(batch_stride)
            .saturating_add((n_heads - 1).saturating_mul(head_stride))
            .saturating_add((n_seq - 1).saturating_mul(seq_stride))
            .saturating_add(head_dim))
    };
    let need_q = stride_bound(&args.q_strides, args.h_q, args.q_len)?;
    if q.len() < need_q {
        return Err(Error::InvalidArgument(format!(
            "sdpa_full: q too small (need {need_q}, got {})",
            q.len()
        )));
    }
    let need_k = stride_bound(&args.k_strides, args.h_kv, args.k_len)?;
    if k.len() < need_k {
        return Err(Error::InvalidArgument(format!(
            "sdpa_full: k too small (need {need_k}, got {})",
            k.len()
        )));
    }
    let need_v = stride_bound(&args.v_strides, args.h_kv, args.k_len)?;
    if v.len() < need_v {
        return Err(Error::InvalidArgument(format!(
            "sdpa_full: v too small (need {need_v}, got {})",
            v.len()
        )));
    }
    let need_o = stride_bound(&args.o_strides, args.h_q, args.q_len)?;
    if out.len() < need_o {
        return Err(Error::InvalidArgument(format!(
            "sdpa_full: out too small (need {need_o}, got {})",
            out.len()
        )));
    }

    let pipeline = library.pipeline_specialized(
        "steel_attention_bfloat16_bq32_bk32_bd64_wm4_wn1_maskbfloat16",
        &[
            FunctionConstant::Bool {
                index: 200,
                value: align_q,
            },
            FunctionConstant::Bool {
                index: 201,
                value: align_k,
            },
            FunctionConstant::Bool {
                index: 300,
                value: false,
            }, // has_mask
            FunctionConstant::Bool {
                index: 301,
                value: args.causal,
            }, // do_causal
            FunctionConstant::Bool {
                index: 302,
                value: false,
            }, // has_sinks
        ],
    )?;

    {
        let encoder = commands.encoder()?;
        encoder.setComputePipelineState(pipeline.metal_pipeline_state());
        // SAFETY: buffer / byte indices match the `steel_attention` MSL
        // signature: q=0, k=1, v=2, o=3, params(AttnParams)=4. The mask
        // params/array (5/6) and sinks (7) are gated behind has_mask /
        // has_sinks (both pinned false), so they are absent from this
        // specialization and must not be bound. Buffers sized to match
        // (checked above).
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(q.metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(k.metal_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(v.metal_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(out.metal_buffer()), 0, 3);
            let params_ptr: NonNull<c_void> = NonNull::from(&params).cast();
            encoder.setBytes_length_atIndex(params_ptr, std::mem::size_of::<AttnParams>(), 4);
        }
    }

    // MLX dispatches `grid = (NQ, H, B)` *threadgroups* with a
    // `(32, wm, wn)` group — see `sdpa_full_self_attention_metal`.
    let grid = MTLSize {
        width: usize::try_from(n_q).expect("n_q >= 1"),
        height: usize::try_from(args.h_q).expect("h_q > 0 checked above"),
        depth: usize::try_from(args.batch).expect("batch > 0 checked above"),
    };
    let group = MTLSize {
        width: 32,
        height: WM,
        depth: WN,
    };
    commands.dispatch_threadgroups(grid, group)
}
