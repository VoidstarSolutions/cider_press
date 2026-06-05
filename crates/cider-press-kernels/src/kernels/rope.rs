// Dispatch routine derived from MLX (https://github.com/ml-explore/mlx).
// Copyright © 2023-2024 Apple Inc. Released under the MIT license — see
// the full upstream notice at `kernels-mlx/COPYING`.

//! Rotary positional-embedding dispatch over MLX's `rope.metal`.
//!
//! Exposes the `rope_bfloat16` entry point with `forward=true`,
//! `traditional=false`, `hs_transpose=false` — the `Qwen2`-inference
//! specialization, computing inverse frequencies on-GPU from `base`
//! (no explicit `freqs` array).
//!
//! Deferred upstream variants: `_single` (`T=1`, contiguous fast
//! path — non-`_single` handles the same shape with marginal extra
//! per-thread bookkeeping), `_freqs` (explicit-frequencies array for
//! `YaRN` / Llama-3 scaling — Qwen2 doesn't use them), `_large` (int64
//! indexing — Qwen2 input sizes fit int32), `vjp` (backward — inference
//! only), `traditional` (GPT-NeoX uses interleaved layout, Qwen2 uses
//! the `(x[:d/2], x[d/2:])` non-traditional split), and `hs_transpose`
//! (head/sequence swap in the input layout — Qwen2's projection
//! outputs are already in the expected `[B, H, T, D]` order). Add when
//! their first consumer arrives.
//!
//! Analogue of MLX's `RoPE::eval_gpu` (`backend/metal/rope.cpp`).

#![allow(clippy::cast_sign_loss)]

use std::ffi::c_void;
use std::ptr::NonNull;

use half::bf16;
use objc2_metal::{MTLComputeCommandEncoder, MTLSize};

use crate::buffer::Buffer;
use crate::commands::Commands;
use crate::error::{Error, Result};
use crate::library::{FunctionConstant, KernelLibrary};

/// MLX's per-thread head fan-out for the non-`_single` rope variants
/// (the templated `N` in `rope_impl`). Hard-coded upstream as the
/// `n_per_thread` constexpr in `backend/metal/rope.cpp`.
const N_PER_THREAD: i32 = 4;

/// Dispatch arguments for [`dispatch_rope_bf16`]. Mirrors the buffer /
/// bytes plumbing in MLX's `RoPE::eval_gpu` for the non-`_single`,
/// non-`_freqs`, non-`_large` case.
///
/// Strides are in *elements*, not bytes — matches the MSL kernel
/// signature `constant const int64_t strides[3]`.
#[derive(Debug, Clone, Copy)]
pub struct RopeArgs {
    /// Total leading-dim product (`B` in MLX terminology). For
    /// per-sample-shaped Q/K we usually pass `1`.
    pub batch: i32,
    /// Product of all dimensions between batch and `T`: number of
    /// heads being rotated. Q for Qwen2.5-0.5B is 14, K is 2.
    pub n_head: i32,
    /// Sequence length (`T`).
    pub seq_len: i32,
    /// Per-head feature dimension (`D`). Must be even (rotation pairs).
    pub head_dim: i32,
    /// Number of leading per-head slots that get rotated (`dims_` in
    /// MLX). Equals `head_dim` for full-rotation models like Qwen2.
    pub rotary_dims: i32,
    /// MLX's `scale_` — multiplies the position before computing the
    /// rotation angle (`1.0` for the standard formula).
    pub scale: f32,
    /// `log2(theta)`. The MSL kernel computes
    /// `inv_freq = exp2(-d * base)` so the host passes the log already
    /// — see `backend/metal/rope.cpp` line `float base = std::log2(base_);`.
    pub base_log2: f32,
    /// Element strides over the (head, sequence, feature) axes of the
    /// input tensor. For a row-contiguous `[B, H, T, D]` tensor:
    /// `[T*D, D, 1]`.
    pub strides: [i64; 3],
    /// Same as `strides` but for the output tensor — usually identical
    /// since we allocate the output row-contiguous.
    pub out_strides: [i64; 3],
    /// Element stride of the `offset` buffer along its leading
    /// dimension. `0` for a 1-element scalar offset.
    pub offset_stride: i64,
}

/// `rope_bfloat16` with `forward=true, traditional=false,
/// hs_transpose=false`.
///
/// Both `src` and `dst` must hold at least
/// `batch * n_head * seq_len * head_dim` elements. The kernel reads
/// `src` and writes `dst`; in-place rotation is allowed (pass the same
/// `Buffer<bf16>` aliased through [`crate::Buffer::clone_handle`]) but
/// the dispatch itself does not handle the alias for you.
///
/// `offset` is a device-side scalar (or per-batch vector when batched)
/// holding the starting sequence position — the kernel adds the
/// `[0, T)` thread coordinate to it to derive the rotated position.
/// MLX models pass this as a 1-element `int32` array.
pub fn dispatch_rope_bf16(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    src: &Buffer<bf16>,
    dst: &mut Buffer<bf16>,
    offset: &Buffer<i32>,
    args: RopeArgs,
) -> Result<()> {
    if args.batch <= 0 || args.n_head <= 0 || args.seq_len <= 0 || args.head_dim <= 0 {
        return Err(Error::InvalidArgument(format!(
            "rope: dims must be positive (batch={}, n_head={}, seq_len={}, head_dim={})",
            args.batch, args.n_head, args.seq_len, args.head_dim
        )));
    }
    if args.head_dim % 2 != 0 {
        return Err(Error::InvalidArgument(format!(
            "rope: head_dim must be even (head_dim={})",
            args.head_dim
        )));
    }
    if args.rotary_dims <= 0 || args.rotary_dims > args.head_dim || args.rotary_dims % 2 != 0 {
        return Err(Error::InvalidArgument(format!(
            "rope: rotary_dims must be even and in (0, head_dim] \
             (rotary_dims={}, head_dim={})",
            args.rotary_dims, args.head_dim
        )));
    }

    let elem_count = (args.batch as usize)
        .checked_mul(args.n_head as usize)
        .and_then(|x| x.checked_mul(args.seq_len as usize))
        .and_then(|x| x.checked_mul(args.head_dim as usize))
        .ok_or_else(|| Error::InvalidArgument("rope: element count overflows usize".into()))?;
    if src.len() < elem_count || dst.len() < elem_count {
        return Err(Error::InvalidArgument(format!(
            "rope: src/dst too small (need {elem_count}, got src={}, dst={})",
            src.len(),
            dst.len(),
        )));
    }
    if offset.is_empty() {
        return Err(Error::InvalidArgument(
            "rope: offset buffer is empty".into(),
        ));
    }

    let pipeline = library.pipeline_specialized(
        "rope_bfloat16",
        &[
            FunctionConstant::Bool {
                index: 1,
                value: true,
            },
            FunctionConstant::Bool {
                index: 2,
                value: false,
            },
            FunctionConstant::Bool {
                index: 3,
                value: false,
            },
        ],
    )?;

    let gx = (args.rotary_dims / 2) as usize;
    let gy = args.seq_len as usize;
    let gz = (args.batch as usize) * (args.n_head as usize).div_ceil(N_PER_THREAD as usize);

    {
        let encoder = commands.encoder()?;
        encoder.setComputePipelineState(pipeline.metal_pipeline_state());
        // SAFETY: slot/dtype layout matches the `rope` kernel signature in
        // rope.metal under the chosen function-constant specialization.
        // Bindings:
        //   0  device T* in            (bf16, elem_count)
        //   1  device T* out           (bf16, elem_count)
        //   2  device const int* offset
        //   3  bytes scale (float)
        //   4  bytes strides[3]  (int64)
        //   5  bytes out_strides[3] (int64)
        //   6  bytes offset_stride (int64)
        //   7  bytes n_head (int)
        //  10  bytes base (float, == log2(theta))
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(src.metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(dst.metal_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(offset.metal_buffer()), 0, 2);

            let scale_ptr: NonNull<c_void> = NonNull::from(&args.scale).cast();
            encoder.setBytes_length_atIndex(scale_ptr, std::mem::size_of::<f32>(), 3);

            let strides_ptr: NonNull<c_void> = NonNull::from(&args.strides).cast();
            encoder.setBytes_length_atIndex(strides_ptr, std::mem::size_of::<[i64; 3]>(), 4);

            let out_strides_ptr: NonNull<c_void> = NonNull::from(&args.out_strides).cast();
            encoder.setBytes_length_atIndex(out_strides_ptr, std::mem::size_of::<[i64; 3]>(), 5);

            let offset_stride_ptr: NonNull<c_void> = NonNull::from(&args.offset_stride).cast();
            encoder.setBytes_length_atIndex(offset_stride_ptr, std::mem::size_of::<i64>(), 6);

            let n_head_ptr: NonNull<c_void> = NonNull::from(&args.n_head).cast();
            encoder.setBytes_length_atIndex(n_head_ptr, std::mem::size_of::<i32>(), 7);

            let base_ptr: NonNull<c_void> = NonNull::from(&args.base_log2).cast();
            encoder.setBytes_length_atIndex(base_ptr, std::mem::size_of::<f32>(), 10);
        }
    }

    let grid = MTLSize {
        width: gx,
        height: gy,
        depth: gz,
    };
    let threadgroup = get_block_dims(gx, gy, gz);
    commands.dispatch_threads(grid, threadgroup)
}

/// Ported `get_block_dims_common` from MLX
/// (`mlx/backend/common/utils.cpp`). Picks per-axis powers of two for
/// the threadgroup such that the product is ≤ 1024 and each axis grows
/// monotonically as the corresponding grid dim does. The original takes
/// a `pow2` ceiling; we hard-code Apple's standard 1024 cap.
fn get_block_dims(dim0: usize, dim1: usize, dim2: usize) -> MTLSize {
    let grid = [dim0, dim1, dim2];
    let mut pows = [0u32; 3];
    let mut sum = 0u32;
    loop {
        let pre = sum;
        for axis in 0..3 {
            if grid[axis] >= 1usize << (pows[axis] + 1) {
                pows[axis] += 1;
                sum += 1;
            }
            if sum == 10 {
                return MTLSize {
                    width: 1 << pows[0],
                    height: 1 << pows[1],
                    depth: 1 << pows[2],
                };
            }
        }
        if sum == pre {
            return MTLSize {
                width: 1 << pows[0],
                height: 1 << pows[1],
                depth: 1 << pows[2],
            };
        }
    }
}
