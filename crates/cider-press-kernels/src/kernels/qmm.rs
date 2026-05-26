// Dispatch routine derived from MLX (https://github.com/ml-explore/mlx).
// Copyright © 2023-2024 Apple Inc. Released under the MIT license — see
// the full upstream notice at `kernels-mlx/COPYING`.

//! Quantized matrix-matrix dispatch over MLX's `quantized.metal`.
//!
//! Wraps the `affine_qmm_t` family — matrix-matrix multiply against an
//! affine-quantized weight matrix (packed `int4`/`int8` lanes, per-group
//! scale + bias) with `transpose=true`. This is the inference hot path for
//! prefill: every transformer projection hits it once per prompt chunk.
//!
//! Analogue of `mlx::core::metal::qmm()` in
//! `mlx/backend/metal/quantized.cpp`. This module covers the
//! `transpose=true, aligned=true (N % 32 == 0), batched=false (B=1)` case.

#![allow(
    clippy::too_many_arguments,
    clippy::many_single_char_names,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation
)]

use std::ffi::c_void;
use std::ptr::NonNull;

use half::bf16;
use objc2_metal::{MTLComputeCommandEncoder, MTLSize};

use crate::buffer::Buffer;
use crate::commands::Commands;
use crate::error::{Error, Result};
use crate::library::KernelLibrary;

/// Kernel-side tile sizes from `mlx/backend/metal/quantized.cpp::qmm()`.
/// `wm = 2, wn = 2, bm = 32, bn = 32`; threadgroup is `(32, wn, wm)`;
/// grid is `((N + bn - 1) / bn, (M + bm - 1) / bm, B)` in *threadgroups*,
/// dispatched via `dispatchThreadgroups:threadsPerThreadgroup:`.
const BM: usize = 32;
const BN: usize = 32;

/// Quantized matrix-matrix multiply: `y = x · dequant(w_q)ᵀ`, all bf16-typed.
///
/// `library` must be a [`KernelLibrary::quantized`]. Encodes one dispatch into
/// `commands`; caller flushes via [`Commands::commit_and_wait`].
///
/// # Shape requirements
///
/// - `w_q.len()    == N * K * bits / 32`  (int4 lanes packed into uint32)
/// - `scales.len() == N * K / group_size`
/// - `biases.len() == N * K / group_size`
/// - `x.len()      == M * K`
/// - `y.len()      == M * N`
/// - `N % 32 == 0` (aligned variant only; generic deferred)
///
/// `group_size` must be in `{32, 64, 128}` and `bits` must be in
/// `{2, 3, 4, 5, 6, 8}` — these are the combinations MLX's
/// `instantiate_quantized_all` macro emits.
///
/// # Notes
///
/// - Only `transpose=true, aligned=true (N % 32 == 0), batched=false (B=1)`
///   is wired. The `_qmm_n_` (non-transposed) and `_alN_false` variants land
///   when a consumer needs them.
pub fn affine_qmm_t_bf16(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    w_q: &Buffer<u32>,
    scales: &Buffer<bf16>,
    biases: &Buffer<bf16>,
    x: &Buffer<bf16>,
    y: &mut Buffer<bf16>,
    m: usize,
    n: usize,
    k: usize,
    group_size: u32,
    bits: u32,
) -> Result<()> {
    let m_i32 = i32::try_from(m)
        .map_err(|_| Error::InvalidArgument(format!("affine_qmm_t: M={m} does not fit in i32")))?;
    let n_i32 = i32::try_from(n)
        .map_err(|_| Error::InvalidArgument(format!("affine_qmm_t: N={n} does not fit in i32")))?;
    let k_i32 = i32::try_from(k)
        .map_err(|_| Error::InvalidArgument(format!("affine_qmm_t: K={k} does not fit in i32")))?;
    if m_i32 <= 0 || n_i32 <= 0 || k_i32 <= 0 {
        return Err(Error::InvalidArgument(format!(
            "affine_qmm_t: M ({m}), N ({n}), K ({k}) must all be positive"
        )));
    }
    if group_size == 0 {
        return Err(Error::InvalidArgument(
            "affine_qmm_t: group_size must be positive".to_owned(),
        ));
    }
    if k % (group_size as usize) != 0 {
        return Err(Error::InvalidArgument(format!(
            "affine_qmm_t: K ({k}) must be a multiple of group_size ({group_size})"
        )));
    }
    // Aligned variant only; generic (_alN_false) deferred.
    if n % 32 != 0 {
        return Err(Error::InvalidArgument(format!(
            "affine_qmm_t: N ({n}) must be a multiple of 32 (aligned variant only)"
        )));
    }

    let expected_w_q = n * k * (bits as usize) / 32;
    if w_q.len() != expected_w_q {
        return Err(Error::InvalidArgument(format!(
            "affine_qmm_t: w_q.len()={} != N*K*bits/32 ({expected_w_q})",
            w_q.len(),
        )));
    }
    let groups_per_row = k / (group_size as usize);
    let expected_groups = n * groups_per_row;
    if scales.len() != expected_groups {
        return Err(Error::InvalidArgument(format!(
            "affine_qmm_t: scales.len()={} != N*K/group_size ({expected_groups})",
            scales.len(),
        )));
    }
    if biases.len() != expected_groups {
        return Err(Error::InvalidArgument(format!(
            "affine_qmm_t: biases.len()={} != N*K/group_size ({expected_groups})",
            biases.len(),
        )));
    }
    if x.len() != m * k {
        return Err(Error::InvalidArgument(format!(
            "affine_qmm_t: x.len()={} != M*K ({})",
            x.len(),
            m * k,
        )));
    }
    if y.len() != m * n {
        return Err(Error::InvalidArgument(format!(
            "affine_qmm_t: y.len()={} != M*N ({})",
            y.len(),
            m * n,
        )));
    }

    // transpose=true, aligned=true (N % 32 == 0), batched=false (B=1).
    let kernel_name = format!("affine_qmm_t_bfloat16_t_gs_{group_size}_b_{bits}_alN_true_batch_0");
    let pipeline = library.pipeline(&kernel_name)?;

    let encoder = commands.encoder()?;
    encoder.setComputePipelineState(pipeline.metal_pipeline_state());
    // SAFETY: bind slots and dtypes match the `affine_qmm_t` MSL signature;
    // buffer-bind order transcribed from quantized.cpp::qmm lines 758–768.
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(w_q.metal_buffer()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(scales.metal_buffer()), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(biases.metal_buffer()), 0, 2);
        encoder.setBuffer_offset_atIndex(Some(x.metal_buffer()), 0, 3);
        encoder.setBuffer_offset_atIndex(Some(y.metal_buffer()), 0, 4);
        let k_ptr: NonNull<c_void> = NonNull::from(&k_i32).cast();
        let n_ptr: NonNull<c_void> = NonNull::from(&n_i32).cast();
        let m_ptr: NonNull<c_void> = NonNull::from(&m_i32).cast();
        encoder.setBytes_length_atIndex(k_ptr, std::mem::size_of::<i32>(), 5);
        encoder.setBytes_length_atIndex(n_ptr, std::mem::size_of::<i32>(), 6);
        encoder.setBytes_length_atIndex(m_ptr, std::mem::size_of::<i32>(), 7);
    }

    // Grid is in *threadgroups* (dispatchThreadgroups:), not threads.
    // B=1 so add_strides_and_shapes is a no-op; no further binds needed.
    let grid = MTLSize {
        width: n.div_ceil(BN),
        height: m.div_ceil(BM),
        depth: 1,
    };
    let threadgroup = MTLSize {
        width: 32,
        height: 2,
        depth: 2,
    };
    encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, threadgroup);
    Ok(())
}
