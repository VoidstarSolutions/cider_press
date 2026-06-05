// Dispatch routine derived from MLX (https://github.com/ml-explore/mlx).
// Copyright © 2023-2024 Apple Inc. Released under the MIT license — see
// the full upstream notice at `kernels-mlx/COPYING`.

//! Quantized matrix-vector dispatch over MLX's `quantized.metal`.
//!
//! Wraps the `affine_qmv` family — matrix-vector multiply against an
//! affine-quantized weight matrix (packed `int4`/`int8` lanes, per-group
//! scale + bias). This is the inference hot path for token generation:
//! every transformer layer hits it once per token.
//!
//! Analogue of `mlx::core::metal::qmv()` in
//! `mlx/backend/metal/quantized.cpp`. Picks the fast variant
//! (`affine_qmv_fast`) when `K % 512 == 0 && N % bn == 0`, generic
//! (`affine_qmv`) otherwise, matching MLX's dispatcher.

#![allow(
    clippy::too_many_arguments,
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

/// Kernel-side tile sizes from `mlx/backend/metal/quantized.cpp::qmv()`.
/// `bn = 8`, `bk = 32`; threadgroup is `(bk, 2, 1)`; grid is
/// `(M, (N + bn - 1) / bn, B)` in *threadgroups*, dispatched via
/// `dispatchThreadgroups:threadsPerThreadgroup:`.
const BN: i32 = 8;
const BK: i32 = 32;

/// Quantized matvec: `y = x · dequant(w_q)ᵀ`, all bf16-typed.
///
/// `library` must be a [`KernelLibrary::quantized`]. Encodes one
/// dispatch into `commands`; caller flushes via
/// [`Commands::commit_and_wait`].
///
/// # Parameters
///
/// - `in_vec_size`: physical K from the weight shape — NOT the activation
///   length. For unpadded weights `x.len() == in_vec_size`; for K-padded
///   weights `x.len() < in_vec_size` and passing `x.len()` would silently
///   compute garbage. Always read from `QuantizedWeight::shape()[1]`.
///
/// # Shape requirements
///
/// Let `K = in_vec_size`, `N = y.len()`. Then:
/// - `w_q.len()  == N * K * bits / 32`  (int4 lanes packed into uint32)
/// - `scales.len() == N * K / group_size`
/// - `biases.len() == N * K / group_size`
///
/// `group_size` must be in `{32, 64, 128}` and `bits` must be in
/// `{2, 3, 4, 5, 6, 8}` — these are the combinations MLX's
/// `instantiate_quantized_all` macro emits. Calling with other values
/// returns [`Error::KernelNotFound`] at
/// the pipeline lookup, because the matching kernel was never
/// instantiated.
///
/// # Notes
///
/// - This function does one `format!` per call to build the kernel name
///   for [`KernelLibrary::pipeline`]. Pipeline cache lookups are O(1)
///   after the first call so the allocation, not the pipeline build,
///   dominates the per-dispatch cost of this function. Acceptable for
///   the runtime layer's needs; revisit if benchmarks point at it.
pub fn affine_qmv_bf16(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    w_q: &Buffer<u32>,
    scales: &Buffer<bf16>,
    biases: &Buffer<bf16>,
    x: &Buffer<bf16>,
    y: &mut Buffer<bf16>,
    in_vec_size: usize,
    group_size: u32,
    bits: u32,
) -> Result<()> {
    // Shape validation. Failing any of these would produce silent garbage
    // on the GPU side (or a div-by-zero panic on `group_size == 0`); reject
    // them with a typed error so callers can recover.
    let k = i32::try_from(in_vec_size).map_err(|_| {
        Error::InvalidArgument(format!("affine_qmv: K={in_vec_size} does not fit in i32"))
    })?;
    // For K-padded weights the physical K exceeds the activation's Rust-logical
    // length. Every Metal allocation is 512-B rounded at
    // `cider_press_kernels::Device::alloc_buffer`, so a 1792-B activation row
    // (896 bf16) has ≥2048-B Metal capacity — enough for a padded-K read of
    // 1024 bf16. Pad weights dequantize to zero so tail values don't affect
    // the result. Assert against the Metal allocation length to catch any
    // future path that skips the 512-B rounding.
    assert!(
        x.metal_alloc_len() >= in_vec_size * 2,
        "affine_qmv: activation Metal allocation too small for K-padded read: \
         metal_alloc_len={} < in_vec_size*2={} (K-padding requires Metal capacity >= physical K * 2)",
        x.metal_alloc_len(),
        in_vec_size * 2,
    );
    let n = i32::try_from(y.len()).map_err(|_| {
        Error::InvalidArgument(format!("affine_qmv: N={} does not fit in i32", y.len()))
    })?;
    if k <= 0 || n <= 0 {
        return Err(Error::InvalidArgument(format!(
            "affine_qmv: K ({k}) and N ({n}) must be positive"
        )));
    }
    if group_size == 0 {
        return Err(Error::InvalidArgument(
            "affine_qmv: group_size must be positive".to_owned(),
        ));
    }
    if k as u32 % group_size != 0 {
        return Err(Error::InvalidArgument(format!(
            "affine_qmv: K ({k}) must be a multiple of group_size ({group_size})"
        )));
    }
    let groups_per_row = (k as usize) / (group_size as usize);
    let expected_w_q = (n as usize) * (k as usize) * (bits as usize) / 32;
    if w_q.len() != expected_w_q {
        return Err(Error::InvalidArgument(format!(
            "affine_qmv: w_q.len()={} != N*K*bits/32 ({expected_w_q})",
            w_q.len(),
        )));
    }
    let expected_groups = (n as usize) * groups_per_row;
    if scales.len() != expected_groups {
        return Err(Error::InvalidArgument(format!(
            "affine_qmv: scales.len()={} != N*K/group_size ({expected_groups})",
            scales.len(),
        )));
    }
    if biases.len() != expected_groups {
        return Err(Error::InvalidArgument(format!(
            "affine_qmv: biases.len()={} != N*K/group_size ({expected_groups})",
            biases.len(),
        )));
    }

    // Mirror MLX's dispatcher: fast variant when both K is a 512-multiple
    // and N is a bn-multiple. See quantized.cpp::qmv.
    let fast = n % BN == 0 && k % 512 == 0;
    let variant = if fast { "qmv_fast" } else { "qmv" };
    // For batch=1 the kernel-name suffix is `_batch_0`; batched dispatch
    // (B > 1) would use `_batch_1` and bind shape/stride buffers. We
    // only do single-batch here for now.
    let kernel_name = format!("affine_{variant}_bfloat16_t_gs_{group_size}_b_{bits}_batch_0");
    let pipeline = library.pipeline(&kernel_name)?;

    {
        let encoder = commands.encoder()?;
        encoder.setComputePipelineState(pipeline.metal_pipeline_state());
        // SAFETY: bind slots and dtypes match the `affine_qmv*` MSL
        // signatures verified bit-exactly against MLX in the qmv parity test.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(w_q.metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(scales.metal_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(biases.metal_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(x.metal_buffer()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(y.metal_buffer()), 0, 4);
            let k_ptr: NonNull<c_void> = NonNull::from(&k).cast();
            let n_ptr: NonNull<c_void> = NonNull::from(&n).cast();
            encoder.setBytes_length_atIndex(k_ptr, std::mem::size_of::<i32>(), 5);
            encoder.setBytes_length_atIndex(n_ptr, std::mem::size_of::<i32>(), 6);
        }
    }

    // Grid is in *threadgroups* (dispatchThreadgroups:), not threads.
    let grid = MTLSize {
        width: 1,
        height: ((n + BN - 1) / BN) as usize,
        depth: 1,
    };
    let threadgroup = MTLSize {
        width: BK as usize,
        height: 2,
        depth: 1,
    };
    commands.dispatch_threadgroups(grid, threadgroup)
}
