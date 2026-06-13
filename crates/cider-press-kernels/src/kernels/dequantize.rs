// Dispatch routine derived from MLX (https://github.com/ml-explore/mlx).
// Copyright © 2023-2024 Apple Inc. Released under the MIT license — see
// the full upstream notice at `kernels-mlx/COPYING`.

//! Affine-dequantize: `out = w * scale + bias` per group, materialising
//! a dense `T` tensor from an affine-quantized `(w_q, scales, biases)`
//! triple.
//!
//! Wraps the `affine_dequantize` kernel instantiated in vendored
//! `quantized.metal` — same library as `affine_qmv`, so no new
//! `KernelLibrary` is needed. Scope mirrors the qmv module: one
//! instantiation, `T = bfloat16_t`, `group_size = 64`, `bits = 4`,
//! the Qwen2 production quantization config.
//!
//! Buffer slot binding mirrors MLX's `affine_dequantize` MSL kernel
//! (`backend/metal/kernels/quantized.h`):
//!
//! | slot | binding |
//! |------|---------|
//! | 0    | `w` (device `uint8_t*`, packed int4)   |
//! | 1    | `scales` (device `bfloat16_t*`)         |
//! | 2    | `biases` (device `bfloat16_t*`)         |
//! | 3    | `out` (device `bfloat16_t*`)            |
//!
//! Grid: 1D, `(out.len() / pack_factor, 1, 1)` thread positions, where
//! `pack_factor = 8 / bits` (2 for bits=4). Each thread reads one
//! packed byte from `w`, looks up the matching `scale`/`bias` for its
//! group, and writes `pack_factor` dequantized values to `out`.
//!
//! Explained in the inference guide: [`docs/inference/weights-and-quantization.md`](../../../../docs/inference/weights-and-quantization.md).

use half::bf16;
use objc2_metal::{MTLComputeCommandEncoder, MTLSize};

use crate::buffer::Buffer;
use crate::commands::Commands;
use crate::error::{Error, Result};
use crate::library::KernelLibrary;

/// Bit-width this dispatch fn is wired for (matches Qwen2's q4 config).
pub const BITS: u32 = 4;

/// Group size this dispatch fn is wired for (matches Qwen2's
/// group-of-64 quantization).
pub const GROUP_SIZE: u32 = 64;

/// Dequantize an affine-quantized `bf16` weight (`group_size=64`,
/// `bits=4`) into a dense `bf16` tensor.
///
/// Shapes (caller-enforced):
/// - `w_q.len() == n_elements / 8` (uint32 elements; each holds 8 int4 values)
/// - `scales.len() == biases.len() == n_elements / group_size`
/// - `out.len() == n_elements`
///
/// `library` must be a [`KernelLibrary::quantized`]. Encodes one
/// dispatch into `commands`; the caller flushes via
/// [`Commands::commit_and_wait`].
pub fn affine_dequantize_bf16_gs64_b4(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    w_q: &Buffer<u32>,
    scales: &Buffer<bf16>,
    biases: &Buffer<bf16>,
    out: &mut Buffer<bf16>,
) -> Result<()> {
    let n_out = out.len();
    if n_out == 0 {
        return Ok(());
    }
    // pack_factor for bits=4 is 2 — each thread iterates over one uint8
    // (two int4 values). MLX's kernel reads one byte (`uint val =
    // w[offset]`) and unpacks `pack_factor` outputs from it.
    let pack_factor: usize = 8 / BITS as usize;
    if n_out % pack_factor != 0 {
        return Err(Error::InvalidArgument(format!(
            "affine_dequantize: out element count {n_out} not a multiple of pack_factor {pack_factor}",
        )));
    }
    if n_out % GROUP_SIZE as usize != 0 {
        return Err(Error::InvalidArgument(format!(
            "affine_dequantize: out element count {n_out} not a multiple of group_size {GROUP_SIZE}",
        )));
    }
    let groups = n_out / GROUP_SIZE as usize;
    if scales.len() != groups || biases.len() != groups {
        return Err(Error::InvalidArgument(format!(
            "affine_dequantize: scales/biases length mismatch (expected {groups}, got scales={}, biases={})",
            scales.len(),
            biases.len(),
        )));
    }
    // Each packed uint32 holds 8 int4 values; we dispatch one thread
    // per uint8 inside that (so 4 threads per uint32 since each uint8
    // = 2 nibbles = pack_factor=2 outputs). Total threads = n_out/pack_factor.
    let total_threads = n_out / pack_factor;
    let bytes_per_uint32: usize = 4;
    let expected_wq_uint32_count = total_threads / bytes_per_uint32;
    if w_q.len() != expected_wq_uint32_count {
        return Err(Error::InvalidArgument(format!(
            "affine_dequantize: w_q length {} != expected uint32 count {expected_wq_uint32_count}",
            w_q.len(),
        )));
    }

    let kernel_name = format!("affine_dequantize_bfloat16_t_gs_{GROUP_SIZE}_b_{BITS}");
    let pipeline = library.pipeline(&kernel_name)?;

    {
        let encoder = commands.encoder()?;
        encoder.setComputePipelineState(pipeline.metal_pipeline_state());
        // SAFETY: slot/dtype binding matches MLX's affine_dequantize MSL
        // signature; the `w` buffer is uint32 in our Rust types but is
        // bound as the same MTLBuffer the kernel reads as `uint8_t*` (a
        // reinterpret-as-bytes view — the buffer's byte length is the same
        // regardless of the Rust element type).
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(w_q.metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(scales.metal_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(biases.metal_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(out.metal_buffer()), 0, 3);
        }
    }

    // 1D grid: one thread per packed byte. Threadgroup width capped at
    // 1024 (Metal limit); we pick min(total_threads, 1024).
    let grid = MTLSize {
        width: total_threads,
        height: 1,
        depth: 1,
    };
    let threadgroup = MTLSize {
        width: total_threads.min(1024),
        height: 1,
        depth: 1,
    };
    commands.dispatch_threads(grid, threadgroup)
}
