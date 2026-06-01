// Dispatch routine derived from MLX (https://github.com/ml-explore/mlx).
// Copyright © 2023-2024 Apple Inc. Released under the MIT license — see
// the full upstream notice at `kernels-mlx/COPYING`.

//! Argmax over the last (contiguous, flat) axis of a bf16 vector,
//! producing a `u32` index. Wraps MLX's `argmax_bfloat16`
//! (`arg_reduce.metal`). The decode hot path's next-token selection.

use half::bf16;
use objc2_metal::{MTLComputeCommandEncoder, MTLSize};
use std::ffi::c_void;
use std::ptr::NonNull;

use crate::buffer::Buffer;
use crate::commands::Commands;
use crate::error::{Error, Result};
use crate::library::KernelLibrary;

/// Argmax over a contiguous `[axis_size]` bf16 vector → `u32` index in
/// `dst[0]`. Single dispatch of MLX's `argmax_bfloat16`. `dst.len()`
/// must be 1 (scalar reduction of a flat axis).
pub fn argmax_bf16(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    src: &Buffer<bf16>,
    dst: &mut Buffer<u32>,
    axis_size: usize,
) -> Result<()> {
    if axis_size == 0 {
        return Err(Error::InvalidArgument(
            "argmax_bf16: axis_size must be positive".to_owned(),
        ));
    }
    if src.len() != axis_size {
        return Err(Error::InvalidArgument(format!(
            "argmax_bf16: src.len() ({}) != axis_size ({})",
            src.len(),
            axis_size
        )));
    }
    if dst.len() != 1 {
        return Err(Error::InvalidArgument(format!(
            "argmax_bf16: dst.len() must be 1, got {}",
            dst.len()
        )));
    }
    let pipeline = library.pipeline("argmax_bfloat16")?;
    let encoder = commands.encoder()?;
    encoder.setComputePipelineState(pipeline.metal_pipeline_state());

    // ndim == 0 (flat scalar reduction): shape/strides slots take a
    // single placeholder element. For the flat ndim==0 case the kernel
    // does not dereference shape/in_strides/out_strides, so single zero
    // placeholders suffice (matching MLX's set_bytes(0, 2/3/4)).
    let shape: [i32; 1] = [0];
    let in_strides: [i64; 1] = [0];
    let out_strides: [i64; 1] = [0];
    let ndim: usize = 0;
    let axis_stride: i64 = 1;

    // SAFETY: bind slots/dtypes match the `arg_reduce_general` MSL
    // signature (in@0, out u32@1, shape@2, in_strides@3, out_strides@4,
    // ndim@5, axis_stride@6, axis_size@7).
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(src.metal_buffer()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(dst.metal_buffer()), 0, 1);
        encoder.setBytes_length_atIndex(
            NonNull::from(&shape).cast::<c_void>(),
            std::mem::size_of_val(&shape),
            2,
        );
        encoder.setBytes_length_atIndex(
            NonNull::from(&in_strides).cast::<c_void>(),
            std::mem::size_of_val(&in_strides),
            3,
        );
        encoder.setBytes_length_atIndex(
            NonNull::from(&out_strides).cast::<c_void>(),
            std::mem::size_of_val(&out_strides),
            4,
        );
        encoder.setBytes_length_atIndex(
            NonNull::from(&ndim).cast::<c_void>(),
            std::mem::size_of::<usize>(),
            5,
        );
        encoder.setBytes_length_atIndex(
            NonNull::from(&axis_stride).cast::<c_void>(),
            std::mem::size_of::<i64>(),
            6,
        );
        encoder.setBytes_length_atIndex(
            NonNull::from(&axis_size).cast::<c_void>(),
            std::mem::size_of::<usize>(),
            7,
        );
    }

    // N_READS = 4 per thread, grid-strided over the axis; one
    // threadgroup. Round the thread count up to a simd multiple (32),
    // capped at the Metal-guaranteed 1024 (the kernel loops to cover
    // larger axes). Mirrors MLX primitives.cpp::ArgReduce::eval_gpu.
    let want = axis_size.div_ceil(4);
    let tg = want.min(1024).div_ceil(32) * 32;
    let grid = MTLSize { width: tg, height: 1, depth: 1 };
    let threadgroup = MTLSize { width: tg, height: 1, depth: 1 };
    encoder.dispatchThreads_threadsPerThreadgroup(grid, threadgroup);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Device, KernelLibrary};

    fn run_argmax(values: &[f32]) -> u32 {
        let device = Device::system_default().expect("device");
        let library = KernelLibrary::arg_reduce(&device).expect("arg_reduce lib");
        let src_host: Vec<bf16> = values.iter().map(|&x| bf16::from_f32(x)).collect();
        let src: Buffer<bf16> = device.upload(&src_host).expect("upload src");
        let mut dst: Buffer<u32> = device.alloc_buffer(1).expect("alloc dst");
        let mut cmds = device.commands().expect("commands");
        argmax_bf16(&mut cmds, &library, &src, &mut dst, src_host.len()).expect("dispatch");
        cmds.commit_and_wait().expect("commit");
        // SAFETY: dst is a [1] u32 buffer, materialized after wait.
        unsafe { dst.as_slice()[0] }
    }

    #[test]
    fn argmax_unique_max_returns_its_index() {
        assert_eq!(run_argmax(&[0.1, 0.5, 0.3, 0.9, 0.2]), 3);
    }

    #[test]
    fn argmax_large_vector() {
        // 4096-wide, max planted at 1234.
        let mut v = vec![0.0f32; 4096];
        v[1234] = 7.0;
        assert_eq!(run_argmax(&v), 1234);
    }

    #[test]
    fn argmax_tie_breaking_is_locked() {
        // Two equal maxima at indices 1 and 3. Lock the kernel's choice
        // as a regression guard (MLX's ArgMax keeps the lower index).
        assert_eq!(run_argmax(&[0.1, 0.9, 0.3, 0.9, 0.2]), 1);
    }
}
