//! Parity tests for the `copy_g_*` (strided same-dtype) dispatch
//! ported from MLX's `copy_gg*` template family.
//!
//! Each test builds a backing buffer, picks a non-trivial src view
//! (transposed / sliced / broadcast / high-rank), runs the kernel into
//! a contiguous dst, and compares against a host-side strided gather.

#![cfg(target_os = "macos")]
#![allow(
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::identity_op,
    clippy::unnecessary_cast
)]

use cider_press_kernels::{Buffer, Device, KernelLibrary, kernels};
use half::bf16;

/// Logical → element-offset (host-side reference). Mirrors what
/// `copy_gg` does on the GPU: sum of `index[d] * stride[d]` over axes.
fn gather_host<T: Copy>(
    src: &[T],
    src_element_offset: usize,
    shape: &[i32],
    strides: &[i64],
) -> Vec<T> {
    let total: usize = shape.iter().map(|&d| d as usize).product();
    let mut out = Vec::with_capacity(total);
    let rank = shape.len();
    let mut idx = vec![0i64; rank];
    for _ in 0..total {
        let mut elem: i64 = 0;
        for d in 0..rank {
            elem += idx[d] * strides[d];
        }
        let phys = (src_element_offset as i64 + elem) as usize;
        out.push(src[phys]);
        // bump row-major
        for d in (0..rank).rev() {
            idx[d] += 1;
            if idx[d] < shape[d] as i64 {
                break;
            }
            idx[d] = 0;
        }
    }
    out
}

fn contiguous_strides(shape: &[i32]) -> Vec<i64> {
    let mut s = vec![1i64; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        s[d] = s[d + 1] * shape[d + 1] as i64;
    }
    s
}

#[test]
fn copy_g_f32_rank2_transpose() {
    let device = Device::system_default().expect("Metal device");
    let library = KernelLibrary::copy(&device).expect("copy.metal");

    // Source is a 4x3 row-major matrix; view it transposed (3x4).
    let rows = 4i32;
    let cols = 3i32;
    let src: Vec<f32> = (0..(rows * cols)).map(|i| i as f32 * 0.5 - 1.0).collect();
    let src_buf: Buffer<f32> = device.upload(&src).expect("upload");

    let view_shape = [cols, rows];
    // Original strides were [cols, 1]. Transpose swaps them.
    let src_strides: [i64; 2] = [1, cols as i64];
    let dst_strides = contiguous_strides(&view_shape);
    let dst_elems: usize = view_shape.iter().map(|&d| d as usize).product();
    let mut dst_buf: Buffer<f32> = device.alloc_buffer(dst_elems).expect("alloc dst");

    let mut commands = device.commands().expect("commands");
    kernels::copy::copy_g_f32(
        &mut commands,
        &library,
        &src_buf,
        0,
        &src_strides,
        &mut dst_buf,
        0,
        &dst_strides,
        &view_shape,
    )
    .expect("dispatch");
    commands.commit_and_wait().expect("commit");

    let expected = gather_host(&src, 0, &view_shape, &src_strides);
    let actual = unsafe { dst_buf.as_mut_slice() };
    assert_eq!(actual, expected.as_slice());
}

#[test]
fn copy_g_bf16_rank3_inner_slice() {
    let device = Device::system_default().expect("Metal device");
    let library = KernelLibrary::copy(&device).expect("copy.metal");

    // 2x4x3 source; slice axis-1 [1..3] → view shape 2x2x3.
    let src: Vec<bf16> = (0..(2 * 4 * 3))
        .map(|i| bf16::from_f32(i as f32 * 0.25 - 5.0))
        .collect();
    let src_buf: Buffer<bf16> = device.upload(&src).expect("upload");

    let view_shape = [2i32, 2, 3];
    let src_strides: [i64; 3] = [12, 3, 1]; // unchanged across the slice
    let dst_strides = contiguous_strides(&view_shape);
    let src_byte_offset = 1 /* axis-1 start */ * 3 * std::mem::size_of::<bf16>();
    let dst_elems: usize = view_shape.iter().map(|&d| d as usize).product();
    let mut dst_buf: Buffer<bf16> = device.alloc_buffer(dst_elems).expect("alloc dst");

    let mut commands = device.commands().expect("commands");
    kernels::copy::copy_g_bf16(
        &mut commands,
        &library,
        &src_buf,
        src_byte_offset,
        &src_strides,
        &mut dst_buf,
        0,
        &dst_strides,
        &view_shape,
    )
    .expect("dispatch");
    commands.commit_and_wait().expect("commit");

    let expected = gather_host(
        &src,
        src_byte_offset / std::mem::size_of::<bf16>(),
        &view_shape,
        &src_strides,
    );
    let actual = unsafe { dst_buf.as_mut_slice() };
    for (i, (a, b)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "mismatch at {i}");
    }
}

#[test]
fn copy_g_f32_broadcast_zero_stride() {
    let device = Device::system_default().expect("Metal device");
    let library = KernelLibrary::copy(&device).expect("copy.metal");

    // Source is a 1x4 vector; broadcast to 3x4 via stride[0] = 0.
    let src: Vec<f32> = vec![1.5, -2.0, 3.25, 0.0];
    let src_buf: Buffer<f32> = device.upload(&src).expect("upload");

    let view_shape = [3i32, 4];
    let src_strides: [i64; 2] = [0, 1];
    let dst_strides = contiguous_strides(&view_shape);
    let dst_elems: usize = view_shape.iter().map(|&d| d as usize).product();
    let mut dst_buf: Buffer<f32> = device.alloc_buffer(dst_elems).expect("alloc dst");

    let mut commands = device.commands().expect("commands");
    kernels::copy::copy_g_f32(
        &mut commands,
        &library,
        &src_buf,
        0,
        &src_strides,
        &mut dst_buf,
        0,
        &dst_strides,
        &view_shape,
    )
    .expect("dispatch");
    commands.commit_and_wait().expect("commit");

    let expected = gather_host(&src, 0, &view_shape, &src_strides);
    let actual = unsafe { dst_buf.as_mut_slice() };
    assert_eq!(actual, expected.as_slice());
    // Spot check: every row equals the input vector.
    for row in actual.chunks(4) {
        assert_eq!(row, src.as_slice());
    }
}

#[test]
fn copy_g_i32_rank4_generic() {
    let device = Device::system_default().expect("Metal device");
    let library = KernelLibrary::copy(&device).expect("copy.metal");

    // Rank 4 forces the generic `ggn2_copy` path (ndim > 3).
    // Source layout: 2x3x4x5 row-major. View it transposed on the
    // outer two axes → shape 3x2x4x5 with strides [20, 60, 5, 1].
    let src: Vec<i32> = (0..(2 * 3 * 4 * 5)).map(|i| i as i32 * 7 - 100).collect();
    let src_buf: Buffer<i32> = device.upload(&src).expect("upload");

    let view_shape = [3i32, 2, 4, 5];
    let src_strides: [i64; 4] = [20, 60, 5, 1];
    let dst_strides = contiguous_strides(&view_shape);
    let dst_elems: usize = view_shape.iter().map(|&d| d as usize).product();
    let mut dst_buf: Buffer<i32> = device.alloc_buffer(dst_elems).expect("alloc dst");

    let mut commands = device.commands().expect("commands");
    kernels::copy::copy_g_i32(
        &mut commands,
        &library,
        &src_buf,
        0,
        &src_strides,
        &mut dst_buf,
        0,
        &dst_strides,
        &view_shape,
    )
    .expect("dispatch");
    commands.commit_and_wait().expect("commit");

    let expected = gather_host(&src, 0, &view_shape, &src_strides);
    let actual = unsafe { dst_buf.as_mut_slice() };
    assert_eq!(actual, expected.as_slice());
}

#[test]
fn copy_g_u32_rank1_byte_offset() {
    let device = Device::system_default().expect("Metal device");
    let library = KernelLibrary::copy(&device).expect("copy.metal");

    // Rank 1 + non-zero src byte offset → gg1_copy with src bound at offset.
    let src: Vec<u32> = (0..16)
        .map(|i| (i as u32).wrapping_mul(0x9E37_79B1))
        .collect();
    let src_buf: Buffer<u32> = device.upload(&src).expect("upload");

    let view_shape = [8i32];
    let src_strides: [i64; 1] = [1];
    let dst_strides: [i64; 1] = [1];
    let src_byte_offset = 4 * std::mem::size_of::<u32>();
    let mut dst_buf: Buffer<u32> = device.alloc_buffer(8).expect("alloc dst");

    let mut commands = device.commands().expect("commands");
    kernels::copy::copy_g_u32(
        &mut commands,
        &library,
        &src_buf,
        src_byte_offset,
        &src_strides,
        &mut dst_buf,
        0,
        &dst_strides,
        &view_shape,
    )
    .expect("dispatch");
    commands.commit_and_wait().expect("commit");

    let actual = unsafe { dst_buf.as_mut_slice() };
    assert_eq!(actual, &src[4..12]);
}
