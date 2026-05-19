//! Identity-copy coverage for the dtypes wired beyond the original
//! Stage-2 spike: `bf16`, `i32`, `u32`. (The `f32` / `f16` paths live
//! in `stage2_copy.rs` and stay there as the spike-era canary.)
//!
//! Each test uploads a sentinel buffer, runs `v_copy<dtype>`, and
//! asserts byte-for-byte identity.

#![cfg(target_os = "macos")]
#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss
)]

use cider_press_kernels::{Buffer, Device, KernelLibrary, kernels};
use half::bf16;

const N: usize = 1024;

#[test]
fn copy_v_bf16_identity() {
    let device = Device::system_default().expect("Metal device");
    let library = KernelLibrary::copy(&device).expect("compile copy.metal");

    let mut src = vec![bf16::ZERO; N];
    for (i, v) in src.iter_mut().enumerate() {
        // Bf16 has limited precision; pick values that survive round-trip.
        *v = bf16::from_f32(i as f32 * 0.5 - 17.0);
    }
    let src_buf: Buffer<bf16> = device.upload(&src).expect("upload src");
    let mut dst_buf: Buffer<bf16> = device.alloc_buffer(N).expect("alloc dst");

    let mut commands = device.commands().expect("commands");
    kernels::copy::copy_v_bf16(&mut commands, &library, &src_buf, &mut dst_buf).expect("dispatch");
    commands.commit_and_wait().expect("commit");

    // SAFETY: commit_and_wait blocked until the GPU finished writing dst_buf.
    let dst = unsafe { dst_buf.as_mut_slice() };
    for (i, (&a, &b)) in src.iter().zip(dst.iter()).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "bf16 mismatch at index {i}");
    }
}

#[test]
fn copy_v_i32_identity() {
    let device = Device::system_default().expect("Metal device");
    let library = KernelLibrary::copy(&device).expect("compile copy.metal");

    let src: Vec<i32> = (0..N).map(|i| (i as i32) * 7 - 3_000).collect();
    let src_buf: Buffer<i32> = device.upload(&src).expect("upload src");
    let mut dst_buf: Buffer<i32> = device.alloc_buffer(N).expect("alloc dst");

    let mut commands = device.commands().expect("commands");
    kernels::copy::copy_v_i32(&mut commands, &library, &src_buf, &mut dst_buf).expect("dispatch");
    commands.commit_and_wait().expect("commit");

    let dst = unsafe { dst_buf.as_mut_slice() };
    assert_eq!(dst, src.as_slice());
}

#[test]
fn copy_v_u32_identity() {
    let device = Device::system_default().expect("Metal device");
    let library = KernelLibrary::copy(&device).expect("compile copy.metal");

    let src: Vec<u32> = (0..N)
        .map(|i| (i as u32).wrapping_mul(0x9E37_79B1))
        .collect();
    let src_buf: Buffer<u32> = device.upload(&src).expect("upload src");
    let mut dst_buf: Buffer<u32> = device.alloc_buffer(N).expect("alloc dst");

    let mut commands = device.commands().expect("commands");
    kernels::copy::copy_v_u32(&mut commands, &library, &src_buf, &mut dst_buf).expect("dispatch");
    commands.commit_and_wait().expect("commit");

    let dst = unsafe { dst_buf.as_mut_slice() };
    assert_eq!(dst, src.as_slice());
}
