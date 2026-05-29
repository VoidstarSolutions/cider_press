//! Loads and dispatches MLX's `copy.metal`.
//!
//! Loads MLX's `copy.metal` (vendored under `kernels-mlx/`, flattened by
//! `build.rs`), compiles it at runtime, and dispatches the simplest
//! vector-copy instantiation against 1D `f32` and `f16` inputs. Asserts
//! the output is bit-identical to the input — the kernel is identity.

#![cfg(target_os = "macos")]
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use cider_press_kernels::{Buffer, Device, KernelLibrary, kernels};
use half::f16;

const N: usize = 1024;

#[test]
fn copy_v_f32_identity() {
    let device = Device::system_default().expect("Metal device");
    let library = KernelLibrary::copy(&device).expect("compile copy.metal");

    let mut src = vec![0_f32; N];
    for (i, v) in src.iter_mut().enumerate() {
        *v = i as f32 * 0.5 - 17.0;
    }
    let src_buf: Buffer<f32> = device.upload(&src).expect("upload src");
    let mut dst_buf: Buffer<f32> = device.alloc_buffer(N).expect("alloc dst");

    let mut commands = device.commands().expect("commands");
    kernels::copy::copy_v_f32(&mut commands, &library, &src_buf, &mut dst_buf).expect("dispatch");
    commands.commit_and_wait().expect("commit");

    // SAFETY: commit_and_wait blocked until the GPU finished writing dst_buf.
    let dst = unsafe { dst_buf.as_mut_slice() };
    for (i, (&a, &b)) in src.iter().zip(dst.iter()).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "f32 mismatch at index {i}");
    }
}

#[test]
fn copy_v_f16_identity() {
    let device = Device::system_default().expect("Metal device");
    let library = KernelLibrary::copy(&device).expect("compile copy.metal");

    let mut src = vec![f16::ZERO; N];
    for (i, v) in src.iter_mut().enumerate() {
        *v = f16::from_f32(i as f32 * 0.25 - 8.0);
    }
    let src_buf: Buffer<f16> = device.upload(&src).expect("upload src");
    let mut dst_buf: Buffer<f16> = device.alloc_buffer(N).expect("alloc dst");

    let mut commands = device.commands().expect("commands");
    kernels::copy::copy_v_f16(&mut commands, &library, &src_buf, &mut dst_buf).expect("dispatch");
    commands.commit_and_wait().expect("commit");

    // SAFETY: commit_and_wait blocked until the GPU finished writing dst_buf.
    let dst = unsafe { dst_buf.as_mut_slice() };
    for (i, (&a, &b)) in src.iter().zip(dst.iter()).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "f16 mismatch at index {i}");
    }
}
