//! Copy-kernel MLX parity test.
//!
//! Loads pre-generated input/expected-output safetensors (produced by
//! `scripts/gen_copy_fixture.py`), runs the MLX `v_copy` kernel via
//! cider-press, and compares against the MLX reference with
//! dtype-appropriate tolerance.
//!
//! The harness exercised here is the same rig the qmv parity test uses
//! to validate `qmv` against MLX's `quantized_matmul`. For copy the
//! reference equals the input, so this test mostly smoke-tests the rig
//! itself.

#![cfg(target_os = "macos")]
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use std::path::PathBuf;

use approx::relative_eq;
use cider_press_kernels::{Buffer, Device, KernelLibrary, kernels};
use half::f16;
use safetensors::SafeTensors;

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("copy.safetensors")
}

#[test]
fn parity_v_copy_f32() {
    let path = fixture_path();
    if !path.exists() {
        eprintln!(
            "skipping: fixture missing at {}; run `uv run scripts/gen_copy_fixture.py`",
            path.display(),
        );
        return;
    }

    let bytes = std::fs::read(&path).expect("read fixture");
    let st = SafeTensors::deserialize(&bytes).expect("parse safetensors");
    let src: Vec<f32> = read_tensor_f32(&st, "src_f32");
    let expected: Vec<f32> = read_tensor_f32(&st, "dst_f32");
    assert_eq!(src.len(), expected.len());

    let device = Device::system_default().expect("Metal device");
    let library = KernelLibrary::copy(&device).expect("compile copy.metal");

    let src_buf: Buffer<f32> = device.upload(&src).expect("upload src");
    let mut dst_buf: Buffer<f32> = device.alloc_buffer(src.len()).expect("alloc dst");

    let mut commands = device.commands().expect("commands");
    kernels::copy::copy_v_f32(&mut commands, &library, &src_buf, &mut dst_buf).expect("dispatch");
    commands.commit_and_wait().expect("commit");

    // SAFETY: commit_and_wait blocked until the GPU finished writing dst_buf.
    let actual = unsafe { dst_buf.as_mut_slice() };
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            relative_eq!(*a, *e, max_relative = 1e-6, epsilon = 1e-7),
            "f32 mismatch at index {i}: actual={a} expected={e}",
        );
    }
}

#[test]
fn parity_v_copy_f16() {
    let path = fixture_path();
    if !path.exists() {
        eprintln!(
            "skipping: fixture missing at {}; run `uv run scripts/gen_copy_fixture.py`",
            path.display(),
        );
        return;
    }

    let bytes = std::fs::read(&path).expect("read fixture");
    let st = SafeTensors::deserialize(&bytes).expect("parse safetensors");
    let src: Vec<f16> = read_tensor_f16(&st, "src_f16");
    let expected: Vec<f16> = read_tensor_f16(&st, "dst_f16");
    assert_eq!(src.len(), expected.len());

    let device = Device::system_default().expect("Metal device");
    let library = KernelLibrary::copy(&device).expect("compile copy.metal");

    let src_buf: Buffer<f16> = device.upload(&src).expect("upload src");
    let mut dst_buf: Buffer<f16> = device.alloc_buffer(src.len()).expect("alloc dst");

    let mut commands = device.commands().expect("commands");
    kernels::copy::copy_v_f16(&mut commands, &library, &src_buf, &mut dst_buf).expect("dispatch");
    commands.commit_and_wait().expect("commit");

    // SAFETY: commit_and_wait blocked until the GPU finished writing dst_buf.
    let actual = unsafe { dst_buf.as_mut_slice() };
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let a32 = a.to_f32();
        let e32 = e.to_f32();
        assert!(
            relative_eq!(a32, e32, max_relative = 1e-3, epsilon = 1e-4),
            "f16 mismatch at index {i}: actual={a32} expected={e32}",
        );
    }
}

fn read_tensor_f32(st: &SafeTensors, name: &str) -> Vec<f32> {
    let view = st
        .tensor(name)
        .unwrap_or_else(|e| panic!("tensor {name}: {e}"));
    assert_eq!(view.dtype(), safetensors::Dtype::F32);
    let bytes = view.data();
    assert!(bytes.len() % 4 == 0);
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn read_tensor_f16(st: &SafeTensors, name: &str) -> Vec<f16> {
    let view = st
        .tensor(name)
        .unwrap_or_else(|e| panic!("tensor {name}: {e}"));
    assert_eq!(view.dtype(), safetensors::Dtype::F16);
    let bytes = view.data();
    assert!(bytes.len() % 2 == 0);
    bytes
        .chunks_exact(2)
        .map(|c| f16::from_le_bytes([c[0], c[1]]))
        .collect()
}
