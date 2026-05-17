//! Stage 4 of the development spike — see `CLAUDE.md`.
//!
//! Dispatches MLX's `affine_qmv_fast_bfloat16_t_gs_64_b_4_batch_0` kernel
//! (bf16 input/output, int4 weights packed into `uint32`, `group_size`=64,
//! per-group scale + bias) and compares the output against MLX's own
//! `mx.quantized_matmul` reference at bf16 tolerance.
//!
//! The binding order, buffer layout, and dispatch geometry here are
//! transcribed from `mlx/backend/metal/quantized.cpp::qmv()`. Any drift
//! from that file will produce silent garbage rather than a compile
//! error — so when MLX bumps the kernel signature, the fixture file
//! becomes the canary.

#![cfg(target_os = "macos")]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::too_many_lines
)]

use std::ffi::c_void;
use std::path::PathBuf;
use std::ptr::NonNull;

use half::bf16;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary, MTLResourceOptions, MTLSize,
};
use safetensors::SafeTensors;

#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {}

const QUANTIZED_SOURCE: &str = include_str!(concat!(env!("OUT_DIR"), "/quantized_inlined.metal"));

// Test dims — must match scripts/gen_stage4_fixtures.py.
const K: i32 = 512;
const N: i32 = 512;
const GROUP_SIZE: usize = 64;
const BITS: usize = 4;
// Kernel-side tile sizes from mlx/backend/metal/quantized.cpp::qmv():
//   bn = 8, bk = 32
//   group_dims = (bk, 2, 1) = (32, 2, 1)
//   grid_dims  = (M, (N + bn - 1) / bn, B) = (1, N/8, 1) in threadgroups
const BN: i32 = 8;
const BK: i32 = 32;

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("stage4_qmv.safetensors")
}

fn mlx_source_unavailable() -> bool {
    QUANTIZED_SOURCE
        .trim()
        .starts_with("// MLX checkout not found")
}

#[test]
fn parity_affine_qmv_fast_bf16_gs64_b4() {
    if mlx_source_unavailable() {
        eprintln!("skipping: MLX checkout not found at build time");
        return;
    }
    let path = fixture_path();
    if !path.exists() {
        eprintln!(
            "skipping: fixture missing at {}; run `uv run scripts/gen_stage4_fixtures.py`",
            path.display(),
        );
        return;
    }

    let bytes = std::fs::read(&path).expect("read fixture");
    let st = SafeTensors::deserialize(&bytes).expect("parse safetensors");

    let w_q = read_u32(&st, "w_q");
    let scales = read_bf16(&st, "scales");
    let biases = read_bf16(&st, "biases");
    let x = read_bf16(&st, "x");
    let y_ref = read_bf16(&st, "y_ref");

    // Sanity-check shapes against the constants.
    assert_eq!(w_q.len(), (N as usize) * (K as usize) * BITS / 32);
    assert_eq!(scales.len(), (N as usize) * (K as usize) / GROUP_SIZE);
    assert_eq!(biases.len(), scales.len());
    assert_eq!(x.len(), K as usize);
    assert_eq!(y_ref.len(), N as usize);

    let device = MTLCreateSystemDefaultDevice().expect("no Metal device");
    let queue = device.newCommandQueue().expect("command queue");
    let source = NSString::from_str(QUANTIZED_SOURCE);
    let library = device
        .newLibraryWithSource_options_error(&source, None)
        .unwrap_or_else(|err| panic!("quantized.metal compile: {err}"));

    // Name comes from the `instantiate_quantized_batched` macro in
    // mlx/backend/metal/kernels/quantized.metal — the type token is the
    // MSL type spelling (`bfloat16_t`), not MLX's `Dtype::bfloat16` name.
    let kernel_name = "affine_qmv_fast_bfloat16_t_gs_64_b_4_batch_0";
    let ns_name = NSString::from_str(kernel_name);
    let function = library
        .newFunctionWithName(&ns_name)
        .unwrap_or_else(|| panic!("kernel {kernel_name} not in library"));
    let pipeline = device
        .newComputePipelineStateWithFunction_error(&function)
        .unwrap_or_else(|err| panic!("pipeline {kernel_name}: {err}"));

    // Device buffers, host-writable (unified memory).
    let w_buf = make_buffer(&device, &w_q);
    let s_buf = make_buffer(&device, &scales);
    let b_buf = make_buffer(&device, &biases);
    let x_buf = make_buffer(&device, &x);
    let y_buf = device
        .newBufferWithLength_options(
            std::mem::size_of_val(y_ref.as_slice()),
            MTLResourceOptions::StorageModeShared,
        )
        .expect("y buffer");

    let cmd = queue.commandBuffer().expect("commandBuffer");
    let encoder = cmd.computeCommandEncoder().expect("encoder");
    encoder.setComputePipelineState(&pipeline);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(&w_buf), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(&s_buf), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(&b_buf), 0, 2);
        encoder.setBuffer_offset_atIndex(Some(&x_buf), 0, 3);
        encoder.setBuffer_offset_atIndex(Some(&y_buf), 0, 4);
        // K (in_vec_size) and N (out_vec_size) are passed as int32 constants.
        let k_ptr: NonNull<c_void> = NonNull::from(&K).cast();
        let n_ptr: NonNull<c_void> = NonNull::from(&N).cast();
        encoder.setBytes_length_atIndex(k_ptr, std::mem::size_of::<i32>(), 5);
        encoder.setBytes_length_atIndex(n_ptr, std::mem::size_of::<i32>(), 6);
    }

    let grid = MTLSize {
        width: 1,
        height: ((N + BN - 1) / BN) as usize,
        depth: 1,
    };
    let threadgroup = MTLSize {
        width: BK as usize,
        height: 2,
        depth: 1,
    };
    encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, threadgroup);
    encoder.endEncoding();

    cmd.commit();
    cmd.waitUntilCompleted();

    // Read y from shared memory.
    let len = N as usize;
    let mut y_out = vec![bf16::ZERO; len];
    unsafe {
        let ptr: NonNull<bf16> = y_buf.contents().cast();
        y_out.copy_from_slice(std::slice::from_raw_parts(ptr.as_ptr(), len));
    }

    // Compare against MLX reference. bf16 has ~3 decimal digits of mantissa,
    // so relative tolerance ~1e-2 is the right bound; fall back to absolute
    // tolerance for values near zero where relative error is meaningless.
    let mut max_rel: f32 = 0.0;
    let mut max_abs: f32 = 0.0;
    let mut worst_idx = 0usize;
    for (i, (a, e)) in y_out.iter().zip(y_ref.iter()).enumerate() {
        let af = a.to_f32();
        let ef = e.to_f32();
        let abs = (af - ef).abs();
        let rel = if ef.abs() > 1e-3 { abs / ef.abs() } else { 0.0 };
        if rel > max_rel {
            max_rel = rel;
            worst_idx = i;
        }
        max_abs = max_abs.max(abs);
    }

    eprintln!(
        "qmv parity: max_relative={max_rel:.3e}, max_absolute={max_abs:.3e}, \
         worst_idx={worst_idx} (got {} vs ref {})",
        y_out[worst_idx].to_f32(),
        y_ref[worst_idx].to_f32(),
    );

    assert!(
        max_rel < 1e-2 && max_abs < 1e-1,
        "parity exceeded bf16 tolerance: max_rel={max_rel:.3e}, max_abs={max_abs:.3e}",
    );
}

fn make_buffer<T: Copy>(
    device: &ProtocolObject<dyn MTLDevice>,
    data: &[T],
) -> Retained<ProtocolObject<dyn MTLBuffer>> {
    let bytes = std::mem::size_of_val(data);
    let buf = device
        .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
        .expect("allocate buffer");
    unsafe {
        let ptr: NonNull<T> = buf.contents().cast();
        std::slice::from_raw_parts_mut(ptr.as_ptr(), data.len()).copy_from_slice(data);
    }
    buf
}

fn read_u32(st: &SafeTensors, name: &str) -> Vec<u32> {
    let view = st
        .tensor(name)
        .unwrap_or_else(|e| panic!("tensor {name}: {e}"));
    assert_eq!(view.dtype(), safetensors::Dtype::U32);
    let bytes = view.data();
    assert!(bytes.len() % 4 == 0);
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn read_bf16(st: &SafeTensors, name: &str) -> Vec<bf16> {
    let view = st
        .tensor(name)
        .unwrap_or_else(|e| panic!("tensor {name}: {e}"));
    assert_eq!(view.dtype(), safetensors::Dtype::BF16);
    let bytes = view.data();
    assert!(bytes.len() % 2 == 0);
    bytes
        .chunks_exact(2)
        .map(|c| bf16::from_le_bytes([c[0], c[1]]))
        .collect()
}
