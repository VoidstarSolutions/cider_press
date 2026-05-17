//! Stage 4 of the development spike — see `CLAUDE.md`.
//!
//! Dispatches MLX's `affine_qmv_fast` (bf16 / int4 / `group_size=64`) via
//! [`kernels::qmv::affine_qmv_bf16`] and compares against MLX's own
//! `mx.quantized_matmul` reference at bf16 tolerance.
//!
//! With the typed dispatch fn in place, this test is the regression
//! canary for both:
//!   - cider-press kernel layer drift (parameters, geometry), and
//!   - upstream MLX kernel-source drift (the mlx-sync workflow gates
//!     PRs on this test).

#![cfg(target_os = "macos")]
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use std::path::PathBuf;

use cider_press_kernels::{Buffer, Device, KernelLibrary, kernels};
use half::bf16;
use safetensors::SafeTensors;

// Test dims — must match scripts/gen_stage4_fixtures.py.
const K: usize = 512;
const N: usize = 512;
const GROUP_SIZE: u32 = 64;
const BITS: u32 = 4;

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("stage4_qmv.safetensors")
}

#[test]
fn parity_affine_qmv_fast_bf16_gs64_b4() {
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

    assert_eq!(x.len(), K);
    assert_eq!(y_ref.len(), N);

    let device = Device::system_default().expect("Metal device");
    let library = KernelLibrary::quantized(&device).expect("compile quantized.metal");

    let w_buf: Buffer<u32> = device.upload(&w_q).expect("upload w_q");
    let s_buf: Buffer<bf16> = device.upload(&scales).expect("upload scales");
    let b_buf: Buffer<bf16> = device.upload(&biases).expect("upload biases");
    let x_buf: Buffer<bf16> = device.upload(&x).expect("upload x");
    let mut y_buf: Buffer<bf16> = device.alloc_buffer(N).expect("alloc y");

    let mut cmds = device.commands().expect("commands");
    kernels::qmv::affine_qmv_bf16(
        &mut cmds, &library, &w_buf, &s_buf, &b_buf, &x_buf, &mut y_buf, GROUP_SIZE, BITS,
    )
    .expect("dispatch");
    cmds.commit_and_wait().expect("commit");

    // SAFETY: commit_and_wait blocked until the GPU finished writing y_buf.
    let y_out: Vec<bf16> = unsafe { y_buf.as_mut_slice() }.to_vec();

    // bf16 has ~3 decimal digits of mantissa; ~1e-2 relative is the right
    // bound. Near-zero values use absolute tolerance instead since
    // relative error is meaningless there.
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
