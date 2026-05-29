//! qmv kernel parity vs MLX.
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

use approx::assert_relative_eq;
use cider_press_kernels::{Buffer, Device, KernelLibrary, kernels};
use cider_press_test_utils::{read_bf16, read_u32};
use half::bf16;
use safetensors::SafeTensors;

// Test dims — must match scripts/gen_qmv_fixture.py.
const K: usize = 512;
const N: usize = 512;
const GROUP_SIZE: u32 = 64;
const BITS: u32 = 4;

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("qmv.safetensors")
}

#[test]
fn parity_affine_qmv_fast_bf16_gs64_b4() {
    let path = fixture_path();
    if !path.exists() {
        eprintln!(
            "skipping: fixture missing at {}; run `uv run scripts/gen_qmv_fixture.py`",
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

    // Per-element parity at bf16 precision. The `epsilon` arm is the
    // absolute-error fallback that approx::relative_eq! kicks into for
    // near-zero values, where relative error becomes meaningless;
    // `0.05` is ~3 bf16 ULPs at scale ~0.5, comfortably above the
    // per-element drift we see when the qmv reduction runs on
    // different Apple Silicon GPU generations (CI runner is M1; some
    // dev machines are M3/M4 with native bfloat). The kernel itself
    // is bit-exact on identical hardware; this tolerance just absorbs
    // hardware-family differences in non-IEEE-strict bf16 ops.
    for (a, e) in y_out.iter().zip(y_ref.iter()) {
        let af = a.to_f32();
        let ef = e.to_f32();
        assert_relative_eq!(af, ef, max_relative = 1e-2, epsilon = 5e-2);
    }
}
