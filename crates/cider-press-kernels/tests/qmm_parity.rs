//! Parity test for `kernels::qmm::affine_qmm_t_bf16`.
//!
//! Dispatches MLX's `affine_qmm_t` (bf16 / int4 / `group_size=64`,
//! `transpose=true`, aligned, unbatched) via [`kernels::qmm::affine_qmm_t_bf16`]
//! and compares against MLX's own `mx.quantized_matmul` reference.
//!
//! Shape: `[M=8, K=896, N=896]` — the Qwen2.5-0.5B `q_proj` prefill shape
//! at T=8. The fixture is generated once by `scripts/gen_qmm_fixture.py`.

#![cfg(target_os = "macos")]
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use std::path::PathBuf;

use approx::assert_relative_eq;
use cider_press_kernels::{Buffer, Device, KernelLibrary, kernels};
use cider_press_test_utils::{read_bf16, read_u32};
use half::bf16;
use safetensors::SafeTensors;

// Test dims — must match scripts/gen_qmm_fixture.py.
const M: usize = 8;
const K: usize = 896;
const N: usize = 896;
const GROUP_SIZE: u32 = 64;
const BITS: u32 = 4;

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("qmm_qwen2.safetensors")
}

#[test]
fn parity_affine_qmm_t_bf16_gs64_b4() {
    let path = fixture_path();
    if !path.exists() {
        eprintln!(
            "skipping: fixture missing at {}; run `uv run scripts/gen_qmm_fixture.py`",
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

    assert_eq!(w_q.len(), N * K * (BITS as usize) / 32);
    assert_eq!(scales.len(), N * K / (GROUP_SIZE as usize));
    assert_eq!(biases.len(), N * K / (GROUP_SIZE as usize));
    assert_eq!(x.len(), M * K);
    assert_eq!(y_ref.len(), M * N);

    let device = Device::system_default().expect("Metal device");
    let library = KernelLibrary::quantized(&device).expect("compile quantized.metal");

    let w_buf: Buffer<u32> = device.upload(&w_q).expect("upload w_q");
    let s_buf: Buffer<bf16> = device.upload(&scales).expect("upload scales");
    let b_buf: Buffer<bf16> = device.upload(&biases).expect("upload biases");
    let x_buf: Buffer<bf16> = device.upload(&x).expect("upload x");
    let mut y_buf: Buffer<bf16> = device.alloc_buffer(M * N).expect("alloc y");

    let mut cmds = device.commands().expect("commands");
    kernels::qmm::affine_qmm_t_bf16(
        &mut cmds, &library, &w_buf, &s_buf, &b_buf, &x_buf, &mut y_buf, M, N, K, GROUP_SIZE, BITS,
    )
    .expect("dispatch");
    cmds.commit_and_wait().expect("commit");

    // SAFETY: commit_and_wait blocked until the GPU finished writing y_buf.
    let y_out: Vec<bf16> = unsafe { y_buf.as_mut_slice() }.to_vec();

    // Per-element parity at bf16 precision. Tolerances match the qmv parity test:
    // the kernel is bit-exact on identical hardware, but 1–2 ULP drift
    // occurs across Apple Silicon generations (M1 CI vs M3/M4 dev machines)
    // because bf16 reduction order can differ between GPU families.
    for (a, e) in y_out.iter().zip(y_ref.iter()) {
        let af = a.to_f32();
        let ef = e.to_f32();
        assert_relative_eq!(af, ef, max_relative = 1e-2, epsilon = 5e-2);
    }
}
