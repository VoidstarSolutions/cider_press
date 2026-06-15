//! Parity test for the non-aligned `affine_qmm_t` variant (`alN_false`).
//!
//! Dispatches MLX's `affine_qmm_t` at an N that is **not** a multiple of the
//! `BN=32` output tile (`N=48`) via [`kernels::qmm::affine_qmm_t_bf16`] and
//! compares against MLX's own `mx.quantized_matmul`. This is the M>1 prefill
//! shape the Qwen3.6-27B GDN `in_proj_a`/`in_proj_b` projections hit
//! (output N=48), which the aligned-only path rejected.
//!
//! The fixture is materialised on demand from MLX via `dump_mlx_op qmm`.

#![cfg(target_os = "macos")]
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use cider_press_kernels::{Buffer, Device, KernelLibrary, kernels};
use cider_press_test_utils::{dump_mlx_op, read_bf16, read_u32, tempdir};
use half::bf16;
use safetensors::SafeTensors;

// N=48 is not a multiple of BN=32, so this exercises the `alN_false`
// (bounds-checked N tail) variant. K is a multiple of group_size=64.
const M: usize = 4;
const K: usize = 64;
const N: usize = 48;
const GROUP_SIZE: u32 = 64;
const BITS: u32 = 4;

#[test]
fn parity_affine_qmm_t_bf16_unaligned_n48() {
    let tmp = tempdir("kernels-qmm-unaligned-parity");
    let path = tmp.join("qmm_n48.safetensors");
    let m_s = M.to_string();
    let n_s = N.to_string();
    let k_s = K.to_string();
    dump_mlx_op(
        &path,
        &[
            "qmm",
            "--m",
            &m_s,
            "--n",
            &n_s,
            "--k",
            &k_s,
            "--group-size",
            "64",
            "--bits",
            "4",
            "--dtype",
            "bf16",
        ],
    );

    let bytes = std::fs::read(&path).expect("read fixture");
    let st = SafeTensors::deserialize(&bytes).expect("parse safetensors");

    let w_q = read_u32(&st, "w_q");
    let scales = read_bf16(&st, "scales");
    let biases = read_bf16(&st, "biases");
    let x = read_bf16(&st, "x");
    let y_ref = read_bf16(&st, "y");

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
    .expect("dispatch non-aligned qmm");
    cmds.commit_and_wait().expect("commit");

    // SAFETY: commit_and_wait blocked until the GPU finished writing y_buf.
    let y_out: Vec<bf16> = unsafe { y_buf.as_mut_slice() }.to_vec();

    // The qmm is bit-exact on identical hardware; allow 1-2 bf16 ULP for
    // cross-generation reduction-order drift, matching the aligned parity test.
    for (a, e) in y_out.iter().zip(y_ref.iter()) {
        let af = a.to_f32();
        let ef = e.to_f32();
        approx::assert_relative_eq!(af, ef, max_relative = 1e-2, epsilon = 5e-2);
    }
}
