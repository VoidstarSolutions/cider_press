//! Parity tests for the vendored `sdpa_vector` kernel against MLX's
//! `mx.fast.scaled_dot_product_attention`. macOS + live Metal device.
#![cfg(target_os = "macos")]

use cider_press_kernels::{Buffer, Device, FunctionConstant, KernelLibrary, kernels::sdpa};
use cider_press_test_utils::{dump_mlx_op, read_bf16, tempdir};
use half::bf16;
use safetensors::SafeTensors;

#[test]
fn sdpa_vector_library_compiles() {
    let device = Device::system_default().expect("Metal device");
    let lib = KernelLibrary::sdpa_vector(&device).expect("compile sdpa_vector");
    // The bf16 head-dim-64 single-pass kernel requires [[function_constant]]
    // specialization: calling pipeline() on a function-constant kernel
    // aborts at dispatch (missing-constant SIGABRT), so pipeline_specialized
    // is the only safe path. Indices 20-25 are bool; 26 is int for 2-pass only.
    // Bind the six bool slots to their simplest no-mask, no-causal values.
    let constants = [
        FunctionConstant::Bool {
            index: 20,
            value: false,
        }, // has_mask
        FunctionConstant::Bool {
            index: 21,
            value: false,
        }, // query_transposed
        FunctionConstant::Bool {
            index: 22,
            value: false,
        }, // do_causal
        FunctionConstant::Bool {
            index: 23,
            value: false,
        }, // bool_mask
        FunctionConstant::Bool {
            index: 24,
            value: false,
        }, // float_mask
        FunctionConstant::Bool {
            index: 25,
            value: false,
        }, // has_sinks
    ];
    lib.pipeline_specialized("sdpa_vector_bfloat16_t_64_64", &constants)
        .expect("sdpa_vector bf16 64/64 pipeline");
}

const H_Q: usize = 14;
const H_KV: usize = 2;
const T_CACHE: usize = 16;
const D: usize = 64;

#[test]
// The cast operands are compile-time consts (small, exact); the lints
// guard against runtime truncation that cannot occur here.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss
)]
fn sdpa_vector_bf16_matches_mlx() {
    let tmp = tempdir("sdpa-vector-parity");
    let fixture = tmp.join("sdpa.safetensors");
    dump_mlx_op(
        &fixture,
        &[
            "sdpa",
            "--batch",
            "1",
            "--h-q",
            &H_Q.to_string(),
            "--h-kv",
            &H_KV.to_string(),
            "--seq-q",
            "1",
            "--seq-kv",
            &T_CACHE.to_string(),
            "--head-dim",
            &D.to_string(),
        ],
    );
    let bytes = std::fs::read(&fixture).expect("read fixture");
    let st = SafeTensors::deserialize(&bytes).expect("parse safetensors");
    let q = read_bf16(&st, "q"); // [1,H_Q,1,D]
    let k = read_bf16(&st, "k"); // [1,H_KV,T_CACHE,D]
    let v = read_bf16(&st, "v");
    let out_ref = read_bf16(&st, "out"); // [1,H_Q,1,D]

    let device = Device::system_default().expect("Metal device");
    let lib = KernelLibrary::sdpa_vector(&device).expect("lib");
    let q_buf: Buffer<bf16> = device.upload(&q).expect("q");
    let k_buf: Buffer<bf16> = device.upload(&k).expect("k");
    let v_buf: Buffer<bf16> = device.upload(&v).expect("v");
    let mut out_buf: Buffer<bf16> = device.alloc_buffer(H_Q * D).expect("out");

    let mut commands = device.commands().expect("commands");
    sdpa::dispatch_sdpa_vector_bf16(
        &mut commands,
        &lib,
        &q_buf,
        &k_buf,
        &v_buf,
        &mut out_buf,
        sdpa::SdpaVectorArgs {
            gqa_factor: (H_Q / H_KV) as i32,
            n_keys: T_CACHE as i32,
            k_head_stride: (T_CACHE * D) as u64,
            k_seq_stride: D as u64,
            v_head_stride: (T_CACHE * D) as u64,
            v_seq_stride: D as u64,
            scale: 1.0 / (D as f32).sqrt(),
            head_dim: D,
        },
    )
    .expect("dispatch sdpa_vector");
    commands.commit_and_wait().expect("commit");

    let out: Vec<bf16> = unsafe { out_buf.as_mut_slice() }.to_vec();
    // SDPA carries exp → allclose, not bit-exact.
    let (atol, rtol) = (0.01f32, 0.02f32);
    let mut max_err = 0.0f32;
    for (i, (&a, &b)) in out.iter().zip(out_ref.iter()).enumerate() {
        let (af, bf) = (a.to_f32(), b.to_f32());
        let err = (af - bf).abs();
        max_err = max_err.max(err);
        assert!(
            err <= atol + rtol * bf.abs(),
            "sdpa mismatch at {i}: got {af}, want {bf} (abs err {err})"
        );
    }
    eprintln!("sdpa_vector parity max abs err: {max_err}");
}
