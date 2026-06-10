//! Compile-smoke for the vendored fused `steel_attention` (`sdpa_full`
//! prefill) kernel: JIT the library and specialize the pinned causal
//! bf16 instantiation. macOS + live Metal device. Also drives
//! `dispatch_sdpa_full_bf16` at the real Qwen2 prefill shape against
//! MLX's `mx.fast.scaled_dot_product_attention`.
#![cfg(target_os = "macos")]

use cider_press_kernels::{Buffer, Device, FunctionConstant, KernelLibrary, kernels::sdpa};
use cider_press_test_utils::{dump_mlx_op, read_bf16, tempdir};
use half::bf16;
use safetensors::SafeTensors;

#[test]
fn steel_attention_library_compiles() {
    let device = Device::system_default().expect("Metal device");
    let lib = KernelLibrary::steel_attention(&device).expect("compile steel_attention");
    // Every instantiation requires [[function_constant]] specialization
    // (200/201 align, 300 has_mask, 301 do_causal, 302 has_sinks); the
    // causal bf16 prefill set is the figure-out-the-shape canary. The
    // host name uses `bfloat16` (no `_t` suffix) because the
    // instantiate_attn macro stringifies a separate `tname` token.
    let constants = [
        FunctionConstant::Bool {
            index: 200,
            value: false,
        }, // align_Q
        FunctionConstant::Bool {
            index: 201,
            value: false,
        }, // align_K
        FunctionConstant::Bool {
            index: 300,
            value: false,
        }, // has_mask
        FunctionConstant::Bool {
            index: 301,
            value: true,
        }, // do_causal
        FunctionConstant::Bool {
            index: 302,
            value: false,
        }, // has_sinks
    ];
    lib.pipeline_specialized(
        "steel_attention_bfloat16_bq32_bk32_bd64_wm4_wn1_maskbfloat16",
        &constants,
    )
    .expect("steel_attention bf16 bd64 causal pipeline");
}

const H_Q: usize = 14;
const H_KV: usize = 2;
const D: usize = 64;
const T: usize = 39;

#[test]
// Cast operands are compile-time consts (small, exact); the lints guard
// against runtime truncation that cannot occur here.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss
)]
fn steel_attention_prefill_causal_matches_mlx() {
    let tmp = tempdir("steel-attention-parity");
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
            &T.to_string(),
            "--seq-kv",
            &T.to_string(),
            "--head-dim",
            &D.to_string(),
            "--dtype",
            "bf16",
            "--causal",
        ],
    );
    let bytes = std::fs::read(&fixture).expect("read fixture");
    let st = SafeTensors::deserialize(&bytes).expect("parse safetensors");
    let q = read_bf16(&st, "q"); // [1, H_Q, T, D]
    let k = read_bf16(&st, "k"); // [1, H_KV, T, D]
    let v = read_bf16(&st, "v");
    let out_ref = read_bf16(&st, "out"); // [1, H_Q, T, D]

    let device = Device::system_default().expect("Metal device");
    let lib = KernelLibrary::steel_attention(&device).expect("lib");
    let q_buf: Buffer<bf16> = device.upload(&q).expect("q");
    let k_buf: Buffer<bf16> = device.upload(&k).expect("k");
    let v_buf: Buffer<bf16> = device.upload(&v).expect("v");
    let mut out_buf: Buffer<bf16> = device.alloc_buffer(H_Q * T * D).expect("out");

    let mut commands = device.commands().expect("commands");
    sdpa::dispatch_sdpa_full_bf16(
        &mut commands,
        &lib,
        &q_buf,
        &k_buf,
        &v_buf,
        &mut out_buf,
        sdpa::SdpaFullArgs {
            batch: 1,
            h_q: H_Q as i32,
            h_kv: H_KV as i32,
            head_dim: D as i32,
            q_len: T as i32,
            k_len: T as i32,
            scale: 1.0 / (D as f32).sqrt(),
            q_strides: [(H_Q * T * D) as i64, (T * D) as i64, D as i64],
            k_strides: [(H_KV * T * D) as i64, (T * D) as i64, D as i64],
            v_strides: [(H_KV * T * D) as i64, (T * D) as i64, D as i64],
            o_strides: [(H_Q * T * D) as i64, (T * D) as i64, D as i64],
            causal: true,
        },
    )
    .expect("dispatch sdpa_full");
    commands.commit_and_wait().expect("commit");

    let out: Vec<bf16> = unsafe { out_buf.as_mut_slice() }.to_vec();
    // Composed-SDPA chain carries exp → combined allclose bound, not
    // bit-exact. Tolerance matches the prefill-causal SDPA bar.
    let (atol, rtol) = (0.03f32, 0.08f32);
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    assert_eq!(out.len(), out_ref.len(), "length mismatch");
    for (i, (&a, &b)) in out.iter().zip(out_ref.iter()).enumerate() {
        let (af, bf) = (a.to_f32(), b.to_f32());
        let abs = (af - bf).abs();
        max_abs = max_abs.max(abs);
        if bf.abs() > 1e-6 {
            max_rel = max_rel.max(abs / bf.abs());
        }
        assert!(
            abs <= atol + rtol * bf.abs(),
            "steel_attention mismatch at {i}: got {af}, want {bf} (abs err {abs})"
        );
    }
    eprintln!("steel_attention parity: max abs err {max_abs}, max rel err {max_rel}");
}

#[test]
// Cast operands are compile-time consts (small, exact); the lints guard
// against runtime truncation that cannot occur here.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss
)]
fn steel_attention_strided_kv_matches_dense() {
    // Regression guard: the prefill KV-cache view is head-interleaved
    // (a `[T, H_kv, D]` slab permuted to `[H_kv, T, D]`), so K/V reach the
    // kernel with head_stride = D and seq_stride = H_kv*D — NOT the dense
    // `[H_kv, T, D]` (head_stride = T*D, seq_stride = D). A later task feeds
    // that strided view straight to the kernel; this proves the kernel honors
    // the non-canonical H/L strides by re-laying identical logical K/V into the
    // interleaved memory order with compensating strides and demanding the same
    // result.
    let tmp = tempdir("steel-attention-strided");
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
            &T.to_string(),
            "--seq-kv",
            &T.to_string(),
            "--head-dim",
            &D.to_string(),
            "--dtype",
            "bf16",
            "--causal",
        ],
    );
    let bytes = std::fs::read(&fixture).expect("read fixture");
    let st = SafeTensors::deserialize(&bytes).expect("parse safetensors");
    let q = read_bf16(&st, "q"); // [1, H_Q, T, D]
    let k = read_bf16(&st, "k"); // [1, H_KV, T, D], row-major: h*T*D + s*D + e
    let v = read_bf16(&st, "v");
    let out_ref = read_bf16(&st, "out"); // [1, H_Q, T, D]

    // Re-lay K/V from dense `[H_kv, T, D]` into head-interleaved
    // `[T, H_kv, D]`: element (h, s, e) moves to s*H_kv*D + h*D + e. The
    // strides passed below (head_stride = D, seq_stride = H_kv*D) make the
    // kernel read this layout as the same logical `[H_kv, T, D]` tensor.
    let interleave = |src: &[bf16]| -> Vec<bf16> {
        let mut dst = vec![bf16::ZERO; H_KV * T * D];
        for h in 0..H_KV {
            for s in 0..T {
                for e in 0..D {
                    dst[s * H_KV * D + h * D + e] = src[h * T * D + s * D + e];
                }
            }
        }
        dst
    };
    let k_il = interleave(&k);
    let v_il = interleave(&v);

    let device = Device::system_default().expect("Metal device");
    let lib = KernelLibrary::steel_attention(&device).expect("lib");
    let q_buf: Buffer<bf16> = device.upload(&q).expect("q");
    let k_buf: Buffer<bf16> = device.upload(&k_il).expect("k");
    let v_buf: Buffer<bf16> = device.upload(&v_il).expect("v");
    let mut out_buf: Buffer<bf16> = device.alloc_buffer(H_Q * T * D).expect("out");

    let mut commands = device.commands().expect("commands");
    sdpa::dispatch_sdpa_full_bf16(
        &mut commands,
        &lib,
        &q_buf,
        &k_buf,
        &v_buf,
        &mut out_buf,
        sdpa::SdpaFullArgs {
            batch: 1,
            h_q: H_Q as i32,
            h_kv: H_KV as i32,
            head_dim: D as i32,
            q_len: T as i32,
            k_len: T as i32,
            scale: 1.0 / (D as f32).sqrt(),
            // Q/O stay dense `[H, T, D]`; only K/V carry the interleaved
            // (head_stride = D, seq_stride = H_kv*D) view.
            q_strides: [(H_Q * T * D) as i64, (T * D) as i64, D as i64],
            k_strides: [(H_KV * T * D) as i64, D as i64, (H_KV * D) as i64],
            v_strides: [(H_KV * T * D) as i64, D as i64, (H_KV * D) as i64],
            o_strides: [(H_Q * T * D) as i64, (T * D) as i64, D as i64],
            causal: true,
        },
    )
    .expect("dispatch sdpa_full");
    commands.commit_and_wait().expect("commit");

    let out: Vec<bf16> = unsafe { out_buf.as_mut_slice() }.to_vec();
    let (atol, rtol) = (0.03f32, 0.08f32);
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    assert_eq!(out.len(), out_ref.len(), "length mismatch");
    for (i, (&a, &b)) in out.iter().zip(out_ref.iter()).enumerate() {
        let (af, bf) = (a.to_f32(), b.to_f32());
        let abs = (af - bf).abs();
        max_abs = max_abs.max(abs);
        if bf.abs() > 1e-6 {
            max_rel = max_rel.max(abs / bf.abs());
        }
        assert!(
            abs <= atol + rtol * bf.abs(),
            "steel_attention strided mismatch at {i}: got {af}, want {bf} (abs err {abs})"
        );
    }
    eprintln!("steel_attention strided parity: max abs err {max_abs}, max rel err {max_rel}");
}
