//! Runtime-layer tests for `Tensor::sdpa`: construction validation here,
//! through-the-op parity in a later task. macOS + Metal.
#![cfg(target_os = "macos")]
#![allow(clippy::cast_precision_loss)]

use cider_press_runtime::{Device, Tensor};
use cider_press_test_utils::{dump_mlx_op, read_bf16, tempdir};
use half::bf16;
use safetensors::SafeTensors;

#[test]
fn sdpa_rejects_unsupported_head_dim() {
    let device = Device::system_default().expect("Metal device");
    // head_dim = 48 is not in the vector kernel's supported set.
    let q = Tensor::from_slice(&device, &[bf16::ONE; 14 * 48], [1, 14, 1, 48]).expect("q");
    let k = Tensor::from_slice(&device, &[bf16::ONE; 2 * 16 * 48], [1, 2, 16, 48]).expect("k");
    let v = Tensor::from_slice(&device, &[bf16::ONE; 2 * 16 * 48], [1, 2, 16, 48]).expect("v");
    let err = Tensor::sdpa(&q, &k, &v, None, 0.1, 7, false).expect_err("bad head_dim must error");
    assert!(
        format!("{err}").contains("head_dim"),
        "error names head_dim: {err}"
    );
}

#[test]
fn sdpa_rejects_gqa_mismatch() {
    let device = Device::system_default().expect("Metal device");
    let q = Tensor::from_slice(&device, &[bf16::ONE; 16 * 64], [1, 16, 1, 64]).expect("q");
    let k = Tensor::from_slice(&device, &[bf16::ONE; 2 * 16 * 64], [1, 2, 16, 64]).expect("k");
    let v = Tensor::from_slice(&device, &[bf16::ONE; 2 * 16 * 64], [1, 2, 16, 64]).expect("v");
    // H_q=16, H_kv=2 → valid gqa_factor is 8; 9 is wrong.
    let err =
        Tensor::sdpa(&q, &k, &v, None, 0.125, 9, false).expect_err("bad gqa_factor must error");
    assert!(format!("{err}").contains("gqa_factor"), "{err}");
}

#[test]
fn sdpa_through_op_matches_mlx() {
    const H_Q: usize = 14;
    const H_KV: usize = 2;
    const T: usize = 16;
    const D: usize = 64;

    let tmp = tempdir("sdpa-op-parity");
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
            &T.to_string(),
            "--head-dim",
            &D.to_string(),
        ],
    );
    let bytes = std::fs::read(&fixture).expect("read");
    let st = SafeTensors::deserialize(&bytes).expect("parse");
    let q = read_bf16(&st, "q");
    let k = read_bf16(&st, "k");
    let v = read_bf16(&st, "v");
    let out_ref = read_bf16(&st, "out");

    let device = Device::system_default().expect("device");
    let qt = Tensor::from_slice(&device, &q, [1, H_Q, 1, D]).expect("q");
    let kt = Tensor::from_slice(&device, &k, [1, H_KV, T, D]).expect("k");
    let vt = Tensor::from_slice(&device, &v, [1, H_KV, T, D]).expect("v");

    let scale = 1.0f32 / (D as f32).sqrt();
    let o = Tensor::sdpa(&qt, &kt, &vt, None, scale, H_Q / H_KV, false).expect("sdpa");
    o.eval().expect("eval");
    let out: Vec<bf16> = o.cpu_to_vec().expect("dense out");

    let (atol, rtol) = (0.01f32, 0.02f32);
    for (i, (&a, &b)) in out.iter().zip(out_ref.iter()).enumerate() {
        let (af, bf) = (a.to_f32(), b.to_f32());
        assert!(
            (af - bf).abs() <= atol + rtol * bf.abs(),
            "mismatch at {i}: {af} vs {bf}"
        );
    }
}
