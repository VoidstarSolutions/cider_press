#![cfg(target_os = "macos")]
#![allow(clippy::cast_precision_loss, clippy::many_single_char_names)]
use cider_press_runtime::{Device, Tensor};
use half::bf16;

#[test]
fn sdpa_accepts_prefill_causal_and_evals() {
    let (h_q, h_kv, t, d) = (14usize, 2usize, 39usize, 64usize);
    let device = Device::system_default().expect("device");
    let mk = |heads: usize| {
        let data = vec![bf16::from_f32(0.01); heads * t * d];
        Tensor::from_slice(&device, &data, [1, heads, t, d]).expect("t")
    };
    let (q, k, v) = (mk(h_q), mk(h_kv), mk(h_kv));
    let scale = 1.0f32 / (d as f32).sqrt();
    let out = Tensor::sdpa(&q, &k, &v, None, scale, h_q / h_kv, true).expect("build sdpa T>1");
    out.eval().expect("eval prefill sdpa");
    assert_eq!(out.shape().dims(), &[1, h_q, t, d]);
}

#[test]
fn sdpa_rejects_prefill_non64_head_dim() {
    // Prefill (T_q>1) routes to steel sdpa_full, instantiated only for D=64.
    // A dim the vector kernel supports (128) must still fail at construction.
    let (h_q, h_kv, t, d) = (8usize, 2usize, 4usize, 128usize);
    let device = Device::system_default().expect("device");
    let mk = |heads: usize| {
        let data = vec![bf16::from_f32(0.01); heads * t * d];
        Tensor::from_slice(&device, &data, [1, heads, t, d]).expect("t")
    };
    let (q, k, v) = (mk(h_q), mk(h_kv), mk(h_kv));
    let scale = 1.0f32 / (d as f32).sqrt();
    let err = Tensor::sdpa(&q, &k, &v, None, scale, h_q / h_kv, true)
        .expect_err("prefill head_dim 128 must be rejected at construction");
    assert!(err.to_string().contains("head_dim 64"), "got: {err}");
}
