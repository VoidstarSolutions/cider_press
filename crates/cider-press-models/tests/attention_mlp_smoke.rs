//! Synthetic (no-checkpoint) shape/dtype/lazy smoke tests for the
//! branch-12b model layers. This task adds the `Mlp::forward` case;
//! Task 2 extends the file with an `Attention::forward` case.
//!
//! Numerical parity against MLX is covered by the gated `mlp_layer0.rs`
//! and `attention_layer0.rs` integration tests. These tests just
//! confirm the lazy graph builds with the right output shape/dtype on
//! zero-filled Qwen2.5-0.5B-shaped weights.

#![cfg(target_os = "macos")]

use cider_press_models::nn::{Linear, Mlp, Module};
use cider_press_runtime::{DType, Device, Quantization, QuantizedWeight, Tensor};
use half::bf16;

// Qwen2.5-0.5B-Instruct dimensions.
const HIDDEN_SIZE: usize = 896;
const INTERMEDIATE: usize = 4864;
const T: usize = 4;

/// Zero-filled `QuantizedWeight` with logical `[n, k]`, q4-gs64.
fn zero_qw(device: &Device, n: usize, k: usize) -> QuantizedWeight {
    let elem_count = n * k;
    let w_bytes = elem_count * 4 / 32 * 4;
    let aux_bytes = (elem_count / 64) * 2;
    QuantizedWeight::from_bytes(
        device,
        [n, k],
        Quantization::Q4_GS64,
        &vec![0u8; w_bytes],
        &vec![0u8; aux_bytes],
        &vec![0u8; aux_bytes],
    )
    .expect("zero QuantizedWeight")
}

#[test]
fn mlp_forward_shape_dtype_and_lazy() {
    let device = Device::shared().expect("device");

    let gate = Linear::new(zero_qw(&device, INTERMEDIATE, HIDDEN_SIZE), None).expect("gate");
    let up = Linear::new(zero_qw(&device, INTERMEDIATE, HIDDEN_SIZE), None).expect("up");
    let down = Linear::new(zero_qw(&device, HIDDEN_SIZE, INTERMEDIATE), None).expect("down");
    let mlp = Mlp::new(gate, up, down);

    let x = Tensor::from_slice(
        &device,
        &vec![bf16::ZERO; T * HIDDEN_SIZE],
        [1usize, T, HIDDEN_SIZE],
    )
    .expect("hidden");

    let out = mlp.forward(&x).expect("mlp forward");
    assert_eq!(out.shape().dims(), &[1, T, HIDDEN_SIZE], "mlp out shape");
    assert_eq!(out.dtype(), DType::BF16, "mlp out dtype");
    assert!(!out.is_materialized(), "mlp out should be lazy");
}
