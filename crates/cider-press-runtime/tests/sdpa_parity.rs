//! Runtime-layer tests for `Tensor::sdpa`: construction validation here,
//! through-the-op parity in a later task. macOS + Metal.
#![cfg(target_os = "macos")]

use cider_press_runtime::{Device, Tensor};
use half::bf16;

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
