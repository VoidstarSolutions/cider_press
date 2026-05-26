//! Runtime-level parity test for `Tensor::quantized_matmul` (M=8 prefill path)
//! against MLX's `mx.quantized_matmul`.
//!
//! Reuses the Task-1 fixture from the kernels crate
//! (`crates/cider-press-kernels/tests/fixtures/qmm_qwen2.safetensors`)
//! but drives the dispatch through the runtime API: build a
//! `QuantizedWeight`, build an `[M=8, K=896]` activation tensor, schedule
//! `quantized_matmul`, `eval()`, and read the result back.

#![cfg(target_os = "macos")]
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use std::path::PathBuf;

use approx::assert_relative_eq;
use cider_press_runtime::{DType, Device, Quantization, QuantizedWeight, Tensor};
use cider_press_test_utils::read_bf16;
use half::bf16;
use safetensors::SafeTensors;

// Test dims — must match scripts/gen_qmm_fixture.py.
const M: usize = 8;
const K: usize = 896;
const N: usize = 896;

fn fixture_path() -> PathBuf {
    // The fixture is committed under the kernels crate; reuse it from
    // there rather than duplicating the binary blob across crates.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates dir")
        .join("cider-press-kernels")
        .join("tests")
        .join("fixtures")
        .join("qmm_qwen2.safetensors")
}

#[test]
fn qmm_through_runtime_matches_mlx() {
    let path = fixture_path();
    assert!(
        path.exists(),
        "missing parity fixture at {}; run `uv run scripts/gen_qmm_fixture.py` \
         from the workspace root to regenerate it",
        path.display(),
    );

    let bytes = std::fs::read(&path).expect("read fixture");
    let st = SafeTensors::deserialize(&bytes).expect("parse safetensors");

    // Pull the raw safetensors bytes for each tensor. The runtime's
    // `QuantizedWeight::from_bytes` takes byte slices directly — no
    // need to deserialize to typed Vecs first.
    let w_bytes = bytes_for(&st, "w_q");
    let scales_bytes = bytes_for(&st, "scales");
    let biases_bytes = bytes_for(&st, "biases");
    let x_raw = read_bf16(&st, "x");
    let y_ref = read_bf16(&st, "y_ref");

    assert_eq!(x_raw.len(), M * K);
    assert_eq!(y_ref.len(), M * N);

    let device = Device::shared().expect("device");

    let qw = QuantizedWeight::from_bytes(
        &device,
        [N, K],
        Quantization::Q4_GS64,
        w_bytes,
        scales_bytes,
        biases_bytes,
    )
    .expect("build QuantizedWeight");

    let x = Tensor::from_slice(&device, &x_raw, [M, K]).expect("upload x");

    let y = x.quantized_matmul(&qw).expect("schedule quantized_matmul");
    assert!(!y.is_materialized(), "quantized_matmul is lazy");
    assert_eq!(y.shape().dims(), &[M, N]);
    assert_eq!(y.dtype(), DType::BF16);

    y.eval().expect("eval");
    assert!(y.is_materialized());

    let y_out = y.cpu_slice::<bf16>().expect("cpu_slice bf16");
    assert_eq!(y_out.len(), M * N);

    // Same tolerance as the kernels-layer test: bit-exact on identical
    // hardware, 1–2 ULP drift across Apple Silicon generations.
    for (a, e) in y_out.iter().zip(y_ref.iter()) {
        let af = a.to_f32();
        let ef = e.to_f32();
        assert_relative_eq!(af, ef, max_relative = 1e-2, epsilon = 5e-2);
    }
}

fn bytes_for<'a>(st: &'a SafeTensors<'a>, name: &str) -> &'a [u8] {
    st.tensor(name)
        .unwrap_or_else(|e| panic!("tensor {name}: {e}"))
        .data()
}
