//! Runtime-level parity test for `Tensor::quantized_matmul` (M=1 decode path)
//! against MLX's `mx.quantized_matmul`.
//!
//! Reuses the Stage-4 fixture from the kernels crate
//! (`crates/cider-press-kernels/tests/fixtures/stage4_qmv.safetensors`)
//! but drives the dispatch through the runtime API: build a
//! `QuantizedWeight`, build an activation tensor, schedule
//! `quantized_matmul`, `eval()`, and read the result back.
//!
//! The kernels-level test [crates/cider-press-kernels/tests/stage4_qmv.rs]
//! has a relaxed bf16 tolerance to absorb GPU-family differences when
//! run on multiple machines. The same kernel is being called here, so
//! the comparison passes at the same tolerance.

#![cfg(target_os = "macos")]
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use std::path::PathBuf;

use approx::assert_relative_eq;
use cider_press_runtime::{DType, Device, Quantization, QuantizedWeight, Tensor};
use cider_press_test_utils::read_bf16;
use half::bf16;
use safetensors::SafeTensors;

const K: usize = 512;
const N: usize = 512;

fn fixture_path() -> PathBuf {
    // The fixture is committed under the kernels crate; reuse it from
    // there rather than duplicating the binary blob across crates.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates dir")
        .join("cider-press-kernels")
        .join("tests")
        .join("fixtures")
        .join("stage4_qmv.safetensors")
}

#[test]
fn qmv_through_runtime_matches_mlx() {
    let path = fixture_path();
    assert!(
        path.exists(),
        "missing parity fixture at {}; run `uv run scripts/gen_stage4_fixtures.py` \
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
    let x_bytes = bytes_for(&st, "x");
    let y_ref = read_bf16(&st, "y_ref");

    assert_eq!(y_ref.len(), N);

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

    let x = Tensor::from_bytes(&device, x_bytes, [K], DType::BF16).expect("upload x");

    let y = x.quantized_matmul(&qw, true).expect("schedule quantized_matmul");
    assert!(!y.is_materialized(), "quantized_matmul is lazy");

    y.eval().expect("eval");
    assert!(y.is_materialized());

    let y_out = y.cpu_slice::<bf16>().expect("cpu_slice bf16");
    assert_eq!(y_out.len(), N);

    // Same kernel as Stage 4 ⇒ same per-element tolerance. The
    // kernels-level test documents why this isn't bit-exact across
    // all Apple Silicon families even though the spike measured
    // max_relative=max_absolute=0 on the development machine.
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
