//! The runtime routes `M > 1, K % 64 == 0` quantized matmuls to the NAX
//! kernel where available and leaves everything else on the classic path.
//! Both branches are checked against MLX fixtures through the public
//! `Tensor::quantized_matmul` API — the NAX branch bit-exact, the classic
//! branch under the existing tolerance.

#![cfg(target_os = "macos")]
#![allow(clippy::many_single_char_names)]

use cider_press_runtime::{Device, Quantization, QuantizedWeight, Tensor};
use cider_press_test_utils::{dump_mlx_op, read_bf16, tempdir};
use half::bf16;
use safetensors::SafeTensors;

/// `(y_out, y_ref)` for an MLX-generated `[m, k] x [n, k]^T` case driven through
/// the runtime op.
fn run_case(tag: &str, m: usize, n: usize, k: usize, group_size: u32) -> (Vec<bf16>, Vec<bf16>) {
    let tmp = tempdir(tag);
    let path = tmp.join("qmm.safetensors");
    let (m_s, n_s, k_s, gs_s) = (
        m.to_string(),
        n.to_string(),
        k.to_string(),
        group_size.to_string(),
    );
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
            &gs_s,
            "--bits",
            "4",
            "--dtype",
            "bf16",
        ],
    );
    let bytes = std::fs::read(&path).expect("read fixture");
    let st = SafeTensors::deserialize(&bytes).expect("parse safetensors");
    let raw = |name: &str| {
        st.tensor(name)
            .unwrap_or_else(|e| panic!("{name}: {e}"))
            .data()
    };
    let x_raw = read_bf16(&st, "x");
    let y_ref = read_bf16(&st, "y");

    let device = Device::shared().expect("device");
    let qw = QuantizedWeight::from_bytes(
        &device,
        [n, k],
        Quantization::new(4, group_size).expect("quantization"),
        raw("w_q"),
        raw("scales"),
        raw("biases"),
    )
    .expect("build QuantizedWeight");
    let x = Tensor::from_slice(&device, &x_raw, [m, k]).expect("upload x");
    let y = x.quantized_matmul(&qw, true).expect("schedule");
    y.eval().expect("eval");
    (y.cpu_slice::<bf16>().expect("cpu_slice").to_vec(), y_ref)
}

#[test]
fn routes_to_nax_bit_exact_when_available() {
    if !Device::shared().expect("device").nax_available() {
        eprintln!("skipping: NAX unavailable on this GPU/OS");
        return;
    }
    let (y, y_ref) = run_case("runtime-qmm-nax-route", 512, 2048, 1024, 64);
    let mismatches = y
        .iter()
        .zip(y_ref.iter())
        .filter(|(a, e)| a.to_bits() != e.to_bits())
        .count();
    assert_eq!(mismatches, 0, "{mismatches}/{} differ from MLX", y.len());
}

#[test]
fn routes_to_classic_when_k_not_multiple_of_64() {
    // K = 96 (group size 32, since 96 is not a whole number of 64-groups):
    // NAX cannot take it; the classic kernel must, within its tolerance.
    let (y, y_ref) = run_case("runtime-qmm-classic-k96", 8, 64, 96, 32);
    for (a, e) in y.iter().zip(y_ref.iter()) {
        let (a, e) = (a.to_f32(), e.to_f32());
        assert!((a - e).abs() <= 5e-2 + 1e-2 * e.abs(), "{a} vs {e}");
    }
}
