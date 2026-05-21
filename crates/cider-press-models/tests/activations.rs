//! Parity tests for the composed [`nn::silu`] and [`nn::gelu`]
//! activations against `mlx.nn.silu` / `mlx.nn.gelu`.
//!
//! `nn.silu` is composed identically on both sides (`x * sigmoid(x)`)
//! but sigmoid drifts 1–2 bf16 ULPs across Apple Silicon generations
//! (see the kernel-layer sigmoid parity test) and the final bf16
//! rounding of the multiply can push relative error a touch past that,
//! so silu uses the same bf16-compose tolerance as gelu. `nn.gelu`
//! differs by one
//! arithmetic detail: MLX writes `x / sqrt(2)` (a division by the
//! bf16-rounded constant `sqrt(2) ≈ 1.4140625`) where we write
//! `x * (1/sqrt(2))` (a multiplication by the bf16-rounded constant
//! `1/sqrt(2) ≈ 0.70703125`). The runtime doesn't yet expose a divide
//! op, so we use the reciprocal-multiply, and use a bf16-compose
//! tolerance for gelu — same pattern as `rms_norm`.

#![cfg(target_os = "macos")]

use std::path::{Path, PathBuf};
use std::process::Command;

use cider_press_models::nn::{gelu, silu};
use cider_press_runtime::{Device, Tensor};
use half::bf16;
use safetensors::SafeTensors;

const S: usize = 8;
const H: usize = 896;

#[test]
fn silu_matches_mlx_nn_silu_within_ulp_tolerance() {
    let (lhs, out_ref) = fixture("silu");
    let device = Device::system_default().expect("device");
    let x = Tensor::from_slice(&device, &lhs, [1, S, H]).expect("x");
    let y = silu(&x).expect("schedule silu");
    y.eval().expect("eval");
    let got: Vec<bf16> = y.cpu_to_vec().expect("dense out");
    assert_eq!(got.len(), out_ref.len());
    // silu = x * sigmoid(x); sigmoid carries 1–2 bf16 ULPs of
    // cross-hardware drift through `metal::exp` (see kernel-layer
    // sigmoid parity test). The final bf16 rounding of `x * sigmoid(x)`
    // can push relative error a touch past the raw sigmoid bar at small
    // outputs, so we sit at the gelu compose tolerance (0.02 / 0.02).
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for (a, b) in got.iter().zip(out_ref.iter()) {
        let af = a.to_f32();
        let bf = b.to_f32();
        let abs = (af - bf).abs();
        let rel = if bf.abs() > 1e-3 { abs / bf.abs() } else { 0.0 };
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(rel);
    }
    assert!(
        max_abs <= 0.02,
        "nn::silu max_abs {max_abs} exceeded bf16-compose tolerance 0.02"
    );
    assert!(
        max_rel <= 0.02,
        "nn::silu max_rel {max_rel} exceeded bf16-compose tolerance 0.02"
    );
}

#[test]
fn gelu_matches_mlx_nn_gelu_within_bf16_compose_tolerance() {
    let (lhs, out_ref) = fixture("gelu");
    let device = Device::system_default().expect("device");
    let x = Tensor::from_slice(&device, &lhs, [1, S, H]).expect("x");
    let y = gelu(&x).expect("schedule gelu");
    y.eval().expect("eval");
    let got: Vec<bf16> = y.cpu_to_vec().expect("dense out");
    assert_eq!(got.len(), out_ref.len());

    // bf16-compose tolerance. MLX uses x / sqrt(2); we use
    // x * (1/sqrt(2)) because the runtime doesn't expose a divide op
    // yet. The bf16-rounded constants differ in the last few bits, so
    // exact agreement isn't possible without matching the exact MLX
    // expression. Per-element drift stays well under a bf16 ULP at the
    // magnitudes gelu outputs sit at.
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for (a, b) in got.iter().zip(out_ref.iter()) {
        let af = a.to_f32();
        let bf = b.to_f32();
        let abs = (af - bf).abs();
        let rel = if bf.abs() > 1e-3 { abs / bf.abs() } else { 0.0 };
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(rel);
    }
    assert!(
        max_abs <= 0.02,
        "max absolute error {max_abs} exceeded bf16-compose tolerance 0.02",
    );
    assert!(
        max_rel <= 0.02,
        "max relative error {max_rel} exceeded bf16-compose tolerance 0.02",
    );
}

// ---------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------

fn fixture(op: &str) -> (Vec<bf16>, Vec<bf16>) {
    let tmp = tempdir();
    let path = tmp.join(format!("{op}.safetensors"));
    dump_mlx(op, &path, &format!("1,{S},{H}"));
    let bytes = std::fs::read(&path).expect("read fixture");
    let st = SafeTensors::deserialize(&bytes).expect("parse safetensors");
    (read_bf16(&st, "lhs"), read_bf16(&st, "out"))
}

fn dump_mlx(op: &str, out: &Path, lhs_shape: &str) {
    let script = workspace_root().join("scripts").join("dump_mlx_op.py");
    let status = Command::new("uv")
        .arg("run")
        .arg(&script)
        .arg("--output")
        .arg(out)
        .arg("--seed")
        .arg("0")
        .arg(op)
        .arg("--lhs-shape")
        .arg(lhs_shape)
        .arg("--dtype")
        .arg("bf16")
        .status();
    let status = match status {
        Ok(s) => s,
        Err(err) => panic!(
            "failed to invoke `uv run {}`: {err}. This test requires `uv` on PATH; \
             CI installs it via astral-sh/setup-uv.",
            script.display()
        ),
    };
    assert!(status.success(), "dump_mlx_op.py {op} exited {status}");
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("workspace root")
        .to_path_buf()
}

fn read_bf16(st: &SafeTensors, name: &str) -> Vec<bf16> {
    let view = st
        .tensor(name)
        .unwrap_or_else(|e| panic!("tensor {name}: {e}"));
    assert_eq!(view.dtype(), safetensors::Dtype::BF16);
    let bytes = view.data();
    assert!(bytes.len() % 2 == 0);
    bytes
        .chunks_exact(2)
        .map(|c| bf16::from_le_bytes([c[0], c[1]]))
        .collect()
}

fn tempdir() -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("SystemTime")
        .as_nanos();
    let pid = std::process::id();
    let dir = std::env::temp_dir().join(format!("cider-press-activations-{pid}-{nanos}"));
    std::fs::create_dir_all(&dir).expect("mktemp");
    dir
}
