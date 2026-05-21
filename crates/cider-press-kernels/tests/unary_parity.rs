//! Parity test for `unary.metal` Square / Rsqrt / Sigmoid / Erf
//! dispatch (branches 5 + 6).
//!
//! Generates fixtures at test time via `uv run scripts/dump_mlx_op.py
//! {square,rsqrt,sigmoid,erf}` and asserts bit-exact equality with MLX
//! for bf16 at a Qwen2-residual-shaped tensor (`[1, 8, 896]`). Each op
//! is a single rounded primitive in MSL, so any drift here is a
//! dispatch-transcription bug.
//!
//! Like the binary parity test, fixtures are kept out of the repo —
//! `dump_mlx_op.py` is the single source of truth for the reference
//! output across kernels-, runtime-, and models-crate parity tests.

#![cfg(target_os = "macos")]

use std::path::{Path, PathBuf};
use std::process::Command;

use cider_press_kernels::{Buffer, Device, KernelLibrary, kernels};
use half::bf16;
use safetensors::SafeTensors;

const S: usize = 8;
const H: usize = 896;

#[test]
fn parity_v_square_bf16() {
    let (lhs, out_ref) = fixture("square");
    let device = Device::system_default().expect("Metal device");
    let library = KernelLibrary::unary(&device).expect("compile unary.metal");

    let src: Buffer<bf16> = device.upload(&lhs).expect("upload lhs");
    let mut dst: Buffer<bf16> = device.alloc_buffer(S * H).expect("alloc out");
    let mut cmds = device.commands().expect("commands");
    kernels::unary::v_square_bf16(&mut cmds, &library, &src, &mut dst).expect("dispatch");
    cmds.commit_and_wait().expect("commit");

    // SAFETY: commit_and_wait synchronised; GPU is done with `dst`.
    let out: Vec<bf16> = unsafe { dst.as_mut_slice() }.to_vec();
    assert_eq!(
        out, out_ref,
        "v_Squarebfloat16bfloat16 must be bit-exact vs MLX"
    );
}

#[test]
fn parity_v_rsqrt_bf16() {
    let (lhs, out_ref) = fixture("rsqrt");
    let device = Device::system_default().expect("Metal device");
    let library = KernelLibrary::unary(&device).expect("compile unary.metal");

    let src: Buffer<bf16> = device.upload(&lhs).expect("upload lhs");
    let mut dst: Buffer<bf16> = device.alloc_buffer(S * H).expect("alloc out");
    let mut cmds = device.commands().expect("commands");
    kernels::unary::v_rsqrt_bf16(&mut cmds, &library, &src, &mut dst).expect("dispatch");
    cmds.commit_and_wait().expect("commit");

    // SAFETY: commit_and_wait synchronised; GPU is done with `dst`.
    let out: Vec<bf16> = unsafe { dst.as_mut_slice() }.to_vec();
    assert_eq!(
        out, out_ref,
        "v_Rsqrtbfloat16bfloat16 must be bit-exact vs MLX"
    );
}

#[test]
fn parity_v_sigmoid_bf16() {
    let (lhs, out_ref) = fixture("sigmoid");
    let device = Device::system_default().expect("Metal device");
    let library = KernelLibrary::unary(&device).expect("compile unary.metal");

    let src: Buffer<bf16> = device.upload(&lhs).expect("upload lhs");
    let mut dst: Buffer<bf16> = device.alloc_buffer(S * H).expect("alloc out");
    let mut cmds = device.commands().expect("commands");
    kernels::unary::v_sigmoid_bf16(&mut cmds, &library, &src, &mut dst).expect("dispatch");
    cmds.commit_and_wait().expect("commit");

    // SAFETY: commit_and_wait synchronised; GPU is done with `dst`.
    let out: Vec<bf16> = unsafe { dst.as_mut_slice() }.to_vec();
    if out != out_ref {
        diagnose_mismatch("v_Sigmoidbfloat16bfloat16", &out, &out_ref);
    }
    assert_eq!(
        out, out_ref,
        "v_Sigmoidbfloat16bfloat16 must be bit-exact vs MLX"
    );
}

fn diagnose_mismatch(label: &str, got: &[bf16], expected: &[bf16]) {
    assert_eq!(got.len(), expected.len());
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut mismatches = 0usize;
    let mut samples: Vec<(usize, f32, f32, f32)> = Vec::new();
    for (i, (a, b)) in got.iter().zip(expected.iter()).enumerate() {
        if a == b {
            continue;
        }
        mismatches += 1;
        let af = a.to_f32();
        let bf = b.to_f32();
        let abs = (af - bf).abs();
        let rel = if bf.abs() > 1e-6 { abs / bf.abs() } else { 0.0 };
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(rel);
        if samples.len() < 8 {
            samples.push((i, af, bf, abs));
        }
    }
    eprintln!(
        "{label}: {mismatches}/{} mismatches, max_abs={max_abs:.6}, max_rel={max_rel:.6}",
        got.len(),
    );
    for (i, g, e, abs) in samples {
        eprintln!("  [{i}] got={g} expected={e} |diff|={abs}");
    }
}

#[test]
fn parity_v_erf_bf16() {
    let (lhs, out_ref) = fixture("erf");
    let device = Device::system_default().expect("Metal device");
    let library = KernelLibrary::unary(&device).expect("compile unary.metal");

    let src: Buffer<bf16> = device.upload(&lhs).expect("upload lhs");
    let mut dst: Buffer<bf16> = device.alloc_buffer(S * H).expect("alloc out");
    let mut cmds = device.commands().expect("commands");
    kernels::unary::v_erf_bf16(&mut cmds, &library, &src, &mut dst).expect("dispatch");
    cmds.commit_and_wait().expect("commit");

    // SAFETY: commit_and_wait synchronised; GPU is done with `dst`.
    let out: Vec<bf16> = unsafe { dst.as_mut_slice() }.to_vec();
    assert_eq!(
        out, out_ref,
        "v_Erfbfloat16bfloat16 must be bit-exact vs MLX"
    );
}

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
    let dir = std::env::temp_dir().join(format!("cider-press-unary-parity-{pid}-{nanos}"));
    std::fs::create_dir_all(&dir).expect("mktemp");
    dir
}
