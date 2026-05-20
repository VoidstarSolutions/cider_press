//! Parity test for `binary.metal` Add dispatch (branch 4 commit 2).
//!
//! Unlike the Stage-3/4 parity tests, the fixture is generated at test
//! time by shelling out to `uv run scripts/dump_mlx_op.py add ...`
//! against MLX's own reference implementation. Branch 4 keeps fixture
//! bytes out of the repo so the activation-dump harness stays the
//! single source of truth for the op's reference output across both
//! the kernels-crate parity test and the later models-crate parity
//! test (commit 4).
//!
//! Two cases:
//!
//! - **vv** — same-shape `[1, S, H]` bf16 add, exercises `vv_Addbfloat16`.
//! - **broadcast** — `[1, S, H] + [H]` bf16 add, exercises
//!   `g3_Addbfloat16` with rhs strides `[0, 0, 1]`. This is the
//!   per-channel-bias shape branch 5 (rmsnorm) and the Q/K/V/O Linear
//!   biases will hit.
//!
//! Bit-exact: bf16 add is a single rounded op, so MLX and us must
//! agree on every byte. If this drifts to a tolerance check, something
//! is wrong with the dispatch — investigate before relaxing.

#![cfg(target_os = "macos")]

use std::path::{Path, PathBuf};
use std::process::Command;

use cider_press_kernels::{Buffer, Device, KernelLibrary, kernels};
use half::bf16;
use safetensors::SafeTensors;

const S: usize = 8;
const H: usize = 896;

fn dump_mlx_add(out: &Path, lhs_shape: &str, rhs_shape: &str) {
    let script = workspace_root().join("scripts").join("dump_mlx_op.py");
    let status = Command::new("uv")
        .arg("run")
        .arg(&script)
        .arg("--output")
        .arg(out)
        .arg("--seed")
        .arg("0")
        .arg("add")
        .arg("--lhs-shape")
        .arg(lhs_shape)
        .arg("--rhs-shape")
        .arg(rhs_shape)
        .arg("--dtype")
        .arg("bf16")
        .status();
    let status = match status {
        Ok(s) => s,
        Err(err) => panic!(
            "failed to invoke `uv run {}`: {err}. \
             This test requires `uv` (https://docs.astral.sh/uv) on PATH \
             so MLX can generate the parity fixture; CI installs it via \
             astral-sh/setup-uv.",
            script.display()
        ),
    };
    assert!(status.success(), "dump_mlx_op.py add exited {status}");
}

fn workspace_root() -> PathBuf {
    // CARGO_MANIFEST_DIR = crates/cider-press-kernels; go up two.
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

#[test]
fn parity_vv_add_bf16() {
    let tmp = tempdir();
    let fixture = tmp.join("add_vv_bf16.safetensors");
    dump_mlx_add(&fixture, &format!("1,{S},{H}"), &format!("1,{S},{H}"));

    let bytes = std::fs::read(&fixture).expect("read fixture");
    let st = SafeTensors::deserialize(&bytes).expect("parse safetensors");
    let lhs = read_bf16(&st, "lhs");
    let rhs = read_bf16(&st, "rhs");
    let out_ref = read_bf16(&st, "out");
    let n = S * H;
    assert_eq!(lhs.len(), n);
    assert_eq!(rhs.len(), n);
    assert_eq!(out_ref.len(), n);

    let device = Device::system_default().expect("Metal device");
    let library = KernelLibrary::binary(&device).expect("compile binary.metal");

    let a: Buffer<bf16> = device.upload(&lhs).expect("upload lhs");
    let b: Buffer<bf16> = device.upload(&rhs).expect("upload rhs");
    let mut c: Buffer<bf16> = device.alloc_buffer(n).expect("alloc out");

    let mut cmds = device.commands().expect("commands");
    kernels::binary::vv_add_bf16(&mut cmds, &library, &a, &b, &mut c).expect("dispatch");
    cmds.commit_and_wait().expect("commit");

    // SAFETY: commit_and_wait synchronised; GPU is done with `c`.
    let out: Vec<bf16> = unsafe { c.as_mut_slice() }.to_vec();
    assert_eq!(out, out_ref, "vv_Addbfloat16 must be bit-exact vs MLX");
}

#[test]
fn parity_g3_add_bf16_broadcast() {
    let tmp = tempdir();
    let fixture = tmp.join("add_bcast_bf16.safetensors");
    dump_mlx_add(&fixture, &format!("1,{S},{H}"), &format!("{H}"));

    let bytes = std::fs::read(&fixture).expect("read fixture");
    let st = SafeTensors::deserialize(&bytes).expect("parse safetensors");
    let lhs = read_bf16(&st, "lhs"); // [1, S, H]
    let rhs = read_bf16(&st, "rhs"); // [H]
    let out_ref = read_bf16(&st, "out"); // [1, S, H]
    let n = S * H;
    assert_eq!(lhs.len(), n);
    assert_eq!(rhs.len(), H);
    assert_eq!(out_ref.len(), n);

    let device = Device::system_default().expect("Metal device");
    let library = KernelLibrary::binary(&device).expect("compile binary.metal");

    let a: Buffer<bf16> = device.upload(&lhs).expect("upload lhs");
    let b: Buffer<bf16> = device.upload(&rhs).expect("upload rhs");
    let mut c: Buffer<bf16> = device.alloc_buffer(n).expect("alloc out");

    // Padded to rank 3, both operands. lhs is contiguous over the
    // logical [1, S, H] shape; rhs is broadcast across the leading two
    // axes so its strides on those axes are zero.
    #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
    let a_strides: [i64; 3] = [(S * H) as i64, H as i64, 1];
    let b_strides: [i64; 3] = [0, 0, 1];
    #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
    let shape: [i32; 3] = [1, S as i32, H as i32];

    let mut cmds = device.commands().expect("commands");
    kernels::binary::g3_add_bf16(
        &mut cmds, &library, &a, &b, &mut c, a_strides, b_strides, shape,
    )
    .expect("dispatch");
    cmds.commit_and_wait().expect("commit");

    // SAFETY: commit_and_wait synchronised; GPU is done with `c`.
    let out: Vec<bf16> = unsafe { c.as_mut_slice() }.to_vec();
    assert_eq!(
        out, out_ref,
        "g3_Addbfloat16 broadcast must be bit-exact vs MLX"
    );
}

/// Cargo-test-local tempdir without pulling in the `tempfile` crate.
/// Lives under `target/` so `cargo clean` collects it; randomised per
/// test invocation to keep parallel test threads from colliding.
fn tempdir() -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("SystemTime")
        .as_nanos();
    let pid = std::process::id();
    let dir = std::env::temp_dir().join(format!("cider-press-binary-parity-{pid}-{nanos}"));
    std::fs::create_dir_all(&dir).expect("mktemp");
    dir
}
