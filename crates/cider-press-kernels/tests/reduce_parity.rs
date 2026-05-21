//! Parity test for `reduce.metal` last-axis Sum dispatch (branch 5).
//!
//! Generates fixtures via `uv run scripts/dump_mlx_op.py row_sum`
//! and asserts bit-exact equality vs MLX for bf16 at `[1, 8, 896]`
//! reducing axis -1.
//!
//! Bit-exactness rationale: the kernels-crate `row_reduce_sum_bf16`
//! dispatches the same MLX `row_reduce_looped_1_reduce_sumbfloat16`
//! kernel that `mx.sum` does. The accumulation order is deterministic
//! within the kernel (simdgroup tree-reduce → threadgroup tree-reduce
//! → write), so identical kernel + identical inputs ⇒ identical bytes.
//!
//! If this test starts failing, suspect either dispatch transcription
//! drift (our bind order or grid math) or an MLX-side change to the
//! reduce kernel itself.

#![cfg(target_os = "macos")]

use std::path::{Path, PathBuf};
use std::process::Command;

use cider_press_kernels::{Buffer, Device, KernelLibrary, kernels};
use half::bf16;
use safetensors::SafeTensors;

const S: usize = 8;
const H: usize = 896;

#[test]
fn parity_row_reduce_sum_bf16() {
    let tmp = tempdir();
    let fixture = tmp.join("row_sum.safetensors");
    dump_mlx_row_sum(&fixture, &format!("1,{S},{H}"));

    let bytes = std::fs::read(&fixture).expect("read fixture");
    let st = SafeTensors::deserialize(&bytes).expect("parse safetensors");
    let lhs = read_bf16(&st, "lhs");
    let out_ref = read_bf16(&st, "out");
    assert_eq!(lhs.len(), S * H);
    assert_eq!(out_ref.len(), S);

    let device = Device::system_default().expect("Metal device");
    let library = KernelLibrary::reduce(&device).expect("compile reduce.metal");

    let src: Buffer<bf16> = device.upload(&lhs).expect("upload lhs");
    let mut dst: Buffer<bf16> = device.alloc_buffer(S).expect("alloc out");

    let mut cmds = device.commands().expect("commands");
    kernels::reduce::row_reduce_sum_bf16(&mut cmds, &library, &src, &[1, S, H], &mut dst)
        .expect("dispatch");
    cmds.commit_and_wait().expect("commit");

    // SAFETY: commit_and_wait synchronised.
    let out: Vec<bf16> = unsafe { dst.as_mut_slice() }.to_vec();
    assert_eq!(
        out, out_ref,
        "row_reduce_sum_bf16 must be bit-exact vs mx.sum"
    );
}

fn dump_mlx_row_sum(out: &Path, lhs_shape: &str) {
    let script = workspace_root().join("scripts").join("dump_mlx_op.py");
    let status = Command::new("uv")
        .arg("run")
        .arg(&script)
        .arg("--output")
        .arg(out)
        .arg("--seed")
        .arg("0")
        .arg("row_sum")
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
    assert!(status.success(), "dump_mlx_op.py row_sum exited {status}");
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
    let dir = std::env::temp_dir().join(format!("cider-press-reduce-parity-{pid}-{nanos}"));
    std::fs::create_dir_all(&dir).expect("mktemp");
    dir
}
