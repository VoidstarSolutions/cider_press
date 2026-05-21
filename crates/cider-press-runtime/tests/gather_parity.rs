//! Runtime-level parity test for `Tensor::gather` against MLX's
//! `mx.take(src, indices, axis=0)`.
//!
//! The kernels-crate test (`kernels-crate::tests::gather_parity`) has
//! already validated the JIT-assembled gather dispatch bit-exactly;
//! this test validates that the runtime threads inputs + shape
//! metadata correctly into that dispatch — including the per-call
//! metadata buffers (`src_shape`, `slice_sizes`, etc.) which the
//! runtime synthesizes rather than the caller.

#![cfg(target_os = "macos")]
#![allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]

use std::path::{Path, PathBuf};
use std::process::Command;

use cider_press_runtime::{Device, Tensor};
use half::bf16;
use safetensors::SafeTensors;

const VOCAB: usize = 128;
const HIDDEN: usize = 64;
const N_INDICES: usize = 8;

#[test]
fn gather_through_runtime_matches_mlx_bit_exact() {
    let (src_host, indices_host, out_ref) = fixture();
    let device = Device::system_default().expect("device");

    let src = Tensor::from_slice(&device, &src_host, [VOCAB, HIDDEN]).expect("src");
    let indices = Tensor::from_slice(&device, &indices_host, [N_INDICES]).expect("indices");

    let out = src.gather(&indices).expect("schedule gather");
    out.eval().expect("eval");
    assert_eq!(out.shape().dims(), &[N_INDICES, HIDDEN]);
    let got: Vec<bf16> = out.cpu_to_vec().expect("dense out");
    assert_eq!(
        got, out_ref,
        "Tensor::gather must match mx.take(src, indices, axis=0) bit-exactly",
    );
}

// ---------------------------------------------------------------------
// helpers (mirror the kernels-crate parity test exactly)
// ---------------------------------------------------------------------

fn fixture() -> (Vec<bf16>, Vec<u32>, Vec<bf16>) {
    let tmp = tempdir();
    let path = tmp.join("gather.safetensors");
    dump_mlx(&path);
    let bytes = std::fs::read(&path).expect("read fixture");
    let st = SafeTensors::deserialize(&bytes).expect("parse safetensors");
    (
        read_bf16(&st, "src"),
        read_u32(&st, "indices"),
        read_bf16(&st, "out"),
    )
}

fn dump_mlx(out: &Path) {
    let script = workspace_root().join("scripts").join("dump_mlx_op.py");
    let status = Command::new("uv")
        .arg("run")
        .arg(&script)
        .arg("--output")
        .arg(out)
        .arg("--seed")
        .arg("0")
        .arg("gather")
        .arg("--vocab")
        .arg(VOCAB.to_string())
        .arg("--hidden")
        .arg(HIDDEN.to_string())
        .arg("--n-indices")
        .arg(N_INDICES.to_string())
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
    assert!(status.success(), "dump_mlx_op.py gather exited {status}");
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

fn read_u32(st: &SafeTensors, name: &str) -> Vec<u32> {
    let view = st
        .tensor(name)
        .unwrap_or_else(|e| panic!("tensor {name}: {e}"));
    assert_eq!(view.dtype(), safetensors::Dtype::U32);
    let bytes = view.data();
    assert!(bytes.len() % 4 == 0);
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn tempdir() -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("SystemTime")
        .as_nanos();
    let pid = std::process::id();
    let dir =
        std::env::temp_dir().join(format!("cider-press-runtime-gather-parity-{pid}-{nanos}"));
    std::fs::create_dir_all(&dir).expect("mktemp");
    dir
}
