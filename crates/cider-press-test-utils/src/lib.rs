//! Shared helpers for cider-press MLX-parity tests.
//!
//! Every parity test across `cider-press-kernels`, `cider-press-runtime`,
//! and `cider-press-models` needs the same four primitives: locate the
//! workspace root (to find `scripts/dump_mlx_op.py`), create a per-
//! invocation tempdir, shell out to `uv run` to materialise a fixture
//! via MLX, and read a bf16 tensor out of the resulting safetensors.
//! Consolidating them here keeps a single source of truth for the
//! invocation contract (`uv` on PATH, `--seed 0`, exit-status check).
//!
//! Dev-only — the crate has `publish = false` and is consumed
//! exclusively via `[dev-dependencies]`.

use std::path::{Path, PathBuf};
use std::process::Command;

use half::bf16;
use safetensors::SafeTensors;

/// Workspace root, derived from the dependent crate's
/// `CARGO_MANIFEST_DIR`. `cider-press-test-utils` lives at
/// `crates/cider-press-test-utils/`; the workspace root is two
/// ancestors up at compile time of *this* crate, which is the same
/// physical directory regardless of which downstream crate links it.
#[must_use]
pub fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("workspace root")
        .to_path_buf()
}

/// Per-invocation tempdir under `std::env::temp_dir()`. `tag` becomes
/// part of the directory name so concurrent test suites stay clear of
/// each other; pid + nanos suffix the path so parallel cargo runs
/// don't collide.
#[must_use]
pub fn tempdir(tag: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("SystemTime")
        .as_nanos();
    let pid = std::process::id();
    let dir = std::env::temp_dir().join(format!("cider-press-{tag}-{pid}-{nanos}"));
    std::fs::create_dir_all(&dir).expect("mktemp");
    dir
}

/// Read a named tensor from `st` as bf16. Panics if the tensor is
/// missing or stored under a non-bf16 dtype.
#[must_use]
pub fn read_bf16(st: &SafeTensors, name: &str) -> Vec<bf16> {
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

/// Read a named tensor from `st` as u32. Panics if the tensor is
/// missing or stored under a non-u32 dtype.
#[must_use]
pub fn read_u32(st: &SafeTensors, name: &str) -> Vec<u32> {
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

/// Invoke `scripts/dump_mlx_op.py` with `--output <out> --seed 0`
/// followed by the caller-supplied args (typically the op name plus
/// its op-specific flags like `--lhs-shape`, `--rhs-shape`, `--dtype`,
/// `--eps`). Panics if `uv` is not on PATH or the script exits
/// non-zero.
pub fn dump_mlx_op(out: &Path, args: &[&str]) {
    let script = workspace_root().join("scripts").join("dump_mlx_op.py");
    let status = Command::new("uv")
        .arg("run")
        .arg(&script)
        .arg("--output")
        .arg(out)
        .arg("--seed")
        .arg("0")
        .args(args)
        .status();
    let status = match status {
        Ok(s) => s,
        Err(err) => panic!(
            "failed to invoke `uv run {}`: {err}. This test requires \
             `uv` (https://docs.astral.sh/uv) on PATH so MLX can \
             generate the parity fixture; CI installs it via \
             astral-sh/setup-uv.",
            script.display()
        ),
    };
    assert!(status.success(), "dump_mlx_op.py {args:?} exited {status}");
}
