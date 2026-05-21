//! Parity test for `affine_dequantize_bf16_gs64_b4` against
//! `mx.dequantize`.
//!
//! Same Qwen2 q4/gs64 config as Stage-4 qmv. The kernel is a pure
//! deterministic function of `(w_q, scales, biases)` so the bar is
//! bit-exact.

#![cfg(target_os = "macos")]

use std::path::{Path, PathBuf};
use std::process::Command;

use cider_press_kernels::kernels::dequantize::affine_dequantize_bf16_gs64_b4;
use cider_press_kernels::{Buffer, Device, KernelLibrary};
use half::bf16;
use safetensors::SafeTensors;

const ROWS: usize = 128;
const COLS: usize = 64; // multiple of group_size=64

#[test]
fn affine_dequantize_bf16_gs64_b4_matches_mlx_bit_exact() {
    let f = fixture();
    let device = Device::system_default().expect("Metal device");
    let library = KernelLibrary::quantized(&device).expect("compile quantized.metal");

    let w_q: Buffer<u32> = device.upload(&f.w_q).expect("upload w_q");
    let scales: Buffer<bf16> = device.upload(&f.scales).expect("upload scales");
    let biases: Buffer<bf16> = device.upload(&f.biases).expect("upload biases");
    let mut out: Buffer<bf16> = device.alloc_buffer(ROWS * COLS).expect("alloc out");

    let mut cmds = device.commands().expect("commands");
    affine_dequantize_bf16_gs64_b4(&mut cmds, &library, &w_q, &scales, &biases, &mut out)
        .expect("dispatch dequantize");
    cmds.commit_and_wait().expect("commit");

    // SAFETY: commit_and_wait synchronised; GPU is done with `out`.
    let got: Vec<bf16> = unsafe { out.as_mut_slice() }.to_vec();
    assert_eq!(
        got, f.out,
        "affine_dequantize must be bit-exact vs mx.dequantize"
    );
}

// ---------------------------------------------------------------------
// fixture / harness boilerplate
// ---------------------------------------------------------------------

struct Fixture {
    w_q: Vec<u32>,
    scales: Vec<bf16>,
    biases: Vec<bf16>,
    out: Vec<bf16>,
}

fn fixture() -> Fixture {
    let tmp = tempdir();
    let path = tmp.join("dequantize.safetensors");
    dump_mlx(&path);
    let bytes = std::fs::read(&path).expect("read fixture");
    let st = SafeTensors::deserialize(&bytes).expect("parse safetensors");
    Fixture {
        w_q: read_u32(&st, "w_q"),
        scales: read_bf16(&st, "scales"),
        biases: read_bf16(&st, "biases"),
        out: read_bf16(&st, "out"),
    }
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
        .arg("dequantize")
        .arg("--rows")
        .arg(ROWS.to_string())
        .arg("--cols")
        .arg(COLS.to_string())
        .arg("--group-size")
        .arg("64")
        .arg("--bits")
        .arg("4")
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
    assert!(status.success(), "dump_mlx_op.py dequantize exited {status}");
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
    let dir = std::env::temp_dir().join(format!("cider-press-dequantize-parity-{pid}-{nanos}"));
    std::fs::create_dir_all(&dir).expect("mktemp");
    dir
}
