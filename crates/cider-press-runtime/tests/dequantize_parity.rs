//! Runtime-level parity tests for `Tensor::dequantize_affine` and the
//! composed `gather → dequantize` pipeline used by `nn::embed_tokens`.
//!
//! Two cases:
//!
//! 1. `dequantize_full_table` — `Tensor::dequantize_affine` on the
//!    components of a `QuantizedWeight` matches `mx.dequantize` on the
//!    same triple. This is the kernels-crate parity again, but
//!    threaded through the runtime's lazy graph.
//! 2. `gather_then_dequantize` — gather rows out of the quantized
//!    triple (using `Tensor::gather` on both `U32` `w_q` and `BF16`
//!    scales/biases), then dequantize the gathered rows. The result
//!    must equal `dequantize_full_table` indexed by the same row IDs.

#![cfg(target_os = "macos")]
#![allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]

use std::path::{Path, PathBuf};
use std::process::Command;

use cider_press_runtime::{Device, Quantization, QuantizedWeight, Tensor};
use half::bf16;
use safetensors::SafeTensors;

const ROWS: usize = 128;
const COLS: usize = 64;
const GROUP_SIZE: u32 = 64;
const BITS: u8 = 4;

#[test]
fn dequantize_full_table_matches_mlx() {
    let f = fixture();
    let device = Device::system_default().expect("device");
    let qw = QuantizedWeight::from_bytes(
        &device,
        [ROWS, COLS],
        Quantization::Q4_GS64,
        &u32_to_bytes(&f.w_q),
        &bf16_to_bytes(&f.scales),
        &bf16_to_bytes(&f.biases),
    )
    .expect("construct QuantizedWeight");

    let (w_q, scales, biases) = qw.components();
    let out = Tensor::dequantize_affine(&w_q, &scales, &biases, GROUP_SIZE, BITS)
        .expect("schedule dequantize");
    out.eval().expect("eval");
    assert_eq!(out.shape().dims(), &[ROWS, COLS]);
    let got: Vec<bf16> = out.cpu_to_vec().expect("dense out");
    assert_eq!(
        got, f.out,
        "Tensor::dequantize_affine must match mx.dequantize bit-exactly"
    );
}

#[test]
fn gather_then_dequantize_matches_dequantize_then_gather() {
    let f = fixture();
    let device = Device::system_default().expect("device");
    let qw = QuantizedWeight::from_bytes(
        &device,
        [ROWS, COLS],
        Quantization::Q4_GS64,
        &u32_to_bytes(&f.w_q),
        &bf16_to_bytes(&f.scales),
        &bf16_to_bytes(&f.biases),
    )
    .expect("construct QuantizedWeight");

    // Pick a few row IDs — sparse, out-of-order, with a repeat to make
    // sure the composition handles duplicate indices correctly.
    let indices_host: Vec<u32> = vec![3, 0, 127, 64, 64, 1];
    let n = indices_host.len();
    let indices = Tensor::from_slice(&device, &indices_host, [n]).expect("indices");

    let (w_q, scales, biases) = qw.components();
    let gw = w_q.gather(&indices).expect("gather w_q");
    let gs = scales.gather(&indices).expect("gather scales");
    let gb = biases.gather(&indices).expect("gather biases");
    let gathered_dq = Tensor::dequantize_affine(&gw, &gs, &gb, GROUP_SIZE, BITS)
        .expect("schedule gathered dequantize");
    gathered_dq.eval().expect("eval");
    assert_eq!(gathered_dq.shape().dims(), &[n, COLS]);
    let gathered_out: Vec<bf16> = gathered_dq.cpu_to_vec().expect("dense out");

    // Reference: dequantize the entire table, then gather the same
    // rows in Rust. Bit-exact equivalence proves both orderings
    // produce identical output (which they must, since affine
    // dequantize is per-row).
    let mut expected: Vec<bf16> = Vec::with_capacity(n * COLS);
    for &row in &indices_host {
        let start = row as usize * COLS;
        expected.extend_from_slice(&f.out[start..start + COLS]);
    }
    assert_eq!(
        gathered_out, expected,
        "gather-then-dequantize must equal dequantize-then-gather"
    );
}

// ---------------------------------------------------------------------
// fixture
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
        .arg(GROUP_SIZE.to_string())
        .arg("--bits")
        .arg(BITS.to_string())
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

fn u32_to_bytes(v: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for &x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

fn bf16_to_bytes(v: &[bf16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 2);
    for &x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

fn tempdir() -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("SystemTime")
        .as_nanos();
    let pid = std::process::id();
    let dir =
        std::env::temp_dir().join(format!("cider-press-runtime-dequantize-parity-{pid}-{nanos}"));
    std::fs::create_dir_all(&dir).expect("mktemp");
    dir
}
