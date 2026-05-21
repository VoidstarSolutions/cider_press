//! Parity tests for the composed [`nn::embed_tokens`] against MLX.
//!
//! Two cases:
//! - Synthetic: generate a random quantized embedding table + indices
//!   via `dump_mlx_op.py embed_tokens`, compare cider-press output
//!   bit-exactly to `mx.take(mx.dequantize(...), indices, axis=0)`.
//!   The composition is per-row identical to MLX's, so bit-exact is
//!   the right bar.
//! - Real-checkpoint (gated on `CIDER_QWEN_CHECKPOINT_PATH`): load
//!   Qwen2.5-0.5B's `embed_tokens` via `load_qwen2_weights`, pull a
//!   handful of token IDs through `nn::embed_tokens`, and validate
//!   against a CPU-computed per-row dequantize over the same loaded
//!   bytes. Catches loader/dispatch wiring drift on real weights.

#![cfg(target_os = "macos")]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
)]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use cider_press_models::nn::embed_tokens;
use cider_press_models::qwen2::{Qwen2Config, load_qwen2_weights};
use cider_press_runtime::{Device, Quantization, QuantizedWeight, Tensor};
use half::bf16;
use safetensors::SafeTensors;

const VOCAB: usize = 256;
const HIDDEN: usize = 128;
const N_INDICES: usize = 6;
const GROUP_SIZE: u32 = 64;
const BITS: u8 = 4;

#[test]
fn embed_tokens_matches_mlx_bit_exact() {
    let f = synthetic_fixture();
    let device = Device::system_default().expect("device");
    let qw = QuantizedWeight::from_bytes(
        &device,
        [VOCAB, HIDDEN],
        Quantization::Q4_GS64,
        &u32_to_bytes(&f.w_q),
        &bf16_to_bytes(&f.scales),
        &bf16_to_bytes(&f.biases),
    )
    .expect("construct QuantizedWeight");
    let indices = Tensor::from_slice(&device, &f.indices, [N_INDICES]).expect("indices");

    let out = embed_tokens(&indices, &qw).expect("schedule embed_tokens");
    out.eval().expect("eval");
    assert_eq!(out.shape().dims(), &[N_INDICES, HIDDEN]);
    let got: Vec<bf16> = out.cpu_to_vec().expect("dense out");
    assert_eq!(
        got, f.out,
        "nn::embed_tokens must match mx.take(mx.dequantize(...), indices) bit-exactly",
    );
}

#[test]
fn real_checkpoint_embed_tokens_matches_cpu_dequantize() {
    let Some(dir) = checkpoint_path() else {
        eprintln!(
            "skipping real_checkpoint_embed_tokens_matches_cpu_dequantize: \
             CIDER_QWEN_CHECKPOINT_PATH not set"
        );
        return;
    };
    let device = Device::system_default().expect("device");
    let config_bytes = fs::read(dir.join("config.json")).expect("read config.json");
    let config = Qwen2Config::from_json_bytes(&config_bytes).expect("parse config");
    let archive_path = dir.join("model.safetensors");
    let archive_bytes = fs::read(&archive_path).expect("read safetensors");
    let archive = SafeTensors::deserialize(&archive_bytes).expect("parse archive");
    let weights = load_qwen2_weights(&archive, &config, &device).expect("load weights");

    // A few token IDs chosen to cover a span of the vocabulary.
    let indices_host: Vec<u32> = vec![0, 1, 100, 5000, 100_000, config.vocab_size as u32 - 1];
    let n = indices_host.len();
    let indices = Tensor::from_slice(&device, &indices_host, [n]).expect("indices");

    let out = embed_tokens(&indices, &weights.embed).expect("schedule");
    out.eval().expect("eval");
    let got: Vec<bf16> = out.cpu_to_vec().expect("dense out");

    // CPU reference: dequantize each selected row manually from the
    // loaded raw bytes. Validates the composition on real weights
    // without depending on a separate MLX dump at the model scale.
    let (w_bytes, s_bytes, b_bytes) = weights.embed.component_bytes();
    let hidden = config.hidden_size as usize;
    let group_size = GROUP_SIZE as usize;
    let bits = BITS as usize;
    let packed_per_row = hidden * bits / 32; // u32 count per row
    let groups_per_row = hidden / group_size;
    let mut expected: Vec<bf16> = Vec::with_capacity(n * hidden);
    for &row in &indices_host {
        let row = row as usize;
        let w_off = row * packed_per_row * 4;
        let s_off = row * groups_per_row * 2;
        let row_w: Vec<u32> = w_bytes[w_off..w_off + packed_per_row * 4]
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let row_s: Vec<bf16> = s_bytes[s_off..s_off + groups_per_row * 2]
            .chunks_exact(2)
            .map(|c| bf16::from_le_bytes([c[0], c[1]]))
            .collect();
        let row_b: Vec<bf16> = b_bytes[s_off..s_off + groups_per_row * 2]
            .chunks_exact(2)
            .map(|c| bf16::from_le_bytes([c[0], c[1]]))
            .collect();
        expected.extend(cpu_dequantize_row(&row_w, &row_s, &row_b, group_size));
    }
    assert_eq!(got.len(), expected.len());
    assert_eq!(
        got, expected,
        "nn::embed_tokens on real Qwen weights must equal CPU per-row dequantize",
    );
}

// ---------------------------------------------------------------------
// CPU reference: int4 affine dequantize, one row at a time.
//
// Mirrors MLX's `affine_dequantize` MSL kernel for bits=4: each u32
// holds 8 int4 nibbles (lowest nibble = lane 0), group_size nibbles
// share one scale + bias. `out[i] = scale * nibble + bias`.
// ---------------------------------------------------------------------

fn cpu_dequantize_row(w: &[u32], scales: &[bf16], biases: &[bf16], group_size: usize) -> Vec<bf16> {
    let logical_len = w.len() * 8;
    let mut out = Vec::with_capacity(logical_len);
    for (i, &packed) in w.iter().enumerate() {
        for lane in 0..8 {
            let nibble = ((packed >> (4 * lane)) & 0x0f) as f32;
            let logical_idx = i * 8 + lane;
            let group = logical_idx / group_size;
            let s = scales[group].to_f32();
            let b = biases[group].to_f32();
            out.push(bf16::from_f32(s * nibble + b));
        }
    }
    out
}

// ---------------------------------------------------------------------
// synthetic fixture
// ---------------------------------------------------------------------

struct Synthetic {
    w_q: Vec<u32>,
    scales: Vec<bf16>,
    biases: Vec<bf16>,
    indices: Vec<u32>,
    out: Vec<bf16>,
}

fn synthetic_fixture() -> Synthetic {
    let tmp = tempdir();
    let path = tmp.join("embed_tokens.safetensors");
    dump_mlx_embed_tokens(&path);
    let bytes = fs::read(&path).expect("read fixture");
    let st = SafeTensors::deserialize(&bytes).expect("parse safetensors");
    Synthetic {
        w_q: read_u32(&st, "w_q"),
        scales: read_bf16(&st, "scales"),
        biases: read_bf16(&st, "biases"),
        indices: read_u32(&st, "indices"),
        out: read_bf16(&st, "out"),
    }
}

fn dump_mlx_embed_tokens(out: &Path) {
    let script = workspace_root().join("scripts").join("dump_mlx_op.py");
    let status = Command::new("uv")
        .arg("run")
        .arg(&script)
        .arg("--output")
        .arg(out)
        .arg("--seed")
        .arg("0")
        .arg("embed_tokens")
        .arg("--vocab")
        .arg(VOCAB.to_string())
        .arg("--hidden")
        .arg(HIDDEN.to_string())
        .arg("--n-indices")
        .arg(N_INDICES.to_string())
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
            "failed to invoke `uv run {}`: {err}. This test requires `uv` on PATH.",
            script.display()
        ),
    };
    assert!(status.success(), "dump_mlx_op.py embed_tokens exited {status}");
}

// ---------------------------------------------------------------------
// boilerplate
// ---------------------------------------------------------------------

fn checkpoint_path() -> Option<PathBuf> {
    let raw = std::env::var("CIDER_QWEN_CHECKPOINT_PATH").ok()?;
    let path = fs::canonicalize(&raw)
        .unwrap_or_else(|err| panic!("CIDER_QWEN_CHECKPOINT_PATH={raw} does not resolve: {err}"));
    assert!(
        path.is_dir(),
        "CIDER_QWEN_CHECKPOINT_PATH={} is not a directory",
        path.display(),
    );
    Some(path)
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
    let dir = std::env::temp_dir().join(format!("cider-press-embed-tokens-{pid}-{nanos}"));
    fs::create_dir_all(&dir).expect("mktemp");
    dir
}
