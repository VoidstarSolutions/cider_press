//! Layer-0 attention real-checkpoint integration test.
//!
//! Closes the deferred integration test from branch 11. Loads
//! Qwen2.5-0.5B layer 0 (gated on `CIDER_QWEN_CHECKPOINT_PATH`), drives
//! the full attention pipeline via [`Attention::forward`]
//! (project Q/K/V → rope → `KvCache` → sdpa → output projection), and
//! compares to `mlx_lm.models.qwen2.Attention.__call__`'s output at
//! bf16-compose tolerance.
//!
//! Two sub-tests:
//! - **`attention_layer0_prefill`** (`T=8, offset=0`): exercises the
//!   `affine_qmm_t` path (M=8) through the projection helpers.
//! - **`attention_layer0_decode`** (`T=1, offset=7`): exercises the
//!   `affine_qmv` path through the projections, plus a primed
//!   `KvCache`. Cache priming on the Python side uses the `KVCache.state`
//!   property setter (`mlx_lm` ≥0.21) to seed the cache with `offset` zero
//!   rows before the attention call.
//!
//! Both tests gate on `CIDER_QWEN_CHECKPOINT_PATH`; when unset they
//! no-op with an `eprintln!` message and return PASS. When set, both
//! sub-tests must pass within `ATOL=5e-2, RTOL=5e-2` (np.allclose-style
//! combined bound).
//!
//! The checkpoint must be an MLX-format Qwen2.5-0.5B-Instruct-4bit
//! directory as produced by `mlx_lm.convert`.

#![cfg(target_os = "macos")]
#![allow(clippy::cast_precision_loss)]

use std::fs;
use std::path::{Path, PathBuf};

use cider_press_models::qwen2::{Qwen2Config, attention, load_qwen2_weights};
use cider_press_runtime::{DType, Device, KvCache, Tensor};
use cider_press_test_utils::{dump_mlx_op, read_bf16, tempdir};
use half::bf16;
use safetensors::SafeTensors;

/// Tolerance for the composed full-attention pipeline.
///
/// The chain is: qmv/qmm projection × 4 (Q, K, V, O) + bias-add + reshape +
/// permute + rope + sdpa (itself ~3 ops: `qk_matmul`, softmax, `av_matmul`).
/// Each bf16 round-trip adds up to ~1 ULP, so `5e-2 abs / 5e-2 rel` gives
/// comfortable headroom while still catching real bugs. Earlier branches
/// use tighter bars for fewer ops; this matches the project's "wider for
/// composed chains" convention established in branch 6.
const ATOL: f32 = 5e-2;
const RTOL: f32 = 5e-2;

// ── Helpers ───────────────────────────────────────────────────────────────────

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

fn assert_close(label: &str, got: &[bf16], expected: &[bf16]) {
    assert_eq!(got.len(), expected.len(), "{label}: length mismatch");
    let mut worst_excess = 0.0f32;
    let mut worst_at = 0usize;
    let mut worst_pair = (0.0f32, 0.0f32);
    for (i, (a, b)) in got.iter().zip(expected.iter()).enumerate() {
        if a == b {
            continue;
        }
        let af = a.to_f32();
        let bfv = b.to_f32();
        let diff = (af - bfv).abs();
        let bound = ATOL + RTOL * bfv.abs();
        let excess = diff - bound;
        if excess > worst_excess {
            worst_excess = excess;
            worst_at = i;
            worst_pair = (af, bfv);
        }
    }
    assert!(
        worst_excess <= 0.0,
        "{label}: tolerance exceeded — worst element [{worst_at}] \
         got={} want={} |diff|={} > atol={ATOL} + rtol={RTOL}*|want|",
        worst_pair.0,
        worst_pair.1,
        (worst_pair.0 - worst_pair.1).abs(),
    );
}

// ── Core runner ───────────────────────────────────────────────────────────────

/// Run the full layer-0 attention pipeline.
///
/// `seq_len` is the query sequence length; `offset` is the number of
/// tokens already in the KV cache before this call (0 for pure prefill).
/// On the Python side `dump_mlx_op.py attention_layer0` mirrors the same
/// semantics: `cache=None` for offset=0, a pre-primed `KVCache` otherwise.
fn run_layer0(checkpoint: &Path, seq_len: usize, offset: usize) {
    // ── 1. Generate MLX reference via dump_mlx_op.py ─────────────────────────
    let tmp = tempdir("models-attention-layer0");
    let fixture_path = tmp.join(format!(
        "attention-layer0-seq{seq_len}-off{offset}.safetensors"
    ));
    let checkpoint_str = checkpoint.to_str().expect("checkpoint path is UTF-8");
    let seq_len_s = seq_len.to_string();
    let offset_s = offset.to_string();

    dump_mlx_op(
        &fixture_path,
        &[
            "attention_layer0",
            "--checkpoint",
            checkpoint_str,
            "--seq-len",
            &seq_len_s,
            "--offset",
            &offset_s,
        ],
    );

    // ── 2. Load fixture ───────────────────────────────────────────────────────
    let fixture_bytes = fs::read(&fixture_path).expect("read fixture");
    let st = SafeTensors::deserialize(&fixture_bytes).expect("parse safetensors");
    let x_ref = read_bf16(&st, "x");
    let y_ref = read_bf16(&st, "y");

    // ── 3. Load checkpoint ────────────────────────────────────────────────────
    let config_bytes = fs::read(checkpoint.join("config.json")).expect("read config.json");
    let config = Qwen2Config::from_json_bytes(&config_bytes).expect("parse config.json");

    let archive_bytes =
        fs::read(checkpoint.join("model.safetensors")).expect("read model.safetensors");
    let archive = SafeTensors::deserialize(&archive_bytes).expect("deserialize safetensors");

    let device = Device::shared().expect("system default device");
    let weights = load_qwen2_weights(&archive, &config, &device).expect("load weights");

    let layer0 = &weights.layers[0];
    let head_dim = config.head_dim().expect("head_dim");

    // ── 4. Build the hidden-state tensor from fixture ─────────────────────────
    let hidden = Tensor::from_slice(&device, &x_ref, [1usize, seq_len, config.hidden_size])
        .expect("hidden tensor");

    // ── 5. Build the Attention module from layer-0 weights ────────────────────
    let attn = attention::Attention::from_weights(&layer0.attention, &config)
        .expect("Attention::from_weights");

    // ── 6. KV cache sized for offset + seq_len (+ headroom) ───────────────────
    let max_tokens = offset + seq_len + 4;
    let mut cache = KvCache::new(
        &device,
        max_tokens,
        config.num_key_value_heads,
        head_dim,
        DType::BF16,
    )
    .expect("KvCache::new");

    if offset > 0 {
        // Prime with `offset` zero rows to mirror the Python side's
        // `KVCache.state = (zeros, zeros)` seed. Shape [offset, H_kv, D_h].
        let pad_elems = offset * config.num_key_value_heads * head_dim;
        let zeros: Vec<bf16> = vec![bf16::ZERO; pad_elems];
        let pad_k = Tensor::from_slice(
            &device,
            &zeros,
            [offset, config.num_key_value_heads, head_dim],
        )
        .expect("pad_k");
        let pad_v = Tensor::from_slice(
            &device,
            &zeros,
            [offset, config.num_key_value_heads, head_dim],
        )
        .expect("pad_v");
        cache.update(&pad_k, &pad_v).expect("cache prime");
    }

    // ── 7. RoPE offset == tokens already cached ───────────────────────────────
    let offset_i32 = i32::try_from(offset).expect("offset fits in i32");
    let offset_t = Tensor::from_slice(&device, &[offset_i32], [1usize]).expect("offset tensor");

    // ── 8. Full attention forward (mask=None mirrors Python mask=None) ────────
    let out = attn
        .forward(&hidden, None, &offset_t, &mut cache)
        .expect("Attention::forward");

    // ── 9. Eval and compare ───────────────────────────────────────────────────
    out.eval().expect("eval");
    let got: Vec<bf16> = out.cpu_to_vec().expect("cpu_to_vec");

    let label = format!("attention_layer0 seq_len={seq_len} offset={offset}");
    assert_close(&label, &got, &y_ref);
}

// ── Test cases ────────────────────────────────────────────────────────────────

#[test]
fn attention_layer0_prefill() {
    let Some(checkpoint) = checkpoint_path() else {
        eprintln!(
            "skipped: set CIDER_QWEN_CHECKPOINT_PATH to a local \
             Qwen2.5-0.5B-Instruct-4bit MLX checkout to run this test"
        );
        return;
    };
    run_layer0(&checkpoint, 8, 0);
}

#[test]
fn attention_layer0_decode() {
    let Some(checkpoint) = checkpoint_path() else {
        eprintln!(
            "skipped: set CIDER_QWEN_CHECKPOINT_PATH to a local \
             Qwen2.5-0.5B-Instruct-4bit MLX checkout to run this test"
        );
        return;
    };
    run_layer0(&checkpoint, 1, 7);
}
