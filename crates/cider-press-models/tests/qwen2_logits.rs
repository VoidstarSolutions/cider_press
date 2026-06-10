//! Full-model `Qwen2Model` MLX-LM parity test.
//!
//! Loads Qwen2.5-0.5B (gated on `CIDER_QWEN_CHECKPOINT_PATH`), tokenizes a
//! fixed short prompt via the MLX dump harness, runs prefill through both
//! `Qwen2Model::forward` and `mlx_lm.models.qwen2.Model`, and compares the
//! [1, T, vocab] logits at bf16-compose tolerance.
//!
//! Skips with a PASS + `eprintln!` when the env var is unset.

#![cfg(target_os = "macos")]
#![allow(clippy::cast_precision_loss)]

use std::fs;
use std::path::{Path, PathBuf};

use cider_press_models::qwen2::{Qwen2Config, Qwen2Model, load_qwen2_weights};
use cider_press_runtime::{DType, Device, KvCache, Tensor};
use cider_press_test_utils::{dump_mlx_op, read_bf16, read_u32, tempdir};
use half::bf16;
use safetensors::SafeTensors;

// 24-layer composed chain. Tolerance is deliberately loose: this test
// catches structural regressions (wrong dispatch, shape bugs, dead
// layers) but not numerical precision regressions. Bf16 ULP drift in
// `qmm_t` (M>1 path; ~0.1% rel_l2 vs MLX) gets amplified
// ~200x by the composed-bf16 SDPA chain (no fused kernel yet) and
// compounds across 24 layers. Empirically the
// worst-element drift on Qwen2.5-0.5B prefill is ~2.7 abs at logits
// magnitudes near 1, so 3.0 abs / 1.0 rel passes comfortably while
// still failing fast on real bugs. Tighten when the fused SDPA kernel
// lands.
const ATOL: f32 = 3.0;
const RTOL: f32 = 1.0;

const PROMPT: &str = "The capital of France is";

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
        let af = a.to_f32();
        let bfv = b.to_f32();
        if !af.is_finite() || !bfv.is_finite() {
            // NaN/Inf would otherwise short-circuit the tolerance math
            // (NaN comparisons fall back to false) — force a failure.
            worst_excess = f32::INFINITY;
            worst_at = i;
            worst_pair = (af, bfv);
            break;
        }
        if a == b {
            continue;
        }
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

fn run_qwen2_logits(checkpoint: &Path) {
    let tmp = tempdir("models-qwen2-logits");
    let fixture_path = tmp.join("qwen2-logits.safetensors");
    let checkpoint_str = checkpoint.to_str().expect("checkpoint path is UTF-8");

    dump_mlx_op(
        &fixture_path,
        &[
            "qwen2_logits",
            "--checkpoint",
            checkpoint_str,
            "--prompt",
            PROMPT,
        ],
    );

    let fixture_bytes = fs::read(&fixture_path).expect("read fixture");
    let st = SafeTensors::deserialize(&fixture_bytes).expect("parse safetensors");
    let input_ids_ref = read_u32(&st, "input_ids");
    let logits_ref = read_bf16(&st, "logits");

    let config_bytes = fs::read(checkpoint.join("config.json")).expect("read config.json");
    let config = Qwen2Config::from_json_bytes(&config_bytes).expect("parse config.json");
    let head_dim = config.head_dim().expect("head_dim");
    let vocab_size = config.vocab_size;
    let num_layers = config.num_hidden_layers;
    let n_kv_heads = config.num_key_value_heads;

    // Token count is whatever the tokenizer produced; infer from the
    // dumped fixture rather than the prompt's character count.
    let t = input_ids_ref.len();
    assert_eq!(
        logits_ref.len(),
        t * vocab_size,
        "fixture logits length should be T * vocab_size; got {} for T={t} vocab={vocab_size}",
        logits_ref.len()
    );

    let archive_bytes =
        fs::read(checkpoint.join("model.safetensors")).expect("read model.safetensors");
    let archive = SafeTensors::deserialize(&archive_bytes).expect("deserialize safetensors");

    let device = Device::shared().expect("system default device");
    let weights = load_qwen2_weights(&archive, &config, &device).expect("load weights");
    let model = Qwen2Model::from_weights(weights, config.clone()).expect("build Qwen2Model");

    let input_ids = Tensor::from_slice(&device, &input_ids_ref, [1usize, t]).expect("input_ids");
    let offset_t = Tensor::from_slice(&device, &[0i32], [1usize]).expect("offset tensor");

    // T tokens go into the cache during prefill; size slabs accordingly with
    // a little headroom.
    let max_tokens = t + 4;
    let mut caches: Vec<KvCache> = (0..num_layers)
        .map(|_| {
            KvCache::new(&device, max_tokens, n_kv_heads, head_dim, DType::BF16).expect("kv cache")
        })
        .collect();

    let logits = model
        .forward(&input_ids, &offset_t, &mut caches)
        .expect("Qwen2Model forward");
    logits.eval().expect("eval logits");
    let got: Vec<bf16> = logits.cpu_to_vec().expect("cpu_to_vec");

    assert_eq!(got.len(), logits_ref.len(), "rust/mlx logits length");
    assert_close(&format!("qwen2_logits T={t}"), &got, &logits_ref);
}

#[test]
fn qwen2_logits_prefill() {
    let Some(checkpoint) = checkpoint_path() else {
        eprintln!(
            "skipped: set CIDER_QWEN_CHECKPOINT_PATH to a local \
             Qwen2.5-0.5B-Instruct-4bit MLX checkout to run this test"
        );
        return;
    };
    run_qwen2_logits(&checkpoint);
}
