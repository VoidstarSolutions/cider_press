#![cfg(target_os = "macos")]

//! Qwen3.5 full text-model (`Qwen35Model`) build + shape smoke test.
//!
//! Env-gated on a real checkpoint (`CIDER_QWEN35_CHECKPOINT_PATH`): full
//! numeric parity is Phase-4 Task B2 — this only asserts the model **builds**
//! and `forward` produces logits of shape `[1, T, vocab_size]`. The skip path
//! (env unset) is the default in CI and compiles + returns early.

use std::{fs, path::PathBuf};

use cider_press_models::qwen3_5::{
    Qwen35Config, Qwen35LayerCache, Qwen35Model, load_qwen35_weights,
};
use cider_press_runtime::{Device, Tensor};
use cider_press_test_utils::{read_bf16, read_u32};
use half::bf16;
use safetensors::SafeTensors;

const FIXTURES: &str = "/tmp/qwen3_5_4b_fixtures.safetensors";

fn checkpoint_path() -> Option<PathBuf> {
    std::env::var_os("CIDER_QWEN35_CHECKPOINT_PATH").map(PathBuf::from)
}

/// Combined np.allclose bound `|a-b| <= atol + rtol*|b|`. Reports max abs/rel.
fn assert_close(label: &str, got: &[bf16], expected: &[bf16], atol: f32, rtol: f32) {
    assert_eq!(got.len(), expected.len(), "{label}: length mismatch");
    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    for (i, (a, b)) in got.iter().zip(expected.iter()).enumerate() {
        let (a, b) = (a.to_f32(), b.to_f32());
        let abs = (a - b).abs();
        max_abs = max_abs.max(abs);
        if b.abs() > 1e-6 {
            max_rel = max_rel.max(abs / b.abs());
        }
        let bound = atol + rtol * b.abs();
        assert!(
            abs <= bound,
            "{label}[{i}]: got={a} want={b} > bound={bound} (max_abs so far={max_abs})"
        );
    }
    println!("{label}: max_abs={max_abs}, max_rel={max_rel}");
}

/// Argmax over the last `[vocab]` row of a `[1, T, vocab]` row-major buffer.
fn last_row_argmax(buf: &[bf16], vocab: usize) -> usize {
    let last = &buf[buf.len() - vocab..];
    last.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.to_f32().total_cmp(&b.to_f32()))
        .map(|(i, _)| i)
        .unwrap()
}

/// Build a `Qwen35Model` from the real 4B checkpoint and run a synthetic
/// prompt through `forward`; assert the logits shape is `[1, T, vocab_size]`.
#[test]
fn qwen35_model_forward_shape() {
    let Some(dir) = checkpoint_path() else {
        eprintln!("skipping: CIDER_QWEN35_CHECKPOINT_PATH unset");
        return;
    };
    let config =
        Qwen35Config::from_json_bytes(&fs::read(dir.join("config.json")).unwrap()).unwrap();
    let archive_bytes = fs::read(dir.join("model.safetensors")).unwrap();
    let archive = SafeTensors::deserialize(&archive_bytes).unwrap();
    let device = Device::shared().unwrap();
    let weights = load_qwen35_weights(&archive, &config, &device).unwrap();

    let model = Qwen35Model::from_weights(weights, config.clone()).unwrap();

    // Synthetic prompt: a short run of low token ids is enough for a shape /
    // it-runs check (numeric parity against mlx-lm is Task B2).
    let ids: Vec<u32> = (0..5u32).collect();
    let t = ids.len();
    let input_ids = Tensor::from_slice(&device, &ids, [1usize, t]).unwrap();
    let offset = Tensor::from_slice(&device, &[0i32], [1usize]).unwrap();

    let mut caches: Vec<Qwen35LayerCache> = (0..config.num_hidden_layers)
        .map(|i| Qwen35LayerCache::for_layer(&config, i, &device, t).unwrap())
        .collect();

    let logits = model.forward(&input_ids, &offset, &mut caches).unwrap();
    logits.eval().unwrap();

    assert_eq!(
        logits.shape().dims(),
        [1, t, config.vocab_size],
        "forward logits shape"
    );

    // forward_last projects only the final position: [1, 1, vocab_size].
    let mut last_caches: Vec<Qwen35LayerCache> = (0..config.num_hidden_layers)
        .map(|i| Qwen35LayerCache::for_layer(&config, i, &device, t).unwrap())
        .collect();
    let last = model
        .forward_last(&input_ids, &offset, &mut last_caches)
        .unwrap();
    last.eval().unwrap();
    assert_eq!(
        last.shape().dims(),
        [1, 1, config.vocab_size],
        "forward_last logits shape"
    );

    println!("Qwen35Model forward [1,{t},{}] OK", config.vocab_size);
}

/// End-to-end parity for the full 32-layer assembly vs mlx-lm on the 4B
/// fixture. Two gates:
///
/// 1. **`final_hidden` (backbone):** post-final-norm hidden (`forward_hidden`,
///    mirrors mlx-lm `lm.norm(x)`). Localizes a failure to the 32 decoder
///    layers vs the head.
/// 2. **logits (end-to-end):** full `forward` (incl. the tied `lm_head`).
///
/// Both use the combined np.allclose bound `|a-b| <= atol + rtol*|b|` with
/// **atol 0.6, rtol 0.1**. That is looser than a single-op parity tol *by
/// design*: this is a 32-layer bf16 residual chain. The per-layer drift was
/// measured to grow **smoothly and monotonically** with depth (embed bit-exact;
/// layer-0/3 in-model outputs ~0.02 abs; the worst final element needs
/// atol ~0.5 at rtol 0.05) — pure accumulation, no per-layer discontinuity that
/// would flag a buggy mixer/cache. The bound is therefore sized to pass genuine
/// deep-chain accumulation while still catching a structural break (a swapped
/// layer or mis-typed cache would diverge by O(1+) on whole rows, far past it).
///
/// The **strongest practical gate is the last-position argmax** matching the
/// fixture exactly — greedy decode consumes only that one logits row, and the
/// argmax is robust to the per-element bf16 drift.
///
/// Env-gated on `CIDER_QWEN35_CHECKPOINT_PATH` + fixture existence.
#[test]
fn qwen35_model_logits_match_mlx() {
    let Some(dir) = checkpoint_path() else {
        eprintln!("skipping: CIDER_QWEN35_CHECKPOINT_PATH unset");
        return;
    };
    if !std::path::Path::new(FIXTURES).exists() {
        eprintln!("skipping: {FIXTURES} missing — run scripts/dump_qwen35_fixtures.py");
        return;
    }
    let config =
        Qwen35Config::from_json_bytes(&fs::read(dir.join("config.json")).unwrap()).unwrap();
    let archive_bytes = fs::read(dir.join("model.safetensors")).unwrap();
    let archive = SafeTensors::deserialize(&archive_bytes).unwrap();
    let device = Device::shared().unwrap();
    let weights = load_qwen35_weights(&archive, &config, &device).unwrap();
    let model = Qwen35Model::from_weights(weights, config.clone()).unwrap();

    let fx_bytes = fs::read(FIXTURES).unwrap();
    let fx = SafeTensors::deserialize(&fx_bytes).unwrap();
    let input_ids = read_u32(&fx, "input_ids"); // [1, T]
    let t = input_ids.len();
    let ids = Tensor::from_slice(&device, &input_ids, [1usize, t]).unwrap();
    let offset = Tensor::from_slice(&device, &[0i32], [1usize]).unwrap();

    let build_caches = || -> Vec<Qwen35LayerCache> {
        (0..config.num_hidden_layers)
            .map(|i| Qwen35LayerCache::for_layer(&config, i, &device, t).unwrap())
            .collect()
    };

    // (1) Backbone: post-final-norm hidden. Tighter — localizes layer vs head.
    let mut caches = build_caches();
    let fh = model.forward_hidden(&ids, &offset, &mut caches).unwrap();
    fh.eval().unwrap();
    let fh_got: Vec<bf16> = fh.cpu_to_vec().unwrap();
    let fh_want = read_bf16(&fx, "final_hidden"); // [1, T, hidden]
    // atol 0.6 / rtol 0.1: the 32-layer bf16 residual chain (see the test
    // doc comment) — the worst post-norm element drifts ~0.5 abs.
    assert_close("final_hidden", &fh_got, &fh_want, 0.6, 0.1);

    // (2) End-to-end: full forward incl. tied lm_head. Fresh caches — the
    // first forward consumed the backbone caches.
    let mut caches2 = build_caches();
    let logits = model.forward(&ids, &offset, &mut caches2).unwrap();
    logits.eval().unwrap();
    let lg_got: Vec<bf16> = logits.cpu_to_vec().unwrap();
    let lg_want = read_bf16(&fx, "logits"); // [1, T, vocab]
    // atol 0.6 / rtol 0.1: same deep-chain accumulation, fed through the tied
    // head. The argmax assertion below is the real end-to-end gate.
    assert_close("logits", &lg_got, &lg_want, 0.6, 0.1);

    // The meaningful end-to-end signal: greedy decode only needs the argmax of
    // the last position. Assert it matches the fixture exactly.
    let got_argmax = last_row_argmax(&lg_got, config.vocab_size);
    let want_argmax = last_row_argmax(&lg_want, config.vocab_size);
    println!("last-position argmax: cider={got_argmax}, mlx={want_argmax}");
    assert_eq!(
        got_argmax, want_argmax,
        "last-position argmax mismatch: cider={got_argmax}, mlx={want_argmax}"
    );
}
