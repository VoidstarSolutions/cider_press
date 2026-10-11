//! End-to-end greedy token-stream parity gate.
//!
//! Asserts cider's `Generator<Qwen35Model>` greedy (argmax) continuation
//! is identical, token-for-token, to mlx-lm's argmax stream on the same
//! RAW prompt (no chat template) + same tokenizer. Greedy is
//! deterministic, so an exact per-step match is the robust correctness
//! gate over the full N-token stream. One test per checkpoint: the
//! single-file 4B and the 3-shard 27B (untied `lm_head`, 64 layers).
//!
//! Gated on the checkpoint env var (`CIDER_QWEN35_CHECKPOINT_PATH` /
//! `CIDER_QWEN35_27B_CHECKPOINT_PATH`) AND the reference JSON written by
//! `scripts/dump_qwen35_greedy.py`. Either missing → the test panics (both
//! are `#[ignore]`d; run with `--ignored`).

#![cfg(target_os = "macos")]

use std::collections::HashSet;
use std::fs;
use std::path::Path;

use cider_press_models::generator::Generator;
use cider_press_models::qwen3_5::{Qwen35Config, Qwen35Model, load_qwen35_weights};
use cider_press_models::safetensors_io::ShardSet;
use cider_press_models::tokenizer::Tokenizer;
use cider_press_runtime::Device;
use cider_press_test_utils::{qwen35_4b_checkpoint_dir, qwen35_27b_checkpoint_dir, require_file};
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    prompt_ids: Vec<u32>,
    continuation_ids: Vec<u32>,
}

fn assert_greedy_stream_matches_mlx(name: &str, dir: &Path, reference_json: &str, regen: &str) {
    let reference: Reference = serde_json::from_str(
        &fs::read_to_string(require_file(reference_json, regen)).expect("read reference json"),
    )
    .expect("parse reference json");
    let Reference {
        prompt_ids,
        continuation_ids: reference,
    } = reference;
    let n = reference.len();
    assert!(!prompt_ids.is_empty(), "reference prompt_ids empty");
    assert!(n > 0, "reference continuation_ids empty");

    let config =
        Qwen35Config::from_json_bytes(&fs::read(dir.join("config.json")).unwrap()).unwrap();
    let shards = ShardSet::read(dir).unwrap();
    let archive = shards.archive().unwrap();
    let device = Device::shared().unwrap();
    let weights = load_qwen35_weights(&archive, &config, &device).unwrap();
    let model = Qwen35Model::from_weights(weights, config).expect("build Qwen35Model");

    // Only used to decode mismatches for diagnostics; the prompt ids come
    // from the reference JSON so tokenization is shared.
    let tokenizer = Tokenizer::from_file(&dir.join("tokenizer.json")).expect("load tokenizer");
    let decode = |id: u32| tokenizer.decode(&[id]).unwrap_or_else(|_| "<?>".into());

    // u32::MAX is an unreachable EOS so the loop runs the full N budget,
    // matching the fixed-N (no early-stop) reference dump.
    let context_window = prompt_ids.len() + n + 8;
    let mut generator =
        Generator::<Qwen35Model>::new(model, context_window, HashSet::from([u32::MAX]))
            .expect("Generator::new");

    let cider: Vec<u32> = generator
        .generate(&prompt_ids, n)
        .expect("generate")
        .map(|r| r.expect("decode step"))
        .collect();

    assert_eq!(
        cider.len(),
        n,
        "{name}: cider yielded {} tokens, expected {n}",
        cider.len()
    );

    if let Some(i) = (0..n).find(|&i| cider[i] != reference[i]) {
        panic!(
            "{name}: greedy stream diverged at continuation index {i} \
             ({i}/{n} tokens matched)\n  \
             cider id = {} ({:?})\n  mlx   id = {} ({:?})\n  \
             cider stream: {:?}\n  mlx   stream: {:?}",
            cider[i],
            decode(cider[i]),
            reference[i],
            decode(reference[i]),
            cider,
            reference,
        );
    }

    eprintln!("{name}: all {n} continuation tokens match mlx-lm");
}

#[test]
#[ignore = "requires CIDER_QWEN35_CHECKPOINT_PATH + /tmp/qwen3_5_4b_greedy.json"]
fn qwen35_greedy_stream_matches_mlx() {
    assert_greedy_stream_matches_mlx(
        "4B greedy",
        &qwen35_4b_checkpoint_dir(),
        "/tmp/qwen3_5_4b_greedy.json",
        "uv run scripts/dump_qwen35_greedy.py",
    );
}

#[test]
#[ignore = "requires CIDER_QWEN35_27B_CHECKPOINT_PATH + /tmp/qwen3_6_27b_greedy.json"]
fn qwen35_27b_greedy_stream_matches_mlx() {
    assert_greedy_stream_matches_mlx(
        "27B greedy",
        &qwen35_27b_checkpoint_dir(),
        "/tmp/qwen3_6_27b_greedy.json",
        "uv run scripts/dump_qwen35_greedy.py --checkpoint mlx-community/Qwen3.6-27B-4bit -n 8 --out /tmp/qwen3_6_27b_greedy.json",
    );
}
