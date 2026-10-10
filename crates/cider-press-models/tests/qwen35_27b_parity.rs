//! End-to-end greedy token-stream parity gate for the **sharded 27B**
//! (Phase-5 Task C2).
//!
//! Asserts cider's `Generator<Qwen35Model>` greedy (argmax) continuation
//! is identical, token-for-token, to mlx-lm's argmax stream on the same
//! RAW prompt (no chat template) + same tokenizer, for the 3-shard
//! `mlx-community/Qwen3.6-27B-4bit` (untied `lm_head`, 64 layers, 48
//! value-heads). Greedy is deterministic, so an exact per-step match is
//! the robust correctness gate.
//!
//! This mirrors the 4B `qwen35_greedy_parity` test, but loads the
//! multi-shard checkpoint via `model.safetensors.index.json` +
//! [`ShardedArchive`] instead of a single `model.safetensors`.
//!
//! Gated on `CIDER_QWEN35_27B_CHECKPOINT_PATH` AND the reference JSON
//! written by `scripts/dump_qwen35_greedy.py` at
//! `/tmp/qwen3_6_27b_greedy.json`. Either missing → the test skips with a
//! message (clean in CI). `#[ignore]` because the 27B is ~15 GB and decode
//! is slow (naive per-token GDN; perf is Phases 6–7).

#![cfg(target_os = "macos")]

use cider_press_test_utils::{require_checkpoint_dir, require_file};
use std::collections::{HashMap, HashSet};
use std::{fs, path::PathBuf};

use cider_press_models::generator::Generator;
use cider_press_models::qwen3_5::{Qwen35Config, Qwen35Model, load_qwen35_weights};
use cider_press_models::safetensors_io::ShardedArchive;
use cider_press_models::tokenizer::Tokenizer;
use cider_press_runtime::Device;
use serde::Deserialize;

const REFERENCE_JSON: &str = "/tmp/qwen3_6_27b_greedy.json";
const REFERENCE_REGEN: &str = "uv run scripts/dump_qwen35_greedy.py --checkpoint mlx-community/Qwen3.6-27B-4bit -n 8 --out /tmp/qwen3_6_27b_greedy.json";

fn checkpoint_path() -> PathBuf {
    require_checkpoint_dir(
        "CIDER_QWEN35_27B_CHECKPOINT_PATH",
        "a local Qwen3.6-27B-4bit MLX checkpoint directory",
    )
}

/// Minimal view of `model.safetensors.index.json`: the `weight_map` routing
/// each tensor key to its shard filename (mirrors the CLI loader).
#[derive(Deserialize)]
struct SafetensorsIndex {
    weight_map: HashMap<String, String>,
}

/// Minimal reader for the reference JSON produced by
/// `scripts/dump_qwen35_greedy.py`. Pulls one flat integer array out by
/// hand — the keys we need are simple `"key": [1, 2, 3]` lists.
fn read_u32_array(json: &str, key: &str) -> Vec<u32> {
    let needle = format!("\"{key}\"");
    let start = json
        .find(&needle)
        .unwrap_or_else(|| panic!("reference JSON missing key {key}"));
    let bracket = json[start..]
        .find('[')
        .map(|i| start + i + 1)
        .expect("array open bracket");
    let end = json[bracket..]
        .find(']')
        .map(|i| bracket + i)
        .expect("array close bracket");
    json[bracket..end]
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<u32>().expect("u32 token id"))
        .collect()
}

#[test]
#[ignore = "requires CIDER_QWEN35_27B_CHECKPOINT_PATH + /tmp/qwen3_6_27b_greedy.json"]
fn qwen35_27b_greedy_stream_matches_mlx() {
    let dir = checkpoint_path();
    let json = fs::read_to_string(require_file(REFERENCE_JSON, REFERENCE_REGEN))
        .expect("read reference json");

    let prompt_ids = read_u32_array(&json, "prompt_ids");
    let reference = read_u32_array(&json, "continuation_ids");
    let n = reference.len();
    assert!(!prompt_ids.is_empty(), "reference prompt_ids empty");
    assert!(n > 0, "reference continuation_ids empty");

    // Config (nested text_config, top-level quantization).
    let config =
        Qwen35Config::from_json_bytes(&fs::read(dir.join("config.json")).unwrap()).unwrap();
    let device = Device::shared().unwrap();

    // Multi-shard load: route each tensor key to its shard via the
    // `model.safetensors.index.json` weight_map (mirrors the CLI's C1 path).
    let index: SafetensorsIndex =
        serde_json::from_slice(&fs::read(dir.join("model.safetensors.index.json")).unwrap())
            .unwrap();
    let mut shard_names: Vec<String> = index.weight_map.values().cloned().collect();
    shard_names.sort();
    shard_names.dedup();
    let shard_bytes: Vec<Vec<u8>> = shard_names
        .iter()
        .map(|name| fs::read(dir.join(name)).unwrap())
        .collect();
    let sharded = ShardedArchive::new(&shard_names, &shard_bytes, &index.weight_map).unwrap();
    let weights = load_qwen35_weights(&sharded, &config, &device).unwrap();
    let model = Qwen35Model::from_weights(weights, config).expect("build Qwen35Model");

    // Tokenizer — only used to decode mismatches for diagnostics. The prompt
    // ids come from the reference JSON so tokenization is shared.
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
        "cider yielded {} tokens, expected {n}",
        cider.len()
    );

    // First-mismatch diagnostics: report index, both ids, both decoded.
    if let Some(i) = (0..n).find(|&i| cider[i] != reference[i]) {
        let matched = i;
        panic!(
            "27B greedy stream diverged at continuation index {i} \
             ({matched}/{n} tokens matched)\n  \
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

    eprintln!("27B greedy stream parity: all {n} continuation tokens match mlx-lm");
}
