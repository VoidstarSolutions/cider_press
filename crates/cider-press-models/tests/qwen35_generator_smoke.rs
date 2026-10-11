//! Env-gated smoke test: drive a real `Qwen35Model` through the generic
//! `Generator<Qwen35Model>`. Verifies the `LanguageModel` plumbing yields the
//! requested number of token ids end-to-end; token *values* are validated by
//! `qwen35_greedy_parity`. `#[ignore]`d; needs `CIDER_QWEN35_CHECKPOINT_PATH`.

#![cfg(target_os = "macos")]

use cider_press_models::safetensors_io::ShardSet;
use cider_press_test_utils::require_checkpoint_dir;
use std::collections::HashSet;
use std::{fs, path::PathBuf};

use cider_press_models::generator::Generator;
use cider_press_models::qwen3_5::{Qwen35Config, Qwen35Model, load_qwen35_weights};
use cider_press_runtime::Device;

fn checkpoint_path() -> PathBuf {
    require_checkpoint_dir(
        "CIDER_QWEN35_CHECKPOINT_PATH",
        "a local Qwen3.5-4B-4bit MLX checkpoint directory",
    )
}

#[test]
#[ignore = "requires a real checkpoint via CIDER_QWEN35_CHECKPOINT_PATH"]
fn qwen35_generator_yields_requested_tokens() {
    let dir = checkpoint_path();

    let config =
        Qwen35Config::from_json_bytes(&fs::read(dir.join("config.json")).unwrap()).unwrap();
    let shards = ShardSet::read(&dir).unwrap();
    let archive = shards.archive().unwrap();
    let device = Device::shared().unwrap();
    let weights = load_qwen35_weights(&archive, &config, &device).unwrap();
    let model = Qwen35Model::from_weights(weights, config).expect("build Qwen35Model");

    // u32::MAX is an unreachable EOS so the loop runs the full budget.
    let mut generator = Generator::<Qwen35Model>::new(model, 256, HashSet::from([u32::MAX]))
        .expect("Generator::new");

    let prompt_ids: Vec<u32> = vec![1, 2, 3, 4];
    let max_new = 4;
    let ids: Vec<u32> = generator
        .generate(&prompt_ids, max_new)
        .expect("generate")
        .map(|r| r.expect("decode step"))
        .collect();

    assert_eq!(
        ids.len(),
        max_new,
        "generator should yield exactly max_new tokens (got {ids:?})"
    );
}
