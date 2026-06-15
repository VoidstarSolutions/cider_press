//! Env-gated smoke test: drive a real `Qwen35Model` through the generic
//! `Generator<Qwen35Model>`. Verifies the `LanguageModel` plumbing yields the
//! requested number of token ids end-to-end; token *values* are validated in
//! C3 (greedy parity). Gated on `CIDER_QWEN35_CHECKPOINT_PATH` (skips in CI).

#![cfg(target_os = "macos")]

use std::collections::HashSet;
use std::{fs, path::PathBuf};

use cider_press_models::generator::Generator;
use cider_press_models::qwen3_5::{Qwen35Config, Qwen35Model, load_qwen35_weights};
use cider_press_runtime::Device;
use safetensors::SafeTensors;

fn checkpoint_path() -> Option<PathBuf> {
    std::env::var_os("CIDER_QWEN35_CHECKPOINT_PATH").map(PathBuf::from)
}

#[test]
#[ignore = "requires a real checkpoint via CIDER_QWEN35_CHECKPOINT_PATH"]
fn qwen35_generator_yields_requested_tokens() {
    let Some(dir) = checkpoint_path() else {
        eprintln!("skipping: CIDER_QWEN35_CHECKPOINT_PATH unset");
        return;
    };

    let config =
        Qwen35Config::from_json_bytes(&fs::read(dir.join("config.json")).unwrap()).unwrap();
    let bytes = fs::read(dir.join("model.safetensors")).unwrap();
    let archive = SafeTensors::deserialize(&bytes).unwrap();
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
