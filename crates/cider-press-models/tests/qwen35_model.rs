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
use safetensors::SafeTensors;

fn checkpoint_path() -> Option<PathBuf> {
    std::env::var_os("CIDER_QWEN35_CHECKPOINT_PATH").map(PathBuf::from)
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
