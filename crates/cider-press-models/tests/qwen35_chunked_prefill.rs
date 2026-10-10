//! Chunked prefill on the real Qwen3.5-4B: a long prompt (well past the
//! unchunked path's memory cliff) prefills and decodes, and on a prompt short
//! enough to run unchunked, the chunked `Generator` greedy stream equals a
//! hand-driven single-forward prefill + `T = 1` decode loop.
//!
//! Gated on `CIDER_QWEN35_CHECKPOINT_PATH`; run with `--ignored`.

#![cfg(target_os = "macos")]

use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;

use cider_press_models::generator::Generator;
use cider_press_models::language_model::LanguageModel;
use cider_press_models::qwen3_5::{Qwen35Config, Qwen35Model, load_qwen35_weights};
use cider_press_models::tokenizer::Tokenizer;
use cider_press_runtime::{Device, Tensor};
use cider_press_test_utils::require_checkpoint_dir;
use safetensors::SafeTensors;

const PASSAGE: &str = "The river bends twice before it reaches the old mill, \
    where the water slows and the town keeps its records of every flood since \
    the bridge was built. Each spring the clerk copies the high-water marks into \
    a ledger, and each autumn the children are taught to read them. ";

const MAX_NEW: usize = 8;

fn checkpoint_dir() -> PathBuf {
    require_checkpoint_dir(
        "CIDER_QWEN35_CHECKPOINT_PATH",
        "a local Qwen3.5-4B-4bit MLX checkpoint directory",
    )
}

fn load(dir: &std::path::Path) -> (Qwen35Model, Tokenizer) {
    let config =
        Qwen35Config::from_json_bytes(&fs::read(dir.join("config.json")).expect("config.json"))
            .expect("config");
    let bytes = fs::read(dir.join("model.safetensors")).expect("model.safetensors");
    let archive = SafeTensors::deserialize(&bytes).expect("deserialize");
    let device = Device::shared().expect("device");
    let weights = load_qwen35_weights(&archive, &config, &device).expect("load weights");
    let model = Qwen35Model::from_weights(weights, config).expect("build Qwen35Model");
    let tokenizer = Tokenizer::from_file(&dir.join("tokenizer.json")).expect("tokenizer");
    (model, tokenizer)
}

/// The first `len` tokens of `PASSAGE` repeated.
fn prompt(tokenizer: &Tokenizer, len: usize) -> Vec<u32> {
    let ids = tokenizer
        .encode(&PASSAGE.repeat(len / 16 + 1))
        .expect("encode");
    assert!(ids.len() >= len, "passage too short: {} < {len}", ids.len());
    ids[..len].to_vec()
}

/// Greedy-decode through the generator, which chunks Qwen3.5 prefill.
fn generate_chunked(model: Qwen35Model, prompt_ids: &[u32]) -> Vec<u32> {
    // u32::MAX is an unreachable EOS so the loop runs the full budget.
    let mut generator =
        Generator::new(model, prompt_ids.len() + MAX_NEW, HashSet::from([u32::MAX]))
            .expect("Generator::new");
    generator
        .generate(prompt_ids, MAX_NEW)
        .expect("generate")
        .map(|r| r.expect("decode step"))
        .collect()
}

/// Greedy-decode with one `forward_last` over the whole prompt (no chunking),
/// then synchronous `T = 1` steps.
fn generate_unchunked(model: &Qwen35Model, prompt_ids: &[u32]) -> Vec<u32> {
    let device = Device::shared().expect("device");
    let mut caches = model
        .make_caches(prompt_ids.len() + MAX_NEW, &device)
        .expect("caches");
    let mut feed = prompt_ids.to_vec();
    let mut pos = 0usize;
    let mut out = Vec::with_capacity(MAX_NEW);
    while out.len() < MAX_NEW {
        let ids = Tensor::from_slice(&device, &feed, [1, feed.len()]).expect("ids");
        let offset =
            Tensor::from_slice(&device, &[i32::try_from(pos).expect("pos")], [1]).expect("offset");
        let next = model
            .forward_last(&ids, &offset, &mut caches)
            .expect("forward_last")
            .argmax(2)
            .expect("argmax");
        next.eval().expect("eval");
        let id = next.cpu_slice::<u32>().expect("host id")[0];
        out.push(id);
        pos += feed.len();
        feed = vec![id];
    }
    out
}

#[test]
#[ignore = "requires CIDER_QWEN35_CHECKPOINT_PATH (Qwen3.5-4B)"]
fn long_prompt_prefills_and_decodes() {
    let (model, tokenizer) = load(&checkpoint_dir());
    let prompt_ids = prompt(&tokenizer, 300);
    let ids = generate_chunked(model, &prompt_ids);
    assert_eq!(ids.len(), MAX_NEW, "got {ids:?}");
    eprintln!(
        "300-token prompt continuation: {:?}",
        tokenizer.decode(&ids).unwrap_or_default()
    );
}

#[test]
#[ignore = "requires CIDER_QWEN35_CHECKPOINT_PATH (Qwen3.5-4B)"]
fn chunked_greedy_matches_unchunked() {
    let dir = checkpoint_dir();
    let (model, tokenizer) = load(&dir);
    // 36 = 16 + 16 + 4: two fill_cache chunks and a ragged forward_last tail,
    // yet short enough that the unchunked oracle's transient stays ~9 GB.
    let prompt_ids = prompt(&tokenizer, 36);
    let unchunked = generate_unchunked(&model, &prompt_ids);
    let chunked = generate_chunked(model, &prompt_ids);
    assert_eq!(
        chunked,
        unchunked,
        "chunked {:?} vs unchunked {:?}",
        tokenizer.decode(&chunked).unwrap_or_default(),
        tokenizer.decode(&unchunked).unwrap_or_default(),
    );
}
