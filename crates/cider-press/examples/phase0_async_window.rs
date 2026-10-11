//! Phase-0 throwaway: time 64 independent 150-token prefill forwards with a
//! bounded `eval_async` window of `k` graphs in flight. `k = 1` is the
//! synchronous baseline. Deleted at the end of phase 0; the numbers live in
//! docs/inference/models/qwen3-embedding.md.
//!
//! Usage: `cargo run --release --example phase0_async_window -- <checkpoint> <prompt.txt> <k> [texts=64]`

use std::collections::VecDeque;
use std::path::Path;
use std::time::Instant;

use cider_press_models::language_model::LanguageModel;
use cider_press_models::qwen2::{Qwen2Config, Qwen2Model, load_qwen2_weights};
use cider_press_models::safetensors_io::ShardSet;
use cider_press_models::tokenizer::Tokenizer;
use cider_press_runtime::{Device, KvCache, PendingEval, Tensor};

type BoxError = Box<dyn std::error::Error>;

#[allow(clippy::cast_precision_loss)]
fn main() -> Result<(), BoxError> {
    let args: Vec<String> = std::env::args().collect();
    let checkpoint = Path::new(&args[1]);
    let prompt = std::fs::read_to_string(&args[2])?;
    let k: usize = args[3].parse()?;
    let texts: usize = args.get(4).map_or(Ok(64), |s| s.parse())?;
    assert!(k >= 1, "window must be >= 1");

    let device = Device::shared()?;
    let config_bytes = std::fs::read(checkpoint.join("config.json"))?;
    let config = Qwen2Config::from_json_bytes(&config_bytes)?;
    let shards = ShardSet::read(checkpoint)?;
    let weights = load_qwen2_weights(&shards.archive()?, &config, &device)?;
    let model = Qwen2Model::from_weights(weights, config)?;
    let tokenizer = Tokenizer::from_file(&checkpoint.join("tokenizer.json"))?;
    let ids = tokenizer.encode(&prompt)?;
    let t = ids.len();
    let input = Tensor::from_slice::<u32>(&device, &ids, [1, t])?;

    // One cache set per window slot: a slot is reused only after its graph
    // completed (`wait`), so `reset` never sees an uncommitted update.
    let mut cache_sets: Vec<Vec<KvCache>> = (0..k)
        .map(|_| model.make_caches(t + 1, &device))
        .collect::<Result<_, _>>()?;

    // Warm-up (Metal JIT): two synchronous forwards.
    for _ in 0..2 {
        let logits = model.forward_last(&input, &mut cache_sets[0])?;
        logits.eval()?;
        model.reset_caches(&mut cache_sets[0])?;
    }

    let mut reference: Option<Vec<u32>> = None;
    let mut inflight: VecDeque<(PendingEval, Tensor)> = VecDeque::with_capacity(k);
    let start = Instant::now();
    for i in 0..texts {
        if inflight.len() == k {
            let (pending, logits) = inflight.pop_front().expect("len == k >= 1");
            pending.wait()?;
            check_argmax(&logits, &mut reference)?;
        }
        let slot = i % k;
        model.reset_caches(&mut cache_sets[slot])?;
        let logits = model.forward_last(&input, &mut cache_sets[slot])?;
        let pending = logits.eval_async()?;
        inflight.push_back((pending, logits));
    }
    while let Some((pending, logits)) = inflight.pop_front() {
        pending.wait()?;
        check_argmax(&logits, &mut reference)?;
    }
    let elapsed = start.elapsed();

    let per_text_ms = elapsed.as_secs_f64() * 1e3 / texts as f64;
    println!(
        "k={k} texts={texts} tokens/text={t} total_ms={:.1} per_text_ms={per_text_ms:.2} texts_per_s={:.1}",
        elapsed.as_secs_f64() * 1e3,
        texts as f64 / elapsed.as_secs_f64()
    );
    Ok(())
}

/// Every forward sees the same prompt, so the greedy token must be identical
/// at every window size; a mismatch means a cache slot was reused too early.
fn check_argmax(logits: &Tensor, reference: &mut Option<Vec<u32>>) -> Result<(), BoxError> {
    let id = logits.argmax(2)?; // [1, 1, vocab] -> [1, 1] U32, as the generator does
    id.eval()?;
    let got = id.cpu_to_vec::<u32>().ok_or("argmax not materialised")?;
    match reference {
        None => *reference = Some(got),
        Some(r) if *r != got => return Err(format!("argmax drift: {r:?} vs {got:?}").into()),
        Some(_) => {}
    }
    Ok(())
}
