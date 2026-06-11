//! `cider-press` CLI.
//!
//! Two subcommands:
//! - `chat`  — load a Qwen2 checkpoint, render the prompt through its
//!   chat template, greedy-decode with streamed detokenization.
//! - `bench` — same load + greedy-decode path, but measure prefill and
//!   decode throughput, peak RSS, and (with `--features profiling`) a
//!   per-span time breakdown; print a report instead of the text.

use std::collections::HashSet;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use clap::{Args, Parser, Subcommand};

use cider_press_models::chat_template::{ChatTemplate, Message};
use cider_press_models::generator::Generator;
use cider_press_models::qwen2::{Qwen2Config, Qwen2Model, load_qwen2_weights};
use cider_press_models::tokenizer::Tokenizer;
use cider_press_runtime::{Device, profile};
use safetensors::SafeTensors;

type BoxError = Box<dyn std::error::Error + Send + Sync>;

const DEFAULT_BENCH_PROMPT: &str = "Write a short paragraph about the city of Seattle.";

#[derive(Parser, Debug)]
#[command(
    name = "cider-press",
    version,
    about = "Cold-pressed LLM inference for Apple Silicon"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Greedy decode + streamed detokenization.
    Chat(ChatArgs),
    /// Benchmark prefill/decode throughput + peak RSS; print a report.
    Bench(BenchArgs),
}

#[derive(Args, Debug)]
struct ChatArgs {
    /// Checkpoint directory holding config.json, tokenizer.json,
    /// `tokenizer_config.json`, and a single-file model.safetensors.
    /// Falls back to `$CIDER_PRESS_CHECKPOINT` when the flag is omitted.
    #[arg(long, env = "CIDER_PRESS_CHECKPOINT")]
    checkpoint: PathBuf,
    /// User message.
    #[arg(long)]
    prompt: String,
    /// Optional system message; omitted means no system turn.
    #[arg(long, conflicts_with = "no_chat_template")]
    system: Option<String>,
    /// Maximum new tokens to generate.
    #[arg(long, default_value_t = 256)]
    max_tokens: usize,
    /// Pre-allocated `KvCache` window in tokens.
    #[arg(long, default_value_t = 4096)]
    context_window: usize,
    /// Skip `ChatTemplate` rendering; encode the bare --prompt.
    /// `tokenizer_config.json` is still read for the EOS token ids.
    #[arg(long)]
    no_chat_template: bool,
}

#[derive(Args, Debug)]
struct BenchArgs {
    /// Checkpoint directory (same layout as `chat`).
    /// Falls back to `$CIDER_PRESS_CHECKPOINT` when the flag is omitted.
    #[arg(long, env = "CIDER_PRESS_CHECKPOINT")]
    checkpoint: PathBuf,
    /// Prompt to benchmark; defaults to a fixed paragraph request.
    #[arg(long, default_value_t = DEFAULT_BENCH_PROMPT.to_owned())]
    prompt: String,
    /// Optional system message.
    #[arg(long, conflicts_with = "no_chat_template")]
    system: Option<String>,
    /// New tokens to generate (the decode workload).
    #[arg(long, default_value_t = 128)]
    max_tokens: usize,
    /// Pre-allocated `KvCache` window in tokens.
    #[arg(long, default_value_t = 4096)]
    context_window: usize,
    /// Decode steps to discard before timing (warm caches / JIT).
    #[arg(long, default_value_t = 8)]
    warmup: usize,
    /// Async-pipeline lookahead depth (see `Generator::set_inflight_depth`).
    /// Default 1 (one decode forward overlapping the readback). 0 disables
    /// lookahead — one token committed and waited at a time, still through
    /// the async-eval + on-GPU id-chaining path (not the historical
    /// pre-pipelining implementation).
    #[arg(long, default_value_t = 1)]
    inflight_depth: usize,
    /// Skip `ChatTemplate` rendering; encode the bare --prompt.
    #[arg(long)]
    no_chat_template: bool,
    /// Run one timed decode token through the profiled eval (one sampled
    /// compute encoder per dispatch) to print a per-OpKind GPU-time
    /// breakdown. Perturbs the encode regime; the production
    /// encode/wait spans are still reported from the normal eval. No-op
    /// without --features profiling.
    #[arg(long)]
    gpu_profile: bool,
    /// After the timed decode window, run ONE prefill forward (T = prompt
    /// length) through the profiled eval and print a per-OpKind GPU
    /// breakdown for the prefill path. No-op without --features profiling.
    #[arg(long)]
    gpu_profile_prefill: bool,
    /// Iterations for the synchronous prefill measurement (mlx-comparable:
    /// forward + eval-wait, warm). Mean is reported as "prefill (sync)".
    #[arg(long, default_value_t = 5)]
    prefill_iters: usize,
}

/// Shared load result for both subcommands.
struct Loaded {
    model: Qwen2Model,
    tokenizer: Tokenizer,
    eos_ids: HashSet<u32>,
    chat_template: ChatTemplate,
}

fn load(checkpoint: &Path, device: &Device) -> Result<Loaded, BoxError> {
    let config_bytes = std::fs::read(checkpoint.join("config.json"))?;
    let config = Qwen2Config::from_json_bytes(&config_bytes)?;

    let archive_bytes = std::fs::read(checkpoint.join("model.safetensors"))?;
    let archive = SafeTensors::deserialize(&archive_bytes)?;
    let weights = load_qwen2_weights(&archive, &config, device)?;
    let model = Qwen2Model::from_weights(weights, config)?;

    let tokenizer = Tokenizer::from_file(&checkpoint.join("tokenizer.json"))?;
    let chat_template =
        ChatTemplate::from_file(&checkpoint.join("tokenizer_config.json"), &tokenizer)?;
    let eos_ids: HashSet<u32> = chat_template.eos_ids().collect();
    Ok(Loaded {
        model,
        tokenizer,
        eos_ids,
        chat_template,
    })
}

fn build_prompt(
    chat_template: &ChatTemplate,
    prompt: &str,
    system: Option<&str>,
    no_chat_template: bool,
) -> Result<String, BoxError> {
    if no_chat_template {
        return Ok(prompt.to_owned());
    }
    let mut messages = Vec::new();
    if let Some(sys) = system {
        messages.push(Message::system(sys));
    }
    messages.push(Message::user(prompt.to_owned()));
    Ok(chat_template.render(&messages)?)
}

fn run_chat(args: &ChatArgs) -> Result<(), BoxError> {
    let device = Device::shared()?;
    let loaded = load(&args.checkpoint, &device)?;

    let prompt = build_prompt(
        &loaded.chat_template,
        &args.prompt,
        args.system.as_deref(),
        args.no_chat_template,
    )?;
    let ids = loaded.tokenizer.encode(&prompt)?;

    let mut generator = Generator::new(loaded.model, args.context_window, loaded.eos_ids)?;
    let mut stream = loaded.tokenizer.decode_stream(true);

    let stdout = io::stdout();
    let mut handle = stdout.lock();
    for id_result in generator.generate(&ids, args.max_tokens)? {
        let id = id_result?;
        if let Some(text) = stream.step(id)? {
            handle.write_all(text.as_bytes())?;
            handle.flush()?;
        }
    }
    writeln!(handle)?;
    Ok(())
}

#[allow(clippy::cast_precision_loss, clippy::too_many_lines)]
fn run_bench(args: &BenchArgs) -> Result<(), BoxError> {
    let device = Device::shared()?;

    let rss_pre_load = cider_press::sys::resident_bytes();
    let t_load = Instant::now();
    let loaded = load(&args.checkpoint, &device)?;
    let load_dur = t_load.elapsed();
    let rss_post_load = cider_press::sys::resident_bytes();

    let prompt = build_prompt(
        &loaded.chat_template,
        &args.prompt,
        args.system.as_deref(),
        args.no_chat_template,
    )?;
    let ids = loaded.tokenizer.encode(&prompt)?;
    let prompt_len = ids.len();

    let mut generator = Generator::new(loaded.model, args.context_window, loaded.eos_ids)?;
    generator.set_inflight_depth(args.inflight_depth);

    // Prefill is performed inside generate() before it returns the iterator.
    let t_prefill = Instant::now();
    let iter = generator.generate(&ids, args.max_tokens)?;
    let prefill_dur = t_prefill.elapsed();

    // Decode: drain the iterator. Discard `warmup` tokens, then reset the
    // profiler and start the decode timer so both the tok/s figure and the
    // span breakdown reflect only the steady-state window.
    let mut produced = 0usize;
    let mut timed = 0usize;
    let mut decode_dur = std::time::Duration::ZERO;
    // With `--warmup 0` the in-loop `produced == args.warmup` check never
    // fires (produced is incremented to >= 1 before it), so start timing
    // before the first token in that case. Otherwise the timer starts when
    // the warmup-th token is produced.
    let mut decode_timer: Option<Instant> = if args.warmup == 0 {
        profile::reset();
        Some(Instant::now())
    } else {
        None
    };
    let mut last_id: Option<u32> = None;
    for id_result in iter {
        last_id = Some(id_result?);
        produced += 1;
        if produced == args.warmup {
            profile::reset();
            decode_timer = Some(Instant::now());
        } else if produced > args.warmup {
            timed += 1;
        }
    }
    if let Some(t) = decode_timer {
        decode_dur = t.elapsed();
    }

    // Capture the timed-window wall-clock spans BEFORE the optional profiled
    // GPU-profile token runs. `gpu_profile_token` performs a real
    // `model.forward`, whose KvCache updates emit `kvcache.update` spans that
    // would otherwise contaminate this table. The GPU table is a separate
    // store (`record_gpu`) drained later, so it is unaffected.
    let spans = profile::drain();

    // Decode GPU profile FIRST: `gpu_profile_token` wants the full post-decode
    // cache context, and the synchronous prefill timing below resets the caches
    // (`prefill_sync` → `reset`), so it must run after this.
    if args.gpu_profile {
        // generate() rejects max_tokens == 0, so last_id is always Some here.
        if let Some(id) = last_id {
            gpu_profile_token(&mut generator, id)?;
        }
    }

    // Synchronous prefill timing (mlx-comparable): forward + eval-wait, warm.
    // The production prefill above is async (eval_async, no GPU wait), so it
    // is not comparable to mlx_lm's synchronous model(x)+mx.eval. This runs
    // the same shape synchronously for an apples-to-apples wall-clock. It
    // resets the caches, so it runs after the decode profile above.
    let prefill_iters = args.prefill_iters.max(1);
    let (prefill_sync_dur, prefill_spans) = {
        // Settle Metal JIT on the synchronous eval() path before timing
        // (the caches are already warm from generate()).
        for _ in 0..2 {
            generator.prefill_sync(&ids)?;
        }
        // Discard the settle-iter wall spans; keep the GPU store intact (the
        // decode GPU breakdown is drained later). drain() touches only the
        // wall-clock store, reset() would wipe both.
        let _ = profile::drain();
        let t = Instant::now();
        for _ in 0..prefill_iters {
            generator.prefill_sync(&ids)?;
        }
        let dur = t.elapsed();
        (dur, profile::drain())
    };

    let rss_post_decode = cider_press::sys::resident_bytes();
    let rss_peak = cider_press::sys::peak_resident_bytes();
    let pool = device.pool_stats();

    // ---- report ----
    let prefill_tok_s = prompt_len as f64 / prefill_dur.as_secs_f64();
    let decode_tok_s = if decode_dur.is_zero() {
        0.0
    } else {
        timed as f64 / decode_dur.as_secs_f64()
    };

    println!("cider-press bench");
    println!("  checkpoint     {}", args.checkpoint.display());
    println!("  prompt tokens  {prompt_len}");
    println!(
        "  generated      {produced} (warmup {}, timed {timed})",
        args.warmup
    );
    println!();
    println!("  load           {:>8.3} s", load_dur.as_secs_f64());
    println!(
        "  prefill        {:>8.3} ms   {prefill_tok_s:>8.1} tok/s ({prompt_len} prompt tokens)",
        prefill_dur.as_secs_f64() * 1e3,
    );
    #[allow(clippy::cast_precision_loss)]
    let prefill_sync_per = prefill_sync_dur.as_secs_f64() / prefill_iters as f64;
    #[allow(clippy::cast_precision_loss)]
    let prefill_sync_tok_s = if prefill_sync_per > 0.0 {
        prompt_len as f64 / prefill_sync_per
    } else {
        0.0
    };
    println!(
        "  prefill (sync) {:>8.3} ms   {prefill_sync_tok_s:>8.1} tok/s (mean of {} iters, eval-wait)",
        prefill_sync_per * 1e3,
        prefill_iters,
    );
    print_prefill_eval_split(&prefill_spans, prefill_iters);
    println!(
        "  decode         {:>8.3} ms   {decode_tok_s:>8.1} tok/s ({timed} tokens)",
        decode_dur.as_secs_f64() * 1e3,
    );
    println!();
    print_rss("  rss pre-load ", rss_pre_load);
    print_rss("  rss post-load", rss_post_load);
    print_rss("  rss post-dec ", rss_post_decode);
    print_rss("  rss peak     ", rss_peak);
    println!();
    print_pool_stats(&pool);
    println!();
    print_span_breakdown(&spans);

    if profile::is_enabled() {
        // Decode breakdown (from gpu_profile_token, if it ran). Labeled so
        // the prefill table below is unambiguous.
        let decode_gpu = profile::drain_gpu();
        if !decode_gpu.is_empty() {
            println!("  [decode]");
            print_gpu_breakdown(&decode_gpu);
        }
    } else if args.gpu_profile {
        println!("  (gpu breakdown unavailable; rebuild with --features profiling)");
    }

    if args.gpu_profile_prefill {
        gpu_profile_prefill(&mut generator, &ids)?;
        if profile::is_enabled() {
            let prefill_gpu = profile::drain_gpu();
            if !prefill_gpu.is_empty() {
                println!("  [prefill]");
                print_gpu_breakdown(&prefill_gpu);
            }
        }
    }
    Ok(())
}

/// After the timed window, run ONE more token through the profiled eval for a
/// per-OpKind GPU breakdown. Outside the timed window so it doesn't skew
/// tok/s; one token suffices (the counter sample buffer is sized per token).
/// The generator's caches still hold the full generated context, so this is a
/// realistic full-context decode step (T=1 at the current cache position), not
/// a cold T=1.
fn gpu_profile_token(generator: &mut Generator, last_id: u32) -> Result<(), BoxError> {
    if !profile::is_enabled() {
        eprintln!("  (--gpu-profile ignored; rebuild with --features profiling)");
        return Ok(());
    }
    generator.profiled_decode_step(last_id)?;
    Ok(())
}

/// Run ONE prefill forward (T = prompt length) through the profiled eval for
/// a per-OpKind GPU breakdown of the prefill path. Outside the timed window.
/// Resets + advances the generator's caches as a side effect (one-shot).
fn gpu_profile_prefill(generator: &mut Generator, ids: &[u32]) -> Result<(), BoxError> {
    if !profile::is_enabled() {
        eprintln!("  (--gpu-profile-prefill ignored; rebuild with --features profiling)");
        return Ok(());
    }
    generator.profiled_prefill(ids)?;
    Ok(())
}

/// Print the wall-clock span breakdown for the timed decode window. When
/// profiling is disabled the spans are empty, so print the standard note.
#[allow(clippy::cast_precision_loss)]
fn print_span_breakdown(spans: &[(&'static str, std::time::Duration, u64)]) {
    if !profile::is_enabled() {
        println!("  (span breakdown unavailable; rebuild with --features profiling)");
        return;
    }
    println!("  span breakdown (timed decode window):");
    println!(
        "    {:<18} {:>10} {:>8} {:>12}",
        "span", "total/ms", "hits", "us/hit"
    );
    for (name, total, hits) in spans {
        let total_ms = total.as_secs_f64() * 1e3;
        let us_hit = if *hits == 0 {
            0.0
        } else {
            total.as_secs_f64() * 1e6 / *hits as f64
        };
        println!("    {name:<18} {total_ms:>10.3} {hits:>8} {us_hit:>12.2}");
    }
}

/// Per-iter CPU-encode and GPU-wait microseconds from drained prefill
/// spans. `tensor.eval.encode` / `tensor.eval.wait` accumulate across all
/// `iters` synchronous prefill evals; divide the totals by `iters`. Missing
/// spans or `iters == 0` yield 0.0 (profiling off ⇒ empty drain).
#[allow(clippy::cast_precision_loss)]
fn eval_split_us(spans: &[(&'static str, std::time::Duration, u64)], iters: usize) -> (f64, f64) {
    if iters == 0 {
        return (0.0, 0.0);
    }
    let total_us = |name: &str| -> f64 {
        spans
            .iter()
            .find(|(n, _, _)| *n == name)
            .map_or(0.0, |(_, d, _)| d.as_secs_f64() * 1e6)
    };
    let n = iters as f64;
    (
        total_us("tensor.eval.encode") / n,
        total_us("tensor.eval.wait") / n,
    )
}

/// Print the prefill CPU-encode vs GPU-wait split (per-iter). The GPU-wait
/// figure is the apples-to-apples comparison point against mlx's synchronous
/// prefill; a wait near mlx with a larger total ⇒ the gap is CPU-encode.
#[allow(clippy::cast_precision_loss)]
fn print_prefill_eval_split(spans: &[(&'static str, std::time::Duration, u64)], iters: usize) {
    if !profile::is_enabled() {
        return;
    }
    let (encode_us, wait_us) = eval_split_us(spans, iters);
    println!("  prefill eval   encode (CPU) {encode_us:>7.1} us   wait (GPU) {wait_us:>7.1} us");
}

/// Print the per-OpKind GPU-time breakdown from `profiled_eval`. No-op when
/// the slice is empty (no `--gpu-profile`, or feature off).
#[allow(clippy::cast_precision_loss)]
fn print_gpu_breakdown(gpu: &[(&'static str, u64, u64)]) {
    if gpu.is_empty() {
        return;
    }
    let grand_total: u64 = gpu.iter().map(|(_, ns, _)| *ns).sum();
    println!();
    println!("  GPU breakdown (counter-sampled, one token; perturbed regime —");
    println!("  compare shares within, not against the production wait span):");
    println!(
        "    {:<20} {:>10} {:>11} {:>12} {:>8}",
        "kind", "total/ms", "dispatches", "us/dispatch", "% gpu"
    );
    for (name, ns, count) in gpu {
        let ms = *ns as f64 / 1e6;
        let us_disp = if *count == 0 {
            0.0
        } else {
            *ns as f64 / 1e3 / *count as f64
        };
        let pct = if grand_total == 0 {
            0.0
        } else {
            *ns as f64 * 100.0 / grand_total as f64
        };
        println!("    {name:<20} {ms:>10.3} {count:>11} {us_disp:>12.2} {pct:>7.1}%");
    }
}

/// Print the buffer-pool counters. Cumulative over the whole run
/// (prefill + warmup + timed decode), so the hit-rate is pessimistic
/// versus warm steady state; `high water` (max bytes simultaneously
/// held in the free-list) is the cap-sizing figure and is windowing-
/// independent.
#[allow(clippy::cast_precision_loss)]
fn print_pool_stats(pool: &cider_press_runtime::PoolStats) {
    let total = pool.hits + pool.misses;
    let hit_rate = if total == 0 {
        0.0
    } else {
        pool.hits as f64 * 100.0 / total as f64
    };
    let mib = |b: usize| b as f64 / (1024.0 * 1024.0);
    println!("  buffer pool (cumulative):");
    println!(
        "    hits {} / misses {}  ({hit_rate:.1}% hit)   high water {:>8.1} MiB",
        pool.hits,
        pool.misses,
        mib(pool.high_water_bytes),
    );
}

#[allow(clippy::cast_precision_loss)]
fn print_rss(label: &str, bytes: Option<u64>) {
    match bytes {
        Some(b) => println!("{label}  {:>8.1} MiB", b as f64 / (1024.0 * 1024.0)),
        None => println!("{label}  (unavailable)"),
    }
}

fn run() -> Result<(), BoxError> {
    match &Cli::parse().command {
        Command::Chat(args) => run_chat(args),
        Command::Bench(args) => run_bench(args),
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn eval_split_us_averages_over_iters() {
        let spans = vec![
            ("tensor.eval.encode", Duration::from_micros(1200), 4),
            ("tensor.eval.wait", Duration::from_micros(31_280), 4),
            ("tensor.eval", Duration::from_micros(32_480), 4),
        ];
        let (encode_us, wait_us) = eval_split_us(&spans, 4);
        assert!((encode_us - 300.0).abs() < 1e-6, "encode {encode_us}");
        assert!((wait_us - 7820.0).abs() < 1e-6, "wait {wait_us}");
    }

    #[test]
    fn eval_split_us_missing_spans_are_zero() {
        let (e, w) = eval_split_us(&[], 5);
        assert_eq!((e, w), (0.0, 0.0));
    }

    #[test]
    fn eval_split_us_zero_iters_is_zero() {
        let spans = vec![("tensor.eval.encode", Duration::from_millis(1), 1)];
        assert_eq!(eval_split_us(&spans, 0), (0.0, 0.0));
    }
}
