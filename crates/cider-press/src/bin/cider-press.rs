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
    #[arg(long)]
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
    #[arg(long)]
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
    /// Skip `ChatTemplate` rendering; encode the bare --prompt.
    #[arg(long)]
    no_chat_template: bool,
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

#[allow(clippy::cast_precision_loss)]
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
    let mut decode_timer: Option<Instant> = None;
    for id_result in iter {
        let _ = id_result?;
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

    let rss_post_decode = cider_press::sys::resident_bytes();
    let rss_peak = cider_press::sys::peak_resident_bytes();
    let spans = profile::drain();

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
    if profile::is_enabled() {
        println!("  span breakdown (timed decode window):");
        println!(
            "    {:<18} {:>10} {:>8} {:>12}",
            "span", "total/ms", "hits", "us/hit"
        );
        for (name, total, hits) in &spans {
            let total_ms = total.as_secs_f64() * 1e3;
            let us_hit = if *hits == 0 {
                0.0
            } else {
                total.as_secs_f64() * 1e6 / *hits as f64
            };
            println!("    {name:<18} {total_ms:>10.3} {hits:>8} {us_hit:>12.2}");
        }
    } else {
        println!("  (span breakdown unavailable; rebuild with --features profiling)");
    }
    Ok(())
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
