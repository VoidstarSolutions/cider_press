//! `cider-press` CLI — load a Qwen2 checkpoint, render the prompt
//! through its chat template, greedy-decode with streamed
//! detokenization.

use std::collections::HashSet;
use std::io::{self, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;

use cider_press_models::chat_template::{ChatTemplate, Message};
use cider_press_models::generator::Generator;
use cider_press_models::qwen2::{Qwen2Config, Qwen2Model, load_qwen2_weights};
use cider_press_models::tokenizer::Tokenizer;
use cider_press_runtime::Device;
use safetensors::SafeTensors;

#[derive(Parser, Debug)]
#[command(
    name = "cider-press",
    version,
    about = "Greedy decode against a Qwen2 checkpoint"
)]
struct Args {
    /// Checkpoint directory holding config.json, tokenizer.json,
    /// `tokenizer_config.json`, and a single-file model.safetensors.
    #[arg(
        long,
        help = "Checkpoint directory holding config.json, tokenizer.json, tokenizer_config.json, and a single-file model.safetensors"
    )]
    checkpoint: PathBuf,
    /// User message.
    #[arg(long)]
    prompt: String,
    /// Optional system message; omitted means no system turn.
    /// Conflicts with --no-chat-template (no template, no role framing).
    #[arg(long, conflicts_with = "no_chat_template")]
    system: Option<String>,
    /// Maximum new tokens to generate.
    #[arg(long, default_value_t = 256)]
    max_tokens: usize,
    /// Pre-allocated `KvCache` window in tokens.
    #[arg(
        long,
        default_value_t = 4096,
        help = "Pre-allocated KvCache window in tokens"
    )]
    context_window: usize,
    /// Skip `ChatTemplate`; encode the bare --prompt.
    #[arg(long, help = "Skip ChatTemplate; encode the bare --prompt")]
    no_chat_template: bool,
}

fn run() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args = Args::parse();

    let device = Device::shared()?;

    let config_bytes = std::fs::read(args.checkpoint.join("config.json"))?;
    let config = Qwen2Config::from_json_bytes(&config_bytes)?;

    let archive_bytes = std::fs::read(args.checkpoint.join("model.safetensors"))?;
    let archive = SafeTensors::deserialize(&archive_bytes)?;
    let weights = load_qwen2_weights(&archive, &config, &device)?;
    let model = Qwen2Model::from_weights(weights, config)?;

    let tokenizer = Tokenizer::from_file(&args.checkpoint.join("tokenizer.json"))?;

    // Build the prompt + eos set. Without a chat template the user is
    // responsible for prompt framing; we still need *some* EOS, so
    // fall back to encoding the tokenizer's default eos_token via the
    // tokenizer_config.json file (always present in HF checkpoints).
    let chat_template =
        ChatTemplate::from_file(&args.checkpoint.join("tokenizer_config.json"), &tokenizer)?;
    let eos_ids: HashSet<u32> = chat_template.eos_ids().collect();

    let prompt = if args.no_chat_template {
        args.prompt.clone()
    } else {
        let mut messages = Vec::new();
        if let Some(sys) = args.system.as_ref() {
            messages.push(Message::system(sys));
        }
        messages.push(Message::user(args.prompt.clone()));
        chat_template.render(&messages)?
    };

    let ids = tokenizer.encode(&prompt)?;

    let mut generator = Generator::new(model, args.context_window, eos_ids)?;
    let mut stream = tokenizer.decode_stream(true); // skip special tokens

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

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
