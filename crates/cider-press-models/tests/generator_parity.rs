//! Gated end-to-end parity test for `Generator` vs `mlx_lm.generate`.
//!
//! Requires `CIDER_QWEN_CHECKPOINT_PATH` pointed at a Qwen2.5-0.5B-Instruct-4bit
//! MLX checkout. Renders a fixed messages list through both the Rust
//! and MLX-Python pipelines under greedy sampling (temp=0.0), and
//! asserts the produced token-id sequences match exactly.

#![cfg(target_os = "macos")]

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use cider_press_models::Tokenizer;
use cider_press_models::chat_template::{ChatTemplate, Message};
use cider_press_models::generator::Generator;
use cider_press_models::qwen2::{Qwen2Config, Qwen2Model, load_qwen2_weights};
use cider_press_runtime::Device;
use cider_press_test_utils::{read_u32, tempdir, workspace_root};
use safetensors::SafeTensors;

const MAX_NEW_TOKENS: usize = 16;

fn fixed_messages() -> Vec<Message> {
    vec![
        Message::system("You are a helpful assistant."),
        Message::user("What is the capital of France?"),
    ]
}

fn checkpoint_path() -> Option<PathBuf> {
    let raw = std::env::var("CIDER_QWEN_CHECKPOINT_PATH").ok()?;
    let path = fs::canonicalize(&raw)
        .unwrap_or_else(|err| panic!("CIDER_QWEN_CHECKPOINT_PATH={raw} does not resolve: {err}"));
    assert!(
        path.is_dir(),
        "CIDER_QWEN_CHECKPOINT_PATH={} is not a directory",
        path.display(),
    );
    Some(path)
}

fn run_mlx_harness(checkpoint: &Path, messages: &[Message], max_tokens: usize) -> Vec<u32> {
    let tmp = tempdir("models-generator-parity");
    let out = tmp.join("ids.safetensors");
    let script = workspace_root()
        .join("scripts")
        .join("dump_mlx_generate.py");
    let messages_json = serde_json::to_string(messages).expect("serialize messages");
    let status = Command::new("uv")
        .arg("run")
        .arg(&script)
        .arg("--checkpoint")
        .arg(checkpoint)
        .arg("--messages")
        .arg(&messages_json)
        .arg("--max-tokens")
        .arg(max_tokens.to_string())
        .arg("--output")
        .arg(&out)
        .status()
        .unwrap_or_else(|err| {
            panic!(
                "failed to invoke `uv run {}`: {err}. This test requires \
                 `uv` (https://docs.astral.sh/uv) on PATH.",
                script.display()
            )
        });
    assert!(status.success(), "dump_mlx_generate.py exited {status}");

    let bytes = fs::read(&out).expect("read mlx output safetensors");
    let st = SafeTensors::deserialize(&bytes).expect("parse safetensors");

    // meta[0] = num_ids, meta[1] = num_prompt_ids — used to disambiguate
    // 1-element placeholder arrays when either list is empty.
    let meta = read_u32(&st, "meta");
    let num_ids = meta[0] as usize;
    let all_ids = read_u32(&st, "ids");
    all_ids[..num_ids].to_vec()
}

#[test]
fn generator_parity_greedy_qwen2() {
    let Some(checkpoint) = checkpoint_path() else {
        eprintln!(
            "skipped: set CIDER_QWEN_CHECKPOINT_PATH to a local \
             Qwen2.5-0.5B-Instruct-4bit MLX checkout to run this test"
        );
        return;
    };

    let messages = fixed_messages();

    // ── Rust side ─────────────────────────────────────────────────────────────
    let device = Device::shared().expect("device");

    let config_bytes = fs::read(checkpoint.join("config.json")).expect("read config.json");
    let config = Qwen2Config::from_json_bytes(&config_bytes).expect("parse config.json");

    let archive_bytes =
        fs::read(checkpoint.join("model.safetensors")).expect("read model.safetensors");
    let archive = SafeTensors::deserialize(&archive_bytes).expect("deserialize safetensors");

    let weights = load_qwen2_weights(&archive, &config, &device).expect("load weights");
    let model = Qwen2Model::from_weights(weights, config.clone()).expect("build Qwen2Model");

    let tokenizer =
        Tokenizer::from_file(&checkpoint.join("tokenizer.json")).expect("load tokenizer");
    let chat_template =
        ChatTemplate::from_file(&checkpoint.join("tokenizer_config.json"), &tokenizer)
            .expect("load chat template");

    let prompt = chat_template.render(&messages).expect("render prompt");
    let ids = tokenizer.encode(&prompt).expect("encode prompt");

    let eos: HashSet<u32> = chat_template.eos_ids().collect();
    let mut generator =
        Generator::new(model, ids.len() + MAX_NEW_TOKENS + 1, eos).expect("Generator::new");

    let rust_ids: Vec<u32> = generator
        .generate(&ids, MAX_NEW_TOKENS)
        .expect("generate")
        .collect::<Result<_, _>>()
        .expect("collect generated ids");

    // ── MLX side ──────────────────────────────────────────────────────────────
    let mlx_ids = run_mlx_harness(&checkpoint, &messages, MAX_NEW_TOKENS);

    assert_eq!(
        rust_ids, mlx_ids,
        "rust_ids != mlx_ids\n  rust = {rust_ids:?}\n  mlx  = {mlx_ids:?}",
    );
}
