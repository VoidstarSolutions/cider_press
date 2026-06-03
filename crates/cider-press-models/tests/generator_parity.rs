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

/// What the MLX harness produced: the sampled token ids and the exact
/// EOS set `mlx_lm` stopped on. The Rust side drives its `Generator` with
/// this same set so a parity mismatch can only mean a decode divergence,
/// never a termination-policy one.
struct MlxRun {
    ids: Vec<u32>,
    eos: HashSet<u32>,
}

fn run_mlx_harness(checkpoint: &Path, messages: &[Message], max_tokens: usize) -> MlxRun {
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

    // meta[0] = num_ids, meta[1] = num_prompt_ids, meta[2] = num_eos —
    // used to disambiguate 1-element placeholder arrays when a list is
    // empty.
    let meta = read_u32(&st, "meta");
    let num_ids = meta[0] as usize;
    let num_eos = meta[2] as usize;
    let ids = read_u32(&st, "ids")[..num_ids].to_vec();
    let eos: HashSet<u32> = read_u32(&st, "eos_ids")[..num_eos]
        .iter()
        .copied()
        .collect();
    MlxRun { ids, eos }
}

/// Build a fresh Generator from the checkpoint (loads weights anew each
/// call, since `Generator::new` consumes the model). Shared by the
/// depth-invariance test.
fn build_generator(checkpoint: &Path, context_window: usize, eos: HashSet<u32>) -> Generator {
    let device = Device::shared().expect("device");
    let config_bytes = fs::read(checkpoint.join("config.json")).expect("read config.json");
    let config = Qwen2Config::from_json_bytes(&config_bytes).expect("parse config.json");
    let archive_bytes =
        fs::read(checkpoint.join("model.safetensors")).expect("read model.safetensors");
    let archive = SafeTensors::deserialize(&archive_bytes).expect("deserialize safetensors");
    let weights = load_qwen2_weights(&archive, &config, &device).expect("load weights");
    let model = Qwen2Model::from_weights(weights, config).expect("build Qwen2Model");
    Generator::new(model, context_window, eos).expect("Generator::new")
}

fn prompt_ids(checkpoint: &Path) -> Vec<u32> {
    let tokenizer =
        Tokenizer::from_file(&checkpoint.join("tokenizer.json")).expect("load tokenizer");
    let chat_template =
        ChatTemplate::from_file(&checkpoint.join("tokenizer_config.json"), &tokenizer)
            .expect("load chat template");
    let prompt = chat_template.render(&fixed_messages()).expect("render prompt");
    tokenizer.encode(&prompt).expect("encode prompt")
}

/// Greedy decode is deterministic regardless of pipeline depth: deeper
/// lookahead commits more tokens ahead but must yield the identical
/// sequence. Exercises on-GPU id chaining at depths 0, 1, 2, and 3,
/// including depth 0 (fully synchronous) which is the regression case.
#[test]
fn greedy_output_is_depth_invariant() {
    let Some(checkpoint) = checkpoint_path() else {
        eprintln!("skipped: set CIDER_QWEN_CHECKPOINT_PATH to run this test");
        return;
    };
    let ids = prompt_ids(&checkpoint);
    let max_new = 24usize;
    let ctx = ids.len() + max_new + 1;
    // Qwen2.5 <|im_end|> = 151645; any fixed non-empty EOS works — all
    // depths use the same set, so they stop (if at all) at the same token.
    let eos: HashSet<u32> = HashSet::from([151_645_u32]);

    let mut sequences = Vec::new();
    for depth in [0usize, 1, 2, 3] {
        let mut g = build_generator(&checkpoint, ctx, eos.clone());
        g.set_inflight_depth(depth);
        let seq: Vec<u32> = g
            .generate(&ids, max_new)
            .expect("gen")
            .collect::<Result<_, _>>()
            .expect("collect");
        sequences.push((depth, seq));
    }
    let (_, ref baseline) = sequences[0];
    for (depth, seq) in &sequences {
        assert_eq!(seq, baseline, "depth {depth} output differs from depth 0");
        assert!(!seq.is_empty(), "depth {depth} produced no tokens");
    }
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

    // Compute Rust-side prompt ids before the MLX harness runs.
    let ids = prompt_ids(&checkpoint);

    // ── MLX side ──────────────────────────────────────────────────────────────
    // Run first so the Rust Generator can adopt MLX's exact EOS set —
    // any parity mismatch is then a decode divergence, not a different
    // stop policy. (chat_template only resolves tokenizer_config.json's
    // eos_token; mlx_lm also folds in generation_config.json, so the two
    // sets are not guaranteed equal.)
    let MlxRun {
        ids: mlx_ids,
        eos: mlx_eos,
    } = run_mlx_harness(&checkpoint, &messages, MAX_NEW_TOKENS);

    let mut generator = build_generator(&checkpoint, ids.len() + MAX_NEW_TOKENS + 1, mlx_eos);

    let rust_ids: Vec<u32> = generator
        .generate(&ids, MAX_NEW_TOKENS)
        .expect("generate")
        .collect::<Result<_, _>>()
        .expect("collect generated ids");

    assert_eq!(
        rust_ids, mlx_ids,
        "rust_ids != mlx_ids\n  rust = {rust_ids:?}\n  mlx  = {mlx_ids:?}",
    );
}
