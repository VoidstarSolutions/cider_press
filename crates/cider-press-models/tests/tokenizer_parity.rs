//! Gated tokenizer parity test.
//!
//! Loads `tokenizer.json` from the same checkpoint pointed at by
//! `CIDER_QWEN_CHECKPOINT_PATH`, shells out to `dump_tokenizer.py` to
//! get the HF reference ids for a fixed corpus, and asserts exact-id
//! equality plus decode round-trip on the Rust side.
//!
//! Skips with a PASS + `eprintln!` when the env var is unset.

#![cfg(target_os = "macos")]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use cider_press_models::Tokenizer;
use cider_press_test_utils::{read_u32, tempdir, workspace_root};
use safetensors::SafeTensors;

/// Same corpus the Python harness encodes — keep these two lists in lockstep.
const CORPUS: &[&str] = &[
    "Hello, world.",
    "日本語のテスト",
    " indented",
    "the quick brown fox",
    "3.14159 and 1e-6",
    "",
];

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

fn dump_tokenizer(checkpoint: &Path, out: &Path) {
    let script = workspace_root().join("scripts").join("dump_tokenizer.py");
    let status = Command::new("uv")
        .args([
            "run",
            script.to_str().expect("script utf-8"),
            "--checkpoint",
            checkpoint.to_str().expect("checkpoint utf-8"),
            "--output",
            out.to_str().expect("output utf-8"),
        ])
        .status()
        .expect("spawn uv run scripts/dump_tokenizer.py");
    assert!(
        status.success(),
        "uv run scripts/dump_tokenizer.py failed (status: {status})"
    );
}

#[test]
fn tokenizer_parity_round_trip() {
    let Some(checkpoint) = checkpoint_path() else {
        eprintln!(
            "skipped: set CIDER_QWEN_CHECKPOINT_PATH to a local \
             Qwen2.5-0.5B-Instruct-4bit MLX checkout to run this test"
        );
        return;
    };

    let tmp = tempdir("models-tokenizer-parity");
    let fixture_path = tmp.join("tokenizer-parity.safetensors");
    dump_tokenizer(&checkpoint, &fixture_path);

    let fixture_bytes = fs::read(&fixture_path).expect("read fixture");
    let st = SafeTensors::deserialize(&fixture_bytes).expect("parse safetensors");

    let tokenizer_json = checkpoint.join("tokenizer.json");
    assert!(
        tokenizer_json.is_file(),
        "expected tokenizer.json at {}",
        tokenizer_json.display(),
    );
    let tokenizer = Tokenizer::from_file(&tokenizer_json).expect("Tokenizer::from_file");

    for (i, &text) in CORPUS.iter().enumerate() {
        // meta = [ids_len, bytes_len]. Used to disambiguate "real ids
        // of length 1" vs "empty placeholder".
        let meta = read_u32(&st, &format!("corpus_{i}_meta"));
        assert_eq!(meta.len(), 2, "corpus_{i}_meta should be length 2");
        let ids_len = meta[0] as usize;

        let ids_raw = read_u32(&st, &format!("corpus_{i}_ids"));
        let expected: &[u32] = if ids_len == 0 {
            &[]
        } else {
            &ids_raw[..ids_len]
        };

        let got = tokenizer.encode(text).expect("encode");
        assert_eq!(
            got, expected,
            "corpus[{i}] = {text:?}: rust ids {got:?} != hf ids {expected:?}"
        );

        // Belt-and-suspenders: the Python harness writes the corpus bytes too;
        // confirm they round-trip cleanly so a future schema drift surfaces in
        // CI rather than silently encoding a different string than HF saw.
        let bytes_len = meta[1] as usize;
        if bytes_len > 0 {
            let bytes_view = st
                .tensor(&format!("corpus_{i}_bytes"))
                .expect("bytes tensor");
            let bytes = bytes_view.data();
            assert_eq!(
                &bytes[..bytes_len],
                text.as_bytes(),
                "corpus[{i}] bytes mismatch: python wrote different utf-8 than rust saw"
            );
        }

        // Decode round-trip: Qwen2.5 uses byte-level BPE, so encode→decode
        // is exact for the strings in this corpus.
        let decoded = tokenizer.decode(&got).expect("decode");
        assert_eq!(
            decoded, text,
            "corpus[{i}] decode round-trip mismatch: got {decoded:?} != {text:?}"
        );
    }
}
