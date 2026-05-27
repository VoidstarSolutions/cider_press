//! Thin wrapper around the `HuggingFace` `tokenizers` crate.
//!
//! Loads a `tokenizer.json` (the file shape HF / mlx-community
//! checkpoints ship next to `config.json`) and exposes bare
//! `encode` / `decode`. Special-token handling and chat-template
//! rendering belong to the caller (branch 14's CLI / decode loop).

use std::path::Path;

use crate::error::{Error, Result};

/// Tokenizer loaded from a `tokenizer.json`, wrapping `tokenizers::Tokenizer`.
///
/// Construct once per process (`from_file` parses the entire
/// `tokenizer.json` up front) and share by reference; `encode` /
/// `decode` are pure functions over `&self`.
#[derive(Debug)]
pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
}

impl Tokenizer {
    /// Load from a `HuggingFace` `tokenizer.json` file.
    pub fn from_file(path: &Path) -> Result<Self> {
        let inner = tokenizers::Tokenizer::from_file(path).map_err(Error::from)?;
        Ok(Self { inner })
    }

    /// Encode raw text to token ids. `add_special_tokens=false` — the
    /// caller is responsible for any BOS/EOS framing it needs.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self.inner.encode(text, false).map_err(Error::from)?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decode token ids back to text. `skip_special_tokens=false` so
    /// the round-trip is exact for the parity test.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.inner.decode(ids, false).map_err(Error::from)
    }
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;
    use tokenizers::models::bpe::BPE;
    use tokenizers::Tokenizer as HfTokenizer;

    use super::*;

    /// Build a minimal BPE tokenizer in-memory and serialize it to a
    /// tempdir to drive `Tokenizer::from_file` through its real load
    /// path. Vocab covers exactly the bytes used in the round-trip
    /// assertion below.
    fn write_minimal_tokenizer(dir: &std::path::Path) -> std::path::PathBuf {
        // BpeBuilder::vocab_and_merges requires AHashMap; build from array.
        let vocab = [
            ("h".to_string(), 0u32),
            ("i".to_string(), 1u32),
            ("[UNK]".to_string(), 2u32),
        ];
        let bpe = BPE::builder()
            .vocab_and_merges(vocab, vec![])
            .unk_token("[UNK]".to_string())
            .build()
            .expect("build BPE");
        let tok = HfTokenizer::new(bpe);
        let path = dir.join("tokenizer.json");
        tok.save(&path, false).expect("save tokenizer.json");
        path
    }

    #[test]
    fn from_file_encode_decode_round_trip() {
        let tmp = TempDir::new().expect("tempdir");
        let path = write_minimal_tokenizer(tmp.path());

        let tok = Tokenizer::from_file(&path).expect("from_file");
        let ids = tok.encode("hi").expect("encode");
        assert!(!ids.is_empty(), "encode should produce at least one id");
        let text = tok.decode(&ids).expect("decode");
        assert!(
            text.contains('h') && text.contains('i'),
            "decode should round-trip both characters; got {text:?}"
        );
    }
}
