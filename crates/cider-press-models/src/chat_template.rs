//! Chat-template rendering for HF-format checkpoints.
//!
//! Loads the `chat_template` field from `tokenizer_config.json`,
//! compiles it once via `minijinja`, and renders a `&[Message]`
//! list into the prompt string the model expects.
//!
//! Lives next to [`crate::tokenizer::Tokenizer`] so the surface that
//! touches HF checkpoint files is in one place. The two types are
//! separate (different responsibilities, different deps) but the
//! caller typically constructs both.

use std::collections::HashSet;
use std::fs;
use std::path::Path;

use minijinja::{context, Environment};
use serde::Serialize;
use serde_json::Value;

use crate::error::{Error, Result};
use crate::tokenizer::Tokenizer;

/// One conversational turn.
#[derive(Debug, Clone, Serialize)]
pub struct Message {
    /// The speaker role (`"system"`, `"user"`, or `"assistant"` after
    /// jinja serialization).
    pub role: Role,
    /// The textual content of the turn.
    pub content: String,
}

impl Message {
    /// Construct a `system` turn.
    pub fn system(content: impl Into<String>) -> Self {
        Self { role: Role::System, content: content.into() }
    }
    /// Construct a `user` turn.
    pub fn user(content: impl Into<String>) -> Self {
        Self { role: Role::User, content: content.into() }
    }
    /// Construct an `assistant` turn.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self { role: Role::Assistant, content: content.into() }
    }
}

/// Conversational role serialized as the lowercase string the
/// HF chat-template convention expects (`"system"`, `"user"`,
/// `"assistant"`).
#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// System turn (instructions / persona).
    System,
    /// User turn.
    User,
    /// Assistant turn.
    Assistant,
}

/// Compiled chat template + resolved EOS ids.
#[derive(Debug)]
pub struct ChatTemplate {
    env: Environment<'static>,
    eos_ids: HashSet<u32>,
}

impl ChatTemplate {
    /// Load `tokenizer_config.json` from `path`, compile its
    /// `chat_template` field, and resolve `eos_token` / `eos_token_id`
    /// to a `HashSet<u32>` via the passed `tokenizer`.
    ///
    /// `eos_token` is accepted either as a plain string or as an
    /// `AddedToken`-shaped object (`{"content": "...", …}`); the
    /// content string is resolved via `tokenizer.encode`. If the
    /// config also has an `eos_token_id` field (single int or list),
    /// those ids are union'd in.
    pub fn from_file(path: &Path, tokenizer: &Tokenizer) -> Result<Self> {
        let bytes = fs::read(path).map_err(|e| {
            Error::InvalidArgument(format!(
                "ChatTemplate::from_file({}): {e}",
                path.display()
            ))
        })?;
        let cfg: Value = serde_json::from_slice(&bytes)?;

        let template_src = cfg
            .get("chat_template")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                Error::InvalidArgument(
                    "tokenizer_config.json: chat_template field missing or non-string".into(),
                )
            })?
            .to_string();

        let mut env = Environment::new();
        env.add_template_owned("chat", template_src)?;

        let mut eos_ids: HashSet<u32> = HashSet::new();

        // eos_token: string OR {"content": "..."} object.
        if let Some(tok) = cfg.get("eos_token") {
            let content = tok.as_str().map(str::to_string).or_else(|| {
                tok.get("content").and_then(Value::as_str).map(str::to_string)
            });
            if let Some(s) = content {
                let ids = tokenizer.encode(&s)?;
                if ids.len() != 1 {
                    return Err(Error::InvalidArgument(format!(
                        "eos_token {s:?} encoded to {} ids; expected 1",
                        ids.len()
                    )));
                }
                eos_ids.insert(ids[0]);
            }
        }

        // eos_token_id: int OR [int, int, ...]
        if let Some(v) = cfg.get("eos_token_id") {
            if let Some(n) = v.as_u64() {
                let id = u32::try_from(n).map_err(|_| {
                    Error::InvalidArgument(format!("eos_token_id value {n} exceeds u32::MAX"))
                })?;
                eos_ids.insert(id);
            } else if let Some(arr) = v.as_array() {
                for item in arr {
                    if let Some(n) = item.as_u64() {
                        let id = u32::try_from(n).map_err(|_| {
                            Error::InvalidArgument(format!(
                                "eos_token_id value {n} exceeds u32::MAX"
                            ))
                        })?;
                        eos_ids.insert(id);
                    }
                }
            }
        }

        if eos_ids.is_empty() {
            return Err(Error::InvalidArgument(
                "tokenizer_config.json: no eos_token / eos_token_id found".into(),
            ));
        }

        Ok(Self { env, eos_ids })
    }

    /// Render `messages` to the prompt string the model expects.
    ///
    /// Evaluates with `add_generation_prompt = true` so the trailing
    /// assistant-turn opener (`<|im_start|>assistant\n` for Qwen2) is
    /// appended.
    pub fn render(&self, messages: &[Message]) -> Result<String> {
        let tmpl = self.env.get_template("chat")?;
        let rendered = tmpl.render(context! {
            messages => messages,
            add_generation_prompt => true,
        })?;
        Ok(rendered)
    }

    /// Iterator over the resolved EOS token ids (union of `eos_token`
    /// and `eos_token_id` from `tokenizer_config.json`).
    pub fn eos_ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.eos_ids.iter().copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use tokenizers::models::bpe::BPE;
    use tokenizers::Tokenizer as HfTokenizer;

    fn minimal_tokenizer(dir: &Path) -> Tokenizer {
        // Put "<eos>" directly in the BPE vocab at id 0 so that
        // `tokenizer.encode("<eos>")` returns `[0]`. BPE requires at
        // least one merge or the builder treats isolated chars as the
        // model, so also provide "[UNK]" at id 1 as the unknown token.
        let vocab = [
            ("<eos>".to_string(), 0u32),
            ("[UNK]".to_string(), 1u32),
        ];
        let bpe = BPE::builder()
            .vocab_and_merges(vocab, vec![])
            .unk_token("[UNK]".to_string())
            .build()
            .expect("build minimal bpe");
        let mut hf = HfTokenizer::new(bpe);
        // Register "<eos>" as a special (non-splitting) token so the
        // tokenizer returns it as a single token rather than splitting
        // into individual characters.
        let added = hf
            .add_special_tokens([tokenizers::AddedToken::from("<eos>".to_string(), true)])
            .expect("add_special_tokens");
        assert_eq!(added, 1, "add_special_tokens failed to register <eos>");
        let path = dir.join("tokenizer.json");
        hf.save(path.to_str().unwrap(), false).expect("save tokenizer");
        Tokenizer::from_file(&path).expect("load tokenizer")
    }

    #[test]
    fn renders_messages_and_resolves_eos() {
        let dir = tempdir().expect("tempdir");
        let tok = minimal_tokenizer(dir.path());

        // Minimal jinja template: concat role+content per turn, then
        // append a generation prompt if requested.
        let template = "{% for m in messages %}<{{ m.role }}>{{ m.content }}</{{ m.role }}>{% endfor %}{% if add_generation_prompt %}<assistant>{% endif %}";

        let cfg = serde_json::json!({
            "chat_template": template,
            "eos_token": "<eos>",
        });
        let cfg_path = dir.path().join("tokenizer_config.json");
        std::fs::write(&cfg_path, serde_json::to_vec(&cfg).unwrap()).expect("write config");

        let ct = ChatTemplate::from_file(&cfg_path, &tok).expect("load template");
        let rendered = ct
            .render(&[Message::system("be brief"), Message::user("hi")])
            .expect("render");
        assert_eq!(
            rendered,
            "<system>be brief</system><user>hi</user><assistant>",
        );

        let ids: HashSet<u32> = ct.eos_ids().collect();
        assert_eq!(ids, HashSet::from([0u32]));
    }

    #[test]
    fn rejects_missing_chat_template() {
        let dir = tempdir().expect("tempdir");
        let tok = minimal_tokenizer(dir.path());
        let cfg = serde_json::json!({ "eos_token": "<eos>" });
        let cfg_path = dir.path().join("tokenizer_config.json");
        std::fs::write(&cfg_path, serde_json::to_vec(&cfg).unwrap()).expect("write config");
        let err = ChatTemplate::from_file(&cfg_path, &tok).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)), "got {err:?}");
    }

    #[test]
    fn rejects_missing_eos() {
        let dir = tempdir().expect("tempdir");
        let tok = minimal_tokenizer(dir.path());
        let cfg = serde_json::json!({ "chat_template": "{{ '' }}" });
        let cfg_path = dir.path().join("tokenizer_config.json");
        std::fs::write(&cfg_path, serde_json::to_vec(&cfg).unwrap()).expect("write config");
        let err = ChatTemplate::from_file(&cfg_path, &tok).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)), "got {err:?}");
    }

    #[test]
    fn resolves_eos_token_as_added_token_object() {
        let dir = tempdir().expect("tempdir");
        let tok = minimal_tokenizer(dir.path());

        let cfg = serde_json::json!({
            "chat_template": "{{ '' }}",
            "eos_token": {
                "content": "<eos>",
                "lstrip": false,
                "rstrip": false,
                "single_word": false,
                "special": true,
            },
        });
        let cfg_path = dir.path().join("tokenizer_config.json");
        std::fs::write(&cfg_path, serde_json::to_vec(&cfg).unwrap())
            .expect("write config");

        let ct = ChatTemplate::from_file(&cfg_path, &tok).expect("load template");
        let ids: HashSet<u32> = ct.eos_ids().collect();
        assert_eq!(ids, HashSet::from([0u32]));
    }
}
