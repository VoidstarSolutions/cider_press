# Branch 14 — `feat/cli + greedy decode` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire `Qwen2Model` + `Tokenizer` into a reusable `Generator` + `ChatTemplate` and surface them through a `cider-press` CLI binary that greedy-decodes a chat-templated prompt with streamed detokenization until EOS or `--max-tokens`.

**Architecture:** Two new modules in `cider-press-models` (`chat_template`, `generator`) plus one new binary in the facade crate (`crates/cider-press/src/bin/cider-press.rs`). Generator is concrete-on-`Qwen2Model` and yields `Result<u32>` ids; the CLI composes `Tokenizer::decode_stream` on top for streamed printing. Chat template is jinja-rendered via `minijinja`.

**Tech Stack:** Rust 2024, `clap` (=4.5.21, derive), `minijinja` (=2.5.0), `tokenizers` (existing), `cider-press-runtime` + `cider-press-models` (existing).

**Spec:** `docs/superpowers/specs/2026-05-28-branch-14-cli-greedy-decode-design.md`

---

## Known API references (verified before writing this plan)

Use these names verbatim; they're what the existing tests and modules
already use.

| Surface | Signature |
|---|---|
| Device | `Device::shared() -> Result<Device>` (the test convention; `Device::system_default` also exists but tests use `shared`) |
| Tensor | `Tensor::from_slice<T: Scalar>(&Device, &[T], impl Into<Shape>) -> Result<Tensor>` — generic, no `_u32` / `_i32` variants |
| Tensor | `Tensor::cpu_to_vec<T: Scalar>(&self) -> Option<Vec<T>>` — `Option`, not `Result` |
| KvCache | `KvCache::new(&Device, max_tokens, n_kv_heads, head_dim, DType) -> Result<KvCache>` — 5 args including the dtype |
| Quantization | `Quantization::Q4_GS64` (associated const), or `Quantization::new(bits, group_size)` |
| Qwen2 loader | `cider_press_models::qwen2::load_qwen2_weights(&SafeTensors, &Qwen2Config, &Device) -> Result<Qwen2Weights>` — takes a deserialized archive, not a path. Caller reads `<checkpoint>/model.safetensors` and `SafeTensors::deserialize`s it. |
| DType | `cider_press_runtime::DType::{BF16, F32, F16, U32, I32}` |
| Tokenizer | `Tokenizer::from_file`, `encode`, `decode` — **no `decode_stream` yet**; this branch adds it. |

---

## File map

**New**
- `crates/cider-press-models/src/chat_template.rs` — `ChatTemplate`, `Role`, `Message`
- `crates/cider-press-models/src/generator.rs` — `Generator`, `GenerateIter`
- `crates/cider-press-models/tests/generator_smoke.rs` — synthetic-model smoke
- `crates/cider-press-models/tests/generator_parity.rs` — gated MLX parity
- `crates/cider-press/src/bin/cider-press.rs` — CLI
- `scripts/dump_mlx_generate.py` — MLX greedy-generate harness

**Modified**
- `Cargo.toml` — workspace deps (`clap`, `minijinja`)
- `crates/cider-press-models/Cargo.toml` — `minijinja`
- `crates/cider-press/Cargo.toml` — `clap`, runtime/models deps, `[[bin]]`
- `crates/cider-press-models/src/lib.rs` — module exports
- `crates/cider-press-models/src/nn.rs` — `pub fn causal_mask`
- `crates/cider-press-models/src/error.rs` — `Error::ChatTemplate`
- `crates/cider-press-models/tests/qwen2_logits.rs` — consume `nn::causal_mask`
- `docs/QWEN_PATH.md` — mark branch 14 done

---

## Task 1: Workspace dep adds (clap + minijinja)

**Files:**
- Modify: `Cargo.toml` (workspace `[workspace.dependencies]`)
- Modify: `crates/cider-press-models/Cargo.toml`
- Modify: `crates/cider-press/Cargo.toml`

- [ ] **Step 1.1: Add the two workspace deps**

Open `Cargo.toml` and add to `[workspace.dependencies]` (after the existing `tokenizers` line; keep block in roughly alphabetical order):

```toml
clap = { version = "=4.5.21", features = ["derive"] }
minijinja = "=2.5.0"
```

- [ ] **Step 1.2: Wire `minijinja` into the models crate**

In `crates/cider-press-models/Cargo.toml`, add to `[dependencies]` (after `tokenizers`):

```toml
minijinja = { workspace = true }
```

- [ ] **Step 1.3: Wire `clap` + runtime/models into the facade crate**

In `crates/cider-press/Cargo.toml`, add to `[dependencies]` (after the existing `cider-press-models` line):

```toml
clap = { workspace = true }
safetensors = { workspace = true }
```

The facade already depends on `cider-press-runtime` and `cider-press-models`; the bin needs `safetensors` directly to deserialize the checkpoint archive before handing it to `load_qwen2_weights`.

Append at the bottom (after `[lints]`):

```toml
[[bin]]
name = "cider-press"
path = "src/bin/cider-press.rs"
```

- [ ] **Step 1.4: Verify workspace resolves**

Run: `cargo check --workspace`
Expected: PASS (warnings about unused deps in facade crate are fine — bin file lands later).

- [ ] **Step 1.5: Commit**

```bash
git add Cargo.toml Cargo.lock crates/cider-press-models/Cargo.toml crates/cider-press/Cargo.toml
git commit -m "$(cat <<'EOF'
build(workspace): add clap + minijinja for branch 14 (CLI + chat template)

clap derive backs the new cider-press CLI binary; minijinja drives the
chat-template rendering inside cider-press-models. Pinned with `=` per
project convention. Facade crate gains a [[bin]] entry; bin source
lands in a later task.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Lift `causal_mask` into `nn.rs`

**Files:**
- Modify: `crates/cider-press-models/src/nn.rs` — add `pub fn causal_mask`
- Modify: `crates/cider-press-models/tests/qwen2_logits.rs` — consume new helper

The current `causal_mask_bf16` lives privately inside `tests/qwen2_logits.rs`. Lift it to a public free function in `nn.rs` so both the existing test and the new `Generator` consume the same construction. Rank is rank-2 `[T, T]` — broadcast-relies-on consumers per the existing test's comment.

- [ ] **Step 2.1: Add `pub fn causal_mask` to `nn.rs`**

At the bottom of `crates/cider-press-models/src/nn.rs`, append:

```rust
/// Build an additive causal attention mask of shape `[T, T]`, BF16.
///
/// `0.0` on/below the diagonal, `-1e4` above. The `-1e4` magnitude
/// keeps the softmax accumulator inside bf16's representable range
/// while still acting as a hard mask (softmax subtracts the max
/// before exp).
///
/// Rank-2 because `sdpa`'s scores tensor is rank-3
/// `[B*H_q, T, T_cache]`; the wired binary path is `g3_*`. Rank-4
/// strided binary is deferred.
pub fn causal_mask(device: &cider_press_runtime::Device, t: usize)
    -> cider_press_runtime::Result<cider_press_runtime::Tensor>
{
    use half::bf16;
    let neg = bf16::from_f32(-1.0e4);
    let zero = bf16::ZERO;
    let mut data = vec![zero; t * t];
    for row in 0..t {
        for col in (row + 1)..t {
            data[row * t + col] = neg;
        }
    }
    cider_press_runtime::Tensor::from_slice(device, &data, [t, t])
}
```

(If `nn.rs` already imports `half::bf16` or `cider_press_runtime::*` at module scope, drop the local `use` and the qualified paths accordingly to match.)

- [ ] **Step 2.2: Consume in `qwen2_logits.rs`**

In `crates/cider-press-models/tests/qwen2_logits.rs`, delete the private `causal_mask_bf16` helper (and its surrounding doc comment) and replace the call site:

Before:
```rust
let mask = causal_mask_bf16(&device, t);
```

After:
```rust
let mask = cider_press_models::nn::causal_mask(&device, t)
    .expect("causal mask");
```

If `cider_press_models::nn` is not already imported in the test, add at the top of the file:

```rust
use cider_press_models::nn;
```

and use `nn::causal_mask(&device, t)`. Whichever style matches the file's existing import convention.

- [ ] **Step 2.3: Run the existing test (compiles only — it's gated)**

Run: `cargo test -p cider-press-models --test qwen2_logits --no-run`
Expected: builds clean, no warnings about the deleted helper.

- [ ] **Step 2.4: Run the full models suite to confirm nothing else broke**

Run: `cargo test -p cider-press-models`
Expected: all non-gated tests PASS.

- [ ] **Step 2.5: Commit**

```bash
git add crates/cider-press-models/src/nn.rs crates/cider-press-models/tests/qwen2_logits.rs
git commit -m "$(cat <<'EOF'
refactor(models): lift causal_mask helper from test into nn module

Branch 14's Generator needs the same explicit `[T, T]` BF16 causal
mask construction that qwen2_logits already builds privately. Move
the helper into nn.rs as `pub fn causal_mask` and update the test to
consume it. Same rank, same -1e4 magnitude.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Add `Error::ChatTemplate` variant

**Files:**
- Modify: `crates/cider-press-models/src/error.rs`

- [ ] **Step 3.1: Add the variant**

In `crates/cider-press-models/src/error.rs`, add inside the `pub enum Error { … }` block (after the existing `Tokenizer` variant, before `Runtime`):

```rust
    /// A chat template (minijinja) failed to load, compile, or render.
    #[error("chat template: {0}")]
    ChatTemplate(#[from] minijinja::Error),
```

- [ ] **Step 3.2: Verify build**

Run: `cargo build -p cider-press-models`
Expected: PASS.

- [ ] **Step 3.3: Commit**

```bash
git add crates/cider-press-models/src/error.rs
git commit -m "$(cat <<'EOF'
feat(models): Error::ChatTemplate variant for branch 14

minijinja::Error wraps load, compile, and render failures from the
upcoming ChatTemplate module. #[from] mirrors the existing pattern
for Tokenizer / Safetensors / Json / Runtime variants.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: `ChatTemplate` module (TDD)

**Files:**
- Create: `crates/cider-press-models/src/chat_template.rs`
- Modify: `crates/cider-press-models/src/lib.rs` (`pub mod chat_template;`)

Test-first. The unit test lives inside the new module under a `#[cfg(test)] mod tests` block (matches the pattern in `tokenizer.rs`).

- [ ] **Step 4.1: Skeleton + failing test**

Create `crates/cider-press-models/src/chat_template.rs`:

```rust
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
        let bytes = fs::read(path).map_err(|e| Error::InvalidArgument(
            format!("ChatTemplate::from_file({}): {e}", path.display()),
        ))?;
        let cfg: Value = serde_json::from_slice(&bytes)?;

        let template_src = cfg.get("chat_template")
            .and_then(Value::as_str)
            .ok_or_else(|| Error::InvalidArgument(
                "tokenizer_config.json: chat_template field missing or non-string".into(),
            ))?
            .to_string();

        let mut env = Environment::new();
        env.add_template_owned("chat", template_src)?;

        let mut eos_ids: HashSet<u32> = HashSet::new();

        // eos_token: string OR {"content": "..."} object.
        if let Some(tok) = cfg.get("eos_token") {
            let content = if let Some(s) = tok.as_str() {
                Some(s.to_string())
            } else if let Some(s) = tok.get("content").and_then(Value::as_str) {
                Some(s.to_string())
            } else {
                None
            };
            if let Some(s) = content {
                let ids = tokenizer.encode(&s)?;
                if ids.len() != 1 {
                    return Err(Error::InvalidArgument(format!(
                        "eos_token {s:?} encoded to {} ids; expected 1", ids.len()
                    )));
                }
                eos_ids.insert(ids[0]);
            }
        }

        // eos_token_id: int OR [int, int, ...]
        if let Some(v) = cfg.get("eos_token_id") {
            if let Some(n) = v.as_u64() {
                eos_ids.insert(n as u32);
            } else if let Some(arr) = v.as_array() {
                for item in arr {
                    if let Some(n) = item.as_u64() {
                        eos_ids.insert(n as u32);
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
    use tokenizers::models::bpe::BpeBuilder;
    use tokenizers::Tokenizer as HfTokenizer;

    fn minimal_tokenizer(dir: &Path) -> Tokenizer {
        // Single-token vocab: "<eos>" => 0. Enough for from_file/encode
        // to resolve the eos string to id 0.
        let vocab = std::collections::HashMap::from([("<eos>".to_string(), 0u32)]);
        let bpe = BpeBuilder::new()
            .vocab_and_merges(vocab, vec![])
            .build()
            .expect("build minimal bpe");
        let hf = HfTokenizer::new(bpe);
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
        std::fs::write(&cfg_path, serde_json::to_vec(&cfg).unwrap())
            .expect("write config");

        let ct = ChatTemplate::from_file(&cfg_path, &tok).expect("load template");
        let rendered = ct.render(&[
            Message::system("be brief"),
            Message::user("hi"),
        ]).expect("render");
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
        std::fs::write(&cfg_path, serde_json::to_vec(&cfg).unwrap())
            .expect("write config");
        let err = ChatTemplate::from_file(&cfg_path, &tok).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)), "got {err:?}");
    }

    #[test]
    fn rejects_missing_eos() {
        let dir = tempdir().expect("tempdir");
        let tok = minimal_tokenizer(dir.path());
        let cfg = serde_json::json!({ "chat_template": "{{ '' }}" });
        let cfg_path = dir.path().join("tokenizer_config.json");
        std::fs::write(&cfg_path, serde_json::to_vec(&cfg).unwrap())
            .expect("write config");
        let err = ChatTemplate::from_file(&cfg_path, &tok).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)), "got {err:?}");
    }
}
```

- [ ] **Step 4.2: Wire module into `lib.rs`**

In `crates/cider-press-models/src/lib.rs`, add (next to the other `pub mod` lines, alphabetical):

```rust
pub mod chat_template;
```

- [ ] **Step 4.3: Run tests — confirm they PASS**

Run: `cargo test -p cider-press-models --lib chat_template`
Expected: 3 tests pass (`renders_messages_and_resolves_eos`, `rejects_missing_chat_template`, `rejects_missing_eos`).

- [ ] **Step 4.4: Commit**

```bash
git add crates/cider-press-models/src/chat_template.rs crates/cider-press-models/src/lib.rs
git commit -m "$(cat <<'EOF'
feat(models): ChatTemplate wrapper over minijinja + tokenizer_config.json

Loads chat_template, eos_token (string OR AddedToken object), and
optional eos_token_id (int OR list) from tokenizer_config.json. Renders
[{role, content}] messages with add_generation_prompt=true, matching
the HF chat-template convention. Unit tests cover happy path + the two
required-field rejections.

Branch 14 surface. Generator consumes eos_ids() in a later task.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: `Generator` module (TDD via synthetic-model smoke)

**Files:**
- Create: `crates/cider-press-models/src/generator.rs`
- Create: `crates/cider-press-models/tests/generator_smoke.rs`
- Modify: `crates/cider-press-models/src/lib.rs` (`pub mod generator;`)

The smoke test builds a tiny synthetic `Qwen2Model` (random weights at hidden=64, vocab=128, layers=2, heads=2, kv_heads=1, head_dim=32) and exercises the iterator semantics end-to-end on real GPU dispatches. No mocking of `forward`.

- [ ] **Step 5.1: Write the failing smoke test first**

Create `crates/cider-press-models/tests/generator_smoke.rs`:

```rust
//! Non-gated smoke test for the `Generator` iterator semantics.
//!
//! Builds a tiny synthetic Qwen2Model from random bytes (no real
//! checkpoint), constructs a Generator over it, and verifies:
//!  - the iterator yields `max_new_tokens` ids when EOS is unreachable;
//!  - the iterator terminates early (yielding exactly 1 id) when the
//!    first sampled id is forced into the eos_ids set;
//!  - validation rejects `context_window = 0` and empty `eos_ids`.

use std::collections::HashSet;

use cider_press_models::generator::Generator;
use cider_press_models::qwen2::{
    Qwen2AttentionWeights, Qwen2Config, Qwen2LayerWeights, Qwen2MlpWeights,
    Qwen2Model, Qwen2Weights,
};
use cider_press_runtime::{Device, Quantization, QuantizedWeight, Tensor};
use half::bf16;

const HIDDEN: usize = 64;
const VOCAB: usize = 128;
const LAYERS: usize = 2;
const HEADS: usize = 2;
const KV_HEADS: usize = 1;
const HEAD_DIM: usize = 32;
const INTERMEDIATE: usize = 128;
const GROUP_SIZE: usize = 64;
const BITS: usize = 4;

fn synthetic_config() -> Qwen2Config {
    // Qwen2Config and Qwen2QuantizationConfig are #[non_exhaustive],
    // so external code cannot use struct-literal construction. Build
    // through the existing JSON parser instead.
    let json = serde_json::json!({
        "hidden_size": HIDDEN,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "intermediate_size": INTERMEDIATE,
        "vocab_size": VOCAB,
        "max_position_embeddings": 256,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1.0e-6,
        "tie_word_embeddings": true,
        "quantization": {
            "group_size": GROUP_SIZE,
            "bits": BITS,
        },
    });
    Qwen2Config::from_json_bytes(&serde_json::to_vec(&json).unwrap())
        .expect("synthetic Qwen2Config")
}

/// Deterministic-but-non-trivial byte pattern (avoids all-zero weights
/// which produce all-zero logits and ambiguous argmaxes).
fn fill(seed: u8, len: usize) -> Vec<u8> {
    (0..len).map(|i| seed.wrapping_add(i as u8 ^ 0xA5)).collect()
}

fn random_qweight(device: &Device, n: usize, k: usize, seed: u8) -> QuantizedWeight {
    let q = Quantization::Q4_GS64; // 4-bit, group_size=64
    let elem_count = n * k;
    let w_bytes = (elem_count * BITS / 8).max(4);
    let aux_bytes = (elem_count / GROUP_SIZE) * std::mem::size_of::<bf16>();
    QuantizedWeight::from_bytes(
        device,
        [n, k],
        q,
        &fill(seed, w_bytes),
        &fill(seed.wrapping_add(1), aux_bytes),
        &fill(seed.wrapping_add(2), aux_bytes),
    )
    .expect("from_bytes")
}

fn random_dense(device: &Device, dim: usize, seed: u8) -> Tensor {
    let data: Vec<bf16> = (0..dim)
        .map(|i| bf16::from_f32(((seed as i32 + i as i32) as f32) * 1e-3))
        .collect();
    Tensor::from_slice(device, &data, [dim]).expect("from_slice")
}

fn synthetic_model(device: &Device) -> Qwen2Model {
    let cfg = synthetic_config();
    let q_dim = HEADS * HEAD_DIM;
    let kv_dim = KV_HEADS * HEAD_DIM;
    let layers = (0..LAYERS)
        .map(|li| {
            let s = (li as u8) * 16;
            Qwen2LayerWeights {
                input_layernorm: random_dense(device, HIDDEN, s + 1),
                post_attention_layernorm: random_dense(device, HIDDEN, s + 2),
                attention: Qwen2AttentionWeights {
                    q_proj: random_qweight(device, q_dim, HIDDEN, s + 3),
                    k_proj: random_qweight(device, kv_dim, HIDDEN, s + 4),
                    v_proj: random_qweight(device, kv_dim, HIDDEN, s + 5),
                    o_proj: random_qweight(device, HIDDEN, q_dim, s + 6),
                    q_bias: random_dense(device, q_dim, s + 7),
                    k_bias: random_dense(device, kv_dim, s + 8),
                    v_bias: random_dense(device, kv_dim, s + 9),
                },
                mlp: Qwen2MlpWeights {
                    gate_proj: random_qweight(device, INTERMEDIATE, HIDDEN, s + 10),
                    up_proj: random_qweight(device, INTERMEDIATE, HIDDEN, s + 11),
                    down_proj: random_qweight(device, HIDDEN, INTERMEDIATE, s + 12),
                },
            }
        })
        .collect();
    let weights = Qwen2Weights {
        embed: random_qweight(device, VOCAB, HIDDEN, 100),
        norm: random_dense(device, HIDDEN, 200),
        layers,
    };
    Qwen2Model::from_weights(weights, cfg).expect("from_weights")
}

#[test]
fn generator_yields_max_new_tokens_when_eos_unreachable() {
    let device = Device::shared().expect("device");
    let model = synthetic_model(&device);
    let mut gen = Generator::new(model, 64, HashSet::from([u32::MAX]))
        .expect("Generator::new");
    let ids: Vec<u32> = gen
        .generate(&[1, 2, 3], 4)
        .expect("generate")
        .collect::<Result<_, _>>()
        .expect("collect");
    assert_eq!(ids.len(), 4);
}

#[test]
fn generator_terminates_when_first_id_is_eos() {
    let device = Device::shared().expect("device");
    let model = synthetic_model(&device);
    // First pass: see what the first sampled id will be.
    let mut gen1 = Generator::new(model, 64, HashSet::from([u32::MAX]))
        .expect("Generator::new");
    let first = gen1
        .generate(&[1, 2, 3], 1)
        .expect("generate")
        .next()
        .expect("first id")
        .expect("first id ok");
    // Rebuild with that id as the only EOS; should yield exactly it then stop.
    let model = synthetic_model(&device);
    let mut gen2 = Generator::new(model, 64, HashSet::from([first]))
        .expect("Generator::new");
    let ids: Vec<u32> = gen2
        .generate(&[1, 2, 3], 8)
        .expect("generate")
        .collect::<Result<_, _>>()
        .expect("collect");
    assert_eq!(ids, vec![first]);
}

#[test]
fn generator_rejects_zero_context_window() {
    let device = Device::shared().expect("device");
    let model = synthetic_model(&device);
    let err = Generator::new(model, 0, HashSet::from([0u32])).unwrap_err();
    assert!(format!("{err}").contains("context_window"), "got: {err}");
}

#[test]
fn generator_rejects_empty_eos_ids() {
    let device = Device::shared().expect("device");
    let model = synthetic_model(&device);
    let err = Generator::new(model, 64, HashSet::new()).unwrap_err();
    assert!(format!("{err}").contains("eos_ids"), "got: {err}");
}
```

- [ ] **Step 5.2: Try to run the test — confirm it fails at compile time**

Run: `cargo test -p cider-press-models --test generator_smoke --no-run`
Expected: FAIL with "unresolved import `cider_press_models::generator`" or similar (the module doesn't exist yet).

- [ ] **Step 5.3: Implement `Generator`**

Create `crates/cider-press-models/src/generator.rs`:

```rust
//! Greedy-decode driver over [`Qwen2Model`] + [`KvCache`].
//!
//! Concrete-on-Qwen2Model for now; lifting to a `LanguageModel`
//! trait waits until a second architecture arrives. The iterator
//! yields `Result<u32>` ids; UTF-8 detokenization is the caller's
//! problem (compose `Tokenizer::decode_stream` on top).

use std::collections::HashSet;

use cider_press_runtime::{DType, Device, KvCache, Tensor};
use half::bf16;

use crate::error::{Error, Result};
use crate::nn::causal_mask;
use crate::qwen2::Qwen2Model;

/// Owns a model + pre-allocated KvCaches + the EOS set. One instance
/// drives one or more `generate` calls; caches are reset on every
/// call so previous runs don't leak into the next prompt.
pub struct Generator {
    model: Qwen2Model,
    caches: Vec<KvCache>,
    eos_ids: HashSet<u32>,
    context_window: usize,
}

impl Generator {
    /// Construct a Generator. Pre-allocates one [`KvCache`] per layer
    /// sized to `context_window` tokens.
    ///
    /// # Errors
    /// - `context_window == 0`
    /// - `eos_ids.is_empty()` — without an EOS the loop has no
    ///   natural terminator besides `max_new_tokens`; we reject so a
    ///   caller can't accidentally rely on that.
    pub fn new(
        model: Qwen2Model,
        context_window: usize,
        eos_ids: HashSet<u32>,
    ) -> Result<Self> {
        if context_window == 0 {
            return Err(Error::InvalidArgument(
                "Generator::new: context_window must be > 0".into(),
            ));
        }
        if eos_ids.is_empty() {
            return Err(Error::InvalidArgument(
                "Generator::new: eos_ids must be non-empty".into(),
            ));
        }
        let device = Device::shared()?;
        let cfg = model.config();
        let head_dim = cfg.head_dim()?;
        let caches = (0..cfg.num_hidden_layers)
            .map(|_| KvCache::new(
                &device,
                context_window,
                cfg.num_key_value_heads,
                head_dim,
                DType::BF16,
            ))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(Self { model, caches, eos_ids, context_window })
    }

    /// Reset all KvCaches to position 0. Called automatically at the
    /// start of [`Self::generate`].
    pub fn reset(&mut self) {
        for cache in self.caches.iter_mut() {
            cache.reset();
        }
    }

    /// Greedy-decode up to `max_new_tokens` ids from `input_ids`.
    ///
    /// Runs prefill (one forward at `T = input_ids.len()`, offset=0,
    /// causal mask) then yields sampled ids one at a time. Iterator
    /// terminates on EOS or after `max_new_tokens` items.
    ///
    /// # Errors
    /// - `input_ids.is_empty()`
    /// - `input_ids.len() + max_new_tokens > context_window`
    pub fn generate(
        &mut self,
        input_ids: &[u32],
        max_new_tokens: usize,
    ) -> Result<GenerateIter<'_>> {
        if input_ids.is_empty() {
            return Err(Error::InvalidArgument(
                "Generator::generate: input_ids must be non-empty".into(),
            ));
        }
        let total = input_ids.len()
            .checked_add(max_new_tokens)
            .ok_or_else(|| Error::InvalidArgument(
                "Generator::generate: input_ids.len() + max_new_tokens overflow".into(),
            ))?;
        if total > self.context_window {
            return Err(Error::InvalidArgument(format!(
                "Generator::generate: input_ids.len() ({}) + max_new_tokens ({}) > context_window ({})",
                input_ids.len(), max_new_tokens, self.context_window,
            )));
        }
        self.reset();
        let device = Device::shared()?;
        let prefill_len = input_ids.len();

        // Prefill: [1, T] ids, offset=0, explicit causal mask.
        let ids_tensor = Tensor::from_slice(&device, input_ids, [1, prefill_len])?;
        let offset = Tensor::from_slice(&device, &[0i32], [1])?;
        let mask = causal_mask(&device, prefill_len)?;
        let logits = self.model.forward(
            &ids_tensor,
            Some(&mask),
            &offset,
            &mut self.caches,
        )?;
        let first = argmax_last_position(&logits, self.model.config().vocab_size)?;

        Ok(GenerateIter {
            gen: self,
            device,
            prefill_len,
            next_input: Some(first),
            step: 0,
            max_new_tokens,
        })
    }
}

/// Iterator yielding `Result<u32>` sampled ids. Borrows `&mut Generator`.
pub struct GenerateIter<'a> {
    gen: &'a mut Generator,
    device: Device,
    prefill_len: usize,
    /// The id to yield next (and to feed into the next forward). `None`
    /// once the iterator has terminated.
    next_input: Option<u32>,
    step: usize,
    max_new_tokens: usize,
}

impl<'a> Iterator for GenerateIter<'a> {
    type Item = Result<u32>;

    fn next(&mut self) -> Option<Self::Item> {
        let id = self.next_input.take()?;
        // Hand the just-decided id to the caller, then prepare the next.
        self.step += 1;
        if id_is_terminal(id, &self.gen.eos_ids) || self.step >= self.max_new_tokens {
            // Don't run another forward; iterator ends after we hand this id back.
            return Some(Ok(id));
        }
        // Forward T=1 to produce the *next* id, store it for the next call.
        let next_id = match self.advance(id) {
            Ok(n) => n,
            Err(e) => return Some(Err(e)),
        };
        self.next_input = Some(next_id);
        Some(Ok(id))
    }
}

impl<'a> GenerateIter<'a> {
    fn advance(&mut self, prev_id: u32) -> Result<u32> {
        let ids = Tensor::from_slice(&self.device, &[prev_id], [1, 1])?;
        let pos = (self.prefill_len + self.step - 1) as i32;
        let offset = Tensor::from_slice(&self.device, &[pos], [1])?;
        let logits = self.gen.model.forward(
            &ids,
            None,
            &offset,
            &mut self.gen.caches,
        )?;
        argmax_last_position(&logits, self.gen.model.config().vocab_size)
    }
}

fn id_is_terminal(id: u32, eos_ids: &HashSet<u32>) -> bool {
    eos_ids.contains(&id)
}

/// Argmax over the last position of a `[1, T, vocab]` BF16 logits tensor.
fn argmax_last_position(logits: &Tensor, vocab: usize) -> Result<u32> {
    let elements: Vec<bf16> = logits.cpu_to_vec().ok_or_else(|| {
        Error::InvalidArgument(
            "argmax_last_position: cpu_to_vec returned None \
             (logits not materialised or dtype mismatch)".into(),
        )
    })?;
    if elements.len() < vocab {
        return Err(Error::InvalidArgument(format!(
            "argmax_last_position: logits has {} elements; need at least vocab={vocab}",
            elements.len(),
        )));
    }
    let start = elements.len() - vocab;
    let (best_i, _) = elements[start..]
        .iter()
        .enumerate()
        .map(|(i, x)| (i as u32, x.to_f32()))
        .fold((0u32, f32::NEG_INFINITY), |(bi, bv), (i, v)| {
            if v > bv { (i, v) } else { (bi, bv) }
        });
    Ok(best_i)
}
```

- [ ] **Step 5.4: Wire module into `lib.rs`**

In `crates/cider-press-models/src/lib.rs`, add (alphabetical with the others):

```rust
pub mod generator;
```

- [ ] **Step 5.5: Build the test target**

Run: `cargo test -p cider-press-models --test generator_smoke --no-run`
Expected: builds clean.

If any of the constructor calls (`Qwen2Model::from_weights`, `Qwen2Weights { … }`) mismatch the real APIs (struct shape may have drifted since this plan was written), fix the test against the real signatures and re-build until clean. Do NOT change the production API to fit the test — the test must compose existing constructors.

- [ ] **Step 5.6: Run the smoke test**

Run: `cargo test -p cider-press-models --test generator_smoke --release`
Expected: all 4 tests PASS. Release mode because `Generator::generate` runs real Metal dispatches; debug-mode bf16 byte-juggling adds latency but should still pass.

- [ ] **Step 5.7: Commit**

```bash
git add crates/cider-press-models/src/generator.rs \
        crates/cider-press-models/src/lib.rs \
        crates/cider-press-models/tests/generator_smoke.rs
git commit -m "$(cat <<'EOF'
feat(models): Generator + GenerateIter for greedy decode (branch 14)

Concrete-on-Qwen2Model driver: pre-allocates KvCaches sized to
context_window, runs prefill with an explicit causal mask, then
yields Result<u32> ids one at a time via decode-time T=1 forwards.
Terminates on EOS or max_new_tokens.

Argmax runs CPU-side over the last-position [vocab] slice (~150KB
at production sizes; the GPU round-trip would dominate).

Smoke test builds a tiny synthetic Qwen2Model (random 4-bit weights,
2 layers, vocab=128, hidden=64) and exercises iterator semantics on
real Metal dispatches: max_new_tokens cap, EOS early-exit,
context_window/eos_ids validation. No MLX dep; runs in CI.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: `dump_mlx_generate.py` harness

**Files:**
- Create: `scripts/dump_mlx_generate.py`

PEP 723 uv script. Same shape as `scripts/dump_tokenizer.py`. Takes `--checkpoint`, a JSON-encoded `--messages` argument, `--max-tokens`, and `--output`. Applies the HF chat template, runs `mlx_lm.generate` with greedy sampling (`temp=0.0`), writes the produced ids to safetensors.

- [ ] **Step 6.1: Author the script**

Create `scripts/dump_mlx_generate.py`:

```python
#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "mlx-lm>=0.20",
#   "transformers>=4.45",
#   "safetensors>=0.4",
#   "numpy>=1.26",
# ]
# ///
"""Dump greedy-generated token ids from mlx_lm for cider-press branch 14.

Loads `mlx_lm.load(<checkpoint>)`, applies the HF chat template to the
passed messages, runs `mlx_lm.generate` with greedy sampling
(temp=0.0), and writes the produced uint32 ids to a safetensors file
the Rust gated parity test compares against.

Usage::

    uv run scripts/dump_mlx_generate.py \\
        --checkpoint /path/to/qwen25-0.5b-instruct-4bit-mlx \\
        --messages '[{"role":"system","content":"..."}, {"role":"user","content":"..."}]' \\
        --max-tokens 16 \\
        --output /tmp/generator-parity.safetensors

Output safetensors keys:
  ids        : uint32 1-D — token ids mlx_lm sampled, excluding the
               prompt (decoded tokens only, in generation order).
  prompt_ids : uint32 1-D — token ids of the rendered chat prompt
               (post chat-template, what mlx_lm received as input).
  meta       : uint32 1-D [num_ids, num_prompt_ids] — disambiguates
               1-element placeholders if either list is empty.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from mlx_lm import load, generate
from safetensors.numpy import save_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--messages",
        required=True,
        help="JSON list of {role, content} dicts",
    )
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    messages = json.loads(args.messages)
    model, tokenizer = load(args.checkpoint)

    # apply_chat_template renders the conversation and (with
    # add_generation_prompt=True) appends the assistant-turn opener.
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    prompt_ids = tokenizer.encode(prompt)

    # mlx_lm.generate returns a string by default; the lower-level
    # generate_step yields (token_id, logprobs). We want the ids so we
    # use generate_step directly with greedy sampling (temp=0.0).
    from mlx_lm.utils import generate_step
    import mlx.core as mx

    sampled: list[int] = []
    prompt_tokens = mx.array(prompt_ids)
    for (token, _logprobs), step in zip(
        generate_step(prompt_tokens, model, temp=0.0),
        range(args.max_tokens),
    ):
        tok_int = int(token.item()) if hasattr(token, "item") else int(token)
        sampled.append(tok_int)
        if tok_int in (tokenizer.eos_token_ids if hasattr(tokenizer, "eos_token_ids")
                       else {tokenizer.eos_token_id}):
            break

    ids_arr = (
        np.array([0], dtype=np.uint32) if not sampled
        else np.array(sampled, dtype=np.uint32)
    )
    prompt_arr = (
        np.array([0], dtype=np.uint32) if not prompt_ids
        else np.array(prompt_ids, dtype=np.uint32)
    )
    meta = np.array([len(sampled), len(prompt_ids)], dtype=np.uint32)

    save_file(
        {"ids": ids_arr, "prompt_ids": prompt_arr, "meta": meta},
        args.output,
    )
    print(f"wrote {args.output}: {len(sampled)} sampled ids, "
          f"{len(prompt_ids)} prompt ids")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6.2: Make it executable**

Run: `chmod +x scripts/dump_mlx_generate.py`

- [ ] **Step 6.3: Commit**

```bash
git add scripts/dump_mlx_generate.py
git commit -m "$(cat <<'EOF'
feat(scripts): dump_mlx_generate.py for greedy-generation parity (branch 14)

PEP 723 uv script that applies the HF chat template, runs mlx_lm's
greedy generate_step (temp=0.0), and writes the produced uint32 token
ids to safetensors. Consumed by the gated generator_parity test in
the next task.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Gated `generator_parity` test

**Files:**
- Create: `crates/cider-press-models/tests/generator_parity.rs`

- [ ] **Step 7.1: Author the test**

Create `crates/cider-press-models/tests/generator_parity.rs`:

```rust
//! Gated end-to-end parity test for `Generator` vs `mlx_lm.generate`.
//!
//! Requires `CIDER_QWEN_CHECKPOINT_PATH` pointed at a Qwen2.5-0.5B-Instruct-4bit
//! MLX checkout. Renders a fixed messages list through both the Rust
//! and MLX-Python pipelines under greedy sampling (temp=0.0), and
//! asserts the produced token-id sequences match exactly.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::Command;

use cider_press_models::chat_template::{ChatTemplate, Message};
use cider_press_models::generator::Generator;
use cider_press_models::qwen2::{Qwen2Config, Qwen2Model, load_qwen2_weights};
use cider_press_models::tokenizer::Tokenizer;
use cider_press_test_utils::tempdir;
use safetensors::SafeTensors;

const MAX_NEW_TOKENS: usize = 16;

fn fixed_messages() -> Vec<Message> {
    vec![
        Message::system("You are a helpful assistant."),
        Message::user("What is the capital of France?"),
    ]
}

fn env_checkpoint_or_skip() -> Option<PathBuf> {
    match std::env::var("CIDER_QWEN_CHECKPOINT_PATH") {
        Ok(v) => Some(PathBuf::from(v)),
        Err(_) => {
            eprintln!("CIDER_QWEN_CHECKPOINT_PATH not set; skipping generator_parity");
            None
        }
    }
}

fn run_mlx_harness(checkpoint: &Path, messages: &[Message], max_tokens: usize)
    -> Vec<u32>
{
    let tmp = tempdir("models-generator-parity");
    let out = tmp.join("ids.safetensors");
    let messages_json = serde_json::to_string(messages).expect("serialize messages");
    let status = Command::new("uv")
        .arg("run")
        .arg("scripts/dump_mlx_generate.py")
        .arg("--checkpoint").arg(checkpoint)
        .arg("--messages").arg(&messages_json)
        .arg("--max-tokens").arg(max_tokens.to_string())
        .arg("--output").arg(&out)
        .status()
        .expect("invoke uv");
    assert!(status.success(), "dump_mlx_generate.py exited non-zero");
    let bytes = std::fs::read(&out).expect("read mlx output");
    let st = SafeTensors::deserialize(&bytes).expect("parse safetensors");
    let meta = st.tensor("meta").expect("meta").data();
    let meta: Vec<u32> = meta
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let num_ids = meta[0] as usize;
    let ids_data = st.tensor("ids").expect("ids").data();
    ids_data[..num_ids * 4]
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
#[ignore = "requires CIDER_QWEN_CHECKPOINT_PATH"]
fn generator_parity_greedy_qwen2() {
    let checkpoint = match env_checkpoint_or_skip() {
        Some(p) => p,
        None => return,
    };

    let messages = fixed_messages();

    // Rust side.
    let device = cider_press_runtime::Device::shared().expect("device");
    let config_bytes = std::fs::read(checkpoint.join("config.json"))
        .expect("read config.json");
    let config = Qwen2Config::from_json_bytes(&config_bytes).expect("parse config");
    let archive_bytes = std::fs::read(checkpoint.join("model.safetensors"))
        .expect("read model.safetensors");
    let archive = SafeTensors::deserialize(&archive_bytes).expect("deserialize");
    let weights = load_qwen2_weights(&archive, &config, &device)
        .expect("load weights");
    let model = Qwen2Model::from_weights(weights, config.clone()).expect("build model");
    let tokenizer = Tokenizer::from_file(&checkpoint.join("tokenizer.json"))
        .expect("load tokenizer");
    let chat_template = ChatTemplate::from_file(
        &checkpoint.join("tokenizer_config.json"),
        &tokenizer,
    ).expect("load chat template");

    let prompt = chat_template.render(&messages).expect("render");
    let ids = tokenizer.encode(&prompt).expect("encode");
    let eos: HashSet<u32> = chat_template.eos_ids().collect();
    let mut gen = Generator::new(model, ids.len() + MAX_NEW_TOKENS + 1, eos)
        .expect("Generator::new");
    let rust_ids: Vec<u32> = gen
        .generate(&ids, MAX_NEW_TOKENS)
        .expect("generate")
        .collect::<Result<_, _>>()
        .expect("collect");

    // MLX side.
    let mlx_ids = run_mlx_harness(&checkpoint, &messages, MAX_NEW_TOKENS);

    assert_eq!(rust_ids, mlx_ids,
        "rust_ids != mlx_ids\n  rust = {rust_ids:?}\n  mlx  = {mlx_ids:?}");
}
```

The exact loader entry point (`safetensors_io::load_qwen2_checkpoint` above is a placeholder) should match what's already used elsewhere — check `crates/cider-press-models/tests/attention_layer0.rs` or `tests/qwen2_logits.rs` for the canonical load pattern and copy it verbatim.

- [ ] **Step 7.2: Build the test target**

Run: `cargo test -p cider-press-models --test generator_parity --no-run`
Expected: builds clean. Fix any loader-API name mismatches against the real surface.

- [ ] **Step 7.3: If a real checkpoint is available, run it locally**

Run: `CIDER_QWEN_CHECKPOINT_PATH=/path/to/checkpoint cargo test -p cider-press-models --test generator_parity --release -- --ignored --nocapture`
Expected: PASS (exact id equality). If 1–2 bf16 ULPs in argmax ties cause divergence at some token N, capture the diff and decide whether to (a) match MLX's tie-break, (b) relax to first-N exact or (c) accept and update the spec — write up the finding before changing the assertion.

If no checkpoint is available in this session, leave the test as written and continue; gated tests are expected to skip in CI without the env var.

- [ ] **Step 7.4: Commit**

```bash
git add crates/cider-press-models/tests/generator_parity.rs
git commit -m "$(cat <<'EOF'
test(models): gated Generator parity vs mlx_lm greedy (branch 14)

Shells out to scripts/dump_mlx_generate.py, asserts the Rust
Generator's sampled ids equal mlx_lm's greedy output for a fixed
system+user prompt at max_new_tokens=16. Gated on
CIDER_QWEN_CHECKPOINT_PATH, same convention every prior branch uses.

The exact-id-equality bar is aspirational; if bf16 argmax ties
diverge across implementations, the failure surfaces the drift so we
can decide tolerance vs tie-break before relaxing the assertion.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: CLI binary

**Files:**
- Create: `crates/cider-press/src/bin/cider-press.rs`

- [ ] **Step 8.1: Author the binary**

Create `crates/cider-press/src/bin/cider-press.rs`:

```rust
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
#[command(name = "cider-press", version, about = "Greedy decode against a Qwen2 checkpoint")]
struct Args {
    /// Checkpoint directory holding config.json, tokenizer.json,
    /// tokenizer_config.json, and the safetensors shards.
    #[arg(long)]
    checkpoint: PathBuf,
    /// User message.
    #[arg(long)]
    prompt: String,
    /// Optional system message; omitted means no system turn.
    #[arg(long)]
    system: Option<String>,
    /// Maximum new tokens to generate.
    #[arg(long, default_value_t = 256)]
    max_tokens: usize,
    /// Pre-allocated KvCache window in tokens.
    #[arg(long, default_value_t = 4096)]
    context_window: usize,
    /// Skip ChatTemplate; encode the bare --prompt.
    #[arg(long)]
    no_chat_template: bool,
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
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
    let chat_template = ChatTemplate::from_file(
        &args.checkpoint.join("tokenizer_config.json"),
        &tokenizer,
    )?;
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
    let mut stream = tokenizer.decode_stream(true);  // skip special tokens

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
```

Notes for the implementer:
- `Tokenizer::decode_stream` lands in step 8.2; the binary file above
  imports it from the existing wrapper.

- [ ] **Step 8.2: Add `Tokenizer::decode_stream`**

The existing `Tokenizer` wrapper in `crates/cider-press-models/src/tokenizer.rs` does NOT yet expose `decode_stream`. Add it.

In `crates/cider-press-models/src/tokenizer.rs`, after the existing `decode` method, add:

```rust
    /// Construct an incremental decoder that buffers BPE tokens until
    /// they form a UTF-8-valid suffix. Push ids with `.step(id)`;
    /// returns `Some(String)` when a printable chunk is ready, `None`
    /// while buffering mid-codepoint.
    ///
    /// Thin pass-through to `tokenizers::Tokenizer::decode_stream`.
    /// `skip_special_tokens=true` suppresses tokens like `<|im_end|>`
    /// from human-visible output.
    pub fn decode_stream(&self, skip_special_tokens: bool)
        -> tokenizers::DecodeStream<'_>
    {
        self.0.decode_stream(skip_special_tokens)
    }
```

(Adjust `self.0` to match the actual field name on the `Tokenizer` struct.)

Add a unit test inside the existing `#[cfg(test)] mod tests` block:

```rust
    #[test]
    fn decode_stream_emits_text() {
        let dir = tempfile::tempdir().expect("tempdir");
        let tok = minimal_bpe_tokenizer(dir.path());  // existing helper
        // encode "hello" through the minimal tokenizer, push ids one at a
        // time, accumulate Some(String) chunks; assert the concatenation
        // round-trips. Exact id list depends on the minimal tokenizer
        // shape; use what tok.encode("hello") returns.
        let ids = tok.encode("hello").expect("encode");
        let mut stream = tok.decode_stream(false);
        let mut out = String::new();
        for id in &ids {
            if let Some(chunk) = stream.step(*id).expect("step") {
                out.push_str(&chunk);
            }
        }
        assert_eq!(out, "hello");
    }
```

If the existing `tokenizer.rs` test setup doesn't already expose a `minimal_bpe_tokenizer(&Path)` helper, factor one out of whatever inline construction the existing `from_file_round_trips` test (or similar) uses.

Run: `cargo test -p cider-press-models --lib tokenizer`
Expected: existing tests still pass, new `decode_stream_emits_text` also passes.

- [ ] **Step 8.3: Verify the binary builds**

Run: `cargo build -p cider-press --bin cider-press --release`
Expected: PASS.

- [ ] **Step 8.4: Smoke-run the CLI against a real checkpoint (if available)**

Run:
```bash
target/release/cider-press \
  --checkpoint $CIDER_QWEN_CHECKPOINT_PATH \
  --system "You are a helpful assistant." \
  --prompt "What is the capital of France?" \
  --max-tokens 32
```

Expected: streamed output, ending with a newline. If garbled output appears for multi-byte characters, `DecodeStream` integration is wrong — fix before claiming success.

If no checkpoint is available, skip this manual smoke and rely on `--help` smoke:
```bash
target/release/cider-press --help
```
Expected: clap-derived help output covering every flag.

- [ ] **Step 8.5: Commit**

```bash
git add crates/cider-press/src/bin/cider-press.rs \
        crates/cider-press-models/src/tokenizer.rs
git commit -m "$(cat <<'EOF'
feat(cli): cider-press greedy-decode binary (branch 14)

clap-derive CLI that loads a Qwen2 checkpoint, renders --prompt
through the checkpoint's chat template, runs Generator::generate
greedy, and prints streamed detokenized output via HF DecodeStream.
--no-chat-template skips template rendering for base-model A/B
testing; --max-tokens caps decode length; --context-window sizes
the KvCache slab.

Also adds Tokenizer::decode_stream as a thin pass-through to the HF
crate's incremental decoder; needed so the CLI can print UTF-8-safe
chunks as tokens land instead of waiting for the full sequence.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Roadmap update

**Files:**
- Modify: `docs/QWEN_PATH.md`

- [ ] **Step 9.1: Mark branch 14 done**

In `docs/QWEN_PATH.md`, find the row:

```
| 14 | `feat/cli + greedy decode` | CLI: load model, take prompt, prefill, decode loop, greedy argmax sampling, streamed detokenize | Generate a coherent completion |
```

Replace with:

```
| 14 | (done) `feat/cli + greedy decode` | New `cider-press-models::chat_template` (minijinja over `tokenizer_config.json`; renders `Message{system,user,assistant}` with `add_generation_prompt=true`; resolves `eos_token`/`eos_token_id` via `Tokenizer`) + `cider-press-models::generator` (`Generator { model, caches, eos_ids, context_window }`, concrete-on-`Qwen2Model`; `generate(ids, max_new_tokens) -> impl Iterator<Item=Result<u32>>` running prefill with `nn::causal_mask` then T=1 decode loop; CPU argmax over the last-position `[vocab]` BF16 slice). New `crates/cider-press/src/bin/cider-press.rs` clap-derive CLI composing the two with `Tokenizer::decode_stream` for streamed UTF-8-safe output. Workspace deps `clap=4.5.21` (derive) and `minijinja=2.5.0`. `nn::causal_mask` lifted out of `tests/qwen2_logits.rs`; `Error::ChatTemplate(minijinja::Error)` added with the `#[from]` pattern. | Unit smoke (`generator_smoke.rs`): synthetic 2-layer/vocab=128 Qwen2Model exercises iterator semantics on real Metal dispatches (max_new_tokens cap, EOS early-exit, context_window/eos_ids validation). Gated `generator_parity_greedy_qwen2` on `CIDER_QWEN_CHECKPOINT_PATH`: shells out to `scripts/dump_mlx_generate.py` and asserts exact id-sequence equality vs `mlx_lm.generate_step(temp=0.0)` for a fixed system+user prompt at `max_new_tokens=16`. |
```

- [ ] **Step 9.2: Update the "What to do next time" section**

Find the section beginning `## What to do next time we sit down`. Currently it points at branch 14. Replace with text that points at branch 15:

```markdown
## What to do next time we sit down

Branches 12c (model), 13 (tokenizer), and 14 (CLI + greedy decode)
are all done. The CLI runs end-to-end on Qwen2.5-0.5B-Instruct-4bit.

**Before merging branch 14:** run the gated `generator_parity_greedy_qwen2`
test locally with `CIDER_QWEN_CHECKPOINT_PATH` set. The exact-id bar
is informative; if bf16 argmax-tie drift causes divergence, capture
the diff before relaxing the assertion.

Branch 15 (`feat/perf-measurement`). Specifically:

1. Bench tokens/sec end-to-end via the CLI on a fixed prompt at
   fixed `--max-tokens`. Time the generate loop excluding load.
2. Compare against `mlx_lm.generate` on the same prompt + greedy.
3. Memory profile (peak RSS during decode).
4. Identify hot spots — likely candidates remain: per-dispatch
   commit/wait overhead (the spike's 1.5× gap), per-step bf16
   round-trip for argmax, per-step KvCache memcpy.
5. Document numbers in `docs/QWEN_PERF.md`.

Open carry-forward items:
- **`LanguageModel` trait extraction**: `Generator` is concrete-on-`Qwen2Model`;
  lift when a second architecture lands.
- **Sampling beyond greedy**: top-k / top-p / temperature.
- **Interactive / multi-turn REPL**: a future `cider-press chat` subcommand;
  the existing Generator just needs an optional `reset()` skip.
- **`assert_close` extraction**: still six sites in `cider-press-models/tests`.
- **Transpose=false direction of `Tensor::quantized_matmul`**: still
  `Unimplemented`; Qwen2.5 ties the LM head so this may never arrive.
- **Generic non-aligned `affine_qmm`**: deferred to its first unaligned
  consumer.
- **GPU argmax**: CPU scan over `[vocab]` is fine at sub-second
  decode speeds; lift to GPU when measurement justifies it.

Branches still ahead of perf measurement: 15 (perf). One branch
between us and `docs/QWEN_PERF.md`.
```

- [ ] **Step 9.3: Commit**

```bash
git add docs/QWEN_PATH.md
git commit -m "$(cat <<'EOF'
docs: mark branch 14 landed; point next step at branch 15 (perf)

CLI runs end-to-end on Qwen2.5-0.5B-Instruct-4bit. Generator +
ChatTemplate are the new model-layer surface; cider-press binary
composes them with Tokenizer::decode_stream for streamed output.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Final verification

- [ ] **Run the full workspace test suite (release)**

Run: `cargo test --workspace --release`
Expected: all non-gated tests PASS (gated parity tests skip without `CIDER_QWEN_CHECKPOINT_PATH`).

- [ ] **Lint clean**

Run: `cargo clippy --all-targets --all-features -- -D warnings`
Expected: no warnings.

- [ ] **Format clean**

Run: `cargo fmt --check`
Expected: no diffs.

- [ ] **Docs clean**

Run: `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps`
Expected: no warnings.

If anything fails in the final-verification block, fix in-place with a fixup commit. The pre-commit hook listed in `CLAUDE.md` runs fmt + clippy + tests + docs and will block a PR push otherwise.
