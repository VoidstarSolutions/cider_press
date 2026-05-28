# Branch 14 — `feat/cli + greedy decode` design

First end-to-end runnable inference: load Qwen2.5-0.5B-Instruct-4bit
from a checkpoint directory, render a chat-templated prompt, prefill,
greedy-decode token-by-token with streamed detokenization until EOS or
a `--max-tokens` cap.

## Goals

- A working `cider-press` CLI binary that takes a prompt and prints a
  generated completion.
- A reusable `Generator` abstraction in `cider-press-models` that
  drives prefill + greedy decode against `Qwen2Model` and a
  pre-allocated set of `KvCache`s.
- Chat-template rendering (`tokenizer_config.json` jinja → string)
  living alongside `Tokenizer` so the surface that touches HF
  checkpoint files is in one place.
- Gated token-parity acceptance test against `mlx_lm.generate` with
  greedy sampling.

## Non-goals

- Top-k / top-p / temperature sampling — greedy only.
- Multi-turn / interactive REPL — one-shot generation per CLI invocation.
- Trait abstraction over models. `Generator` is concrete on
  `Qwen2Model`; lift to a `LanguageModel` trait when (and only when)
  a second model lands.
- Perf measurement, command-buffer batching, allocator pooling —
  branch 15's territory.

## Architecture

Three layers, each with one clear job, dependencies flowing upward
(facade → models → runtime):

```
crates/cider-press/src/bin/cider-press.rs   ← clap CLI, ~80 lines
   │
   ├─→ cider-press-models::ChatTemplate     ← jinja render
   ├─→ cider-press-models::Tokenizer        ← encode + decode_stream
   └─→ cider-press-models::Generator        ← prefill + greedy loop
          │
          ├─→ Qwen2Model (existing)
          └─→ KvCache × num_layers (existing)
```

### `cider-press-models::chat_template` (new module)

```rust
pub enum Role { System, User, Assistant }
pub struct Message { pub role: Role, pub content: String }

pub struct ChatTemplate {
    env: minijinja::Environment<'static>,
    eos_token_id: u32,
    extra_eos_ids: Vec<u32>,
}

impl ChatTemplate {
    pub fn from_file(tokenizer_config_path: &Path, tokenizer: &Tokenizer)
        -> Result<Self>;
    pub fn render(&self, messages: &[Message]) -> Result<String>;
    pub fn eos_ids(&self) -> impl Iterator<Item = u32> + '_;
}
```

- `from_file` reads `tokenizer_config.json`, pulls out:
  - `chat_template` (jinja string) — compiled into the `minijinja::Environment`
  - `eos_token` — accepted either as a plain string (`"<|im_end|>"`)
    or as an `AddedToken`-shaped object (`{"content": "...", ...}`);
    the `.content` is resolved to a `u32` id via the passed `Tokenizer`
  - `eos_token_id` (int or `Vec<int>`, optional) — union'd with the
    resolved `eos_token`
- `render` evaluates the template with
  `{messages: [...], add_generation_prompt: true}`. The
  `add_generation_prompt = true` appends the trailing
  `<|im_start|>assistant\n` (or model-equivalent) so the next token
  the model produces is the start of the assistant turn.
- `eos_ids()` exposes the combined set so `Generator` can build its
  `HashSet<u32>` without `ChatTemplate` having to know about
  generation policy.

### `cider-press-models::generator` (new module)

```rust
pub struct Generator {
    model: Qwen2Model,
    caches: Vec<KvCache>,
    eos_ids: HashSet<u32>,
    context_window: usize,
    offset: u32,
}

impl Generator {
    pub fn new(model: Qwen2Model, context_window: usize, eos_ids: HashSet<u32>)
        -> Result<Self>;
    pub fn reset(&mut self);
    pub fn generate<'a>(&'a mut self, input_ids: &[u32], max_new_tokens: usize)
        -> Result<GenerateIter<'a>>;
}

pub struct GenerateIter<'a> { /* borrows &mut Generator */ }
impl<'a> Iterator for GenerateIter<'a> {
    type Item = Result<u32>;
}
```

- `new` pre-allocates one `KvCache` per transformer layer, each sized
  to `context_window` tokens. Rejects `context_window = 0` and
  `eos_ids.is_empty()` (the loop would never terminate gracefully).
- `generate` validates `input_ids.len() + max_new_tokens <= context_window`,
  calls `reset()`, runs prefill (one forward at `T = input_ids.len()`,
  `offset = 0`, explicit `[1, 1, T, T]` BF16 causal mask with
  `-1e4` upper triangle), argmaxes the *last* position's logits, and
  hands the resulting first id off to `GenerateIter` as the first
  yielded value.
- `GenerateIter::next` runs decode steps: build `[1, 1]` U32 input
  from the prior id, `offset = I32 [prefill_len + step]`, `mask = None`
  (T=1 attends every cached position; no triangle to mask), forward,
  argmax the single position, terminate if id ∈ `eos_ids` or
  `step >= max_new_tokens`. Returns `None` on termination, `Some(Err)`
  if a forward call fails.
- Argmax: `Tensor::cpu_to_vec` over the `[vocab]` slice, linear scan
  for the max-value index. Vocab is ~150 k bf16 = ~300 KB per step,
  trivially fast on CPU. No GPU argmax op needed.
- The Generator holds the model by value (moved in at construction)
  so the CLI can `let gen = Generator::new(model, ...)` and forget
  about the model after that.

### Causal mask helper

Lift the explicit `[1, 1, T, T]` BF16 causal mask construction out of
`crates/cider-press-models/tests/qwen2_logits.rs` into a new
`pub fn causal_mask(t: usize) -> Result<Tensor>` in
`cider-press-models::nn`. Both the test and `Generator::generate`
consume it. Same `-1e4` upper-triangle convention.

### `crates/cider-press/src/bin/cider-press.rs` (new binary)

```rust
#[derive(clap::Parser)]
struct Args {
    /// Checkpoint dir holding config.json, tokenizer.json,
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
    /// Skip ChatTemplate; encode the bare `--prompt`. For base-model
    /// behavior or A/B with the parity test.
    #[arg(long)]
    no_chat_template: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    // Load config, weights, model, tokenizer.
    // Build ChatTemplate (unless --no-chat-template).
    // Build Generator with eos_ids from ChatTemplate.
    // Render messages → prompt string → ids.
    // Open tokenizer.decode_stream(skip_special_tokens=true).
    // for id in generator.generate(&ids, max_tokens)? {
    //     if let Some(text) = stream.step(id?)? {
    //         print!("{text}"); stdout.flush()?;
    //     }
    // }
    // println!();
    Ok(())
}
```

~80 lines total.

## Data flow

**Startup**
1. clap parses args.
2. `Qwen2Config::from_file(checkpoint/config.json)`
3. `Qwen2Weights::load(checkpoint, &device)`
4. `Qwen2Model::from_weights(weights, config)`
5. `Tokenizer::from_file(checkpoint/tokenizer.json)`
6. `ChatTemplate::from_file(checkpoint/tokenizer_config.json, &tokenizer)`
7. `Generator::new(model, args.context_window, chat_template.eos_ids().collect())`

**Per prompt**
1. Build `Vec<Message>` from `--system` (if any) + `--prompt`.
2. `let text = if args.no_chat_template { args.prompt.clone() }
                else { chat_template.render(&messages)? };`
3. `let ids = tokenizer.encode(&text)?;` — `add_special_tokens=false`
   (template already injected the special-token markers).
4. `let mut stream = tokenizer.decode_stream(true);`
5. `for id_result in generator.generate(&ids, args.max_tokens)? { ... }`

## Acceptance

### Gated token-parity test

`crates/cider-press-models/tests/generator_parity.rs`, gated on
`CIDER_QWEN_CHECKPOINT_PATH` per the existing pattern.

```rust
#[test]
#[ignore = "requires CIDER_QWEN_CHECKPOINT_PATH"]
fn generator_parity_greedy_qwen2() -> Result<()> {
    let checkpoint = env_checkpoint_or_skip();

    // Fixed inputs.
    let messages = [
        Message::system("You are a helpful assistant."),
        Message::user("What is the capital of France?"),
    ];
    let max_new_tokens = 16;

    // Rust side.
    let (model, tokenizer, chat_template) = load_all(&checkpoint)?;
    let prompt = chat_template.render(&messages)?;
    let ids = tokenizer.encode(&prompt)?;
    let mut gen = Generator::new(
        model, ids.len() + max_new_tokens + 1, chat_template.eos_ids().collect()
    )?;
    let rust_ids = gen.generate(&ids, max_new_tokens)?
        .collect::<Result<Vec<u32>>>()?;

    // MLX side — shell out to the dump harness, read u32 ids back.
    let mlx_ids = run_mlx_generate_harness(
        &checkpoint, &messages, max_new_tokens
    )?;

    assert_eq!(rust_ids, mlx_ids);
    Ok(())
}
```

### MLX harness

`scripts/dump_mlx_generate.py` — PEP 723 uv script. Same shape as the
other dump scripts. Takes `--checkpoint` and a JSON-encoded messages
argument plus `--max-tokens`, applies the chat template via HF's
`tokenizer.apply_chat_template`, runs `mlx_lm.generate` with
`temp=0.0` (greedy), writes the produced u32 ids and the rendered
prompt string as metadata into a safetensors tensor in a temp file
whose path it prints to stdout. Test shells out via `uv run` and
reads from the printed path.

Tolerance: exact id equality. If 1–2 ULP bf16 drift in argmax ties
causes intermittent divergence (this is the worst-case outcome), the
test failure is informative enough to drive the follow-up: either
introduce a tie-break convention matching MLX's, or relax to first-N
exact-equality plus tolerance on the rest. We discover that on the
real run, not by speculation.

### Unit smoke (always runs)

`crates/cider-press-models/tests/generator_smoke.rs` — non-gated,
no MLX dep, no real checkpoint.

- Construct a tiny synthetic Qwen2Model (e.g. 2 layers, hidden=64,
  vocab=128, 1 KV head, head_dim=32) with random weights.
- Construct a Generator over it with a 64-token context window and a
  small `eos_ids` set.
- Assert: (a) prefill + N=4 decode steps run without panicking,
  (b) iterator yields at most `max_new_tokens` items, (c) when an EOS
  id is forced into the model output (e.g. by zeroing all logits
  except the eos index), the iterator terminates early.

## Workspace dependency adds

In `[workspace.dependencies]`:

```toml
clap = { version = "=4.5.21", features = ["derive"] }
minijinja = "=2.5.0"
```

Both pinned (`=`) per project convention. `clap` only used by the
facade-crate binary; `minijinja` only used by
`cider-press-models::chat_template`.

## Files touched

**New**
- `crates/cider-press-models/src/chat_template.rs`
- `crates/cider-press-models/src/generator.rs`
- `crates/cider-press-models/tests/generator_smoke.rs`
- `crates/cider-press-models/tests/generator_parity.rs`
- `crates/cider-press/src/bin/cider-press.rs`
- `scripts/dump_mlx_generate.py`

**Modified**
- `Cargo.toml` — workspace deps
- `crates/cider-press-models/Cargo.toml` — `minijinja` dep
- `crates/cider-press-models/src/lib.rs` — `pub mod chat_template; pub mod generator;`
- `crates/cider-press-models/src/nn.rs` — `pub fn causal_mask(t)` lifted from test
- `crates/cider-press-models/src/error.rs` — `Error::ChatTemplate(minijinja::Error)`
- `crates/cider-press-models/tests/qwen2_logits.rs` — consume `nn::causal_mask`
- `crates/cider-press/Cargo.toml` — `clap` dep, declare `[[bin]]`
- `docs/QWEN_PATH.md` — mark branch 14 done in the roadmap table

## Open follow-ups (deferred past this branch)

- **`LanguageModel` trait** — extract `Generator`'s `Qwen2Model` bound
  to a trait when the second model arrives. Currently a single
  implementation; the trait would have no callers.
- **Sampling beyond greedy** — top-k / top-p / temperature. Greedy
  suffices for the deterministic parity test; richer sampling waits
  on a real user-facing need.
- **Interactive / multi-turn REPL** — one-shot per invocation for now.
  A future `cider-press chat` subcommand would persist the KvCache
  across turns (its `reset()` becomes optional rather than mandatory).
- **GPU argmax** — CPU scan over `[vocab]` is fine at sub-second
  decode speeds. Lift to GPU when measurement justifies it.
- **`assert_close` extraction** — still six sites; not a branch-14
  concern.

## Why this shape

- **Generator as a concrete struct in models, not a trait** —
  one model, one binding. The trait line lifts cleanly later because
  the only `Qwen2Model`-specific call inside `Generator` is
  `model.forward(...)` whose signature would be the trait method.
- **ChatTemplate as a distinct type from Tokenizer** — Tokenizer is
  pure encode/decode over the BPE; ChatTemplate is jinja rendering
  over a messages list. Different responsibilities, different
  dependencies (`minijinja` vs `tokenizers`), different test
  surfaces. Bundling them would couple jinja to every Tokenizer user.
- **Generator yields `Result<u32>`, not `Result<String>`** — the
  user picked id-in / id-out for cleanest separation. The CLI
  composes `tokenizer.decode_stream` on top; a non-CLI consumer
  (e.g. a future bench harness) can skip detokenization entirely.
- **CPU argmax** — vocab fits in a 300 KB slice; the GPU round-trip
  would dominate the actual scan cost.
