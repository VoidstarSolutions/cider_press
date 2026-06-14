# Tokenization

The first and last thing inference does to text. A language model never
sees characters — it operates on a fixed vocabulary of integer **token
IDs**. Tokenization is the bijection at the boundary: prompt text → IDs on
the way in, sampled IDs → text on the way out.

## Concept

A tokenizer maps a string to a sequence of integers drawn from a fixed
vocabulary (≈151k entries for Qwen2.5). The mapping is **subword**: common
words become a single token, rare words split into several pieces, and any
string at all — including text the tokenizer has never seen — is always
representable, because the vocabulary bottoms out at individual bytes.

Modern LLMs use **byte-level BPE** (byte-pair encoding). It starts from the
256 raw bytes and greedily merges the most frequent adjacent pair, over and
over, until the vocabulary reaches its target size; the learned merge list
*is* the tokenizer. Encoding replays those merges on the input. Two
properties fall out of this and both matter for inference:

- **Subword granularity** is the compromise between a character vocabulary
  (tiny, but sequences are long and the model must relearn spelling) and a
  word vocabulary (short sequences, but no way to spell an unseen word).
  BPE gives short sequences *and* total coverage — frequent strings cost
  one token, novel strings degrade gracefully to a few.
- **Byte-level coverage** means there is no out-of-vocabulary failure mode
  and no UTF-8 hazard at the vocabulary level: a single emoji or CJK
  character is several bytes, hence often several tokens, and a token
  boundary can fall *inside* a multi-byte codepoint. That last point is why
  detokenization has to be incremental (below).

Tokenization is not the whole story for an instruct model. The raw vocab
encodes content, but the model was fine-tuned on conversations wrapped in
**special tokens** that mark turn boundaries and roles. A **chat template**
is the (model-specific) recipe that takes a list of role/content turns and
renders the exact string the model expects — for Qwen2.5 that means framing
each turn with `<|im_start|>{role}` … `<|im_end|>` and appending the
trailing `<|im_start|>assistant\n` opener so the model knows it is its turn
to speak. The template runs *before* tokenization; its output is the string
that actually gets encoded. Skip it, or render it wrong, and the model is
being asked a question in a format it was never trained on.

## In cider-press

Tokenization is host-side Rust, entirely upstream of the runtime and the
GPU. Three pieces live in the models crate:

- [`Tokenizer`](../../crates/cider-press-models/src/tokenizer.rs) is a thin
  wrapper over the HuggingFace
  [`tokenizers`](https://github.com/huggingface/tokenizers) crate. It loads
  the `tokenizer.json` that ships next to `config.json` in every
  mlx-community / HF checkpoint and exposes `encode` (text → `Vec<u32>`) and
  `decode` (`&[u32]` → `String`). `encode` passes
  `add_special_tokens=false` deliberately: turn framing is the chat
  template's job, not the raw encoder's, so the two layers don't both try to
  insert specials.
- [`chat_template.rs`](../../crates/cider-press-models/src/chat_template.rs)
  loads the `chat_template` field from `tokenizer_config.json` (falling back
  to a sibling `chat_template.jinja` for newer `transformers` layouts),
  compiles it once with `minijinja`, and renders a `&[Message]` list into
  the prompt string. It also resolves the EOS token(s) from the same config
  (`eos_token` and/or `eos_token_id`) into the `HashSet<u32>` the
  [`Generator`](../../crates/cider-press-models/src/generator.rs) uses as
  its stop set. So the input flow is: messages → `render` → `encode` →
  token IDs into prefill.
- Detokenization on the way out **streams**: the generate loop in
  [`generator.rs`](../../crates/cider-press-models/src/generator.rs) yields one
  sampled `u32` at a time and delegates detokenization to the caller, which
  feeds each ID into `Tokenizer::decode_stream` (in
  [`tokenizer.rs`](../../crates/cider-press-models/src/tokenizer.rs)). That is
  where the buffering lives: it holds BPE pieces until they form a
  UTF-8-valid suffix and only then emits a printable chunk. That buffering
  is exactly the byte-level hazard from the Concept section: a token can end
  mid-codepoint, and naively decoding ID-by-ID would emit replacement
  characters at those seams. `decode_stream` is passed
  `skip_special_tokens=true` so control tokens like `<|im_end|>` never reach
  the terminal, while the model still sees them as stop signals.

```text
"hi"  --encode-->  [chat template frames the turn]  -->  [token ids]  --> prefill
                                                              ...
            sampled id  --decode_stream.step(id)-->  Some("…") | None (buffering)
```

**Model variations.** The *code* here is model-agnostic — the same
`tokenizer`-crate wrapper, the same minijinja renderer, the same streaming
decoder serve any HF-format checkpoint. The *data* is per-model: each model
ships its own `tokenizer.json` (its own learned merges and vocabulary) and
its own chat template (its own special tokens and turn framing). Qwen2.5
uses byte-level BPE with the `<|im_start|>` / `<|im_end|>` ChatML framing;
a Llama or Mistral checkpoint drops in a different `tokenizer.json` and a
different template with no code change. See
[models/qwen2.5.md](./models/qwen2.5.md) for Qwen2.5's specifics.

## Performance

Tokenization is **off the GPU critical path**. Encoding the prompt and
rendering the chat template are pure host-side string work that happens once,
before the first forward pass; detokenization is a few microseconds of host
work per generated token, overlapped with the next token's GPU work in the
[execution model](./execution-model.md)'s decode pipeline. Neither shows up
in the prefill/decode wall-clock numbers that the rest of this guide chases.

The one cost worth naming is load-time:
[`Tokenizer::from_file`](../../crates/cider-press-models/src/tokenizer.rs)
parses the entire `tokenizer.json` — the full ≈151k-entry vocabulary and
merge list — up front, once per process. That is a one-time construction
cost amortized across every prompt the process serves; the type is built
once and shared by reference, and `encode`/`decode` are pure functions over
`&self` with no per-call setup.

## Open levers

None are material for the single-stream interactive workload `cider-press`
runs today. Tokenization is host-side and one-time-per-prompt, so it is not
where any measurable latency lives.

- **Batched prompt tokenization** — encoding many prompts at once would only
  matter to a **serving layer** fielding concurrent requests, where the
  host-side encode of one request could otherwise stall others. There is no
  such layer today (single-stream chat / bench), so this is noted, not open.
- **`tokenizer.json` parse on startup** — the one-time `from_file` cost
  contributes to cold-start latency but is dwarfed by the Metal JIT compile
  of the flattened kernel source on first run (tens of seconds, cached
  cross-process thereafter — see the [execution model](./execution-model.md)).
  Optimizing the parse before that is closed as not worth it.
