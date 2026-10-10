#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "mlx-lm>=0.20",
# ]
# ///
"""Dump a greedy (argmax) continuation from mlx_lm for the cider-press
Qwen3.5 token-stream parity gate.

Loads `mlx_lm.load(<checkpoint>)`, tokenizes a fixed RAW prompt (NO chat
template — `tokenizer.encode(prompt)`), greedy-generates N tokens (argmax
/ temp=0, deterministic), and writes the *continuation* token ids (the
generated ids, excluding the prompt) plus the prompt ids and prompt
string to a JSON file the Rust gated parity test reads.

Both sides use the RAW prompt + the same tokenizer so the comparison is
apples-to-apples (cider's chat template doesn't render Qwen3.5 yet).

Usage::

    uv run scripts/dump_qwen35_greedy.py
    uv run scripts/dump_qwen35_greedy.py --checkpoint <dir> -n 24 \\
        --out /tmp/qwen3_5_4b_greedy.json

Output JSON keys:
  prompt        : str        — the raw prompt string (no chat template).
  prompt_ids    : list[int]  — token ids of the raw prompt.
  continuation_ids : list[int] — argmax-sampled ids, excluding the prompt.
  n             : int        — len(continuation_ids).
  checkpoint    : str        — the resolved checkpoint id/path.
"""

from __future__ import annotations

import argparse
import json

from mlx_lm import load, stream_generate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--checkpoint",
        default="mlx-community/Qwen3.5-4B-4bit",
    )
    parser.add_argument(
        "--prompt",
        default="Reverse a string in Rust.\n",
        help="RAW prompt — encoded directly, NO chat template.",
    )
    parser.add_argument("-n", "--num-tokens", type=int, default=24)
    parser.add_argument("--out", default="/tmp/qwen3_5_4b_greedy.json")
    args = parser.parse_args()

    model, tokenizer = load(args.checkpoint)

    # RAW prompt — no apply_chat_template. add_special_tokens defaults
    # vary by tokenizer; match the cider Rust side (encode with no
    # special tokens) by passing the string through .encode and trusting
    # the HF tokenizer's default framing for this model (Qwen adds none).
    prompt_ids = tokenizer.encode(args.prompt)

    # No `sampler` selects argmax — deterministic greedy (temp=0). Run a
    # fixed N tokens; for a fixed-N run we do NOT stop on EOS so the
    # stream length is exactly N (the Rust side mirrors this with an
    # unreachable EOS).
    continuation: list[int] = []
    for response in stream_generate(
        model, tokenizer, prompt_ids, max_tokens=args.num_tokens
    ):
        continuation.append(int(response.token))
        if len(continuation) >= args.num_tokens:
            break

    out = {
        "prompt": args.prompt,
        "prompt_ids": [int(t) for t in prompt_ids],
        "continuation_ids": continuation,
        "n": len(continuation),
        "checkpoint": args.checkpoint,
    }
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)

    decoded = tokenizer.decode(continuation)
    print(f"checkpoint:       {args.checkpoint}")
    print(f"prompt:           {args.prompt!r}")
    print(f"prompt_ids ({len(prompt_ids)}): {list(prompt_ids)}")
    print(f"continuation_ids ({len(continuation)}): {continuation}")
    print(f"decoded continuation: {decoded!r}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
