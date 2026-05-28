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

    # `stream_generate` yields GenerationResponse objects with a
    # `.token` field. Passing no `sampler` selects argmax — equivalent
    # to the old `temp=0.0` greedy path. mlx_lm filters the EOS token
    # internally; we keep the explicit set check as a safety net.
    from mlx_lm import stream_generate

    eos_set = (
        set(tokenizer.eos_token_ids)
        if hasattr(tokenizer, "eos_token_ids")
        else {tokenizer.eos_token_id}
    )
    sampled: list[int] = []
    for response in stream_generate(
        model, tokenizer, prompt_ids, max_tokens=args.max_tokens
    ):
        tok_int = int(response.token)
        sampled.append(tok_int)
        if tok_int in eos_set:
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
