#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "transformers>=4.45",
#   "safetensors>=0.4",
#   "numpy>=1.26",
# ]
# ///
"""Dump tokenizer parity fixtures for cider-press branch 13.

Loads `transformers.AutoTokenizer.from_pretrained(<checkpoint>)` and
encodes a fixed corpus of strings. Writes the corpus and the per-string
token ids to a safetensors file the Rust gated parity test compares
against.

Usage::

    uv run scripts/dump_tokenizer.py \\
        --checkpoint /path/to/qwen25-0.5b-instruct-4bit-mlx \\
        --output /tmp/tokenizer-parity.safetensors

Output safetensors keys (one set per corpus entry i):
  corpus_<i>_ids   : uint32 1-D — token ids transformers produced. If
                                  the source string is empty, this is a
                                  single-element placeholder [0]; the
                                  Rust side uses the matching `meta`
                                  tensor to disambiguate.
  corpus_<i>_bytes : uint8  1-D — UTF-8 bytes of the source string.
                                  Single-byte [0] placeholder when the
                                  source is the empty string (safetensors
                                  >=0.4 cannot serialize 0-length tensors).
  corpus_<i>_meta  : uint32 2-D — [ids_len, bytes_len]. Disambiguates the
                                  placeholders above.
"""

from __future__ import annotations

import argparse

import numpy as np
from safetensors.numpy import save_file
from transformers import AutoTokenizer


CORPUS: tuple[str, ...] = (
    "Hello, world.",
    "日本語のテスト",
    " indented",
    "the quick brown fox",
    "3.14159 and 1e-6",
    "",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dump tokenizer parity fixtures for cider-press branch 13."
    )
    parser.add_argument(
        "--checkpoint", required=True, help="path to HF/MLX checkpoint directory"
    )
    parser.add_argument(
        "--output", required=True, help="path to write the safetensors fixture"
    )
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.checkpoint)
    out: dict[str, np.ndarray] = {}
    for i, text in enumerate(CORPUS):
        ids = tok.encode(text, add_special_tokens=False)
        ids_arr = (
            np.array([0], dtype=np.uint32)
            if len(ids) == 0
            else np.array(ids, dtype=np.uint32)
        )
        bytes_arr = (
            np.array([0], dtype=np.uint8)
            if len(text) == 0
            else np.frombuffer(text.encode("utf-8"), dtype=np.uint8).copy()
        )
        # uint32 [ids_len, bytes_len]; the Rust side reads ids_len to know
        # whether `ids_arr` is real or a 1-element placeholder.
        meta = np.array([len(ids), len(text.encode("utf-8"))], dtype=np.uint32)
        out[f"corpus_{i}_ids"] = ids_arr
        out[f"corpus_{i}_bytes"] = bytes_arr
        out[f"corpus_{i}_meta"] = meta

    save_file(out, args.output)
    print(
        f"wrote {args.output} ({sum(v.nbytes for v in out.values())} bytes): "
        f"{len(CORPUS)} corpus entries"
    )


if __name__ == "__main__":
    main()
