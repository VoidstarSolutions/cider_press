#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "mlx-lm>=0.31",
# ]
# ///
"""Phase-0/1 golden fixtures for the Qwen3.5/3.6 Gated-DeltaNet port.

Runs the **prefill forward** (cache=None, T>1) of the reference mlx-lm model on a
fixed prompt and captures stage-by-stage activations from the *same* checkpoint
cider-press loads — so the Rust parity tests compare bit-for-bit (bf16) against
the incumbent. Granularity matches the roadmap phase gates:

  - embed              : embedding output            (loader / embed step)
  - layer0_in/out      : one **GDN** layer (linear)  (Phase 3 gate)
  - layer3_in/out      : one **gated-attention** layer (Phase 2 gate)
  - final_hidden       : post final-norm             (assembly)
  - logits             : lm_head output (tied on 4B, separate on 27B) (Phase 4 gate)

Saved with ``mx.save_safetensors`` (bf16 preserved; cider reads it directly).

Usage::

    uv run scripts/dump_qwen35_fixtures.py                       # 4B dev vehicle
    uv run scripts/dump_qwen35_fixtures.py \\
        --checkpoint mlx-community/Qwen3.6-27B-4bit \\
        --out /tmp/qwen3_6_27b_fixtures.safetensors
"""

from __future__ import annotations

import argparse

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask, create_ssm_mask


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="mlx-community/Qwen3.5-4B-4bit")
    ap.add_argument("--prompt", default="Reverse a string in Rust.")
    ap.add_argument("--out", default="/tmp/qwen3_5_4b_fixtures.safetensors")
    args = ap.parse_args()

    model, tok = load(args.checkpoint)
    lm = model.language_model.model  # Qwen3_5TextModel

    ids = mx.array(tok.encode(args.prompt), dtype=mx.uint32)[None]  # [1, S]
    caps: dict[str, mx.array] = {"input_ids": ids}

    # Manual layer loop so we can tap individual blocks; cache=None => prefill.
    h = lm.embed_tokens(ids)
    caps["embed"] = h
    fa_mask = create_attention_mask(h, None)
    ssm_mask = create_ssm_mask(h, None)

    x = h
    for i, layer in enumerate(lm.layers):
        mask = ssm_mask if layer.is_linear else fa_mask
        if i in (0, 3):
            caps[f"layer{i}_in"] = x
        x = layer(x, mask=mask, cache=None)
        if i in (0, 3):
            caps[f"layer{i}_out"] = x

    caps["final_hidden"] = lm.norm(x)
    caps["logits"] = model(ids)  # full forward incl. lm_head (tied on 4B)

    # Validate the layer-type taps before writing, so a surprising layout fails
    # loudly instead of saving a mislabeled fixture.
    assert lm.layers[0].is_linear, "layer 0 expected to be a GDN (linear) layer"
    assert not lm.layers[3].is_linear, "layer 3 expected to be a full-attention layer"

    mx.eval(list(caps.values()))
    mx.save_safetensors(args.out, caps)

    print(f"# wrote {args.out}")
    for k, v in caps.items():
        print(f"{k} : {tuple(v.shape)} : {v.dtype}")


if __name__ == "__main__":
    main()
