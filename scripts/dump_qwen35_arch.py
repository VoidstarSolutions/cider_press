#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "mlx-lm>=0.31",
#   "huggingface-hub>=0.25",
#   "safetensors>=0.4",
# ]
# ///
"""Phase-0 architecture probe for the Qwen3.5/3.6 Gated-DeltaNet port.

Locks the two facts Phase 0 needs before any Rust is written:

  1. The authoritative ``config.json`` (every hyperparameter).
  2. The exact safetensors weight-key layout (key -> shape, dtype) — this
     *is* the loader spec for Phase 1 (the GDN params: the **split**
     ``in_proj_qkv``/``in_proj_z``/``in_proj_a``/``in_proj_b`` projections —
     the checkpoint splits the reference's fused ``in_proj_qkvz``/``in_proj_ba``
     — ``A_log``, ``dt_bias``, ``conv1d.weight``, the per-layer linear-vs-full
     split, norms, lm_head/tied embedding).

It also smoke-tests the **text-only** path so we resolve open-question §5 in
ROADMAP.md: does ``mlx-lm`` load this ``Qwen3_5ForConditionalGeneration``
checkpoint standalone, or do we need ``mlx-vlm``? The layout dump is
tool-agnostic (reads safetensors headers directly), so it succeeds even if the
generate smoke-test cannot.

Usage::

    uv run scripts/dump_qwen35_arch.py            # defaults to the 4B dev vehicle
    uv run scripts/dump_qwen35_arch.py --checkpoint mlx-community/Qwen3.6-27B-4bit
    uv run scripts/dump_qwen35_arch.py --prompt "Write a Rust fn that reverses a string."
"""

from __future__ import annotations

import argparse
import glob
import json
import os

from huggingface_hub import snapshot_download
from safetensors import safe_open


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="mlx-community/Qwen3.5-4B-4bit")
    ap.add_argument("--prompt", default="Reverse a string in Rust.")
    ap.add_argument("--max-tokens", type=int, default=8)
    args = ap.parse_args()

    print(f"# snapshot_download({args.checkpoint}) — first run pulls the checkpoint")
    local = snapshot_download(args.checkpoint)

    # 1. Authoritative config.
    cfg_path = os.path.join(local, "config.json")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    print("\n=== config.json ===")
    print(json.dumps(cfg, indent=2))

    # 2. Weight-key layout (the loader spec) — read headers only, no tensor loads.
    print("\n=== safetensors layout (key : shape : dtype) ===")
    n = 0
    for shard in sorted(glob.glob(os.path.join(local, "*.safetensors"))):
        with safe_open(shard, framework="numpy") as st:
            for k in st.keys():
                t = st.get_slice(k)
                print(f"{k} : {tuple(t.get_shape())} : {t.get_dtype()}")
                n += 1
    print(f"# {n} tensors total")

    # 3. Text-only smoke test — resolves the mlx-lm vs mlx-vlm question.
    print("\n=== text-only generate smoke-test (mlx-lm) ===")
    try:
        from mlx_lm import generate, load

        model, tokenizer = load(args.checkpoint)
        messages = [{"role": "user", "content": args.prompt}]
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        out = generate(
            model, tokenizer, prompt=prompt, max_tokens=args.max_tokens, verbose=True
        )
        print(f"\n# OK: mlx-lm loads + generates text-only.\n# output: {out!r}")
    except Exception as exc:  # noqa: BLE001 — surface the exact failure mode
        print(f"\n# mlx-lm text-only path FAILED: {type(exc).__name__}: {exc}")
        print("# -> fall back to mlx-vlm (add 'mlx-vlm' to deps) or a text-submodule extraction.")


if __name__ == "__main__":
    main()
