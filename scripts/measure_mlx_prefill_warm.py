#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "mlx-lm>=0.31",
# ]
# ///
"""Warmed mlx_lm prefill-forward throughput on a checkpoint, for the
cider-press prefill band (docs/inference/models/qwen3-embedding.md §1).

Runs `model(ids)` (the full forward, including `lm_head` over ALL T
positions — more work than cider's last-row head, so this is a conservative
MLX number) after 3 warm-up forwards, then --rounds rounds of --reps evals
each, reporting the median round with min–max. No KV cache is passed, so
every forward is a from-scratch prefill. Tokenized with
add_special_tokens=False, matching cider's Tokenizer::encode.

Usage::

    uv run scripts/measure_mlx_prefill_warm.py --checkpoint <mlx dir> \\
        /tmp/embed-phase0/p150.txt /tmp/embed-phase0/p512.txt
"""

from __future__ import annotations

import argparse
import statistics
import time

import mlx.core as mx
from mlx_lm import load


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("prompts", nargs="+")
    args = p.parse_args()
    model, tok = load(args.checkpoint)
    print(f"mlx {mx.__version__} {mx.device_info()['device_name']} ({mx.device_info()['architecture']})")
    for path in args.prompts:
        with open(path, encoding="utf-8") as f:
            ids = tok.encode(f.read(), add_special_tokens=False)
        x = mx.array([ids])
        for _ in range(3):
            mx.eval(model(x))
        rounds: list[float] = []
        for _ in range(args.rounds):
            mx.synchronize()
            t0 = time.perf_counter()
            for _ in range(args.reps):
                mx.eval(model(x))
            mx.synchronize()
            rounds.append((time.perf_counter() - t0) / args.reps)
        med = statistics.median(rounds)
        print(
            f"T={len(ids)} forward {med * 1e3:.2f} ms ({min(rounds) * 1e3:.2f}–{max(rounds) * 1e3:.2f})"
            f"  {len(ids) / med:.0f} tok/s ({len(ids) / max(rounds):.0f}–{len(ids) / min(rounds):.0f})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
