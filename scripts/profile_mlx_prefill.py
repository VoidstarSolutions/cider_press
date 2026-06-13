#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "mlx-lm>=0.20",
#   "mlx>=0.18",
# ]
# ///
"""Profile mlx_lm PREFILL (prompt eval): wall-clock timing and optional Metal GPU capture.

Mirrors mlx_lm.generate_step's prefill structure (for a prompt below one
prefill_step_size chunk): a cache-fill forward over the first T-1 tokens with
only the KV-cache state eval'd (the head is pruned via the lazily-unused
output), then the final token as a T=1 step whose logits are eval'd. Two evals
= two command-buffer commits, matching cider's `fill_cache` + T=1 step
(`Generator::prefill_sync`) for an apples-to-apples comparison. Measures warm
wall-clock ms and tok/s (mean over --iters runs after --warmup warm-up passes),
and optionally writes a Metal GPU trace via mx.metal.start_capture /
stop_capture for interactive analysis in Xcode Instruments.

The checkpoint is expected to be a 4-bit affine group_size=64 MLX-format
checkpoint loadable by mlx_lm.load directly.

Usage::

    uv run scripts/profile_mlx_prefill.py \\
        --checkpoint /path/to/qwen25-0.5b-instruct-4bit-mlx

    # with GPU capture (requires MTL_CAPTURE_ENABLED=1):
    MTL_CAPTURE_ENABLED=1 uv run scripts/profile_mlx_prefill.py \\
        --checkpoint /path/to/qwen25-0.5b-instruct-4bit-mlx \\
        --trace-out /tmp/mlx_prefill.gputrace
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--prompt",
        default="Write a short paragraph about the city of Seattle.",
    )
    parser.add_argument("--system", default=None)
    parser.add_argument(
        "--trace-out",
        default=None,
        metavar="PATH",
        help="write a Metal GPU capture (.gputrace) to this path (requires MTL_CAPTURE_ENABLED=1)",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=5)
    args = parser.parse_args()
    if args.iters < 1:
        parser.error("--iters must be >= 1 (the timed mean divides by it)")

    model, tokenizer = load(args.checkpoint)

    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": args.prompt})
    prompt_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True
    )
    x = mx.array(prompt_ids)
    T = x.shape[0]
    print(f"prompt tokens: {T}")

    def prefill() -> mx.array:
        # Mirror mlx_lm.generate_step: cache-fill the first T-1 tokens (eval
        # only the cache state, head pruned via the lazily-unused output), then
        # run the final token as a T=1 step and eval its logits. Two evals =
        # two command-buffer commits, matching cider's fill_cache + T=1 step.
        cache = make_prompt_cache(model)
        if T > 1:
            model(x[:-1][None], cache=cache)
            mx.eval([c.state for c in cache])
        logits = model(x[-1:][None], cache=cache)
        mx.eval(logits)
        return logits

    # warm-up
    for _ in range(args.warmup):
        prefill()

    # optional GPU capture (single warm pass inside the capture window).
    # Capture is best-effort: a start failure (e.g. MTL_CAPTURE_ENABLED unset)
    # is non-fatal so the timing run below still happens. If start succeeds,
    # stop_capture runs in a finally so the capture session is never left open.
    if args.trace_out is not None:
        try:
            mx.metal.start_capture(args.trace_out)
        except Exception as exc:
            print(f"GPU capture unavailable: {exc} (need MTL_CAPTURE_ENABLED=1)")
        else:
            try:
                prefill()
            finally:
                mx.metal.stop_capture()
            print(f"wrote GPU capture: {args.trace_out}")

    # timed loop
    elapsed: list[float] = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        prefill()
        elapsed.append(time.perf_counter() - t0)

    mean_ms = sum(elapsed) / len(elapsed) * 1e3
    tok_s = T / (mean_ms / 1e3)
    print(f"prefill: {mean_ms:.3f} ms  {tok_s:.1f} tok/s (mean of {args.iters} iters)")


if __name__ == "__main__":
    main()
