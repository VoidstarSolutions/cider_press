#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "mlx-lm>=0.20",
#   "mlx>=0.18",
# ]
# ///
"""Measure mlx_lm greedy throughput + peak memory on a fixed prompt.

The cider-press `bench` subcommand reports prefill/decode tok/s and peak
RSS on the same prompt + max-tokens; this script produces the comparable
MLX numbers so docs/QWEN_PERF.md can table the ratio. Manual one-shot
tool (Python is one-shot infra here), mirroring measure_stage5_mlx.py.

Uses `mlx_lm.stream_generate` (mlx_lm 0.20+ API) with no sampler
argument for greedy/argmax decoding — matching dump_mlx_generate.py.
The reference snippet used `generate_step(prompt, model, temp=0.0)` but
that API was removed in mlx_lm 0.20; `stream_generate` is the working
replacement.

Usage::

    uv run scripts/measure_qwen_mlx.py \\
        --checkpoint /path/to/qwen25-0.5b-instruct-4bit-mlx \\
        --prompt "Write a short paragraph about the city of Seattle." \\
        --max-tokens 128
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx
from mlx_lm import load, stream_generate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--prompt",
        default="Write a short paragraph about the city of Seattle.",
    )
    parser.add_argument("--system", default=None)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=8)
    args = parser.parse_args()

    model, tokenizer = load(args.checkpoint)

    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": args.prompt})

    # apply_chat_template with tokenize=False + encode matches
    # dump_mlx_generate.py's working pattern for mlx_lm 0.20+.
    prompt_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    prompt_ids = tokenizer.encode(prompt_text)
    prompt_len = len(prompt_ids)

    mx.reset_peak_memory()

    # Drive stream_generate (mlx_lm 0.20+ greedy: no sampler = argmax).
    # We time the first response as the prefill step (processes the whole
    # prompt) and subsequent responses as decode tokens.
    gen = stream_generate(model, tokenizer, prompt_ids, max_tokens=args.max_tokens)

    t_prefill_start = time.perf_counter()
    first_response = next(gen)
    mx.eval()  # flush any pending Metal work
    prefill_dur = time.perf_counter() - t_prefill_start

    produced = 1  # the prefill/first token above
    timed = 0
    decode_dur = 0.0
    decode_start: float | None = None

    for response in gen:
        mx.eval()  # synchronise each step for an honest per-token time
        produced += 1
        if produced == args.warmup:
            decode_start = time.perf_counter()
        elif produced > args.warmup:
            timed += 1
        if produced >= args.max_tokens:
            break

    if decode_start is not None:
        decode_dur = time.perf_counter() - decode_start

    peak_mib = mx.get_peak_memory() / (1024 * 1024)
    prefill_tok_s = prompt_len / prefill_dur if prefill_dur else 0.0
    decode_tok_s = timed / decode_dur if decode_dur else 0.0

    print("mlx_lm bench")
    print(f"  checkpoint     {args.checkpoint}")
    print(f"  prompt tokens  {prompt_len}")
    print(f"  generated      {produced} (warmup {args.warmup}, timed {timed})")
    print()
    print(f"  prefill        {prefill_dur * 1e3:8.3f} ms   {prefill_tok_s:8.1f} tok/s")
    print(f"  decode         {decode_dur * 1e3:8.3f} ms   {decode_tok_s:8.1f} tok/s")
    print(f"  peak memory    {peak_mib:8.1f} MiB")


if __name__ == "__main__":
    main()
