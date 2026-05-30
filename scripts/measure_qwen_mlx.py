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
tool (Python is one-shot infra here), mirroring measure_qmv_mlx.py.

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
    parser.add_argument(
        "--warmup",
        type=int,
        default=8,
        help=(
            "informational only; mlx_lm's generation_tps already excludes"
            " prompt-eval time"
        ),
    )
    args = parser.parse_args()

    model, tokenizer = load(args.checkpoint)

    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": args.prompt})
    prompt_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    prompt_ids = tokenizer.encode(prompt_text)

    # Drive mlx_lm's own pipelined generation and read its built-in
    # measurements from the terminal GenerationResponse. This matches the
    # numbers `mlx_lm.generate --verbose` reports — injecting our own
    # mx.eval() per step would serialize mlx_lm's async pipeline and make
    # its decode throughput look slower than it really is.
    last = None
    for response in stream_generate(
        model, tokenizer, prompt_ids, max_tokens=args.max_tokens
    ):
        last = response
    if last is None:
        raise SystemExit("stream_generate produced no tokens")

    prompt_len = last.prompt_tokens
    prefill_tok_s = last.prompt_tps
    decode_tok_s = last.generation_tps
    generated = last.generation_tokens
    # last.peak_memory is in GB; report MiB to match `cider-press bench`.
    peak_mib = last.peak_memory * 1e9 / (1024 * 1024)

    print("mlx_lm bench")
    print(f"  checkpoint     {args.checkpoint}")
    print(f"  prompt tokens  {prompt_len}")
    print(f"  generated      {generated}")
    print()
    print(f"  prefill        {prefill_tok_s:8.1f} tok/s")
    print(f"  decode         {decode_tok_s:8.1f} tok/s")
    print(f"  peak memory    {peak_mib:8.1f} MiB")


if __name__ == "__main__":
    main()
