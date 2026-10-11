#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "mlx>=0.18",
# ]
# ///
"""MLX 4-bit `quantized_matmul` vs bf16 `matmul` throughput at the
Qwen3-Embedding-0.6B prefill shapes (docs/inference/models/qwen3-embedding.md §6).

For M in {150, 512, 2048} and the five distinct per-layer projection shapes
(q 1024→2048, k/v 1024→1024, o 2048→1024, gate/up 1024→3072, down 3072→1024;
k/v and gate/up counted once each), times `mx.quantized_matmul(x, wq, s, b,
transpose=True, group_size=64, bits=4)` and `x @ w.T` on random bf16 data:
--warmup evals, then --rounds rounds of --reps dispatches accumulated under one
`mx.eval` (amortised, as scripts/measure_qmm_mlx.py does). Reports the median
round's TFLOPS (2·M·K·N / t summed over the shapes) with min–max. On a GPU
where MLX's `is_nax_available()` holds, these are the NAX kernels.

Usage::

    uv run scripts/measure_qmm_vs_gemm_mlx.py
"""

from __future__ import annotations

import argparse
import statistics
import time

import mlx.core as mx

SHAPES = [("q_proj", 1024, 2048), ("kv_proj", 1024, 1024), ("o_proj", 2048, 1024), ("gate_up", 1024, 3072), ("down", 3072, 1024)]
GROUP_SIZE = 64
BITS = 4


def bench(fn, warmup: int, reps: int) -> float:
    for _ in range(warmup):
        mx.eval(fn())
    mx.synchronize()
    t0 = time.perf_counter()
    acc = None
    for _ in range(reps):
        y = fn()
        acc = y if acc is None else acc + y
    mx.eval(acc)
    mx.synchronize()
    return (time.perf_counter() - t0) / reps


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--reps", type=int, default=100)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--m", type=int, nargs="+", default=[150, 512, 2048])
    args = p.parse_args()
    print(f"mlx {mx.__version__} {mx.device_info()['device_name']} ({mx.device_info()['architecture']})")
    for m in args.m:
        tq_rounds: list[float] = []
        tg_rounds: list[float] = []
        flops = sum(2 * m * k * n for _, k, n in SHAPES)
        tensors = []
        for _, k, n in SHAPES:
            x = mx.random.normal((m, k)).astype(mx.bfloat16)
            w = mx.random.normal((n, k)).astype(mx.bfloat16)
            wq, s, b = mx.quantize(w, group_size=GROUP_SIZE, bits=BITS)
            tensors.append((x, w, wq, s, b))
        for _ in range(args.rounds):
            tq = tg = 0.0
            for x, w, wq, s, b in tensors:
                tq += bench(lambda: mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=GROUP_SIZE, bits=BITS), args.warmup, args.reps)
                tg += bench(lambda: x @ w.T, args.warmup, args.reps)
            tq_rounds.append(tq)
            tg_rounds.append(tg)
        tq_med, tg_med = statistics.median(tq_rounds), statistics.median(tg_rounds)
        print(
            f"M={m:5d}  qmm4 {flops / tq_med / 1e12:5.2f} TFLOPS ({flops / max(tq_rounds) / 1e12:.2f}–{flops / min(tq_rounds) / 1e12:.2f})"
            f"   bf16 gemm {flops / tg_med / 1e12:5.2f} TFLOPS ({flops / max(tg_rounds) / 1e12:.2f}–{flops / min(tg_rounds) / 1e12:.2f})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
