#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "mlx>=0.18",
# ]
# ///
"""Time MLX's own `mx.quantized_matmul` at the same shapes the Stage 5
Rust test measures, so the two numbers can be compared apples-to-apples.

The Rust harness times raw kernel dispatch (commit + waitUntilCompleted).
MLX's path includes its own graph/dispatch layer on top of the same
Metal kernel, so the difference between the two is, by definition, the
graph/dispatch overhead we're trying to characterize.

Run: `uv run scripts/measure_stage5_mlx.py`
"""

from __future__ import annotations

import time

import mlx.core as mx

GROUP_SIZE = 64
BITS = 4
WARMUP = 50
TIMED = 1000


def bench(dim: int) -> None:
    k = dim
    n = dim
    mx.random.seed(0)

    w = (mx.random.uniform(shape=(n, k)) - 0.5).astype(mx.bfloat16)
    x = (mx.random.uniform(shape=(k,)) - 0.5).astype(mx.bfloat16)
    w_q, scales, biases = mx.quantize(w, group_size=GROUP_SIZE, bits=BITS)
    mx.eval(w_q, scales, biases, x)

    def one() -> mx.array:
        return mx.quantized_matmul(
            x, w_q, scales, biases,
            transpose=True, group_size=GROUP_SIZE, bits=BITS,
        )

    for _ in range(WARMUP):
        y = one()
        mx.eval(y)

    t0 = time.perf_counter()
    for _ in range(TIMED):
        y = one()
        mx.eval(y)
    t1 = time.perf_counter()

    elapsed = t1 - t0
    per_us = (elapsed / TIMED) * 1e6
    rate = TIMED / elapsed
    print(f"  {k:>5}x{n:<5}   {elapsed * 1000:>8.2f} ms   {per_us:>9.2f} us   {rate:>9.0f} disp/s")


def main() -> None:
    print(f"stage 5 (MLX Python): mx.quantized_matmul, warmup={WARMUP}, timed={TIMED}")
    print("  shape (K=N)   total          per-dispatch    rate")
    bench(512)
    bench(4096)


if __name__ == "__main__":
    main()
