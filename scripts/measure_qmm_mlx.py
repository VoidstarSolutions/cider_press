#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "mlx>=0.18",
# ]
# ///
"""MLX qmm effective-bandwidth comparison over Qwen2.5-0.5B prefill shapes.

Times MLX's `mx.quantized_matmul` at the eight Qwen2.5-0.5B prefill shapes
(M=39), reporting per-dispatch latency and effective GB/s. Mirrors the shapes
in `crates/cider-press-kernels/tests/qmm_dispatch_bench.rs` so cider-press
and MLX numbers can be compared column-for-column.

Timing is amortized: REPS dispatches are queued before a single mx.eval so
each rep sees GPU-execution cost rather than the full host round-trip. Each
rep adds a cheap .sum() reduction to keep the result live; that cost is
constant across shapes and far smaller than qmm time on every shape.

Run: `uv run scripts/measure_qmm_mlx.py`
"""

from __future__ import annotations

import time

import mlx.core as mx

GROUP_SIZE = 64
BITS = 4
WARMUP = 50
REPS = 200  # dispatches amortized under one mx.eval
M = 39


def qmm_bytes(k: int, n: int) -> int:
    weights = n * k * BITS // 8
    scales_biases = 2 * (n * (k // GROUP_SIZE)) * 2  # bf16 scales + biases
    x = M * k * 2  # bf16 x matrix
    y = M * n * 2  # bf16 y output
    return weights + scales_biases + x + y


def bench(k: int, n: int, label: str) -> None:
    mx.random.seed(0)
    w = (mx.random.uniform(shape=(n, k)) - 0.5).astype(mx.bfloat16)
    x = (mx.random.uniform(shape=(M, k)) - 0.5).astype(mx.bfloat16)
    w_q, scales, biases = mx.quantize(w, group_size=GROUP_SIZE, bits=BITS)
    mx.eval(w_q, scales, biases, x)

    for _ in range(WARMUP):
        mx.eval(mx.quantized_matmul(
            x, w_q, scales=scales, biases=biases,
            transpose=True, group_size=GROUP_SIZE, bits=BITS,
        ))

    # Queue REPS dispatches before a single mx.eval to amortize host overhead.
    # .sum() keeps each qmm live (prevents dead-code elimination) and reduces
    # each output to a scalar so the accumulator stays memory-flat.
    # One un-timed amortized pass first to JIT the .astype().sum() graph for
    # this shape before the timed run.
    acc = mx.zeros((1,), dtype=mx.float32)
    for _ in range(REPS):
        acc = acc + mx.quantized_matmul(
            x, w_q, scales=scales, biases=biases,
            transpose=True, group_size=GROUP_SIZE, bits=BITS,
        ).astype(mx.float32).sum()
    mx.eval(acc)

    t0 = time.perf_counter()
    acc = mx.zeros((1,), dtype=mx.float32)
    for _ in range(REPS):
        acc = acc + mx.quantized_matmul(
            x, w_q, scales=scales, biases=biases,
            transpose=True, group_size=GROUP_SIZE, bits=BITS,
        ).astype(mx.float32).sum()
    mx.eval(acc)
    elapsed = time.perf_counter() - t0

    per_dispatch = elapsed / REPS
    per_us = per_dispatch * 1e6
    eff_gbps = qmm_bytes(k, n) / per_dispatch / 1e9
    print(f"  {label:<10}  M={M:>2} K={k:>4} N={n:>6}  {per_us:>9.2f} us   {eff_gbps:>6.0f} GB/s")


def main() -> None:
    print(f"qmm (MLX amortized): mx.quantized_matmul, M={M}, warmup={WARMUP}, reps={REPS} per eval")
    print("  (each rep includes a cheap .sum() to keep result live; cost is constant across shapes)")
    print("  shape                            per-dispatch    eff BW")
    for k, n, label in [
        (896, 896, "q_proj"),
        (896, 128, "k_proj"),
        (896, 128, "v_proj"),
        (896, 896, "o_proj"),
        (896, 4864, "gate"),
        (896, 4864, "up"),
        (4864, 896, "down"),
        (896, 151936, "lm_head"),
    ]:
        bench(k, n, label)


if __name__ == "__main__":
    main()
