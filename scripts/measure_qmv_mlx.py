#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "mlx>=0.18",
# ]
# ///
"""MLX qmv effective-bandwidth comparison over Qwen2.5-0.5B production shapes.

Times MLX's `mx.quantized_matmul` at the five Qwen2.5-0.5B decode shapes
plus a forced-fast control, reporting per-dispatch latency and effective
GB/s. Mirrors the shapes in `crates/cider-press-kernels/tests/qmv_dispatch_bench.rs`
so cider-press and MLX numbers can be compared column-for-column.

Run: `uv run scripts/measure_qmv_mlx.py`
"""

from __future__ import annotations

import time

import mlx.core as mx

GROUP_SIZE = 64
BITS = 4
WARMUP = 50
TIMED = 1000


def qmv_bytes(k: int, n: int) -> int:
    weights = n * k * BITS // 8
    scales_biases = 2 * (n * (k // GROUP_SIZE)) * 2  # bf16 scales + biases
    return weights + scales_biases + k * 2 + n * 2


def bench(k: int, n: int, label: str) -> None:
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
        mx.eval(one())

    t0 = time.perf_counter()
    for _ in range(TIMED):
        mx.eval(one())
    elapsed = time.perf_counter() - t0

    per_us = (elapsed / TIMED) * 1e6
    eff_gbps = qmv_bytes(k, n) / (elapsed / TIMED) / 1e9
    print(f"  {k:>5}x{n:<6} {label:<20} {per_us:>9.2f} us   {eff_gbps:>6.0f} GB/s")


def main() -> None:
    print(f"qmv (MLX Python): mx.quantized_matmul, warmup={WARMUP}, timed={TIMED}")
    print("  shape        label                per-dispatch    eff BW")
    for k, n, label in [
        (896, 896, "q/o_proj"),
        (896, 128, "k/v_proj"),
        (896, 4864, "gate/up_proj"),
        (4864, 896, "down_proj"),
        (896, 151936, "lm_head"),
        (1024, 1024, "forced-fast control"),
    ]:
        bench(k, n, label)


if __name__ == "__main__":
    main()
