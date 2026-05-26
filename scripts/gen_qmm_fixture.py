#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "mlx>=0.18",
#   "numpy>=1.26",
# ]
# ///
"""Generate parity fixtures for the qmm_t parity test.

Produces tests/fixtures/qmm_qwen2.safetensors containing:
  - w_q     : packed int4 weight matrix, uint32 [N, K * bits / 32]
  - scales  : per-group scales,           bf16   [N, K / group_size]
  - biases  : per-group biases,           bf16   [N, K / group_size]
  - x       : input matrix,               bf16   [M, K]
  - y_ref   : MLX reference output,       bf16   [M, N]

Dimensions match a Qwen2.5-0.5B q_proj prefill shape (T=8):
  M=8 (sequence length), K=896 (hidden dim), N=896 (output dim).
N % 32 == 0 so the aligned kernel variant applies.

Run: `uv run scripts/gen_qmm_fixture.py`
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

OUT = (
    Path(__file__).resolve().parents[1]
    / "crates"
    / "cider-press-kernels"
    / "tests"
    / "fixtures"
    / "qmm_qwen2.safetensors"
)

M = 8
K = 896
N = 896
GROUP_SIZE = 64
BITS = 4


def main() -> None:
    mx.random.seed(0)

    w = (mx.random.uniform(shape=(N, K)) - 0.5).astype(mx.bfloat16)
    x = (mx.random.uniform(shape=(M, K)) - 0.5).astype(mx.bfloat16)

    w_q, scales, biases = mx.quantize(w, group_size=GROUP_SIZE, bits=BITS)

    # mx.quantized_matmul with transpose=True computes y = x @ dequant(w_q).T,
    # i.e. y[m, n] = sum_k x[m, k] * dequant(w)[n, k]. That's what
    # affine_qmm_t computes: one threadgroup per (output_row, output_col) tile.
    y_ref = mx.quantized_matmul(
        x,
        w_q,
        scales,
        biases,
        transpose=True,
        group_size=GROUP_SIZE,
        bits=BITS,
    )

    mx.eval(w_q, scales, biases, x, y_ref)

    print(
        f"w_q={w_q.shape}/{w_q.dtype}  "
        f"scales={scales.shape}/{scales.dtype}  "
        f"biases={biases.shape}/{biases.dtype}  "
        f"x={x.shape}/{x.dtype}  "
        f"y_ref={y_ref.shape}/{y_ref.dtype}"
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(OUT),
        {
            "w_q": w_q,
            "scales": scales,
            "biases": biases,
            "x": x,
            "y_ref": y_ref,
        },
    )
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
