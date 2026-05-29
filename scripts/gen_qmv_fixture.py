#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "mlx>=0.18",
#   "numpy>=1.26",
# ]
# ///
"""Generate the qmv parity fixture.

Produces tests/fixtures/qmv.safetensors containing:
  - w_q     : packed int4 weight matrix, uint32 [N, K * bits / 32]
  - scales  : per-group scales,           bf16   [N, K / group_size]
  - biases  : per-group biases,           bf16   [N, K / group_size]
  - x       : input vector,               bf16   [K]
  - y_ref   : MLX reference output,       bf16   [N]

# Test dims
Dimensions are kept small (K = N = 512) so the fixture stays ~150 KB and
can be committed. The fast variant of the kernel still applies because
K % 512 == 0 and N % 8 == 0 — those are the conditions MLX uses to
select `affine_qmv_fast` over the general `affine_qmv`.

Run: `uv run scripts/gen_qmv_fixture.py`
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "qmv.safetensors"

K = 512
N = 512
GROUP_SIZE = 64
BITS = 4


def main() -> None:
    mx.random.seed(0)

    # Random bf16 weight in [-1, 1]-ish range. Match MLX's typical inference
    # ranges so the quantization error is representative.
    w = (mx.random.uniform(shape=(N, K)) - 0.5).astype(mx.bfloat16)
    x = (mx.random.uniform(shape=(K,)) - 0.5).astype(mx.bfloat16)

    w_q, scales, biases = mx.quantize(w, group_size=GROUP_SIZE, bits=BITS)

    # mx.quantized_matmul computes y = x @ dequant(w_q).T when transpose=True,
    # i.e. y[n] = sum_k x[k] * dequant(w)[n, k]. That's exactly what
    # affine_qmv computes: one threadgroup per output row.
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
