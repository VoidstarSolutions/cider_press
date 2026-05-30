#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "mlx>=0.18",
#   "numpy>=1.26",
#   "safetensors>=0.4",
# ]
# ///
"""Generate the copy-kernel parity fixture.

This is a one-shot fixture generator, not project infrastructure. It uses
MLX as the reference oracle so the same harness extends to the qmv parity
fixture, where the reference is genuinely non-trivial. For copy itself the
expected output equals the input — we still round-trip through MLX so
the harness exercises the same code path it will use later.

Run with `uv run scripts/gen_copy_fixture.py` (no requirements.txt
needed; PEP 723 inline metadata above declares the deps).
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np
from safetensors.numpy import save_file

OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "copy.safetensors"
N = 1024


def main() -> None:
    rng = np.random.default_rng(seed=0)

    src_f32 = rng.standard_normal(N, dtype=np.float32) * 4.0 - 1.0
    src_f16 = (rng.standard_normal(N).astype(np.float32) * 2.0 + 0.5).astype(np.float16)

    # MLX as the reference oracle. For copy the expected output equals the
    # input, but we still pass it through `mx.array` -> `np.asarray` so
    # this script stays representative of the qmv parity flow.
    dst_f32 = np.asarray(mx.array(src_f32))
    dst_f16 = np.asarray(mx.array(src_f16))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "src_f32": src_f32,
            "dst_f32": dst_f32,
            "src_f16": src_f16,
            "dst_f16": dst_f16,
        },
        str(OUT),
    )
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
