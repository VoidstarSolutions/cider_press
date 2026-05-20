#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "mlx>=0.18",
#   "numpy>=1.26",
# ]
# ///
"""Generic MLX activation-dump harness for per-op parity fixtures.

Each `cider-press` op branch landing after branch 3 (`feat/weight-loading`)
adds one case to this script: a small function that runs MLX's reference
implementation of the op against deterministically-seeded inputs and
writes both inputs and outputs to a safetensors file that the Rust-side
parity test compares against.

Usage::

    uv run scripts/dump_mlx_op.py <op> --output PATH [op-specific args]

Listing available ops::

    uv run scripts/dump_mlx_op.py --help

Each op's flags and the keys it writes are documented on its
`add_<op>_parser` function below. The convention every op follows:

- Inputs are seeded via `--seed` (default `0`).
- Inputs and outputs go into the same safetensors with descriptive keys
  (e.g. ``lhs``, ``rhs``, ``out`` for `add`).
- The output path is the caller's choice; the harness doesn't impose a
  fixtures/ subdirectory.

To add a new op:

1. Write `run_<op>(args) -> dict[str, mx.array]` that produces the
   input/output dict.
2. Write `add_<op>_parser(subparsers)` that wires its CLI flags.
3. Register both in the ``OPS`` table at the bottom of this file.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path

import mlx.core as mx


# ---------------------------------------------------------------------------
# add: element-wise add with broadcasting (consumed by branch 4
# `feat/elementwise-add`). Writes `lhs`, `rhs`, `out`.
# ---------------------------------------------------------------------------


def _parse_shape(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(",") if part)


_FLOAT_DTYPES = {
    "f32": mx.float32,
    "f16": mx.float16,
    "bf16": mx.bfloat16,
}


def _float_dtype(name: str) -> mx.Dtype:
    if name not in _FLOAT_DTYPES:
        raise SystemExit(
            f"unknown dtype {name!r}; expected one of {sorted(_FLOAT_DTYPES)}"
        )
    return _FLOAT_DTYPES[name]


def add_add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("add", help="Element-wise add with NumPy broadcasting")
    p.add_argument("--lhs-shape", required=True, help="comma-separated, e.g. 2,3,4")
    p.add_argument("--rhs-shape", required=True, help="comma-separated, e.g. 4")
    p.add_argument("--dtype", default="bf16", help="one of f32, f16, bf16")


def run_add(args: argparse.Namespace) -> dict[str, mx.array]:
    lhs_shape = _parse_shape(args.lhs_shape)
    rhs_shape = _parse_shape(args.rhs_shape)
    dtype = _float_dtype(args.dtype)
    lhs = (mx.random.uniform(shape=lhs_shape) - 0.5).astype(dtype)
    rhs = (mx.random.uniform(shape=rhs_shape) - 0.5).astype(dtype)
    out = mx.add(lhs, rhs)
    mx.eval(lhs, rhs, out)
    return {"lhs": lhs, "rhs": rhs, "out": out}


# ---------------------------------------------------------------------------
# square / rsqrt: element-wise unary ops (consumed by branch 5
# `feat/rmsnorm`). Writes `lhs`, `out`. Use uniform-in-[0.1, 1.1]
# inputs for rsqrt to avoid the rsqrt(0) trap; square accepts any
# real input.
# ---------------------------------------------------------------------------


def add_square_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("square", help="Element-wise x*x")
    p.add_argument("--lhs-shape", required=True, help="comma-separated, e.g. 1,8,896")
    p.add_argument("--dtype", default="bf16", help="one of f32, f16, bf16")


def run_square(args: argparse.Namespace) -> dict[str, mx.array]:
    shape = _parse_shape(args.lhs_shape)
    dtype = _float_dtype(args.dtype)
    lhs = (mx.random.uniform(shape=shape) - 0.5).astype(dtype)
    out = mx.square(lhs)
    mx.eval(lhs, out)
    return {"lhs": lhs, "out": out}


def add_rsqrt_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("rsqrt", help="Element-wise 1/sqrt(x); inputs are uniform in [0.1, 1.1]")
    p.add_argument("--lhs-shape", required=True, help="comma-separated, e.g. 1,8,896")
    p.add_argument("--dtype", default="bf16", help="one of f32, f16, bf16")


def run_rsqrt(args: argparse.Namespace) -> dict[str, mx.array]:
    shape = _parse_shape(args.lhs_shape)
    dtype = _float_dtype(args.dtype)
    # Shift the uniform distribution into [0.1, 1.1] so we never feed
    # rsqrt() a zero/near-zero. The runtime test does not need to
    # validate rsqrt's behaviour on degenerate inputs — that's a
    # property of the hardware op, not our dispatch.
    lhs = (mx.random.uniform(shape=shape) + 0.1).astype(dtype)
    out = mx.rsqrt(lhs)
    mx.eval(lhs, out)
    return {"lhs": lhs, "out": out}


# ---------------------------------------------------------------------------
# mul: element-wise multiply with broadcasting. Writes `lhs`, `rhs`, `out`.
# Shares the add-style argparser and seeding convention.
# ---------------------------------------------------------------------------


def add_mul_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("mul", help="Element-wise multiply with NumPy broadcasting")
    p.add_argument("--lhs-shape", required=True, help="comma-separated, e.g. 2,3,4")
    p.add_argument("--rhs-shape", required=True, help="comma-separated, e.g. 4")
    p.add_argument("--dtype", default="bf16", help="one of f32, f16, bf16")


def run_mul(args: argparse.Namespace) -> dict[str, mx.array]:
    lhs_shape = _parse_shape(args.lhs_shape)
    rhs_shape = _parse_shape(args.rhs_shape)
    dtype = _float_dtype(args.dtype)
    lhs = (mx.random.uniform(shape=lhs_shape) - 0.5).astype(dtype)
    rhs = (mx.random.uniform(shape=rhs_shape) - 0.5).astype(dtype)
    out = mx.multiply(lhs, rhs)
    mx.eval(lhs, rhs, out)
    return {"lhs": lhs, "rhs": rhs, "out": out}


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------


Runner = Callable[[argparse.Namespace], dict[str, mx.array]]
ParserBuilder = Callable[[argparse._SubParsersAction], None]


OPS: dict[str, tuple[ParserBuilder, Runner]] = {
    "add": (add_add_parser, run_add),
    "mul": (add_mul_parser, run_mul),
    "square": (add_square_parser, run_square),
    "rsqrt": (add_rsqrt_parser, run_rsqrt),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate per-op MLX parity fixtures for cider-press.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="path to write the safetensors fixture to",
    )
    parser.add_argument("--seed", type=int, default=0, help="MLX random seed (default 0)")
    subparsers = parser.add_subparsers(dest="op", required=True, metavar="op")
    for parser_builder, _runner in OPS.values():
        parser_builder(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.op not in OPS:  # argparse should already enforce, defense in depth.
        raise SystemExit(f"unknown op {args.op!r}")
    mx.random.seed(args.seed)
    _, runner = OPS[args.op]
    tensors = runner(args)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(args.output), tensors)

    summary = ", ".join(f"{k}={v.shape}/{v.dtype}" for k, v in tensors.items())
    print(f"wrote {args.output} ({args.output.stat().st_size} bytes): {summary}")


if __name__ == "__main__":
    main()
