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
import mlx.nn as nn


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
# row_sum: sum-reduce along the last axis (consumed by branch 5
# `feat/rmsnorm`). Writes `lhs`, `out`. Output shape drops the last
# axis (i.e. caller picks keep_dim semantics at the runtime layer).
# ---------------------------------------------------------------------------


def add_row_sum_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("row_sum", help="Sum-reduce along the last axis")
    p.add_argument("--lhs-shape", required=True, help="comma-separated, e.g. 1,8,896")
    p.add_argument("--dtype", default="bf16", help="one of f32, f16, bf16")


def run_row_sum(args: argparse.Namespace) -> dict[str, mx.array]:
    shape = _parse_shape(args.lhs_shape)
    dtype = _float_dtype(args.dtype)
    lhs = (mx.random.uniform(shape=shape) - 0.5).astype(dtype)
    out = mx.sum(lhs, axis=-1, keepdims=False)
    mx.eval(lhs, out)
    return {"lhs": lhs, "out": out}


# ---------------------------------------------------------------------------
# rms_norm: composed RMSNorm against MLX's reference (consumed by
# branch 5 `feat/rmsnorm`). Writes `x`, `gamma`, `out`. eps is bound
# at the CLI; the runtime test side uses the same value.
# ---------------------------------------------------------------------------


def add_rms_norm_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "rms_norm",
        help="MLX mx.fast.rms_norm(x, gamma, eps) — last-axis RMSNorm",
    )
    p.add_argument("--x-shape", required=True, help="comma-separated, e.g. 1,8,896")
    p.add_argument("--eps", type=float, default=1e-6)
    p.add_argument("--dtype", default="bf16", help="one of f32, f16, bf16")


def run_rms_norm(args: argparse.Namespace) -> dict[str, mx.array]:
    shape = _parse_shape(args.x_shape)
    if not shape:
        raise SystemExit("rms_norm: --x-shape must be non-empty")
    dtype = _float_dtype(args.dtype)
    hidden = shape[-1]
    x = (mx.random.uniform(shape=shape) - 0.5).astype(dtype)
    gamma = (mx.random.uniform(shape=(hidden,)) + 0.5).astype(dtype)
    out = mx.fast.rms_norm(x, gamma, args.eps)
    mx.eval(x, gamma, out)
    return {"x": x, "gamma": gamma, "out": out}


# ---------------------------------------------------------------------------
# sigmoid / erf: element-wise unary primitives (consumed by branch 6
# `feat/silu-and-gelu`). MLX itself composes `silu = x * sigmoid(x)`
# and exact `gelu = 0.5 * x * (1 + erf(x / sqrt(2)))` from these two,
# so the kernels-crate parity bar is bit-exact against the MLX
# primitive (not against `nn.silu` / `nn.gelu`). Writes `lhs`, `out`.
# Inputs span [-0.5, 0.5] (zero-centred) so both ops see the
# non-saturating regions of their respective curves.
# ---------------------------------------------------------------------------


def add_sigmoid_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("sigmoid", help="Element-wise 1/(1+exp(-x))")
    p.add_argument("--lhs-shape", required=True, help="comma-separated, e.g. 1,8,896")
    p.add_argument("--dtype", default="bf16", help="one of f32, f16, bf16")


def run_sigmoid(args: argparse.Namespace) -> dict[str, mx.array]:
    shape = _parse_shape(args.lhs_shape)
    dtype = _float_dtype(args.dtype)
    lhs = (mx.random.uniform(shape=shape) - 0.5).astype(dtype)
    out = mx.sigmoid(lhs)
    mx.eval(lhs, out)
    return {"lhs": lhs, "out": out}


def add_erf_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("erf", help="Element-wise Gauss error function")
    p.add_argument("--lhs-shape", required=True, help="comma-separated, e.g. 1,8,896")
    p.add_argument("--dtype", default="bf16", help="one of f32, f16, bf16")


def run_erf(args: argparse.Namespace) -> dict[str, mx.array]:
    shape = _parse_shape(args.lhs_shape)
    dtype = _float_dtype(args.dtype)
    lhs = (mx.random.uniform(shape=shape) - 0.5).astype(dtype)
    out = mx.erf(lhs)
    mx.eval(lhs, out)
    return {"lhs": lhs, "out": out}


# ---------------------------------------------------------------------------
# silu / gelu: composed activations against MLX's mlx.nn references
# (consumed by branch 6 `feat/silu-and-gelu`). silu = x * sigmoid(x);
# gelu is the exact erf-based variant (mlx.nn.gelu, not the tanh
# approx). Writes `lhs`, `out`. The composed-path tolerance on the
# Rust side is bf16-compose (each intermediate rounds to bf16); under
# f32 we expect bit-exact since both sides reduce to the same
# arithmetic in the same order.
# ---------------------------------------------------------------------------


def add_silu_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("silu", help="x * sigmoid(x) via mlx.nn.silu")
    p.add_argument("--lhs-shape", required=True, help="comma-separated, e.g. 1,8,896")
    p.add_argument("--dtype", default="bf16", help="one of f32, f16, bf16")


def run_silu(args: argparse.Namespace) -> dict[str, mx.array]:
    shape = _parse_shape(args.lhs_shape)
    dtype = _float_dtype(args.dtype)
    lhs = (mx.random.uniform(shape=shape) - 0.5).astype(dtype)
    out = nn.silu(lhs)
    mx.eval(lhs, out)
    return {"lhs": lhs, "out": out}


def add_gelu_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "gelu",
        help="0.5 * x * (1 + erf(x / sqrt(2))) via mlx.nn.gelu (exact)",
    )
    p.add_argument("--lhs-shape", required=True, help="comma-separated, e.g. 1,8,896")
    p.add_argument("--dtype", default="bf16", help="one of f32, f16, bf16")


def run_gelu(args: argparse.Namespace) -> dict[str, mx.array]:
    shape = _parse_shape(args.lhs_shape)
    dtype = _float_dtype(args.dtype)
    lhs = (mx.random.uniform(shape=shape) - 0.5).astype(dtype)
    out = nn.gelu(lhs)
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
    "row_sum": (add_row_sum_parser, run_row_sum),
    "rms_norm": (add_rms_norm_parser, run_rms_norm),
    "sigmoid": (add_sigmoid_parser, run_sigmoid),
    "erf": (add_erf_parser, run_erf),
    "silu": (add_silu_parser, run_silu),
    "gelu": (add_gelu_parser, run_gelu),
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
