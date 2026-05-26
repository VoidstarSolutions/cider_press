#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "mlx>=0.18",
#   "mlx_lm>=0.21",
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
# gather: axis-0 embedding-style lookup (consumed by branch 7
# `feat/gather`). Writes `src`, `indices`, `out`. Indices are u32
# uniformly drawn over `[0, vocab)`; src is bf16 uniform in [-0.5, 0.5).
# The kernel is a pure data-mover so bit-exact equality holds for any
# dtype.
# ---------------------------------------------------------------------------


_GATHER_SRC_DTYPES = dict(_FLOAT_DTYPES, u32=mx.uint32)


def add_gather_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "gather",
        help="axis-0 gather (embedding lookup): out[t, h] = src[indices[t], h]",
    )
    p.add_argument("--vocab", type=int, required=True, help="source first-axis size, e.g. 128")
    p.add_argument("--hidden", type=int, required=True, help="source second-axis size, e.g. 64")
    p.add_argument("--n-indices", type=int, required=True, help="length of the rank-1 index tensor")
    p.add_argument(
        "--dtype",
        default="bf16",
        help=f"src/out dtype: one of {sorted(_GATHER_SRC_DTYPES)}",
    )


def run_gather(args: argparse.Namespace) -> dict[str, mx.array]:
    if args.dtype not in _GATHER_SRC_DTYPES:
        raise SystemExit(
            f"gather: unknown dtype {args.dtype!r}; expected one of {sorted(_GATHER_SRC_DTYPES)}"
        )
    dtype = _GATHER_SRC_DTYPES[args.dtype]
    if dtype == mx.uint32:
        # u32 is the packed-quantized-weight gather case (uint32 carries
        # packed int4 values). Generate uniformly-random bits to avoid
        # the kernel accidentally working on a degenerate `src = 0`.
        src = mx.random.randint(0, 2**31 - 1, shape=(args.vocab, args.hidden)).astype(dtype)
    else:
        src = (mx.random.uniform(shape=(args.vocab, args.hidden)) - 0.5).astype(dtype)
    indices = mx.random.randint(0, args.vocab, shape=(args.n_indices,)).astype(mx.uint32)
    out = mx.take(src, indices, axis=0)
    mx.eval(src, indices, out)
    return {"src": src, "indices": indices, "out": out}


# ---------------------------------------------------------------------------
# dequantize: affine-dequantize a packed (w_q, scales, biases) triple
# back to a dense bf16 tensor. Consumed by branch 7 (feat/gather) for
# the embedding-table integration. Writes `w_q`, `scales`, `biases`,
# `out`. Generates a random bf16 source, runs mx.quantize, then
# mx.dequantize — the round-trip lets the Rust side compare against
# the exact bytes MLX would produce given the same quantized triple.
# ---------------------------------------------------------------------------


def add_dequantize_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "dequantize",
        help="mx.dequantize on a quantized (w_q, scales, biases) triple",
    )
    p.add_argument("--rows", type=int, required=True, help="rows of the original (dense) weight, e.g. 128")
    p.add_argument("--cols", type=int, required=True, help="cols of the original weight (multiple of group_size)")
    p.add_argument("--group-size", type=int, default=64)
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--dtype", default="bf16", help="value dtype: one of f32, f16, bf16")


def run_dequantize(args: argparse.Namespace) -> dict[str, mx.array]:
    dtype = _float_dtype(args.dtype)
    dense = (mx.random.uniform(shape=(args.rows, args.cols)) - 0.5).astype(dtype)
    w_q, scales, biases = mx.quantize(dense, group_size=args.group_size, bits=args.bits)
    out = mx.dequantize(w_q, scales, biases, group_size=args.group_size, bits=args.bits)
    mx.eval(w_q, scales, biases, out)
    return {"w_q": w_q, "scales": scales, "biases": biases, "out": out}


# ---------------------------------------------------------------------------
# embed_tokens: gather rows of a quantized embedding table by token IDs
# and dequantize the gathered rows. The exact composition nn::embed_tokens
# wires at the models layer. Writes (w_q, scales, biases, indices, out)
# where out = mx.take(mx.dequantize(...), indices, axis=0). Bit-exact bar
# because each row's dequantize is independent of every other row.
# ---------------------------------------------------------------------------


def add_embed_tokens_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "embed_tokens",
        help="gather + dequantize the rows of a quantized embedding table",
    )
    p.add_argument("--vocab", type=int, required=True, help="rows of embedding table")
    p.add_argument("--hidden", type=int, required=True, help="cols of embedding table (multiple of group_size)")
    p.add_argument("--n-indices", type=int, required=True, help="length of the rank-1 index tensor")
    p.add_argument("--group-size", type=int, default=64)
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--dtype", default="bf16", help="value dtype: one of f32, f16, bf16")


def run_embed_tokens(args: argparse.Namespace) -> dict[str, mx.array]:
    dtype = _float_dtype(args.dtype)
    dense = (mx.random.uniform(shape=(args.vocab, args.hidden)) - 0.5).astype(dtype)
    w_q, scales, biases = mx.quantize(dense, group_size=args.group_size, bits=args.bits)
    indices = mx.random.randint(0, args.vocab, shape=(args.n_indices,)).astype(mx.uint32)
    dq = mx.dequantize(w_q, scales, biases, group_size=args.group_size, bits=args.bits)
    out = mx.take(dq, indices, axis=0)
    mx.eval(w_q, scales, biases, indices, out)
    return {"w_q": w_q, "scales": scales, "biases": biases, "indices": indices, "out": out}


# ---------------------------------------------------------------------------
# rope: rotary positional embedding (consumed by branch 9 `feat/rope`).
# Writes `lhs`, `out`. Input shape is `[B, H, T, D]`; the rotated leading
# `dims` slots are computed with Qwen2-style non-traditional layout
# (`(x[:d/2], x[d/2:])`) by default. `offset` is the starting position
# along the sequence axis (the MLX binding accepts a Python int and
# materializes the device-side scalar internally).
# ---------------------------------------------------------------------------


def add_rope_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "rope",
        help="rotary positional embedding via mx.fast.rope",
    )
    p.add_argument(
        "--lhs-shape",
        required=True,
        help="comma-separated [B, H, T, D], e.g. 1,14,4,64",
    )
    p.add_argument(
        "--dims",
        type=int,
        required=True,
        help="number of leading head-dim slots to rotate (== D for Qwen2)",
    )
    p.add_argument("--base", type=float, default=10000.0, help="theta base (default 10000)")
    p.add_argument("--scale", type=float, default=1.0, help="position scale (default 1.0)")
    p.add_argument("--offset", type=int, default=0, help="starting sequence position")
    p.add_argument(
        "--traditional",
        action="store_true",
        help="use traditional (GPT-NeoX interleaved) rotation",
    )
    p.add_argument("--dtype", default="bf16", help="one of f32, f16, bf16")


def run_rope(args: argparse.Namespace) -> dict[str, mx.array]:
    shape = _parse_shape(args.lhs_shape)
    if len(shape) != 4:
        raise SystemExit(
            f"rope: lhs-shape must have 4 dims [B, H, T, D], got {shape!r}"
        )
    head_dim = shape[-1]
    if args.dims < 1 or args.dims > head_dim:
        raise SystemExit(
            f"rope: --dims must be in [1, D] where D={head_dim}, got {args.dims}"
        )
    dtype = _float_dtype(args.dtype)
    lhs = (mx.random.uniform(shape=shape) - 0.5).astype(dtype)
    out = mx.fast.rope(
        lhs,
        args.dims,
        traditional=args.traditional,
        base=args.base,
        scale=args.scale,
        offset=args.offset,
    )
    mx.eval(lhs, out)
    return {"lhs": lhs, "out": out}


# ---------------------------------------------------------------------------
# softmax: numerically-stable last-axis softmax (consumed by branch 10
# `feat/softmax`). Writes `lhs`, `out`. Drives `mx.softmax(x, axis=-1,
# precise=...)`. Inputs are uniform in [-3, 3] so the max-subtraction
# produces a non-trivial exp range without overflowing bf16.
# ---------------------------------------------------------------------------


def add_sdpa_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "sdpa",
        help="GQA-broadcast scaled dot-product attention via mx.fast.scaled_dot_product_attention",
    )
    p.add_argument("--batch", type=int, required=True)
    p.add_argument("--h-q", type=int, required=True)
    p.add_argument("--h-kv", type=int, required=True)
    p.add_argument("--seq-q", type=int, required=True, help="Q sequence length (T)")
    p.add_argument(
        "--seq-kv",
        type=int,
        required=True,
        help="K/V sequence length (T_cache)",
    )
    p.add_argument("--head-dim", type=int, required=True)
    p.add_argument("--dtype", default="bf16", help="one of f32, f16, bf16")
    p.add_argument(
        "--causal",
        action="store_true",
        help="apply a causal mask aligned to the last `seq_q` rows of the [seq_q, seq_kv] grid",
    )


def run_sdpa(args: argparse.Namespace) -> dict[str, mx.array]:
    dtype = _float_dtype(args.dtype)
    b, h_q, h_kv = args.batch, args.h_q, args.h_kv
    t, t_kv, d = args.seq_q, args.seq_kv, args.head_dim
    if h_q % h_kv != 0:
        raise SystemExit(f"h_q ({h_q}) must be divisible by h_kv ({h_kv})")
    q = (mx.random.uniform(shape=(b, h_q, t, d)) * 2.0 - 1.0).astype(dtype)
    k = (mx.random.uniform(shape=(b, h_kv, t_kv, d)) * 2.0 - 1.0).astype(dtype)
    v = (mx.random.uniform(shape=(b, h_kv, t_kv, d)) * 2.0 - 1.0).astype(dtype)
    scale = 1.0 / (d**0.5)
    if args.causal:
        # Causal mask aligned to the last `t` rows of the [t, t_kv] grid
        # — same shape MLX's `mask="causal"` shortcut produces when
        # `t_kv >= t`.
        offsets = t_kv - t
        i = mx.arange(t).reshape(t, 1)
        j = mx.arange(t_kv).reshape(1, t_kv)
        mask_bool = (j <= (i + offsets))
        mask = mx.where(mask_bool, mx.array(0.0, dtype=dtype), mx.array(-mx.inf, dtype=dtype))
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
        result = {"q": q, "k": k, "v": v, "mask": mask, "out": out}
    else:
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
        result = {"q": q, "k": k, "v": v, "out": out}
    mx.eval(*result.values())
    return result


def add_matmul_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "matmul",
        help="batched dense matmul via mx.matmul",
    )
    p.add_argument(
        "--lhs-shape",
        required=True,
        help="comma-separated, e.g. 2,14,4,64 ([..., M, K])",
    )
    p.add_argument(
        "--rhs-shape",
        required=True,
        help="comma-separated, e.g. 2,14,64,128 ([..., K, N])",
    )
    p.add_argument("--dtype", default="bf16", help="one of f32, f16, bf16")


def run_matmul(args: argparse.Namespace) -> dict[str, mx.array]:
    lhs_shape = _parse_shape(args.lhs_shape)
    rhs_shape = _parse_shape(args.rhs_shape)
    dtype = _float_dtype(args.dtype)
    lhs = (mx.random.uniform(shape=lhs_shape) * 2.0 - 1.0).astype(dtype)
    rhs = (mx.random.uniform(shape=rhs_shape) * 2.0 - 1.0).astype(dtype)
    out = mx.matmul(lhs, rhs)
    mx.eval(lhs, rhs, out)
    return {"lhs": lhs, "rhs": rhs, "out": out}


# ---------------------------------------------------------------------------
# qmm: quantized matmul — covers the qmv path (M=1 decode) and the
# qmm_t path (M>1 prefill). Consumed by branch 11b `feat/quantized-matmul`.
# Writes `x`, `w_q`, `scales`, `biases`, `y`.
# ---------------------------------------------------------------------------


def add_qmm_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "qmm",
        help="Quantized matmul (qmv path for M=1, qmm_t for M>1)",
    )
    p.add_argument("--m", type=int, required=True, help="activation rows (1 for decode)")
    p.add_argument("--n", type=int, required=True, help="output cols (weight rows)")
    p.add_argument("--k", type=int, required=True, help="inner dim (weight cols / activation cols)")
    p.add_argument("--group-size", type=int, default=64)
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--dtype", default="bf16", help="one of f32, f16, bf16")


def run_qmm(args: argparse.Namespace) -> dict[str, mx.array]:
    dtype = _float_dtype(args.dtype)
    # Uniform [-0.5, 0.5) matches gen_qmm_fixture.py / gen_stage4_fixtures.py:
    # at K=896, normal-distributed inputs produce output magnitudes large
    # enough that bf16 LSB precision exceeds the M>1 tolerance bar.
    w = (mx.random.uniform(shape=(args.n, args.k)) - 0.5).astype(dtype)
    w_q, scales, biases = mx.quantize(w, group_size=args.group_size, bits=args.bits)
    if args.m == 1:
        x = (mx.random.uniform(shape=(args.k,)) - 0.5).astype(dtype)
    else:
        x = (mx.random.uniform(shape=(args.m, args.k)) - 0.5).astype(dtype)
    y = mx.quantized_matmul(
        x, w_q, scales, biases,
        transpose=True,
        group_size=args.group_size,
        bits=args.bits,
    )
    mx.eval(x, w_q, scales, biases, y)
    return {"x": x, "w_q": w_q, "scales": scales, "biases": biases, "y": y}


# ---------------------------------------------------------------------------
# attention_layer0: run layer-0 self-attention on a loaded Qwen2.5 MLX
# checkpoint. Writes `x`, `y`. Loads the checkpoint via `mlx_lm.utils.load`,
# constructs a random hidden state, and calls `model.model.layers[0].self_attn`
# directly. For offset=0 (prefill) the cache is None; for offset>0 (decode)
# the KVCache is pre-primed with `offset` zero rows via the `state` property
# setter.
#
# This is a checkpoint-loading case: it requires `mlx_lm` (>=0.21) and a
# local MLX-format checkpoint directory. It is registered in the OPS table
# and invoked via the `attention_layer0` sub-parser; the Rust integration
# test calls it via the `dump_mlx_op` helper with a `--checkpoint` arg.
# ---------------------------------------------------------------------------


def add_attention_layer0_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "attention_layer0",
        help="Run layer-0 self-attention on a loaded Qwen2.5 MLX checkpoint",
    )
    p.add_argument("--checkpoint", required=True, help="path to the MLX checkpoint directory")
    p.add_argument("--seq-len", type=int, required=True, help="sequence length (T)")
    p.add_argument("--offset", type=int, default=0, help="KV cache offset (0 for prefill)")


def run_attention_layer0(args: argparse.Namespace) -> dict[str, mx.array]:
    from mlx_lm.utils import load
    from mlx_lm.models.cache import KVCache

    model, _tokenizer = load(args.checkpoint)
    attn = model.model.layers[0].self_attn
    hidden_size: int = model.args.hidden_size
    n_kv_heads: int = attn.n_kv_heads
    # head_dim = hidden_size // n_heads; same formula as Attention.__init__
    head_dim: int = hidden_size // model.args.num_attention_heads

    x = (mx.random.uniform(shape=(1, args.seq_len, hidden_size)) - 0.5).astype(mx.bfloat16)

    if args.offset == 0:
        # Prefill: no cache — mlx_lm.Qwen2Attention.__call__ uses offset=0 RoPE
        # and omits the KV-cache update when cache=None.
        out = attn(x, mask=None, cache=None)
    else:
        # Decode: pre-prime the KVCache with `offset` zero rows so that
        # mlx_lm's attention uses `cache.offset` for RoPE positioning.
        # `KVCache.state` setter assigns keys/values and sets
        # `self.offset = keys.shape[2]`, giving us arbitrary-offset seeding
        # without patching mlx_lm internals.
        cache = KVCache()
        pad_k = mx.zeros((1, n_kv_heads, args.offset, head_dim), dtype=mx.bfloat16)
        pad_v = mx.zeros_like(pad_k)
        cache.state = (pad_k, pad_v)
        out = attn(x, mask=None, cache=cache)

    mx.eval(x, out)
    return {"x": x, "y": out}


def add_softmax_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "softmax",
        help="last-axis softmax via mx.softmax(x, axis=-1, precise=...)",
    )
    p.add_argument("--lhs-shape", required=True, help="comma-separated, e.g. 1,14,4,4")
    p.add_argument(
        "--precise",
        action="store_true",
        help="use float accumulator (mx.softmax precise=True)",
    )
    p.add_argument("--dtype", default="bf16", help="one of f32, f16, bf16")


def run_softmax(args: argparse.Namespace) -> dict[str, mx.array]:
    shape = _parse_shape(args.lhs_shape)
    dtype = _float_dtype(args.dtype)
    # Inputs in [-3, 3] cover a useful exp range while staying well
    # within bf16's representable region after max-subtraction.
    lhs = (mx.random.uniform(shape=shape) * 6.0 - 3.0).astype(dtype)
    out = mx.softmax(lhs, axis=-1, precise=args.precise)
    mx.eval(lhs, out)
    return {"lhs": lhs, "out": out}


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
    "gather": (add_gather_parser, run_gather),
    "dequantize": (add_dequantize_parser, run_dequantize),
    "embed_tokens": (add_embed_tokens_parser, run_embed_tokens),
    "rope": (add_rope_parser, run_rope),
    "softmax": (add_softmax_parser, run_softmax),
    "matmul": (add_matmul_parser, run_matmul),
    "sdpa": (add_sdpa_parser, run_sdpa),
    "qmm": (add_qmm_parser, run_qmm),
    "attention_layer0": (add_attention_layer0_parser, run_attention_layer0),
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
