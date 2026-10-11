# Vendored MLX Metal kernels

This directory contains a subset of the [MLX](https://github.com/ml-explore/mlx)
Metal kernel sources, vendored verbatim and used by `cider-press-kernels`
at build time via the flattener in `crates/cider-press-kernels/build.rs`.

MLX is MIT-licensed (© 2023 Apple Inc.); the upstream `LICENSE` is
preserved here as `COPYING` and applies to every file in this directory.

## Upstream commit

```text
06eb7483f03bca49a80e596f4089d02a507a2cf9  (2026-10-10)
```

The full upstream path is preserved under
`kernels-mlx/mlx/backend/metal/kernels/...` so MLX's own
`#include "mlx/backend/metal/kernels/..."` directives resolve naturally
when the flattener treats `kernels-mlx/` as its include root.

## Files

Entry points (compiled to JITed `MTLLibrary`s by cider-press-kernels):

- `mlx/backend/metal/kernels/copy.metal`
- `mlx/backend/metal/kernels/binary.metal`
- `mlx/backend/metal/kernels/unary.metal`
- `mlx/backend/metal/kernels/reduce.metal`
- `mlx/backend/metal/kernels/arg_reduce.metal`
- `mlx/backend/metal/kernels/rope.metal`
- `mlx/backend/metal/kernels/rms_norm.metal`
- `mlx/backend/metal/kernels/softmax.metal`
- `mlx/backend/metal/kernels/quantized.metal`
- `mlx/backend/metal/kernels/sdpa_vector_only.metal` (cider-press entry point; upstream uses `scaled_dot_product_attention.metal`)
- `mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.metal` (fused `sdpa_full` prefill path)

Transitive headers reached from those entry points (kept in sync by the
sync workflow below):

- `mlx/backend/metal/kernels/{copy,binary,binary_ops,unary,unary_ops,reduce,reduce_utils,quantized,quantized_utils,sdpa_vector,softmax,utils,bf16,bf16_math,complex,defines,logging,atomic,cexpf,erf,expm1f,fp8}.h`
- `mlx/backend/metal/kernels/reduction/{ops,reduce_all,reduce_col,reduce_init,reduce_row}.h`
- `mlx/backend/metal/kernels/steel/{defines,utils}.h`
- `mlx/backend/metal/kernels/steel/gemm/{gemm,loader,mma,params,transforms}.h`
- `mlx/backend/metal/kernels/steel/attn/{attn,loader,mma,params,transforms}.h`
- `mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h`
- `mlx/backend/metal/kernels/steel/utils/{integral_constant,type_traits}.h`

JIT-only kernel headers (no precompiled `.metal` upstream; the source
string is assembled at dispatch time per instantiation, mirroring
MLX's own `indexing.cpp` / `softmax.cpp` / etc.):

- `mlx/backend/metal/kernels/indexing/{indexing,gather}.h`

System includes (`<metal_stdlib>`, `<metal_simdgroup>`, etc.) are *not*
vendored — they're resolved by Apple's Metal compiler at JIT time.

## Compile options

The vendored kernels are parity-tested **bit-exact** against MLX fixtures.
That only holds if both sides hand the Metal compiler the same policy, so
`KernelLibrary::compile_options()` (`crates/cider-press-kernels/src/library.rs`)
mirrors MLX's `Device::build_library_` in `mlx/backend/metal/device.cpp`:

- `fastMathEnabled = false`
- `languageVersion` = `get_metal_version()`: 4.0 on macOS 26+, 3.2 on
  macOS 15+, else 3.1
- `enableLogging` only in debug builds (`#ifndef NDEBUG`)

`device.cpp` is **not** vendored (it is C++ dispatch code, not a kernel), so
the sync script prints MLX's current block at the end of every sync. If it
differs from the bullets above, update `compile_options()` in the same PR as
the kernel bump and re-run the full parity suite.

## Sync procedure

`scripts/sync_mlx_kernels.sh` copies the canonical file list from a
local MLX checkout into this directory. Set `MLX_DIR` to the checkout
path; the script errors out cleanly if it is unset. After syncing,
update the "Upstream commit" SHA above and run
`cargo test --workspace --release` — the qmv parity test is the canary
for MLX-side changes that break us.

```sh
git clone https://github.com/ml-explore/mlx.git ../mlx
MLX_DIR=../mlx ./scripts/sync_mlx_kernels.sh
```

The `.github/workflows/mlx-sync.yml` GitHub Actions workflow runs this
on a schedule against the upstream `main` branch and, when drift is
detected, opens a pull request from the fixed `mlx-sync/upstream`
branch — or updates the already-open one, so there is never more than
one sync PR.

## Adding a new entry point

1. Add the new `.metal` file to the list in `sync_mlx_kernels.sh`.
2. Run the sync. The flattener will discover transitive headers; copy
   each one mentioned in the build error into the script.
3. Add the new entry-point stem to `ENTRY_POINTS` in
   `crates/cider-press-kernels/build.rs`.
