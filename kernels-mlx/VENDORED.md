# Vendored MLX Metal kernels

This directory contains a subset of the [MLX](https://github.com/ml-explore/mlx)
Metal kernel sources, vendored verbatim and used by `cider-press-kernels`
at build time via the flattener in `crates/cider-press-kernels/build.rs`.

MLX is MIT-licensed (© 2023 Apple Inc.); the upstream `LICENSE` is
preserved here as `COPYING` and applies to every file in this directory.

## Upstream commit

```text
7b7c12407f85b494e3e6d1cd3888650d224f362c  (2026-05-15)
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
- `mlx/backend/metal/kernels/rope.metal`
- `mlx/backend/metal/kernels/quantized.metal`

Transitive headers reached from those entry points (kept in sync by the
sync workflow below):

- `mlx/backend/metal/kernels/{copy,binary,binary_ops,unary,unary_ops,reduce,reduce_utils,quantized,quantized_utils,utils,bf16,bf16_math,complex,defines,logging,atomic,cexpf,erf,expm1f,fp8}.h`
- `mlx/backend/metal/kernels/reduction/{ops,reduce_all,reduce_col,reduce_init,reduce_row}.h`
- `mlx/backend/metal/kernels/steel/{defines,utils}.h`
- `mlx/backend/metal/kernels/steel/gemm/{gemm,loader,mma,params,transforms}.h`
- `mlx/backend/metal/kernels/steel/utils/{integral_constant,type_traits}.h`

JIT-only kernel headers (no precompiled `.metal` upstream; the source
string is assembled at dispatch time per instantiation, mirroring
MLX's own `indexing.cpp` / `softmax.cpp` / etc.):

- `mlx/backend/metal/kernels/indexing/{indexing,gather}.h`

System includes (`<metal_stdlib>`, `<metal_simdgroup>`, etc.) are *not*
vendored — they're resolved by Apple's Metal compiler at JIT time.

## Sync procedure

`scripts/sync_mlx_kernels.sh` copies the canonical file list from a
local MLX checkout into this directory. Set `MLX_DIR` to the checkout
path; the script errors out cleanly if it is unset. After syncing,
update the "Upstream commit" SHA above and run
`cargo test --workspace --release` — Stage 4's bit-exact parity test
is the canary for MLX-side changes that break us.

```sh
git clone https://github.com/ml-explore/mlx.git ~/src/mlx
MLX_DIR=~/src/mlx ./scripts/sync_mlx_kernels.sh
```

The `.github/workflows/mlx-sync.yml` GitHub Actions workflow runs this
on a schedule against the upstream `main` branch and opens a pull
request when drift is detected.

## Adding a new entry point

1. Add the new `.metal` file to the list in `sync_mlx_kernels.sh`.
2. Run the sync. The flattener will discover transitive headers; copy
   each one mentioned in the build error into the script.
3. Add the new entry-point stem to `ENTRY_POINTS` in
   `crates/cider-press-kernels/build.rs`.
