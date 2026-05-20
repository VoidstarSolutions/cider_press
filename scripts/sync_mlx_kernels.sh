#!/usr/bin/env bash
# Copy the MLX Metal kernel files we vendor into kernels-mlx/ from a
# local MLX checkout. Usage:
#
#   MLX_DIR=/path/to/mlx ./scripts/sync_mlx_kernels.sh
#
# After running, update the "Upstream commit" SHA in
# kernels-mlx/VENDORED.md and run `cargo test --workspace --release` to
# confirm MLX-side changes still produce bit-exact qmv parity.

set -euo pipefail

if [ -z "${MLX_DIR:-}" ]; then
    echo "error: MLX_DIR is not set." >&2
    echo "       Usage: MLX_DIR=/path/to/mlx ./scripts/sync_mlx_kernels.sh" >&2
    exit 1
fi
DEST=$(cd "$(dirname "$0")/.." && pwd)/kernels-mlx

if [ ! -d "$MLX_DIR/mlx/backend/metal/kernels" ]; then
    echo "error: \$MLX_DIR=$MLX_DIR does not look like an MLX checkout." >&2
    exit 1
fi

FILES=(
    mlx/backend/metal/kernels/copy.metal
    mlx/backend/metal/kernels/copy.h
    mlx/backend/metal/kernels/binary.metal
    mlx/backend/metal/kernels/binary.h
    mlx/backend/metal/kernels/binary_ops.h
    mlx/backend/metal/kernels/quantized.metal
    mlx/backend/metal/kernels/quantized.h
    mlx/backend/metal/kernels/quantized_utils.h
    mlx/backend/metal/kernels/utils.h
    mlx/backend/metal/kernels/bf16.h
    mlx/backend/metal/kernels/bf16_math.h
    mlx/backend/metal/kernels/complex.h
    mlx/backend/metal/kernels/defines.h
    mlx/backend/metal/kernels/logging.h
    mlx/backend/metal/kernels/steel/defines.h
    mlx/backend/metal/kernels/steel/utils.h
    mlx/backend/metal/kernels/steel/gemm/gemm.h
    mlx/backend/metal/kernels/steel/gemm/loader.h
    mlx/backend/metal/kernels/steel/gemm/mma.h
    mlx/backend/metal/kernels/steel/gemm/params.h
    mlx/backend/metal/kernels/steel/gemm/transforms.h
    mlx/backend/metal/kernels/steel/utils/integral_constant.h
    mlx/backend/metal/kernels/steel/utils/type_traits.h
)

for f in "${FILES[@]}"; do
    src="$MLX_DIR/$f"
    dst="$DEST/$f"
    if [ ! -f "$src" ]; then
        echo "error: missing upstream file: $src" >&2
        exit 1
    fi
    mkdir -p "$(dirname "$dst")"
    cp "$src" "$dst"
done

cp "$MLX_DIR/LICENSE" "$DEST/COPYING"

SHA=$(cd "$MLX_DIR" && git rev-parse HEAD)
DATE=$(cd "$MLX_DIR" && git log -1 --format=%cs)
echo "synced ${#FILES[@]} files from MLX $SHA ($DATE)"
echo "next: update kernels-mlx/VENDORED.md upstream-commit SHA, then run cargo test --workspace --release"
