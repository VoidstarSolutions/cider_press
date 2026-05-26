// Cider-press own kernel (not derived from MLX): naive batched bf16
// matrix multiply. One thread per output element, float accumulator.
//
// Scope: a correct, simple GEMM for branch-11 SDPA (Q @ K^T and
// attn @ V). Perf is intentionally unoptimized — branch 15 will
// migrate to MLX's tiled `steel_gemm` family if the measurement
// points there. Inputs must be contiguous in their last two
// dimensions; transposed views materialize via `Tensor::copy` first.
//
// Layout:
//   A: [Batch, M, K] row-major contiguous bf16
//   B: [Batch, K, N] row-major contiguous bf16
//   C: [Batch, M, N] row-major contiguous bf16
// where `Batch = product(leading dims)` flattened by the host. The
// host passes per-batch element strides (not byte strides) for A, B,
// and C in case a future caller wants to alias B across batches; for
// the unbroadcast case `a_batch_stride = M * K`, etc.

#include <metal_stdlib>

using namespace metal;

kernel void gemm_bfloat16(
    device const bfloat* A          [[buffer(0)]],
    device const bfloat* B          [[buffer(1)]],
    device bfloat* C                [[buffer(2)]],
    constant uint& M                [[buffer(3)]],
    constant uint& N                [[buffer(4)]],
    constant uint& K                [[buffer(5)]],
    constant uint& a_batch_stride   [[buffer(6)]],
    constant uint& b_batch_stride   [[buffer(7)]],
    constant uint& c_batch_stride   [[buffer(8)]],
    uint3 gid                       [[thread_position_in_grid]]
) {
    uint m = gid.x;
    uint n = gid.y;
    uint b = gid.z;
    if (m >= M || n >= N) {
        return;
    }

    device const bfloat* a_row = A + b * a_batch_stride + m * K;
    device const bfloat* b_col = B + b * b_batch_stride + n;

    float acc = 0.0f;
    for (uint k = 0; k < K; ++k) {
        acc += float(a_row[k]) * float(b_col[k * N]);
    }
    C[b * c_batch_stride + m * N + n] = bfloat(acc);
}
