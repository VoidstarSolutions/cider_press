// Copyright © 2024-25 Apple Inc.
//
// Single-pass sdpa_vector instantiations only (cider-press Plan A).
// MLX's upstream scaled_dot_product_attention.metal also instantiates
// the 2pass split-K variant; that lands later.

#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/sdpa_vector.h"

using namespace metal;

#define instantiate_sdpa_vector(type, qk_dim, value_dim)  \
  instantiate_kernel(                                      \
      "sdpa_vector_" #type "_" #qk_dim "_" #value_dim,     \
      sdpa_vector,                                         \
      type,                                                \
      qk_dim,                                              \
      value_dim)

#define instantiate_sdpa_vector_heads(type) \
  instantiate_sdpa_vector(type, 64, 64)     \
  instantiate_sdpa_vector(type, 96, 96)     \
  instantiate_sdpa_vector(type, 128, 128)   \
  instantiate_sdpa_vector(type, 256, 256)

instantiate_sdpa_vector_heads(float)
instantiate_sdpa_vector_heads(bfloat16_t)
instantiate_sdpa_vector_heads(float16_t)
