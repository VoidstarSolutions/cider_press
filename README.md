# cider-press

> Cold-pressed LLM inference for Apple Silicon.

An inference-only tensor runtime and model loader targeting Apple Silicon
exclusively. Scoped tightly so it can lean on the unified memory model and
ship hand-tuned Metal kernels without paying for a general-purpose device
abstraction.

**Status:** pre-alpha stub. Nothing works yet.

## Why another one?

Existing Rust ML frameworks (candle, burn) are general-purpose and eager.
That's the right design for portability but leaves perf on the table on
Apple Silicon, where:

- Unified memory means host↔device copies are free, but eager frameworks
  still pay for lifetime/refcount conservatism around the abstraction.
- Lazy graphs enable kernel fusion and command-buffer batching, worth
  roughly 10–25% on transformer inference.
- MLX has already done the hard work on quantized matmul and SDPA Metal
  kernels; those `.metal` sources are MIT-licensed and reusable.

`cider-press` is the experiment of pulling those ideas into idiomatic Rust
with a tight, inference-only scope.

## Non-goals

- Training / autograd
- CUDA, ROCm, Vulkan, WebGPU
- Windows, Linux
- Generic tensor library — this exists to run LLMs, not to be a numpy.

## Documentation

A step-by-step guide to how inference works in this codebase lives in
[`docs/inference/`](docs/inference/README.md) — one file per stage of the
forward pass, each also serving as the rustdoc for its module.

## Acknowledgments

The Metal kernels and many design decisions here derive from
[MLX](https://github.com/ml-explore/mlx) (MIT, © 2023 Apple Inc.). Any
files derived from MLX retain the MIT notice.

## License

MIT.
