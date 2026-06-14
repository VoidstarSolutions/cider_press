# Weights and quantization

Every projection in the forward pass is a matmul against a learned weight
matrix, and those matrices are most of the model. For
Qwen2.5-0.5B-Instruct-4bit the entire parameter set ships **4-bit
quantized** — the weights are stored at a quarter of their `bf16` width,
and dequantized on the fly inside the matmul kernels. This page is about
how those bytes are laid out, how `cider-press` loads them, and why
quantization is the single biggest reason decode runs as fast as it does.

## Concept

Decode is **memory-bandwidth-bound**, not compute-bound. Generating one
token reads *every* weight in the model once (to project a single
activation vector through each layer), but does only a vector-times-matrix
worth of arithmetic per weight. The GPU finishes the math long before the
next weight arrives from memory, so the wall-clock cost of a decode step is
dominated by *how many bytes of weight have to stream through the memory
system*. Halve the bytes per weight and you roughly halve the time.

That is the whole case for quantization: store each weight in fewer bits.
The scheme here is **affine** (a.k.a. zero-point) quantization. A real
weight `w` is approximated as

```text
w ≈ scale * q + bias
```

where `q` is a small integer (4 bits, so `0..15`) and `scale`/`bias` are
per-group floating-point constants. The matrix is sliced into **groups of
`group_size = 64` consecutive scalars along the inner (`K`) axis**, and each
group carries its own `(scale, bias)` pair. Within a group, `scale` sets the
spacing of the 16 representable levels and `bias` sets where they start, so a
group spanning a narrow range of values gets a fine grid and a group spanning
a wide range gets a coarse one — the per-group constants are what let 4 bits
cover the full dynamic range of a layer without one global scale crushing the
small weights to zero.

Physically that is three buffers, not one:

- **Packed lanes** — the 4-bit integers `q`, bit-packed into `u32` words.
  At 4 bits, one `u32` holds exactly 8 scalars (`32 / bits`). This is the
  bulk of the bytes and where the bandwidth saving lives.
- **Scales** — one `bf16` per group (`K / group_size` per row).
- **Biases** — one `bf16` per group, the affine zero-point. (These are the
  *quantization* biases — distinct from any additive Linear bias the layer
  may also carry; see the footgun below.)

What's lost is precision: every weight in a group snaps to one of 16 levels,
so the reconstruction `scale * q + bias` is close to the original `bf16`
weight but not equal to it. What's gained is bandwidth — for `group_size =
64`, a 4-bit group costs `64 * 4` packed bits plus two `bf16` side-channel
values, ≈ `4.5` bits per weight versus `16` for `bf16`, roughly a 3.5×
reduction in bytes moved per decode step. The quantization itself is done
offline (by the checkpoint publisher); inference only ever *reads* these
buffers and reconstructs `w` inside the kernel.

## In cider-press

Loading is host-side and one-time, structured as three layers.

[`load_qwen2_weights`](../../crates/cider-press-models/src/qwen2/weights.rs)
is the key-mapping loader: it walks the HuggingFace safetensors archive,
pulling each tensor out by its `model.layers.{i}.…` key into a typed
[`Qwen2Weights`](../../crates/cider-press-models/src/qwen2/weights.rs)
struct, validating every dtype and shape against the predictions from
[`Qwen2Config`](../../crates/cider-press-models/src/qwen2/config.rs) as it
goes. The per-key byte-reading is delegated to
[`safetensors_io.rs`](../../crates/cider-press-models/src/safetensors_io.rs):
`read_dense_tensor` for the plain `bf16` tensors (the RMSNorm gammas, the
Linear biases) and `read_quantized_weight` for the three-key quantized
triple — `{base}.weight` (`u32` packed lanes), `{base}.scales` (`bf16`),
`{base}.biases` (`bf16`) — which it assembles into one runtime object.

That object is
[`QuantizedWeight`](../../crates/cider-press-runtime/src/quantized.rs), a
newtype over `Tensor` that keeps the three component buffers together and
carries a
[`Quantization`](../../crates/cider-press-runtime/src/quantization.rs)
descriptor (`bits = 4`, `group_size = 64`). The newtype exists to encode
"this is a quantized weight" in the type system: the
[`quantized_matmul`](../../crates/cider-press-runtime/src/tensor.rs) op
takes a `&QuantizedWeight`, so a dense tensor can't be handed to it by
accident. When a weight is consumed as a *table* rather than a matmul
operand — the embedding lookup reuses the tied head's `[vocab, D]` weight —
the dequant kernel
[`affine_dequantize_bf16_gs64_b4`](../../crates/cider-press-kernels/src/kernels/dequantize.rs)
materializes the gathered rows back to dense `bf16`, wrapping the vendored
`affine_dequantize` instantiation in
[`quantized.metal`](../../kernels-mlx/mlx/backend/metal/kernels/quantized.metal)
(`out = w * scale + bias` per group, one thread per packed byte). In the
projection path the dequantize is *fused into* the matmul kernel (`qmv` /
`qmm`) and never materializes a dense weight — see
[quantized matmul](./quantized-matmul.md).

**The `.bias` vs `.biases` footgun.** The naming collides badly. Q, K, and
V each carry **two** unrelated things whose keys differ by one letter:

- `q_proj.bias` (**singular**) — a dense `bf16` *additive Linear bias*,
  shape `[H_q * D_h]`, added to the projection output *after* the matmul.
  Only Q/K/V have it; the output projection and the MLP projections do not.
  The loader names these `q_bias` / `k_bias` / `v_bias`.
- `q_proj.biases` (**plural**) — the per-group quantization zero-point on
  the 4-bit qmv ABI, shape `[N, K / group_size]`, consumed *inside*
  dequantization. **Every** quantized weight has it, including the output
  projection (which has `.biases` but no `.bias`).

They are semantically unrelated despite the near-identical name; mixing them
up silently corrupts the projection rather than erroring. The loader keeps
them in entirely separate fields — `q_bias` is a `Tensor`, the quantization
biases ride along inside the `QuantizedWeight` — precisely so the two can't
be confused at the call site.

**Model variations.** The quantization scheme is per-checkpoint metadata,
read from `config.json` (`bits`, `group_size`) — a model published at 8-bit,
or `group_size = 32`, drops in by changing the descriptor, no code change,
as long as `group_size` is a power of two and `group_size * bits` is a
multiple of 32 (a whole number of `u32` words spans a group; enforced by
[`Quantization::validate`](../../crates/cider-press-runtime/src/quantization.rs)).
*Which* tensors are quantized is also a model choice: Qwen2.5-0.5B quantizes
everything including the embedding/head table, while other checkpoints leave
the embedding dense. The additive-bias question is likewise per-family —
Qwen2.5 puts a Linear `.bias` only on Q/K/V; Llama-family models typically
carry none. And the head is **tied** here (`tie_word_embeddings = true`), so
there is no separate `lm_head` weight: the embedding table serves both the
input lookup and the output projection. Untied models ship a distinct
`lm_head` and the loader would need to read it (today
`tie_word_embeddings = false` is explicitly rejected). See
[models/qwen2.5.md](./models/qwen2.5.md).

## Performance

The loader's correctness bar is **loaded bytes == safetensors bytes**: every
tensor that passes `load_qwen2_weights` is byte-for-byte identical to what's
in the archive (the round-trip test in `weights.rs` memcmps every dense
tensor and every quantized triple against the source). Quantization isn't
something `cider-press` *does* — it consumes weights already quantized
offline — so there is no quantization error introduced at load; the only
arithmetic approximation is the offline one baked into the checkpoint, and
the dequant kernels reproduce MLX's reconstruction bit-exactly (the qmv/qmm
parity tests are the canary).

The memory footprint (process RSS via mach `task_info`; see the
[execution-model](./execution-model.md) page's memory section):

| mark        | cider-press RSS |
|-------------|----------------:|
| pre-load    | ~14 MiB         |
| post-load   | ~697 MiB        |
| post-decode | ~913 MiB        |
| peak        | ~913 MiB        |

A caveat on those numbers: cider-press samples **process RSS**, which
includes the ~278 MB safetensors archive read into a host `Vec` during load
plus the resident weight buffers, whereas `mlx_lm` reports its **Metal
buffer high-water mark** (~329 MiB) — they measure different things and
aren't directly comparable. The peak rose from ~900 to ~913 MiB with the
decode-path qmv padding (the q/k/v/o projections keep both an original and a
K-padded copy resident, ~28 MiB / ~3%); see
[quantized matmul](./quantized-matmul.md) for why the padded twins exist.

The one large *latency* cost attributable to the kernels is **Metal JIT
compilation**, and it is a load-time / first-token artifact, not a
steady-state cost. `cider-press` ships the kernels as flattened `.metal`
source; `newLibraryWithSource:` compiles them on first use. The first
compile of the large flattened source is **~43 s** of cold-start, but Metal
**caches the compiled library to disk cross-process**, so every subsequent
run of the binary is sub-100 ms. Cold first-token latency is therefore a
one-time JIT artifact, not something inference pays per token or per process
after the cache warms — it never appears in the warm prefill/decode numbers
the rest of this guide chases. See the
[execution model](./execution-model.md).

## Open levers

- **`.metallib` precompilation** — the ~43 s cold-start JIT (above) is the
  one user-visible weight-side latency, and it is removable. Shipping the
  kernels precompiled to a `.metallib` instead of flattened source would
  drop the first-token cold-start from a shipped binary entirely; it does
  not touch steady-state tok/s (the cache already absorbs it within a dev
  loop). This is a packaging concern, deferred — see the
  [execution model](./execution-model.md).
- **`quantized_matmul(transpose=false)`** — only the standard
  `y = x @ W^T` direction is implemented in the runtime; the
  `transpose=false` direction returns `Err` at construction
  ([`tensor.rs`](../../crates/cider-press-runtime/src/tensor.rs)). It was
  deferred to its first real consumer, and there may never be one for
  Qwen2.5: the tied head reuses the embedding table *transposed* (i.e.
  `transpose=true`), so the existing direction covers the whole forward
  pass. Wiring it is a known small task gated on a model that needs it.
- **Freeing the prefill weight twins after warm-up** — the ~28 MiB of
  duplicate K-padded decode weights (above) could be dropped once a serving
  context has finished prefill, recovering the ~3% peak-RSS bump. YAGNI for
  the current single-stream workload; noted, not open.
