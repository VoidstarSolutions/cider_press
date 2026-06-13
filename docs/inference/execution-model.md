# Execution model

Everything else in this guide describes *what* the forward pass computes; this
page is about *how* it runs. `cider-press` does not execute eagerly. Building a
`Tensor` expression records nodes into a lazy graph and touches no GPU; one
synchronous `eval()` is the single dispatch boundary that turns the graph into
command buffers, runs them, and caches results. That one design choice is what
lets the runtime batch dispatches, elide redundant work, and pipeline the CPU
and GPU — and it is why prefill and decode, the *same* op graph at two sequence
lengths, have such different performance characteristics. This is the
runtime-systems layer the sibling pages forward-link to whenever they say "off
the critical path", "hidden by the pipeline", or "the eval boundary".

## Concept

A `Tensor` is a cheap handle (an `Arc` over immutable metadata). Composing ops
on it — `a + b`, a projection, an attention — does no arithmetic; it records an
**op node** whose inputs are the tensors it reads. The transitive closure of
those inputs *is* the graph: there is no separate `Graph` type, a tensor
self-describes its own lineage (the same model as MLX's `array`). Work happens
only when something asks for a result — `eval()` — which is the one place the
GPU is touched.

Why a graph at all, rather than running each op as it's written:

- **Batch dispatches.** A whole transformer forward is hundreds of ops. Encoding
  them into shared command buffers and committing in chunks — rather than one
  submit per op — is what amortizes per-dispatch overhead and lets the GPU stay
  fed.
- **Elide redundant work.** Seeing the whole graph before dispatching lets the
  runtime drop work that isn't needed: degenerate copies that are already
  contiguous become zero-copy views, redundant memory barriers are elided when
  no hazard exists, and an already-materialized subgraph is simply skipped.
- **Detach after eval.** Once a node's result is cached, its inputs can be
  dropped — the lineage that produced it is no longer needed to read it. Without
  this, each evaluated tensor would pin its entire upstream graph (see
  Performance).

The eval boundary is **synchronous by design** — `eval()` waits for the GPU
before returning, so a reader only ever sees "not computed" or "ready", never an
in-flight state. This is deliberate and permanent (see
[In cider-press](#in-cider-press)): MLX's own `mx.eval` is synchronous too. The
*internal* pipelining that overlaps CPU and GPU work lives below the boundary,
not in the public API.

**Prefill vs. decode** are the same graph at two sizes, and the split is the
single most important performance fact about inference:

- **Prefill** processes the whole prompt at once — one `eval()` over a `[1, T, D]`
  graph for a `T`-token prompt. Each weight is reused across all `T` rows, so the
  [quantized matmul](./quantized-matmul.md) is the matrix-matrix (`qmm`) shape and
  the work is compute-bound; the GPU has plenty to chew on per dispatch.
- **Decode** generates one token per step — one `eval()` over a `[1, 1, D]` graph,
  every step. Each weight is read exactly once for a single activation row
  (`qmv`), so there is almost no arithmetic reuse and the step is
  **memory-bandwidth-bound** on streaming the weights. The whole step is a deep
  *chain* of small dependent dispatches (~536 per token).

That difference drives everything: prefill runs at <50% GPU occupancy (one big
synchronous eval, the levers below are about *concurrency*), while decode is a
bandwidth-bound dependent chain where the lever is overlapping its CPU encode
with the prior token's GPU execution.

**The CPU-encode / GPU-execute split.** Every `eval()` has two costs: the CPU
walks the graph and *encodes* dispatches into command buffers, and the GPU then
*executes* them. In decode these can overlap — token *N+1*'s graph-build and
encode can run on the CPU while token *N*'s command buffers run on the GPU,
because greedy generation already knows it will need token *N+1*. Pipelining the
two is the difference between paying `encode + execute` per token and paying
`max(encode, execute)`.

## In cider-press

[`eval`](../../crates/cider-press-runtime/src/eval.rs) is the synchronous
boundary. It runs four passes over the graph rooted at the tensor being
evaluated:

1. **Schedule** (`build_order`) — collect the unevaluated op nodes reachable
   from the root in **Kahn level (wave) order**: a valid topological order in
   which every op appears after its producers, *and* mutually independent ops sit
   adjacent. Already-cached nodes are terminal (their whole subtree is skipped);
   views own no storage, so a dependency reached through a view contributes the
   view's *source* op. Wave order is not incidental — it is what makes barrier
   elision pay (below).
2. **Encode** (`encode_ops`) — allocate each op's output buffer (from the pool),
   record its read/written buffer keys, and encode its dispatch, committing a
   command buffer every
   [`OPS_PER_COMMAND_BUFFER`](../../crates/cider-press-runtime/src/eval.rs) (50)
   ops. The committed handles are returned **un-waited**.
3. **Populate caches** (`populate_caches`) — set each node's
   `OnceLock<LeafStorage>` from its output buffer. Synchronous `eval` does this
   after waiting every chunk; the bytes are host-readable by then.
4. **Detach** (`detach_order`) — clear each just-evaluated node's op inputs, so an
   evaluated tensor stops retaining its producers (MLX's detach-on-eval; see
   Performance for why this is load-bearing).

[`eval_async`](../../crates/cider-press-runtime/src/eval.rs) shares all four
passes but **commits without waiting**, returning a
[`PendingEval`](../../crates/cider-press-runtime/src/pending_eval.rs) that owns
the in-flight command buffers and pins the topo order's buffers until the GPU
finishes. It populates caches at commit time (so a *dependent* graph can chain
off the buffers GPU-side immediately) but sets an `in_flight` flag on each node
so *host* reads stay gated until `PendingEval::wait` (or its blocking drop)
observes completion. This is what pipelines decode: the generator commits token
*N+1*'s `eval_async` while holding token *N*'s `PendingEval`, so *N+1*'s encode
overlaps *N*'s GPU execution.

### Materialization state

A `TensorInner` carries immutable op metadata plus an `OnceLock<LeafStorage>`
cache, rather than a mutable `Source` enum. The two fields encode complementary
facts — *lineage* (the op) and *materialization* (the cache) — and the cache
makes exactly one monotonic transition, empty → populated. This means no lock on
the read path (`cpu_bytes` borrows straight from the `OnceLock`), no race window
between mutations, and `Send + Sync` falls out for free (it is `Arc<TensorInner>`
+ immutable metadata + a `Send + Sync` `OnceLock`). `eval_async`'s in-flight
state is a *third* logical state layered on top with the separate `in_flight`
flag, not a fourth cache variant — the design's `Source::Pending` sketch was
never needed.

### The buffer pool

Decode allocates ~1000 op-output buffers per token. The
[`BufferPool`](../../crates/cider-press-runtime/src/buffer_pool.rs) is a
size-keyed free-list: an op output's buffer returns to the pool when its tensor
node drops (between tokens) and is handed back out to the next token's eval, so
steady-state decode stops calling `newBufferWithLength:`. The load-bearing rule
is **only pool-minted buffers return** — the
[`KvCache`](./kv-cache.md) slab and host/constant leaves are minted *unpooled*
and never recycled, or they would corrupt cached data. Reuse is **cross-token**,
not within-eval: a token's whole graph stays live for its command-buffer chunks
and frees only as the next token starts (this is the gap to MLX's ~329 MiB peak;
see Open levers).

### Cross-buffer ordering: the per-output fence map

cider's buffers are allocated `HazardTrackingModeUntracked` — Metal does *not*
automatically order one command buffer against the next — so all cross-chunk,
cross-token, and pool-recycle ordering rests on an explicit fence scheme in
[`commands.rs`](../../crates/cider-press-kernels/src/commands.rs). This is the
**per-output fence map** ported faithfully from MLX (`Device::prev_outputs`): a
closing encoder waits only the prior-writer fences of the buffers it actually
reads or overwrites, then updates its own fence at close (capturing the whole
buffer — `updateFence` must be encoded at encoder *close*, before
`endEncoding`, or it signals immediately and orders nothing). Non-dependent
chunks therefore don't drain the GPU at the seam. Intra-encoder ordering is a
buffer-scope `memoryBarrier` that is **elided** unless an op reads a buffer
written since the last barrier — which is why the Kahn-wave schedule matters:
within a wave the first op pays the barrier and the rest elide and overlap.
(Two env escape hatches survive for bisecting sync bugs:
`CIDER_PRESS_TRACKED_BUFFERS=1` restores automatic hazard tracking,
`CIDER_PRESS_NO_BARRIER_ELISION=1` puts a barrier before every dispatch.)

### The sync-public / pipelined-internal decision

The caller-visible API is synchronous — no `async fn`, by deliberate up-front
decision — while the pipelining that closes the performance gap lives entirely
*inside* the eval boundary. The reasoning:

- **Sequential token generation has zero pipelining headroom *above* eval.** It
  is strictly data-dependent: token *N+1* needs token *N*'s logits. An async
  public API would buy nothing for this workload and would force the entire user
  codepath (tokenizer, sampler, CLI) to be async — contagion for no benefit.
- **Batched serving is a scheduler concern that sits *above* the engine, not
  inside it.** vLLM, llama.cpp, and MLX-LM all structure it this way: a
  synchronous per-request engine with an async scheduler on top. The runtime
  crate is the engine.
- **MLX's `mx.eval` is itself synchronous.** The per-dispatch gap lived in
  *internal* pipelining, not in the public API shape — and closing it (async
  decode pipelining, below) never touched an op signature.

The runtime is already `Send + Sync`, and the `eval_async` / `PendingEval`
machinery is exactly the groundwork a batched-serving scheduler would build on.

**Model variations:** none. This is the model-agnostic engine — the lazy graph,
the eval boundary, the pool, the fence map are identical for every model. This
is where a `LanguageModel` trait (to abstract over architectures) and a
batched-serving scheduler (to fan out concurrent requests over the sync engine)
would attach.

## Performance

**Async decode pipelining (detach-on-eval).** Decode was synchronous — one `commit_and_wait` per token —
so token *N+1*'s CPU work could not overlap token *N*'s GPU work. Making `eval`
non-blocking (`eval_async` + a `PendingEval` waited later) and chaining the
argmax id on-GPU (no readback) lets the generator run a depth-1 lookahead: token
*N+1* is encoded and committed while token *N* runs. Measured (M4 Max, warm,
post-fusion): **~317 → ~420 tok/s, a ~1.33× overlap**; the win hides *all*
per-token CPU work (the lazy graph-build, not just the encode slice) under the
GPU. Depth-1 is the natural depth for greedy decode; `--inflight-depth 2`
measured no further gain.

**Detach-on-eval was the enabler, and it fixed a sharp regression** (same
section). Chaining the argmax id on-GPU first *regressed* decode: feeding the
previous token's argmax (an op tensor) into the next forward made each token's
graph hold an `Arc` to the prior token's graph — a backward chain through the
whole decode. Buffer-pool reuse collapsed **98% → 0%** (~975 buffers
re-allocated per token), encode ballooned **~0.55 → ~3.2 ms/token**, and decode
fell *below* the synchronous baseline. Clearing each evaluated node's inputs
after caching (MLX's `array::detach()`) restored the pool to ~97% and unlocked
the pipelining gain. This is the same mechanism the [KV cache](./kv-cache.md)
relies on — basing each `slice_update` on the slab leaf rather than the prior
step's op-tensor — to keep the cache from pinning per-step graphs; together they
reached **~98% pool hit-rate**.

**Decode-step breakdown.**
`tensor.eval` is ~90% of the decode step and is **GPU-bound**: the encode/wait
split is ~14% CPU encode, ~86% GPU wait (was ~38%/62% before the pool). The
~0.51 ms/token CPU encode is small because per-op allocation is a pool hit ~98%
of the time; the GPU wait is the bottleneck, dominated by the 4-bit `qmv`
matvecs. The **GPU op-kind breakdown** (counter-sampled, within-table shares):
`quantized_matmul` ~41%, `binary` ~24% (residual/RoPE/SwiGLU elementwise),
`copy` ~11%, `unary` ~7%, `reduce` ~6%, `rope` ~4%, `sdpa` ~4%. The
attention copy/score-matmul chain collapsed into the single ~4% `sdpa` bucket
after the fused decode kernel landed (see [attention](./attention.md)).

**Memory.** Peak process RSS is **~900–916 MiB**
(vs `mlx_lm`'s ~329 MiB Metal high-water — these measure different things; cider
samples process RSS including the host safetensors `Vec`). The peak had
*regressed* to ~1192 MiB when the `SliceUpdate` collapse held all 24 layers'
intermediates live at once; it was resolved back to ~900 MiB not by the pool's
byte cap (free-list high-water is only ~94 MiB) but by the
[KvCache](./kv-cache.md) slab-base change that stopped per-step graphs being
pinned. The remaining gap to `mlx_lm` is **within-eval reuse** (see Open
levers).

**The decode dependent chain is *not* removable serialization.**
Decode is ~92% a sequential dependency
chain (the residual stream plus each layer's q→rope→slice-update→sdpa→o-proj and
rms→gate→silu→mul→down). The concurrent-encoder spike measured this directly:
**concurrent dispatch with correct hazard barriers is byte-identical to serial
but ~13% slower** (~210 vs ~240 tok/s), because ~900 of ~975 dispatches need a
barrier; the no-barrier "floor" (~510 tok/s) is fast only because it ignores
data dependencies and **computes garbage** — it is not an achievable target. The
real decode win came from the **MLX encoder/sync-model port** (untracked
buffers + the fence chain + concurrent encoders + barrier elision in Kahn-wave
order): the *barrier had to become rare, not merely cheap*. Kahn-wave scheduling
was the unlock (+22%, ~461 → ~561 tok/s), closing the last gap to parity (~561
vs `mlx_lm` ~564, ~1.00×), on top of which [qmv fast-path
padding](./quantized-matmul.md) (A1) put decode **~1.05× ahead** (~599 vs ~568
tok/s). The **large-workload validation** (724-token prompt, 1024 generated
tokens) confirms the lead holds at scale: **~513 vs ~488 tok/s, ~1.05×** (both
runtimes degrade proportionally as the KV cache grows).

## Open levers

The consolidated roadmap. Decode is at parity-plus and effectively closed on the
kernel side; the live work is prefill, memory, and features. Each lever links to
the step doc that owns its detail.

- **Within-eval buffer reuse** — the next memory lever, toward `mlx_lm`'s
  ~329 MiB peak (cider ~900 MiB). MLX frees intermediates *during* eval
  (depth-first) and reuses buffers within a single token; cider's pool reclaims
  **cross-token** only, keeping every intermediate of a token live simultaneously
  across its command-buffer chunks. Mid-eval freeing is the gap. (Note the
  large-workload **pool-churn** finding closed as a no-op: a 6.3% hit-rate under
  the large workload was prefill one-shot buffers squatting the byte cap, off the
  critical path — decode tok/s and peak RSS unmoved across a cap sweep.)
- **`.metallib` precompilation** — removes the cold-start JIT cost (the first
  compile of the flattened kernel source is tens of seconds, ~43 s) from
  first-token latency. A shipping concern, not a dev-loop one (the JIT result is
  cached cross-process). See [weights and quantization](./weights-and-quantization.md).
- **Chunked prefill via `fill_cache`** — for long prompts under bounded memory.
  The MLX-faithful T−1-fill + T=1-step split was measured a *pessimization* for
  short prompts and reverted, but
  [`fill_cache`](../../crates/cider-press-models/src/qwen2/model.rs) and its
  equivalence test were kept as the building block: chunking the cache fill caps
  peak working set on prompts longer than one chunk. See [KV cache](./kv-cache.md).
- **Prefill concurrency** — the <50%-occupancy floor. Prefill runs synchronously
  per `eval()`, and *every* dispatch-count lever was refuted: the cadence probe
  (fewer/bigger command buffers made GPU-wait *worse*), copy-count reduction, and
  the per-output fence map all measured no-ops. The reason is that the prefill
  chunks are **genuinely chain-dependent** — a 50-op chunk spans ~3–4
  sequential transformer layers, so the fence map waits exactly the fences the
  dependency chain already required. The ~2.2 ms "bubble" is **real
  data-dependency latency**, not over-serialization of independent work. Beating
  the floor needs genuine *concurrency* (overlapping independent work), not a
  leaner dispatch stream. See [attention](./attention.md).
- **Batched-serving scheduler** — concurrent requests fanned out over the
  synchronous engine. The `Send + Sync` runtime and the `eval_async` /
  `PendingEval` / `in_flight` machinery that stood in for the `Source::Pending`
  sketch already exist; the
  scheduler sits *above* the engine, where a `LanguageModel` trait would also
  attach.
- **Top-k / top-p / temperature sampling** — the clearest open feature. Only
  greedy argmax is wired today; there is **no** partial sampler scaffold. Adding
  stochastic sampling means a new decoding-strategy seam ahead of the argmax plus
  supporting kernels. See [sampling](./sampling.md).
- **MoE routing / expert dispatch** — top-k router + `gather_qmm` expert
  dispatch, the path to the eventual mixture-of-experts target. See
  [feed-forward](./feed-forward.md).
- **A2 (cider-owned small-N `qmv`): closed, no-go** — the zero-cost-k/v ceiling
  was only +1.07%, hidden by the async decode pipeline. There is no kernel-side
  decode headroom left worth owning. See [quantized matmul](./quantized-matmul.md).
