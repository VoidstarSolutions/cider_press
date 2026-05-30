# Runtime design: lazy `Tensor` + graph + `eval()`

Planning doc for the next slice of `cider-press-runtime`. Picks up
where `CLAUDE.md`'s post-spike Findings left off. Scope is the layer
that turns the data primitives (`DType`, `Shape`, `Strides`, `Layout`,
`Quantization`, the `Tensor` skeleton) into something that can hold
real buffers, build a graph of operations, and execute them on the
GPU via `cider-press-kernels`.

This is a *planning* doc, not a spec. Decisions are recorded with
their reason so future-me can challenge them when something doesn't
fit. Code lands in follow-up commits.

---

## Goals (this slice)

1. **Storage.** A `Tensor` can be backed by a real Metal buffer, with
   a typed constructor surface (`zeros`, `from_slice`, `from_bytes`)
   that allocates via the kernels-layer `Device`.
2. **Lazy graph.** A `Tensor` can also represent the *result* of an
   op on other tensors — no GPU work runs at construction.
3. **`eval()`.** Walking the graph from an unevaluated `Tensor`
   allocates output buffers, encodes dispatches into a `Commands`,
   commits, and waits. After `eval()`, every reachable node is a
   materialized leaf.
4. **First op coverage.** Enough ops to demonstrate the pattern
   end-to-end. Concretely: `copy` (the identity mover) and one
   quantized matmul (`qmv`, bit-exact vs MLX). That's two ops with
   very different dispatch shapes — enough to stress-test the
   abstraction without over-extending it.

## Decided up-front: sync public API

This one is called out separately because, unlike a deferred
optimization, it shapes every op signature in the runtime. **The
caller-visible API is synchronous. No `async fn`. Forever.**

The reasoning, briefly:

- Sequential token generation is the dominant workload and is
  strictly data-dependent (token N+1 needs token N's logits). It
  has zero pipelining headroom *above* the eval boundary. An async
  API would buy nothing for this case and would force the entire
  user codepath (tokenizer, sampler, CLI) to be async — contagion
  for no benefit.
- Batched serving is real, but it's a *scheduler* concern that sits
  above the inference engine, not inside it. vLLM, llama.cpp, and
  MLX-LM all structure things this way: sync per-request engine,
  async scheduler on top. The runtime crate is the engine.
- The 1.5× per-dispatch perf gap to MLX lives in *internal* dispatch
  pipelining, not in the public API shape — MLX's `mx.eval` is
  itself sync. Closing the gap is an implementation concern; see
  "What pipelining needs" below.

Picking an async runtime (tokio / smol / executor-agnostic
futures) would be a heavy API tax that doesn't pay for itself on
this hardware and workload.

## Non-goals (deferred)

- **Buffer pool / allocator.** Use raw `Device::new_buffer` per
  allocation. The MLX `allocator.cpp` design is the reference for
  later, but pooling is an optimization, not a design constraint on
  the public API.
- **Cross-buffer command-buffer batching / internal pipelining.**
  First cut: one `Commands` per `eval()`, synchronous commit +
  wait. The 1.5× per-dispatch perf gap to MLX lives here — but
  closing it is a follow-up, not a prerequisite. The graph shape
  we pick here is what *enables* batching later (see
  `Source::Pending` below); we don't need to land the scheduling
  logic now.
- **Autograd.** Inference only. Ops don't carry backward closures.
- **Multi-device, multi-stream.** Single shared device, single
  command queue. Apple-Silicon-only scope makes this trivial.
- **In-place ops, views, broadcasting tricks.** Get the eager case
  right first; views and broadcasting are layered on top by
  decorating `Layout` rather than threading new variants through
  every op.

## Mental model

Same as MLX's `array` + lazy eval, in idiomatic Rust:

- `Tensor` is a cheap-to-clone handle. Cloning bumps an `Arc`.
- Behind the `Arc` is a `TensorInner` carrying `Shape`, `DType`,
  `Layout`, and a `Source`.
- `Source` is the discriminant for *how this tensor's values are
  produced*. Three variants in this slice:
  - `Placeholder` — metadata only (existing). Stays around for
    test/debug scaffolding; not used by op nodes.
  - `Leaf(Buffer)` — materialized. Backed by a real `MTLBuffer`
    (via the kernels-crate `Buffer` wrapper). Either user-supplied
    (from `zeros` / `from_slice`) or the result of a completed
    `eval()`.
  - `Op(OpNode)` — produced by an op. Carries the op kind and its
    input tensors. The op metadata is immutable; what changes at
    `eval()` time is a separate `OnceLock<Buffer>` cache (see
    "Materialization state" below). An op tensor with a populated
    cache is the post-eval state — the lineage is preserved.
- Building a graph: `Tensor::copy(&a)` returns a new `Tensor` whose
  `Source` is `Op { kind: Copy, inputs: [a.clone()] }`. No GPU
  work yet.
- `t.eval()` walks the DAG rooted at `t`, allocates output buffers
  for every unevaluated `Op` node, encodes them into one `Commands`,
  commits, waits, and populates each op's result cache. After
  `eval()`, `t` and every dependency are materialized.

### What pipelining needs (future `Source::Pending`)

The sync-public / pipelined-internal split (see "Decided up-front"
above) is what closes the 1.5× perf gap to MLX without changing op
signatures. The data model has to admit it without rework. One
future variant does the job:

```rust
enum Source {
    Placeholder,
    Leaf(Buffer),                // CPU-readable, GPU-done
    Op(OpNode),                  // not yet encoded
    Pending(InFlight),           // committed, not yet completed (future)
}

struct InFlight {
    buffer: Buffer,                              // already allocated
    command_buffer: Retained<MTLCommandBuffer>,  // status / waitUntilCompleted
}
```

In the implemented model (immutable op metadata + `OnceLock<Buffer>`
cache; see "Materialization state" below), pipelining is just a
third state for the cache slot: instead of two states (empty,
populated) we'd have three (empty, in-flight, ready). Two viable
shapes for that extension:

- Add an `in_flight: OnceLock<InFlight>` parallel field, set when
  eval encodes and commits but doesn't wait; `cache` is populated
  from a completion handler.
- Replace `OnceLock<Buffer>` with a small custom state machine
  (`Empty | Pending(InFlight) | Ready(Buffer)`) behind a `Mutex`,
  paying the read-path lock cost in exchange for atomic transitions.

The first-slice `eval()` skips the in-flight state entirely — it
calls `waitUntilCompleted` before populating the cache, so readers
only ever see the empty → ready transition. Picking between the
two extension shapes can wait until the scheduler design is
concrete; the public API doesn't depend on the choice. Metal's
`MTLCommandBuffer` exposes `status` and `waitUntilCompleted`
natively — no futures executor required to express "in flight" or
"wait."

## The graph: data-only, not a separate type

There is *no* separate `Graph` type. The graph is the transitive
closure of `Op::inputs` reachable from the `Tensor` you call
`eval()` on. This mirrors MLX (`array` self-describes its lineage)
and keeps the API surface narrow. It also means subgraphs are
implicit — calling `eval()` on `b` after `eval()` on `a` only
evaluates what's actually unmaterialized.

The op enum is closed (we own it). No `dyn Op` trait until we have
a reason — extension by enum variant is fine while the op set is
small. When it grows past ~20 ops the enum-vs-trait tradeoff is
worth revisiting; not before.

```rust
enum OpKind {
    Copy,
    Qmv { /* group_size, bits, etc. -- see Quantization */ },
    // grows
}

struct OpNode {
    kind: OpKind,
    inputs: SmallVec<[Tensor; 4]>, // most ops have ≤4 inputs
}
```

## Type-system leverage

Once the sync-API decision is locked in (see "Decided up-front"),
the type system isn't carrying any async / materialization-state
load. That frees it for the places it earns its keep: turning
op-level preconditions into compile-time guarantees, and keeping
construction-time information available without forcing every
helper to be generic.

Things we *do* lean on the type system for:

- **Newtypes at the op boundary.** Most ops are layout-agnostic; a
  few aren't. `qmv` requires a quantized weight with scales and
  biases; an attention kernel requires a contiguous KV-cache slab;
  a softmax wants a logits tensor of a specific rank. Encode that
  in the signature:
  ```rust
  fn qmv(w: &QuantizedWeight, x: &Tensor) -> Tensor
  ```
  not `fn qmv(a: &Tensor, b: &Tensor) -> Result<Tensor>`.
  `QuantizedWeight` is a thin newtype around `Tensor` that only
  `Tensor::quantized_from_bytes(...)` can construct; the
  `Layout::Quantized(_)` check happens once, at construction.
  Thereafter the op signatures are infallible at the type level.
  Same pattern will apply to `KvCache`, `AttentionMask`,
  `LogitsTensor`, etc. as they show up.
- **Dtype-typed at construction, dtype-erased on the handle.**
  Already what we do — `Tensor::placeholder::<f32>(...)` takes the
  dtype as a type parameter for ergonomic typed construction;
  `Tensor::dense_placeholder(shape, DType::BF16)` is the
  runtime-typed escape hatch for model loading. The handle stores
  `DType` dynamically so a `Vec<Tensor>` of mixed dtypes is
  expressible without `Box<dyn Anything>`.
- **Sealed `OpKind` enum.** Closed set, no `dyn Op` trait. We own
  every op; the extensibility tax of a trait isn't worth paying
  for a hypothetical third-party extension that isn't in scope.
  Switch to a trait if/when the variant count gets unwieldy or
  external extension becomes a real requirement.
- **`Arc`-based identity for graph dedup.** `Arc::ptr_eq` is the
  equivalence relation for "same node." Common subexpression
  elimination is mechanical (walk the graph, collect by pointer
  identity) rather than requiring us to invent a hash for ops.
- **`Send + Sync` for `Tensor`.** Falls out of `Arc<TensorInner>` +
  immutable op metadata + `OnceLock<Buffer<u8>>` (with
  `Buffer<u8>: Send + Sync` added in the kernels crate). That's
  what lets a future batched-serving scheduler layer sit on top of
  the runtime without us having to retrofit thread-safety into
  core types.

Things we deliberately *don't* encode in the type system:

- **Materialization state on the handle** (typestate `Tensor<Lazy>`
  / `Tensor<Pending>` / `Tensor<Ready>`). Catches no real bug —
  the right behavior for `t.cpu_bytes()` on a not-yet-eval'd tensor
  is to implicitly evaluate, not to refuse to compile — and it
  would force every helper that takes a `Tensor` to be generic
  over state. Net negative.
- **Const-generic rank** (`Tensor<F32, 3>`). LLM forward passes
  reshape, broadcast, and slice constantly; the static rank ends
  up dynamic at every interesting boundary. Not worth the tax.
- **Compile-time dtype on the handle.** Same reason — model
  loading produces dtype-at-runtime tensors and the generic
  explosion downstream would be severe. We get the wins of typed
  construction via the typed constructor surface, then erase.
- **Compile-time device** (`Tensor<Gpu>` etc.). Single-platform
  scope makes the abstraction empty.

The newtype-at-the-op-boundary pattern is the high-leverage one to
get right before the first op signatures cast: it's both the most
load-bearing for safety and the hardest to retrofit later without
breaking callers.

## Storage: `Buffer` ownership

The kernels crate already owns `Buffer` (an `MTLBuffer` wrapper).
The runtime crate's `Source::Leaf` carries a `Buffer` directly.
Lifetimes are managed by `Arc<TensorInner>` — when the last clone
drops, the inner drops, the buffer drops, the `MTLBuffer` releases.

`Device` handle: `Tensor`s capture an `Arc<Device>` on construction.
- Single-device scope makes this implicit (created once at startup
  or lazily on first allocation, lives forever in practice).
- Carrying it on every tensor means `t.eval()` works without an
  explicit device arg — ergonomic.
- Op constructors assert inputs share the same `Arc<Device>` (cheap
  `Arc::ptr_eq`). Trivially true today; the check is a guard for
  later.

Open: do we expose `Device` as a public type, or hide it behind a
process-global `default_device()` like MLX? Lean toward **expose
it** — explicit > magical for a library — but provide a
`Device::shared()` convenience that returns the lazily-initialized
global. Easy to use, easy to test, no global mutation pain because
device handles are immutable.

## Materialization state: `Option<OpNode>` + `OnceLock<Buffer>`

`eval()` needs each unevaluated node to acquire a materialized
buffer so that downstream ops see real bytes.

Two viable shapes were on the table:

1. **`Mutex<Source>` inside `TensorInner`.** One enum, locked
   discriminant; eval rewrites `Source::Op` → `Source::Leaf(buffer)`.
2. **Split fields: immutable op metadata + `OnceLock<Buffer>` cache.**
   Op metadata is set at construction and never mutated; the buffer
   slot transitions empty → populated exactly once, monotonically.

**Picked (2).** Originally this doc preferred (1), but implementation
revealed option (2) is materially simpler:

- **No lock on the read path.** `cpu_bytes()` can return `&[u8]`
  borrowed directly from the `OnceLock` storage — no `MutexGuard`
  lifetime leaking into the public API, no closure form needed.
- **No race window between mutations.** There's exactly one
  transition (cache empty → populated); the op-metadata field never
  moves. Compare with a `Mutex<Source>` rewrite, which has to
  atomically swap a whole variant.
- **State enumeration is still clean.** `(op, cache)` deterministically
  encodes four states: placeholder (None, None), host-constructed
  leaf (None, Some), unevaluated op (Some, None), evaluated op
  (Some, Some). The "two fields encoding one state" smell the
  earlier draft worried about doesn't materialize — the fields
  encode *complementary* facts (lineage vs. materialization), not
  the same fact twice.
- **`Send + Sync` falls out** the same way: `OnceLock<Buffer<u8>>`
  is `Send + Sync` given `Buffer<u8>: Send + Sync` (added to the
  kernels crate alongside this slice).

The trade-off this gives up: an evaluated op tensor reports both
`is_op() == true` and `is_materialized() == true`. That's accurate
(it's an op-produced tensor whose result is cached) and not
misleading — `is_op` queries lineage, `is_materialized` queries
readiness.

## `eval()` algorithm (first cut)

```text
fn eval(&self):
    1. Collect topological order of unevaluated Op nodes reachable
       from self. Reverse post-order DFS, dedup by Arc pointer
       identity. Skip nodes whose OnceLock cache is already
       populated (host leaves and previously-evaluated ops) and
       Placeholders (no op, no cache).
    2. For each Op node in order:
         a. Allocate output Buffer (via Device::alloc_buffer).
         b. Look up the dispatcher for OpKind in a match.
         c. Dispatcher encodes into the shared Commands, given:
              - input Buffers (already materialized by prior iters,
                resolved via the side-table of in-progress outputs
                or the input's OnceLock cache)
              - output Buffer
              - shape/dtype/layout from TensorInner
    3. Commands.commit_and_wait().
    4. For each Op node in order, populate its OnceLock<LeafStorage>
       cache with the now-valid output buffer. The op metadata
       (Option<OpNode>) stays put — lineage is preserved.
```

Simplicity choices:

- One `Commands` per `eval()`. No mid-walk commit. Closes the door
  on the perf-batching question, opens it back up when we need it.
- Synchronous wait. No completion handlers, no async API yet. The
  perf gap this leaves on the table is the same one MLX already
  beats us by 1.5× on; we're not trying to close it here.
- Output buffer size is computed from `Shape` + `DType` (dense) or
  `Shape` + `Quantization` (quantized) — both already implemented
  on the data primitives.

## Quantized tensors as first-class

Per memory `feedback_api_design`: quantized layouts ride along
from day one. Concretely:

- `Tensor::quantized_placeholder` already exists. Add
  `Tensor::quantized_from_bytes(shape, quantization, w, scales,
  biases)` — takes the three packed buffers MLX's `qmv` needs and
  constructs a single `Tensor` whose `Layout::Quantized(_)` carries
  the descriptor. Internally this is *one* `Tensor` of dtype
  `U32` (the packing word) plus side-channel `Buffer`s for scales
  and biases.
- Open question: should scales/biases be separate `Tensor`s
  (composite op input list), or fields on a `QuantizedLeaf` struct?
  Lean toward **fields on the leaf** — they're an inseparable unit
  in MLX's data model, and treating them as independent tensors
  invites the user to mismatch them. The composite-input shape
  is easy to switch to later if it turns out we want it (e.g.
  shared scales across multiple weight tensors).

This is the first place the abstraction will feel awkward, which is
exactly why we want it in the first slice — surface the awkwardness
before the API ossifies.

## File layout in `cider-press-runtime`

What actually landed (the planning sketch in earlier drafts was a
finer-grained split with an `ops/` subdir; the implementation kept
it flatter because the per-op dispatcher is small enough to live
inline in `eval.rs` for now):

```text
src/
├── device.rs        # Arc<DeviceInner> handle, shared(), kernel-lib caches
├── dtype.rs
├── error.rs
├── eval.rs          # topo walk + per-OpKind dispatcher
├── layout.rs
├── lib.rs
├── quantization.rs
├── quantized.rs     # QuantizedWeight newtype + matvec op builder
├── shape.rs
├── strides.rs
└── tensor.rs        # Tensor + TensorInner + OpKind + op_tensor builder
```

A finer split (per-op file under `ops/`) becomes worth it when the
op count grows past the point where a single match in `eval.rs` is
unwieldy — probably around SDPA + attention prerequisites.

## Commit plan

Stage-by-stage, one logical concern per commit (per memory
`feedback_workflow`):

1. **`Device` handle + `Source::Leaf`.** Add `device.rs`. Replace
   `Tensor` constructors that today produce `Placeholder` with a
   typed surface: `zeros`, `from_slice`, `from_bytes`. Each
   allocates a real `Buffer`. `Placeholder` stays as the
   metadata-only variant for tests.
2. **`OpKind` + lazy graph.** Add `ops/mod.rs` and `source.rs`.
   Add `Tensor::copy(&self)` as the first op — it returns a new
   `Tensor` with `Source::Op { kind: Copy, inputs: [self.clone()] }`.
   No `eval()` yet; new tests assert the graph shape (rank, dtype,
   etc. propagate correctly through the op).
3. **`eval()` for `Copy`.** Topo walk, single `Commands`, synchronous
   commit + wait. New test: `Tensor::from_slice(&[1,2,3]).copy().eval()`
   produces a leaf whose buffer reads back as `[1,2,3]`.
4. **`Qmv` op + quantized leaf.** Add
   `Tensor::quantized_from_bytes`, `Tensor::qmv(&w, &x)`, and the
   dispatcher in `ops/qmv.rs`. Reuse the qmv parity fixture as the
   parity test, but now driven through the runtime API instead of
   raw kernels-crate code.

Each step ends with green tests, formatted code, and updated
`lib.rs` rustdoc reflecting the new status. `CLAUDE.md` gets a
status line update at the end of step 4 (the slice is "done" when
both example ops are wired through the runtime).

## Risks & things I expect to get wrong

- **`OnceLock<Buffer>` race semantics.** Two threads concurrently
  evaluating the same node both produce an output; one set wins,
  the other's buffer is dropped. Wastes work but is correct (both
  produce the same bytes). Fine for now; revisit when concurrent
  eval becomes a thing.
- **Op enum vs trait.** I'll regret picking enum the moment we
  want to extend ops outside the crate. Acceptable trade for the
  inference-only single-crate scope; revisit if/when `cider-press-models`
  starts wanting custom ops.
- **Quantized leaf shape.** The "fields on the leaf" decision is
  the place I most expect to revise once `qmm`, `dequantize`, and
  attention's KV-cache quant land. Document the alternative
  (composite-input form) in the code so the switch is mechanical.
- **Buffer reuse.** Per-`eval()` allocation will look slow on a
  real forward pass. The pool design is deliberately deferred but
  is the next obvious follow-up; reserve the file slot for it
  (`src/pool.rs`) once `eval()` lands.

## After this slice: framework gaps for inference

The target use case is **Qwen2.5-0.5B-Instruct** running end-to-end
(see `docs/ARCHITECTURE.md` for the architecture map). It's the
smallest published dense Qwen — same architecture family as the
eventual MoE target minus the router and expert dispatch, so every
op carries forward. Picking a concrete target now
keeps the gap-fill work honest: each addition has a known consumer.

Two framework primitives we know we'll need before the op flood:

### Views (covers reshape, transpose, slicing, *and* broadcasting)

Attention needs per-head reshape + transpose of `[B, T, H_q*D_h]`
into `[B, H_q, T, D_h]` without a copy. KV cache reads need slicing
the populated prefix `[..T_cache, ...]`. RMSNorm and elementwise ops
need to broadcast `[D]` gamma over `[B, T, D]` — which is just a
view with 0 strides on the leading dims, the MLX answer.

Recommended shape: a third option alongside `op` and host-leaf
construction. Sketch:

```rust
struct TensorInner {
    shape, dtype, layout, device,
    op: Option<OpNode>,
    view: Option<ViewSource>,        // new
    cache: OnceLock<LeafStorage>,
}

struct ViewSource {
    source: Tensor,
    byte_offset: usize,
}
```

`cpu_bytes()` on a view follows the chain to a materialized buffer
and slices. Dispatchers that require contiguous inputs trigger an
internal `Copy` to materialize a strided view; later, strided-aware
kernels (MLX's `g_copy` family etc.) can be wired in for perf.
Mutually exclusive with `op` at construction; `cache` populates
only for materialized leaves, never for pure views.

**Trigger to land**: first op that needs reshape (any attention
work) — likely the SDPA branch.

### `KvCache` type (separate from `Tensor`)

The decode loop appends one (K, V) row per token to a pre-allocated
slab. Semantics are *set-by-position-monotonically*, which doesn't
fit the `OnceLock<LeafStorage>` "set once" model. Recommended
shape: a separate type, not a `Tensor` variant.

```rust
pub struct KvCache {
    device: Device,
    keys: Buffer<u8>,          // [max_T, H_kv, D_h]
    values: Buffer<u8>,        // [max_T, H_kv, D_h]
    dtype: DType,
    head_dim: usize,
    n_heads: usize,
    position: usize,           // how many slots are filled
}

impl KvCache {
    pub fn new(device: &Device, max_tokens: usize,
               n_heads: usize, head_dim: usize, dtype: DType) -> Result<Self>;
    pub fn update(&mut self, k: &Tensor, v: &Tensor) -> Result<()>;
    pub fn keys_view(&self) -> Tensor;     // view of populated prefix
    pub fn values_view(&self) -> Tensor;
}
```

`update` runs a `Copy` (or a fused write kernel later) into the
slab. The views read by SDPA piggyback on the Views primitive
above. The `update` op is *eager* — it commits to the GPU
immediately — because the decode loop blocks on the result anyway;
no graph value to gain from laziness.

**Trigger to land**: SDPA branch (Views must come first).

### Smaller framework choices that follow from the above

- **Multi-output ops**: not needed for dense Qwen2.5 (SDPA can be
  split into matmul + softmax + matmul; KV updates happen
  out-of-graph). Defer until the first fused op that needs it.
- **Op trait vs. enum**: enum keeps winning for now. Revisit if/when
  the variant count gets unwieldy (>~20) or third-party ops become
  a real requirement.
- **In-place ops**: `KvCache::update` is the only in-place case in
  Qwen, and it lives outside the graph. Defer broader in-place
  semantics until needed.

## After Qwen2.5-0.5B runs

These are the perf and ergonomic follow-ups that only matter once
there is something running to measure:

- Buffer pool / allocator (reference: MLX's `allocator.cpp`).
- Cross-`eval()` command-buffer batching — the lever that should
  close the 1.5× per-dispatch perf gap to MLX. Within a single
  `eval()` we already batch; this would batch across the decode
  loop's per-token `eval()` calls.
- `.metallib` precompilation strategy for shipping (cold-start
  matters for first-token latency in production).
- The full MoE story: top-k routing, expert dispatch (MLX's
  `gather_qmm`), the loop-or-fuse decision.
