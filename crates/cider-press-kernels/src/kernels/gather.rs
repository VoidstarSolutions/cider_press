// Dispatch routine derived from MLX (https://github.com/ml-explore/mlx).
// Copyright © 2023-2024 Apple Inc. Released under the MIT license — see
// the full upstream notice at `kernels-mlx/COPYING`.

//! JIT-assembled gather dispatch.
//!
//! Unlike the contiguous unary / binary / reduce families we vendor as
//! single `.metal` files, MLX's gather kernel has no precompiled
//! source upstream: the kernel body is assembled at dispatch time per
//! instantiation tuple `(T, IdxT, NIDX, IDX_NDIM, LocT)`, mirroring
//! MLX's own `backend/metal/indexing.cpp`. We mirror that factoring:
//! a flattened header bundle (`GATHER_HEADERS` — `utils.h` + the two
//! `indexing/*.h`) is prepended to a per-instantiation wrapper string
//! produced from `GATHER_WRAPPER_TEMPLATE`, compiled, and dispatched.
//!
//! Scope: one instantiation suitable for Qwen2's
//! embedding lookup — `T = bfloat16_t`, `IdxT = uint`, `NIDX = 1`,
//! `IDX_NDIM = 1`, `LocT = int`, `axis = 0`, source rank 2. The
//! [`make_source`] helper is generic over the instantiation tuple so
//! adding f32/f16 source dtypes or other index dtypes is one call
//! site change.
//!
//! Buffer slot binding mirrors MLX's `Gather::eval_gpu` verbatim:
//!
//! | slot | binding |
//! |------|---------|
//! | 0    | `src` (device) |
//! | 1    | `out` (device) |
//! | 2    | `src_shape` (constant int*) |
//! | 3    | `src_strides` (constant `int64_t*`) |
//! | 4    | `src_ndim` (constant `size_t&`) |
//! | 5    | `slice_sizes` (constant int*) |
//! | 6    | `axes` (constant int*) |
//! | 7    | `idx_shapes` (constant int*) |
//! | 8    | `idx_strides` (constant `int64_t*`) |
//! | 9    | `idx_contigs` (constant bool*) |
//! | 10   | `idx_ndim` (constant int&) |
//! | 20+i | `idx_i` (device `IdxT*`) |
//!
//! Explained in the inference guide: [`docs/inference/embedding.md`](../../../../docs/inference/embedding.md).

use std::ffi::c_void;
use std::ptr::NonNull;

use objc2_metal::{MTLComputeCommandEncoder, MTLSize};

use crate::buffer::Buffer;
use crate::commands::Commands;
use crate::error::{Error, Result};
use crate::library::KernelLibrary;

/// Flattened `utils.h` + `indexing/{indexing,gather}.h` header bundle,
/// produced by `build.rs`. Prepend to a per-instantiation wrapper
/// source to form a complete compilable MSL string.
pub const GATHER_HEADERS: &str =
    include_str!(concat!(env!("OUT_DIR"), "/gather_headers_inlined.metal"));

/// Per-instantiation wrapper template, copied verbatim from MLX's
/// `backend/metal/jit/indexing.h` (the `gather_kernels` constant).
/// Positional placeholders match MLX's `fmt::format` call site exactly:
///
/// * `{0}` — `type_to_name(out) + idx_type_name` (function-name suffix
///   before the rank/LocT suffix; e.g. `bfloat16uint32`)
/// * `{1}` — output C type (e.g. `bfloat16_t`)
/// * `{2}` — index C type (e.g. `uint`)
/// * `{3}` — `NIDX` literal (e.g. `1`)
/// * `{4}` — `idx_args` (one `const device T* idxN [[buffer(20+N)]],`
///   per index source)
/// * `{5}` — `idx_arr` (`idx0, idx1, ...` — initializer list for the
///   `Indices<>` struct)
/// * `{6}` — `IDX_NDIM` literal
/// * `{7}` — `LocT` (`int` or `int64_t`)
///
/// The complete function name produced is
/// `gather{type_to_name(out)}{idx_type_name}_{NIDX}_{IDX_NDIM}_{LocT}`.
const GATHER_WRAPPER_TEMPLATE: &str = r"
[[kernel]] void gather{0}_{3}_{6}_{7}(
    const device {1}* src [[buffer(0)]],
    device {1}* out [[buffer(1)]],
    const constant int* src_shape [[buffer(2)]],
    const constant int64_t* src_strides [[buffer(3)]],
    const constant size_t& src_ndim [[buffer(4)]],
    const constant int* slice_sizes [[buffer(5)]],
    const constant int* axes [[buffer(6)]],
    const constant int* idx_shapes [[buffer(7)]],
    const constant int64_t* idx_strides [[buffer(8)]],
    const constant bool* idx_contigs [[buffer(9)]],
    const constant int& idx_ndim [[buffer(10)]],
    {4}
    uint3 index [[thread_position_in_grid]],
    uint3 grid_dim [[threads_per_grid]]) {{
  Indices<{2}, {3}> idxs{{
    {{ {5} }}, idx_shapes, idx_strides, idx_contigs, idx_ndim}};

  return gather_impl<{1}, {2}, {3}, {6}, {7}>(
      src,
      out,
      src_shape,
      src_strides,
      src_ndim,
      slice_sizes,
      axes,
      idxs,
      index,
      grid_dim);
}}
";

/// Instantiation parameters for one gather kernel specialization.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Instantiation {
    /// MSL C type of the source/output values (e.g. `"bfloat16_t"`).
    pub value_type: &'static str,
    /// Short name used in the function name (e.g. `"bfloat16"`).
    pub value_name: &'static str,
    /// MSL C type of the index buffer values (e.g. `"uint"`).
    pub index_type: &'static str,
    /// Short name used in the function name (e.g. `"uint32"`).
    pub index_name: &'static str,
    /// Number of index sources (matches `NIDX` template parameter).
    pub nidx: u32,
    /// Rank of each index tensor (matches `IDX_NDIM` template parameter).
    pub idx_ndim: u32,
    /// 32-bit (`int`) or 64-bit (`int64_t`) locator type for byte
    /// offsets. Use 32-bit when both src + indices stay under 2^31
    /// elements; 64-bit otherwise. MLX picks per-dispatch.
    pub loc_type: LocType,
}

/// Locator-type choice driving the `LocT` template parameter.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LocType {
    /// 32-bit `int`. Sufficient when `src.size() * sizeof(T)` fits in
    /// 2^31 bytes — the embedding lookup case for any reasonable
    /// vocabulary.
    Int32,
    /// 64-bit `int64_t`. Required for >2GB tensors.
    Int64,
}

impl LocType {
    fn as_str(self) -> &'static str {
        match self {
            Self::Int32 => "int",
            Self::Int64 => "int64_t",
        }
    }
}

impl Instantiation {
    /// `bf16` source, `u32` indices, single index buffer of rank 1,
    /// 32-bit locator. Suitable for Qwen2 embedding lookup (one rank-1
    /// `[T]` indices tensor → `[T, hidden]` output from a `[vocab,
    /// hidden]` source).
    #[must_use]
    pub const fn bf16_u32_rank1() -> Self {
        Self {
            value_type: "bfloat16_t",
            value_name: "bfloat16",
            index_type: "uint",
            index_name: "uint32",
            nidx: 1,
            idx_ndim: 1,
            loc_type: LocType::Int32,
        }
    }

    /// `u32` source, `u32` indices, single index buffer of rank 1,
    /// 32-bit locator. Used for gathering rows of an affine-quantized
    /// weight's packed `w` buffer (int4 packed 8-per-`uint32`).
    #[must_use]
    pub const fn u32_u32_rank1() -> Self {
        Self {
            value_type: "uint",
            value_name: "uint32",
            index_type: "uint",
            index_name: "uint32",
            nidx: 1,
            idx_ndim: 1,
            loc_type: LocType::Int32,
        }
    }

    /// Kernel function name MLX would produce for this instantiation
    /// (the `[[host_name]]`-equivalent — the wrapper isn't a template,
    /// so the function name is what `newFunctionWithName:` looks up).
    #[must_use]
    pub fn kernel_name(&self) -> String {
        format!(
            "gather{}{}_{}_{}_{}",
            self.value_name,
            self.index_name,
            self.nidx,
            self.idx_ndim,
            self.loc_type.as_str(),
        )
    }

    fn idx_args(&self) -> String {
        (0..self.nidx)
            .map(|i| {
                format!(
                    "const device {ty} *idx{i} [[buffer({slot})]],",
                    ty = self.index_type,
                    slot = 20 + i,
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn idx_arr(&self) -> String {
        (0..self.nidx)
            .map(|i| format!("idx{i}"))
            .collect::<Vec<_>>()
            .join(",")
    }
}

/// Assemble a complete, compilable MSL source string for the given
/// instantiation. Concatenates [`GATHER_HEADERS`] with a wrapper
/// expanded from `GATHER_WRAPPER_TEMPLATE`.
///
/// The corresponding kernel name (the `newFunctionWithName:` argument)
/// is [`Instantiation::kernel_name`].
#[must_use]
pub fn make_source(inst: &Instantiation) -> String {
    let wrapper = GATHER_WRAPPER_TEMPLATE
        .replace("{0}", &format!("{}{}", inst.value_name, inst.index_name))
        .replace("{1}", inst.value_type)
        .replace("{2}", inst.index_type)
        .replace("{3}", &inst.nidx.to_string())
        .replace("{4}", &inst.idx_args())
        .replace("{5}", &inst.idx_arr())
        .replace("{6}", &inst.idx_ndim.to_string())
        .replace("{7}", inst.loc_type.as_str())
        // The template uses {{ / }} as escaped braces in C++'s fmt
        // syntax; in our plain string-replace world they're literal
        // single braces that the MSL compiler needs.
        .replace("{{", "{")
        .replace("}}", "}");
    format!("{GATHER_HEADERS}\n{wrapper}")
}

/// Compute the 3D `(grid_dim, threadgroup_dim)` pair for a gather
/// dispatch. Mirrors MLX's `get_block_dims` logic: partition up to
/// `2^pow2` (default 1024) threads across 3 dimensions, growing each
/// dim by powers of 2 until either the dim's size or the budget runs
/// out.
fn block_dims(dim0: usize, dim1: usize, dim2: usize) -> MTLSize {
    let mut pows = [0u32; 3];
    let mut sum = 0u32;
    let axis_lens = [dim0, dim1, dim2];
    loop {
        let presum = sum;
        for (i, len) in axis_lens.iter().enumerate() {
            if *len >= 1usize << (pows[i] + 1) {
                pows[i] += 1;
                sum += 1;
            }
            if sum == 10 {
                break;
            }
        }
        if sum == presum || sum == 10 {
            break;
        }
    }
    MTLSize {
        width: 1usize << pows[0],
        height: 1usize << pows[1],
        depth: 1usize << pows[2],
    }
}

/// Buffers + per-call shape/stride metadata for one gather dispatch.
///
/// Metadata buffers use the exact Metal types the kernel binds (i32 /
/// i64 / u8 for `bool`). The caller owns every buffer and keeps it
/// alive through `commit_and_wait`.
pub struct GatherDispatch<'a, T> {
    /// Source tensor (the embedding table for the embedding case).
    pub src: &'a Buffer<T>,
    /// Output tensor, sized `idx_count * slice_size`.
    pub out: &'a mut Buffer<T>,
    /// Index tensors. `inst.nidx` must match `indices.len()`.
    pub indices: &'a [&'a Buffer<u32>],
    /// `src.shape()` as `int32` values.
    pub src_shape: &'a Buffer<i32>,
    /// `src.strides()` as `int64` values (element strides, not byte
    /// strides — MLX's convention).
    pub src_strides: &'a Buffer<i64>,
    /// `src.ndim()` (passed via `setBytes`, not a buffer).
    pub src_ndim: u64,
    /// Per-axis slice sizes: `1` for each axis indexed by `axes`, the
    /// matching `src.shape(ax)` otherwise.
    pub slice_sizes: &'a Buffer<i32>,
    /// Gathered axes (one entry per index source).
    pub axes: &'a Buffer<i32>,
    /// Per-index packed shapes (`indices[i].shape()` concatenated).
    pub idx_shapes: &'a Buffer<i32>,
    /// Per-index packed strides (`indices[i].strides()` concatenated).
    pub idx_strides: &'a Buffer<i64>,
    /// Per-index row-contiguous flags (one `u8`, value 0 or 1, per
    /// index source — Metal's `bool` is 1 byte but we keep the Rust
    /// representation explicit).
    pub idx_contigs: &'a Buffer<u8>,
    /// Common rank of all index tensors (passed via `setBytes`).
    pub idx_ndim: i32,
    /// Grid dimensions `(dim0, dim1, dim2)`:
    /// - `dim0 = indices[0].shape(0)` (first index axis)
    /// - `dim1 = indices[0].size() / dim0` (remaining index axes)
    /// - `dim2 = slice_size` (product of preserved src axes)
    pub grid: (usize, usize, usize),
}

/// Dispatch one gather kernel call into `commands`. The caller is
/// responsible for: building `library` from [`make_source`], populating
/// every metadata buffer with the bytes the kernel expects, and
/// committing the encoded command buffer via [`Commands::commit_and_wait`].
///
/// This intentionally takes the raw buffers — the runtime layer wraps
/// it with type-safe `Tensor` semantics. Generic over `T`: gather's
/// Metal binding only cares about the buffer's *size*, not its
/// element type, so the Rust dispatch function is dtype-agnostic. The
/// caller picks an [`Instantiation`] whose `value_type` matches `T`'s
/// MSL representation.
#[allow(clippy::needless_pass_by_value)]
pub fn dispatch<T>(
    commands: &mut Commands<'_>,
    library: &KernelLibrary,
    inst: &Instantiation,
    args: GatherDispatch<'_, T>,
) -> Result<()> {
    if args.indices.len() != inst.nidx as usize {
        return Err(Error::InvalidArgument(format!(
            "gather: instantiation expects {} index buffer(s), got {}",
            inst.nidx,
            args.indices.len(),
        )));
    }

    let kernel_name = inst.kernel_name();
    let pipeline = library.pipeline(&kernel_name)?;

    let src_ndim = args.src_ndim;
    let idx_ndim = args.idx_ndim;
    let (d0, d1, d2) = args.grid;

    {
        let encoder = commands.encoder()?;
        encoder.setComputePipelineState(pipeline.metal_pipeline_state());
        // SAFETY: slot/dtype bindings match MLX's `Gather::eval_gpu` in
        // `backend/metal/indexing.cpp` verbatim. The runtime layer owns
        // buffer typing; this function is called only via the gated
        // `Tensor::gather` path which enforces dtype agreement.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(args.src.metal_buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(args.out.metal_buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(args.src_shape.metal_buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(args.src_strides.metal_buffer()), 0, 3);
            let ndim_ptr: NonNull<c_void> = NonNull::from(&src_ndim).cast();
            encoder.setBytes_length_atIndex(ndim_ptr, std::mem::size_of::<u64>(), 4);
            encoder.setBuffer_offset_atIndex(Some(args.slice_sizes.metal_buffer()), 0, 5);
            encoder.setBuffer_offset_atIndex(Some(args.axes.metal_buffer()), 0, 6);
            encoder.setBuffer_offset_atIndex(Some(args.idx_shapes.metal_buffer()), 0, 7);
            encoder.setBuffer_offset_atIndex(Some(args.idx_strides.metal_buffer()), 0, 8);
            encoder.setBuffer_offset_atIndex(Some(args.idx_contigs.metal_buffer()), 0, 9);
            let idx_ndim_ptr: NonNull<c_void> = NonNull::from(&idx_ndim).cast();
            encoder.setBytes_length_atIndex(idx_ndim_ptr, std::mem::size_of::<i32>(), 10);
            for (i, buf) in args.indices.iter().enumerate() {
                encoder.setBuffer_offset_atIndex(Some(buf.metal_buffer()), 0, 20 + i);
            }
        }
    }

    let grid = MTLSize {
        width: d0,
        height: d1,
        depth: d2,
    };
    let threadgroup = block_dims(d0, d1, d2);
    commands.dispatch_threads(grid, threadgroup)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_instantiation_kernel_name_matches_mlx_format() {
        let inst = Instantiation::bf16_u32_rank1();
        assert_eq!(inst.kernel_name(), "gatherbfloat16uint32_1_1_int");
    }

    #[test]
    fn u32_instantiation_kernel_name_matches_mlx_format() {
        let inst = Instantiation::u32_u32_rank1();
        assert_eq!(inst.kernel_name(), "gatheruint32uint32_1_1_int");
    }

    #[test]
    fn embedding_instantiation_idx_args_uses_slot_20() {
        let inst = Instantiation::bf16_u32_rank1();
        assert_eq!(inst.idx_args(), "const device uint *idx0 [[buffer(20)]],");
        assert_eq!(inst.idx_arr(), "idx0");
    }

    #[test]
    fn block_dims_packs_three_dimensions() {
        // 1 thread per axis when the axis is degenerate.
        let dims = block_dims(1, 1, 1);
        assert_eq!((dims.width, dims.height, dims.depth), (1, 1, 1));

        // Pure-z dispatch (slice_size=896 — typical embedding hidden
        // dim). With dim0=dim1=1, all 1024 threads end up on z.
        let dims = block_dims(1, 1, 896);
        assert_eq!((dims.width, dims.height, dims.depth), (1, 1, 512));

        // Two-dimension dispatch (indices count + slice). Budget of
        // 1024 is split greedily as MLX does.
        let dims = block_dims(8, 1, 896);
        assert_eq!((dims.width, dims.height, dims.depth), (8, 1, 128));
    }
}
