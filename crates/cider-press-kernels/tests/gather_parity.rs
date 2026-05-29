//! Parity test for the JIT-assembled `gather` dispatch.
//!
//! Drives the kernels-crate `gather::dispatch` against a fixture
//! produced by `uv run scripts/dump_mlx_op.py gather` (which calls
//! `mx.take(src, indices, axis=0)`). Gather is a pure data-mover, so
//! the bar is bit-exact equality for any dtype.
//!
//! Scope: the embedding-shaped instantiation only — `bf16` source,
//! `u32` indices, single rank-1 index tensor, axis 0, rank-2 source.

#![cfg(target_os = "macos")]
#![allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]

use std::path::Path;

use cider_press_kernels::kernels::gather::{self, GatherDispatch, Instantiation, make_source};
use cider_press_kernels::{Buffer, Device, KernelLibrary};
use cider_press_test_utils::{dump_mlx_op, read_bf16, read_u32, tempdir};
use half::bf16;
use safetensors::SafeTensors;

const VOCAB: usize = 128;
const HIDDEN: usize = 64;
const N_INDICES: usize = 8;

// Smaller u32 case to keep test data trivial; gather is dtype-agnostic
// so size doesn't matter for parity.
const U32_VOCAB: usize = 32;
const U32_HIDDEN: usize = 16;
const U32_N: usize = 4;

#[test]
fn gather_axis0_bf16_u32_rank1_matches_mlx_bit_exact() {
    let (src_host, indices_host, out_ref) = fixture();
    assert_eq!(src_host.len(), VOCAB * HIDDEN);
    assert_eq!(indices_host.len(), N_INDICES);
    assert_eq!(out_ref.len(), N_INDICES * HIDDEN);

    let device = Device::system_default().expect("Metal device");
    let inst = Instantiation::bf16_u32_rank1();
    let source = make_source(&inst);
    let library = KernelLibrary::from_source(&device, &source).expect("compile JIT gather");

    let src: Buffer<bf16> = device.upload(&src_host).expect("upload src");
    let indices: Buffer<u32> = device.upload(&indices_host).expect("upload indices");
    let mut out: Buffer<bf16> = device.alloc_buffer(N_INDICES * HIDDEN).expect("alloc out");

    // Per MLX's Gather::eval_gpu binding order:
    let src_shape: Buffer<i32> = device
        .upload(&[VOCAB as i32, HIDDEN as i32])
        .expect("src_shape");
    let src_strides: Buffer<i64> = device.upload(&[HIDDEN as i64, 1]).expect("src_strides");
    // axis 0 is gathered (slice size 1); axis 1 is preserved.
    let slice_sizes: Buffer<i32> = device.upload(&[1, HIDDEN as i32]).expect("slice_sizes");
    let axes: Buffer<i32> = device.upload(&[0]).expect("axes");
    let idx_shapes: Buffer<i32> = device.upload(&[N_INDICES as i32]).expect("idx_shapes");
    let idx_strides: Buffer<i64> = device.upload(&[1]).expect("idx_strides");
    let idx_contigs: Buffer<u8> = device.upload(&[1u8]).expect("idx_contigs");

    let mut cmds = device.commands().expect("commands");
    {
        let idx_refs = [&indices];
        gather::dispatch(
            &mut cmds,
            &library,
            &inst,
            GatherDispatch {
                src: &src,
                out: &mut out,
                indices: &idx_refs,
                src_shape: &src_shape,
                src_strides: &src_strides,
                src_ndim: 2,
                slice_sizes: &slice_sizes,
                axes: &axes,
                idx_shapes: &idx_shapes,
                idx_strides: &idx_strides,
                idx_contigs: &idx_contigs,
                idx_ndim: 1,
                // dim0 = first index axis size; dim1 = remaining index
                // axes (1 here since indices are rank-1); dim2 = slice
                // size = product of preserved src axes (just hidden).
                grid: (N_INDICES, 1, HIDDEN),
            },
        )
        .expect("dispatch gather");
    }
    cmds.commit_and_wait().expect("commit");

    // SAFETY: commit_and_wait synchronised; GPU is done with `out`.
    let got: Vec<bf16> = unsafe { out.as_mut_slice() }.to_vec();
    assert_eq!(
        got, out_ref,
        "gather must be bit-exact vs mx.take(src, indices, axis=0)"
    );
}

#[test]
fn gather_axis0_u32_u32_rank1_matches_mlx_bit_exact() {
    // u32 gather is the packed-quantized-weight case (uint32 rows).
    let (src_host, indices_host, out_ref) = u32_fixture();
    assert_eq!(src_host.len(), U32_VOCAB * U32_HIDDEN);
    assert_eq!(out_ref.len(), U32_N * U32_HIDDEN);

    let device = Device::system_default().expect("Metal device");
    let inst = Instantiation::u32_u32_rank1();
    let source = make_source(&inst);
    let library = KernelLibrary::from_source(&device, &source).expect("compile JIT gather (u32)");

    let src: Buffer<u32> = device.upload(&src_host).expect("upload u32 src");
    let indices: Buffer<u32> = device.upload(&indices_host).expect("upload indices");
    let mut out: Buffer<u32> = device.alloc_buffer(U32_N * U32_HIDDEN).expect("alloc out");

    let src_shape: Buffer<i32> = device
        .upload(&[U32_VOCAB as i32, U32_HIDDEN as i32])
        .expect("src_shape");
    let src_strides: Buffer<i64> = device.upload(&[U32_HIDDEN as i64, 1]).expect("src_strides");
    let slice_sizes: Buffer<i32> = device.upload(&[1, U32_HIDDEN as i32]).expect("slice_sizes");
    let axes: Buffer<i32> = device.upload(&[0]).expect("axes");
    let idx_shapes: Buffer<i32> = device.upload(&[U32_N as i32]).expect("idx_shapes");
    let idx_strides: Buffer<i64> = device.upload(&[1]).expect("idx_strides");
    let idx_contigs: Buffer<u8> = device.upload(&[1u8]).expect("idx_contigs");

    let mut cmds = device.commands().expect("commands");
    {
        let idx_refs = [&indices];
        gather::dispatch(
            &mut cmds,
            &library,
            &inst,
            GatherDispatch {
                src: &src,
                out: &mut out,
                indices: &idx_refs,
                src_shape: &src_shape,
                src_strides: &src_strides,
                src_ndim: 2,
                slice_sizes: &slice_sizes,
                axes: &axes,
                idx_shapes: &idx_shapes,
                idx_strides: &idx_strides,
                idx_contigs: &idx_contigs,
                idx_ndim: 1,
                grid: (U32_N, 1, U32_HIDDEN),
            },
        )
        .expect("dispatch u32 gather");
    }
    cmds.commit_and_wait().expect("commit");

    // SAFETY: commit_and_wait synchronised; GPU is done with `out`.
    let got: Vec<u32> = unsafe { out.as_mut_slice() }.to_vec();
    assert_eq!(
        got, out_ref,
        "u32 gather must be bit-exact vs mx.take on a uint32 src"
    );
}

// ---------------------------------------------------------------------
// fixture / harness boilerplate (mirrors the other parity tests)
// ---------------------------------------------------------------------

fn fixture() -> (Vec<bf16>, Vec<u32>, Vec<bf16>) {
    let tmp = tempdir("kernels-gather-parity");
    let path = tmp.join("gather.safetensors");
    dump_mlx_gather(&path, VOCAB, HIDDEN, N_INDICES, "bf16");
    let bytes = std::fs::read(&path).expect("read fixture");
    let st = SafeTensors::deserialize(&bytes).expect("parse safetensors");
    (
        read_bf16(&st, "src"),
        read_u32(&st, "indices"),
        read_bf16(&st, "out"),
    )
}

fn u32_fixture() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let tmp = tempdir("kernels-gather-parity");
    let path = tmp.join("gather_u32.safetensors");
    dump_mlx_gather(&path, U32_VOCAB, U32_HIDDEN, U32_N, "u32");
    let bytes = std::fs::read(&path).expect("read u32 fixture");
    let st = SafeTensors::deserialize(&bytes).expect("parse safetensors");
    (
        read_u32(&st, "src"),
        read_u32(&st, "indices"),
        read_u32(&st, "out"),
    )
}

fn dump_mlx_gather(out: &Path, vocab: usize, hidden: usize, n: usize, dtype: &str) {
    let vocab_s = vocab.to_string();
    let hidden_s = hidden.to_string();
    let n_s = n.to_string();
    dump_mlx_op(
        out,
        &[
            "gather",
            "--vocab",
            &vocab_s,
            "--hidden",
            &hidden_s,
            "--n-indices",
            &n_s,
            "--dtype",
            dtype,
        ],
    );
}
