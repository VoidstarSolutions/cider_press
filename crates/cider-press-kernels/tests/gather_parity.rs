//! Parity test for the JIT-assembled `gather` dispatch (branch 7).
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

use std::path::{Path, PathBuf};
use std::process::Command;

use cider_press_kernels::kernels::gather::{
    self, GatherDispatch, Instantiation, make_source,
};
use cider_press_kernels::{Buffer, Device, KernelLibrary};
use half::bf16;
use safetensors::SafeTensors;

const VOCAB: usize = 128;
const HIDDEN: usize = 64;
const N_INDICES: usize = 8;

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
    let src_strides: Buffer<i64> = device
        .upload(&[HIDDEN as i64, 1])
        .expect("src_strides");
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

// ---------------------------------------------------------------------
// fixture / harness boilerplate (mirrors the other parity tests)
// ---------------------------------------------------------------------

fn fixture() -> (Vec<bf16>, Vec<u32>, Vec<bf16>) {
    let tmp = tempdir();
    let path = tmp.join("gather.safetensors");
    dump_mlx(&path);
    let bytes = std::fs::read(&path).expect("read fixture");
    let st = SafeTensors::deserialize(&bytes).expect("parse safetensors");
    (read_bf16(&st, "src"), read_u32(&st, "indices"), read_bf16(&st, "out"))
}

fn dump_mlx(out: &Path) {
    let script = workspace_root().join("scripts").join("dump_mlx_op.py");
    let status = Command::new("uv")
        .arg("run")
        .arg(&script)
        .arg("--output")
        .arg(out)
        .arg("--seed")
        .arg("0")
        .arg("gather")
        .arg("--vocab")
        .arg(VOCAB.to_string())
        .arg("--hidden")
        .arg(HIDDEN.to_string())
        .arg("--n-indices")
        .arg(N_INDICES.to_string())
        .arg("--dtype")
        .arg("bf16")
        .status();
    let status = match status {
        Ok(s) => s,
        Err(err) => panic!(
            "failed to invoke `uv run {}`: {err}. This test requires `uv` on PATH; \
             CI installs it via astral-sh/setup-uv.",
            script.display()
        ),
    };
    assert!(status.success(), "dump_mlx_op.py gather exited {status}");
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("workspace root")
        .to_path_buf()
}

fn read_bf16(st: &SafeTensors, name: &str) -> Vec<bf16> {
    let view = st
        .tensor(name)
        .unwrap_or_else(|e| panic!("tensor {name}: {e}"));
    assert_eq!(view.dtype(), safetensors::Dtype::BF16);
    let bytes = view.data();
    assert!(bytes.len() % 2 == 0);
    bytes
        .chunks_exact(2)
        .map(|c| bf16::from_le_bytes([c[0], c[1]]))
        .collect()
}

fn read_u32(st: &SafeTensors, name: &str) -> Vec<u32> {
    let view = st
        .tensor(name)
        .unwrap_or_else(|e| panic!("tensor {name}: {e}"));
    assert_eq!(view.dtype(), safetensors::Dtype::U32);
    let bytes = view.data();
    assert!(bytes.len() % 4 == 0);
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn tempdir() -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("SystemTime")
        .as_nanos();
    let pid = std::process::id();
    let dir = std::env::temp_dir().join(format!("cider-press-gather-parity-{pid}-{nanos}"));
    std::fs::create_dir_all(&dir).expect("mktemp");
    dir
}
