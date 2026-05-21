//! Runtime-level parity test for `Tensor::gather` against MLX's
//! `mx.take(src, indices, axis=0)`.
//!
//! The kernels-crate test (`kernels-crate::tests::gather_parity`) has
//! already validated the JIT-assembled gather dispatch bit-exactly;
//! this test validates that the runtime threads inputs + shape
//! metadata correctly into that dispatch — including the per-call
//! metadata buffers (`src_shape`, `slice_sizes`, etc.) which the
//! runtime synthesizes rather than the caller.

#![cfg(target_os = "macos")]
#![allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]

use cider_press_runtime::{Device, Tensor};
use cider_press_test_utils::{dump_mlx_op, read_bf16, read_u32, tempdir};
use half::bf16;
use safetensors::SafeTensors;

const VOCAB: usize = 128;
const HIDDEN: usize = 64;
const N_INDICES: usize = 8;

#[test]
fn gather_through_runtime_matches_mlx_bit_exact() {
    let (src_host, indices_host, out_ref) = fixture();
    let device = Device::system_default().expect("device");

    let src = Tensor::from_slice(&device, &src_host, [VOCAB, HIDDEN]).expect("src");
    let indices = Tensor::from_slice(&device, &indices_host, [N_INDICES]).expect("indices");

    let out = src.gather(&indices).expect("schedule gather");
    out.eval().expect("eval");
    assert_eq!(out.shape().dims(), &[N_INDICES, HIDDEN]);
    let got: Vec<bf16> = out.cpu_to_vec().expect("dense out");
    assert_eq!(
        got, out_ref,
        "Tensor::gather must match mx.take(src, indices, axis=0) bit-exactly",
    );
}

// ---------------------------------------------------------------------
// helpers (mirror the kernels-crate parity test exactly)
// ---------------------------------------------------------------------

fn fixture() -> (Vec<bf16>, Vec<u32>, Vec<bf16>) {
    let tmp = tempdir("runtime-gather-parity");
    let path = tmp.join("gather.safetensors");
    let vocab_s = VOCAB.to_string();
    let hidden_s = HIDDEN.to_string();
    let n_s = N_INDICES.to_string();
    dump_mlx_op(
        &path,
        &[
            "gather",
            "--vocab",
            &vocab_s,
            "--hidden",
            &hidden_s,
            "--n-indices",
            &n_s,
            "--dtype",
            "bf16",
        ],
    );
    let bytes = std::fs::read(&path).expect("read fixture");
    let st = SafeTensors::deserialize(&bytes).expect("parse safetensors");
    (
        read_bf16(&st, "src"),
        read_u32(&st, "indices"),
        read_bf16(&st, "out"),
    )
}
