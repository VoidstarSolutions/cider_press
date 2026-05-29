//! Parity test for `affine_dequantize_bf16_gs64_b4` against
//! `mx.dequantize`.
//!
//! Same Qwen2 q4/gs64 config as the qmv parity test. The kernel is a pure
//! deterministic function of `(w_q, scales, biases)` so the bar is
//! bit-exact.

#![cfg(target_os = "macos")]

use cider_press_kernels::kernels::dequantize::affine_dequantize_bf16_gs64_b4;
use cider_press_kernels::{Buffer, Device, KernelLibrary};
use cider_press_test_utils::{dump_mlx_op, read_bf16, read_u32, tempdir};
use half::bf16;
use safetensors::SafeTensors;

const ROWS: usize = 128;
const COLS: usize = 64; // multiple of group_size=64

#[test]
fn affine_dequantize_bf16_gs64_b4_matches_mlx_bit_exact() {
    let f = fixture();
    let device = Device::system_default().expect("Metal device");
    let library = KernelLibrary::quantized(&device).expect("compile quantized.metal");

    let w_q: Buffer<u32> = device.upload(&f.w_q).expect("upload w_q");
    let scales: Buffer<bf16> = device.upload(&f.scales).expect("upload scales");
    let biases: Buffer<bf16> = device.upload(&f.biases).expect("upload biases");
    let mut out: Buffer<bf16> = device.alloc_buffer(ROWS * COLS).expect("alloc out");

    let mut cmds = device.commands().expect("commands");
    affine_dequantize_bf16_gs64_b4(&mut cmds, &library, &w_q, &scales, &biases, &mut out)
        .expect("dispatch dequantize");
    cmds.commit_and_wait().expect("commit");

    // SAFETY: commit_and_wait synchronised; GPU is done with `out`.
    let got: Vec<bf16> = unsafe { out.as_mut_slice() }.to_vec();
    assert_eq!(
        got, f.out,
        "affine_dequantize must be bit-exact vs mx.dequantize"
    );
}

// ---------------------------------------------------------------------
// fixture / harness boilerplate
// ---------------------------------------------------------------------

struct Fixture {
    w_q: Vec<u32>,
    scales: Vec<bf16>,
    biases: Vec<bf16>,
    out: Vec<bf16>,
}

fn fixture() -> Fixture {
    let tmp = tempdir("kernels-dequantize-parity");
    let path = tmp.join("dequantize.safetensors");
    let rows_s = ROWS.to_string();
    let cols_s = COLS.to_string();
    dump_mlx_op(
        &path,
        &[
            "dequantize",
            "--rows",
            &rows_s,
            "--cols",
            &cols_s,
            "--group-size",
            "64",
            "--bits",
            "4",
            "--dtype",
            "bf16",
        ],
    );
    let bytes = std::fs::read(&path).expect("read fixture");
    let st = SafeTensors::deserialize(&bytes).expect("parse safetensors");
    Fixture {
        w_q: read_u32(&st, "w_q"),
        scales: read_bf16(&st, "scales"),
        biases: read_bf16(&st, "biases"),
        out: read_bf16(&st, "out"),
    }
}
