//! Parity test for `Tensor::slice_update` against MLX's in-place
//! `out[offset:offset+rows] = src`. `SliceUpdate` is a pure data mover
//! (a region copy via the vendored `copy_g` kernel), so the result is
//! bit-exact: the slab region is overwritten by src, the rest unchanged.

#![cfg(target_os = "macos")]

use cider_press_runtime::{Device, Tensor};
use cider_press_test_utils::{dump_mlx_op, read_bf16, tempdir};
use half::bf16;
use safetensors::SafeTensors;

fn run_parity(slab_rows: usize, src_rows: usize, offset: usize) {
    let tmp = tempdir("slice-update-parity");
    let fixture = tmp.join("slice_update_bf16.safetensors");
    let slab_shape = format!("{slab_rows},2,3");
    let src_rows_str = src_rows.to_string();
    let offset_str = offset.to_string();
    dump_mlx_op(
        &fixture,
        &[
            "slice_update",
            "--slab-shape",
            &slab_shape,
            "--src-rows",
            &src_rows_str,
            "--offset",
            &offset_str,
            "--dtype",
            "bf16",
        ],
    );

    let bytes = std::fs::read(&fixture).expect("read fixture");
    let st = SafeTensors::deserialize(&bytes).expect("parse safetensors");
    let slab_vals = read_bf16(&st, "slab");
    let src_vals = read_bf16(&st, "src");
    let out_ref = read_bf16(&st, "out");

    let device = Device::shared().expect("device");
    let slab = Tensor::from_slice(&device, &slab_vals, [slab_rows, 2, 3]).expect("slab");
    let src = Tensor::from_slice(&device, &src_vals, [src_rows, 2, 3]).expect("src");

    let updated = slab.slice_update(&src, offset).expect("slice_update");
    updated.eval().expect("eval");
    let out: Vec<bf16> = updated.cpu_to_vec().expect("out");

    assert_eq!(
        out, out_ref,
        "slice_update (slab_rows={slab_rows}, src_rows={src_rows}, offset={offset}) \
         must match MLX in-place assignment"
    );
}

#[test]
fn parity_slice_update_bf16_mid_offset() {
    // Non-zero offset: non-empty prefix (rows 0-2), written window (3-4),
    // non-empty suffix (row 5).
    run_parity(6, 2, 3);
}

#[test]
fn parity_slice_update_bf16_offset_zero() {
    // Head write: no prefix — catches an off-by-one in the start index.
    run_parity(6, 2, 0);
}
