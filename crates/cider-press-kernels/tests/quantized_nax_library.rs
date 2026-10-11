//! `quantized_nax.metal` (MLX's tensor-core quantized matmul family) compiles
//! under Metal 4.0 and exposes the bf16 / gs64 / 4-bit `qmm_t_nax` variants.
//! Skips on a machine without NAX: the source needs Metal 4.0 and MLX never
//! builds it there either.

use cider_press_kernels::{Device, KernelLibrary};

#[test]
fn quantized_nax_library_exposes_bf16_gs64_b4_qmm_t_variants() {
    let device = Device::system_default().expect("Metal device");
    if !device.nax_available() {
        eprintln!("skipping: NAX unavailable on this GPU/OS");
        return;
    }
    let library = KernelLibrary::quantized_nax(&device).expect("compile quantized_nax.metal");
    for name in [
        "affine_qmm_t_nax_bfloat16_t_gs_64_b_4_bm64_bn64_bk64_wm2_wn2_alN_true_batch_0",
        "affine_qmm_t_nax_bfloat16_t_gs_64_b_4_bm64_bn64_bk64_wm2_wn2_alN_false_batch_0",
        "affine_qmm_t_nax_bfloat16_t_gs_64_b_4_bm32_bn64_bk64_wm2_wn2_alN_true_batch_0",
        "affine_qmm_t_nax_bfloat16_t_gs_64_b_4_bm32_bn64_bk64_wm2_wn2_alN_false_batch_0",
    ] {
        library
            .pipeline(name)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
    }
}
