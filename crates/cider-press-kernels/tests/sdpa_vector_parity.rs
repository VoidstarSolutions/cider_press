//! Parity tests for the vendored `sdpa_vector` kernel against MLX's
//! `mx.fast.scaled_dot_product_attention`. macOS + live Metal device.
#![cfg(target_os = "macos")]

use cider_press_kernels::{Device, FunctionConstant, KernelLibrary};

#[test]
fn sdpa_vector_library_compiles() {
    let device = Device::system_default().expect("Metal device");
    let lib = KernelLibrary::sdpa_vector(&device).expect("compile sdpa_vector");
    // The bf16 head-dim-64 single-pass kernel requires [[function_constant]]
    // specialization (indices 20-25 are bool; 26 is int for 2-pass only).
    // Bind the six bool slots to their simplest no-mask, no-causal values.
    let constants = [
        FunctionConstant::Bool {
            index: 20,
            value: false,
        }, // has_mask
        FunctionConstant::Bool {
            index: 21,
            value: false,
        }, // query_transposed
        FunctionConstant::Bool {
            index: 22,
            value: false,
        }, // do_causal
        FunctionConstant::Bool {
            index: 23,
            value: false,
        }, // bool_mask
        FunctionConstant::Bool {
            index: 24,
            value: false,
        }, // float_mask
        FunctionConstant::Bool {
            index: 25,
            value: false,
        }, // has_sinks
    ];
    lib.pipeline_specialized("sdpa_vector_bfloat16_t_64_64", &constants)
        .expect("sdpa_vector bf16 64/64 pipeline");
}
