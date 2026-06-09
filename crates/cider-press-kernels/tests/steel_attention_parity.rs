//! Compile-smoke for the vendored fused `steel_attention` (`sdpa_full`
//! prefill) kernel: JIT the library and specialize the pinned causal
//! bf16 instantiation. macOS + live Metal device.
#![cfg(target_os = "macos")]

use cider_press_kernels::{Device, FunctionConstant, KernelLibrary};

#[test]
fn steel_attention_library_compiles() {
    let device = Device::system_default().expect("Metal device");
    let lib = KernelLibrary::steel_attention(&device).expect("compile steel_attention");
    // Every instantiation requires [[function_constant]] specialization
    // (200/201 align, 300 has_mask, 301 do_causal, 302 has_sinks); the
    // causal bf16 prefill set is the figure-out-the-shape canary. The
    // host name uses `bfloat16` (no `_t` suffix) because the
    // instantiate_attn macro stringifies a separate `tname` token.
    let constants = [
        FunctionConstant::Bool {
            index: 200,
            value: false,
        }, // align_Q
        FunctionConstant::Bool {
            index: 201,
            value: false,
        }, // align_K
        FunctionConstant::Bool {
            index: 300,
            value: false,
        }, // has_mask
        FunctionConstant::Bool {
            index: 301,
            value: true,
        }, // do_causal
        FunctionConstant::Bool {
            index: 302,
            value: false,
        }, // has_sinks
    ];
    lib.pipeline_specialized(
        "steel_attention_bfloat16_bq32_bk32_bd64_wm4_wn1_maskbfloat16",
        &constants,
    )
    .expect("steel_attention bf16 bd64 causal pipeline");
}
