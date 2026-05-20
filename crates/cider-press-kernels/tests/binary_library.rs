//! Smoke test for the vendored MLX `binary.metal` library.
//!
//! Confirms the flattened MSL source compiles via
//! `newLibraryWithSource:`, and that a representative pipeline lookup
//! per op family (`Add`, `Multiply`) for bf16 resolves — i.e. the
//! `instantiate_binary_*` macros produced the host names we expect to
//! dispatch from the kernels crate. Bit-exact / parity checks live in
//! `binary_parity.rs` once the dispatch wrappers land in commit 2.

#![cfg(target_os = "macos")]

use cider_press_kernels::{Device, KernelLibrary};

#[test]
fn binary_library_compiles() {
    let device = Device::system_default().expect("Metal device");
    KernelLibrary::binary(&device).expect("compile binary.metal");
}

#[test]
fn binary_bf16_pipelines_resolve() {
    let device = Device::system_default().expect("Metal device");
    let library = KernelLibrary::binary(&device).expect("compile binary.metal");

    // Contiguous vector-vector fast path: this is the kernel commit 3
    // will dispatch when both operands are dense and same-shape.
    library
        .pipeline("vv_Addbfloat16")
        .expect("vv_Addbfloat16 pipeline");
    library
        .pipeline("vv_Multiplybfloat16")
        .expect("vv_Multiplybfloat16 pipeline");

    // Strided rank-2: representative of the broadcast path. The
    // `g{1,2,3}_*` family is what commit 3 picks when one or both
    // inputs are non-contiguous views (e.g. a broadcast bias).
    library
        .pipeline("g2_Addbfloat16")
        .expect("g2_Addbfloat16 pipeline");
}
