//! Smoke test for the vendored MLX `unary.metal` library.
//!
//! Confirms the flattened MSL source compiles via
//! `newLibraryWithSource:`, and that representative pipeline lookups
//! per op family (`Square`, `Rsqrt` — the two unary ops the rmsnorm
//! composition needs) for bf16 resolve. Bit-exact parity for these
//! kernels lands in `unary_parity.rs` once the dispatch wrappers
//! arrive in commit 3 of this branch.
//!
//! Naming convention (from `instantiate_unary_*` in
//! `kernels-mlx/.../unary.metal`): host names concatenate the op
//! token with `in_tname` + `out_tname` — for same-dtype float ops
//! like Square/Rsqrt, that's e.g. `v_Squarebfloat16bfloat16` for the
//! `N = 1` contiguous instantiation.

#![cfg(target_os = "macos")]

use cider_press_kernels::{Device, KernelLibrary};

#[test]
fn unary_library_compiles() {
    let device = Device::system_default().expect("Metal device");
    KernelLibrary::unary(&device).expect("compile unary.metal");
}

#[test]
fn unary_bf16_pipelines_resolve() {
    let device = Device::system_default().expect("Metal device");
    let library = KernelLibrary::unary(&device).expect("compile unary.metal");

    // Contiguous fast path: this is the kernel branch 5's
    // `Tensor::square`/`Tensor::rsqrt` will dispatch when the input is
    // dense and contiguous (which the rmsnorm composition always is).
    library
        .pipeline("v_Squarebfloat16bfloat16")
        .expect("v_Squarebfloat16bfloat16 pipeline");
    library
        .pipeline("v_Rsqrtbfloat16bfloat16")
        .expect("v_Rsqrtbfloat16bfloat16 pipeline");

    // Strided generic: representative of the view path. Once branch
    // 5's runtime wiring lands we'll exercise this with a parity test
    // against a non-contiguous (e.g. transposed) input.
    library
        .pipeline("gn1_Squarebfloat16bfloat16")
        .expect("gn1_Squarebfloat16bfloat16 pipeline");
}
