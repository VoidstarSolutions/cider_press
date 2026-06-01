//! Smoke test for the vendored MLX `arg_reduce.metal` library.
//!
//! Confirms the flattened MSL source compiles via
//! `newLibraryWithSource:`. Dispatch-level tests (argmax correctness,
//! tie-break locking) land in Task 2 alongside the dispatch wrapper.

#![cfg(target_os = "macos")]

use cider_press_kernels::{Device, KernelLibrary};

#[test]
fn arg_reduce_library_compiles() {
    let device = Device::system_default().expect("Metal device");
    KernelLibrary::arg_reduce(&device).expect("compile arg_reduce.metal");
}
