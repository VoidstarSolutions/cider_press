//! Smoke test for the vendored MLX `reduce.metal` library.
//!
//! Confirms the flattened MSL source compiles via
//! `newLibraryWithSource:`, and that a representative pipeline lookup
//! per op family resolves for the bf16 dtype combinations the rmsnorm
//! and softmax compositions will need.
//!
//! Naming convention (from `instantiate_*_reduce*` in
//! `kernels-mlx/.../reduce.metal`): host names are built by
//! token-pasting the op name (e.g. `sum`) with the input tname (e.g.
//! `bfloat16`) — no separator. So `Sum` over bf16 yields the suffix
//! `sumbfloat16`, prefixed by which reduction template
//! (`all_reduce_`, `row_reduce_simple_`, `init_reduce_`, etc.).
//!
//! Bit-exact parity for these kernels lands in `reduce_parity.rs`
//! once the dispatch wrappers arrive in commit 4 of this branch. The
//! pipelines exercised here cover all four op families (Sum / Prod /
//! Max / Min) since branch 5 wires the full reduce surface up front.

#![cfg(target_os = "macos")]

use cider_press_kernels::{Device, KernelLibrary};

#[test]
fn reduce_library_compiles() {
    let device = Device::system_default().expect("Metal device");
    KernelLibrary::reduce(&device).expect("compile reduce.metal");
}

#[test]
fn reduce_bf16_pipelines_resolve() {
    let device = Device::system_default().expect("Metal device");
    let library = KernelLibrary::reduce(&device).expect("compile reduce.metal");

    // Initializer kernels — every dispatch starts by seeding the
    // output buffer with the op's identity value (0 for sum, 1 for
    // prod, +inf for min, -inf for max).
    for op in ["sum", "prod", "min", "max"] {
        let name = format!("init_reduce_{op}bfloat16");
        library
            .pipeline(&name)
            .unwrap_or_else(|err| panic!("{name} pipeline: {err}"));
    }

    // `all_reduce_*` is the full-tensor reduction kernel — collapses
    // every element to a single output. RMSNorm doesn't use it
    // directly (it wants per-row sums), but softmax-temperature and
    // norm scalars will.
    for op in ["sum", "prod", "min", "max"] {
        let name = format!("all_reduce_{op}bfloat16");
        library
            .pipeline(&name)
            .unwrap_or_else(|err| panic!("{name} pipeline: {err}"));
    }

    // `row_reduce_simple_*` compiles in MLX's reduce.metal but isn't
    // wired by the current Rust dispatcher — `encode_row_reduce`
    // always uses `row_reduce_looped_*` regardless of row size. The
    // pipeline-resolution check here just guarantees the simple
    // variants stay buildable for when a future dispatcher selects
    // them on rows that fit a single threadgroup.
    for op in ["sum", "prod", "min", "max"] {
        let name = format!("row_reduce_simple_{op}bfloat16");
        library
            .pipeline(&name)
            .unwrap_or_else(|err| panic!("{name} pipeline: {err}"));
    }
}
