//! The JIT compile options must mirror MLX's (`mlx/backend/metal/device.cpp`,
//! `Device::build_library_`): fast math OFF and the highest Metal language
//! version the running OS supports. The vendored kernels are declared
//! bit-exact against MLX fixtures, which only holds under the same policy.

#![cfg(target_os = "macos")]

use cider_press_kernels::KernelLibrary;
use objc2_foundation::{NSOperatingSystemVersion, NSProcessInfo};
use objc2_metal::MTLLanguageVersion;

fn os_at_least(major: isize) -> bool {
    NSProcessInfo::processInfo().isOperatingSystemAtLeastVersion(NSOperatingSystemVersion {
        majorVersion: major,
        minorVersion: 0,
        patchVersion: 0,
    })
}

#[test]
fn compile_options_disable_fast_math() {
    let options = KernelLibrary::compile_options();
    #[allow(deprecated)]
    let fast_math = options.fastMathEnabled();
    assert!(!fast_math, "MLX compiles with fastMathEnabled = false");
}

#[test]
fn compile_options_pick_highest_language_version_for_this_os() {
    let expected = if os_at_least(26) {
        MTLLanguageVersion::Version4_0
    } else if os_at_least(15) {
        MTLLanguageVersion::Version3_2
    } else {
        MTLLanguageVersion::Version3_1
    };
    assert_eq!(KernelLibrary::compile_options().languageVersion(), expected);
}

#[test]
fn every_vendored_library_compiles_under_the_mlx_options() {
    let device = cider_press_kernels::Device::system_default().expect("device");
    for (name, build) in [
        ("copy", KernelLibrary::copy as fn(&_) -> _),
        ("binary", KernelLibrary::binary),
        ("unary", KernelLibrary::unary),
        ("reduce", KernelLibrary::reduce),
        ("arg_reduce", KernelLibrary::arg_reduce),
        ("rope", KernelLibrary::rope),
        ("rms_norm", KernelLibrary::rms_norm),
        ("softmax", KernelLibrary::softmax),
        ("quantized", KernelLibrary::quantized),
        ("sdpa_vector", KernelLibrary::sdpa_vector),
        ("steel_attention", KernelLibrary::steel_attention),
        ("matmul", KernelLibrary::matmul),
    ] {
        build(&device).unwrap_or_else(|e| panic!("{name}: {e}"));
    }
}
