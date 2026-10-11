//! The JIT compile options must mirror MLX's: fast math OFF (MLX passes
//! `-fno-fast-math` to the ahead-of-time `mlx.metallib` build in
//! `kernels/CMakeLists.txt`, and `setFastMathEnabled(false)` on its JIT path in
//! `device.cpp`) and the highest Metal language version the OS supports. The
//! vendored kernels are declared bit-exact against MLX fixtures, which only
//! holds under the same policy.
//!
//! On macOS 15+ the deprecated `fastMathEnabled` flag is implemented in terms
//! of the newer knobs, and it sets BOTH: `mathMode = Safe` and
//! `mathFloatingPointFunctions = Precise` (whereas `setMathMode(Safe)` alone
//! leaves the FP functions Fast). The assertions pin the effective knobs, not
//! the flag, so a "modernising" refactor cannot silently weaken the policy.

use cider_press_kernels::KernelLibrary;
use objc2_metal::{MTLLanguageVersion, MTLMathFloatingPointFunctions, MTLMathMode};

#[test]
fn compile_options_select_safe_math_and_precise_fp_functions() {
    let options = KernelLibrary::compile_options();
    assert_eq!(options.mathMode(), MTLMathMode::Safe);
    assert_eq!(
        options.mathFloatingPointFunctions(),
        MTLMathFloatingPointFunctions::Precise
    );
}

#[test]
fn compile_options_language_version_is_within_mlx_range() {
    let version = KernelLibrary::compile_options().languageVersion();
    assert!(
        version >= MTLLanguageVersion::Version3_1 && version <= MTLLanguageVersion::Version4_0,
        "MLX selects 3.1, 3.2 or 4.0; got {version:?}"
    );
    assert_eq!(version, KernelLibrary::metal_language_version());
}
