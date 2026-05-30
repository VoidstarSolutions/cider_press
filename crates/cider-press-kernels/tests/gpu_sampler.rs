//! Integration tests for the GPU counter-sampling facility. All are
//! gated on a live Metal device that supports stage-boundary counter
//! sampling; they SKIP (return early, not fail) when unsupported so a
//! counter-less CI runner stays green.
#![cfg(target_os = "macos")]

use cider_press_kernels::Device;

#[test]
fn device_reports_stage_boundary_sampling_support() {
    let device = Device::system_default().expect("Metal device");
    // On Apple Silicon this is always true; we only assert the call
    // works and the period is sane when supported.
    if !device.supports_stage_boundary_sampling() {
        eprintln!("SKIP: device lacks stage-boundary counter sampling");
        return;
    }
    let period = device.gpu_timestamp_period_ns();
    assert!(
        period > 0.0 && period.is_finite(),
        "gpu timestamp period must be a positive finite ns/tick, got {period}"
    );
}
