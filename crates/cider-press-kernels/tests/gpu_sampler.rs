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
        (0.1..=10_000.0).contains(&period),
        "gpu timestamp period ns/tick out of sane range, got {period}"
    );
}

use cider_press_kernels::GpuSampler;

#[test]
fn sampler_allocates_for_capacity_and_starts_empty() {
    let device = Device::system_default().expect("Metal device");
    if !device.supports_stage_boundary_sampling() {
        eprintln!("SKIP: no stage-boundary sampling");
        return;
    }
    // Capacity is in *ops*; each op consumes a start+end pair.
    let sampler = GpuSampler::new(&device, 4).expect("sampler for 4 ops");
    assert_eq!(sampler.recorded_segments().len(), 0, "no segments yet");
    assert_eq!(sampler.sample_capacity(), 8, "4 ops => 8 timestamps");
}
