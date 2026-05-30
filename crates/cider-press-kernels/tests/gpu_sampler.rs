//! Integration tests for the GPU counter-sampling facility. All are
//! gated on a live Metal device that supports stage-boundary counter
//! sampling; they SKIP (return early, not fail) when unsupported so a
//! counter-less CI runner stays green.
#![cfg(target_os = "macos")]

use cider_press_kernels::{Device, GpuSampler, KernelLibrary, kernels::copy};

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

#[test]
fn profiled_commands_time_each_copy_dispatch() {
    let device = Device::system_default().expect("Metal device");
    if !device.supports_stage_boundary_sampling() {
        eprintln!("SKIP: no stage-boundary sampling");
        return;
    }
    // Build the copy library the way every kernels-crate test does
    // (see tests/copy_kernel.rs). NOT device.copy_library() — that's on
    // the runtime Device wrapper, not this crate's Device.
    let library = KernelLibrary::copy(&device).expect("compile copy.metal");

    let src = device.upload(&[1.0f32; 256]).expect("src");
    let mut dst_a = device.alloc_buffer::<f32>(256).expect("dst a");
    let mut dst_b = device.alloc_buffer::<f32>(256).expect("dst b");

    let mut commands = device.commands_profiled(2).expect("profiled commands");

    commands.begin_profiled_op("copy");
    copy::copy_v_f32(&mut commands, &library, &src, &mut dst_a).expect("copy a");
    commands.begin_profiled_op("copy");
    copy::copy_v_f32(&mut commands, &library, &src, &mut dst_b).expect("copy b");

    let segments = commands.commit_wait_resolve().expect("resolve");
    assert_eq!(segments.len(), 2, "two profiled ops => two segments");
    for seg in &segments {
        assert_eq!(seg.label, "copy");
        assert!(seg.end_tick >= seg.start_tick, "monotonic ticks: {seg:?}");
    }
}
