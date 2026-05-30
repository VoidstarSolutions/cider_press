//! Unit coverage for the `profile` facility. Gated on the `profiling`
//! feature: with the feature off, `drain` always returns empty (the
//! whole point), so the assertions only make sense with it on. Run:
//! `cargo test -p cider-press-runtime --features profiling --test profiling`.
#![cfg(feature = "profiling")]

use cider_press_runtime::profile;

#[test]
fn accumulates_named_spans_with_hit_counts() {
    profile::reset();
    {
        let _a = profile::span("alpha");
        let _b = profile::span("beta");
    }
    {
        let _a = profile::span("alpha");
    }

    let stats = profile::drain();
    let alpha = stats
        .iter()
        .find(|(n, _, _)| *n == "alpha")
        .expect("alpha recorded");
    let beta = stats
        .iter()
        .find(|(n, _, _)| *n == "beta")
        .expect("beta recorded");
    assert_eq!(alpha.2, 2, "alpha hit twice");
    assert_eq!(beta.2, 1, "beta hit once");

    // drain cleared the accumulator.
    assert!(
        profile::drain().is_empty(),
        "drain leaves the profiler empty"
    );
}

#[test]
fn drain_is_sorted_by_name() {
    profile::reset();
    let z = profile::span("zeta");
    drop(z);
    let a = profile::span("alpha");
    drop(a);

    let stats = profile::drain();
    let names: Vec<&str> = stats.iter().map(|(n, _, _)| *n).collect();
    assert_eq!(names, ["alpha", "zeta"]);
}

/// A single `eval()` emits the encode/wait split, and the two
/// sub-spans nest within the outer `tensor.eval` total (they time
/// disjoint sub-intervals of the same call, so their sum cannot exceed
/// it). Guards against the spans drifting out of `eval` or being
/// double-counted. Needs a Metal device, so macOS-only.
#[cfg(target_os = "macos")]
#[test]
fn eval_emits_encode_and_wait_spans() {
    use cider_press_runtime::{Device, Tensor};
    use half::bf16;

    profile::reset();

    let device = Device::system_default().expect("Metal device");
    let a = Tensor::from_slice(&device, &[bf16::ONE; 4], [1, 4]).expect("a");
    let b = Tensor::from_slice(&device, &[bf16::ONE; 4], [1, 4]).expect("b");
    a.add(&b).expect("schedule add").eval().expect("eval");

    let stats = profile::drain();
    let total = |name: &str| {
        stats
            .iter()
            .find(|(n, _, _)| *n == name)
            .unwrap_or_else(|| panic!("{name} span recorded"))
    };
    let eval = total("tensor.eval");
    let encode = total("tensor.eval.encode");
    let wait = total("tensor.eval.wait");

    assert_eq!(eval.2, 1, "one eval call");
    assert_eq!(encode.2, 1, "one encode span per eval");
    assert_eq!(wait.2, 1, "one wait span per eval");
    assert!(
        encode.1 + wait.1 <= eval.1,
        "encode ({:?}) + wait ({:?}) must nest within tensor.eval ({:?})",
        encode.1,
        wait.1,
        eval.1,
    );
}

#[test]
fn gpu_accumulator_sums_by_label_with_counts() {
    profile::reset();
    profile::record_gpu("gpu.qmv", 1000);
    profile::record_gpu("gpu.qmv", 500);
    profile::record_gpu("gpu.copy", 200);

    let stats = profile::drain_gpu();
    let qmv = stats.iter().find(|(n, _, _)| *n == "gpu.qmv").expect("qmv");
    let copy = stats
        .iter()
        .find(|(n, _, _)| *n == "gpu.copy")
        .expect("copy");
    assert_eq!(qmv.1, 1500, "summed ns");
    assert_eq!(qmv.2, 2, "two hits");
    assert_eq!(copy.1, 200);
    assert_eq!(copy.2, 1);
    // drain clears
    assert!(profile::drain_gpu().is_empty());
}
