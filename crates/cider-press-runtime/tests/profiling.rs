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
    assert!(profile::drain().is_empty(), "drain leaves the profiler empty");
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
