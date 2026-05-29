//! General performance-instrumentation facility.
//!
//! Drop a named span at any point you want to time:
//!
//! ```ignore
//! let _span = cider_press_runtime::profile::span("decode.argmax");
//! // ... work ...
//! // span records elapsed time on drop, keyed by name.
//! ```
//!
//! When the `profiling` cargo feature is **off** (the default), [`span`]
//! returns a zero-sized guard whose drop is empty, so instrumentation
//! compiles away to nothing — no branch, no timer, no allocation. When
//! the feature is **on**, each guard records its lifetime into a
//! thread-local accumulator keyed by the span name; totals and hit
//! counts are read with [`drain`] and cleared with [`reset`].
//!
//! Accumulation is thread-local rather than threaded through call
//! signatures because instrumented work (graph eval, KV-cache writes,
//! sampling) crosses crate boundaries deep into the model layer;
//! threading a collector would pollute every signature. The eval
//! boundary is synchronous and single-threaded, so thread-local is
//! sound for the cases instrumented here. A caller that wants spans
//! recorded on a worker thread reads [`drain`] on that same thread.

#[cfg(feature = "profiling")]
mod imp {
    use std::cell::RefCell;
    use std::time::{Duration, Instant};

    thread_local! {
        static PROFILER: RefCell<Vec<(&'static str, Duration, u64)>> =
            const { RefCell::new(Vec::new()) };
    }

    /// RAII timing guard; records elapsed time on drop under its name.
    #[must_use = "binding the span to `_` drops it immediately, timing nothing"]
    pub struct Span {
        name: &'static str,
        start: Instant,
    }

    impl Drop for Span {
        fn drop(&mut self) {
            let elapsed = self.start.elapsed();
            PROFILER.with(|p| {
                let mut v = p.borrow_mut();
                if let Some(entry) = v.iter_mut().find(|(n, _, _)| *n == self.name) {
                    entry.1 += elapsed;
                    entry.2 += 1;
                } else {
                    v.push((self.name, elapsed, 1));
                }
            });
        }
    }

    /// Start timing a named span. The returned guard records on drop.
    #[must_use = "binding the span to `_` drops it immediately, timing nothing"]
    pub fn span(name: &'static str) -> Span {
        Span {
            name,
            start: Instant::now(),
        }
    }

    /// Clear all accumulated spans on the current thread.
    pub fn reset() {
        PROFILER.with(|p| p.borrow_mut().clear());
    }

    /// Take and clear the accumulated spans, sorted by name:
    /// `(name, total_elapsed, hit_count)`.
    #[must_use]
    pub fn drain() -> Vec<(&'static str, Duration, u64)> {
        PROFILER.with(|p| {
            let mut out = std::mem::take(&mut *p.borrow_mut());
            out.sort_by_key(|e| e.0);
            out
        })
    }

    /// Whether the `profiling` feature was compiled in.
    #[must_use]
    pub const fn is_enabled() -> bool {
        true
    }
}

#[cfg(not(feature = "profiling"))]
mod imp {
    use std::time::Duration;

    /// Zero-sized no-op guard (feature off).
    #[must_use = "binding the span to `_` drops it immediately, timing nothing"]
    pub struct Span;

    #[inline]
    pub fn span(_name: &'static str) -> Span {
        Span
    }

    #[inline]
    pub fn reset() {}

    #[inline]
    #[must_use]
    pub fn drain() -> Vec<(&'static str, Duration, u64)> {
        Vec::new()
    }

    /// Whether the `profiling` feature was compiled in.
    #[inline]
    #[must_use]
    pub const fn is_enabled() -> bool {
        false
    }
}

pub use imp::{Span, drain, is_enabled, reset, span};
