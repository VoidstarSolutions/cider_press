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

    thread_local! {
        static GPU_PROFILER: RefCell<Vec<(&'static str, u64, u64)>> =
            const { RefCell::new(Vec::new()) };
    }

    /// Record `ns` of GPU time under `name`, accumulating with hit count.
    /// Separate from the wall-clock span store: these are resolved GPU
    /// counter intervals, not RAII span lifetimes.
    pub fn record_gpu(name: &'static str, ns: u64) {
        GPU_PROFILER.with(|p| {
            let mut v = p.borrow_mut();
            if let Some(entry) = v.iter_mut().find(|(n, _, _)| *n == name) {
                entry.1 += ns;
                entry.2 += 1;
            } else {
                v.push((name, ns, 1));
            }
        });
    }

    /// Take and clear accumulated GPU segments, sorted by name:
    /// `(name, total_ns, hit_count)`.
    #[must_use]
    pub fn drain_gpu() -> Vec<(&'static str, u64, u64)> {
        GPU_PROFILER.with(|p| {
            let mut out = std::mem::take(&mut *p.borrow_mut());
            out.sort_by_key(|e| e.0);
            out
        })
    }

    /// Clear all accumulated spans on the current thread.
    pub fn reset() {
        PROFILER.with(|p| p.borrow_mut().clear());
        GPU_PROFILER.with(|p| p.borrow_mut().clear());
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

    #[inline]
    pub fn record_gpu(_name: &'static str, _ns: u64) {}

    #[inline]
    #[must_use]
    pub fn drain_gpu() -> Vec<(&'static str, u64, u64)> {
        Vec::new()
    }

    /// Whether the `profiling` feature was compiled in.
    #[inline]
    #[must_use]
    pub const fn is_enabled() -> bool {
        false
    }
}

/// `os_signpost` interval markers for Instruments (Points of Interest /
/// Metal System Trace). No-op unless the `profiling` feature is on.
pub mod signpost {
    /// The eval phase a marker brackets. The discriminant is the
    /// `os_signpost_id_t` — distinct per region, stable across begin/end.
    #[cfg_attr(not(feature = "profiling"), allow(dead_code))]
    #[derive(Clone, Copy)]
    pub enum Region {
        Encode = 1,
        Wait = 2,
    }

    /// RAII guard returned by [`interval_begin`]; its drop emits the matching
    /// `END` marker, so a `?`/panic unwind still closes the interval balanced.
    /// Drop it explicitly at a phase boundary to end the interval early.
    #[must_use]
    pub struct Marker(#[cfg_attr(not(feature = "profiling"), allow(dead_code))] Region);

    impl Drop for Marker {
        fn drop(&mut self) {
            #[cfg(feature = "profiling")]
            {
                const INTERVAL_END: u8 = 2;
                ffi::emit(INTERVAL_END, self.0 as u64, name_of(self.0).as_ptr());
            }
        }
    }

    #[cfg(feature = "profiling")]
    mod ffi {
        use std::ffi::c_char;
        use std::os::raw::c_void;
        use std::sync::OnceLock;

        #[repr(transparent)]
        struct OsLog(*mut c_void);
        // SAFETY: os_log_t is an immutable, thread-safe handle.
        unsafe impl Send for OsLog {}
        unsafe impl Sync for OsLog {}

        // `os_signpost_interval_begin/end` are header-only macros; the real C
        // entry point they expand to is `_os_signpost_emit_with_name_impl`
        // (asm symbol `__os_signpost_emit_with_name_impl`, in libsystem_trace
        // via the dyld shared cache). `_os_signpost_emit_with_type` named in
        // the task spec is itself a macro, not a linkable symbol — it resolves
        // neither statically (no SDK .tbd export) nor at runtime (dlsym NULL).
        // The impl requires a non-null `dso` (image handle, for Instruments to
        // resolve the binary) and a non-null format buffer — passing NULL for
        // either traps (SIGTRAP). We pass `&__dso_handle` and a zeroed buffer.
        unsafe extern "C" {
            fn os_log_create(subsystem: *const c_char, category: *const c_char) -> *mut c_void;
            fn _os_signpost_emit_with_name_impl(
                dso: *mut c_void,
                log: *mut c_void,
                typ: u8,
                spid: u64,
                name: *const c_char,
                format: *const c_char,
                buf: *mut u8,
                size: u32,
            );
            #[link_name = "__dso_handle"]
            static DSO_HANDLE: c_void;
        }

        fn log() -> *mut c_void {
            static LOG: OnceLock<OsLog> = OnceLock::new();
            LOG.get_or_init(|| {
                let subsystem = c"com.cider-press.prefill";
                let category = c"PointsOfInterest";
                // SAFETY: both pointers are valid nul-terminated statics.
                OsLog(unsafe { os_log_create(subsystem.as_ptr(), category.as_ptr()) })
            })
            .0
        }

        /// Emit one signpost interval marker (begin or end). `typ` is the
        /// `os_signpost_type_t` (begin = 1, end = 2).
        pub(super) fn emit(typ: u8, spid: u64, name: *const c_char) {
            const BUF_LEN: u32 = 16;
            let mut buf = [0u8; BUF_LEN as usize];
            // SAFETY: `log()` returns a valid os_log_t; `name`/format are
            // valid nul-terminated statics; `DSO_HANDLE` is this image's
            // header; `buf` is a valid writable region matching `size`.
            unsafe {
                _os_signpost_emit_with_name_impl(
                    std::ptr::addr_of!(DSO_HANDLE).cast_mut(),
                    log(),
                    typ,
                    spid,
                    name,
                    c"".as_ptr(),
                    buf.as_mut_ptr(),
                    BUF_LEN,
                );
            }
        }
    }

    #[cfg(feature = "profiling")]
    fn name_of(r: Region) -> &'static std::ffi::CStr {
        match r {
            Region::Encode => c"eval.encode",
            Region::Wait => c"eval.wait",
        }
    }

    /// Open an interval for `region`, returning an RAII [`Marker`] whose drop
    /// emits the matching end. Drop it at the phase boundary to end early.
    pub fn interval_begin(region: Region) -> Marker {
        #[cfg(feature = "profiling")]
        {
            const INTERVAL_BEGIN: u8 = 1;
            ffi::emit(INTERVAL_BEGIN, region as u64, name_of(region).as_ptr());
        }
        Marker(region)
    }
}

pub use imp::{Span, drain, drain_gpu, is_enabled, record_gpu, reset, span};

#[cfg(test)]
mod tests {
    #[test]
    fn signpost_emit_does_not_panic() {
        // No-op when profiling is off; FFI emit (begin + drop-end) when on.
        // Either way: no panic.
        let s = super::signpost::interval_begin(super::signpost::Region::Encode);
        drop(s);
    }
}
