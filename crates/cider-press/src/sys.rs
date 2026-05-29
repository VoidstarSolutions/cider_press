//! In-process resident-memory sampling via mach `task_info`.
//!
//! Queries `MACH_TASK_BASIC_INFO` on the current task. `resident_size`
//! is the current RSS; `resident_size_max` is the process-lifetime peak
//! RSS (a high-water mark that never decreases). Both are in bytes.
//!
//! The mach symbols live in libSystem, which is always linked on macOS,
//! so no `#[link]` directive is needed. `MACH_TASK_BASIC_INFO` (flavor
//! 20) maps to a stable, fixed-layout struct — preferred here over
//! `TASK_VM_INFO`'s `phys_footprint`, whose field offset is brittle.

use std::mem;

const MACH_TASK_BASIC_INFO: u32 = 20;

/// Mirrors `mach_task_basic_info` (`<mach/task_info.h>`). `mach_vm_size_t`
/// is `u64`; `time_value_t` is two `integer_t` (`i32`); `policy_t` and
/// `integer_t` are `i32`. Size is 48 bytes = 12 `natural_t` (`u32`) words,
/// which is `MACH_TASK_BASIC_INFO_COUNT`.
///
/// All fields must be present even if not read — the struct layout must match
/// the C ABI exactly so `task_info` writes into the correct offsets.
#[repr(C)]
#[allow(dead_code)]
struct MachTaskBasicInfo {
    virtual_size: u64,
    resident_size: u64,
    resident_size_max: u64,
    user_time: [i32; 2],
    system_time: [i32; 2],
    policy: i32,
    suspend_count: i32,
}

#[allow(non_upper_case_globals)]
unsafe extern "C" {
    // The `mach_task_self()` C macro expands to this global mach_port_t.
    static mach_task_self_: u32;
    fn task_info(
        target_task: u32,
        flavor: u32,
        task_info_out: *mut i32,
        task_info_out_cnt: *mut u32,
    ) -> i32;
}

fn query() -> Option<MachTaskBasicInfo> {
    let mut info = mem::MaybeUninit::<MachTaskBasicInfo>::uninit();
    #[allow(clippy::cast_possible_truncation)]
    let mut count = (mem::size_of::<MachTaskBasicInfo>() / mem::size_of::<u32>()) as u32;
    // SAFETY: task_info writes exactly `count` natural_t words into the
    // buffer; `count` is sized from the struct, and the struct layout
    // matches MACH_TASK_BASIC_INFO. On KERN_SUCCESS (0) the struct is
    // fully initialized.
    let kr = unsafe {
        task_info(
            mach_task_self_,
            MACH_TASK_BASIC_INFO,
            info.as_mut_ptr().cast::<i32>(),
            &raw mut count,
        )
    };
    if kr == 0 {
        // SAFETY: KERN_SUCCESS means task_info initialized the struct.
        Some(unsafe { info.assume_init() })
    } else {
        None
    }
}

/// Current resident set size in bytes, or `None` if the mach query fails.
#[must_use]
pub fn resident_bytes() -> Option<u64> {
    query().map(|i| i.resident_size)
}

/// Process-lifetime peak resident set size in bytes, or `None` on failure.
#[must_use]
pub fn peak_resident_bytes() -> Option<u64> {
    query().map(|i| i.resident_size_max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn samples_nonzero_and_peak_at_least_current() {
        let current = resident_bytes().expect("resident_bytes query");
        let peak = peak_resident_bytes().expect("peak_resident_bytes query");
        assert!(current > 0, "current RSS should be positive");
        assert!(peak >= current, "peak RSS should be >= current RSS");
    }
}
