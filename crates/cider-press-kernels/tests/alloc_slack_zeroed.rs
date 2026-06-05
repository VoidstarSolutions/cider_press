//! Allocation-site slack-zeroing invariant.
//!
//! `Device::alloc_buffer` rounds the `MTLBuffer` up to 512 B (so K-padded qmv
//! can read past the logical row), then zeroes the slack `[byte_len..
//! metal_alloc_len)` once at birth. This makes the tail read back as 0.0 for
//! the buffer's lifetime — the load-bearing guarantee behind the qmv padded
//! parity (kills `bias·Σx_tail` AND the `0 * NaN = NaN` accum poisoning that
//! zero scale/bias alone cannot).
//!
//! This is the invariant's directly unit-testable surface: allocate a buffer
//! whose logical byte length is NOT a 512-multiple, then read the slack bytes
//! back through the host `contents()` pointer and assert every one is 0.

#![cfg(target_os = "macos")]

use cider_press_kernels::{Buffer, Device};
use objc2_metal::MTLBuffer;

/// Read the underlying `MTLBuffer` bytes `[0..metal_alloc_len)` through the
/// shared-storage host pointer. Mirrors `Buffer::as_slice`'s SAFETY reasoning
/// (shared storage, no concurrent dispatch), but reaches the FULL allocation —
/// `as_slice` exposes only the logical `len` elements, never the slack.
fn full_alloc_bytes<T>(buf: &Buffer<T>) -> &[u8] {
    let alloc_len = buf.metal_alloc_len();
    let ptr = buf.metal_buffer().contents().as_ptr().cast::<u8>();
    // SAFETY: `ptr` is the base of the shared-storage MTLBuffer; `alloc_len`
    // is its reported byte length; no GPU dispatch references these freshly
    // allocated test buffers, so a host read is race-free.
    unsafe { std::slice::from_raw_parts(ptr, alloc_len) }
}

#[test]
fn alloc_slack_reads_back_zero() {
    let device = Device::system_default().expect("no Metal device available");

    // Sizes whose byte length is NOT a 512-multiple, so there is real slack.
    // 700 u8 -> 700 B logical, rounded to 1024 B: 324 B slack.
    // 300 u32 -> 1200 B logical, rounded to 1536 B: 336 B slack.
    // 896 bf16 (the actual qmv activation row) -> 1792 B, rounded to 2048 B:
    //   256 B slack — the exact padded-qmv tail region.
    for len in [700usize, 1usize, 511usize] {
        let buf: Buffer<u8> = device.alloc_buffer(len).expect("alloc u8 buffer");
        let logical = buf.byte_len();
        let alloc = buf.metal_alloc_len();
        assert!(
            alloc > logical,
            "len={len}: expected slack (alloc {alloc} > logical {logical})"
        );
        assert_eq!(
            alloc,
            logical.next_multiple_of(512),
            "len={len}: alloc must be the 512-rounded logical size"
        );
        let all = full_alloc_bytes(&buf);
        assert!(
            all[logical..].iter().all(|&b| b == 0),
            "len={len}: slack bytes [{logical}..{alloc}) must read back as 0"
        );
    }

    // The actual qmv activation row: 896 bf16 = 1792 B -> 2048 B, 256 B slack.
    let act: Buffer<half::bf16> = device.alloc_buffer(896).expect("alloc bf16 activation");
    let logical = act.byte_len();
    let alloc = act.metal_alloc_len();
    assert_eq!(logical, 1792);
    assert_eq!(alloc, 2048, "896 bf16 must round 1792 B -> 2048 B");
    let all = full_alloc_bytes(&act);
    assert!(
        all[logical..].iter().all(|&b| b == 0),
        "qmv activation slack [1792..2048) must be zero (the padded-K tail region)"
    );
}

#[test]
fn alloc_exact_512_multiple_has_no_slack() {
    // 512 u8 = 512 B is already a 512-multiple: no slack, no memset, and the
    // whole allocation is the logical region.
    let device = Device::system_default().expect("no Metal device available");
    let buf: Buffer<u8> = device.alloc_buffer(512).expect("alloc u8 buffer");
    assert_eq!(buf.byte_len(), 512);
    assert_eq!(buf.metal_alloc_len(), 512, "512 B is already 512-aligned");
}
