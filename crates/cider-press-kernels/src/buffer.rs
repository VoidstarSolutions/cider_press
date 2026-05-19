//! Typed Metal buffer handle.
//!
//! [`Buffer<T>`] wraps an `MTLBuffer` of shared (unified-memory) storage,
//! carrying the element type at compile time. Kernel dispatch functions
//! accept typed buffers so dtype mismatches against the underlying
//! kernel signature fail to compile rather than producing silent
//! garbage at runtime.
//!
//! Allocation is on [`crate::device::Device`]; this module just defines
//! the handle and host-side access.

use std::marker::PhantomData;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLBuffer;

/// Typed handle to a shared-storage [`MTLBuffer`].
///
/// `T` is the element type. The buffer's length in elements is
/// remembered separately from the underlying byte length so
/// [`Buffer::as_slice`] etc. can return correctly-typed slices without
/// the caller re-deriving the count.
pub struct Buffer<T> {
    raw: Retained<ProtocolObject<dyn MTLBuffer>>,
    len: usize,
    _marker: PhantomData<T>,
}

// SAFETY: Apple's documentation guarantees `MTLBuffer` (and its parent
// `MTLResource`) is safe to reference from multiple threads. The unsafe
// `as_slice` / `as_mut_slice` accessors below carry the explicit data-race
// contract against GPU dispatch; Send+Sync on the handle just lets it cross
// thread boundaries, which is needed for the runtime layer's
// `Arc<TensorInner>` to itself be `Send+Sync`.
unsafe impl<T: Send> Send for Buffer<T> {}
unsafe impl<T: Sync> Sync for Buffer<T> {}

impl<T> Buffer<T> {
    /// Construct a typed handle from a freshly-allocated raw buffer.
    ///
    /// # Safety
    ///
    /// The raw buffer must have been allocated with at least
    /// `len * size_of::<T>()` bytes and with shared storage; `T` must
    /// have the alignment Metal would have given the underlying type
    /// (true for the scalar types we use: `f32`, `u32`, `i32`, `f16`,
    /// `bf16`, etc.).
    pub(crate) unsafe fn from_raw_parts(
        raw: Retained<ProtocolObject<dyn MTLBuffer>>,
        len: usize,
    ) -> Self {
        Self {
            raw,
            len,
            _marker: PhantomData,
        }
    }

    /// Length in elements of `T`.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer holds zero elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Logical length of the buffer in bytes: `len * size_of::<T>()`.
    ///
    /// This is the byte count cider-press requested from Metal. The
    /// underlying `MTLBuffer`'s page-aligned allocation may be larger,
    /// but that rounded-up size is not exposed here. Use
    /// `self.metal_buffer().length()` if you need it.
    #[must_use]
    pub fn byte_len(&self) -> usize {
        self.len * size_of::<T>()
    }

    /// Escape hatch to the underlying [`MTLBuffer`] for use with raw
    /// Metal APIs not yet wrapped here (e.g. binding to a kernel for
    /// which we don't have a typed dispatch function). Typed dispatch
    /// functions in [`crate::kernels`] should be preferred for routine
    /// work.
    #[must_use]
    pub fn metal_buffer(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.raw
    }

    /// Reinterpret this buffer as one with element type `U`, sharing
    /// the same underlying `MTLBuffer` storage (Retained clone — a
    /// refcount bump, no copy).
    ///
    /// Intended as the byte-erasure escape hatch for callers that
    /// store buffers dtype-erased (typically the runtime layer's
    /// `Buffer<u8>` leaf) and need a typed view at op-dispatch time.
    /// The original buffer remains valid; both views co-own the
    /// `MTLBuffer` for as long as either is alive.
    ///
    /// # Safety
    ///
    /// The caller asserts:
    ///
    /// - the buffer's byte length is a multiple of `size_of::<U>()`
    ///   (the new `len()` is `byte_len() / size_of::<U>()`);
    /// - the bytes contain a valid sequence of `U` values, or `U` is
    ///   a type for which any bit pattern is well-defined (the scalar
    ///   types `cider-press` uses — `f32`, `f16`, `bf16`, `i32`,
    ///   `u32` — all qualify).
    ///
    /// Alignment is provided by Metal: shared-storage allocations are
    /// page-aligned (≥ 4 KiB), which exceeds any scalar `U`'s
    /// requirement.
    #[must_use]
    pub unsafe fn reinterpret_as<U>(&self) -> Buffer<U> {
        const {
            assert!(
                size_of::<U>() > 0,
                "Buffer<U>: zero-sized element types are not supported"
            );
        }
        let new_len = self.byte_len() / size_of::<U>();
        Buffer {
            raw: self.raw.clone(),
            len: new_len,
            _marker: PhantomData,
        }
    }
}

impl<T: Copy> Buffer<T> {
    /// View the buffer's contents as a `&[T]` through the unified-memory
    /// host pointer.
    ///
    /// # Safety
    ///
    /// The caller must ensure no GPU dispatch is concurrently writing
    /// the buffer. In practice, only read host-side after the
    /// command-buffer that wrote it has completed (e.g. after
    /// `Commands::commit_and_wait`).
    #[must_use]
    pub unsafe fn as_slice(&self) -> &[T] {
        // Metal's `contents()` is documented to return non-null for
        // any successfully-allocated buffer, but defending against an
        // empty backing allocation costs nothing.
        if self.len == 0 {
            return &[];
        }
        let ptr = self.raw.contents().cast::<T>().as_ptr();
        unsafe { std::slice::from_raw_parts(ptr, self.len) }
    }

    /// Mutable view of the buffer's contents as `&mut [T]`.
    ///
    /// # Safety
    ///
    /// Same constraint as [`Buffer::as_slice`]: no GPU dispatch may be
    /// reading or writing this buffer concurrently.
    pub unsafe fn as_mut_slice(&mut self) -> &mut [T] {
        if self.len == 0 {
            return &mut [];
        }
        let ptr = self.raw.contents().cast::<T>().as_ptr();
        unsafe { std::slice::from_raw_parts_mut(ptr, self.len) }
    }
}
