//! Scalar dtypes the runtime understands.
//!
//! [`DType`] is the dynamic tag carried by every [`Tensor`](crate::Tensor);
//! [`Scalar`] is the compile-time bridge from a Rust scalar type to the
//! matching [`DType`] discriminant. The two together let the runtime stay
//! generic over element type where useful (host-side uploads, parity
//! tests) while keeping a single dynamic-typed [`Tensor`] handle on the
//! graph side.
//!
//! Quantized weight layouts are *not* a [`DType`]. They live on
//! [`Layout`](crate::Layout) because a packed-int4 tensor is still
//! u32-typed at the buffer level; the `bits`/`group_size` are layout
//! metadata, not a scalar type. This keeps the dtype set small enough
//! to exhaustively match in dispatch code.

use half::{bf16, f16};

/// Scalar element type tag.
///
/// Every [`Tensor`](crate::Tensor) carries one of these. The variants
/// are intentionally only the *storage* dtypes used by kernels — there
/// is no "quantized" variant here; that's a [`Layout`](crate::Layout)
/// distinction sitting on top of [`DType::U32`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    /// IEEE-754 single-precision binary32.
    F32,
    /// IEEE-754 half-precision binary16.
    F16,
    /// "Brain" 16-bit float (8 exponent bits, 7 mantissa bits).
    BF16,
    /// Two's-complement 32-bit integer.
    I32,
    /// Unsigned 32-bit integer. Also the storage type for packed
    /// quantized weights ([`Layout::Quantized`](crate::Layout::Quantized)).
    U32,
}

impl DType {
    /// Size of one element in bytes.
    #[must_use]
    pub const fn size_bytes(self) -> usize {
        match self {
            DType::F32 | DType::I32 | DType::U32 => 4,
            DType::F16 | DType::BF16 => 2,
        }
    }

    /// Whether this dtype is a floating-point type.
    #[must_use]
    pub const fn is_float(self) -> bool {
        matches!(self, DType::F32 | DType::F16 | DType::BF16)
    }

    /// Whether this dtype is an integer type.
    #[must_use]
    pub const fn is_int(self) -> bool {
        matches!(self, DType::I32 | DType::U32)
    }

    /// Short stable identifier, suitable for embedding in kernel names
    /// or in user-facing error strings.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            DType::I32 => "i32",
            DType::U32 => "u32",
        }
    }
}

impl core::fmt::Display for DType {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.name())
    }
}

mod private {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for half::f16 {}
    impl Sealed for half::bf16 {}
    impl Sealed for i32 {}
    impl Sealed for u32 {}
}

/// Compile-time bridge from a Rust scalar type to its [`DType`].
///
/// Implemented (sealed) for every type [`DType`] knows about. Use this
/// when you want a function generic over element type to look up the
/// matching dynamic tag — e.g. when uploading a `&[T]` and tagging the
/// resulting tensor.
pub trait Scalar: Copy + private::Sealed + 'static {
    /// The [`DType`] discriminant for this scalar type.
    const DTYPE: DType;
}

impl Scalar for f32 {
    const DTYPE: DType = DType::F32;
}
impl Scalar for f16 {
    const DTYPE: DType = DType::F16;
}
impl Scalar for bf16 {
    const DTYPE: DType = DType::BF16;
}
impl Scalar for i32 {
    const DTYPE: DType = DType::I32;
}
impl Scalar for u32 {
    const DTYPE: DType = DType::U32;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn size_bytes_matches_rust_size_of() {
        assert_eq!(DType::F32.size_bytes(), std::mem::size_of::<f32>());
        assert_eq!(DType::F16.size_bytes(), std::mem::size_of::<f16>());
        assert_eq!(DType::BF16.size_bytes(), std::mem::size_of::<bf16>());
        assert_eq!(DType::I32.size_bytes(), std::mem::size_of::<i32>());
        assert_eq!(DType::U32.size_bytes(), std::mem::size_of::<u32>());
    }

    #[test]
    fn scalar_trait_maps_to_dtype() {
        assert_eq!(<f32 as Scalar>::DTYPE, DType::F32);
        assert_eq!(<f16 as Scalar>::DTYPE, DType::F16);
        assert_eq!(<bf16 as Scalar>::DTYPE, DType::BF16);
        assert_eq!(<i32 as Scalar>::DTYPE, DType::I32);
        assert_eq!(<u32 as Scalar>::DTYPE, DType::U32);
    }

    #[test]
    fn float_int_partition() {
        for dt in [DType::F32, DType::F16, DType::BF16] {
            assert!(dt.is_float());
            assert!(!dt.is_int());
        }
        for dt in [DType::I32, DType::U32] {
            assert!(dt.is_int());
            assert!(!dt.is_float());
        }
    }
}
