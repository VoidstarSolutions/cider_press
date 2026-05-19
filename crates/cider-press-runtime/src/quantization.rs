//! Descriptor for MLX-style affine quantization layouts.
//!
//! A quantized weight matrix is bit-packed into `u32` words: for
//! `bits` ∈ {2, 4, 8} one word holds exactly `32 / bits` scalars,
//! and for `bits` ∈ {3, 5, 6} lanes straddle word boundaries — the
//! invariant the kernels rely on is that `group_size * bits` is a
//! multiple of 32, so a whole number of `u32` words spans one group
//! (see [`Quantization::validate`]). The total word count is always
//! `scalar_count * bits / 32`; see
//! [`Quantization::packed_word_count`].
//!
//! The matrix is divided into groups of `group_size` scalars along
//! the inner (K) axis; each group has one scale and one bias
//! (typically as bf16). At the runtime layer these three buffers
//! (packed weights, scales, biases) live together on a single
//! [`Tensor`](crate::Tensor) in `LeafStorage::Quantized`, and are
//! bound side-by-side at dispatch time the way MLX's
//! `quantized_matmul` ABI expects. Dequantization is
//! `x = w * scale + bias` per group.
//!
//! This descriptor lives on [`Layout::Quantized`](crate::Layout::Quantized);
//! the packed `u32` buffer is the [`Tensor`](crate::Tensor) it
//! decorates. The scale and bias tensors are *separate*
//! [`Tensor`](crate::Tensor)s — they share a logical group count with
//! the packed weight but are otherwise independent dense arrays. Ops
//! that consume quantized weights (e.g. `qmv`) take all three.

use crate::error::{Error, Result};

/// Supported `bits` values, mirroring MLX's affine quantization
/// catalogue: 2, 3, 4, 5, 6, 8 (the set
/// `instantiate_quantized_all()` emits in
/// `quantized.metal`).
const SUPPORTED_BITS: [u8; 6] = [2, 3, 4, 5, 6, 8];

/// Descriptor for an MLX-style affine quantization layout.
///
/// `bits` is the per-scalar width of the packed values; `group_size`
/// is the number of scalars sharing one (scale, bias) pair along the
/// quantized axis. Fields are private so every value goes through
/// [`Quantization::new`] (or the named constants), which enforces
/// the invariants downstream helpers rely on.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Quantization {
    bits: u8,
    group_size: u32,
}

impl Quantization {
    /// 4-bit, group size 64 — the dominant production config for
    /// Llama / Qwen 4-bit quants and the layout the Stage 4 spike
    /// validated bit-exactly against MLX.
    pub const Q4_GS64: Self = Self {
        bits: 4,
        group_size: 64,
    };

    /// Construct with validation. See [`Quantization::validate`] for
    /// the accepted set.
    pub fn new(bits: u8, group_size: u32) -> Result<Self> {
        let q = Self { bits, group_size };
        q.validate()?;
        Ok(q)
    }

    /// Bit width of each packed scalar.
    #[must_use]
    pub const fn bits(self) -> u8 {
        self.bits
    }

    /// Number of scalars sharing one scale/bias pair.
    #[must_use]
    pub const fn group_size(self) -> u32 {
        self.group_size
    }

    /// Reject malformed descriptors.
    ///
    /// Constraints:
    /// - `bits ∈ {2, 3, 4, 5, 6, 8}` (the MLX-supported set).
    /// - `group_size > 0` and a power of two (the kernels assume this
    ///   for the tile math).
    /// - `group_size * bits` is a multiple of 32, so a whole number of
    ///   packed `u32` words spans one group.
    pub fn validate(self) -> Result<()> {
        if !SUPPORTED_BITS.contains(&self.bits) {
            return Err(Error::InvalidArgument(format!(
                "unsupported quantization bits {} — expected one of {SUPPORTED_BITS:?}",
                self.bits
            )));
        }
        if self.group_size == 0 || !self.group_size.is_power_of_two() {
            return Err(Error::InvalidArgument(format!(
                "group_size {} must be a positive power of two",
                self.group_size
            )));
        }
        let group_bits = u64::from(self.group_size) * u64::from(self.bits);
        if group_bits % 32 != 0 {
            return Err(Error::InvalidArgument(format!(
                "group_size {} × bits {} = {} bits per group does not fit in whole u32 words",
                self.group_size, self.bits, group_bits
            )));
        }
        Ok(())
    }

    /// Number of `u32` words needed to pack `scalar_count` scalars.
    ///
    /// Uses the canonical MLX formula `total_bits / 32` — works for
    /// every supported `bits` value, including ones that don't divide
    /// 32 (3, 5, 6) because the [`Self::validate`] precondition
    /// guarantees `group_size * bits` is a multiple of 32, and
    /// `scalar_count` is required to be a multiple of `group_size`
    /// (enforced below).
    ///
    /// Errors if `scalar_count` is not a multiple of `group_size`
    /// (partial groups aren't supported) or if the bit-count
    /// computation overflows `usize`.
    pub fn packed_word_count(self, scalar_count: usize) -> Result<usize> {
        let gs = self.group_size as usize;
        if scalar_count % gs != 0 {
            return Err(Error::InvalidArgument(format!(
                "scalar_count {scalar_count} is not divisible by group_size {gs}",
            )));
        }
        let total_bits = scalar_count
            .checked_mul(self.bits as usize)
            .ok_or_else(|| {
                Error::InvalidArgument(format!(
                    "scalar_count {scalar_count} × bits {} overflows usize",
                    self.bits,
                ))
            })?;
        debug_assert!(
            total_bits % 32 == 0,
            "validate() guarantees group_size * bits is a multiple of 32"
        );
        Ok(total_bits / 32)
    }

    /// Number of groups (= number of (scale, bias) pairs) for
    /// `scalar_count` scalars along the quantized axis.
    pub fn group_count(self, scalar_count: usize) -> Result<usize> {
        let gs = self.group_size as usize;
        if scalar_count % gs != 0 {
            return Err(Error::InvalidArgument(format!(
                "scalar_count {scalar_count} is not divisible by group_size {gs}",
            )));
        }
        Ok(scalar_count / gs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q4_gs64_validates() {
        Quantization::Q4_GS64.validate().unwrap();
        assert_eq!(Quantization::Q4_GS64.bits(), 4);
        assert_eq!(Quantization::Q4_GS64.group_size(), 64);
    }

    #[test]
    fn all_supported_bit_widths_validate() {
        for &bits in &SUPPORTED_BITS {
            Quantization::new(bits, 64).unwrap();
        }
    }

    #[test]
    fn rejects_unsupported_bits() {
        assert!(Quantization::new(1, 64).is_err());
        assert!(Quantization::new(0, 64).is_err());
        assert!(Quantization::new(7, 64).is_err());
        assert!(Quantization::new(16, 64).is_err());
    }

    #[test]
    fn rejects_non_power_of_two_group_size() {
        assert!(Quantization::new(4, 0).is_err());
        assert!(Quantization::new(4, 48).is_err());
    }

    #[test]
    fn packed_word_count_4_gs64() {
        // 64 scalars × 4 bits = 256 bits = 8 u32 words per group.
        let q = Quantization::Q4_GS64;
        assert_eq!(q.packed_word_count(64).unwrap(), 8);
        assert_eq!(q.packed_word_count(4096).unwrap(), 512);
        assert!(q.packed_word_count(7).is_err()); // not a multiple of group_size
    }

    #[test]
    fn packed_word_count_non_divisor_bits() {
        // bits=3, group_size=32: 32 × 3 = 96 bits = 3 words per group.
        // The old `scalar_count / (32/bits)` formula gave the wrong
        // answer here (32 / 10 = 3 → wrong unit anyway, and rejected
        // 32 as non-divisible). The bits-based formula gets it right.
        let q3 = Quantization::new(3, 32).unwrap();
        assert_eq!(q3.packed_word_count(32).unwrap(), 3);
        assert_eq!(q3.packed_word_count(64).unwrap(), 6);

        // bits=5, group_size=32: 32 × 5 = 160 bits = 5 words per group.
        let q5 = Quantization::new(5, 32).unwrap();
        assert_eq!(q5.packed_word_count(32).unwrap(), 5);
        assert_eq!(q5.packed_word_count(128).unwrap(), 20);

        // bits=6, group_size=32: 32 × 6 = 192 bits = 6 words per group.
        let q6 = Quantization::new(6, 32).unwrap();
        assert_eq!(q6.packed_word_count(32).unwrap(), 6);
        assert_eq!(q6.packed_word_count(64).unwrap(), 12);
    }

    #[test]
    fn group_count_basics() {
        let q = Quantization::Q4_GS64;
        assert_eq!(q.group_count(64).unwrap(), 1);
        assert_eq!(q.group_count(4096).unwrap(), 64);
        assert!(q.group_count(100).is_err());
    }
}
