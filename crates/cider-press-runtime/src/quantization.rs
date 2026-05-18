//! Descriptor for MLX-style affine quantization layouts.
//!
//! A quantized weight matrix is packed into `u32` words: each word
//! holds `32 / bits` scalars. The matrix is divided into groups of
//! `group_size` scalars along the inner (K) axis; each group has one
//! scale and one bias stored separately (typically as bf16 or f16
//! tensors). Dequantization is `x = w * scale + bias` per group.
//!
//! This descriptor lives on [`Layout::Quantized`](crate::Layout::Quantized);
//! the packed `u32` buffer is the [`Tensor`](crate::Tensor) it
//! decorates. The scale and bias tensors are *separate*
//! [`Tensor`](crate::Tensor)s — they share a logical group count with
//! the packed weight but are otherwise independent dense arrays. Ops
//! that consume quantized weights (e.g. `qmv`) take all three.

use crate::error::{Error, Result};

/// Supported `bits` values, copied from MLX's affine quantization
/// catalogue: 2, 3, 4, 6, 8.
const SUPPORTED_BITS: [u8; 5] = [2, 3, 4, 6, 8];

/// Descriptor for an MLX-style affine quantization layout.
///
/// `bits` is the per-scalar width of the packed values; `group_size`
/// is the number of scalars sharing one (scale, bias) pair along the
/// quantized axis.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Quantization {
    /// Bit width of each packed scalar (one of {2, 3, 4, 6, 8}).
    pub bits: u8,
    /// Number of scalars sharing one scale/bias pair.
    pub group_size: u32,
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

    /// Reject malformed descriptors.
    ///
    /// Constraints:
    /// - `bits ∈ {2, 3, 4, 6, 8}` (the MLX-supported set).
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

    /// Number of scalars packed into one `u32` word.
    #[must_use]
    pub const fn scalars_per_word(self) -> u32 {
        32 / self.bits as u32
    }

    /// Number of `u32` words needed to pack `scalar_count` scalars.
    ///
    /// Errors if `scalar_count` is not a multiple of
    /// [`scalars_per_word`](Self::scalars_per_word) — partial packing
    /// isn't supported.
    pub fn packed_word_count(self, scalar_count: usize) -> Result<usize> {
        let per_word = self.scalars_per_word() as usize;
        if scalar_count % per_word != 0 {
            return Err(Error::InvalidArgument(format!(
                "scalar_count {scalar_count} is not divisible by {per_word} (= 32 / {} bits)",
                self.bits
            )));
        }
        Ok(scalar_count / per_word)
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
        assert_eq!(Quantization::Q4_GS64.scalars_per_word(), 8);
    }

    #[test]
    fn all_supported_bit_widths_validate() {
        for &bits in &SUPPORTED_BITS {
            Quantization::new(bits, 64).unwrap();
        }
    }

    #[test]
    fn rejects_unsupported_bits() {
        assert!(Quantization::new(5, 64).is_err());
        assert!(Quantization::new(0, 64).is_err());
        assert!(Quantization::new(16, 64).is_err());
    }

    #[test]
    fn rejects_non_power_of_two_group_size() {
        assert!(Quantization::new(4, 0).is_err());
        assert!(Quantization::new(4, 48).is_err());
    }

    #[test]
    fn packed_word_count_4_gs64() {
        // 64 scalars / 8 per word = 8 u32s per group.
        let q = Quantization::Q4_GS64;
        assert_eq!(q.packed_word_count(64).unwrap(), 8);
        assert_eq!(q.packed_word_count(4096).unwrap(), 512);
        assert!(q.packed_word_count(7).is_err());
    }

    #[test]
    fn group_count_basics() {
        let q = Quantization::Q4_GS64;
        assert_eq!(q.group_count(64).unwrap(), 1);
        assert_eq!(q.group_count(4096).unwrap(), 64);
        assert!(q.group_count(100).is_err());
    }
}
