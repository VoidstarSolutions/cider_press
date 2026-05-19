//! Logical shape of a tensor.
//!
//! [`Shape`] is a thin newtype around `Vec<usize>` so that ergonomic
//! constructors and accessors can be added without re-deriving them on
//! every caller. Rank zero ("scalar") and rank one ("vector") are both
//! valid; an empty shape is a scalar.
//!
//! Strides live in [`Strides`](crate::Strides), not here — a [`Shape`]
//! is a pure logical descriptor with no notion of memory order. The
//! pair (Shape, Strides) is what describes a view; this type alone
//! describes only "how many elements and along which axes."

/// Logical shape: an ordered list of dimension lengths.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct Shape(Vec<usize>);

impl Shape {
    /// Construct a shape from any iterable of dimensions.
    #[must_use]
    pub fn new(dims: impl IntoIterator<Item = usize>) -> Self {
        Self(dims.into_iter().collect())
    }

    /// The scalar shape (rank zero).
    #[must_use]
    pub const fn scalar() -> Self {
        Self(Vec::new())
    }

    /// Number of dimensions.
    #[must_use]
    pub fn rank(&self) -> usize {
        self.0.len()
    }

    /// Whether this is the scalar (rank-zero) shape.
    #[must_use]
    pub fn is_scalar(&self) -> bool {
        self.0.is_empty()
    }

    /// Dimensions as a slice.
    #[must_use]
    pub fn dims(&self) -> &[usize] {
        &self.0
    }

    /// Length along the given axis, or `None` if `axis >= rank()`.
    #[must_use]
    pub fn dim(&self, axis: usize) -> Option<usize> {
        self.0.get(axis).copied()
    }

    /// Total element count = product of dimensions.
    ///
    /// Returns `1` for the scalar shape (empty product). Panics if
    /// the product overflows `usize` — an `elem_count` that doesn't
    /// fit in a machine word can't be allocated anyway, and
    /// downstream `from_raw_parts` paths assume this value is
    /// faithful (silent wrap would be unsound, not just slow).
    #[must_use]
    pub fn elem_count(&self) -> usize {
        self.0
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .expect("shape element count overflowed usize")
    }
}

impl From<Vec<usize>> for Shape {
    fn from(dims: Vec<usize>) -> Self {
        Self(dims)
    }
}

impl From<&[usize]> for Shape {
    fn from(dims: &[usize]) -> Self {
        Self(dims.to_vec())
    }
}

impl<const N: usize> From<[usize; N]> for Shape {
    fn from(dims: [usize; N]) -> Self {
        Self(dims.to_vec())
    }
}

impl core::fmt::Display for Shape {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("[")?;
        for (i, d) in self.0.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{d}")?;
        }
        f.write_str("]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_elem_count_is_one() {
        let s = Shape::scalar();
        assert!(s.is_scalar());
        assert_eq!(s.rank(), 0);
        assert_eq!(s.elem_count(), 1);
    }

    #[test]
    fn elem_count_is_product() {
        let s = Shape::from([2, 3, 4]);
        assert_eq!(s.rank(), 3);
        assert_eq!(s.elem_count(), 24);
        assert_eq!(s.dims(), &[2, 3, 4]);
        assert_eq!(s.dim(0), Some(2));
        assert_eq!(s.dim(3), None);
    }

    #[test]
    fn zero_dim_zeros_count() {
        let s = Shape::from([4, 0, 3]);
        assert_eq!(s.elem_count(), 0);
    }

    #[test]
    fn display() {
        assert_eq!(Shape::scalar().to_string(), "[]");
        assert_eq!(Shape::from([2, 3, 4]).to_string(), "[2, 3, 4]");
    }

    #[test]
    #[should_panic(expected = "overflowed")]
    fn elem_count_panics_on_overflow() {
        // usize::MAX × 2 overflows; checked accumulator panics rather
        // than silently wrapping (which would yield an under-sized
        // allocation downstream).
        let s = Shape::from([usize::MAX, 2]);
        let _ = s.elem_count();
    }
}
