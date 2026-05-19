//! Memory strides for a tensor view.
//!
//! Strides are signed (`isize`) because views like reverse, flip, and
//! some transpose conventions need negative steps. Strides are
//! expressed in *elements*, not bytes — multiplying by the dtype size
//! is the caller's job (typically at kernel-dispatch time).
//!
//! [`Strides::contiguous`] derives row-major (C-order) strides from a
//! [`Shape`]. This is the layout we materialize fresh allocations as
//! and the layout most ops assume by default; non-contiguous layouts
//! arise from view-producing ops (transpose, slice, etc.) and from
//! quantized weight matrices.

use crate::Shape;

/// Per-axis step in *elements* between successive entries along that axis.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct Strides(Vec<isize>);

impl Strides {
    /// Construct directly from a list of signed strides.
    #[must_use]
    pub fn new(strides: impl IntoIterator<Item = isize>) -> Self {
        Self(strides.into_iter().collect())
    }

    /// Row-major (C-order) contiguous strides for `shape`.
    ///
    /// The last axis steps by 1, then each preceding axis steps by the
    /// product of the dimensions after it. Returns an empty stride list
    /// for the scalar shape.
    #[must_use]
    pub fn contiguous(shape: &Shape) -> Self {
        let dims = shape.dims();
        if dims.is_empty() {
            return Self(Vec::new());
        }
        let mut strides = vec![1isize; dims.len()];
        for axis in (0..dims.len() - 1).rev() {
            let next =
                isize::try_from(dims[axis + 1]).expect("shape dimension does not fit in isize");
            strides[axis] = strides[axis + 1]
                .checked_mul(next)
                .expect("contiguous stride computation overflowed isize");
        }
        Self(strides)
    }

    /// Number of strides — should match the rank of the associated shape.
    #[must_use]
    pub fn rank(&self) -> usize {
        self.0.len()
    }

    /// View as a slice.
    #[must_use]
    pub fn as_slice(&self) -> &[isize] {
        &self.0
    }

    /// Whether these strides describe the row-major contiguous layout
    /// of `shape`.
    #[must_use]
    pub fn is_contiguous(&self, shape: &Shape) -> bool {
        self == &Self::contiguous(shape)
    }
}

impl From<Vec<isize>> for Strides {
    fn from(s: Vec<isize>) -> Self {
        Self(s)
    }
}

impl From<&[isize]> for Strides {
    fn from(s: &[isize]) -> Self {
        Self(s.to_vec())
    }
}

impl<const N: usize> From<[isize; N]> for Strides {
    fn from(s: [isize; N]) -> Self {
        Self(s.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contiguous_for_scalar_is_empty() {
        let s = Strides::contiguous(&Shape::scalar());
        assert_eq!(s.rank(), 0);
        assert!(s.as_slice().is_empty());
    }

    #[test]
    fn contiguous_row_major_3d() {
        let strides = Strides::contiguous(&Shape::from([2, 3, 4]));
        // Last axis = 1, middle = 4, first = 12.
        assert_eq!(strides.as_slice(), &[12, 4, 1]);
    }

    #[test]
    fn contiguous_1d() {
        let strides = Strides::contiguous(&Shape::from([7]));
        assert_eq!(strides.as_slice(), &[1]);
    }

    #[test]
    fn is_contiguous_round_trips() {
        let shape = Shape::from([2, 3, 4]);
        let strides = Strides::contiguous(&shape);
        assert!(strides.is_contiguous(&shape));
        let transposed = Strides::from([1, 4, 12]);
        assert!(!transposed.is_contiguous(&shape));
    }
}
