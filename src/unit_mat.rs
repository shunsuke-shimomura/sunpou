//! Unit-aware matrix type.
//!
//! `UnitMat<DR, DC, R, C>` represents an R×C matrix where element `(i,j)` has
//! dimension `DR / DC` conceptually. When multiplied by a vector of dimension
//! `DC`, the result has dimension `DR`.
//!
//! More precisely: `UnitMat<DR, DC, R, C> * UnitVec<DC, C> → UnitVec<DR, R>`.

use core::marker::PhantomData;
use core::ops::{Add, Mul, Neg, Sub};
use nalgebra::SMatrix;

use crate::dim::{Dim, DimMultiply};
use crate::scalar::Scalar;
use crate::unit_vec::UnitVec;

/// An R×C matrix. Multiplication `UnitMat<DR, DC, R, C> * UnitVec<DC, C>`
/// produces `UnitVec<DR, R>`.
///
/// The physical interpretation: each element has dimension `DR / DC`, so that
/// multiplying by a vector of dimension `DC` yields dimension `DR`.
#[repr(transparent)]
pub struct UnitMat<DR, DC, const R: usize, const C: usize> {
    value: SMatrix<f64, R, C>,
    _dim: PhantomData<(DR, DC)>,
}

impl<DR, DC, const R: usize, const C: usize> Clone for UnitMat<DR, DC, R, C> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<DR, DC, const R: usize, const C: usize> Copy for UnitMat<DR, DC, R, C>
where
    SMatrix<f64, R, C>: Copy,
{
}

impl<DR, DC, const R: usize, const C: usize> PartialEq for UnitMat<DR, DC, R, C> {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<DR, DC, const R: usize, const C: usize> core::fmt::Debug for UnitMat<DR, DC, R, C> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "UnitMat({:?})", self.value)
    }
}

impl<DR, DC, const R: usize, const C: usize> UnitMat<DR, DC, R, C> {
    /// Create from a raw nalgebra matrix.
    #[inline(always)]
    pub fn from_raw_unchecked(value: SMatrix<f64, R, C>) -> Self {
        Self {
            value,
            _dim: PhantomData,
        }
    }

    /// Extract the raw nalgebra matrix.
    #[inline(always)]
    pub fn into_raw(self) -> SMatrix<f64, R, C> {
        self.value
    }

    /// Borrow the raw nalgebra matrix.
    #[inline(always)]
    pub fn as_raw(&self) -> &SMatrix<f64, R, C> {
        &self.value
    }

    /// Transpose: `UnitMat<DR, DC, R, C> → UnitMat<DC, DR, C, R>`.
    #[inline(always)]
    pub fn transpose(&self) -> UnitMat<DC, DR, C, R> {
        UnitMat::from_raw_unchecked(self.value.transpose())
    }
}

// ---- Identity (square, same dim) ----

impl<D, const N: usize> UnitMat<D, D, N, N> {
    /// Identity matrix. Only available when DR == DC (dimensionless elements).
    #[inline(always)]
    pub fn identity() -> Self {
        Self::from_raw_unchecked(SMatrix::identity())
    }
}

// ---- Zeros ----

impl<DR, DC, const R: usize, const C: usize> UnitMat<DR, DC, R, C> {
    /// Zero matrix.
    #[inline(always)]
    pub fn zeros() -> Self {
        Self::from_raw_unchecked(SMatrix::zeros())
    }
}

// ---- Same-type add/sub ----

impl<DR, DC, const R: usize, const C: usize> Add for UnitMat<DR, DC, R, C> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self::from_raw_unchecked(self.value + rhs.value)
    }
}

impl<DR, DC, const R: usize, const C: usize> Sub for UnitMat<DR, DC, R, C> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self::from_raw_unchecked(self.value - rhs.value)
    }
}

impl<DR, DC, const R: usize, const C: usize> Neg for UnitMat<DR, DC, R, C> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self::from_raw_unchecked(-self.value)
    }
}

// ---- Matrix-vector multiplication ----
// UnitMat<DR, DC, R, C> * UnitVec<DC, C> → UnitVec<DR, R>

impl<DR, DC, const R: usize, const C: usize> Mul<UnitVec<DC, C>> for UnitMat<DR, DC, R, C> {
    type Output = UnitVec<DR, R>;
    #[inline(always)]
    fn mul(self, rhs: UnitVec<DC, C>) -> UnitVec<DR, R> {
        UnitVec::from_raw_unchecked(self.value * rhs.into_raw())
    }
}

impl<DR, DC, const R: usize, const C: usize> Mul<&UnitVec<DC, C>> for &UnitMat<DR, DC, R, C> {
    type Output = UnitVec<DR, R>;
    #[inline(always)]
    fn mul(self, rhs: &UnitVec<DC, C>) -> UnitVec<DR, R> {
        UnitVec::from_raw_unchecked(self.value * rhs.as_raw())
    }
}

// ---- Matrix-matrix multiplication ----
// UnitMat<DR, DM, R, K> * UnitMat<DM, DC, K, C> → UnitMat<DR, DC, R, C>
// The "middle" dimension DM cancels out.

impl<DR, DM, DC, const R: usize, const K: usize, const C: usize>
    Mul<UnitMat<DM, DC, K, C>> for UnitMat<DR, DM, R, K>
{
    type Output = UnitMat<DR, DC, R, C>;
    #[inline(always)]
    fn mul(self, rhs: UnitMat<DM, DC, K, C>) -> UnitMat<DR, DC, R, C> {
        UnitMat::from_raw_unchecked(self.value * rhs.value)
    }
}

impl<DR, DM, DC, const R: usize, const K: usize, const C: usize>
    Mul<&UnitMat<DM, DC, K, C>> for &UnitMat<DR, DM, R, K>
{
    type Output = UnitMat<DR, DC, R, C>;
    #[inline(always)]
    fn mul(self, rhs: &UnitMat<DM, DC, K, C>) -> UnitMat<DR, DC, R, C> {
        UnitMat::from_raw_unchecked(self.value * rhs.value)
    }
}

// ---- f64 scaling ----

impl<DR, DC, const R: usize, const C: usize> Mul<f64> for UnitMat<DR, DC, R, C> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: f64) -> Self {
        Self::from_raw_unchecked(self.value * rhs)
    }
}

// ---- Scalar multiplication ----
// Scalar<DS> * UnitMat<DR, DC, R, C> → UnitMat<DimMul<DS, DR>, DC, R, C>
// (scales the row dimension)

impl<LS, MS, TS, IS, ThS, NS, JS, LR, MR, TR, IR, ThR, NR, JR, DC, const R: usize, const C: usize>
    Mul<UnitMat<Dim<LR, MR, TR, IR, ThR, NR, JR>, DC, R, C>>
    for Scalar<Dim<LS, MS, TS, IS, ThS, NS, JS>>
where
    Dim<LS, MS, TS, IS, ThS, NS, JS>: DimMultiply<Dim<LR, MR, TR, IR, ThR, NR, JR>>,
{
    type Output = UnitMat<
        <Dim<LS, MS, TS, IS, ThS, NS, JS> as DimMultiply<
            Dim<LR, MR, TR, IR, ThR, NR, JR>,
        >>::Output,
        DC,
        R,
        C,
    >;
    #[inline(always)]
    fn mul(
        self,
        rhs: UnitMat<Dim<LR, MR, TR, IR, ThR, NR, JR>, DC, R, C>,
    ) -> Self::Output {
        UnitMat::from_raw_unchecked(rhs.value * self.into_raw())
    }
}

// ---- Square matrix inverse ----

impl<DR, DC, const N: usize> UnitMat<DR, DC, N, N> {
    /// Inverse: `UnitMat<DR, DC, N, N>⁻¹ → UnitMat<DC, DR, N, N>`.
    ///
    /// Returns `None` if the matrix is singular.
    #[inline(always)]
    pub fn try_inverse(&self) -> Option<UnitMat<DC, DR, N, N>> {
        self.value.try_inverse().map(UnitMat::from_raw_unchecked)
    }
}
