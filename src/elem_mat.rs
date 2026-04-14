//! Frame-less element-dimension matrix.
//!
//! `ElemMat<E, R, C>` is a matrix where every element has SI dimension `E`.
//! When multiplied by `UnitVec<D, C>`, the output is `UnitVec<DimMul<E, D>, R>`.
//!
//! This replaces the old `UnitMat<DR, DC, R, C>` with a simpler, more flexible
//! model: the output dimension is inferred from `E * input_dim` rather than
//! being fixed at construction time.

use core::marker::PhantomData;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use nalgebra::SMatrix;

use crate::dim::{Dim, DimDivide, DimMultiply};
use crate::unit_vec::UnitVec;

/// A frame-less R×C matrix with element dimension `E`.
///
/// - `ElemMat<E, R, C> * UnitVec<D, C> → UnitVec<DimMul<E, D>, R>`
/// - `ElemMat<E1, R, K> * ElemMat<E2, K, C> → ElemMat<DimMul<E1, E2>, R, C>`
/// - Transpose: `ElemMat<E, R, C> → ElemMat<DimInv<E>, C, R>`
/// - Inverse: `ElemMat<E, N, N> → ElemMat<DimInv<E>, N, N>`
#[repr(transparent)]
pub struct ElemMat<E, const R: usize, const C: usize> {
    value: SMatrix<f64, R, C>,
    _dim: PhantomData<E>,
}

impl<E, const R: usize, const C: usize> Clone for ElemMat<E, R, C> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl<E, const R: usize, const C: usize> Copy for ElemMat<E, R, C>
where
    SMatrix<f64, R, C>: Copy,
{
}

impl<E, const R: usize, const C: usize> PartialEq for ElemMat<E, R, C> {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<E, const R: usize, const C: usize> core::fmt::Debug for ElemMat<E, R, C> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "ElemMat({:?})", self.value)
    }
}

/// Type alias for the "inverse dimension" of E: Dimensionless / E.
pub type DimInv<E> = <crate::aliases::Dimensionless as DimDivide<E>>::Output;

impl<E, const R: usize, const C: usize> ElemMat<E, R, C> {
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

    /// Transpose: element dimension inverts to 1/E.
    ///
    /// If `M` maps `Vec<D> → Vec<E*D>`, then `Mᵀ` maps `Vec<D> → Vec<D/E>`.
    #[inline(always)]
    pub fn transpose(&self) -> ElemMat<DimInv<E>, C, R>
    where
        crate::aliases::Dimensionless: DimDivide<E>,
    {
        ElemMat::from_raw_unchecked(self.value.transpose())
    }

    /// Zero matrix.
    #[inline(always)]
    pub fn zeros() -> Self {
        Self::from_raw_unchecked(SMatrix::zeros())
    }
}

// ---- Identity (square, dimensionless elements) ----

impl<const N: usize> ElemMat<crate::aliases::Dimensionless, N, N> {
    /// Dimensionless identity matrix.
    #[inline(always)]
    pub fn identity() -> Self {
        Self::from_raw_unchecked(SMatrix::identity())
    }
}

// ---- Same-type add/sub ----

impl<E, const R: usize, const C: usize> Add for ElemMat<E, R, C> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self::from_raw_unchecked(self.value + rhs.value)
    }
}

impl<E, const R: usize, const C: usize> Sub for ElemMat<E, R, C> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self::from_raw_unchecked(self.value - rhs.value)
    }
}

impl<E, const R: usize, const C: usize> Neg for ElemMat<E, R, C> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self::from_raw_unchecked(-self.value)
    }
}

impl<E, const R: usize, const C: usize> AddAssign for ElemMat<E, R, C> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        self.value += rhs.value;
    }
}

impl<E, const R: usize, const C: usize> SubAssign for ElemMat<E, R, C> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        self.value -= rhs.value;
    }
}

// ---- Matrix-vector multiplication ----
// ElemMat<E, R, C> * UnitVec<D, C> → UnitVec<DimMul<E, D>, R>

impl<L1, M1, T1, I1, Th1, N1, J1, L2, M2, T2, I2, Th2, N2, J2, const R: usize, const C: usize>
    Mul<UnitVec<Dim<L2, M2, T2, I2, Th2, N2, J2>, C>>
    for ElemMat<Dim<L1, M1, T1, I1, Th1, N1, J1>, R, C>
where
    Dim<L1, M1, T1, I1, Th1, N1, J1>: DimMultiply<Dim<L2, M2, T2, I2, Th2, N2, J2>>,
{
    type Output = UnitVec<
        <Dim<L1, M1, T1, I1, Th1, N1, J1> as DimMultiply<
            Dim<L2, M2, T2, I2, Th2, N2, J2>,
        >>::Output,
        R,
    >;
    #[inline(always)]
    fn mul(self, rhs: UnitVec<Dim<L2, M2, T2, I2, Th2, N2, J2>, C>) -> Self::Output {
        UnitVec::from_raw_unchecked(self.value * rhs.into_raw())
    }
}

// ---- Matrix-matrix multiplication ----
// ElemMat<E1, R, K> * ElemMat<E2, K, C> → ElemMat<DimMul<E1, E2>, R, C>

impl<L1, M1, T1, I1, Th1, N1, J1, L2, M2, T2, I2, Th2, N2, J2, const R: usize, const K: usize, const C: usize>
    Mul<ElemMat<Dim<L2, M2, T2, I2, Th2, N2, J2>, K, C>>
    for ElemMat<Dim<L1, M1, T1, I1, Th1, N1, J1>, R, K>
where
    Dim<L1, M1, T1, I1, Th1, N1, J1>: DimMultiply<Dim<L2, M2, T2, I2, Th2, N2, J2>>,
{
    type Output = ElemMat<
        <Dim<L1, M1, T1, I1, Th1, N1, J1> as DimMultiply<
            Dim<L2, M2, T2, I2, Th2, N2, J2>,
        >>::Output,
        R,
        C,
    >;
    #[inline(always)]
    fn mul(self, rhs: ElemMat<Dim<L2, M2, T2, I2, Th2, N2, J2>, K, C>) -> Self::Output {
        ElemMat::from_raw_unchecked(self.value * rhs.value)
    }
}

// ---- f64 scaling ----

impl<E, const R: usize, const C: usize> Mul<f64> for ElemMat<E, R, C> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: f64) -> Self {
        Self::from_raw_unchecked(self.value * rhs)
    }
}

impl<E, const R: usize, const C: usize> MulAssign<f64> for ElemMat<E, R, C> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: f64) {
        self.value *= rhs;
    }
}

// ---- Square matrix inverse ----

impl<E, const N: usize> ElemMat<E, N, N> {
    /// Inverse: element dimension becomes 1/E.
    #[inline(always)]
    pub fn try_inverse(&self) -> Option<ElemMat<DimInv<E>, N, N>>
    where
        crate::aliases::Dimensionless: DimDivide<E>,
    {
        self.value.try_inverse().map(ElemMat::from_raw_unchecked)
    }
}

// ---- TransposeBlock for block matrix support ----

impl<E, const R: usize, const C: usize> crate::block::TransposeBlock for ElemMat<E, R, C>
where
    crate::aliases::Dimensionless: DimDivide<E>,
{
    type Output = ElemMat<DimInv<E>, C, R>;
    #[inline(always)]
    fn block_transpose(self) -> Self::Output {
        self.transpose()
    }
}
