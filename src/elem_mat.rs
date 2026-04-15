//! Frame-less element-dimension matrix with SI prefix support.

use core::marker::PhantomData;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use nalgebra::SMatrix;
use typenum::Z0;

use crate::dim::{Dim, DimDivide, DimMultiply};
use crate::unit_vec::UnitVec;

/// Type alias: inverse dimension = Dimensionless / E.
pub type DimInv<E> = <crate::aliases::Dimensionless as DimDivide<E>>::Output;

/// Frame-less R×C matrix with element dimension `E` and prefix `P`.
#[repr(transparent)]
pub struct ElemMat<E, const R: usize, const C: usize, P = Z0> {
    value: SMatrix<f64, R, C>,
    _marker: PhantomData<(E, P)>,
}

impl<E, const R: usize, const C: usize, P> Clone for ElemMat<E, R, C, P> {
    #[inline(always)]
    fn clone(&self) -> Self { *self }
}
impl<E, const R: usize, const C: usize, P> Copy for ElemMat<E, R, C, P>
where SMatrix<f64, R, C>: Copy {}
impl<E, const R: usize, const C: usize, P> PartialEq for ElemMat<E, R, C, P> {
    fn eq(&self, other: &Self) -> bool { self.value == other.value }
}
impl<E, const R: usize, const C: usize, P> core::fmt::Debug for ElemMat<E, R, C, P> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "ElemMat({:?})", self.value)
    }
}

impl<E, const R: usize, const C: usize, P> ElemMat<E, R, C, P> {
    #[inline(always)]
    pub fn from_raw_unchecked(value: SMatrix<f64, R, C>) -> Self {
        Self { value, _marker: PhantomData }
    }
    #[inline(always)]
    pub fn into_raw(self) -> SMatrix<f64, R, C> { self.value }
    #[inline(always)]
    pub fn as_raw(&self) -> &SMatrix<f64, R, C> { &self.value }

    /// Transpose: element dim → 1/E, prefix → negated.
    #[inline(always)]
    pub fn transpose(&self) -> ElemMat<DimInv<E>, C, R, <Z0 as Sub<P>>::Output>
    where
        crate::aliases::Dimensionless: DimDivide<E>,
        Z0: Sub<P>,
    {
        ElemMat::from_raw_unchecked(self.value.transpose())
    }

    #[inline(always)]
    pub fn zeros() -> Self { Self::from_raw_unchecked(SMatrix::zeros()) }
}

// Identity: dimensionless, base prefix
impl<const N: usize> ElemMat<crate::aliases::Dimensionless, N, N> {
    #[inline(always)]
    pub fn identity() -> Self { Self::from_raw_unchecked(SMatrix::identity()) }
}

// ---- Add/Sub/Neg (same E, same P) ----
impl<E, const R: usize, const C: usize, P> Add for ElemMat<E, R, C, P> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self { Self::from_raw_unchecked(self.value + rhs.value) }
}
impl<E, const R: usize, const C: usize, P> Sub for ElemMat<E, R, C, P> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self { Self::from_raw_unchecked(self.value - rhs.value) }
}
impl<E, const R: usize, const C: usize, P> Neg for ElemMat<E, R, C, P> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self { Self::from_raw_unchecked(-self.value) }
}
impl<E, const R: usize, const C: usize, P> AddAssign for ElemMat<E, R, C, P> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) { self.value += rhs.value; }
}
impl<E, const R: usize, const C: usize, P> SubAssign for ElemMat<E, R, C, P> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) { self.value -= rhs.value; }
}

// ---- Mat * Vec (cross-dim, cross-prefix) ----
impl<L1, M1, T1, I1, Th1, N1, J1, PM, L2, M2, T2, I2, Th2, N2, J2, PV, const R: usize, const C: usize>
    Mul<UnitVec<Dim<L2, M2, T2, I2, Th2, N2, J2>, C, PV>>
    for ElemMat<Dim<L1, M1, T1, I1, Th1, N1, J1>, R, C, PM>
where
    Dim<L1, M1, T1, I1, Th1, N1, J1>: DimMultiply<Dim<L2, M2, T2, I2, Th2, N2, J2>>,
    PM: Add<PV>,
{
    type Output = UnitVec<
        <Dim<L1, M1, T1, I1, Th1, N1, J1> as DimMultiply<Dim<L2, M2, T2, I2, Th2, N2, J2>>>::Output,
        R,
        <PM as Add<PV>>::Output,
    >;
    #[inline(always)]
    fn mul(self, rhs: UnitVec<Dim<L2, M2, T2, I2, Th2, N2, J2>, C, PV>) -> Self::Output {
        UnitVec::from_raw_unchecked(self.value * rhs.into_raw())
    }
}

// ---- Mat * Mat (cross-dim, cross-prefix) ----
impl<L1, M1, T1, I1, Th1, N1, J1, P1, L2, M2, T2, I2, Th2, N2, J2, P2, const R: usize, const K: usize, const C: usize>
    Mul<ElemMat<Dim<L2, M2, T2, I2, Th2, N2, J2>, K, C, P2>>
    for ElemMat<Dim<L1, M1, T1, I1, Th1, N1, J1>, R, K, P1>
where
    Dim<L1, M1, T1, I1, Th1, N1, J1>: DimMultiply<Dim<L2, M2, T2, I2, Th2, N2, J2>>,
    P1: Add<P2>,
{
    type Output = ElemMat<
        <Dim<L1, M1, T1, I1, Th1, N1, J1> as DimMultiply<Dim<L2, M2, T2, I2, Th2, N2, J2>>>::Output,
        R, C,
        <P1 as Add<P2>>::Output,
    >;
    #[inline(always)]
    fn mul(self, rhs: ElemMat<Dim<L2, M2, T2, I2, Th2, N2, J2>, K, C, P2>) -> Self::Output {
        ElemMat::from_raw_unchecked(self.value * rhs.value)
    }
}

// ---- f64 scaling ----
impl<E, const R: usize, const C: usize, P> Mul<f64> for ElemMat<E, R, C, P> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: f64) -> Self { Self::from_raw_unchecked(self.value * rhs) }
}
impl<E, const R: usize, const C: usize, P> MulAssign<f64> for ElemMat<E, R, C, P> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: f64) { self.value *= rhs; }
}

// ---- Inverse ----
impl<E, const N: usize, P> ElemMat<E, N, N, P> {
    #[inline(always)]
    pub fn try_inverse(&self) -> Option<ElemMat<DimInv<E>, N, N, <Z0 as Sub<P>>::Output>>
    where
        crate::aliases::Dimensionless: DimDivide<E>,
        Z0: Sub<P>,
    {
        self.value.try_inverse().map(ElemMat::from_raw_unchecked)
    }
}

// ---- TransposeBlock ----
impl<E, const R: usize, const C: usize, P> crate::block::TransposeBlock for ElemMat<E, R, C, P>
where
    crate::aliases::Dimensionless: DimDivide<E>,
    Z0: Sub<P>,
{
    type Output = ElemMat<DimInv<E>, C, R, <Z0 as Sub<P>>::Output>;
    #[inline(always)]
    fn block_transpose(self) -> Self::Output { self.transpose() }
}
