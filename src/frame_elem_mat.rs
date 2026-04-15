//! Frame-tagged element-dimension matrix with SI prefix support.

use core::marker::PhantomData;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use nalgebra::SMatrix;
use typenum::Z0;

use crate::dim::{Dim, DimDivide, DimMultiply};
use crate::elem_mat::{DimInv, ElemMat};
use crate::frame_vec::FrameVec;
use crate::scalar::Scalar;

/// Frame-tagged R×C matrix with element dimension `E` and prefix `P`.
///
/// `FrameElemMat<F, E, 3, 3, P> * FrameVec<F, D, PV> → FrameVec<F, E×D, P+PV>`
#[repr(transparent)]
pub struct FrameElemMat<F, E, const R: usize, const C: usize, P = Z0> {
    value: SMatrix<f64, R, C>,
    _marker: PhantomData<(F, E, P)>,
}

impl<F, E, const R: usize, const C: usize, P> Clone for FrameElemMat<F, E, R, C, P> {
    #[inline(always)]
    fn clone(&self) -> Self { *self }
}
impl<F, E, const R: usize, const C: usize, P> Copy for FrameElemMat<F, E, R, C, P>
where SMatrix<f64, R, C>: Copy {}
impl<F, E, const R: usize, const C: usize, P> PartialEq for FrameElemMat<F, E, R, C, P> {
    fn eq(&self, other: &Self) -> bool { self.value == other.value }
}
impl<F, E, const R: usize, const C: usize, P> core::fmt::Debug for FrameElemMat<F, E, R, C, P> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "FrameElemMat({:?})", self.value)
    }
}

impl<F, E, const R: usize, const C: usize, P> FrameElemMat<F, E, R, C, P> {
    #[inline(always)]
    pub fn from_raw(value: SMatrix<f64, R, C>) -> Self {
        Self { value, _marker: PhantomData }
    }
    #[inline(always)]
    pub fn from_elem_mat(m: &ElemMat<E, R, C, P>) -> Self {
        Self::from_raw(*m.as_raw())
    }
    #[inline(always)]
    pub fn to_elem_mat(&self) -> ElemMat<E, R, C, P> {
        ElemMat::from_raw(self.value)
    }
    #[inline(always)]
    pub fn into_raw(self) -> SMatrix<f64, R, C> { self.value }
    #[inline(always)]
    pub fn as_raw(&self) -> &SMatrix<f64, R, C> { &self.value }

    #[inline(always)]
    pub fn transpose(&self) -> FrameElemMat<F, DimInv<E>, C, R, <Z0 as Sub<P>>::Output>
    where
        crate::aliases::Dimensionless: DimDivide<E>,
        Z0: Sub<P>,
    {
        FrameElemMat::from_raw(self.value.transpose())
    }

    #[inline(always)]
    pub fn zeros() -> Self { Self::from_raw(SMatrix::zeros()) }
}

// Identity
impl<F, const N: usize> FrameElemMat<F, crate::aliases::Dimensionless, N, N> {
    #[inline(always)]
    pub fn identity() -> Self { Self::from_raw(SMatrix::identity()) }
}

// ---- Add/Sub/Neg ----
impl<F, E, const R: usize, const C: usize, P> Add for FrameElemMat<F, E, R, C, P> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self { Self::from_raw(self.value + rhs.value) }
}
impl<F, E, const R: usize, const C: usize, P> Sub for FrameElemMat<F, E, R, C, P> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self { Self::from_raw(self.value - rhs.value) }
}
impl<F, E, const R: usize, const C: usize, P> Neg for FrameElemMat<F, E, R, C, P> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self { Self::from_raw(-self.value) }
}
impl<F, E, const R: usize, const C: usize, P> AddAssign for FrameElemMat<F, E, R, C, P> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) { self.value += rhs.value; }
}
impl<F, E, const R: usize, const C: usize, P> SubAssign for FrameElemMat<F, E, R, C, P> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) { self.value -= rhs.value; }
}

// ---- Mat * FrameVec (frame checked, dim+prefix inferred) ----
impl<F, LE, ME, TE, IE, ThE, NE, JE, PM, LD, MD, TD, ID, ThD, ND, JD, PV>
    Mul<FrameVec<F, Dim<LD, MD, TD, ID, ThD, ND, JD>, PV>>
    for FrameElemMat<F, Dim<LE, ME, TE, IE, ThE, NE, JE>, 3, 3, PM>
where
    Dim<LE, ME, TE, IE, ThE, NE, JE>: DimMultiply<Dim<LD, MD, TD, ID, ThD, ND, JD>>,
    PM: Add<PV>,
{
    type Output = FrameVec<
        F,
        <Dim<LE, ME, TE, IE, ThE, NE, JE> as DimMultiply<Dim<LD, MD, TD, ID, ThD, ND, JD>>>::Output,
        <PM as Add<PV>>::Output,
    >;
    #[inline(always)]
    fn mul(self, rhs: FrameVec<F, Dim<LD, MD, TD, ID, ThD, ND, JD>, PV>) -> Self::Output {
        FrameVec::from_raw(self.value * rhs.into_raw())
    }
}

impl<F, LE, ME, TE, IE, ThE, NE, JE, PM, LD, MD, TD, ID, ThD, ND, JD, PV>
    Mul<&FrameVec<F, Dim<LD, MD, TD, ID, ThD, ND, JD>, PV>>
    for &FrameElemMat<F, Dim<LE, ME, TE, IE, ThE, NE, JE>, 3, 3, PM>
where
    Dim<LE, ME, TE, IE, ThE, NE, JE>: DimMultiply<Dim<LD, MD, TD, ID, ThD, ND, JD>>,
    PM: Add<PV>,
{
    type Output = FrameVec<
        F,
        <Dim<LE, ME, TE, IE, ThE, NE, JE> as DimMultiply<Dim<LD, MD, TD, ID, ThD, ND, JD>>>::Output,
        <PM as Add<PV>>::Output,
    >;
    #[inline(always)]
    fn mul(self, rhs: &FrameVec<F, Dim<LD, MD, TD, ID, ThD, ND, JD>, PV>) -> Self::Output {
        FrameVec::from_raw(self.value * rhs.as_raw())
    }
}

// ---- Mat * Mat (same frame, dims multiply, prefixes add) ----
impl<F, LE1, ME1, TE1, IE1, ThE1, NE1, JE1, P1, LE2, ME2, TE2, IE2, ThE2, NE2, JE2, P2, const R: usize, const K: usize, const C: usize>
    Mul<FrameElemMat<F, Dim<LE2, ME2, TE2, IE2, ThE2, NE2, JE2>, K, C, P2>>
    for FrameElemMat<F, Dim<LE1, ME1, TE1, IE1, ThE1, NE1, JE1>, R, K, P1>
where
    Dim<LE1, ME1, TE1, IE1, ThE1, NE1, JE1>: DimMultiply<Dim<LE2, ME2, TE2, IE2, ThE2, NE2, JE2>>,
    P1: Add<P2>,
{
    type Output = FrameElemMat<
        F,
        <Dim<LE1, ME1, TE1, IE1, ThE1, NE1, JE1> as DimMultiply<Dim<LE2, ME2, TE2, IE2, ThE2, NE2, JE2>>>::Output,
        R, C,
        <P1 as Add<P2>>::Output,
    >;
    #[inline(always)]
    fn mul(self, rhs: FrameElemMat<F, Dim<LE2, ME2, TE2, IE2, ThE2, NE2, JE2>, K, C, P2>) -> Self::Output {
        FrameElemMat::from_raw(self.value * rhs.value)
    }
}

// ---- f64 scaling ----
impl<F, E, const R: usize, const C: usize, P> Mul<f64> for FrameElemMat<F, E, R, C, P> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: f64) -> Self { Self::from_raw(self.value * rhs) }
}
impl<F, E, const R: usize, const C: usize, P> MulAssign<f64> for FrameElemMat<F, E, R, C, P> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: f64) { self.value *= rhs; }
}

// ---- Scalar * Mat ----
impl<F, LS, MS, TS, IS, ThS, NS, JS, PS, LE, ME, TE, IE, ThE, NE, JE, PM, const R: usize, const C: usize>
    Mul<FrameElemMat<F, Dim<LE, ME, TE, IE, ThE, NE, JE>, R, C, PM>>
    for Scalar<Dim<LS, MS, TS, IS, ThS, NS, JS>, PS>
where
    Dim<LS, MS, TS, IS, ThS, NS, JS>: DimMultiply<Dim<LE, ME, TE, IE, ThE, NE, JE>>,
    PS: Add<PM>,
{
    type Output = FrameElemMat<
        F,
        <Dim<LS, MS, TS, IS, ThS, NS, JS> as DimMultiply<Dim<LE, ME, TE, IE, ThE, NE, JE>>>::Output,
        R, C,
        <PS as Add<PM>>::Output,
    >;
    #[inline(always)]
    fn mul(self, rhs: FrameElemMat<F, Dim<LE, ME, TE, IE, ThE, NE, JE>, R, C, PM>) -> Self::Output {
        FrameElemMat::from_raw(rhs.value * self.into_raw())
    }
}

// ---- Inverse ----
impl<F, E, const N: usize, P> FrameElemMat<F, E, N, N, P> {
    #[inline(always)]
    pub fn try_inverse(&self) -> Option<FrameElemMat<F, DimInv<E>, N, N, <Z0 as Sub<P>>::Output>>
    where
        crate::aliases::Dimensionless: DimDivide<E>,
        Z0: Sub<P>,
    {
        self.value.try_inverse().map(FrameElemMat::from_raw)
    }
}

// ---- TransposeBlock ----
impl<F, E, const R: usize, const C: usize, P> crate::block::TransposeBlock for FrameElemMat<F, E, R, C, P>
where
    crate::aliases::Dimensionless: DimDivide<E>,
    Z0: Sub<P>,
{
    type Output = FrameElemMat<F, DimInv<E>, C, R, <Z0 as Sub<P>>::Output>;
    #[inline(always)]
    fn block_transpose(self) -> Self::Output { self.transpose() }
}
