//! Frame-tagged element-dimension matrix.
//!
//! `FrameElemMat<F, E, R, C>` is a matrix operating within coordinate frame `F`
//! where every element has SI dimension `E`.
//!
//! - `FrameElemMat<F, E, R, C> * FrameVec<F, D> → FrameVec<F, DimMul<E, D>>`
//! - Frame `F` must match between matrix and vector (compile-time checked)
//! - Output dimension is inferred from `E * input_dim`
//!
//! # No need for rescale_dims
//!
//! Unlike the old `FrameUnitMat<F, DR, DC>`, this type does not need explicit
//! dimensional reinterpretation. The same matrix object works with any input:
//!
//! ```rust,ignore
//! let inertia = FrameElemMat::<Body, MomentOfInertia, 3, 3>::from_raw_unchecked(i_raw);
//!
//! // I * ω → AngularMomentum  (MomentOfInertia * AngularVelocity)
//! let ang_mom = inertia * omega;
//!
//! // I * ω̇ → Torque  (MomentOfInertia * AngularAcceleration)
//! let torque = inertia * omega_dot;
//!
//! // I⁻¹ * τ → AngularAcceleration  ((1/MomentOfInertia) * Torque)
//! let omega_dot = inertia.try_inverse().unwrap() * torque;
//! ```
//!
//! # Gain matrices
//!
//! Control gains carry the element dimension `output_dim / input_dim`:
//!
//! - Kp (attitude): element dim = Torque (since angle is dimensionless)
//! - Kv (attitude): element dim = Torque × Time = AngularMomentum
//! - Kp (position): element dim = Force / Length (= InvTime² × Mass)
//! - Kv (position): element dim = Force / Velocity (= Mass)

use core::marker::PhantomData;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use nalgebra::SMatrix;

use crate::dim::{Dim, DimDivide, DimMultiply};
use crate::elem_mat::{DimInv, ElemMat};
use crate::frame_vec::FrameVec;
use crate::scalar::Scalar;

/// A matrix operating within coordinate frame `F` with element dimension `E`.
///
/// `FrameElemMat<F, E, 3, 3> * FrameVec<F, D> → FrameVec<F, E*D>`
#[repr(transparent)]
pub struct FrameElemMat<F, E, const R: usize, const C: usize> {
    value: SMatrix<f64, R, C>,
    _marker: PhantomData<(F, E)>,
}

impl<F, E, const R: usize, const C: usize> Clone for FrameElemMat<F, E, R, C> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl<F, E, const R: usize, const C: usize> Copy for FrameElemMat<F, E, R, C>
where
    SMatrix<f64, R, C>: Copy,
{
}

impl<F, E, const R: usize, const C: usize> PartialEq for FrameElemMat<F, E, R, C> {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<F, E, const R: usize, const C: usize> core::fmt::Debug for FrameElemMat<F, E, R, C> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "FrameElemMat({:?})", self.value)
    }
}

impl<F, E, const R: usize, const C: usize> FrameElemMat<F, E, R, C> {
    /// Create from a raw nalgebra matrix.
    #[inline(always)]
    pub fn from_raw_unchecked(value: SMatrix<f64, R, C>) -> Self {
        Self {
            value,
            _marker: PhantomData,
        }
    }

    /// Create from a frame-less ElemMat by attaching a frame tag.
    #[inline(always)]
    pub fn from_elem_mat(m: &ElemMat<E, R, C>) -> Self {
        Self::from_raw_unchecked(*m.as_raw())
    }

    /// Strip the frame tag, returning a frame-less ElemMat.
    #[inline(always)]
    pub fn to_elem_mat(&self) -> ElemMat<E, R, C> {
        ElemMat::from_raw_unchecked(self.value)
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
    #[inline(always)]
    pub fn transpose(&self) -> FrameElemMat<F, DimInv<E>, C, R>
    where
        crate::aliases::Dimensionless: DimDivide<E>,
    {
        FrameElemMat::from_raw_unchecked(self.value.transpose())
    }

    /// Zero matrix.
    #[inline(always)]
    pub fn zeros() -> Self {
        Self::from_raw_unchecked(SMatrix::zeros())
    }
}

// ---- Identity (dimensionless elements) ----

impl<F, const N: usize> FrameElemMat<F, crate::aliases::Dimensionless, N, N> {
    /// Dimensionless identity matrix in frame F.
    #[inline(always)]
    pub fn identity() -> Self {
        Self::from_raw_unchecked(SMatrix::identity())
    }
}

// ---- Same-type add/sub ----

impl<F, E, const R: usize, const C: usize> Add for FrameElemMat<F, E, R, C> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self::from_raw_unchecked(self.value + rhs.value)
    }
}

impl<F, E, const R: usize, const C: usize> Sub for FrameElemMat<F, E, R, C> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self::from_raw_unchecked(self.value - rhs.value)
    }
}

impl<F, E, const R: usize, const C: usize> Neg for FrameElemMat<F, E, R, C> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self::from_raw_unchecked(-self.value)
    }
}

impl<F, E, const R: usize, const C: usize> AddAssign for FrameElemMat<F, E, R, C> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        self.value += rhs.value;
    }
}

impl<F, E, const R: usize, const C: usize> SubAssign for FrameElemMat<F, E, R, C> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        self.value -= rhs.value;
    }
}

// ---- Matrix-FrameVec multiplication (FRAME CHECKED, DIMENSION INFERRED) ----

impl<F, LE, ME, TE, IE, ThE, NE, JE, LD, MD, TD, ID, ThD, ND, JD>
    Mul<FrameVec<F, Dim<LD, MD, TD, ID, ThD, ND, JD>>>
    for FrameElemMat<F, Dim<LE, ME, TE, IE, ThE, NE, JE>, 3, 3>
where
    Dim<LE, ME, TE, IE, ThE, NE, JE>: DimMultiply<Dim<LD, MD, TD, ID, ThD, ND, JD>>,
{
    type Output = FrameVec<
        F,
        <Dim<LE, ME, TE, IE, ThE, NE, JE> as DimMultiply<
            Dim<LD, MD, TD, ID, ThD, ND, JD>,
        >>::Output,
    >;
    #[inline(always)]
    fn mul(self, rhs: FrameVec<F, Dim<LD, MD, TD, ID, ThD, ND, JD>>) -> Self::Output {
        FrameVec::from_raw_unchecked(self.value * rhs.into_raw())
    }
}

impl<F, LE, ME, TE, IE, ThE, NE, JE, LD, MD, TD, ID, ThD, ND, JD>
    Mul<&FrameVec<F, Dim<LD, MD, TD, ID, ThD, ND, JD>>>
    for &FrameElemMat<F, Dim<LE, ME, TE, IE, ThE, NE, JE>, 3, 3>
where
    Dim<LE, ME, TE, IE, ThE, NE, JE>: DimMultiply<Dim<LD, MD, TD, ID, ThD, ND, JD>>,
{
    type Output = FrameVec<
        F,
        <Dim<LE, ME, TE, IE, ThE, NE, JE> as DimMultiply<
            Dim<LD, MD, TD, ID, ThD, ND, JD>,
        >>::Output,
    >;
    #[inline(always)]
    fn mul(self, rhs: &FrameVec<F, Dim<LD, MD, TD, ID, ThD, ND, JD>>) -> Self::Output {
        FrameVec::from_raw_unchecked(self.value * rhs.as_raw())
    }
}

// ---- Matrix-matrix multiplication (same frame, elem dims multiply) ----

impl<F, LE1, ME1, TE1, IE1, ThE1, NE1, JE1, LE2, ME2, TE2, IE2, ThE2, NE2, JE2, const R: usize, const K: usize, const C: usize>
    Mul<FrameElemMat<F, Dim<LE2, ME2, TE2, IE2, ThE2, NE2, JE2>, K, C>>
    for FrameElemMat<F, Dim<LE1, ME1, TE1, IE1, ThE1, NE1, JE1>, R, K>
where
    Dim<LE1, ME1, TE1, IE1, ThE1, NE1, JE1>: DimMultiply<Dim<LE2, ME2, TE2, IE2, ThE2, NE2, JE2>>,
{
    type Output = FrameElemMat<
        F,
        <Dim<LE1, ME1, TE1, IE1, ThE1, NE1, JE1> as DimMultiply<
            Dim<LE2, ME2, TE2, IE2, ThE2, NE2, JE2>,
        >>::Output,
        R,
        C,
    >;
    #[inline(always)]
    fn mul(self, rhs: FrameElemMat<F, Dim<LE2, ME2, TE2, IE2, ThE2, NE2, JE2>, K, C>) -> Self::Output {
        FrameElemMat::from_raw_unchecked(self.value * rhs.value)
    }
}

// ---- f64 scaling ----

impl<F, E, const R: usize, const C: usize> Mul<f64> for FrameElemMat<F, E, R, C> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: f64) -> Self {
        Self::from_raw_unchecked(self.value * rhs)
    }
}

impl<F, E, const R: usize, const C: usize> MulAssign<f64> for FrameElemMat<F, E, R, C> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: f64) {
        self.value *= rhs;
    }
}

// ---- Scalar multiplication ----

impl<F, LS, MS, TS, IS, ThS, NS, JS, LE, ME, TE, IE, ThE, NE, JE, const R: usize, const C: usize>
    Mul<FrameElemMat<F, Dim<LE, ME, TE, IE, ThE, NE, JE>, R, C>>
    for Scalar<Dim<LS, MS, TS, IS, ThS, NS, JS>>
where
    Dim<LS, MS, TS, IS, ThS, NS, JS>: DimMultiply<Dim<LE, ME, TE, IE, ThE, NE, JE>>,
{
    type Output = FrameElemMat<
        F,
        <Dim<LS, MS, TS, IS, ThS, NS, JS> as DimMultiply<
            Dim<LE, ME, TE, IE, ThE, NE, JE>,
        >>::Output,
        R,
        C,
    >;
    #[inline(always)]
    fn mul(self, rhs: FrameElemMat<F, Dim<LE, ME, TE, IE, ThE, NE, JE>, R, C>) -> Self::Output {
        FrameElemMat::from_raw_unchecked(rhs.value * self.into_raw())
    }
}

// ---- Square matrix inverse ----

impl<F, E, const N: usize> FrameElemMat<F, E, N, N> {
    /// Inverse: element dimension becomes 1/E.
    #[inline(always)]
    pub fn try_inverse(&self) -> Option<FrameElemMat<F, DimInv<E>, N, N>>
    where
        crate::aliases::Dimensionless: DimDivide<E>,
    {
        self.value
            .try_inverse()
            .map(FrameElemMat::from_raw_unchecked)
    }
}

// ---- TransposeBlock for block matrix support ----

impl<F, E, const R: usize, const C: usize> crate::block::TransposeBlock
    for FrameElemMat<F, E, R, C>
where
    crate::aliases::Dimensionless: DimDivide<E>,
{
    type Output = FrameElemMat<F, DimInv<E>, C, R>;
    #[inline(always)]
    fn block_transpose(self) -> Self::Output {
        self.transpose()
    }
}
