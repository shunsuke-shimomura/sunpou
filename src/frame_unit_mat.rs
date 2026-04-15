//! Frame-tagged unit-aware matrix type.
//!
//! `FrameUnitMat<F, DR, DC, R, C>` is a matrix that operates **within** a single
//! coordinate frame `F`. It can only multiply `FrameVec<F, DC>` (same frame),
//! producing `FrameVec<F, DR>`.
//!
//! This prevents accidental application of an ECI-defined STM to an ECEF state
//! vector, which would silently produce wrong results.
//!
//! # Relationship to other types
//!
//! - `UnitMat<DR, DC, R, C>` — frame-less, for pure math or frame-agnostic use
//! - `FrameUnitMat<F, DR, DC, R, C>` — frame-tagged, for within-frame operations
//! - `Rotation<F1, F2>` — frame transformation (between different frames)
//!
//! # Gain matrices
//!
//! Control gains like Kp (position gain) and Kv (velocity gain) are matrices
//! that convert one physical quantity to another within the same frame.
//! For example, a PD attitude controller:
//!
//! ```text
//! τ = -Kp · θ - Kv · ω
//! ```
//!
//! where τ is torque [N·m], θ is angle [rad] (dimensionless), ω is angular
//! velocity [rad/s]. The gains carry the unit conversion:
//!
//! - Kp: [N·m / rad] = [N·m] → `FrameUnitMat<Body, Torque, Dimensionless, 3, 3>`
//! - Kv: [N·m / (rad/s)] = [N·m·s] → `FrameUnitMat<Body, Torque, InvTime, 3, 3>`
//!
//! The "gain has no units" intuition is misleading — the gain's units are
//! precisely `output_dim / input_dim`, and sunpou tracks this.

use core::marker::PhantomData;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use nalgebra::SMatrix;

use crate::dim::{Dim, DimMultiply};
use crate::frame_vec::FrameVec;
use crate::scalar::Scalar;

/// A matrix operating within coordinate frame `F`.
///
/// `FrameUnitMat<F, DR, DC, R, C> * FrameVec<F, DC> → FrameVec<F, DR>`
///
/// The frame `F` must match between matrix and vector — mismatched frames
/// produce a compile error.
#[repr(transparent)]
pub struct FrameUnitMat<F, DR, DC, const R: usize, const C: usize> {
    value: SMatrix<f64, R, C>,
    _marker: PhantomData<(F, DR, DC)>,
}

impl<F, DR, DC, const R: usize, const C: usize> Clone for FrameUnitMat<F, DR, DC, R, C> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<F, DR, DC, const R: usize, const C: usize> Copy for FrameUnitMat<F, DR, DC, R, C>
where
    SMatrix<f64, R, C>: Copy,
{
}

impl<F, DR, DC, const R: usize, const C: usize> PartialEq for FrameUnitMat<F, DR, DC, R, C> {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<F, DR, DC, const R: usize, const C: usize> core::fmt::Debug
    for FrameUnitMat<F, DR, DC, R, C>
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "FrameUnitMat({:?})", self.value)
    }
}

impl<F, DR, DC, const R: usize, const C: usize> FrameUnitMat<F, DR, DC, R, C> {
    /// Create from a raw nalgebra matrix. Caller ensures correct frame and units.
    #[inline(always)]
    pub fn from_raw_unchecked(value: SMatrix<f64, R, C>) -> Self {
        Self {
            value,
            _marker: PhantomData,
        }
    }

    /// Create from a frame-less UnitMat by attaching a frame tag.
    #[inline(always)]
    pub fn from_unit_mat(m: &crate::unit_mat::UnitMat<DR, DC, R, C>) -> Self {
        Self::from_raw_unchecked(*m.as_raw())
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

    /// Strip the frame tag, returning a frame-less UnitMat.
    #[inline(always)]
    pub fn to_unit_mat(&self) -> crate::unit_mat::UnitMat<DR, DC, R, C> {
        crate::unit_mat::UnitMat::from_raw_unchecked(self.value)
    }

    /// Transpose: `FrameUnitMat<F, DR, DC, R, C> → FrameUnitMat<F, DC, DR, C, R>`.
    /// Frame is preserved.
    #[inline(always)]
    pub fn transpose(&self) -> FrameUnitMat<F, DC, DR, C, R> {
        FrameUnitMat::from_raw_unchecked(self.value.transpose())
    }

    /// Zero matrix.
    #[inline(always)]
    pub fn zeros() -> Self {
        Self::from_raw_unchecked(SMatrix::zeros())
    }

    /// Rescale both row and column dimensions by the same factor `S`.
    ///
    /// This reinterprets the matrix without changing any numerical values.
    /// The element dimension `DR / DC` is preserved because both are scaled
    /// by the same factor: `(DR * S) / (DC * S) = DR / DC`.
    ///
    /// # Example: inertia tensor reinterpretation
    ///
    /// The inertia tensor I [kg·m²] can be interpreted as:
    /// - `I · ω = L`: maps AngularVelocity → AngularMomentum
    /// - `I · ω̇ = τ`: maps AngularAcceleration → Torque
    ///
    /// These share the same element dimension (MomentOfInertia), but differ
    /// by a factor of InvTime on both row and column dimensions:
    ///
    /// ```text
    /// AngularMomentum = Torque × Time        (row: ×InvTime)
    /// AngularVelocity = AngularAcceleration × Time  (col: ×InvTime)
    /// ```
    ///
    /// So `rescale_dims::<InvTime>()` converts between interpretations:
    ///
    /// ```rust,ignore
    /// let i_vel: FrameUnitMat<Body, AngularMomentum, AngularVelocity, 3, 3> = ...;
    /// let i_acc = i_vel.rescale_dims::<InvTime>();
    /// // i_acc: FrameUnitMat<Body, Torque, AngularAcceleration, 3, 3>
    /// ```
    #[inline(always)]
    pub fn rescale_dims<S>(
        self,
    ) -> FrameUnitMat<F, <DR as DimMultiply<S>>::Output, <DC as DimMultiply<S>>::Output, R, C>
    where
        DR: DimMultiply<S>,
        DC: DimMultiply<S>,
    {
        FrameUnitMat::from_raw_unchecked(self.value)
    }
}

// ---- Identity (square, same dim) ----

impl<F, D, const N: usize> FrameUnitMat<F, D, D, N, N> {
    /// Identity matrix. Only available when DR == DC.
    #[inline(always)]
    pub fn identity() -> Self {
        Self::from_raw_unchecked(SMatrix::identity())
    }
}

// ---- Same-type add/sub ----

impl<F, DR, DC, const R: usize, const C: usize> Add for FrameUnitMat<F, DR, DC, R, C> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self::from_raw_unchecked(self.value + rhs.value)
    }
}

impl<F, DR, DC, const R: usize, const C: usize> Sub for FrameUnitMat<F, DR, DC, R, C> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self::from_raw_unchecked(self.value - rhs.value)
    }
}

impl<F, DR, DC, const R: usize, const C: usize> Neg for FrameUnitMat<F, DR, DC, R, C> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self::from_raw_unchecked(-self.value)
    }
}

impl<F, DR, DC, const R: usize, const C: usize> AddAssign for FrameUnitMat<F, DR, DC, R, C> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        self.value += rhs.value;
    }
}

impl<F, DR, DC, const R: usize, const C: usize> SubAssign for FrameUnitMat<F, DR, DC, R, C> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        self.value -= rhs.value;
    }
}

// ---- Matrix-FrameVec multiplication (FRAME CHECKED) ----
// FrameUnitMat<F, DR, DC, R, C> * FrameVec<F, DC> → FrameVec<F, DR>
// The frame F must match!

impl<F, DR, DC> Mul<FrameVec<F, DC>> for FrameUnitMat<F, DR, DC, 3, 3> {
    type Output = FrameVec<F, DR>;
    #[inline(always)]
    fn mul(self, rhs: FrameVec<F, DC>) -> FrameVec<F, DR> {
        FrameVec::from_raw_unchecked(self.value * rhs.into_raw())
    }
}

impl<F, DR, DC> Mul<&FrameVec<F, DC>> for &FrameUnitMat<F, DR, DC, 3, 3> {
    type Output = FrameVec<F, DR>;
    #[inline(always)]
    fn mul(self, rhs: &FrameVec<F, DC>) -> FrameVec<F, DR> {
        FrameVec::from_raw_unchecked(self.value * rhs.as_raw())
    }
}

// ---- Matrix-matrix multiplication (same frame) ----
// FrameUnitMat<F, DR, DM, R, K> * FrameUnitMat<F, DM, DC, K, C> → FrameUnitMat<F, DR, DC, R, C>

impl<F, DR, DM, DC, const R: usize, const K: usize, const C: usize>
    Mul<FrameUnitMat<F, DM, DC, K, C>> for FrameUnitMat<F, DR, DM, R, K>
{
    type Output = FrameUnitMat<F, DR, DC, R, C>;
    #[inline(always)]
    fn mul(self, rhs: FrameUnitMat<F, DM, DC, K, C>) -> FrameUnitMat<F, DR, DC, R, C> {
        FrameUnitMat::from_raw_unchecked(self.value * rhs.value)
    }
}

impl<F, DR, DM, DC, const R: usize, const K: usize, const C: usize>
    Mul<&FrameUnitMat<F, DM, DC, K, C>> for &FrameUnitMat<F, DR, DM, R, K>
{
    type Output = FrameUnitMat<F, DR, DC, R, C>;
    #[inline(always)]
    fn mul(self, rhs: &FrameUnitMat<F, DM, DC, K, C>) -> FrameUnitMat<F, DR, DC, R, C> {
        FrameUnitMat::from_raw_unchecked(self.value * rhs.value)
    }
}

// ---- f64 scaling ----

impl<F, DR, DC, const R: usize, const C: usize> Mul<f64> for FrameUnitMat<F, DR, DC, R, C> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: f64) -> Self {
        Self::from_raw_unchecked(self.value * rhs)
    }
}

impl<F, DR, DC, const R: usize, const C: usize> MulAssign<f64>
    for FrameUnitMat<F, DR, DC, R, C>
{
    #[inline(always)]
    fn mul_assign(&mut self, rhs: f64) {
        self.value *= rhs;
    }
}

// ---- Scalar multiplication (cross-dimension, preserves frame) ----

impl<F, LS, MS, TS, IS, ThS, NS, JS, LR, MR, TR, IR, ThR, NR, JR, DC, const R: usize, const C: usize>
    Mul<FrameUnitMat<F, Dim<LR, MR, TR, IR, ThR, NR, JR>, DC, R, C>>
    for Scalar<Dim<LS, MS, TS, IS, ThS, NS, JS>>
where
    Dim<LS, MS, TS, IS, ThS, NS, JS>: DimMultiply<Dim<LR, MR, TR, IR, ThR, NR, JR>>,
{
    type Output = FrameUnitMat<
        F,
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
        rhs: FrameUnitMat<F, Dim<LR, MR, TR, IR, ThR, NR, JR>, DC, R, C>,
    ) -> Self::Output {
        FrameUnitMat::from_raw_unchecked(rhs.value * self.into_raw())
    }
}

// ---- Square matrix inverse ----

impl<F, DR, DC, const N: usize> FrameUnitMat<F, DR, DC, N, N> {
    /// Inverse: `FrameUnitMat<F, DR, DC, N, N>⁻¹ → FrameUnitMat<F, DC, DR, N, N>`.
    /// Frame is preserved.
    #[inline(always)]
    pub fn try_inverse(&self) -> Option<FrameUnitMat<F, DC, DR, N, N>> {
        self.value
            .try_inverse()
            .map(FrameUnitMat::from_raw_unchecked)
    }
}

// ---- TransposeBlock for block matrix support ----

impl<F, DR, DC, const R: usize, const C: usize> crate::block::TransposeBlock
    for FrameUnitMat<F, DR, DC, R, C>
{
    type Output = FrameUnitMat<F, DC, DR, C, R>;
    #[inline(always)]
    fn block_transpose(self) -> Self::Output {
        self.transpose()
    }
}
