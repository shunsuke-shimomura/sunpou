//! Frame-tagged 3D vector with SI dimension.

use core::marker::PhantomData;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use nalgebra::Vector3;

use crate::dim::{Dim, DimMultiply};
use crate::scalar::Scalar;

/// A 3D vector tagged with coordinate frame `F` and SI dimension `D`.
///
/// Frame markers are user-defined zero-sized types. This type is independent
/// of any specific frame library (e.g. arika).
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq)]
pub struct FrameVec<F, D> {
    value: Vector3<f64>,
    _marker: PhantomData<(F, D)>,
}

impl<F, D> core::fmt::Debug for FrameVec<F, D> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "FrameVec([{}, {}, {}])",
            self.value.x, self.value.y, self.value.z
        )
    }
}

impl<F, D> core::fmt::Display for FrameVec<F, D> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "[{}, {}, {}]", self.value.x, self.value.y, self.value.z)
    }
}

impl<F, D> Default for FrameVec<F, D> {
    #[inline(always)]
    fn default() -> Self {
        Self::from_raw_unchecked(Vector3::zeros())
    }
}

impl<F, D> FrameVec<F, D> {
    /// Create from components. Caller ensures SI base units and correct frame.
    #[inline(always)]
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self {
            value: Vector3::new(x, y, z),
            _marker: PhantomData,
        }
    }

    /// Create from a raw nalgebra Vector3.
    #[inline(always)]
    pub fn from_raw_unchecked(value: Vector3<f64>) -> Self {
        Self {
            value,
            _marker: PhantomData,
        }
    }

    /// Extract the raw nalgebra Vector3.
    #[inline(always)]
    pub fn into_raw(self) -> Vector3<f64> {
        self.value
    }

    /// Borrow the raw nalgebra Vector3.
    #[inline(always)]
    pub fn as_raw(&self) -> &Vector3<f64> {
        &self.value
    }

    /// X component.
    #[inline(always)]
    pub fn x(&self) -> f64 {
        self.value.x
    }

    /// Y component.
    #[inline(always)]
    pub fn y(&self) -> f64 {
        self.value.y
    }

    /// Z component.
    #[inline(always)]
    pub fn z(&self) -> f64 {
        self.value.z
    }

    /// Euclidean norm.
    #[inline(always)]
    pub fn norm(&self) -> Scalar<D> {
        Scalar::from_raw_unchecked(self.value.norm())
    }

    /// Squared norm. Returns scalar with dimension D².
    #[inline(always)]
    pub fn norm_squared<D2>(&self) -> Scalar<D2>
    where
        D: DimMultiply<D, Output = D2>,
    {
        Scalar::from_raw_unchecked(self.value.norm_squared())
    }

    /// Convert to a frame-less UnitVec (strip the frame tag).
    #[inline(always)]
    pub fn to_unit_vec(&self) -> crate::unit_vec::UnitVec<D, 3> {
        crate::unit_vec::UnitVec::from_raw_unchecked(
            nalgebra::SVector::from([self.value.x, self.value.y, self.value.z]),
        )
    }

    /// Create from a UnitVec (add frame tag).
    #[inline(always)]
    pub fn from_unit_vec(v: &crate::unit_vec::UnitVec<D, 3>) -> Self {
        let raw = v.as_raw();
        Self::new(raw[0], raw[1], raw[2])
    }

    /// Normalize to unit length. Returns a dimensionless frame vector.
    /// Returns `None` if the vector is zero.
    #[inline(always)]
    pub fn try_normalize(
        &self,
        min_norm: f64,
    ) -> Option<FrameVec<F, crate::aliases::Dimensionless>> {
        self.value
            .try_normalize(min_norm)
            .map(FrameVec::from_raw_unchecked)
    }
}

// ---- Heterogeneous dot product (same frame required) ----

impl<F, D1> FrameVec<F, D1> {
    /// Dot product. Same frame required, dimensions may differ.
    #[inline(always)]
    pub fn dot<D2>(&self, rhs: &FrameVec<F, D2>) -> Scalar<<D1 as DimMultiply<D2>>::Output>
    where
        D1: DimMultiply<D2>,
    {
        Scalar::from_raw_unchecked(self.value.dot(&rhs.value))
    }

    /// Cross product. Same frame required, dimensions may differ.
    #[inline(always)]
    pub fn cross<D2>(
        &self,
        rhs: &FrameVec<F, D2>,
    ) -> FrameVec<F, <D1 as DimMultiply<D2>>::Output>
    where
        D1: DimMultiply<D2>,
    {
        FrameVec::from_raw_unchecked(self.value.cross(&rhs.value))
    }
}

// ---- Same-frame, same-dimension add/sub ----

impl<F, D> Add for FrameVec<F, D> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self::from_raw_unchecked(self.value + rhs.value)
    }
}

impl<F, D> Sub for FrameVec<F, D> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self::from_raw_unchecked(self.value - rhs.value)
    }
}

impl<F, D> Neg for FrameVec<F, D> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self::from_raw_unchecked(-self.value)
    }
}

// ---- Scalar multiplication (cross-dimension, preserves frame) ----

impl<F, L1, M1, T1, I1, Th1, N1, J1, L2, M2, T2, I2, Th2, N2, J2>
    Mul<FrameVec<F, Dim<L2, M2, T2, I2, Th2, N2, J2>>>
    for Scalar<Dim<L1, M1, T1, I1, Th1, N1, J1>>
where
    Dim<L1, M1, T1, I1, Th1, N1, J1>: DimMultiply<Dim<L2, M2, T2, I2, Th2, N2, J2>>,
{
    type Output = FrameVec<
        F,
        <Dim<L1, M1, T1, I1, Th1, N1, J1> as DimMultiply<
            Dim<L2, M2, T2, I2, Th2, N2, J2>,
        >>::Output,
    >;
    #[inline(always)]
    fn mul(self, rhs: FrameVec<F, Dim<L2, M2, T2, I2, Th2, N2, J2>>) -> Self::Output {
        FrameVec::from_raw_unchecked(rhs.value * self.into_raw())
    }
}

// ---- f64 scaling ----

impl<F, D> Mul<f64> for FrameVec<F, D> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: f64) -> Self {
        Self::from_raw_unchecked(self.value * rhs)
    }
}

impl<F, D> Mul<FrameVec<F, D>> for f64 {
    type Output = FrameVec<F, D>;
    #[inline(always)]
    fn mul(self, rhs: FrameVec<F, D>) -> FrameVec<F, D> {
        FrameVec::from_raw_unchecked(rhs.value * self)
    }
}

// ---- Compound assignment ----

impl<F, D> AddAssign for FrameVec<F, D> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        self.value += rhs.value;
    }
}

impl<F, D> SubAssign for FrameVec<F, D> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        self.value -= rhs.value;
    }
}

impl<F, D> MulAssign<f64> for FrameVec<F, D> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: f64) {
        self.value *= rhs;
    }
}

// ---- Reference ops ----

impl<F, D> Add for &FrameVec<F, D> {
    type Output = FrameVec<F, D>;
    #[inline(always)]
    fn add(self, rhs: Self) -> FrameVec<F, D> {
        FrameVec::from_raw_unchecked(self.value + rhs.value)
    }
}

impl<F, D> Sub for &FrameVec<F, D> {
    type Output = FrameVec<F, D>;
    #[inline(always)]
    fn sub(self, rhs: Self) -> FrameVec<F, D> {
        FrameVec::from_raw_unchecked(self.value - rhs.value)
    }
}
