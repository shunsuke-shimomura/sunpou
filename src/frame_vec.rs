//! Frame-tagged 3D vector with SI dimension and prefix.

use core::marker::PhantomData;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use nalgebra::Vector3;
use typenum::{Integer, Z0};

use crate::dim::{Dim, DimMultiply};
use crate::scalar::Scalar;

/// A 3D vector tagged with coordinate frame `F`, SI dimension `D`, and prefix `P`.
#[repr(transparent)]
pub struct FrameVec<F, D, P = Z0> {
    value: Vector3<f64>,
    _marker: PhantomData<(F, D, P)>,
}

impl<F, D, P> Clone for FrameVec<F, D, P> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl<F, D, P> Copy for FrameVec<F, D, P> {}

impl<F, D, P> PartialEq for FrameVec<F, D, P> {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<F, D, P> core::fmt::Debug for FrameVec<F, D, P> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "FrameVec([{}, {}, {}])",
            self.value.x, self.value.y, self.value.z
        )
    }
}

impl<F, D, P> core::fmt::Display for FrameVec<F, D, P> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "[{}, {}, {}]", self.value.x, self.value.y, self.value.z)
    }
}

impl<F, D, P> Default for FrameVec<F, D, P> {
    #[inline(always)]
    fn default() -> Self {
        Self::from_raw_unchecked(Vector3::zeros())
    }
}

impl<F, D, P> FrameVec<F, D, P> {
    /// Create from components.
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

    #[inline(always)]
    pub fn x(&self) -> f64 {
        self.value.x
    }
    #[inline(always)]
    pub fn y(&self) -> f64 {
        self.value.y
    }
    #[inline(always)]
    pub fn z(&self) -> f64 {
        self.value.z
    }

    /// Euclidean norm.
    #[inline(always)]
    pub fn norm(&self) -> Scalar<D, P> {
        Scalar::from_raw_unchecked(self.value.norm())
    }

    /// Squared norm. Returns scalar with dimension D², prefix 2P.
    #[inline(always)]
    pub fn norm_squared(&self) -> Scalar<<D as DimMultiply<D>>::Output, <P as Add<P>>::Output>
    where
        D: DimMultiply<D>,
        P: Add<P>,
    {
        Scalar::from_raw_unchecked(self.value.norm_squared())
    }

    /// Convert to a frame-less UnitVec (strip the frame tag).
    #[inline(always)]
    pub fn to_unit_vec(&self) -> crate::unit_vec::UnitVec<D, 3, P> {
        crate::unit_vec::UnitVec::from_raw_unchecked(nalgebra::SVector::from([
            self.value.x,
            self.value.y,
            self.value.z,
        ]))
    }

    /// Create from a UnitVec (add frame tag).
    #[inline(always)]
    pub fn from_unit_vec(v: &crate::unit_vec::UnitVec<D, 3, P>) -> Self {
        let raw = v.as_raw();
        Self::new(raw[0], raw[1], raw[2])
    }

    /// Normalize to unit length.
    #[inline(always)]
    pub fn try_normalize(
        &self,
        min_norm: f64,
    ) -> Option<FrameVec<F, crate::aliases::Dimensionless>> {
        self.value
            .try_normalize(min_norm)
            .map(FrameVec::from_raw_unchecked)
    }

    /// Rescale to a different prefix.
    #[inline(always)]
    pub fn rescale<P2>(self) -> FrameVec<F, D, P2>
    where
        P: Sub<P2>,
        <P as Sub<P2>>::Output: Integer,
    {
        let factor = crate::prefix::pow10_i32(
            <<P as Sub<P2>>::Output as Integer>::to_i64() as i32,
        );
        FrameVec::from_raw_unchecked(self.value * factor)
    }
}

// ---- Heterogeneous dot product (same frame, cross-dim, cross-prefix) ----

impl<F, D1, P1> FrameVec<F, D1, P1> {
    /// Dot product. Same frame required, dimensions and prefixes may differ.
    #[inline(always)]
    pub fn dot<D2, P2>(
        &self,
        rhs: &FrameVec<F, D2, P2>,
    ) -> Scalar<<D1 as DimMultiply<D2>>::Output, <P1 as Add<P2>>::Output>
    where
        D1: DimMultiply<D2>,
        P1: Add<P2>,
    {
        Scalar::from_raw_unchecked(self.value.dot(&rhs.value))
    }

    /// Cross product. Same frame required, dimensions and prefixes may differ.
    #[inline(always)]
    pub fn cross<D2, P2>(
        &self,
        rhs: &FrameVec<F, D2, P2>,
    ) -> FrameVec<F, <D1 as DimMultiply<D2>>::Output, <P1 as Add<P2>>::Output>
    where
        D1: DimMultiply<D2>,
        P1: Add<P2>,
    {
        FrameVec::from_raw_unchecked(self.value.cross(&rhs.value))
    }
}

// ---- Same-frame, same-dimension, same-prefix add/sub ----

impl<F, D, P> Add for FrameVec<F, D, P> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self::from_raw_unchecked(self.value + rhs.value)
    }
}

impl<F, D, P> Sub for FrameVec<F, D, P> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self::from_raw_unchecked(self.value - rhs.value)
    }
}

impl<F, D, P> Neg for FrameVec<F, D, P> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self::from_raw_unchecked(-self.value)
    }
}

// ---- Scalar multiplication (cross-dim, cross-prefix, preserves frame) ----

impl<F, L1, M1, T1, I1, Th1, N1, J1, P1, L2, M2, T2, I2, Th2, N2, J2, P2>
    Mul<FrameVec<F, Dim<L2, M2, T2, I2, Th2, N2, J2>, P2>>
    for Scalar<Dim<L1, M1, T1, I1, Th1, N1, J1>, P1>
where
    Dim<L1, M1, T1, I1, Th1, N1, J1>: DimMultiply<Dim<L2, M2, T2, I2, Th2, N2, J2>>,
    P1: Add<P2>,
{
    type Output = FrameVec<
        F,
        <Dim<L1, M1, T1, I1, Th1, N1, J1> as DimMultiply<
            Dim<L2, M2, T2, I2, Th2, N2, J2>,
        >>::Output,
        <P1 as Add<P2>>::Output,
    >;
    #[inline(always)]
    fn mul(
        self,
        rhs: FrameVec<F, Dim<L2, M2, T2, I2, Th2, N2, J2>, P2>,
    ) -> Self::Output {
        FrameVec::from_raw_unchecked(rhs.value * self.into_raw())
    }
}

// ---- f64 scaling (preserves prefix) ----

impl<F, D, P> Mul<f64> for FrameVec<F, D, P> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: f64) -> Self {
        Self::from_raw_unchecked(self.value * rhs)
    }
}

impl<F, D, P> Mul<FrameVec<F, D, P>> for f64 {
    type Output = FrameVec<F, D, P>;
    #[inline(always)]
    fn mul(self, rhs: FrameVec<F, D, P>) -> FrameVec<F, D, P> {
        FrameVec::from_raw_unchecked(rhs.value * self)
    }
}

// ---- Compound assignment ----

impl<F, D, P> AddAssign for FrameVec<F, D, P> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        self.value += rhs.value;
    }
}

impl<F, D, P> SubAssign for FrameVec<F, D, P> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        self.value -= rhs.value;
    }
}

impl<F, D, P> MulAssign<f64> for FrameVec<F, D, P> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: f64) {
        self.value *= rhs;
    }
}

// ---- Reference ops ----

impl<F, D, P> Add for &FrameVec<F, D, P> {
    type Output = FrameVec<F, D, P>;
    #[inline(always)]
    fn add(self, rhs: Self) -> FrameVec<F, D, P> {
        FrameVec::from_raw_unchecked(self.value + rhs.value)
    }
}

impl<F, D, P> Sub for &FrameVec<F, D, P> {
    type Output = FrameVec<F, D, P>;
    #[inline(always)]
    fn sub(self, rhs: Self) -> FrameVec<F, D, P> {
        FrameVec::from_raw_unchecked(self.value - rhs.value)
    }
}
