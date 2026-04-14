//! Unit-aware scalar type.

use core::marker::PhantomData;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign, Div, DivAssign};
use crate::dim::{Dim, DimDivide, DimMultiply};

/// A scalar value tagged with SI dimension `D`.
#[repr(transparent)]
pub struct Scalar<D> {
    value: f64,
    _dim: PhantomData<D>,
}

impl<D> Clone for Scalar<D> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl<D> Copy for Scalar<D> {}

impl<D> PartialEq for Scalar<D> {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<D> PartialOrd for Scalar<D> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.value.partial_cmp(&other.value)
    }
}

impl<D> core::fmt::Debug for Scalar<D> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Scalar({})", self.value)
    }
}

impl<D> core::fmt::Display for Scalar<D> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Display::fmt(&self.value, f)
    }
}

impl<D> Default for Scalar<D> {
    #[inline(always)]
    fn default() -> Self {
        Self::from_raw_unchecked(0.0)
    }
}

impl<D> Scalar<D> {
    /// Create a scalar from a raw `f64`. The caller must ensure the value
    /// is expressed in SI base units for dimension `D`.
    #[inline(always)]
    pub fn from_raw_unchecked(value: f64) -> Self {
        Self {
            value,
            _dim: PhantomData,
        }
    }

    /// Extract the raw `f64` value.
    #[inline(always)]
    pub fn into_raw(self) -> f64 {
        self.value
    }

    /// Borrow the raw value.
    #[inline(always)]
    pub fn as_raw(&self) -> &f64 {
        &self.value
    }

    /// Absolute value, preserving dimension.
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self::from_raw_unchecked(if self.value < 0.0 { -self.value } else { self.value })
    }
}

// ---- Same-dimension add/sub ----

impl<D> Add for Scalar<D> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self::from_raw_unchecked(self.value + rhs.value)
    }
}

impl<D> Sub for Scalar<D> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self::from_raw_unchecked(self.value - rhs.value)
    }
}

impl<D> Neg for Scalar<D> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self::from_raw_unchecked(-self.value)
    }
}

// ---- Cross-dimension mul/div ----

impl<L1, M1, T1, I1, Th1, N1, J1, L2, M2, T2, I2, Th2, N2, J2>
    Mul<Scalar<Dim<L2, M2, T2, I2, Th2, N2, J2>>>
    for Scalar<Dim<L1, M1, T1, I1, Th1, N1, J1>>
where
    Dim<L1, M1, T1, I1, Th1, N1, J1>: DimMultiply<Dim<L2, M2, T2, I2, Th2, N2, J2>>,
{
    type Output = Scalar<
        <Dim<L1, M1, T1, I1, Th1, N1, J1> as DimMultiply<
            Dim<L2, M2, T2, I2, Th2, N2, J2>,
        >>::Output,
    >;
    #[inline(always)]
    fn mul(self, rhs: Scalar<Dim<L2, M2, T2, I2, Th2, N2, J2>>) -> Self::Output {
        Scalar::from_raw_unchecked(self.value * rhs.value)
    }
}

impl<L1, M1, T1, I1, Th1, N1, J1, L2, M2, T2, I2, Th2, N2, J2>
    Div<Scalar<Dim<L2, M2, T2, I2, Th2, N2, J2>>>
    for Scalar<Dim<L1, M1, T1, I1, Th1, N1, J1>>
where
    Dim<L1, M1, T1, I1, Th1, N1, J1>: DimDivide<Dim<L2, M2, T2, I2, Th2, N2, J2>>,
{
    type Output = Scalar<
        <Dim<L1, M1, T1, I1, Th1, N1, J1> as DimDivide<
            Dim<L2, M2, T2, I2, Th2, N2, J2>,
        >>::Output,
    >;
    #[inline(always)]
    fn div(self, rhs: Scalar<Dim<L2, M2, T2, I2, Th2, N2, J2>>) -> Self::Output {
        Scalar::from_raw_unchecked(self.value / rhs.value)
    }
}

// ---- Dimensionless f64 scaling ----

impl<D> Mul<f64> for Scalar<D> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: f64) -> Self {
        Self::from_raw_unchecked(self.value * rhs)
    }
}

impl<D> Div<f64> for Scalar<D> {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: f64) -> Self {
        Self::from_raw_unchecked(self.value / rhs)
    }
}

impl<D> Mul<Scalar<D>> for f64 {
    type Output = Scalar<D>;
    #[inline(always)]
    fn mul(self, rhs: Scalar<D>) -> Scalar<D> {
        Scalar::from_raw_unchecked(self * rhs.value)
    }
}

// ---- Compound assignment (same-dimension) ----

impl<D> AddAssign for Scalar<D> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        self.value += rhs.value;
    }
}

impl<D> SubAssign for Scalar<D> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        self.value -= rhs.value;
    }
}

impl<D> MulAssign<f64> for Scalar<D> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: f64) {
        self.value *= rhs;
    }
}

impl<D> DivAssign<f64> for Scalar<D> {
    #[inline(always)]
    fn div_assign(&mut self, rhs: f64) {
        self.value /= rhs;
    }
}

// ---- Reference ops ----

impl<D> Add for &Scalar<D> {
    type Output = Scalar<D>;
    #[inline(always)]
    fn add(self, rhs: Self) -> Scalar<D> {
        Scalar::from_raw_unchecked(self.value + rhs.value)
    }
}

impl<D> Sub for &Scalar<D> {
    type Output = Scalar<D>;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Scalar<D> {
        Scalar::from_raw_unchecked(self.value - rhs.value)
    }
}
