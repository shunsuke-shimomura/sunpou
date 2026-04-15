//! Unit-aware scalar type with SI prefix support.
//!
//! `Scalar<D, P>` stores a value in units of `10^P × [SI base unit for D]`.
//! The prefix `P` defaults to `Z0` (base SI), so existing code is unaffected.
//!
//! When multiplying/dividing, prefixes combine: P1+P2 / P1-P2.
//! Addition requires matching prefix (compile-time enforced), preventing
//! accidental loss of precision from scale mismatch.

use core::marker::PhantomData;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use typenum::{Integer, Z0};

use crate::dim::{Dim, DimDivide, DimMultiply};

/// A scalar value tagged with SI dimension `D` and prefix `P` (power of 10).
///
/// The internal `f64` represents `value × 10^P` in SI base units.
/// For example, `Scalar<Length, P3>` with value 7.0 means 7.0 km = 7000 m.
///
/// `P` defaults to `Z0` (no prefix = SI base units) for backward compatibility.
#[repr(transparent)]
pub struct Scalar<D, P = Z0> {
    value: f64,
    _marker: PhantomData<(D, P)>,
}

impl<D, P> Clone for Scalar<D, P> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl<D, P> Copy for Scalar<D, P> {}

impl<D, P> PartialEq for Scalar<D, P> {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<D, P> PartialOrd for Scalar<D, P> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.value.partial_cmp(&other.value)
    }
}

impl<D, P> core::fmt::Debug for Scalar<D, P> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Scalar({})", self.value)
    }
}

impl<D, P> core::fmt::Display for Scalar<D, P> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Display::fmt(&self.value, f)
    }
}

impl<D, P> Default for Scalar<D, P> {
    #[inline(always)]
    fn default() -> Self {
        Self::from_raw(0.0)
    }
}

impl<D, P> Scalar<D, P> {
    /// Create a scalar from a raw `f64` in the prefixed unit system.
    ///
    /// For `Scalar<Length, Kilo>`, the value is in kilometers.
    /// For `Scalar<Length, Z0>` (or just `Scalar<Length>`), the value is in meters.
    #[inline(always)]
    pub fn from_raw(value: f64) -> Self {
        Self {
            value,
            _marker: PhantomData,
        }
    }

    /// Extract the raw `f64` value (in the prefixed unit).
    #[inline(always)]
    pub fn into_raw(self) -> f64 {
        self.value
    }

    /// Borrow the raw value.
    #[inline(always)]
    pub fn as_raw(&self) -> &f64 {
        &self.value
    }

    /// Absolute value, preserving dimension and prefix.
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self::from_raw(if self.value < 0.0 {
            -self.value
        } else {
            self.value
        })
    }

    /// Convert the value to SI base units (prefix = Z0).
    ///
    /// Returns `self.value * 10^P`.
    #[inline(always)]
    pub fn to_base_value(self) -> f64
    where
        P: Integer,
    {
        self.value * crate::prefix::pow10_i32(P::to_i64() as i32)
    }

    /// Rescale to a different prefix. Multiplies internal value by `10^(P - P2)`.
    ///
    /// ```rust,ignore
    /// let km = Scalar::<Length, Kilo>::from_raw(7.0); // 7 km
    /// let m: Scalar<Length, Base> = km.rescale();               // 7000 m
    /// ```
    #[inline(always)]
    pub fn rescale<P2>(self) -> Scalar<D, P2>
    where
        P: Sub<P2>,
        <P as Sub<P2>>::Output: Integer,
    {
        let exp = <<P as Sub<P2>>::Output as Integer>::to_i64() as i32;
        Scalar::from_raw(self.value * crate::prefix::pow10_i32(exp))
    }
}

// ---- Same-dimension, same-prefix add/sub ----

impl<D, P> Add for Scalar<D, P> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self::from_raw(self.value + rhs.value)
    }
}

impl<D, P> Sub for Scalar<D, P> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self::from_raw(self.value - rhs.value)
    }
}

impl<D, P> Neg for Scalar<D, P> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self::from_raw(-self.value)
    }
}

// ---- Cross-dimension, cross-prefix mul/div ----

impl<L1, M1, T1, I1, Th1, N1, J1, P1, L2, M2, T2, I2, Th2, N2, J2, P2>
    Mul<Scalar<Dim<L2, M2, T2, I2, Th2, N2, J2>, P2>>
    for Scalar<Dim<L1, M1, T1, I1, Th1, N1, J1>, P1>
where
    Dim<L1, M1, T1, I1, Th1, N1, J1>: DimMultiply<Dim<L2, M2, T2, I2, Th2, N2, J2>>,
    P1: Add<P2>,
{
    type Output = Scalar<
        <Dim<L1, M1, T1, I1, Th1, N1, J1> as DimMultiply<
            Dim<L2, M2, T2, I2, Th2, N2, J2>,
        >>::Output,
        <P1 as Add<P2>>::Output,
    >;
    #[inline(always)]
    fn mul(self, rhs: Scalar<Dim<L2, M2, T2, I2, Th2, N2, J2>, P2>) -> Self::Output {
        Scalar::from_raw(self.value * rhs.value)
    }
}

impl<L1, M1, T1, I1, Th1, N1, J1, P1, L2, M2, T2, I2, Th2, N2, J2, P2>
    Div<Scalar<Dim<L2, M2, T2, I2, Th2, N2, J2>, P2>>
    for Scalar<Dim<L1, M1, T1, I1, Th1, N1, J1>, P1>
where
    Dim<L1, M1, T1, I1, Th1, N1, J1>: DimDivide<Dim<L2, M2, T2, I2, Th2, N2, J2>>,
    P1: Sub<P2>,
{
    type Output = Scalar<
        <Dim<L1, M1, T1, I1, Th1, N1, J1> as DimDivide<
            Dim<L2, M2, T2, I2, Th2, N2, J2>,
        >>::Output,
        <P1 as Sub<P2>>::Output,
    >;
    #[inline(always)]
    fn div(self, rhs: Scalar<Dim<L2, M2, T2, I2, Th2, N2, J2>, P2>) -> Self::Output {
        Scalar::from_raw(self.value / rhs.value)
    }
}

// ---- Dimensionless f64 scaling (preserves prefix) ----

impl<D, P> Mul<f64> for Scalar<D, P> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: f64) -> Self {
        Self::from_raw(self.value * rhs)
    }
}

impl<D, P> Div<f64> for Scalar<D, P> {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: f64) -> Self {
        Self::from_raw(self.value / rhs)
    }
}

impl<D, P> Mul<Scalar<D, P>> for f64 {
    type Output = Scalar<D, P>;
    #[inline(always)]
    fn mul(self, rhs: Scalar<D, P>) -> Scalar<D, P> {
        Scalar::from_raw(self * rhs.value)
    }
}

// ---- Compound assignment (same-dimension, same-prefix) ----

impl<D, P> AddAssign for Scalar<D, P> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        self.value += rhs.value;
    }
}

impl<D, P> SubAssign for Scalar<D, P> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        self.value -= rhs.value;
    }
}

impl<D, P> MulAssign<f64> for Scalar<D, P> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: f64) {
        self.value *= rhs;
    }
}

impl<D, P> DivAssign<f64> for Scalar<D, P> {
    #[inline(always)]
    fn div_assign(&mut self, rhs: f64) {
        self.value /= rhs;
    }
}

// ---- Reference ops ----

impl<D, P> Add for &Scalar<D, P> {
    type Output = Scalar<D, P>;
    #[inline(always)]
    fn add(self, rhs: Self) -> Scalar<D, P> {
        Scalar::from_raw(self.value + rhs.value)
    }
}

impl<D, P> Sub for &Scalar<D, P> {
    type Output = Scalar<D, P>;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Scalar<D, P> {
        Scalar::from_raw(self.value - rhs.value)
    }
}

// ---- Dimensionless Scalar from f64 (base prefix only) ----

impl From<f64> for Scalar<crate::aliases::Dimensionless> {
    #[inline(always)]
    fn from(value: f64) -> Self {
        Self::from_raw(value)
    }
}

impl From<Scalar<crate::aliases::Dimensionless>> for f64 {
    #[inline(always)]
    fn from(s: Scalar<crate::aliases::Dimensionless>) -> f64 {
        s.value
    }
}
