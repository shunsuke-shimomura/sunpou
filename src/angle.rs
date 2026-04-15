//! Angle types with rad/deg distinction at the type level.
//!
//! Both radians and degrees are physically dimensionless, but mixing them
//! is a common source of bugs. This module provides `Angle<U>` where
//! `U` is either `Rad` or `Deg`, preventing silent mixing.
//!
//! ```rust,ignore
//! use sunpou::angle::{Angle, Rad, Deg};
//!
//! let a = Angle::<Rad>::new(1.57);
//! let b = Angle::<Deg>::new(90.0);
//!
//! // a + b → compile error (Rad ≠ Deg)
//! let c = a + b.to_rad();  // OK: convert first
//! ```

use core::marker::PhantomData;
use core::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};

/// Marker: radians.
pub struct Rad;
/// Marker: degrees.
pub struct Deg;

const DEG_TO_RAD: f64 = core::f64::consts::PI / 180.0;
const RAD_TO_DEG: f64 = 180.0 / core::f64::consts::PI;

/// An angle value tagged with unit `U` (either `Rad` or `Deg`).
#[repr(transparent)]
#[derive(PartialEq, PartialOrd)]
pub struct Angle<U> {
    value: f64,
    _unit: PhantomData<U>,
}

impl<U> Clone for Angle<U> {
    #[inline(always)]
    fn clone(&self) -> Self { *self }
}
impl<U> Copy for Angle<U> {}

impl<U> core::fmt::Debug for Angle<U> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Angle({})", self.value)
    }
}

impl core::fmt::Display for Angle<Rad> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{} rad", self.value)
    }
}

impl core::fmt::Display for Angle<Deg> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{} deg", self.value)
    }
}

impl<U> Angle<U> {
    /// Create an angle from a raw f64.
    #[inline(always)]
    pub fn new(value: f64) -> Self {
        Self { value, _unit: PhantomData }
    }

    /// Extract the raw f64 value.
    #[inline(always)]
    pub fn value(self) -> f64 {
        self.value
    }

    /// Absolute value.
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self::new(if self.value < 0.0 { -self.value } else { self.value })
    }
}

impl Angle<Rad> {
    /// Convert radians to degrees.
    #[inline(always)]
    pub fn to_deg(self) -> Angle<Deg> {
        Angle::new(self.value * RAD_TO_DEG)
    }

    /// Sine.
    #[inline(always)]
    pub fn sin(self) -> f64 {
        nalgebra::ComplexField::sin(self.value)
    }

    /// Cosine.
    #[inline(always)]
    pub fn cos(self) -> f64 {
        nalgebra::ComplexField::cos(self.value)
    }

    /// Tangent.
    #[inline(always)]
    pub fn tan(self) -> f64 {
        nalgebra::ComplexField::tan(self.value)
    }
}

impl Angle<Deg> {
    /// Convert degrees to radians.
    #[inline(always)]
    pub fn to_rad(self) -> Angle<Rad> {
        Angle::new(self.value * DEG_TO_RAD)
    }
}

// ---- Arithmetic (same unit only) ----

impl<U> Add for Angle<U> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self { Self::new(self.value + rhs.value) }
}

impl<U> Sub for Angle<U> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self { Self::new(self.value - rhs.value) }
}

impl<U> Neg for Angle<U> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self { Self::new(-self.value) }
}

impl<U> AddAssign for Angle<U> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) { self.value += rhs.value; }
}

impl<U> SubAssign for Angle<U> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) { self.value -= rhs.value; }
}

impl<U> Mul<f64> for Angle<U> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: f64) -> Self { Self::new(self.value * rhs) }
}

impl<U> Div<f64> for Angle<U> {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: f64) -> Self { Self::new(self.value / rhs) }
}

impl<U> Default for Angle<U> {
    #[inline(always)]
    fn default() -> Self { Self::new(0.0) }
}
