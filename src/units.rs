//! Unit literal extension methods and display names.
//!
//! Provides ergonomic construction of unit-tagged scalars via methods on `f64`:
//!
//! ```rust,ignore
//! use sunpou::units::UnitLiteral;
//!
//! let d = 7.0.km();        // Scalar<Length, Kilo>
//! let v = 7.5.km_per_s();  // Scalar<Velocity, Kilo>
//! let m = 100.0.kg();      // Scalar<Mass>
//! let t = 60.0.s();        // Scalar<Time>
//! let f = 980.0.n();       // Scalar<Force>  (newtons)
//! ```

use crate::aliases::*;
use crate::prefix;
use crate::scalar::Scalar;

/// Extension trait for `f64` providing unit literal constructors.
pub trait UnitLiteral {
    // ---- Length ----
    /// Meters (SI base).
    fn m(self) -> Scalar<Length>;
    /// Kilometers.
    fn km(self) -> Scalar<Length, prefix::Kilo>;
    /// Millimeters.
    fn mm(self) -> Scalar<Length, prefix::Milli>;

    // ---- Mass ----
    /// Kilograms (SI base).
    fn kg(self) -> Scalar<Mass>;

    // ---- Time ----
    /// Seconds (SI base).
    fn s(self) -> Scalar<Time>;
    /// Milliseconds.
    fn ms(self) -> Scalar<Time, prefix::Milli>;

    // ---- Velocity ----
    /// Meters per second.
    fn m_per_s(self) -> Scalar<Velocity>;
    /// Kilometers per second.
    fn km_per_s(self) -> Scalar<Velocity, prefix::Kilo>;

    // ---- Acceleration ----
    /// Meters per second squared.
    fn m_per_s2(self) -> Scalar<Acceleration>;

    // ---- Force ----
    /// Newtons.
    fn n(self) -> Scalar<Force>;
    /// Kilonewtons.
    fn kn(self) -> Scalar<Force, prefix::Kilo>;

    // ---- Energy ----
    /// Joules.
    fn j(self) -> Scalar<Energy>;

    // ---- Torque ----
    /// Newton-meters.
    fn nm(self) -> Scalar<Torque>;

    // ---- Angular velocity ----
    /// Radians per second.
    fn rad_per_s(self) -> Scalar<AngularVelocity>;
}

impl UnitLiteral for f64 {
    #[inline(always)]
    fn m(self) -> Scalar<Length> { Scalar::from_raw(self) }
    #[inline(always)]
    fn km(self) -> Scalar<Length, prefix::Kilo> { Scalar::from_raw(self) }
    #[inline(always)]
    fn mm(self) -> Scalar<Length, prefix::Milli> { Scalar::from_raw(self) }

    #[inline(always)]
    fn kg(self) -> Scalar<Mass> { Scalar::from_raw(self) }

    #[inline(always)]
    fn s(self) -> Scalar<Time> { Scalar::from_raw(self) }
    #[inline(always)]
    fn ms(self) -> Scalar<Time, prefix::Milli> { Scalar::from_raw(self) }

    #[inline(always)]
    fn m_per_s(self) -> Scalar<Velocity> { Scalar::from_raw(self) }
    #[inline(always)]
    fn km_per_s(self) -> Scalar<Velocity, prefix::Kilo> { Scalar::from_raw(self) }

    #[inline(always)]
    fn m_per_s2(self) -> Scalar<Acceleration> { Scalar::from_raw(self) }

    #[inline(always)]
    fn n(self) -> Scalar<Force> { Scalar::from_raw(self) }
    #[inline(always)]
    fn kn(self) -> Scalar<Force, prefix::Kilo> { Scalar::from_raw(self) }

    #[inline(always)]
    fn j(self) -> Scalar<Energy> { Scalar::from_raw(self) }

    #[inline(always)]
    fn nm(self) -> Scalar<Torque> { Scalar::from_raw(self) }

    #[inline(always)]
    fn rad_per_s(self) -> Scalar<AngularVelocity> { Scalar::from_raw(self) }
}
