//! Common SI dimension type aliases.

use typenum::{N1, N2, P1, P2, Z0};

use crate::dim::Dim;

// ---- Base dimensions ----
pub type Dimensionless = Dim<Z0, Z0, Z0, Z0, Z0, Z0, Z0>;
pub type Length = Dim<P1, Z0, Z0>;
pub type Mass = Dim<Z0, P1, Z0>;
pub type Time = Dim<Z0, Z0, P1>;
pub type ElectricCurrent = Dim<Z0, Z0, Z0, P1>;
pub type Temperature = Dim<Z0, Z0, Z0, Z0, P1>;
pub type AmountOfSubstance = Dim<Z0, Z0, Z0, Z0, Z0, P1>;
pub type LuminousIntensity = Dim<Z0, Z0, Z0, Z0, Z0, Z0, P1>;

// ---- Derived dimensions (mechanics) ----
pub type Area = Dim<P2, Z0, Z0>;
pub type Velocity = Dim<P1, Z0, N1>;
pub type Acceleration = Dim<P1, Z0, N2>;
pub type Force = Dim<P1, P1, N2>;
pub type Momentum = Dim<P1, P1, N1>;
pub type Energy = Dim<P2, P1, N2>;
pub type Power = Dim<P2, P1, typenum::N3>;
pub type InvTime = Dim<Z0, Z0, N1>;
pub type AngularVelocity = Dim<Z0, Z0, N1>; // rad/s (angle is dimensionless)

// ---- Derived dimensions (rotational mechanics) ----
/// Moment of inertia: kg·m²
pub type MomentOfInertia = Dim<P2, P1, Z0>;
/// Torque: kg·m²/s² = N·m
pub type Torque = Dim<P2, P1, N2>;
/// Angular momentum: kg·m²/s
pub type AngularMomentum = Dim<P2, P1, N1>;

// ---- Useful compound dimensions ----
/// Angular acceleration: rad/s² (1/s²)
pub type AngularAcceleration = Dim<Z0, Z0, N2>;

/// Length × Velocity = m²/s (specific angular momentum)
pub type LengthVelocity = Dim<P2, Z0, N1>;
