//! Orbital state transition matrix (STM) using block matrices.
//!
//! Demonstrates:
//! - BlockMat2x2 for composing matrices with different unit dimensions
//! - BlockVec2 for composing vectors with different unit dimensions
//! - Type-safe state propagation: Φ * x₀ = x₁

use nalgebra::{Matrix3, SVector};
use sunpou::block::{BlockMat2x2, BlockVec2};
use sunpou::prelude::*;
use sunpou::unit_mat::UnitMat;
use sunpou::unit_vec::UnitVec;

/// Orbital state vector: [position (m), velocity (m/s)]
type OrbitalState = BlockVec2<UnitVec<Length, 3>, UnitVec<Velocity, 3>>;

/// State transition matrix for Keplerian motion (simplified).
///
/// ```text
/// Φ = | I    dt·I  |    dimensions: | dimensionless  time      |
///     | 0    I     |                | inv-time       dimensionless |
/// ```
///
/// (For this simplified example, ∂v/∂r₀ = 0, i.e. no gravity gradient)
type OrbitalStm = BlockMat2x2<
    UnitMat<Length, Length, 3, 3>,       // ∂r/∂r₀: [m/m] = dimensionless
    UnitMat<Length, Velocity, 3, 3>,     // ∂r/∂v₀: [m/(m/s)] = s
    UnitMat<Velocity, Length, 3, 3>,     // ∂v/∂r₀: [(m/s)/m] = 1/s
    UnitMat<Velocity, Velocity, 3, 3>,   // ∂v/∂v₀: [(m/s)/(m/s)] = dimensionless
>;

fn main() {
    // Initial state
    let r0 = UnitVec::<Length, 3>::from_raw_unchecked(SVector::from([7000e3, 0.0, 0.0]));
    let v0 = UnitVec::<Velocity, 3>::from_raw_unchecked(SVector::from([0.0, 7.5e3, 0.0]));
    let x0 = OrbitalState::new(r0, v0);

    println!("Initial state:");
    println!("  r₀ = {:?} m", x0.upper.as_raw().as_slice());
    println!("  v₀ = {:?} m/s", x0.lower.as_raw().as_slice());

    // Build STM for dt = 60 seconds
    let dt = 60.0;
    let stm = OrbitalStm::new(
        UnitMat::from_raw_unchecked(Matrix3::identity()),         // I
        UnitMat::from_raw_unchecked(Matrix3::identity() * dt),    // dt·I
        UnitMat::from_raw_unchecked(Matrix3::zeros()),            // 0
        UnitMat::from_raw_unchecked(Matrix3::identity()),         // I
    );

    // Propagate: x₁ = Φ * x₀
    // Type system verifies:
    //   upper: UnitMat<Length,Length> * UnitVec<Length> + UnitMat<Length,Velocity> * UnitVec<Velocity>
    //        = UnitVec<Length> + UnitVec<Length> = UnitVec<Length> ✓
    //   lower: UnitMat<Velocity,Length> * UnitVec<Length> + UnitMat<Velocity,Velocity> * UnitVec<Velocity>
    //        = UnitVec<Velocity> + UnitVec<Velocity> = UnitVec<Velocity> ✓
    let x1: OrbitalState = stm * x0;

    println!("\nAfter {} s:", dt);
    println!("  r₁ = {:?} m", x1.upper.as_raw().as_slice());
    println!("  v₁ = {:?} m/s", x1.lower.as_raw().as_slice());

    // Verify: position should advance by v₀ * dt
    let expected_r = 7000e3; // x unchanged (v₀ has no x component)
    let expected_r_y = 7.5e3 * dt; // y = vy * dt
    println!("\nExpected r₁_y = {} m", expected_r_y);
    println!("Actual   r₁_y = {} m", x1.upper.as_raw()[1]);
    assert!((x1.upper.as_raw()[0] - expected_r).abs() < 1e-10);
    assert!((x1.upper.as_raw()[1] - expected_r_y).abs() < 1e-10);

    println!("\nSTM propagation type-checked at compile time!");
}
