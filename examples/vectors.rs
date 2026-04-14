//! Vector operations with heterogeneous-dimension dot and cross products.
//!
//! Demonstrates:
//! - UnitVec with dimension tagging
//! - Same-dimension and cross-dimension dot product
//! - Cross product producing correct output dimension

use nalgebra::SVector;
use uolgebra::prelude::*;
use uolgebra::scalar::Scalar;
use uolgebra::unit_vec::UnitVec;

fn main() {
    // Position and velocity vectors
    let pos = UnitVec::<Length, 3>::from_raw_unchecked(SVector::from([7000e3, 0.0, 0.0]));
    let vel = UnitVec::<Velocity, 3>::from_raw_unchecked(SVector::from([0.0, 7.5e3, 0.0]));

    // Same-dimension addition
    let pos2 = UnitVec::<Length, 3>::from_raw_unchecked(SVector::from([0.0, 100.0, 0.0]));
    let total_pos = pos + pos2;
    println!("pos + offset = {:?}", total_pos.as_raw().as_slice());

    // Heterogeneous cross product: r × v → specific angular momentum (m²/s)
    let h: UnitVec<LengthVelocity, 3> = pos.cross(&vel);
    println!("h = r × v = {:?} m²/s", h.as_raw().as_slice());

    // Dot product: force · displacement → energy
    let force = UnitVec::<Force, 3>::from_raw_unchecked(SVector::from([10.0, 0.0, 0.0]));
    let disp = UnitVec::<Length, 3>::from_raw_unchecked(SVector::from([5.0, 0.0, 0.0]));
    let work: Scalar<Energy> = force.dot(&disp);
    println!("W = F · d = {} J", work.into_raw());

    // Scalar × vector (cross-dimension)
    let mass = Scalar::<Mass>::from_raw_unchecked(10.0);
    let accel = UnitVec::<Acceleration, 3>::from_raw_unchecked(SVector::from([0.0, 0.0, 9.8]));
    let force_vec: UnitVec<Force, 3> = mass * accel;
    println!("F = m * a = {:?} N", force_vec.as_raw().as_slice());

    // Norm
    println!("|pos| = {} m", pos.norm().into_raw());

    // The following would NOT compile:
    // let _ = pos + vel;  // Length + Velocity → compile error

    println!("\nAll vector operations type-checked at compile time!");
}
