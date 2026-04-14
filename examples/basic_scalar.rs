//! Basic scalar operations with compile-time unit checking.
//!
//! Demonstrates:
//! - Creating unit-tagged scalars
//! - Arithmetic that produces correct output dimensions
//! - Compile-time prevention of invalid operations

use uolgebra::prelude::*;
use uolgebra::scalar::Scalar;

fn main() {
    // Newton's second law: F = m * a
    let mass = Scalar::<Mass>::from_raw_unchecked(100.0); // 100 kg
    let accel = Scalar::<Acceleration>::from_raw_unchecked(9.8); // 9.8 m/s²
    let force: Scalar<Force> = mass * accel; // 980 N
    println!("F = m * a = {} N", force.into_raw());

    // Velocity from distance / time
    let distance = Scalar::<Length>::from_raw_unchecked(1000.0); // 1000 m
    let time = Scalar::<Time>::from_raw_unchecked(10.0); // 10 s
    let velocity: Scalar<Velocity> = distance / time; // 100 m/s
    println!("v = d / t = {} m/s", velocity.into_raw());

    // Energy: W = F * d
    let work: Scalar<Energy> = force * distance;
    println!("W = F * d = {} J", work.into_raw());

    // Same-dimension addition
    let d1 = Scalar::<Length>::from_raw_unchecked(3.0);
    let d2 = Scalar::<Length>::from_raw_unchecked(4.0);
    let total = d1 + d2;
    println!("3 m + 4 m = {} m", total.into_raw());

    // f64 scaling
    let doubled = total * 2.0;
    println!("doubled = {} m", doubled.into_raw());

    // The following would NOT compile (dimension mismatch):
    // let _ = mass + distance;  // Mass + Length → compile error
    // let _ = force + velocity; // Force + Velocity → compile error

    println!("\nAll operations type-checked at compile time!");
}
