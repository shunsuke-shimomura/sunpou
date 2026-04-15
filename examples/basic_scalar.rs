//! Basic scalar operations with compile-time unit and prefix checking.
//!
//! Demonstrates:
//! - Creating unit-tagged scalars with SI prefixes
//! - Arithmetic that produces correct output dimensions AND prefixes
//! - Rescaling between prefixes

use sunpou::prefix::*;
use sunpou::prelude::*;
use sunpou::scalar::Scalar;

fn main() {
    // Newton's second law: F = m * a (Base prefix = SI base units, the default)
    // Scalar::<Mass, Base> and Scalar::<Mass> are identical — Base can be omitted.
    let mass = Scalar::<Mass, Base>::from_raw_unchecked(100.0); // 100 kg
    let accel = Scalar::<Acceleration>::from_raw_unchecked(9.8); // 9.8 m/s² (Base omitted)
    let force: Scalar<Force> = mass * accel; // 980 N, prefix: Base + Base = Base
    println!("F = m * a = {} N", force.into_raw());

    // === SI Prefix support ===

    // Distance in km, time in s → velocity in km/s
    let distance = Scalar::<Length, Kilo>::from_raw_unchecked(100.0); // 100 km
    let time = Scalar::<Time>::from_raw_unchecked(10.0); // 10 s
    let velocity: Scalar<Velocity, Kilo> = distance / time; // 10 km/s
    println!("v = d / t = {} km/s", velocity.into_raw());

    // Cross-prefix multiplication: prefixes add automatically
    // 3 km × 4 km = 12 (prefix: Kilo+Kilo = Mega)
    let d1 = Scalar::<Length, Kilo>::from_raw_unchecked(3.0);
    let d2 = Scalar::<Length, Kilo>::from_raw_unchecked(4.0);
    let area: Scalar<Area, Mega> = d1 * d2;
    println!("3 km × 4 km = {} Mm² (= {} m²)", area.into_raw(), area.to_base_value());

    // Rescale between prefixes
    let km = Scalar::<Length, Kilo>::from_raw_unchecked(7.0); // 7 km
    let m: Scalar<Length, Base> = km.rescale(); // 7000 m
    println!("7 km = {} m", m.into_raw());

    // Same-prefix addition
    let total_km = d1 + d2;
    println!("3 km + 4 km = {} km", total_km.into_raw());

    // The following would NOT compile:
    // let _ = mass + Scalar::<Length>::from_raw_unchecked(1.0);  // Mass + Length
    // let _ = d1 + m;  // Kilo + Base (different prefix)

    println!("\nAll operations type-checked at compile time!");
}
