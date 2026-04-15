//! Basic scalar operations with compile-time unit and prefix checking.
//!
//! Demonstrates:
//! - Unit literal extension methods (7.0.km(), 9.8.m_per_s2())
//! - Arithmetic with automatic dimension and prefix tracking
//! - Rescaling between prefixes

use sunpou::prefix::*;
use sunpou::prelude::*;  // includes UnitLiteral trait
use sunpou::scalar::Scalar;

fn main() {
    // === Unit literals — the ergonomic way ===

    // Newton's second law: F = m * a
    let force: Scalar<Force> = 100.0.kg() * 9.8.m_per_s2();
    println!("F = m * a = {} N", force.into_raw());

    // Velocity from distance / time
    let velocity: Scalar<Velocity, Kilo> = 100.0.km() / 10.0.s();
    println!("v = d / t = {} km/s", velocity.into_raw());

    // Energy: W = F * d
    let work: Scalar<Energy> = 980.0.n() * 1000.0.m();
    println!("W = F * d = {} J", work.into_raw());

    // === Cross-prefix arithmetic ===

    // 3 km × 4 km = 12 Mm² (prefix: Kilo+Kilo = Mega)
    let area: Scalar<Area, Mega> = 3.0.km() * 4.0.km();
    println!("3 km × 4 km = {} Mm² (= {} m²)", area.into_raw(), area.to_base_value());

    // === Rescale between prefixes ===
    let km = 7.0.km();
    let m: Scalar<Length, Base> = km.rescale();
    println!("7 km = {} m", m.into_raw());

    // === from_raw — explicit construction (Base prefix is the default, can be omitted) ===
    let mass = Scalar::<Mass, Base>::from_raw(100.0); // equivalent to 100.0.kg()
    let accel = Scalar::<Acceleration>::from_raw(9.8); // Base omitted
    let _f: Scalar<Force> = mass * accel;

    // Same-prefix addition
    let total = 3.0.km() + 4.0.km();
    println!("3 km + 4 km = {} km", total.into_raw());

    // The following would NOT compile:
    // let _ = 3.0.km() + 4.0.m();     // Kilo + Base → prefix mismatch
    // let _ = 3.0.kg() + 4.0.m();     // Mass + Length → dimension mismatch

    println!("\nAll operations type-checked at compile time!");
}
