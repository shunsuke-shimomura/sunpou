//! Vector operations with heterogeneous-dimension dot/cross products and prefixes.

use sunpou::prefix::*;
use sunpou::prelude::*;
use sunpou::scalar::Scalar;
use sunpou::unit_vec::UnitVec;

fn main() {
    // Position and velocity in km scale (natural for orbital mechanics)
    let pos = UnitVec::<Length, 3, Kilo>::new(7000.0, 0.0, 0.0); // 7000 km
    let vel = UnitVec::<Velocity, 3, Kilo>::new(0.0, 7.5, 0.0);  // 7.5 km/s

    // Same-prefix addition
    let pos2 = UnitVec::<Length, 3, Kilo>::new(0.0, 0.1, 0.0);
    let total_pos = pos + pos2;
    println!("pos + offset = [{}, {}, {}] km", total_pos.x(), total_pos.y(), total_pos.z());

    // Cross-dim, cross-prefix: r × v → specific angular momentum
    // Kilo + Kilo = Mega
    let h: UnitVec<LengthVelocity, 3, Mega> = pos.cross(&vel);
    println!("h = r × v = [{}, {}, {}] Mm²/s", h.x(), h.y(), h.z());

    // Dot product: force · displacement → energy
    let force = UnitVec::<Force, 3, Kilo>::new(10.0, 0.0, 0.0); // 10 kN
    let disp = UnitVec::<Length, 3, Kilo>::new(5.0, 0.0, 0.0);  // 5 km
    let work: Scalar<Energy, Mega> = force.dot(&disp); // Kilo+Kilo = Mega
    println!("W = F · d = {} MJ", work.into_raw());

    // Scalar × vector (cross-dimension, cross-prefix)
    let mass = Scalar::<Mass>::from_raw_unchecked(10.0); // 10 kg (base)
    let accel = UnitVec::<Acceleration, 3>::new(0.0, 0.0, 9.8);  // m/s² (base)
    let force_vec: UnitVec<Force, 3> = mass * accel;
    println!("F = m * a = [{}, {}, {}] N", force_vec.x(), force_vec.y(), force_vec.z());

    // Norm preserves prefix
    let r: Scalar<Length, Kilo> = pos.norm();
    println!("|pos| = {} km = {} m", r.into_raw(), r.to_base_value());

    // Rescale vector
    let pos_m: UnitVec<Length, 3, Base> = pos.rescale();
    println!("pos in m = [{}, {}, {}]", pos_m.x(), pos_m.y(), pos_m.z());

    println!("\nAll vector operations type-checked at compile time!");
}
