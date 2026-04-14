//! Frame-tagged vectors and coordinate transformations.
//!
//! Demonstrates:
//! - FrameVec with both frame and dimension tags
//! - Rotation between frames preserving dimension
//! - Compile-time prevention of cross-frame operations

use uolgebra::prelude::*;
use uolgebra::frame_vec::FrameVec;
use uolgebra::rotation::Rotation;
use uolgebra::scalar::Scalar;

// User-defined frame markers (zero-sized types)
struct Eci;
struct Ecef;
struct Body;

fn main() {
    // Satellite position in ECI frame
    let pos_eci = FrameVec::<Eci, Length>::new(7000e3, 0.0, 0.0);
    let vel_eci = FrameVec::<Eci, Velocity>::new(0.0, 7.5e3, 0.0);

    println!("pos_eci = [{}, {}, {}] m", pos_eci.x(), pos_eci.y(), pos_eci.z());
    println!("vel_eci = [{}, {}, {}] m/s", vel_eci.x(), vel_eci.y(), vel_eci.z());

    // ECI → ECEF rotation (e.g. from Earth Rotation Angle)
    let era = 1.5; // radians
    let rot_eci_ecef = Rotation::<Eci, Ecef>::from_angle_z(era);

    // Transform position to ECEF (dimension is preserved)
    let pos_ecef: FrameVec<Ecef, Length> = rot_eci_ecef.transform(&pos_eci);
    println!(
        "pos_ecef = [{:.1}, {:.1}, {:.1}] m",
        pos_ecef.x(), pos_ecef.y(), pos_ecef.z()
    );

    // Transform velocity to ECEF
    let vel_ecef: FrameVec<Ecef, Velocity> = rot_eci_ecef.transform(&vel_eci);
    println!(
        "vel_ecef = [{:.1}, {:.1}, {:.1}] m/s",
        vel_ecef.x(), vel_ecef.y(), vel_ecef.z()
    );

    // Round-trip: ECI → ECEF → ECI
    let pos_back: FrameVec<Eci, Length> = rot_eci_ecef.inverse().transform(&pos_ecef);
    let err = (pos_eci.as_raw() - pos_back.into_raw()).norm();
    println!("Round-trip error: {} m", err);

    // Composition: ECI → ECEF → Body
    let rot_ecef_body = Rotation::<Ecef, Body>::from_angle_z(0.3);
    let rot_eci_body: Rotation<Eci, Body> = rot_eci_ecef.then(&rot_ecef_body);
    let pos_body: FrameVec<Body, Length> = rot_eci_body.transform(&pos_eci);
    println!(
        "pos_body = [{:.1}, {:.1}, {:.1}] m",
        pos_body.x(), pos_body.y(), pos_body.z()
    );

    // Cross-dimension dot product (same frame required)
    let force_eci = FrameVec::<Eci, Force>::new(1.0, 0.0, 0.0);
    let disp_eci = FrameVec::<Eci, Length>::new(10.0, 0.0, 0.0);
    let work: Scalar<Energy> = force_eci.dot(&disp_eci);
    println!("Work = {} J", work.into_raw());

    // Cross product: r × v → angular momentum per unit mass
    let h: FrameVec<Eci, LengthVelocity> = pos_eci.cross(&vel_eci);
    println!("h = r × v = [{:.1e}, {:.1e}, {:.1e}] m²/s", h.x(), h.y(), h.z());

    // The following would NOT compile:
    // let _ = pos_eci + pos_ecef;    // Eci + Ecef → compile error
    // let _ = pos_eci.dot(&pos_ecef); // dot across frames → compile error

    println!("\nAll frame operations type-checked at compile time!");
}
