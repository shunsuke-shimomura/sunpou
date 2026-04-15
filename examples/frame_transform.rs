//! Frame-tagged vectors with coordinate transformations and prefixes.

use sunpou::prefix::*;
use sunpou::prelude::*;
use sunpou::frame_vec::FrameVec;
use sunpou::rotation::Rotation;
use sunpou::scalar::Scalar;

struct Eci;
struct Ecef;
struct Body;

fn main() {
    // Satellite position in ECI, km scale
    let pos_eci = FrameVec::<Eci, Length, Kilo>::new(7000.0, 0.0, 0.0);
    let vel_eci = FrameVec::<Eci, Velocity, Kilo>::new(0.0, 7.5, 0.0);

    println!("pos_eci = [{}, {}, {}] km", pos_eci.x(), pos_eci.y(), pos_eci.z());
    println!("vel_eci = [{}, {}, {}] km/s", vel_eci.x(), vel_eci.y(), vel_eci.z());

    // ECI → ECEF rotation (prefix preserved through rotation)
    let era = 1.5;
    let rot_eci_ecef = Rotation::<Eci, Ecef>::from_angle_z(era);

    let pos_ecef: FrameVec<Ecef, Length, Kilo> = rot_eci_ecef.transform(&pos_eci);
    println!(
        "pos_ecef = [{:.1}, {:.1}, {:.1}] km",
        pos_ecef.x(), pos_ecef.y(), pos_ecef.z()
    );

    // Round-trip: ECI → ECEF → ECI
    let pos_back: FrameVec<Eci, Length, Kilo> = rot_eci_ecef.inverse().transform(&pos_ecef);
    let err = (pos_eci.as_raw() - pos_back.into_raw()).norm();
    println!("Round-trip error: {} km", err);

    // Composition: ECI → ECEF → Body
    let rot_ecef_body = Rotation::<Ecef, Body>::from_angle_z(0.3);
    let rot_eci_body: Rotation<Eci, Body> = rot_eci_ecef.then(&rot_ecef_body);
    let pos_body: FrameVec<Body, Length, Kilo> = rot_eci_body.transform(&pos_eci);
    println!(
        "pos_body = [{:.1}, {:.1}, {:.1}] km",
        pos_body.x(), pos_body.y(), pos_body.z()
    );

    // Cross-dimension dot product (Kilo+Kilo = Mega)
    let force_eci = FrameVec::<Eci, Force, Kilo>::new(1.0, 0.0, 0.0); // 1 kN
    let disp_eci = FrameVec::<Eci, Length, Kilo>::new(10.0, 0.0, 0.0); // 10 km
    let work: Scalar<Energy, Mega> = force_eci.dot(&disp_eci);
    println!("Work = {} MJ", work.into_raw());

    // Angular momentum: r × v
    let h: FrameVec<Eci, LengthVelocity, Mega> = pos_eci.cross(&vel_eci);
    println!("h = [{:.1e}, {:.1e}, {:.1e}] Mm²/s", h.x(), h.y(), h.z());

    // The following would NOT compile:
    // let _ = pos_eci + pos_ecef;  // frame mismatch
    // let _ = pos_eci + pos_eci.rescale::<Base>();  // prefix mismatch
    //
    // Note: Base is the default prefix. FrameVec<Eci, Length> == FrameVec<Eci, Length, Base>

    println!("\nAll frame operations type-checked at compile time!");
}
