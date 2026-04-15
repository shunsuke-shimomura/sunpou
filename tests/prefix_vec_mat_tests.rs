//! Tests for SI prefix on vectors and matrices.

use nalgebra::Matrix3;
use sunpou::aliases::*;
use sunpou::elem_mat::ElemMat;
use sunpou::frame_elem_mat::FrameElemMat;
use sunpou::frame_vec::FrameVec;
use sunpou::prefix::*;
use sunpou::scalar::Scalar;
use sunpou::unit_vec::UnitVec;

struct Eci;
struct Body;

// ============================================================================
// UnitVec with prefix
// ============================================================================

#[test]
fn unitvec_kilo_construction() {
    let v = UnitVec::<Length, 3, Kilo>::new(7.0, 0.0, 0.0); // 7 km
    assert_eq!(v.x(), 7.0);
}

#[test]
fn unitvec_add_same_prefix() {
    let a = UnitVec::<Length, 3, Kilo>::new(3.0, 0.0, 0.0);
    let b = UnitVec::<Length, 3, Kilo>::new(4.0, 0.0, 0.0);
    let c = a + b;
    assert_eq!(c.x(), 7.0); // 7 km
}

#[test]
fn unitvec_dot_cross_prefix() {
    // 3 km · 4 km = 12 (km²) → prefix Mega
    let a = UnitVec::<Length, 3, Kilo>::new(3.0, 0.0, 0.0);
    let b = UnitVec::<Length, 3, Kilo>::new(4.0, 0.0, 0.0);
    let result: Scalar<Area, Mega> = a.dot(&b);
    assert_eq!(result.into_raw(), 12.0);
}

#[test]
fn unitvec_cross_cross_prefix() {
    // (km) × (km/s) → prefix Kilo+Kilo = Mega for specific angular momentum
    let r = UnitVec::<Length, 3, Kilo>::new(7.0, 0.0, 0.0);
    let v = UnitVec::<Velocity, 3, Kilo>::new(0.0, 7.5, 0.0);
    let h: UnitVec<LengthVelocity, 3, Mega> = r.cross(&v);
    assert!((h.z() - 52.5).abs() < 1e-12);
}

#[test]
fn scalar_times_unitvec_cross_prefix() {
    // kilo-mass * kilo-velocity → mega-momentum
    let mass = Scalar::<Mass, Kilo>::from_raw_unchecked(2.0);  // 2000 kg
    let vel = UnitVec::<Velocity, 3, Kilo>::new(1.0, 0.0, 0.0); // 1 km/s
    let momentum: UnitVec<Momentum, 3, Mega> = mass * vel;
    assert_eq!(momentum.x(), 2.0);
}

#[test]
fn unitvec_rescale() {
    let km = UnitVec::<Length, 3, Kilo>::new(7.0, 0.0, 0.0);
    let m: UnitVec<Length, 3, Base> = km.rescale();
    assert!((m.x() - 7000.0).abs() < 1e-10);
}

#[test]
fn unitvec_norm_preserves_prefix() {
    let v = UnitVec::<Length, 3, Kilo>::new(3.0, 4.0, 0.0);
    let n: Scalar<Length, Kilo> = v.norm();
    assert!((n.into_raw() - 5.0).abs() < 1e-14);
}

// ============================================================================
// FrameVec with prefix
// ============================================================================

#[test]
fn framevec_kilo_construction() {
    let v = FrameVec::<Eci, Length, Kilo>::new(7.0, 0.0, 0.0);
    assert_eq!(v.x(), 7.0);
}

#[test]
fn framevec_dot_cross_prefix() {
    let f = FrameVec::<Eci, Force, Kilo>::new(10.0, 0.0, 0.0); // 10 kN
    let d = FrameVec::<Eci, Length, Kilo>::new(5.0, 0.0, 0.0);  // 5 km
    let work: Scalar<Energy, Mega> = f.dot(&d);
    assert_eq!(work.into_raw(), 50.0); // 50 MJ
}

#[test]
fn framevec_rescale() {
    let km = FrameVec::<Eci, Length, Kilo>::new(7.0, 0.0, 0.0);
    let m: FrameVec<Eci, Length, Base> = km.rescale();
    assert!((m.x() - 7000.0).abs() < 1e-10);
}

#[test]
fn framevec_cross_cross_prefix() {
    let r = FrameVec::<Eci, Length, Kilo>::new(7.0, 0.0, 0.0);
    let v = FrameVec::<Eci, Velocity, Kilo>::new(0.0, 7.5, 0.0);
    let h: FrameVec<Eci, LengthVelocity, Mega> = r.cross(&v);
    assert!((h.z() - 52.5).abs() < 1e-12);
}

// ============================================================================
// ElemMat with prefix
// ============================================================================

#[test]
fn elemmat_times_unitvec_cross_prefix() {
    // ElemMat in base prefix * UnitVec in kilo → output in kilo
    let m = ElemMat::<Mass, 3, 3>::from_raw_unchecked(Matrix3::identity() * 10.0);
    let v = UnitVec::<Velocity, 3, Kilo>::new(1.0, 0.0, 0.0);
    let p: UnitVec<Momentum, 3, Kilo> = m * v;
    assert_eq!(p.x(), 10.0); // 10 kilo-momentum-units
}

// ============================================================================
// FrameElemMat with prefix
// ============================================================================

#[test]
fn frame_elemmat_times_framevec_cross_prefix() {
    let inertia = FrameElemMat::<Body, MomentOfInertia, 3, 3>::from_raw_unchecked(
        Matrix3::identity() * 100.0,
    );
    // Angular velocity in milli-rad/s
    let omega = FrameVec::<Body, AngularVelocity, Milli>::new(1.0, 0.0, 0.0);

    // MoI(base) * AngVel(milli) → AngMom(milli)
    let ang_mom: FrameVec<Body, AngularMomentum, Milli> = inertia * omega;
    assert_eq!(ang_mom.x(), 100.0); // 100 milli-(kg·m²/s)
}

// ============================================================================
// Orbital mechanics in km
// ============================================================================

#[test]
fn orbital_state_in_km() {
    // Position in km, velocity in km/s — natural scale for orbit mechanics
    let r = FrameVec::<Eci, Length, Kilo>::new(7000.0, 0.0, 0.0);   // 7000 km
    let v = FrameVec::<Eci, Velocity, Kilo>::new(0.0, 7.5, 0.0);   // 7.5 km/s

    // Specific angular momentum h = r × v
    let h: FrameVec<Eci, LengthVelocity, Mega> = r.cross(&v);
    // 7000 * 7.5 = 52500 (Mega units = 10^6 m²/s)
    assert!((h.z() - 52500.0).abs() < 1e-8);

    // To base SI: 52500 × 10^6 = 52.5e9 m²/s
    assert!((h.norm().to_base_value() - 52500e6).abs() < 1e3);
}

// ============================================================================
// Zero-cost
// ============================================================================

#[test]
fn prefix_vec_zero_cost() {
    use core::mem::size_of;
    assert_eq!(
        size_of::<UnitVec<Length, 3, Kilo>>(),
        size_of::<UnitVec<Length, 3>>(),
    );
    assert_eq!(
        size_of::<FrameVec<Eci, Length, Nano>>(),
        size_of::<FrameVec<Eci, Length>>(),
    );
}
