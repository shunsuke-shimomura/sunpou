//! Tests for usability improvements: unit literals, from_raw rename, Display.

use sunpou::prelude::*;
use sunpou::prefix::*;
use sunpou::scalar::Scalar;
use sunpou::units::UnitLiteral;

// ============================================================================
// A: Unit literal extension methods
// ============================================================================

#[test]
fn literal_meters() {
    let d = 1000.0.m();
    assert_eq!(d.into_raw(), 1000.0);
    let _: Scalar<Length> = d; // type check
}

#[test]
fn literal_kilometers() {
    let d = 7.0.km();
    assert_eq!(d.into_raw(), 7.0);
    let _: Scalar<Length, Kilo> = d;
}

#[test]
fn literal_millimeters() {
    let d = 500.0.mm();
    assert_eq!(d.into_raw(), 500.0);
    let _: Scalar<Length, Milli> = d;
}

#[test]
fn literal_kilograms() {
    let m = 100.0.kg();
    let _: Scalar<Mass> = m;
    assert_eq!(m.into_raw(), 100.0);
}

#[test]
fn literal_seconds() {
    let t = 60.0.s();
    let _: Scalar<Time> = t;
}

#[test]
fn literal_milliseconds() {
    let t = 500.0.ms();
    let _: Scalar<Time, Milli> = t;
}

#[test]
fn literal_velocity() {
    let v = 7.5.km_per_s();
    let _: Scalar<Velocity, Kilo> = v;
}

#[test]
fn literal_acceleration() {
    let a = 9.8.m_per_s2();
    let _: Scalar<Acceleration> = a;
}

#[test]
fn literal_force() {
    let f = 980.0.n();
    let _: Scalar<Force> = f;
}

#[test]
fn literal_kilonewtons() {
    let f = 1.5.kn();
    let _: Scalar<Force, Kilo> = f;
}

#[test]
fn literal_energy() {
    let e = 100.0.j();
    let _: Scalar<Energy> = e;
}

#[test]
fn literal_torque() {
    let t = 5.0.nm();
    let _: Scalar<Torque> = t;
}

#[test]
fn literal_angular_velocity() {
    let w = 0.1.rad_per_s();
    let _: Scalar<AngularVelocity> = w;
}

// ---- Arithmetic with literals ----

#[test]
fn literal_f_equals_ma() {
    let force: Scalar<Force> = 100.0.kg() * 9.8.m_per_s2();
    assert!((force.into_raw() - 980.0).abs() < 1e-12);
}

#[test]
fn literal_v_equals_d_over_t() {
    let vel: Scalar<Velocity, Kilo> = 100.0.km() / 10.0.s();
    assert!((vel.into_raw() - 10.0).abs() < 1e-12);
}

#[test]
fn literal_work_equals_f_times_d() {
    let work: Scalar<Energy> = 50.0.n() * 3.0.m();
    assert!((work.into_raw() - 150.0).abs() < 1e-12);
}

#[test]
fn literal_cross_prefix_mul() {
    // 3 km × 4 km = 12 Mm²
    let area: Scalar<Area, Mega> = 3.0.km() * 4.0.km();
    assert_eq!(area.into_raw(), 12.0);
}

// ============================================================================
// D: from_raw (was from_raw_unchecked) — just verify it works
// ============================================================================

#[test]
fn from_raw_works() {
    let s = Scalar::<Length>::from_raw(5.0);
    assert_eq!(s.into_raw(), 5.0);
}

#[test]
fn from_raw_with_prefix() {
    let s = Scalar::<Length, Kilo>::from_raw(7.0);
    assert_eq!(s.into_raw(), 7.0);
    assert!((s.to_base_value() - 7000.0).abs() < 1e-10);
}
