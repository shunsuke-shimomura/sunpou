//! Tests for Display with unit strings.

extern crate alloc;
use alloc::format;

use sunpou::prelude::*;
use sunpou::prefix::*;
use sunpou::units::UnitLiteral;

#[test]
fn display_length_base() {
    let s = 1000.0.m();
    assert_eq!(format!("{s}"), "1000 m");
}

#[test]
fn display_length_kilo() {
    let s = 7.0.km();
    assert_eq!(format!("{s}"), "7 km");
}

#[test]
fn display_length_milli() {
    let s = 500.0.mm();
    assert_eq!(format!("{s}"), "500 mm");
}

#[test]
fn display_mass() {
    let s = 100.0.kg();
    assert_eq!(format!("{s}"), "100 kg");
}

#[test]
fn display_time() {
    let s = 60.0.s();
    assert_eq!(format!("{s}"), "60 s");
}

#[test]
fn display_velocity_base() {
    let s = 7.5.m_per_s();
    assert_eq!(format!("{s}"), "7.5 m·s⁻¹");
}

#[test]
fn display_velocity_kilo() {
    let s = 7.5.km_per_s();
    assert_eq!(format!("{s}"), "7.5 km·s⁻¹");
}

#[test]
fn display_acceleration() {
    let s = 9.8.m_per_s2();
    assert_eq!(format!("{s}"), "9.8 m·s⁻²");
}

#[test]
fn display_force() {
    // Force = kg·m·s⁻²
    let s = 980.0.n();
    assert_eq!(format!("{s}"), "980 m·kg·s⁻²");
}

#[test]
fn display_force_kilo() {
    let s = 1.5.kn();
    assert_eq!(format!("{s}"), "1.5 km·kg·s⁻²");
}

#[test]
fn display_energy() {
    // Energy = kg·m²·s⁻²
    let s = 100.0.j();
    assert_eq!(format!("{s}"), "100 m²·kg·s⁻²");
}

#[test]
fn display_torque() {
    // Torque = kg·m²·s⁻² (same dim as Energy)
    let s = 5.0.nm();
    assert_eq!(format!("{s}"), "5 m²·kg·s⁻²");
}

#[test]
fn display_dimensionless() {
    use sunpou::scalar::Scalar;
    let s = Scalar::<Dimensionless>::from_raw(42.0);
    assert_eq!(format!("{s}"), "42");
}

#[test]
fn display_dimensionless_prefix() {
    // Dimensionless with Kilo → should show prefix only
    use sunpou::scalar::Scalar;
    let s = Scalar::<Dimensionless, Kilo>::from_raw(3.0);
    assert_eq!(format!("{s}"), "3 k");
}

#[test]
fn display_angular_velocity() {
    let s = 0.1.rad_per_s();
    assert_eq!(format!("{s}"), "0.1 s⁻¹");
}

#[test]
fn display_with_format_precision() {
    let s = 7.123456.km();
    assert_eq!(format!("{s:.2}"), "7.12 km");
}

#[test]
fn display_area_mega() {
    let area = 3.0.km() * 4.0.km();
    assert_eq!(format!("{area}"), "12 Mm²");
}
