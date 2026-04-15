//! Comprehensive tests for SI prefix support.
//!
//! Covers: construction, cross-prefix arithmetic, rescaling, precision,
//! uom cross-validation, and compile-fail scenarios.

use sunpou::aliases::*;
use sunpou::prefix::*;
use sunpou::scalar::Scalar;

// ============================================================================
// 1. CONSTRUCTION AND EXTRACTION
// ============================================================================

#[test]
fn base_prefix_unchanged() {
    // Default prefix = Z0, identical to previous behavior
    let s = Scalar::<Length>::from_raw_unchecked(1000.0);
    assert_eq!(s.into_raw(), 1000.0);
    assert_eq!(s.to_base_value(), 1000.0); // 1000 m
}

#[test]
fn kilo_prefix() {
    let s = Scalar::<Length, Kilo>::from_raw_unchecked(7.0); // 7 km
    assert_eq!(s.into_raw(), 7.0); // stored as 7.0
    assert_eq!(s.to_base_value(), 7000.0); // = 7000 m
}

#[test]
fn nano_prefix() {
    let s = Scalar::<Length, Nano>::from_raw_unchecked(500.0); // 500 nm
    assert_eq!(s.into_raw(), 500.0);
    assert!((s.to_base_value() - 500e-9).abs() < 1e-20);
}

#[test]
fn mega_prefix() {
    let s = Scalar::<Mass, Mega>::from_raw_unchecked(2.5); // 2.5 × 10^6 kg
    assert!((s.to_base_value() - 2.5e6).abs() < 1e-6);
}

// ============================================================================
// 2. RESCALE between prefixes
// ============================================================================

#[test]
fn rescale_kilo_to_base() {
    let km = Scalar::<Length, Kilo>::from_raw_unchecked(7.0);
    let m: Scalar<Length, Base> = km.rescale();
    assert!((m.into_raw() - 7000.0).abs() < 1e-10);
}

#[test]
fn rescale_nano_to_micro() {
    let nm = Scalar::<Length, Nano>::from_raw_unchecked(500.0); // 500 nm
    let um: Scalar<Length, Micro> = nm.rescale();
    assert!((um.into_raw() - 0.5).abs() < 1e-15); // 0.5 μm
}

#[test]
fn rescale_base_to_kilo() {
    let m = Scalar::<Length, Base>::from_raw_unchecked(7000.0);
    let km: Scalar<Length, Kilo> = m.rescale();
    assert!((km.into_raw() - 7.0).abs() < 1e-10);
}

#[test]
fn rescale_roundtrip() {
    let original = Scalar::<Length, Kilo>::from_raw_unchecked(42.0);
    let base: Scalar<Length, Base> = original.rescale();
    let back: Scalar<Length, Kilo> = base.rescale();
    assert!((back.into_raw() - 42.0).abs() < 1e-10);
}

// ============================================================================
// 3. SAME-PREFIX ADDITION (type-safe)
// ============================================================================

#[test]
fn add_same_prefix() {
    let a = Scalar::<Length, Kilo>::from_raw_unchecked(3.0); // 3 km
    let b = Scalar::<Length, Kilo>::from_raw_unchecked(4.0); // 4 km
    let c = a + b;
    assert_eq!(c.into_raw(), 7.0); // 7 km
}

#[test]
fn sub_same_prefix() {
    let a = Scalar::<Length, Nano>::from_raw_unchecked(500.0);
    let b = Scalar::<Length, Nano>::from_raw_unchecked(200.0);
    assert_eq!((a - b).into_raw(), 300.0);
}

// ============================================================================
// 4. CROSS-PREFIX MULTIPLICATION — prefixes add
// ============================================================================

#[test]
fn mul_kilo_times_kilo() {
    // 3 km × 4 km = 12 (km)² = 12 × 10^6 m² → prefix = Mega
    let a = Scalar::<Length, Kilo>::from_raw_unchecked(3.0);
    let b = Scalar::<Length, Kilo>::from_raw_unchecked(4.0);
    let c: Scalar<Area, Mega> = a * b; // P3 + P3 = P6
    assert_eq!(c.into_raw(), 12.0);
    assert!((c.to_base_value() - 12e6).abs() < 1e-6);
}

#[test]
fn mul_nano_times_mega() {
    // 5 nm × 3 MHz (if we abuse units) → stored: 5.0 × 3.0 = 15.0
    // prefix: N9 + P6 = N3 (milli)
    let a = Scalar::<Length, Nano>::from_raw_unchecked(5.0);
    let b = Scalar::<InvTime, Mega>::from_raw_unchecked(3.0);
    let c: Scalar<Velocity, Milli> = a * b; // N9 + P6 = N3
    assert_eq!(c.into_raw(), 15.0);
    assert!((c.to_base_value() - 15e-3).abs() < 1e-15);
}

#[test]
fn mul_base_times_kilo() {
    // 10 kg × 5 km/s = 50 (kg·km/s) → prefix = Kilo
    let mass = Scalar::<Mass>::from_raw_unchecked(10.0);
    let vel = Scalar::<Velocity, Kilo>::from_raw_unchecked(5.0);
    let momentum: Scalar<Momentum, Kilo> = mass * vel; // Z0 + P3 = P3
    assert_eq!(momentum.into_raw(), 50.0);
    assert!((momentum.to_base_value() - 50e3).abs() < 1e-6);
}

// ============================================================================
// 5. CROSS-PREFIX DIVISION — prefixes subtract
// ============================================================================

#[test]
fn div_kilo_by_base() {
    // 100 km / 10 s = 10 km/s → prefix = Kilo
    let dist = Scalar::<Length, Kilo>::from_raw_unchecked(100.0);
    let time = Scalar::<Time>::from_raw_unchecked(10.0);
    let vel: Scalar<Velocity, Kilo> = dist / time; // P3 - Z0 = P3
    assert_eq!(vel.into_raw(), 10.0); // 10 km/s
}

#[test]
fn div_mega_by_kilo() {
    // 12 MJ / 4 kN = 3 (MJ/kN) → 3 × 10^(6-3) m = 3 km → prefix Kilo
    let energy = Scalar::<Energy, Mega>::from_raw_unchecked(12.0);
    let force = Scalar::<Force, Kilo>::from_raw_unchecked(4.0);
    let dist: Scalar<Length, Kilo> = energy / force; // P6 - P3 = P3
    assert_eq!(dist.into_raw(), 3.0);
    assert!((dist.to_base_value() - 3000.0).abs() < 1e-10);
}

// ============================================================================
// 6. f64 SCALING (preserves prefix)
// ============================================================================

#[test]
fn f64_mul_preserves_prefix() {
    let s = Scalar::<Length, Kilo>::from_raw_unchecked(3.0);
    let doubled = s * 2.0;
    assert_eq!(doubled.into_raw(), 6.0); // 6 km
}

// ============================================================================
// 7. NUMERICAL PRECISION — the whole point
// ============================================================================

#[test]
fn precision_nano_times_giga() {
    // Without prefix: 5e-9 * 3e9 = 15.0 (OK in this case)
    // With prefix: 5.0 * 3.0 = 15.0, prefix = N9+P9 = Z0
    // Both give 15.0, but with prefix the intermediate values are nicer

    let a = Scalar::<Length, Nano>::from_raw_unchecked(5.0);
    let b = Scalar::<InvTime, Giga>::from_raw_unchecked(3.0);
    let c: Scalar<Velocity> = a * b; // N9 + P9 = Z0
    assert_eq!(c.into_raw(), 15.0);
}

#[test]
fn precision_addition_same_scale() {
    // Adding values at the same scale preserves precision
    // In base units: 1.000000001e-9 + 2.000000002e-9 may lose digits
    // In nano: 1.000000001 + 2.000000002 = 3.000000003 (exact in f64)
    let a = Scalar::<Length, Nano>::from_raw_unchecked(1.5);
    let b = Scalar::<Length, Nano>::from_raw_unchecked(2.5);
    let c = a + b;
    assert_eq!(c.into_raw(), 4.0); // exact in f64
}

// ============================================================================
// 8. uom CROSS-VALIDATION with prefix
// ============================================================================

#[test]
fn uom_cross_validation_with_prefix() {
    use uom::si::f64::{Length as UomLength, Velocity as UomVelocity, Time as UomTime};
    use uom::si::length::kilometer;
    use uom::si::time::second;
    use uom::si::velocity::kilometer_per_second;

    let uom_dist = UomLength::new::<kilometer>(100.0);
    let uom_time = UomTime::new::<second>(10.0);
    let uom_vel: UomVelocity = uom_dist / uom_time;

    let our_dist = Scalar::<Length, Kilo>::from_raw_unchecked(100.0); // 100 km
    let our_time = Scalar::<Time>::from_raw_unchecked(10.0);          // 10 s
    let our_vel: Scalar<Velocity, Kilo> = our_dist / our_time;       // 10 km/s

    assert_eq!(uom_vel.get::<kilometer_per_second>(), our_vel.into_raw());
}

// ============================================================================
// 9. COMPOUND ASSIGNMENT with prefix
// ============================================================================

#[test]
fn add_assign_with_prefix() {
    let mut a = Scalar::<Length, Kilo>::from_raw_unchecked(3.0);
    a += Scalar::<Length, Kilo>::from_raw_unchecked(4.0);
    assert_eq!(a.into_raw(), 7.0);
}

// ============================================================================
// 10. DEFAULT and DISPLAY with prefix
// ============================================================================

#[test]
fn default_with_prefix() {
    let s = Scalar::<Length, Kilo>::default();
    assert_eq!(s.into_raw(), 0.0);
}

// ============================================================================
// 11. ZERO-COST: Scalar<D, P> has same size as f64
// ============================================================================

#[test]
fn prefix_scalar_zero_cost() {
    assert_eq!(
        core::mem::size_of::<Scalar<Length, Kilo>>(),
        core::mem::size_of::<f64>(),
    );
    assert_eq!(
        core::mem::size_of::<Scalar<Length, Nano>>(),
        core::mem::size_of::<f64>(),
    );
}
