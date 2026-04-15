//! Tests for Round 3 features: conversions, indexing, Display, From<f64>

use sunpou::aliases::*;
use sunpou::frame_vec::FrameVec;
use sunpou::scalar::Scalar;
use sunpou::unit_vec::UnitVec;
use sunpou::rotation::Rotation;

extern crate alloc;
use alloc::format;

struct Eci;
struct Ecef;

// ---------------------------------------------------------------------------
// FrameVec ↔ UnitVec conversion
// ---------------------------------------------------------------------------

#[test]
fn framevec_to_unitvec() {
    let fv = FrameVec::<Eci, Length>::new(1.0, 2.0, 3.0);
    let uv: UnitVec<Length, 3> = fv.to_unit_vec();
    assert_eq!(uv.x(), 1.0);
    assert_eq!(uv.y(), 2.0);
    assert_eq!(uv.z(), 3.0);
}

#[test]
fn unitvec_to_framevec() {
    let uv = UnitVec::<Velocity, 3>::new(4.0, 5.0, 6.0);
    let fv = FrameVec::<Ecef, Velocity>::from_unit_vec(&uv);
    assert_eq!(fv.x(), 4.0);
    assert_eq!(fv.y(), 5.0);
    assert_eq!(fv.z(), 6.0);
}

#[test]
fn framevec_unitvec_roundtrip() {
    let original = FrameVec::<Eci, Length>::new(7000e3, 100.0, -50.0);
    let uv = original.to_unit_vec();
    let back = FrameVec::<Eci, Length>::from_unit_vec(&uv);
    assert_eq!(original.into_raw(), back.into_raw());
}

// ---------------------------------------------------------------------------
// Dimensionless Scalar ↔ f64
// ---------------------------------------------------------------------------

#[test]
fn dimensionless_from_f64() {
    let s: Scalar<Dimensionless> = Scalar::from(3.14);
    assert_eq!(s.into_raw(), 3.14);
}

#[test]
fn f64_from_dimensionless() {
    let s = Scalar::<Dimensionless>::from_raw(2.718);
    let v: f64 = s.into();
    assert_eq!(v, 2.718);
}

// ---------------------------------------------------------------------------
// UnitVec indexing and iteration
// ---------------------------------------------------------------------------

#[test]
fn unitvec_index() {
    let v = UnitVec::<Length, 3>::new(1.0, 2.0, 3.0);
    assert_eq!(v[0], 1.0);
    assert_eq!(v[1], 2.0);
    assert_eq!(v[2], 3.0);
}

#[test]
fn unitvec_iter() {
    let v = UnitVec::<Length, 3>::new(1.0, 2.0, 3.0);
    let sum: f64 = v.iter().sum();
    assert_eq!(sum, 6.0);
}

#[test]
fn unitvec_len() {
    let v = UnitVec::<Length, 3>::zeros();
    assert_eq!(v.len(), 3);
    assert!(!v.is_empty());
}

#[test]
fn unitvec_from_slice() {
    let data = [1.0, 2.0, 3.0];
    let v = UnitVec::<Velocity, 3>::from_slice(&data);
    assert_eq!(v[0], 1.0);
    assert_eq!(v[1], 2.0);
    assert_eq!(v[2], 3.0);
}

// ---------------------------------------------------------------------------
// Display
// ---------------------------------------------------------------------------

#[test]
fn scalar_display() {
    let s = Scalar::<Length>::from_raw(42.5);
    assert_eq!(format!("{s}"), "42.5 m");
}

#[test]
fn unitvec_display() {
    let v = UnitVec::<Length, 3>::new(1.0, 2.0, 3.0);
    assert_eq!(format!("{v}"), "[1, 2, 3]");
}

#[test]
fn framevec_display() {
    let v = FrameVec::<Eci, Length>::new(1.0, 2.0, 3.0);
    assert_eq!(format!("{v}"), "[1, 2, 3]");
}

// ---------------------------------------------------------------------------
// Default
// ---------------------------------------------------------------------------

#[test]
fn scalar_default() {
    let s = Scalar::<Mass>::default();
    assert_eq!(s.into_raw(), 0.0);
}

#[test]
fn unitvec_default() {
    let v = UnitVec::<Velocity, 3>::default();
    assert_eq!(v[0], 0.0);
    assert_eq!(v[1], 0.0);
    assert_eq!(v[2], 0.0);
}

#[test]
fn framevec_default() {
    let v = FrameVec::<Eci, Length>::default();
    assert_eq!(v.x(), 0.0);
}

// ---------------------------------------------------------------------------
// Rotation to_matrix
// ---------------------------------------------------------------------------

#[test]
fn rotation_to_matrix() {
    let rot = Rotation::<Eci, Ecef>::from_angle_z(0.0);
    let mat = rot.to_matrix();
    // Identity rotation → identity matrix
    let raw = mat.into_raw();
    assert!((raw - nalgebra::Matrix3::identity()).norm() < 1e-15);
}

#[test]
fn rotation_to_matrix_consistency() {
    let angle = 1.2;
    let rot = Rotation::<Eci, Ecef>::from_angle_z(angle);
    let mat = rot.to_matrix();
    let v = FrameVec::<Eci, Length>::new(1.0, 0.0, 0.0);

    // Transform via rotation
    let via_rot = rot.transform(&v);
    // Transform via matrix * raw vector
    let via_mat = mat.into_raw() * v.into_raw();

    assert!((via_rot.into_raw() - via_mat).norm() < 1e-15);
}
