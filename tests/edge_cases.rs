//! Edge case tests: NaN, infinity, zero, singular matrices.

use nalgebra::Matrix3;
use sunpou::aliases::*;
use sunpou::frame_vec::FrameVec;
use sunpou::scalar::Scalar;
use sunpou::unit_mat::UnitMat;
use sunpou::unit_vec::UnitVec;

struct Eci;

// ---------------------------------------------------------------------------
// NaN propagation
// ---------------------------------------------------------------------------

#[test]
fn scalar_nan_propagation() {
    let a = Scalar::<Length>::from_raw(f64::NAN);
    let b = Scalar::<Length>::from_raw(1.0);
    assert!((a + b).into_raw().is_nan());
    assert!((a * b).into_raw().is_nan());
}

#[test]
fn unitvec_nan_norm() {
    let v = UnitVec::<Length, 3>::new(f64::NAN, 0.0, 0.0);
    assert!(v.norm().into_raw().is_nan());
}

// ---------------------------------------------------------------------------
// Infinity
// ---------------------------------------------------------------------------

#[test]
fn scalar_infinity() {
    let a = Scalar::<Length>::from_raw(f64::INFINITY);
    let b = Scalar::<Length>::from_raw(1.0);
    assert!((a + b).into_raw().is_infinite());
}

#[test]
fn unitvec_infinity_norm() {
    let v = UnitVec::<Length, 3>::new(f64::INFINITY, 0.0, 0.0);
    assert!(v.norm().into_raw().is_infinite());
}

// ---------------------------------------------------------------------------
// Zero vectors
// ---------------------------------------------------------------------------

#[test]
fn unitvec_zero_normalize() {
    let v = UnitVec::<Length, 3>::zeros();
    assert!(v.try_normalize(1e-10).is_none());
}

#[test]
fn framevec_zero_normalize() {
    let v = FrameVec::<Eci, Length>::new(0.0, 0.0, 0.0);
    assert!(v.try_normalize(1e-10).is_none());
}

#[test]
fn unitvec_nonzero_normalize() {
    let v = UnitVec::<Length, 3>::new(3.0, 4.0, 0.0);
    let n = v.try_normalize(1e-10).unwrap();
    let expected_norm = 1.0;
    assert!((n.norm().into_raw() - expected_norm).abs() < 1e-15);
}

// ---------------------------------------------------------------------------
// Singular matrix
// ---------------------------------------------------------------------------

#[test]
fn singular_matrix_inverse() {
    let m = UnitMat::<Velocity, Length, 3, 3>::from_raw(Matrix3::zeros());
    assert!(m.try_inverse().is_none());
}

#[test]
fn identity_inverse() {
    let m = UnitMat::<Length, Length, 3, 3>::identity();
    let inv = m.try_inverse().unwrap();
    assert_eq!(inv.into_raw(), Matrix3::identity());
}

// ---------------------------------------------------------------------------
// Compound assignment
// ---------------------------------------------------------------------------

#[test]
fn scalar_add_assign() {
    let mut a = Scalar::<Length>::from_raw(3.0);
    a += Scalar::<Length>::from_raw(4.0);
    assert_eq!(a.into_raw(), 7.0);
}

#[test]
fn scalar_sub_assign() {
    let mut a = Scalar::<Length>::from_raw(10.0);
    a -= Scalar::<Length>::from_raw(3.0);
    assert_eq!(a.into_raw(), 7.0);
}

#[test]
fn scalar_mul_assign_f64() {
    let mut a = Scalar::<Length>::from_raw(5.0);
    a *= 3.0;
    assert_eq!(a.into_raw(), 15.0);
}

#[test]
fn unitvec_add_assign() {
    let mut a = UnitVec::<Length, 3>::new(1.0, 2.0, 3.0);
    a += UnitVec::<Length, 3>::new(4.0, 5.0, 6.0);
    assert_eq!(a.as_raw().as_slice(), &[5.0, 7.0, 9.0]);
}

#[test]
fn unitvec_mul_assign_f64() {
    let mut v = UnitVec::<Length, 3>::new(1.0, 2.0, 3.0);
    v *= 2.0;
    assert_eq!(v.as_raw().as_slice(), &[2.0, 4.0, 6.0]);
}

#[test]
fn framevec_add_assign() {
    let mut a = FrameVec::<Eci, Length>::new(1.0, 0.0, 0.0);
    a += FrameVec::<Eci, Length>::new(0.0, 1.0, 0.0);
    assert_eq!(a.x(), 1.0);
    assert_eq!(a.y(), 1.0);
}

#[test]
fn unitmat_add_assign() {
    let mut m = UnitMat::<Length, Length, 3, 3>::from_raw(Matrix3::identity());
    m += UnitMat::from_raw(Matrix3::identity());
    assert_eq!(m.into_raw(), Matrix3::identity() * 2.0);
}

// ---------------------------------------------------------------------------
// Reference ops
// ---------------------------------------------------------------------------

#[test]
fn scalar_ref_add() {
    let a = Scalar::<Length>::from_raw(3.0);
    let b = Scalar::<Length>::from_raw(4.0);
    assert_eq!((&a + &b).into_raw(), 7.0);
    // Originals still usable
    assert_eq!(a.into_raw(), 3.0);
}

#[test]
fn unitvec_ref_add() {
    let a = UnitVec::<Length, 3>::new(1.0, 2.0, 3.0);
    let b = UnitVec::<Length, 3>::new(4.0, 5.0, 6.0);
    let c = &a + &b;
    assert_eq!(c.as_raw().as_slice(), &[5.0, 7.0, 9.0]);
    // Originals still usable
    assert_eq!(a.x(), 1.0);
}

#[test]
fn framevec_ref_sub() {
    let a = FrameVec::<Eci, Length>::new(10.0, 20.0, 30.0);
    let b = FrameVec::<Eci, Length>::new(1.0, 2.0, 3.0);
    let c = &a - &b;
    assert_eq!(c.x(), 9.0);
    assert_eq!(a.x(), 10.0);
}

// ---------------------------------------------------------------------------
// UnitVec convenience constructors
// ---------------------------------------------------------------------------

#[test]
fn unitvec3_new() {
    let v = UnitVec::<Length, 3>::new(1.0, 2.0, 3.0);
    assert_eq!(v.x(), 1.0);
    assert_eq!(v.y(), 2.0);
    assert_eq!(v.z(), 3.0);
}

#[test]
fn unitvec_zeros() {
    let v = UnitVec::<Velocity, 3>::zeros();
    assert_eq!(v.as_raw().as_slice(), &[0.0, 0.0, 0.0]);
}
