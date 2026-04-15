//! Cross-validation: every sunpou operation must produce identical results
//! to the equivalent raw nalgebra operation.

use nalgebra::{Matrix3, SVector, Vector3};
use sunpou::aliases::*;
use sunpou::frame_vec::FrameVec;
use sunpou::rotation::Rotation;
use sunpou::scalar::Scalar;
use sunpou::unit_mat::UnitMat;
use sunpou::unit_vec::UnitVec;

struct Eci;
struct Ecef;

// ---------------------------------------------------------------------------
// Scalar
// ---------------------------------------------------------------------------

#[test]
fn scalar_add() {
    let a = 3.0_f64;
    let b = 4.0_f64;
    let sa = Scalar::<Length>::from_raw(a);
    let sb = Scalar::<Length>::from_raw(b);
    assert_eq!((sa + sb).into_raw(), a + b);
}

#[test]
fn scalar_sub() {
    let a = 10.0_f64;
    let b = 3.0_f64;
    let sa = Scalar::<Length>::from_raw(a);
    let sb = Scalar::<Length>::from_raw(b);
    assert_eq!((sa - sb).into_raw(), a - b);
}

#[test]
fn scalar_mul_cross_dim() {
    let m = 100.0_f64;
    let a = 9.8_f64;
    let sm = Scalar::<Mass>::from_raw(m);
    let sa = Scalar::<Acceleration>::from_raw(a);
    let result: Scalar<Force> = sm * sa;
    assert_eq!(result.into_raw(), m * a);
}

#[test]
fn scalar_div_cross_dim() {
    let d = 100.0_f64;
    let t = 5.0_f64;
    let sd = Scalar::<Length>::from_raw(d);
    let st = Scalar::<Time>::from_raw(t);
    let result: Scalar<Velocity> = sd / st;
    assert_eq!(result.into_raw(), d / t);
}

#[test]
fn scalar_neg() {
    let s = Scalar::<Length>::from_raw(5.0);
    assert_eq!((-s).into_raw(), -5.0);
}

#[test]
fn scalar_f64_mul() {
    let s = Scalar::<Length>::from_raw(3.0);
    assert_eq!((s * 2.0).into_raw(), 6.0);
    assert_eq!((2.0 * s).into_raw(), 6.0);
}

// ---------------------------------------------------------------------------
// UnitVec
// ---------------------------------------------------------------------------

#[test]
fn unitvec_add() {
    let raw_a = SVector::<f64, 3>::new(1.0, 2.0, 3.0);
    let raw_b = SVector::<f64, 3>::new(4.0, 5.0, 6.0);
    let a = UnitVec::<Length, 3>::from_raw(raw_a);
    let b = UnitVec::<Length, 3>::from_raw(raw_b);
    assert_eq!((a + b).into_raw(), raw_a + raw_b);
}

#[test]
fn unitvec_sub() {
    let raw_a = SVector::<f64, 3>::new(1.0, 2.0, 3.0);
    let raw_b = SVector::<f64, 3>::new(4.0, 5.0, 6.0);
    let a = UnitVec::<Length, 3>::from_raw(raw_a);
    let b = UnitVec::<Length, 3>::from_raw(raw_b);
    assert_eq!((a - b).into_raw(), raw_a - raw_b);
}

#[test]
fn unitvec_dot_same_dim() {
    let raw_a = SVector::<f64, 3>::new(1.0, 2.0, 3.0);
    let raw_b = SVector::<f64, 3>::new(4.0, 5.0, 6.0);
    let expected = raw_a.dot(&raw_b);
    let a = UnitVec::<Length, 3>::from_raw(raw_a);
    let b = UnitVec::<Length, 3>::from_raw(raw_b);
    let result: Scalar<Area> = a.dot(&b);
    assert_eq!(result.into_raw(), expected);
}

#[test]
fn unitvec_dot_cross_dim() {
    let raw_a = SVector::<f64, 3>::new(1.0, 2.0, 3.0);
    let raw_b = SVector::<f64, 3>::new(4.0, 5.0, 6.0);
    let expected = raw_a.dot(&raw_b);
    let a = UnitVec::<Force, 3>::from_raw(raw_a);
    let b = UnitVec::<Length, 3>::from_raw(raw_b);
    let result: Scalar<Energy> = a.dot(&b);
    assert_eq!(result.into_raw(), expected);
}

#[test]
fn unitvec_cross_same_dim() {
    let raw_a = SVector::<f64, 3>::new(1.0, 2.0, 3.0);
    let raw_b = SVector::<f64, 3>::new(4.0, 5.0, 6.0);
    let expected = Vector3::new(1.0, 2.0, 3.0).cross(&Vector3::new(4.0, 5.0, 6.0));
    let a = UnitVec::<Length, 3>::from_raw(raw_a);
    let b = UnitVec::<Length, 3>::from_raw(raw_b);
    let result: UnitVec<Area, 3> = a.cross(&b);
    assert_eq!(result.into_raw().as_slice(), expected.as_slice());
}

#[test]
fn unitvec_cross_cross_dim() {
    let raw_a = SVector::<f64, 3>::new(1.0, 0.0, 0.0);
    let raw_b = SVector::<f64, 3>::new(0.0, 7.5, 0.0);
    let expected = Vector3::new(1.0, 0.0, 0.0).cross(&Vector3::new(0.0, 7.5, 0.0));
    let a = UnitVec::<Length, 3>::from_raw(raw_a);
    let b = UnitVec::<Velocity, 3>::from_raw(raw_b);
    // Length × Velocity = m²/s (specific angular momentum)
    let result: UnitVec<LengthVelocity, 3> = a.cross(&b);
    assert_eq!(result.into_raw().as_slice(), expected.as_slice());
}

#[test]
fn unitvec_scalar_mul() {
    let raw_v = SVector::<f64, 3>::new(1.0, 2.0, 3.0);
    let s = Scalar::<Mass>::from_raw(10.0);
    let v = UnitVec::<Acceleration, 3>::from_raw(raw_v);
    let result: UnitVec<Force, 3> = s * v;
    assert_eq!(result.into_raw(), raw_v * 10.0);
}

#[test]
fn unitvec_f64_mul() {
    let raw = SVector::<f64, 3>::new(1.0, 2.0, 3.0);
    let v = UnitVec::<Length, 3>::from_raw(raw);
    assert_eq!((v * 2.0).into_raw(), raw * 2.0);
    assert_eq!((2.0 * v).into_raw(), raw * 2.0);
}

#[test]
fn unitvec_norm() {
    let raw = SVector::<f64, 3>::new(3.0, 4.0, 0.0);
    let v = UnitVec::<Length, 3>::from_raw(raw);
    assert_eq!(v.norm().into_raw(), raw.norm());
}

// ---------------------------------------------------------------------------
// FrameVec
// ---------------------------------------------------------------------------

#[test]
fn framevec_add() {
    let raw_a = Vector3::new(1.0, 2.0, 3.0);
    let raw_b = Vector3::new(4.0, 5.0, 6.0);
    let a = FrameVec::<Eci, Length>::from_raw(raw_a);
    let b = FrameVec::<Eci, Length>::from_raw(raw_b);
    assert_eq!((a + b).into_raw(), raw_a + raw_b);
}

#[test]
fn framevec_dot_cross_dim() {
    let raw_a = Vector3::new(1.0, 0.0, 0.0);
    let raw_b = Vector3::new(10.0, 0.0, 0.0);
    let expected = raw_a.dot(&raw_b);
    let a = FrameVec::<Eci, Force>::new(1.0, 0.0, 0.0);
    let b = FrameVec::<Eci, Length>::new(10.0, 0.0, 0.0);
    let result: Scalar<Energy> = a.dot(&b);
    assert_eq!(result.into_raw(), expected);
}

#[test]
fn framevec_cross_cross_dim() {
    let raw_a = Vector3::new(7000e3, 0.0, 0.0);
    let raw_b = Vector3::new(0.0, 7.5e3, 0.0);
    let expected = raw_a.cross(&raw_b);
    let a = FrameVec::<Eci, Length>::new(7000e3, 0.0, 0.0);
    let b = FrameVec::<Eci, Velocity>::new(0.0, 7.5e3, 0.0);
    let result: FrameVec<Eci, LengthVelocity> = a.cross(&b);
    assert_eq!(result.into_raw(), expected);
}

// ---------------------------------------------------------------------------
// Rotation
// ---------------------------------------------------------------------------

#[test]
fn rotation_transform() {
    let angle = core::f64::consts::FRAC_PI_2;
    let rot = Rotation::<Eci, Ecef>::from_angle_z(angle);
    let raw = Vector3::new(1.0, 0.0, 0.0);
    let v = FrameVec::<Eci, Length>::from_raw(raw);
    let result: FrameVec<Ecef, Length> = rot.transform(&v);
    let expected = nalgebra::UnitQuaternion::from_axis_angle(&Vector3::z_axis(), angle)
        .transform_vector(&raw);
    assert!((result.into_raw() - expected).norm() < 1e-15);
}

#[test]
fn rotation_inverse() {
    let angle = 0.5;
    let rot = Rotation::<Eci, Ecef>::from_angle_z(angle);
    let v = FrameVec::<Eci, Velocity>::new(1.0, 2.0, 3.0);
    let v_ecef = rot.transform(&v);
    let v_back: FrameVec<Eci, Velocity> = rot.inverse().transform(&v_ecef);
    assert!((v.into_raw() - v_back.into_raw()).norm() < 1e-15);
}

#[test]
fn rotation_compose() {
    struct Mid;
    let r1 = Rotation::<Eci, Mid>::from_angle_z(0.3);
    let r2 = Rotation::<Mid, Ecef>::from_angle_z(0.7);
    let r_composed: Rotation<Eci, Ecef> = r1.then(&r2);
    let r_direct = Rotation::<Eci, Ecef>::from_angle_z(1.0);
    let v = FrameVec::<Eci, Length>::new(1.0, 2.0, 3.0);
    let diff = (r_composed.transform(&v).into_raw() - r_direct.transform(&v).into_raw()).norm();
    assert!(diff < 1e-14);
}

// ---------------------------------------------------------------------------
// UnitMat
// ---------------------------------------------------------------------------

#[test]
fn unitmat_mul_vec() {
    let raw_m = Matrix3::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
    let raw_v = Vector3::new(1.0, 2.0, 3.0);
    let expected = raw_m * raw_v;
    // UnitMat<Velocity, Length, 3, 3> * UnitVec<Length, 3> → UnitVec<Velocity, 3>
    let m = UnitMat::<Velocity, Length, 3, 3>::from_raw(raw_m);
    let v = UnitVec::<Length, 3>::from_raw(SVector::from([raw_v.x, raw_v.y, raw_v.z]));
    let result: UnitVec<Velocity, 3> = m * v;
    assert_eq!(result.into_raw().as_slice(), expected.as_slice());
}

#[test]
fn unitmat_mul_mat() {
    let raw_a = Matrix3::new(1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0);
    let raw_b = Matrix3::new(4.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 6.0);
    let expected = raw_a * raw_b;
    // UnitMat<Velocity, Force, 3, 3> * UnitMat<Force, Length, 3, 3> → UnitMat<Velocity, Length, 3, 3>
    let a = UnitMat::<Velocity, Force, 3, 3>::from_raw(raw_a);
    let b = UnitMat::<Force, Length, 3, 3>::from_raw(raw_b);
    let result: UnitMat<Velocity, Length, 3, 3> = a * b;
    assert_eq!(result.into_raw(), expected);
}

#[test]
fn unitmat_transpose() {
    let raw = Matrix3::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
    let m = UnitMat::<Velocity, Length, 3, 3>::from_raw(raw);
    let t: UnitMat<Length, Velocity, 3, 3> = m.transpose();
    assert_eq!(t.into_raw(), raw.transpose());
}

#[test]
fn unitmat_inverse() {
    let raw = Matrix3::new(1.0, 2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    let m = UnitMat::<Velocity, Length, 3, 3>::from_raw(raw);
    let inv: UnitMat<Length, Velocity, 3, 3> = m.try_inverse().unwrap();
    let raw_inv = raw.try_inverse().unwrap();
    assert!((inv.into_raw() - raw_inv).norm() < 1e-15);
}

#[test]
fn unitmat_identity() {
    let id = UnitMat::<Length, Length, 3, 3>::identity();
    let raw_v = SVector::<f64, 3>::new(1.0, 2.0, 3.0);
    let v = UnitVec::<Length, 3>::from_raw(raw_v);
    let result = id * v;
    assert_eq!(result.into_raw(), raw_v);
}
