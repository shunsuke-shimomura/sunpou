//! Static assertions: verify that uolgebra types have identical memory layout
//! to the underlying nalgebra/f64 types (zero-cost abstraction).

use core::mem::{align_of, size_of};
use nalgebra::{Matrix3, SMatrix, SVector, UnitQuaternion, Vector3};
use uolgebra::aliases::*;
use uolgebra::frame_vec::FrameVec;
use uolgebra::rotation::Rotation;
use uolgebra::scalar::Scalar;
use uolgebra::unit_mat::UnitMat;
use uolgebra::unit_vec::UnitVec;

struct Eci;
struct Ecef;

// ---------------------------------------------------------------------------
// Scalar<D> == f64
// ---------------------------------------------------------------------------

#[test]
fn scalar_size() {
    assert_eq!(size_of::<Scalar<Length>>(), size_of::<f64>());
    assert_eq!(align_of::<Scalar<Length>>(), align_of::<f64>());
}

#[test]
fn scalar_all_dims_same_size() {
    assert_eq!(size_of::<Scalar<Mass>>(), size_of::<f64>());
    assert_eq!(size_of::<Scalar<Velocity>>(), size_of::<f64>());
    assert_eq!(size_of::<Scalar<Force>>(), size_of::<f64>());
    assert_eq!(size_of::<Scalar<Energy>>(), size_of::<f64>());
    assert_eq!(size_of::<Scalar<Dimensionless>>(), size_of::<f64>());
}

// ---------------------------------------------------------------------------
// UnitVec<D, N> == SVector<f64, N>
// ---------------------------------------------------------------------------

#[test]
fn unitvec3_size() {
    assert_eq!(
        size_of::<UnitVec<Length, 3>>(),
        size_of::<SVector<f64, 3>>()
    );
    assert_eq!(
        align_of::<UnitVec<Length, 3>>(),
        align_of::<SVector<f64, 3>>()
    );
}

#[test]
fn unitvec6_size() {
    assert_eq!(
        size_of::<UnitVec<Velocity, 6>>(),
        size_of::<SVector<f64, 6>>()
    );
}

// ---------------------------------------------------------------------------
// FrameVec<F, D> == Vector3<f64>
// ---------------------------------------------------------------------------

#[test]
fn framevec_size() {
    assert_eq!(
        size_of::<FrameVec<Eci, Length>>(),
        size_of::<Vector3<f64>>()
    );
    assert_eq!(
        align_of::<FrameVec<Eci, Length>>(),
        align_of::<Vector3<f64>>()
    );
    // Different frame, same size
    assert_eq!(
        size_of::<FrameVec<Ecef, Velocity>>(),
        size_of::<Vector3<f64>>()
    );
}

// ---------------------------------------------------------------------------
// UnitMat<DR, DC, R, C> == SMatrix<f64, R, C>
// ---------------------------------------------------------------------------

#[test]
fn unitmat3x3_size() {
    assert_eq!(
        size_of::<UnitMat<Velocity, Length, 3, 3>>(),
        size_of::<Matrix3<f64>>()
    );
    assert_eq!(
        align_of::<UnitMat<Velocity, Length, 3, 3>>(),
        align_of::<Matrix3<f64>>()
    );
}

#[test]
fn unitmat6x6_size() {
    assert_eq!(
        size_of::<UnitMat<Length, Velocity, 6, 6>>(),
        size_of::<SMatrix<f64, 6, 6>>()
    );
}

// ---------------------------------------------------------------------------
// Rotation<F1, F2> == UnitQuaternion<f64>
// ---------------------------------------------------------------------------

#[test]
fn rotation_size() {
    assert_eq!(
        size_of::<Rotation<Eci, Ecef>>(),
        size_of::<UnitQuaternion<f64>>()
    );
    assert_eq!(
        align_of::<Rotation<Eci, Ecef>>(),
        align_of::<UnitQuaternion<f64>>()
    );
}
