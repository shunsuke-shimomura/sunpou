//! Frame rotation (transformation between coordinate frames).

use core::marker::PhantomData;
use nalgebra::{Matrix3, UnitQuaternion, Vector3};

use crate::frame_vec::FrameVec;

/// A rotation from frame `From` to frame `To`.
///
/// Internally stores a unit quaternion. Zero overhead over `UnitQuaternion<f64>`.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct Rotation<From, To> {
    quat: UnitQuaternion<f64>,
    _marker: PhantomData<(From, To)>,
}

impl<F1, F2> core::fmt::Debug for Rotation<F1, F2> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Rotation({:?})", self.quat)
    }
}

impl<F1, F2> Rotation<F1, F2> {
    /// Create from a raw unit quaternion.
    #[inline(always)]
    pub fn from_raw_unchecked(quat: UnitQuaternion<f64>) -> Self {
        Self {
            quat,
            _marker: PhantomData,
        }
    }

    /// Create from a 3×3 rotation matrix.
    #[inline(always)]
    pub fn from_matrix_unchecked(mat: &Matrix3<f64>) -> Self {
        Self {
            quat: UnitQuaternion::from_matrix(mat),
            _marker: PhantomData,
        }
    }

    /// Create a rotation about the Z axis by the given angle (radians).
    #[inline(always)]
    pub fn from_angle_z(angle: f64) -> Self {
        Self {
            quat: UnitQuaternion::from_axis_angle(&Vector3::z_axis(), angle),
            _marker: PhantomData,
        }
    }

    /// Create an identity rotation (same frame).
    #[inline(always)]
    pub fn identity() -> Self {
        Self {
            quat: UnitQuaternion::identity(),
            _marker: PhantomData,
        }
    }

    /// Get the underlying unit quaternion.
    #[inline(always)]
    pub fn into_raw(self) -> UnitQuaternion<f64> {
        self.quat
    }

    /// Transform a vector from frame `F1` to frame `F2`.
    /// Dimension is preserved.
    #[inline(always)]
    pub fn transform<D, P>(&self, v: &FrameVec<F1, D, P>) -> FrameVec<F2, D, P> {
        FrameVec::from_raw_unchecked(self.quat.transform_vector(v.as_raw()))
    }

    /// Inverse rotation: `F2 → F1`.
    #[inline(always)]
    pub fn inverse(&self) -> Rotation<F2, F1> {
        Rotation::from_raw_unchecked(self.quat.inverse())
    }

    /// Compose: `(F1→F2).then(F2→F3) = F1→F3`.
    #[inline(always)]
    pub fn then<F3>(&self, other: &Rotation<F2, F3>) -> Rotation<F1, F3> {
        Rotation::from_raw_unchecked(other.quat * self.quat)
    }

    /// Convert to a 3×3 rotation matrix (dimensionless).
    #[inline(always)]
    pub fn to_matrix(
        &self,
    ) -> crate::unit_mat::UnitMat<
        crate::aliases::Dimensionless,
        crate::aliases::Dimensionless,
        3,
        3,
    > {
        crate::unit_mat::UnitMat::from_raw_unchecked(*self.quat.to_rotation_matrix().matrix())
    }
}
