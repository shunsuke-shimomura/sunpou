//! Frame-tagged unit direction vector on the 3D unit sphere.
//!
//! `FrameDirection<F>` is a unit vector in coordinate frame `F`, representing
//! a direction (e.g., magnetic field direction in body frame, sun direction in ECI).
//!
//! The error manifold is 2D (tangent plane to the sphere), making this suitable
//! for UKF estimation where the sigma points live in the 2D tangent space.
//!
//! This is the sunpou counterpart of structured-estimator's `Direction` type,
//! with compile-time frame safety added.

use core::marker::PhantomData;
use nalgebra::{Matrix3, Unit, UnitVector3, Vector2, Vector3};

/// A unit direction vector in coordinate frame `F`.
///
/// Internally stores an orthonormal basis (ONB) where the first column is
/// the direction vector and columns 2-3 span the tangent plane.
pub struct FrameDirection<F> {
    basis: Matrix3<f64>,
    _frame: PhantomData<F>,
}

impl<F> Clone for FrameDirection<F> {
    fn clone(&self) -> Self {
        Self { basis: self.basis, _frame: PhantomData }
    }
}

impl<F> core::fmt::Debug for FrameDirection<F> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "FrameDirection({:?})", self.basis.column(0).into_owned())
    }
}

impl<F> FrameDirection<F> {
    /// Create from a unit direction vector.
    pub fn from_dir(dir: UnitVector3<f64>) -> Self {
        let nvec = dir.into_inner();

        // Choose a reference vector not parallel to dir
        let r = if nvec.x.abs() < 0.5 {
            Vector3::new(1.0, 0.0, 0.0)
        } else {
            Vector3::new(0.0, 1.0, 0.0)
        };

        let t1 = nvec.cross(&r).normalize();
        let t2 = nvec.cross(&t1).normalize();

        Self {
            basis: Matrix3::from_columns(&[nvec, t1, t2]),
            _frame: PhantomData,
        }
    }

    /// Get the unit direction vector.
    #[inline(always)]
    pub fn dir(&self) -> UnitVector3<f64> {
        Unit::new_normalize(self.basis.column(0).into_owned())
    }

    /// Get the 3×2 tangent basis (columns 2-3 of the ONB).
    #[inline(always)]
    pub fn basis_2d(&self) -> nalgebra::Matrix3x2<f64> {
        self.basis.fixed_columns::<2>(1).into_owned()
    }

    /// Get the underlying raw basis matrix.
    #[inline(always)]
    pub fn basis(&self) -> &Matrix3<f64> {
        &self.basis
    }

    /// Apply a 2D tangent-space perturbation (for sigma point operations).
    pub fn perturb(&self, sigma: &Vector2<f64>) -> Self {
        let theta_3d = self.basis_2d() * *sigma;
        if theta_3d.norm() < 1e-10 {
            self.clone()
        } else {
            let axis = theta_3d / theta_3d.norm();
            let angle = theta_3d.norm();
            let cos_a = nalgebra::ComplexField::cos(angle);
            let sin_a = nalgebra::ComplexField::sin(angle);
            let rotated = cos_a * self.dir().into_inner() + sin_a * axis;
            Self::from_dir(Unit::new_normalize(rotated))
        }
    }

    /// Compute 2D tangent-space error from `self` to `criteria`.
    pub fn error_from(&self, criteria: &Self) -> Vector2<f64> {
        let dot = self.dir().dot(&criteria.dir()).clamp(-1.0, 1.0);
        let angle = nalgebra::ComplexField::acos(dot);
        let u = self.dir().into_inner() - dot * criteria.dir().into_inner();
        let u_norm = u.norm();
        if u_norm < 1e-10 {
            Vector2::zeros()
        } else {
            let axis = u / u_norm;
            let axisangle = axis * angle;
            criteria.basis_2d().transpose() * axisangle
        }
    }

    /// Strip the frame tag (for interop with frame-less Direction).
    #[inline(always)]
    pub fn into_raw_basis(self) -> Matrix3<f64> {
        self.basis
    }

    /// Create from a raw basis matrix (caller ensures correct frame).
    #[inline(always)]
    pub fn from_raw_basis(basis: Matrix3<f64>) -> Self {
        Self { basis, _frame: PhantomData }
    }
}

impl<F> Default for FrameDirection<F> {
    fn default() -> Self {
        Self {
            basis: Matrix3::identity(),
            _frame: PhantomData,
        }
    }
}

// ---- Rotation support: Rotation3 * FrameDirection → FrameDirection ----

impl<F> core::ops::Mul<FrameDirection<F>> for nalgebra::Rotation3<f64> {
    type Output = FrameDirection<F>;

    fn mul(self, rhs: FrameDirection<F>) -> FrameDirection<F> {
        let new_basis = self * rhs.basis;
        FrameDirection::from_raw_basis(new_basis)
    }
}
