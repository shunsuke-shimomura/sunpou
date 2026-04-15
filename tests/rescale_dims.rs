//! Tests for rescale_dims: safe dimensional reinterpretation of matrices.

use nalgebra::Matrix3;
use sunpou::aliases::*;
use sunpou::frame_unit_mat::FrameUnitMat;
use sunpou::frame_vec::FrameVec;
use sunpou::unit_mat::UnitMat;

struct Body;
struct Eci;

// ============================================================================
// 1. Inertia tensor: the motivating example
// ============================================================================

/// I · ω = L  ↔  I · ω̇ = τ
/// Same matrix, rescaled by InvTime on both dimensions.
#[test]
fn inertia_tensor_rescale() {
    let i_raw = Matrix3::new(100.0, 0.0, 0.0, 0.0, 200.0, 0.0, 0.0, 0.0, 150.0);

    // Start with the I·ω = L interpretation
    let i_vel = FrameUnitMat::<Body, AngularMomentum, AngularVelocity, 3, 3>::from_raw_unchecked(i_raw);

    // Rescale to I·ω̇ = τ interpretation
    let i_acc: FrameUnitMat<Body, Torque, AngularAcceleration, 3, 3> =
        i_vel.rescale_dims::<InvTime>();

    // Numerical values are identical
    assert_eq!(i_vel.into_raw(), i_acc.into_raw());

    // Both interpretations work correctly
    let omega = FrameVec::<Body, AngularVelocity>::new(0.1, 0.0, 0.0);
    let ang_mom: FrameVec<Body, AngularMomentum> = i_vel * omega;
    assert!((ang_mom.x() - 10.0).abs() < 1e-15);

    let omega_dot = FrameVec::<Body, AngularAcceleration>::new(0.1, 0.0, 0.0);
    let torque: FrameVec<Body, Torque> = i_acc * omega_dot;
    assert!((torque.x() - 10.0).abs() < 1e-15);
}

/// Rescale and then use inverse for Euler equation
#[test]
fn inertia_rescale_then_inverse() {
    let i_raw = Matrix3::new(100.0, 0.0, 0.0, 0.0, 200.0, 0.0, 0.0, 0.0, 150.0);

    let i_vel = FrameUnitMat::<Body, AngularMomentum, AngularVelocity, 3, 3>::from_raw_unchecked(i_raw);

    // Rescale to torque interpretation, then invert
    let i_acc: FrameUnitMat<Body, Torque, AngularAcceleration, 3, 3> =
        i_vel.rescale_dims::<InvTime>();
    let i_acc_inv = i_acc.try_inverse().unwrap();
    // i_acc_inv: FrameUnitMat<Body, AngularAcceleration, Torque, 3, 3>

    let net_torque = FrameVec::<Body, Torque>::new(0.0, 0.0, 1.0);
    let omega_dot: FrameVec<Body, AngularAcceleration> = i_acc_inv * net_torque;

    // ω̇_z = τ_z / I_zz = 1.0 / 150.0
    assert!((omega_dot.z() - 1.0 / 150.0).abs() < 1e-15);
}

// ============================================================================
// 2. Round-trip: rescale and rescale back
// ============================================================================

#[test]
fn rescale_roundtrip() {
    let i_raw = Matrix3::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
    let original = FrameUnitMat::<Body, AngularMomentum, AngularVelocity, 3, 3>::from_raw_unchecked(i_raw);

    // Rescale forward: ×InvTime
    let rescaled: FrameUnitMat<Body, Torque, AngularAcceleration, 3, 3> =
        original.rescale_dims::<InvTime>();

    // Rescale back: ×Time
    let back: FrameUnitMat<Body, AngularMomentum, AngularVelocity, 3, 3> =
        rescaled.rescale_dims::<Time>();

    assert_eq!(original.into_raw(), back.into_raw());
}

// ============================================================================
// 3. UnitMat rescale_dims (frame-less version)
// ============================================================================

#[test]
fn unit_mat_rescale() {
    let raw = Matrix3::identity() * 42.0;

    let m = UnitMat::<AngularMomentum, AngularVelocity, 3, 3>::from_raw_unchecked(raw);
    let m2: UnitMat<Torque, AngularAcceleration, 3, 3> = m.rescale_dims::<InvTime>();

    assert_eq!(m.into_raw(), m2.into_raw());
}

// ============================================================================
// 4. Other physics examples of rescaling
// ============================================================================

/// Spring constant: F = -k · x
/// k maps Length → Force: UnitMat<Force, Length>
/// The same k can map Velocity → Power/Length... but more usefully:
/// Rescale by InvTime: maps Velocity → Force*InvTime... not directly useful.
///
/// Better example: mass matrix in structural dynamics.
/// M · ẍ = F  ↔  M · ẋ = impulse (M·v = p)
/// M maps Acceleration → Force, or Velocity → Momentum
#[test]
fn mass_matrix_rescale() {
    let m_raw = Matrix3::identity() * 10.0; // 10 kg (diagonal)

    // M · a = F
    let m_force = FrameUnitMat::<Eci, Force, Acceleration, 3, 3>::from_raw_unchecked(m_raw);

    // Rescale: both ×Time → M · v = p (momentum)
    let m_momentum: FrameUnitMat<Eci, Momentum, Velocity, 3, 3> =
        m_force.rescale_dims::<Time>();

    assert_eq!(m_force.into_raw(), m_momentum.into_raw());

    // Use: F = M · a
    let accel = FrameVec::<Eci, Acceleration>::new(0.0, 0.0, 9.8);
    let force: FrameVec<Eci, Force> = m_force * accel;
    assert!((force.z() - 98.0).abs() < 1e-12);

    // Use: p = M · v
    let vel = FrameVec::<Eci, Velocity>::new(1.0, 0.0, 0.0);
    let momentum: FrameVec<Eci, Momentum> = m_momentum * vel;
    assert!((momentum.x() - 10.0).abs() < 1e-12);
}

/// Damping matrix: F = -C · v, or equivalently impulse = -C · x
/// C maps Velocity → Force: FrameUnitMat<F, Force, Velocity>
/// Rescale by Time: maps Length → Momentum (same numerical damping)
#[test]
fn damping_matrix_rescale() {
    let c_raw = Matrix3::identity() * 5.0; // 5 N·s/m

    let c_vel = FrameUnitMat::<Eci, Force, Velocity, 3, 3>::from_raw_unchecked(c_raw);
    let c_pos: FrameUnitMat<Eci, Momentum, Length, 3, 3> = c_vel.rescale_dims::<Time>();

    assert_eq!(c_vel.into_raw(), c_pos.into_raw());
}

// ============================================================================
// 5. Rescale preserves frame
// ============================================================================

#[test]
fn rescale_preserves_frame() {
    let m = FrameUnitMat::<Body, AngularMomentum, AngularVelocity, 3, 3>::from_raw_unchecked(
        Matrix3::identity(),
    );

    // After rescaling, frame is still Body
    let m2: FrameUnitMat<Body, Torque, AngularAcceleration, 3, 3> =
        m.rescale_dims::<InvTime>();

    // Can multiply with Body-frame vectors
    let omega_dot = FrameVec::<Body, AngularAcceleration>::new(1.0, 0.0, 0.0);
    let _torque: FrameVec<Body, Torque> = m2 * omega_dot;
}

// ============================================================================
// 6. Full Euler equation using rescale_dims
// ============================================================================

/// Complete Euler equation test using only one from_raw_unchecked call
/// and rescale_dims for the second interpretation.
#[test]
fn euler_equation_with_rescale() {
    let i_raw = Matrix3::new(100.0, 0.0, 0.0, 0.0, 200.0, 0.0, 0.0, 0.0, 150.0);

    // Define I once as I·ω = L
    let i_vel = FrameUnitMat::<Body, AngularMomentum, AngularVelocity, 3, 3>::from_raw_unchecked(i_raw);

    // Derive the Euler equation interpretation via rescale
    let i_acc_inv = i_vel
        .rescale_dims::<InvTime>() // → FrameUnitMat<Body, Torque, AngularAcceleration>
        .try_inverse()             // → FrameUnitMat<Body, AngularAcceleration, Torque>
        .unwrap();

    let omega = FrameVec::<Body, AngularVelocity>::new(0.1, 0.05, 0.0);
    let external_torque = FrameVec::<Body, Torque>::new(0.0, 0.0, 1.0);

    // Gyroscopic term: ω × (I·ω)
    let ang_mom: FrameVec<Body, AngularMomentum> = i_vel * omega;
    let gyro: FrameVec<Body, Torque> = omega.cross(&ang_mom);

    // ω̇ = I⁻¹ · (τ_ext - ω × L)
    let omega_dot: FrameVec<Body, AngularAcceleration> = i_acc_inv * (external_torque - gyro);

    // Cross-validate with raw nalgebra
    let raw_omega = nalgebra::Vector3::new(0.1, 0.05, 0.0);
    let raw_tau = nalgebra::Vector3::new(0.0, 0.0, 1.0);
    let raw_gyro = raw_omega.cross(&(i_raw * raw_omega));
    let raw_omega_dot = i_raw.try_inverse().unwrap() * (raw_tau - raw_gyro);

    assert!((omega_dot.into_raw() - raw_omega_dot).norm() < 1e-12);
}
