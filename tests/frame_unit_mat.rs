//! Comprehensive tests for FrameUnitMat: frame-safe matrix operations.
//!
//! Covers physics, control, and math use cases with frame checking.

use nalgebra::{Matrix3, Vector3};
use uolgebra::aliases::*;
use uolgebra::block::{BlockMat2x2, BlockVec2};
use uolgebra::frame_unit_mat::FrameUnitMat;
use uolgebra::frame_vec::FrameVec;
use uolgebra::rotation::Rotation;
// Scalar used in some physics examples
use uolgebra::unit_mat::UnitMat;

// Frame markers
struct Eci;
struct Ecef;
struct Body;
struct Rsw; // Radial / Along-track / Cross-track

// ============================================================================
// 1. BASIC OPERATIONS — nalgebra cross-validation
// ============================================================================

#[test]
fn frame_mat_mul_vec_cross_validation() {
    let raw_m = Matrix3::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
    let raw_v = Vector3::new(1.0, 2.0, 3.0);
    let expected = raw_m * raw_v;

    let m = FrameUnitMat::<Eci, Velocity, Length, 3, 3>::from_raw_unchecked(raw_m);
    let v = FrameVec::<Eci, Length>::from_raw_unchecked(raw_v);
    let result: FrameVec<Eci, Velocity> = m * v;

    assert_eq!(result.into_raw(), expected);
}

#[test]
fn frame_mat_mul_mat_cross_validation() {
    let raw_a = Matrix3::new(1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0);
    let raw_b = Matrix3::new(4.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 6.0);
    let expected = raw_a * raw_b;

    let a = FrameUnitMat::<Body, Torque, AngularVelocity, 3, 3>::from_raw_unchecked(raw_a);
    let b = FrameUnitMat::<Body, AngularVelocity, Dimensionless, 3, 3>::from_raw_unchecked(raw_b);
    let result: FrameUnitMat<Body, Torque, Dimensionless, 3, 3> = a * b;

    assert_eq!(result.into_raw(), expected);
}

#[test]
fn frame_mat_transpose() {
    let raw = Matrix3::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
    let m = FrameUnitMat::<Eci, Velocity, Length, 3, 3>::from_raw_unchecked(raw);
    let t: FrameUnitMat<Eci, Length, Velocity, 3, 3> = m.transpose();
    assert_eq!(t.into_raw(), raw.transpose());
}

#[test]
fn frame_mat_inverse() {
    let raw = Matrix3::new(1.0, 2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    let m = FrameUnitMat::<Body, Torque, AngularVelocity, 3, 3>::from_raw_unchecked(raw);
    let inv: FrameUnitMat<Body, AngularVelocity, Torque, 3, 3> = m.try_inverse().unwrap();
    let raw_inv = raw.try_inverse().unwrap();
    assert!((inv.into_raw() - raw_inv).norm() < 1e-15);
}

#[test]
fn frame_mat_identity() {
    let id = FrameUnitMat::<Eci, Length, Length, 3, 3>::identity();
    let v = FrameVec::<Eci, Length>::new(1.0, 2.0, 3.0);
    let result = id * v;
    assert_eq!(result.into_raw(), v.into_raw());
}

#[test]
fn frame_mat_zero_cost() {
    // FrameUnitMat should have identical size to raw SMatrix
    assert_eq!(
        core::mem::size_of::<FrameUnitMat<Eci, Length, Velocity, 3, 3>>(),
        core::mem::size_of::<Matrix3<f64>>()
    );
}

// ============================================================================
// 2. FRAME SAFETY — frame mismatch prevented at compile time
//    (See also compile_fail/frame_mat_wrong_frame.rs)
// ============================================================================

#[test]
fn frame_mat_preserves_frame() {
    // Matrix in ECI frame operates on ECI vector → result is ECI
    let m = FrameUnitMat::<Eci, Velocity, Length, 3, 3>::from_raw_unchecked(Matrix3::identity());
    let v = FrameVec::<Eci, Length>::new(1.0, 2.0, 3.0);
    let result: FrameVec<Eci, Velocity> = m * v;
    assert_eq!(result.x(), 1.0);
}

// ============================================================================
// 3. PHYSICS: Orbital STM with frame tracking
// ============================================================================

/// State transition matrix (Keplerian, simplified) in ECI frame.
/// Φ maps ECI state at t₀ to ECI state at t₁.
#[test]
fn orbital_stm_in_eci() {
    let dt = 60.0;

    type StmEci = BlockMat2x2<
        FrameUnitMat<Eci, Length, Length, 3, 3>,
        FrameUnitMat<Eci, Length, Velocity, 3, 3>,
        FrameUnitMat<Eci, Velocity, Length, 3, 3>,
        FrameUnitMat<Eci, Velocity, Velocity, 3, 3>,
    >;
    type StateEci = BlockVec2<FrameVec<Eci, Length>, FrameVec<Eci, Velocity>>;

    let stm = StmEci::new(
        FrameUnitMat::from_raw_unchecked(Matrix3::identity()),
        FrameUnitMat::from_raw_unchecked(Matrix3::identity() * dt),
        FrameUnitMat::from_raw_unchecked(Matrix3::zeros()),
        FrameUnitMat::from_raw_unchecked(Matrix3::identity()),
    );

    let x0 = StateEci::new(
        FrameVec::<Eci, Length>::new(7000e3, 0.0, 0.0),
        FrameVec::<Eci, Velocity>::new(0.0, 7.5e3, 0.0),
    );

    let x1: StateEci = stm * x0;

    // Position should advance by v * dt in the y direction
    assert!((x1.upper.x() - 7000e3).abs() < 1e-6);
    assert!((x1.upper.y() - 7.5e3 * dt).abs() < 1e-6);
    // Velocity unchanged (no gravity gradient in simplified model)
    assert!((x1.lower.y() - 7.5e3).abs() < 1e-6);
}

/// Gravity gradient matrix (∂a/∂r) in ECI: maps position perturbation to
/// acceleration perturbation. Has dimension Acceleration/Length = 1/s².
#[test]
fn gravity_gradient_in_eci() {
    // Simplified: G = -(μ/r³) * (I - 3 r̂ r̂ᵀ)
    // For a satellite at [R, 0, 0]:
    //   G = diag(2μ/R³, -μ/R³, -μ/R³)
    let mu = 3.986e14; // m³/s²
    let r = 7000e3; // m
    let r3 = r * r * r;

    let g_raw = Matrix3::new(
        2.0 * mu / r3, 0.0, 0.0,
        0.0, -mu / r3, 0.0,
        0.0, 0.0, -mu / r3,
    );

    // Dimension: Acceleration / Length = (m/s²) / m = 1/s²
    // In UnitMat terms: DR = Acceleration, DC = Length
    let g = FrameUnitMat::<Eci, Acceleration, Length, 3, 3>::from_raw_unchecked(g_raw);

    // Apply to a position perturbation → get acceleration perturbation
    let dr = FrameVec::<Eci, Length>::new(100.0, 0.0, 0.0); // 100m radial
    let da: FrameVec<Eci, Acceleration> = g * dr;

    // Radial perturbation should cause outward acceleration (positive, tidal)
    assert!(da.x() > 0.0);
    assert!((da.x() - 2.0 * mu / r3 * 100.0).abs() < 1e-10);
}

// ============================================================================
// 4. PHYSICS: Inertia tensor and angular momentum
// ============================================================================

/// Inertia tensor in body frame: I · ω = L (angular momentum)
/// [kg·m²] · [rad/s] = [kg·m²/s]
#[test]
fn inertia_tensor_angular_momentum() {
    // Diagonal inertia tensor for a simple spacecraft
    let i_raw = Matrix3::new(
        100.0, 0.0, 0.0,
        0.0, 200.0, 0.0,
        0.0, 0.0, 150.0,
    );

    // Inertia: DR = AngularMomentum, DC = AngularVelocity
    // Because I * ω = L: [kg·m²] * [1/s] = [kg·m²/s]
    let inertia = FrameUnitMat::<Body, AngularMomentum, AngularVelocity, 3, 3>::from_raw_unchecked(i_raw);

    let omega = FrameVec::<Body, AngularVelocity>::new(0.1, 0.0, 0.0); // 0.1 rad/s about x

    let ang_mom: FrameVec<Body, AngularMomentum> = inertia * omega;

    // L = I * ω = [100 * 0.1, 0, 0] = [10, 0, 0] kg·m²/s
    assert!((ang_mom.x() - 10.0).abs() < 1e-15);
    assert!((ang_mom.y()).abs() < 1e-15);
}

/// Euler's equation: I · ω̇ = τ - ω × (I · ω)
///
/// The same inertia matrix I appears in two roles with different unit types:
/// 1. I · ω = L:  maps AngularVelocity → AngularMomentum  (I_vel)
/// 2. I · ω̇ = τ:  maps AngularAcceleration → Torque       (I_acc)
///
/// Both use the same numerical values but different type annotations.
/// Then I_acc⁻¹ maps Torque → AngularAcceleration.
///
/// ω × L has dimension AngularVelocity × AngularMomentum
///   = (1/s) × (kg·m²/s) = kg·m²/s² = Torque ✓
#[test]
fn euler_equation_torque() {
    let i_raw = Matrix3::new(100.0, 0.0, 0.0, 0.0, 200.0, 0.0, 0.0, 0.0, 150.0);

    // I for computing angular momentum: I · ω = L
    let i_vel = FrameUnitMat::<Body, AngularMomentum, AngularVelocity, 3, 3>::from_raw_unchecked(i_raw);
    // I for Euler equation: I · ω̇ = τ_net → I⁻¹ · τ_net = ω̇
    let i_acc = FrameUnitMat::<Body, Torque, AngularAcceleration, 3, 3>::from_raw_unchecked(i_raw);

    let omega = FrameVec::<Body, AngularVelocity>::new(0.1, 0.05, 0.0);
    let torque = FrameVec::<Body, Torque>::new(0.0, 0.0, 1.0); // 1 N·m about z

    // ω × (I · ω): cross product of AngularVelocity and AngularMomentum
    let i_omega: FrameVec<Body, AngularMomentum> = i_vel * omega;
    let gyroscopic: FrameVec<Body, Torque> = omega.cross(&i_omega);
    // AngularVelocity × AngularMomentum = (1/s) × (kg·m²/s) = kg·m²/s² = Torque ✓

    // Net torque: τ - ω × (I·ω)
    let net_torque: FrameVec<Body, Torque> = torque - gyroscopic;

    // ω̇ = I⁻¹ · τ_net
    let i_acc_inv = i_acc.try_inverse().unwrap();
    let omega_dot: FrameVec<Body, AngularAcceleration> = i_acc_inv * net_torque;

    // Verify against raw nalgebra computation
    let raw_omega = Vector3::new(0.1, 0.05, 0.0);
    let raw_torque = Vector3::new(0.0, 0.0, 1.0);
    let raw_gyro = raw_omega.cross(&(i_raw * raw_omega));
    let raw_omega_dot = i_raw.try_inverse().unwrap() * (raw_torque - raw_gyro);

    assert!((omega_dot.into_raw() - raw_omega_dot).norm() < 1e-12);
}

// ============================================================================
// 5. CONTROL: PD controller with gain matrices
// ============================================================================

/// PD attitude controller: τ = -Kp · θ_err - Kv · ω_err
///
/// Kp: position gain [N·m / rad] = [N·m] (since rad is dimensionless)
///   → FrameUnitMat<Body, Torque, Dimensionless, 3, 3>
///
/// Kv: velocity gain [N·m / (rad/s)] = [N·m·s]
///   → FrameUnitMat<Body, Torque, AngularVelocity, 3, 3>
///
/// Note: The "gain has no units" intuition is wrong — the gain matrix
/// has units of (output_dim / input_dim). uolgebra tracks this precisely.
#[test]
fn pd_attitude_controller() {
    // Gain matrices (diagonal for simplicity)
    let kp_val = 10.0; // N·m / rad
    let kv_val = 5.0;  // N·m·s / rad

    // Kp: Torque / Dimensionless → maps angle error to torque
    let kp = FrameUnitMat::<Body, Torque, Dimensionless, 3, 3>::from_raw_unchecked(
        Matrix3::identity() * kp_val,
    );

    // Kv: Torque / AngularVelocity → maps rate error to torque
    let kv = FrameUnitMat::<Body, Torque, AngularVelocity, 3, 3>::from_raw_unchecked(
        Matrix3::identity() * kv_val,
    );

    // Attitude error (small angle, dimensionless)
    let theta_err = FrameVec::<Body, Dimensionless>::new(0.1, -0.05, 0.02);

    // Angular velocity error
    let omega_err = FrameVec::<Body, AngularVelocity>::new(0.01, -0.005, 0.002);

    // τ = -Kp · θ_err - Kv · ω_err
    let torque_p: FrameVec<Body, Torque> = kp * theta_err;
    let torque_v: FrameVec<Body, Torque> = kv * omega_err;
    let torque: FrameVec<Body, Torque> = -torque_p - torque_v;

    // Verify numerically
    let expected_x = -(kp_val * 0.1 + kv_val * 0.01);
    assert!((torque.x() - expected_x).abs() < 1e-12);
}

/// Position-based force controller: F = -Kp · Δr - Kv · Δv
///
/// Kp: [N/m] = [kg/s²] → Force / Length
/// Kv: [N/(m/s)] = [kg/s] → Force / Velocity
#[test]
fn pd_position_controller() {
    let kp = FrameUnitMat::<Eci, Force, Length, 3, 3>::from_raw_unchecked(
        Matrix3::identity() * 0.5,
    );
    let kv = FrameUnitMat::<Eci, Force, Velocity, 3, 3>::from_raw_unchecked(
        Matrix3::identity() * 2.0,
    );

    let dr = FrameVec::<Eci, Length>::new(100.0, 0.0, 0.0);
    let dv = FrameVec::<Eci, Velocity>::new(1.0, 0.0, 0.0);

    let force: FrameVec<Eci, Force> = -(kp * dr) - (kv * dv);

    // F_x = -(0.5 * 100 + 2.0 * 1.0) = -52.0 N
    assert!((force.x() - (-52.0)).abs() < 1e-12);
}

/// Reaction wheel torque distribution matrix.
/// Maps desired body torque to individual wheel torques via pseudo-inverse.
/// If wheels are aligned with body axes, this is identity (simplified).
#[test]
fn reaction_wheel_distribution() {
    // Distribution matrix: maps body torque command → wheel torque commands
    // For 3 orthogonal wheels aligned to body axes: identity
    let dist = FrameUnitMat::<Body, Torque, Torque, 3, 3>::from_raw_unchecked(
        Matrix3::identity(),
    );

    let cmd_torque = FrameVec::<Body, Torque>::new(0.01, -0.02, 0.005);
    let wheel_torque: FrameVec<Body, Torque> = dist * cmd_torque;

    assert_eq!(wheel_torque.x(), 0.01);
    assert_eq!(wheel_torque.y(), -0.02);
}

// ============================================================================
// 6. EKF with frame-safe covariance propagation
// ============================================================================

#[test]
fn ekf_frame_safe_propagation() {
    type StmEci = BlockMat2x2<
        FrameUnitMat<Eci, Length, Length, 3, 3>,
        FrameUnitMat<Eci, Length, Velocity, 3, 3>,
        FrameUnitMat<Eci, Velocity, Length, 3, 3>,
        FrameUnitMat<Eci, Velocity, Velocity, 3, 3>,
    >;
    type CovEci = BlockMat2x2<
        FrameUnitMat<Eci, Length, Length, 3, 3>,
        FrameUnitMat<Eci, Length, Velocity, 3, 3>,
        FrameUnitMat<Eci, Velocity, Length, 3, 3>,
        FrameUnitMat<Eci, Velocity, Velocity, 3, 3>,
    >;
    type StateEci = BlockVec2<FrameVec<Eci, Length>, FrameVec<Eci, Velocity>>;

    let dt = 10.0;
    let phi = StmEci::new(
        FrameUnitMat::from_raw_unchecked(Matrix3::identity()),
        FrameUnitMat::from_raw_unchecked(Matrix3::identity() * dt),
        FrameUnitMat::from_raw_unchecked(Matrix3::zeros()),
        FrameUnitMat::from_raw_unchecked(Matrix3::identity()),
    );

    let p0 = CovEci::new(
        FrameUnitMat::from_raw_unchecked(Matrix3::identity() * 100.0),
        FrameUnitMat::from_raw_unchecked(Matrix3::zeros()),
        FrameUnitMat::from_raw_unchecked(Matrix3::zeros()),
        FrameUnitMat::from_raw_unchecked(Matrix3::identity() * 1.0),
    );

    let q = CovEci::new(
        FrameUnitMat::from_raw_unchecked(Matrix3::identity() * 0.1),
        FrameUnitMat::from_raw_unchecked(Matrix3::zeros()),
        FrameUnitMat::from_raw_unchecked(Matrix3::zeros()),
        FrameUnitMat::from_raw_unchecked(Matrix3::identity() * 0.01),
    );

    // State propagation
    let x0 = StateEci::new(
        FrameVec::new(7000e3, 0.0, 0.0),
        FrameVec::new(0.0, 7.5e3, 0.0),
    );
    let x1: StateEci = phi * x0;

    // Covariance: P₁ = Φ P₀ Φᵀ + Q
    let phi_p0: CovEci = phi * p0;
    let phi_t = phi.transpose();
    let p1: CovEci = phi_p0 * phi_t + q;

    // P_rr should grow due to velocity uncertainty coupling
    assert!(p1.a.as_raw()[(0, 0)] > 100.0);
    // Cross-covariance P_rv should be non-zero
    assert!(p1.b.as_raw()[(0, 0)].abs() > 0.0);
    // x1 position should advance
    assert!((x1.upper.y() - 7.5e3 * dt).abs() < 1e-6);
}

// ============================================================================
// 7. MATH: Rotation → FrameUnitMat and consistency
// ============================================================================

#[test]
fn rotation_as_frame_unit_mat() {
    let angle = 1.0;
    let rot = Rotation::<Eci, Ecef>::from_angle_z(angle);
    let rot_mat = rot.to_matrix(); // UnitMat<Dimensionless, Dimensionless, 3, 3>

    // Wrap as FrameUnitMat (this is a within-ECI operation)
    // Note: Rotation changes frames, but the rotation matrix itself can be
    // used within a frame for coordinate math.
    let _frame_mat = FrameUnitMat::<Eci, Dimensionless, Dimensionless, 3, 3>::from_unit_mat(&rot_mat);
}

// ============================================================================
// 8. CONVERSION: UnitMat ↔ FrameUnitMat
// ============================================================================

#[test]
fn unit_mat_to_frame_unit_mat() {
    let um = UnitMat::<Velocity, Length, 3, 3>::from_raw_unchecked(Matrix3::identity());
    let fum = FrameUnitMat::<Eci, Velocity, Length, 3, 3>::from_unit_mat(&um);
    assert_eq!(fum.into_raw(), Matrix3::identity());
}

#[test]
fn frame_unit_mat_to_unit_mat() {
    let fum = FrameUnitMat::<Body, Torque, AngularVelocity, 3, 3>::from_raw_unchecked(
        Matrix3::identity() * 5.0,
    );
    let um: UnitMat<Torque, AngularVelocity, 3, 3> = fum.to_unit_mat();
    assert_eq!(um.into_raw(), Matrix3::identity() * 5.0);
}

// ============================================================================
// 9. COMPOUND ASSIGNMENT
// ============================================================================

#[test]
fn frame_mat_add_assign() {
    let mut m = FrameUnitMat::<Eci, Length, Length, 3, 3>::from_raw_unchecked(Matrix3::identity());
    m += FrameUnitMat::from_raw_unchecked(Matrix3::identity());
    assert_eq!(m.into_raw(), Matrix3::identity() * 2.0);
}

// ============================================================================
// 10. RSW FRAME: along-track / cross-track operations
// ============================================================================

/// RSW (Radial/Along-track/Cross-track) frame force resolution.
/// Thruster force in body frame → resolve in RSW for orbit analysis.
#[test]
fn rsw_force_resolution() {
    // A force applied in RSW frame (e.g., along-track thrust for orbit raising)
    let thrust_rsw = FrameVec::<Rsw, Force>::new(0.0, 0.5, 0.0); // 0.5 N along-track

    // Kp gain in RSW: convert position error to force command
    let kp_rsw = FrameUnitMat::<Rsw, Force, Length, 3, 3>::from_raw_unchecked(
        Matrix3::identity() * 0.001,
    );

    let dr_rsw = FrameVec::<Rsw, Length>::new(0.0, 1000.0, 0.0); // 1km along-track error
    let correction: FrameVec<Rsw, Force> = kp_rsw * dr_rsw;

    let total_force = thrust_rsw + correction;
    assert!((total_force.y() - 1.5).abs() < 1e-12); // 0.5 + 0.001*1000 = 1.5 N
}
