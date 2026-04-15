//! Tests for ElemMat and FrameElemMat: element-dimension matrix model.
//!
//! Demonstrates that the same matrix object works with any input dimension,
//! with output dimension inferred automatically.

use nalgebra::{Matrix3, Vector3};
use sunpou::aliases::*;
use sunpou::block::{BlockMat2x2, BlockVec2};
use sunpou::elem_mat::ElemMat;
use sunpou::frame_elem_mat::FrameElemMat;
use sunpou::frame_vec::FrameVec;
use sunpou::unit_vec::UnitVec;

struct Eci;
struct Body;

// ============================================================================
// 1. THE KEY FEATURE: same matrix, different inputs, automatic output inference
// ============================================================================

/// Inertia tensor: ONE object, TWO uses, ZERO rescale_dims.
#[test]
fn inertia_tensor_automatic_inference() {
    let i_raw = Matrix3::new(100.0, 0.0, 0.0, 0.0, 200.0, 0.0, 0.0, 0.0, 150.0);
    let inertia = FrameElemMat::<Body, MomentOfInertia, 3, 3>::from_raw_unchecked(i_raw);

    // Use 1: I * ω = L (MomentOfInertia * AngularVelocity = AngularMomentum)
    let omega = FrameVec::<Body, AngularVelocity>::new(0.1, 0.0, 0.0);
    let ang_mom: FrameVec<Body, AngularMomentum> = inertia * omega;
    assert!((ang_mom.x() - 10.0).abs() < 1e-15);

    // Use 2: I * ω̇ = τ (MomentOfInertia * AngularAcceleration = Torque)
    let omega_dot = FrameVec::<Body, AngularAcceleration>::new(0.0, 0.0, 0.5);
    let torque: FrameVec<Body, Torque> = inertia * omega_dot;
    assert!((torque.z() - 75.0).abs() < 1e-15); // 150 * 0.5

    // Use 3: I⁻¹ * τ = ω̇ (1/MomentOfInertia * Torque = AngularAcceleration)
    let i_inv = inertia.try_inverse().unwrap();
    let omega_dot_back: FrameVec<Body, AngularAcceleration> = i_inv * torque;
    assert!((omega_dot_back.z() - 0.5).abs() < 1e-14);

    // Use 4: I⁻¹ * L = ω (1/MomentOfInertia * AngularMomentum = AngularVelocity)
    let omega_back: FrameVec<Body, AngularVelocity> = i_inv * ang_mom;
    assert!((omega_back.x() - 0.1).abs() < 1e-14);
}

/// Mass matrix: M * a = F, M * v = p, same object.
#[test]
fn mass_matrix_automatic_inference() {
    let mass_val = 10.0;
    let m = FrameElemMat::<Eci, Mass, 3, 3>::from_raw_unchecked(Matrix3::identity() * mass_val);

    // M * a = F
    let accel = FrameVec::<Eci, Acceleration>::new(0.0, 0.0, 9.8);
    let force: FrameVec<Eci, Force> = m * accel;
    assert!((force.z() - 98.0).abs() < 1e-12);

    // M * v = p (momentum)
    let vel = FrameVec::<Eci, Velocity>::new(100.0, 0.0, 0.0);
    let momentum: FrameVec<Eci, Momentum> = m * vel;
    assert!((momentum.x() - 1000.0).abs() < 1e-12);
}

// ============================================================================
// 2. EULER EQUATION — full test with single from_raw_unchecked
// ============================================================================

#[test]
fn euler_equation_complete() {
    let i_raw = Matrix3::new(100.0, 0.0, 0.0, 0.0, 200.0, 0.0, 0.0, 0.0, 150.0);
    let inertia = FrameElemMat::<Body, MomentOfInertia, 3, 3>::from_raw_unchecked(i_raw);
    let i_inv = inertia.try_inverse().unwrap();

    let omega = FrameVec::<Body, AngularVelocity>::new(0.1, 0.05, 0.0);
    let ext_torque = FrameVec::<Body, Torque>::new(0.0, 0.0, 1.0);

    // I * ω = L
    let ang_mom: FrameVec<Body, AngularMomentum> = inertia * omega;

    // ω × L = gyroscopic torque
    let gyro: FrameVec<Body, Torque> = omega.cross(&ang_mom);

    // ω̇ = I⁻¹ * (τ_ext - gyro)
    let omega_dot: FrameVec<Body, AngularAcceleration> = i_inv * (ext_torque - gyro);

    // Cross-validate
    let raw_omega = Vector3::new(0.1, 0.05, 0.0);
    let raw_tau = Vector3::new(0.0, 0.0, 1.0);
    let raw_gyro = raw_omega.cross(&(i_raw * raw_omega));
    let raw_omega_dot = i_raw.try_inverse().unwrap() * (raw_tau - raw_gyro);

    assert!((omega_dot.into_raw() - raw_omega_dot).norm() < 1e-12);
}

// ============================================================================
// 3. PD CONTROLLER — gain dimensions inferred automatically
// ============================================================================

#[test]
fn pd_attitude_controller_elem() {
    // Kp element dim = Torque (since angle error is Dimensionless)
    // Kp * θ_err(Dimensionless) → Torque * Dimensionless = Torque
    let kp = FrameElemMat::<Body, Torque, 3, 3>::from_raw_unchecked(
        Matrix3::identity() * 10.0,
    );

    // Kv element dim = AngularMomentum = Torque * Time
    // Kv * ω_err(AngularVelocity=1/s) → AngularMomentum * (1/s) = Torque
    let kv = FrameElemMat::<Body, AngularMomentum, 3, 3>::from_raw_unchecked(
        Matrix3::identity() * 5.0,
    );

    let theta_err = FrameVec::<Body, Dimensionless>::new(0.1, -0.05, 0.02);
    let omega_err = FrameVec::<Body, AngularVelocity>::new(0.01, -0.005, 0.002);

    // Both produce Torque — type inference handles it
    let torque_p: FrameVec<Body, Torque> = kp * theta_err;
    let torque_v: FrameVec<Body, Torque> = kv * omega_err;
    let cmd: FrameVec<Body, Torque> = -torque_p - torque_v;

    let expected_x = -(10.0 * 0.1 + 5.0 * 0.01);
    assert!((cmd.x() - expected_x).abs() < 1e-12);
}

#[test]
fn pd_position_controller_elem() {
    // Kp: Force / Length → element dim = Force/Length
    // But Force/Length = kg/s², let's use a type alias approach
    // Kp * Δr(Length) → (Force/Length) * Length = Force  ← but we need DimDiv for element dim
    // Alternative: think of Kp as having element dim = Acceleration * Mass / Length
    // Actually simpler: Kp_raw = 0.5 N/m
    // Use ElemMat directly — the element dim must make output = Force when input = Length
    // So E * Length = Force → E = Force / Length = Acceleration (since F=ma, F/L = a * m/L... no)
    // Actually: Force = kg·m/s², Length = m, so Force/Length = kg/s² ← no standard alias
    // Let's just define the gain with the right type:

    // F = -Kp * dr - Kv * dv
    // Kp: element dim = Force/Length = kg/s² (no alias, use Dim directly)
    // Kv: element dim = Force/Velocity = kg/s (= Mass * InvTime... = Mass actually wait)
    // Force/Velocity = (kg·m/s²)/(m/s) = kg/s

    // Actually, for position control it's cleaner to split:
    // F = -kp * dr means the matrix elements have unit [N/m]
    // We use Mass (kg) for Kv since F/v = kg·m/s² / (m/s) = kg/s... hmm

    // Let's just compute numerically and verify:
    let kp_raw = Matrix3::identity() * 0.5; // 0.5 N/m
    let kv_raw = Matrix3::identity() * 2.0; // 2.0 N·s/m

    // Using ElemMat: the key insight is what E to pick.
    // Kp * Length = Force → E_kp = Force / Length
    // We need DimDiv<Force, Length> as the element type.
    // Force/Length = Dim<P1,P1,N2> / Dim<P1,Z0,Z0> = Dim<Z0,P1,N2> (= Mass * InvTime²)
    type ForcePerLength = <Force as sunpou::dim::DimDivide<Length>>::Output;
    type ForcePerVelocity = <Force as sunpou::dim::DimDivide<Velocity>>::Output;

    let kp = FrameElemMat::<Eci, ForcePerLength, 3, 3>::from_raw_unchecked(kp_raw);
    let kv = FrameElemMat::<Eci, ForcePerVelocity, 3, 3>::from_raw_unchecked(kv_raw);

    let dr = FrameVec::<Eci, Length>::new(100.0, 0.0, 0.0);
    let dv = FrameVec::<Eci, Velocity>::new(1.0, 0.0, 0.0);

    let force: FrameVec<Eci, Force> = -(kp * dr) - (kv * dv);
    assert!((force.x() - (-52.0)).abs() < 1e-12);
}

// ============================================================================
// 4. TRANSPOSE AND INVERSE — element dim correctly inverts
// ============================================================================

#[test]
fn transpose_inverts_elem_dim() {
    let raw = Matrix3::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
    let m = ElemMat::<Mass, 3, 3>::from_raw_unchecked(raw);

    // Transpose: E → 1/E
    let mt = m.transpose();
    assert_eq!(mt.into_raw(), raw.transpose());

    // If we multiply mt by a Force vector, output = (1/Mass) * Force = Acceleration
    let f = UnitVec::<Force, 3>::from_raw_unchecked(nalgebra::SVector::from([1.0, 0.0, 0.0]));
    let _a: UnitVec<Acceleration, 3> = mt * f;
}

#[test]
fn inverse_inverts_elem_dim() {
    let raw = Matrix3::new(2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0);
    let m = ElemMat::<Mass, 3, 3>::from_raw_unchecked(raw);
    let m_inv = m.try_inverse().unwrap();

    // M * v = p, M⁻¹ * p = v
    let p = UnitVec::<Momentum, 3>::from_raw_unchecked(nalgebra::SVector::from([6.0, 9.0, 12.0]));
    let v: UnitVec<Velocity, 3> = m_inv * p;
    assert!((v[0] - 3.0).abs() < 1e-14);
    assert!((v[1] - 3.0).abs() < 1e-14);
    assert!((v[2] - 3.0).abs() < 1e-14);
}

// ============================================================================
// 5. BLOCK MATRIX with FrameElemMat
// ============================================================================

#[test]
fn orbital_stm_frame_elem() {
    let dt = 60.0;

    // STM blocks: each has its element dimension
    // ∂r/∂r₀ = I (Dimensionless)
    // ∂r/∂v₀ = dt*I (element dim = Time, since Length/Velocity = Time)
    // ∂v/∂r₀ = 0 (element dim = InvTime, since Velocity/Length = 1/s)
    // ∂v/∂v₀ = I (Dimensionless)
    type StmEci = BlockMat2x2<
        FrameElemMat<Eci, Dimensionless, 3, 3>,
        FrameElemMat<Eci, Time, 3, 3>,
        FrameElemMat<Eci, InvTime, 3, 3>,
        FrameElemMat<Eci, Dimensionless, 3, 3>,
    >;
    type StateEci = BlockVec2<FrameVec<Eci, Length>, FrameVec<Eci, Velocity>>;

    let stm = StmEci::new(
        FrameElemMat::from_raw_unchecked(Matrix3::identity()),
        FrameElemMat::from_raw_unchecked(Matrix3::identity() * dt),
        FrameElemMat::from_raw_unchecked(Matrix3::zeros()),
        FrameElemMat::from_raw_unchecked(Matrix3::identity()),
    );

    let x0 = StateEci::new(
        FrameVec::new(7000e3, 0.0, 0.0),
        FrameVec::new(0.0, 7.5e3, 0.0),
    );

    let x1: StateEci = stm * x0;
    assert!((x1.upper.y() - 7.5e3 * dt).abs() < 1e-6);
}

// ============================================================================
// 6. ZERO-COST VERIFICATION
// ============================================================================

#[test]
fn elem_mat_zero_cost() {
    assert_eq!(
        core::mem::size_of::<ElemMat<Mass, 3, 3>>(),
        core::mem::size_of::<Matrix3<f64>>(),
    );
    assert_eq!(
        core::mem::size_of::<FrameElemMat<Body, MomentOfInertia, 3, 3>>(),
        core::mem::size_of::<Matrix3<f64>>(),
    );
}

// ============================================================================
// 7. IDENTITY AND DIMENSIONLESS
// ============================================================================

#[test]
fn dimensionless_identity() {
    let id = FrameElemMat::<Eci, Dimensionless, 3, 3>::identity();
    let v = FrameVec::<Eci, Length>::new(1.0, 2.0, 3.0);
    // Dimensionless * Length = Length
    let result: FrameVec<Eci, Length> = id * v;
    assert_eq!(result.into_raw(), v.into_raw());
}

// ============================================================================
// 8. GRAVITY GRADIENT
// ============================================================================

#[test]
fn gravity_gradient_elem() {
    let mu = 3.986e14;
    let r = 7000e3;
    let r3 = r * r * r;

    // Element dim of gravity gradient: Acceleration/Length = 1/s²
    // (= InvTime * InvTime... but we use AngularAcceleration = 1/s² as alias)
    let g = FrameElemMat::<Eci, AngularAcceleration, 3, 3>::from_raw_unchecked(Matrix3::new(
        2.0 * mu / r3, 0.0, 0.0,
        0.0, -mu / r3, 0.0,
        0.0, 0.0, -mu / r3,
    ));

    let dr = FrameVec::<Eci, Length>::new(100.0, 0.0, 0.0);
    // (1/s²) * m = m/s² = Acceleration
    let da: FrameVec<Eci, Acceleration> = g * dr;
    assert!(da.x() > 0.0);
}
