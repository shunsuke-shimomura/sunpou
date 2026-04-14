//! Extended Kalman Filter (EKF) with type-safe covariance propagation.
//!
//! Demonstrates the full EKF prediction step with compile-time unit checking:
//! - State propagation: x₁ = Φ * x₀
//! - Covariance propagation: P₁ = Φ * P₀ * Φᵀ + Q
//!
//! All matrix dimensions are verified at compile time.

use nalgebra::{Matrix3, SVector};
use uolgebra::block::{BlockMat2x2, BlockVec2};
use uolgebra::prelude::*;
use uolgebra::unit_mat::UnitMat;
use uolgebra::unit_vec::UnitVec;

// Type aliases for the orbital EKF

/// State vector: [position, velocity]
type State = BlockVec2<UnitVec<Length, 3>, UnitVec<Velocity, 3>>;

/// State transition matrix
type Stm = BlockMat2x2<
    UnitMat<Length, Length, 3, 3>,
    UnitMat<Length, Velocity, 3, 3>,
    UnitMat<Velocity, Length, 3, 3>,
    UnitMat<Velocity, Velocity, 3, 3>,
>;

/// Covariance matrix: P[i][j] has dimension dim(state[i]) * dim(state[j])
///
/// ```text
/// P = | P_rr [m²]       P_rv [m·(m/s)]     |
///     | P_vr [(m/s)·m]   P_vv [(m/s)²]      |
/// ```
type Covariance = BlockMat2x2<
    UnitMat<Length, Length, 3, 3>,
    UnitMat<Length, Velocity, 3, 3>,
    UnitMat<Velocity, Length, 3, 3>,
    UnitMat<Velocity, Velocity, 3, 3>,
>;

/// STM transpose: Φᵀ has transposed block dimensions
///
/// ```text
/// Φᵀ = | (∂r/∂r₀)ᵀ  (∂v/∂r₀)ᵀ |
///      | (∂r/∂v₀)ᵀ  (∂v/∂v₀)ᵀ |
/// ```
type StmTranspose = BlockMat2x2<
    UnitMat<Length, Length, 3, 3>,
    UnitMat<Length, Velocity, 3, 3>,
    UnitMat<Velocity, Length, 3, 3>,
    UnitMat<Velocity, Velocity, 3, 3>,
>;

fn stm_transpose(phi: &Stm) -> StmTranspose {
    // Φᵀ: swap off-diagonal blocks AND transpose each block
    StmTranspose::new(
        phi.a.transpose(), // (∂r/∂r₀)ᵀ : Length/Length
        phi.c.transpose(), // (∂v/∂r₀)ᵀ : Length/Velocity  (note: c and b swap!)
        phi.b.transpose(), // (∂r/∂v₀)ᵀ : Velocity/Length
        phi.d.transpose(), // (∂v/∂v₀)ᵀ : Velocity/Velocity
    )
}

fn main() {
    println!("=== EKF Prediction Step (Orbital Mechanics) ===\n");

    // Initial state
    let x0 = State::new(
        UnitVec::<Length, 3>::from_raw_unchecked(SVector::from([7000e3, 0.0, 0.0])),
        UnitVec::<Velocity, 3>::from_raw_unchecked(SVector::from([0.0, 7.5e3, 0.0])),
    );

    // Initial covariance
    let p0 = Covariance::new(
        UnitMat::from_raw_unchecked(Matrix3::identity() * 100.0), // σ_r² = 100 m²
        UnitMat::from_raw_unchecked(Matrix3::zeros()),
        UnitMat::from_raw_unchecked(Matrix3::zeros()),
        UnitMat::from_raw_unchecked(Matrix3::identity() * 0.01), // σ_v² = 0.01 (m/s)²
    );

    // Process noise
    let q = Covariance::new(
        UnitMat::from_raw_unchecked(Matrix3::identity() * 1.0), // 1 m²
        UnitMat::from_raw_unchecked(Matrix3::zeros()),
        UnitMat::from_raw_unchecked(Matrix3::zeros()),
        UnitMat::from_raw_unchecked(Matrix3::identity() * 0.001), // 0.001 (m/s)²
    );

    // State transition matrix (dt = 60 s)
    let dt = 60.0;
    let phi = Stm::new(
        UnitMat::from_raw_unchecked(Matrix3::identity()),
        UnitMat::from_raw_unchecked(Matrix3::identity() * dt),
        UnitMat::from_raw_unchecked(Matrix3::zeros()),
        UnitMat::from_raw_unchecked(Matrix3::identity()),
    );

    // === Prediction Step ===

    // State propagation: x₁ = Φ * x₀
    let x1: State = phi * x0;

    // Covariance propagation: P₁ = Φ * P₀ * Φᵀ + Q
    // Step 1: Φ * P₀
    let phi_p0: Covariance = phi * p0;
    // Step 2: (Φ * P₀) * Φᵀ
    let phi_t = stm_transpose(&phi);
    let phi_p0_phit: Covariance = phi_p0 * phi_t;
    // Step 3: + Q
    let p1: Covariance = phi_p0_phit + q;

    // === Output ===

    println!("State (after {} s):", dt);
    println!("  r = {:?} m", x1.upper.as_raw().as_slice());
    println!("  v = {:?} m/s", x1.lower.as_raw().as_slice());

    println!("\nCovariance P₁:");
    println!("  P_rr diagonal = [{:.2}, {:.2}, {:.2}] m²",
        p1.a.as_raw()[(0, 0)], p1.a.as_raw()[(1, 1)], p1.a.as_raw()[(2, 2)]);
    println!("  P_vv diagonal = [{:.4}, {:.4}, {:.4}] (m/s)²",
        p1.d.as_raw()[(0, 0)], p1.d.as_raw()[(1, 1)], p1.d.as_raw()[(2, 2)]);

    // Cross-covariance should now be non-zero due to STM coupling
    println!("  P_rv[0,0] = {:.2} m·(m/s)", p1.b.as_raw()[(0, 0)]);

    println!("\n=== All EKF operations type-checked at compile time! ===");
}
