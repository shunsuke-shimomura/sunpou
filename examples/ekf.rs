//! Extended Kalman Filter (EKF) with frame-safe, prefix-aware covariance propagation.
//!
//! Demonstrates the full EKF prediction step:
//! - State propagation: x₁ = Φ * x₀
//! - Covariance propagation: P₁ = Φ * P₀ * Φᵀ + Q
//!
//! All dimensions, frames, and prefixes are verified at compile time.
//! Uses ElemMat (element-dimension model) — no rescale_dims needed.

use nalgebra::Matrix3;
use sunpou::block::BlockMat2x2;
use sunpou::block::BlockVec2;
use sunpou::frame_elem_mat::FrameElemMat;
use sunpou::prefix::*;
use sunpou::prelude::*;
use sunpou::frame_vec::FrameVec;

struct Eci;

// ---- Type aliases ----

/// State vector in ECI, km scale
type State = BlockVec2<FrameVec<Eci, Length, Kilo>, FrameVec<Eci, Velocity, Kilo>>;

/// STM block element dimensions:
/// | Dimensionless  Time     |
/// | InvTime        Dimensionless |
type Stm = BlockMat2x2<
    FrameElemMat<Eci, Dimensionless, 3, 3>,
    FrameElemMat<Eci, Time, 3, 3>,
    FrameElemMat<Eci, InvTime, 3, 3>,
    FrameElemMat<Eci, Dimensionless, 3, 3>,
>;

/// Covariance block element dimensions:
/// P_rr: Dimensionless (km² / km² when row=col=Length)
///   ... but actually the covariance P[i][j] = E[δx_i δx_j]
///   For position error (km), P_rr has units km² → element dim = Dimensionless (since row/col are both Length)
///
/// More precisely, with ElemMat model:
///   Φ P Φᵀ requires element dims to compose correctly.
///   P_rr: elem dim = Dimensionless (same as Φ_rr)
///   P_rv: elem dim = Time (same as Φ_rv)
///   P_vr: elem dim = InvTime
///   P_vv: elem dim = Dimensionless
type Covariance = BlockMat2x2<
    FrameElemMat<Eci, Dimensionless, 3, 3>,
    FrameElemMat<Eci, Time, 3, 3>,
    FrameElemMat<Eci, InvTime, 3, 3>,
    FrameElemMat<Eci, Dimensionless, 3, 3>,
>;

fn main() {
    println!("=== EKF Prediction Step (Orbital Mechanics, km scale) ===\n");

    // Initial state in km
    let x0 = State::new(
        FrameVec::new(7000.0, 0.0, 0.0),    // 7000 km
        FrameVec::new(0.0, 7.5, 0.0),       // 7.5 km/s
    );

    // Initial covariance (in km²/km·(km/s)/(km/s)² units via prefixes)
    let p0 = Covariance::new(
        FrameElemMat::from_raw_unchecked(Matrix3::identity() * 0.01),  // σ_r = 0.1 km → 0.01 km²
        FrameElemMat::from_raw_unchecked(Matrix3::zeros()),
        FrameElemMat::from_raw_unchecked(Matrix3::zeros()),
        FrameElemMat::from_raw_unchecked(Matrix3::identity() * 1e-6),  // σ_v = 0.001 km/s → 1e-6
    );

    // Process noise
    let q = Covariance::new(
        FrameElemMat::from_raw_unchecked(Matrix3::identity() * 1e-4),
        FrameElemMat::from_raw_unchecked(Matrix3::zeros()),
        FrameElemMat::from_raw_unchecked(Matrix3::zeros()),
        FrameElemMat::from_raw_unchecked(Matrix3::identity() * 1e-8),
    );

    // STM (dt = 60 s)
    let dt = 60.0;
    let phi = Stm::new(
        FrameElemMat::from_raw_unchecked(Matrix3::identity()),
        FrameElemMat::from_raw_unchecked(Matrix3::identity() * dt),
        FrameElemMat::from_raw_unchecked(Matrix3::zeros()),
        FrameElemMat::from_raw_unchecked(Matrix3::identity()),
    );

    // === Prediction Step ===

    // State propagation: x₁ = Φ * x₀
    let x1: State = phi * x0;

    // Covariance propagation: P₁ = Φ * P₀ * Φᵀ + Q
    let phi_p0: Covariance = phi * p0;
    let phi_t = phi.transpose();
    let phi_p0_phit: Covariance = phi_p0 * phi_t;
    let p1: Covariance = phi_p0_phit + q;

    // === Output ===

    println!("State (after {} s):", dt);
    println!("  r = [{:.3}, {:.3}, {:.3}] km", x1.upper.x(), x1.upper.y(), x1.upper.z());
    println!("  v = [{:.4}, {:.4}, {:.4}] km/s", x1.lower.x(), x1.lower.y(), x1.lower.z());

    println!("\nCovariance P₁ (km units):");
    println!("  P_rr diagonal = [{:.6}, {:.6}, {:.6}] km²",
        p1.a.as_raw()[(0, 0)], p1.a.as_raw()[(1, 1)], p1.a.as_raw()[(2, 2)]);
    println!("  P_vv diagonal = [{:.2e}, {:.2e}, {:.2e}] (km/s)²",
        p1.d.as_raw()[(0, 0)], p1.d.as_raw()[(1, 1)], p1.d.as_raw()[(2, 2)]);
    println!("  P_rv[0,0] = {:.6} km·(km/s)", p1.b.as_raw()[(0, 0)]);

    println!("\n=== All EKF operations: frame-safe, prefix-aware, dimension-checked! ===");
}
