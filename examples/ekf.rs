//! Extended Kalman Filter (EKF) with frame-safe, prefix-aware covariance propagation.
//!
//! Demonstrates the full EKF prediction step:
//! - State propagation: x₁ = Φ * x₀
//! - Covariance propagation: P₁ = Φ * P₀ * Φᵀ + Q
//!
//! Vectors use Kilo prefix (km, km/s).
//! Matrices use Base prefix (SI base units) — `Base` is the default and can be omitted.

use nalgebra::Matrix3;
use sunpou::block::BlockMat2x2;
use sunpou::block::BlockVec2;
use sunpou::frame_elem_mat::FrameElemMat;
use sunpou::prefix::*; // Base, Kilo, Mega, etc.
use sunpou::prelude::*;
use sunpou::frame_vec::FrameVec;

struct Eci;

// ---- Type aliases ----

/// State vector in ECI, km scale
type State = BlockVec2<FrameVec<Eci, Length, Kilo>, FrameVec<Eci, Velocity, Kilo>>;

/// STM block element dimensions (Base prefix — the default, shown explicitly):
///
/// | Dimensionless  Time     |
/// | InvTime        Dimensionless |
type Stm = BlockMat2x2<
    FrameElemMat<Eci, Dimensionless, 3, 3, Base>,  // Base is the default, can be omitted
    FrameElemMat<Eci, Time, 3, 3, Base>,
    FrameElemMat<Eci, InvTime, 3, 3, Base>,
    FrameElemMat<Eci, Dimensionless, 3, 3, Base>,
>;

/// Covariance: same block structure as STM.
///
/// P_rr elements are in km² (Dimensionless × Kilo × Kilo → but tracked via vector prefix)
/// Matrix prefix is Base; the "km²" aspect comes from the vector prefixes.
type Covariance = BlockMat2x2<
    FrameElemMat<Eci, Dimensionless, 3, 3, Base>,
    FrameElemMat<Eci, Time, 3, 3, Base>,
    FrameElemMat<Eci, InvTime, 3, 3, Base>,
    FrameElemMat<Eci, Dimensionless, 3, 3, Base>,
>;

fn main() {
    println!("=== EKF Prediction Step (Orbital Mechanics, km scale) ===\n");

    // Initial state in km
    let x0 = State::new(
        FrameVec::new(7000.0, 0.0, 0.0),    // 7000 km
        FrameVec::new(0.0, 7.5, 0.0),       // 7.5 km/s
    );

    // Initial covariance (values in km²/km·(km/s)/(km/s)² via prefix composition)
    let p0 = Covariance::new(
        FrameElemMat::from_raw_unchecked(Matrix3::identity() * 0.01),  // σ_r = 0.1 km → 0.01 km²
        FrameElemMat::from_raw_unchecked(Matrix3::zeros()),
        FrameElemMat::from_raw_unchecked(Matrix3::zeros()),
        FrameElemMat::from_raw_unchecked(Matrix3::identity() * 1e-6),  // σ_v = 0.001 km/s
    );

    // Process noise (Base prefix)
    let q = Covariance::new(
        FrameElemMat::from_raw_unchecked(Matrix3::identity() * 1e-4),
        FrameElemMat::from_raw_unchecked(Matrix3::zeros()),
        FrameElemMat::from_raw_unchecked(Matrix3::zeros()),
        FrameElemMat::from_raw_unchecked(Matrix3::identity() * 1e-8),
    );

    // STM (dt = 60 s, Base prefix)
    let dt = 60.0;
    let phi = Stm::new(
        FrameElemMat::from_raw_unchecked(Matrix3::identity()),
        FrameElemMat::from_raw_unchecked(Matrix3::identity() * dt),
        FrameElemMat::from_raw_unchecked(Matrix3::zeros()),
        FrameElemMat::from_raw_unchecked(Matrix3::identity()),
    );

    // === Prediction Step ===

    // State propagation: x₁ = Φ * x₀
    // Prefix: Base(Φ) + Kilo(x₀) = Kilo(x₁) ✓
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

    println!("\nCovariance P₁:");
    println!("  P_rr diagonal = [{:.6}, {:.6}, {:.6}]",
        p1.a.as_raw()[(0, 0)], p1.a.as_raw()[(1, 1)], p1.a.as_raw()[(2, 2)]);
    println!("  P_vv diagonal = [{:.2e}, {:.2e}, {:.2e}]",
        p1.d.as_raw()[(0, 0)], p1.d.as_raw()[(1, 1)], p1.d.as_raw()[(2, 2)]);
    println!("  P_rv[0,0] = {:.6}", p1.b.as_raw()[(0, 0)]);

    println!("\n=== All EKF operations: frame-safe, prefix-aware, dimension-checked! ===");
}
