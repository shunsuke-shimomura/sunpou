//! Orbital state transition matrix (STM) using FrameElemMat and block matrices.
//!
//! Demonstrates frame-safe, prefix-aware, dimension-checked state propagation
//! using the element-dimension matrix model.

use nalgebra::Matrix3;
use sunpou::block::{BlockMat2x2, BlockVec2};
use sunpou::frame_elem_mat::FrameElemMat;
use sunpou::prefix::*;
use sunpou::prelude::*;
use sunpou::frame_vec::FrameVec;

struct Eci;

/// State vector in ECI, km scale
type StateEci = BlockVec2<FrameVec<Eci, Length, Kilo>, FrameVec<Eci, Velocity, Kilo>>;

/// STM in ECI frame, element dimensions:
/// | Dimensionless  Time     |
/// | InvTime        Dimensionless |
type StmEci = BlockMat2x2<
    FrameElemMat<Eci, Dimensionless, 3, 3>,
    FrameElemMat<Eci, Time, 3, 3>,
    FrameElemMat<Eci, InvTime, 3, 3>,
    FrameElemMat<Eci, Dimensionless, 3, 3>,
>;

fn main() {
    let x0 = StateEci::new(
        FrameVec::<Eci, Length, Kilo>::new(7000.0, 0.0, 0.0),
        FrameVec::<Eci, Velocity, Kilo>::new(0.0, 7.5, 0.0),
    );

    println!("Initial state:");
    println!("  r₀ = [{}, {}, {}] km", x0.upper.x(), x0.upper.y(), x0.upper.z());
    println!("  v₀ = [{}, {}, {}] km/s", x0.lower.x(), x0.lower.y(), x0.lower.z());

    // STM for dt = 60 seconds
    let dt = 60.0;
    let stm = StmEci::new(
        FrameElemMat::from_raw_unchecked(Matrix3::identity()),
        FrameElemMat::from_raw_unchecked(Matrix3::identity() * dt),
        FrameElemMat::from_raw_unchecked(Matrix3::zeros()),
        FrameElemMat::from_raw_unchecked(Matrix3::identity()),
    );

    // x₁ = Φ * x₀ — dimension, frame, and prefix all checked
    let x1: StateEci = stm * x0;

    println!("\nAfter {} s:", dt);
    println!("  r₁ = [{:.3}, {:.3}, {:.3}] km", x1.upper.x(), x1.upper.y(), x1.upper.z());
    println!("  v₁ = [{:.3}, {:.3}, {:.3}] km/s", x1.lower.x(), x1.lower.y(), x1.lower.z());

    let expected_y = 7.5 * dt;
    println!("\nExpected r₁_y = {} km", expected_y);
    println!("Actual   r₁_y = {} km", x1.upper.y());
    assert!((x1.upper.y() - expected_y).abs() < 1e-10);

    println!("\nSTM propagation: frame-safe, prefix-aware, dimension-checked!");
}
