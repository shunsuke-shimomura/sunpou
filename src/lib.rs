//! # uolgebra — Unit-aware linear algebra
//!
//! Compile-time SI dimension checking for scalars, vectors, matrices, and block
//! matrices. `no_std` compatible, zero-cost abstraction over nalgebra.

#![no_std]

pub mod dim;
pub mod scalar;
pub mod unit_vec;
pub mod frame_vec;
pub mod rotation;
pub mod unit_mat;
pub mod frame_unit_mat;
pub mod block;
pub mod aliases;

pub mod prelude {
    pub use crate::aliases::*;
    pub use crate::block::{BlockMat2x2, BlockVec2};
    pub use crate::frame_unit_mat::FrameUnitMat;
    pub use crate::dim::Dim;
    pub use crate::frame_vec::FrameVec;
    pub use crate::rotation::Rotation;
    pub use crate::scalar::Scalar;
    pub use crate::unit_mat::UnitMat;
    pub use crate::unit_vec::UnitVec;
}
