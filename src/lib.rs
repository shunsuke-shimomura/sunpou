//! # sunpou — Unit-aware linear algebra
//!
//! Compile-time SI dimension checking for scalars, vectors, matrices, and block
//! matrices. `no_std` compatible, zero-cost abstraction over nalgebra.

#![no_std]

pub mod dim;
pub mod prefix;
pub mod scalar;
pub mod unit_vec;
pub mod frame_vec;
pub mod rotation;
pub mod elem_mat;
pub mod frame_elem_mat;
pub mod block;
pub mod aliases;
pub mod dim_name;
pub mod units;

// Legacy modules — kept for backward compatibility, prefer ElemMat/FrameElemMat
#[doc(hidden)]
pub mod unit_mat;
#[doc(hidden)]
pub mod frame_unit_mat;

pub mod prelude {
    pub use crate::aliases::*;
    pub use crate::block::{BlockMat2x2, BlockVec2};
    pub use crate::dim::Dim;
    pub use crate::elem_mat::ElemMat;
    pub use crate::frame_elem_mat::FrameElemMat;
    pub use crate::frame_vec::FrameVec;
    pub use crate::rotation::Rotation;
    pub use crate::scalar::Scalar;
    pub use crate::unit_vec::UnitVec;
    pub use crate::units::UnitLiteral;
}
