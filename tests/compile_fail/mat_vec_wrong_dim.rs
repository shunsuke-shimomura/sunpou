/// Matrix-vector multiplication with wrong input dimension must not compile.
use nalgebra::{Matrix3, SVector};
use sunpou::aliases::*;
use sunpou::unit_mat::UnitMat;
use sunpou::unit_vec::UnitVec;

fn main() {
    // Matrix expects input dimension = Length, but we give Velocity
    let m = UnitMat::<Velocity, Length, 3, 3>::from_raw(Matrix3::identity());
    let v = UnitVec::<Velocity, 3>::from_raw(SVector::from([1.0, 2.0, 3.0]));
    let _ = m * v;
}
