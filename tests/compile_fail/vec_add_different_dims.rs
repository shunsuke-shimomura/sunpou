/// Adding vectors with different dimensions must not compile.
use nalgebra::SVector;
use sunpou::aliases::*;
use sunpou::unit_vec::UnitVec;

fn main() {
    let a = UnitVec::<Length, 3>::from_raw(SVector::from([1.0, 2.0, 3.0]));
    let b = UnitVec::<Velocity, 3>::from_raw(SVector::from([4.0, 5.0, 6.0]));
    let _ = a + b;
}
