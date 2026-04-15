/// Adding scalars with different dimensions must not compile.
use sunpou::aliases::*;
use sunpou::scalar::Scalar;

fn main() {
    let length = Scalar::<Length>::from_raw_unchecked(1.0);
    let mass = Scalar::<Mass>::from_raw_unchecked(2.0);
    let _ = length + mass;
}
