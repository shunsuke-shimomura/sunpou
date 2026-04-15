/// Adding scalars with different prefixes must not compile.
/// This prevents accidental precision loss from scale mismatch.
use sunpou::aliases::*;
use sunpou::prefix::*;
use sunpou::scalar::Scalar;

fn main() {
    let km = Scalar::<Length, Kilo>::from_raw(3.0);
    let m = Scalar::<Length, Base>::from_raw(4000.0);
    // Same dimension but different prefix → must NOT compile
    let _ = km + m;
}
