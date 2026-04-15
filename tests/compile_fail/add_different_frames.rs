/// Adding frame vectors in different frames must not compile.
use sunpou::aliases::*;
use sunpou::frame_vec::FrameVec;

struct Eci;
struct Ecef;

fn main() {
    let a = FrameVec::<Eci, Length>::new(1.0, 0.0, 0.0);
    let b = FrameVec::<Ecef, Length>::new(0.0, 1.0, 0.0);
    let _ = a + b;
}
