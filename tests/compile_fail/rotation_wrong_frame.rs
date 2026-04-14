/// Transforming a vector with wrong source frame must not compile.
use uolgebra::aliases::*;
use uolgebra::frame_vec::FrameVec;
use uolgebra::rotation::Rotation;

struct Eci;
struct Ecef;
struct Body;

fn main() {
    let rot = Rotation::<Eci, Ecef>::from_angle_z(0.5);
    // Vector is in Body frame, but rotation expects Eci
    let v = FrameVec::<Body, Length>::new(1.0, 0.0, 0.0);
    let _ = rot.transform(&v);
}
