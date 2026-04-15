/// FrameUnitMat in ECI frame must not multiply FrameVec in ECEF frame.
use nalgebra::Matrix3;
use sunpou::aliases::*;
use sunpou::frame_unit_mat::FrameUnitMat;
use sunpou::frame_vec::FrameVec;

struct Eci;
struct Ecef;

fn main() {
    // Matrix defined in ECI frame
    let stm = FrameUnitMat::<Eci, Velocity, Length, 3, 3>::from_raw(
        Matrix3::identity(),
    );
    // Vector in ECEF frame
    let v = FrameVec::<Ecef, Length>::new(1.0, 0.0, 0.0);
    // This must NOT compile — frame mismatch!
    let _ = stm * v;
}
