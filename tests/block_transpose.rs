//! Test BlockMat2x2::transpose()

use nalgebra::Matrix3;
use uolgebra::aliases::*;
use uolgebra::block::BlockMat2x2;
use uolgebra::unit_mat::UnitMat;

type Stm = BlockMat2x2<
    UnitMat<Length, Length, 3, 3>,
    UnitMat<Length, Velocity, 3, 3>,
    UnitMat<Velocity, Length, 3, 3>,
    UnitMat<Velocity, Velocity, 3, 3>,
>;

type StmT = BlockMat2x2<
    UnitMat<Length, Length, 3, 3>,
    UnitMat<Length, Velocity, 3, 3>,
    UnitMat<Velocity, Length, 3, 3>,
    UnitMat<Velocity, Velocity, 3, 3>,
>;

#[test]
fn block_transpose_basic() {
    let dt = 60.0;
    let stm = Stm::new(
        UnitMat::from_raw_unchecked(Matrix3::identity()),
        UnitMat::from_raw_unchecked(Matrix3::identity() * dt),
        UnitMat::from_raw_unchecked(Matrix3::zeros()),
        UnitMat::from_raw_unchecked(Matrix3::identity()),
    );

    let stm_t: StmT = stm.transpose();

    // Diagonal blocks are transposed (identity stays identity)
    assert_eq!(stm_t.a.into_raw(), Matrix3::identity());
    assert_eq!(stm_t.d.into_raw(), Matrix3::identity());

    // Off-diagonal blocks are swapped AND transposed
    // Original b = dt*I → becomes new c (transposed = dt*I)
    // Original c = 0 → becomes new b (transposed = 0)
    assert_eq!(stm_t.b.into_raw(), Matrix3::zeros());
    assert_eq!(stm_t.c.into_raw(), Matrix3::identity() * dt);
}
