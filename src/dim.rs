//! Type-level SI dimension representation.
//!
//! Each dimension is encoded as `Dim<L, M, T, I, Th, N, J>` where each
//! parameter is a `typenum` integer representing the exponent of the
//! corresponding SI base quantity.

use core::marker::PhantomData;
use core::ops::{Add, Sub};
use typenum::Z0;

/// SI dimension encoded at the type level.
///
/// - `L`: Length (m)
/// - `M`: Mass (kg)
/// - `T`: Time (s)
/// - `I`: Electric current (A)
/// - `Th`: Thermodynamic temperature (K)
/// - `N`: Amount of substance (mol)
/// - `J`: Luminous intensity (cd)
pub struct Dim<L = Z0, M = Z0, T = Z0, I = Z0, Th = Z0, N = Z0, J = Z0> {
    _marker: PhantomData<(L, M, T, I, Th, N, J)>,
}

// ---------------------------------------------------------------------------
// Dimension arithmetic
// ---------------------------------------------------------------------------

/// Result of multiplying two dimensions (exponents are added).
pub type DimMul<D1, D2> = <D1 as DimMultiply<D2>>::Output;

/// Result of dividing two dimensions (exponents are subtracted).
pub type DimDiv<D1, D2> = <D1 as DimDivide<D2>>::Output;

/// Trait for dimension multiplication.
pub trait DimMultiply<Rhs> {
    type Output;
}

/// Trait for dimension division.
pub trait DimDivide<Rhs> {
    type Output;
}

impl<L1, M1, T1, I1, Th1, N1, J1, L2, M2, T2, I2, Th2, N2, J2>
    DimMultiply<Dim<L2, M2, T2, I2, Th2, N2, J2>> for Dim<L1, M1, T1, I1, Th1, N1, J1>
where
    L1: Add<L2>,
    M1: Add<M2>,
    T1: Add<T2>,
    I1: Add<I2>,
    Th1: Add<Th2>,
    N1: Add<N2>,
    J1: Add<J2>,
{
    type Output = Dim<
        <L1 as Add<L2>>::Output,
        <M1 as Add<M2>>::Output,
        <T1 as Add<T2>>::Output,
        <I1 as Add<I2>>::Output,
        <Th1 as Add<Th2>>::Output,
        <N1 as Add<N2>>::Output,
        <J1 as Add<J2>>::Output,
    >;
}

impl<L1, M1, T1, I1, Th1, N1, J1, L2, M2, T2, I2, Th2, N2, J2>
    DimDivide<Dim<L2, M2, T2, I2, Th2, N2, J2>> for Dim<L1, M1, T1, I1, Th1, N1, J1>
where
    L1: Sub<L2>,
    M1: Sub<M2>,
    T1: Sub<T2>,
    I1: Sub<I2>,
    Th1: Sub<Th2>,
    N1: Sub<N2>,
    J1: Sub<J2>,
{
    type Output = Dim<
        <L1 as Sub<L2>>::Output,
        <M1 as Sub<M2>>::Output,
        <T1 as Sub<T2>>::Output,
        <I1 as Sub<I2>>::Output,
        <Th1 as Sub<Th2>>::Output,
        <N1 as Sub<N2>>::Output,
        <J1 as Sub<J2>>::Output,
    >;
}

/// Marker trait: `D` is the dimensionless type (`Dim<Z0,Z0,Z0,Z0,Z0,Z0,Z0>`).
pub trait IsDimensionless {}
impl IsDimensionless for Dim<Z0, Z0, Z0, Z0, Z0, Z0, Z0> {}

/// Marker trait: two dimensions are the same.
/// Automatically satisfied when `D1` and `D2` are the same concrete type.
pub trait SameDim<D> {}
impl<D> SameDim<D> for D {}
