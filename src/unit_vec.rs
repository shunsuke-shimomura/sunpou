//! Unit-aware N-dimensional vector.

use core::marker::PhantomData;
use core::ops::{Add, Mul, Neg, Sub};
use nalgebra::SVector;

use crate::dim::{Dim, DimMultiply};
use crate::scalar::Scalar;

/// An N-dimensional vector with all components sharing SI dimension `D`.
#[repr(transparent)]
pub struct UnitVec<D, const N: usize> {
    value: SVector<f64, N>,
    _dim: PhantomData<D>,
}

impl<D, const N: usize> Clone for UnitVec<D, N> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<D, const N: usize> Copy for UnitVec<D, N>
where
    SVector<f64, N>: Copy,
{
}

impl<D, const N: usize> PartialEq for UnitVec<D, N> {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<D, const N: usize> core::fmt::Debug for UnitVec<D, N> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "UnitVec({:?})", self.value.as_slice())
    }
}

impl<D, const N: usize> UnitVec<D, N> {
    /// Create from a raw nalgebra vector. Caller ensures SI base units.
    #[inline(always)]
    pub fn from_raw_unchecked(value: SVector<f64, N>) -> Self {
        Self {
            value,
            _dim: PhantomData,
        }
    }

    /// Extract the raw nalgebra vector.
    #[inline(always)]
    pub fn into_raw(self) -> SVector<f64, N> {
        self.value
    }

    /// Borrow the raw nalgebra vector.
    #[inline(always)]
    pub fn as_raw(&self) -> &SVector<f64, N> {
        &self.value
    }

    /// Euclidean norm. Returns a scalar with the same dimension.
    #[inline(always)]
    pub fn norm(&self) -> Scalar<D> {
        Scalar::from_raw_unchecked(self.value.norm())
    }

    /// Squared norm. Returns a scalar with dimension D².
    #[inline(always)]
    pub fn norm_squared<D2>(&self) -> Scalar<D2>
    where
        D: DimMultiply<D, Output = D2>,
    {
        Scalar::from_raw_unchecked(self.value.norm_squared())
    }
}

// ---- Heterogeneous dot product ----

impl<D1, const N: usize> UnitVec<D1, N> {
    /// Dot product with possibly different dimension.
    /// `UnitVec<D1, N> · UnitVec<D2, N> → Scalar<DimMul<D1, D2>>`
    #[inline(always)]
    pub fn dot<D2>(&self, rhs: &UnitVec<D2, N>) -> Scalar<<D1 as DimMultiply<D2>>::Output>
    where
        D1: DimMultiply<D2>,
    {
        Scalar::from_raw_unchecked(self.value.dot(&rhs.value))
    }
}

// ---- Heterogeneous cross product (3D only) ----

impl<D1> UnitVec<D1, 3> {
    /// Cross product with possibly different dimension.
    /// `UnitVec<D1, 3> × UnitVec<D2, 3> → UnitVec<DimMul<D1, D2>, 3>`
    #[inline(always)]
    pub fn cross<D2>(&self, rhs: &UnitVec<D2, 3>) -> UnitVec<<D1 as DimMultiply<D2>>::Output, 3>
    where
        D1: DimMultiply<D2>,
    {
        UnitVec::from_raw_unchecked(self.value.cross(&rhs.value))
    }
}

// ---- Same-dimension add/sub ----

impl<D, const N: usize> Add for UnitVec<D, N> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self::from_raw_unchecked(self.value + rhs.value)
    }
}

impl<D, const N: usize> Sub for UnitVec<D, N> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self::from_raw_unchecked(self.value - rhs.value)
    }
}

impl<D, const N: usize> Neg for UnitVec<D, N> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self::from_raw_unchecked(-self.value)
    }
}

// ---- Scalar multiplication (cross-dimension) ----

impl<L1, M1, T1, I1, Th1, N1, J1, L2, M2, T2, I2, Th2, N2, J2, const K: usize>
    Mul<UnitVec<Dim<L2, M2, T2, I2, Th2, N2, J2>, K>>
    for Scalar<Dim<L1, M1, T1, I1, Th1, N1, J1>>
where
    Dim<L1, M1, T1, I1, Th1, N1, J1>: DimMultiply<Dim<L2, M2, T2, I2, Th2, N2, J2>>,
{
    type Output = UnitVec<
        <Dim<L1, M1, T1, I1, Th1, N1, J1> as DimMultiply<
            Dim<L2, M2, T2, I2, Th2, N2, J2>,
        >>::Output,
        K,
    >;
    #[inline(always)]
    fn mul(self, rhs: UnitVec<Dim<L2, M2, T2, I2, Th2, N2, J2>, K>) -> Self::Output {
        UnitVec::from_raw_unchecked(rhs.value * self.into_raw())
    }
}

// ---- f64 scaling ----

impl<D, const N: usize> Mul<f64> for UnitVec<D, N> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: f64) -> Self {
        Self::from_raw_unchecked(self.value * rhs)
    }
}

impl<D, const N: usize> Mul<UnitVec<D, N>> for f64 {
    type Output = UnitVec<D, N>;
    #[inline(always)]
    fn mul(self, rhs: UnitVec<D, N>) -> UnitVec<D, N> {
        UnitVec::from_raw_unchecked(rhs.value * self)
    }
}
