//! Unit-aware N-dimensional vector with SI prefix support.
//!
//! `UnitVec<D, N, P>` stores N components in units of `10^P × [SI base unit for D]`.
//! Default `P = Z0` for backward compatibility.

use core::marker::PhantomData;
use core::ops::{Add, AddAssign, Index, Mul, MulAssign, Neg, Sub, SubAssign};
use nalgebra::SVector;
use typenum::{Integer, Z0};

use crate::dim::{Dim, DimMultiply};
use crate::scalar::Scalar;

/// An N-dimensional vector with all components sharing SI dimension `D` and prefix `P`.
#[repr(transparent)]
pub struct UnitVec<D, const N: usize, P = Z0> {
    value: SVector<f64, N>,
    _marker: PhantomData<(D, P)>,
}

impl<D, const N: usize, P> Clone for UnitVec<D, N, P> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<D, const N: usize, P> Copy for UnitVec<D, N, P>
where
    SVector<f64, N>: Copy,
{
}

impl<D, const N: usize, P> PartialEq for UnitVec<D, N, P> {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<D, const N: usize, P> core::fmt::Debug for UnitVec<D, N, P> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "UnitVec({:?})", self.value.as_slice())
    }
}

impl<D, const N: usize, P> core::fmt::Display for UnitVec<D, N, P> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "[")?;
        for (i, v) in self.value.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            core::fmt::Display::fmt(v, f)?;
        }
        write!(f, "]")
    }
}

impl<D, const N: usize, P> Default for UnitVec<D, N, P> {
    #[inline(always)]
    fn default() -> Self {
        Self::zeros()
    }
}

impl<D, const N: usize, P> UnitVec<D, N, P> {
    /// Create from a slice. Panics if `slice.len() != N`.
    #[inline(always)]
    pub fn from_slice(slice: &[f64]) -> Self {
        Self::from_raw(SVector::from_column_slice(slice))
    }

    /// Create from a raw nalgebra vector.
    #[inline(always)]
    pub fn from_raw(value: SVector<f64, N>) -> Self {
        Self {
            value,
            _marker: PhantomData,
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

    /// Iterate over components as `f64` references.
    #[inline(always)]
    pub fn iter(&self) -> impl Iterator<Item = &f64> {
        self.value.iter()
    }

    /// Number of components.
    #[inline(always)]
    pub fn len(&self) -> usize {
        N
    }

    /// Always false (N is const > 0 in practice, but needed for clippy).
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        N == 0
    }

    /// Create a zero vector.
    #[inline(always)]
    pub fn zeros() -> Self {
        Self::from_raw(SVector::zeros())
    }

    /// Euclidean norm. Returns a scalar with same dimension and prefix.
    #[inline(always)]
    pub fn norm(&self) -> Scalar<D, P> {
        Scalar::from_raw(self.value.norm())
    }

    /// Normalize to unit length. Returns a dimensionless unit vector (base prefix).
    #[inline(always)]
    pub fn try_normalize(
        &self,
        min_norm: f64,
    ) -> Option<UnitVec<crate::aliases::Dimensionless, N>> {
        self.value
            .try_normalize(min_norm)
            .map(UnitVec::from_raw)
    }

    /// Rescale to a different prefix. Multiplies all components by `10^(P - P2)`.
    #[inline(always)]
    pub fn rescale<P2>(self) -> UnitVec<D, N, P2>
    where
        P: Sub<P2>,
        <P as Sub<P2>>::Output: Integer,
    {
        let factor = crate::prefix::pow10_i32(
            <<P as Sub<P2>>::Output as Integer>::to_i64() as i32,
        );
        UnitVec::from_raw(self.value * factor)
    }
}

// ---- Convenience constructors for 3D ----

impl<D, P> UnitVec<D, 3, P> {
    /// Create a 3D vector from components.
    #[inline(always)]
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self::from_raw(SVector::from([x, y, z]))
    }

    /// X component.
    #[inline(always)]
    pub fn x(&self) -> f64 {
        self.value[0]
    }

    /// Y component.
    #[inline(always)]
    pub fn y(&self) -> f64 {
        self.value[1]
    }

    /// Z component.
    #[inline(always)]
    pub fn z(&self) -> f64 {
        self.value[2]
    }
}

// ---- Heterogeneous dot product (cross-dim, cross-prefix) ----

impl<D1, const N: usize, P1> UnitVec<D1, N, P1> {
    /// Dot product: `UnitVec<D1, N, P1> · UnitVec<D2, N, P2> → Scalar<D1×D2, P1+P2>`
    #[inline(always)]
    pub fn dot<D2, P2>(
        &self,
        rhs: &UnitVec<D2, N, P2>,
    ) -> Scalar<<D1 as DimMultiply<D2>>::Output, <P1 as Add<P2>>::Output>
    where
        D1: DimMultiply<D2>,
        P1: Add<P2>,
    {
        Scalar::from_raw(self.value.dot(&rhs.value))
    }
}

// ---- Heterogeneous cross product (3D, cross-dim, cross-prefix) ----

impl<D1, P1> UnitVec<D1, 3, P1> {
    /// Cross product: `UnitVec<D1, 3, P1> × UnitVec<D2, 3, P2> → UnitVec<D1×D2, 3, P1+P2>`
    #[inline(always)]
    pub fn cross<D2, P2>(
        &self,
        rhs: &UnitVec<D2, 3, P2>,
    ) -> UnitVec<<D1 as DimMultiply<D2>>::Output, 3, <P1 as Add<P2>>::Output>
    where
        D1: DimMultiply<D2>,
        P1: Add<P2>,
    {
        UnitVec::from_raw(self.value.cross(&rhs.value))
    }
}

// ---- Same-dimension, same-prefix add/sub ----

impl<D, const N: usize, P> Add for UnitVec<D, N, P> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self::from_raw(self.value + rhs.value)
    }
}

impl<D, const N: usize, P> Sub for UnitVec<D, N, P> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self::from_raw(self.value - rhs.value)
    }
}

impl<D, const N: usize, P> Neg for UnitVec<D, N, P> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self::from_raw(-self.value)
    }
}

// ---- Scalar multiplication (cross-dim, cross-prefix) ----

impl<L1, M1, T1, I1, Th1, N1, J1, P1, L2, M2, T2, I2, Th2, N2, J2, P2, const K: usize>
    Mul<UnitVec<Dim<L2, M2, T2, I2, Th2, N2, J2>, K, P2>>
    for Scalar<Dim<L1, M1, T1, I1, Th1, N1, J1>, P1>
where
    Dim<L1, M1, T1, I1, Th1, N1, J1>: DimMultiply<Dim<L2, M2, T2, I2, Th2, N2, J2>>,
    P1: Add<P2>,
{
    type Output = UnitVec<
        <Dim<L1, M1, T1, I1, Th1, N1, J1> as DimMultiply<
            Dim<L2, M2, T2, I2, Th2, N2, J2>,
        >>::Output,
        K,
        <P1 as Add<P2>>::Output,
    >;
    #[inline(always)]
    fn mul(
        self,
        rhs: UnitVec<Dim<L2, M2, T2, I2, Th2, N2, J2>, K, P2>,
    ) -> Self::Output {
        UnitVec::from_raw(rhs.value * self.into_raw())
    }
}

// ---- f64 scaling (preserves prefix) ----

impl<D, const N: usize, P> Mul<f64> for UnitVec<D, N, P> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: f64) -> Self {
        Self::from_raw(self.value * rhs)
    }
}

impl<D, const N: usize, P> Mul<UnitVec<D, N, P>> for f64 {
    type Output = UnitVec<D, N, P>;
    #[inline(always)]
    fn mul(self, rhs: UnitVec<D, N, P>) -> UnitVec<D, N, P> {
        UnitVec::from_raw(rhs.value * self)
    }
}

// ---- Compound assignment (same prefix) ----

impl<D, const N: usize, P> AddAssign for UnitVec<D, N, P> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        self.value += rhs.value;
    }
}

impl<D, const N: usize, P> SubAssign for UnitVec<D, N, P> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        self.value -= rhs.value;
    }
}

impl<D, const N: usize, P> MulAssign<f64> for UnitVec<D, N, P> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: f64) {
        self.value *= rhs;
    }
}

// ---- Reference ops ----

impl<D, const N: usize, P> Add for &UnitVec<D, N, P> {
    type Output = UnitVec<D, N, P>;
    #[inline(always)]
    fn add(self, rhs: Self) -> UnitVec<D, N, P> {
        UnitVec::from_raw(self.value + rhs.value)
    }
}

impl<D, const N: usize, P> Sub for &UnitVec<D, N, P> {
    type Output = UnitVec<D, N, P>;
    #[inline(always)]
    fn sub(self, rhs: Self) -> UnitVec<D, N, P> {
        UnitVec::from_raw(self.value - rhs.value)
    }
}

// ---- Indexing ----

impl<D, const N: usize, P> Index<usize> for UnitVec<D, N, P> {
    type Output = f64;
    #[inline(always)]
    fn index(&self, index: usize) -> &f64 {
        &self.value[index]
    }
}
