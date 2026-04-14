//! Unit-aware N-dimensional vector.

use core::marker::PhantomData;
use core::ops::{Add, AddAssign, Index, Mul, MulAssign, Neg, Sub, SubAssign};
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

impl<D, const N: usize> core::fmt::Display for UnitVec<D, N> {
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

impl<D, const N: usize> Default for UnitVec<D, N> {
    #[inline(always)]
    fn default() -> Self {
        Self::zeros()
    }
}

impl<D, const N: usize> UnitVec<D, N> {
    /// Create from a slice. Panics if `slice.len() != N`.
    #[inline(always)]
    pub fn from_slice(slice: &[f64]) -> Self {
        Self::from_raw_unchecked(SVector::from_column_slice(slice))
    }

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
        Self::from_raw_unchecked(SVector::zeros())
    }

    /// Euclidean norm. Returns a scalar with the same dimension.
    #[inline(always)]
    pub fn norm(&self) -> Scalar<D> {
        Scalar::from_raw_unchecked(self.value.norm())
    }

    /// Normalize to unit length. Returns a dimensionless unit vector.
    /// Returns `None` if the vector is zero.
    #[inline(always)]
    pub fn try_normalize(&self, min_norm: f64) -> Option<UnitVec<crate::aliases::Dimensionless, N>> {
        self.value
            .try_normalize(min_norm)
            .map(UnitVec::from_raw_unchecked)
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

// ---- Convenience constructors for 3D ----

impl<D> UnitVec<D, 3> {
    /// Create a 3D vector from components.
    #[inline(always)]
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self::from_raw_unchecked(SVector::from([x, y, z]))
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

// ---- Compound assignment ----

impl<D, const N: usize> AddAssign for UnitVec<D, N> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        self.value += rhs.value;
    }
}

impl<D, const N: usize> SubAssign for UnitVec<D, N> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        self.value -= rhs.value;
    }
}

impl<D, const N: usize> MulAssign<f64> for UnitVec<D, N> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: f64) {
        self.value *= rhs;
    }
}

// ---- Reference ops ----

impl<D, const N: usize> Add for &UnitVec<D, N> {
    type Output = UnitVec<D, N>;
    #[inline(always)]
    fn add(self, rhs: Self) -> UnitVec<D, N> {
        UnitVec::from_raw_unchecked(self.value + rhs.value)
    }
}

impl<D, const N: usize> Sub for &UnitVec<D, N> {
    type Output = UnitVec<D, N>;
    #[inline(always)]
    fn sub(self, rhs: Self) -> UnitVec<D, N> {
        UnitVec::from_raw_unchecked(self.value - rhs.value)
    }
}

// ---- Indexing ----

impl<D, const N: usize> Index<usize> for UnitVec<D, N> {
    type Output = f64;
    #[inline(always)]
    fn index(&self, index: usize) -> &f64 {
        &self.value[index]
    }
}
