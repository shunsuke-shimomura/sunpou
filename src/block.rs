//! Generic block matrix and block vector types.
//!
//! These types compose smaller `UnitMat` / `UnitVec` blocks into larger
//! structures while preserving type-level dimension checking through
//! trait bounds.

use core::ops::{Add, Mul};

// ---------------------------------------------------------------------------
// BlockVec2 — two-block column vector
// ---------------------------------------------------------------------------

/// A column vector composed of two sub-vectors.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlockVec2<U, L> {
    pub upper: U,
    pub lower: L,
}

impl<U, L> BlockVec2<U, L> {
    /// Create a new block vector from upper and lower parts.
    #[inline(always)]
    pub fn new(upper: U, lower: L) -> Self {
        Self { upper, lower }
    }
}

impl<U: Add<Output = U>, L: Add<Output = L>> Add for BlockVec2<U, L> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self {
            upper: self.upper + rhs.upper,
            lower: self.lower + rhs.lower,
        }
    }
}

// ---------------------------------------------------------------------------
// BlockVec3 — three-block column vector
// ---------------------------------------------------------------------------

/// A column vector composed of three sub-vectors.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlockVec3<A, B, C> {
    pub a: A,
    pub b: B,
    pub c: C,
}

impl<A, B, C> BlockVec3<A, B, C> {
    #[inline(always)]
    pub fn new(a: A, b: B, c: C) -> Self {
        Self { a, b, c }
    }
}

impl<A: Add<Output = A>, B: Add<Output = B>, C: Add<Output = C>> Add for BlockVec3<A, B, C> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self {
            a: self.a + rhs.a,
            b: self.b + rhs.b,
            c: self.c + rhs.c,
        }
    }
}

// ---------------------------------------------------------------------------
// BlockMat2x2 — 2×2 block matrix
// ---------------------------------------------------------------------------

/// A 2×2 block matrix.
///
/// ```text
/// | a  b |
/// | c  d |
/// ```
///
/// Multiplication with `BlockVec2<U, L>` requires:
/// - `A*U + B*L` is valid and results in the same type
/// - `C*U + D*L` is valid and results in the same type
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlockMat2x2<A, B, C, D> {
    pub a: A,
    pub b: B,
    pub c: C,
    pub d: D,
}

impl<A, B, C, D> BlockMat2x2<A, B, C, D> {
    #[inline(always)]
    pub fn new(a: A, b: B, c: C, d: D) -> Self {
        Self { a, b, c, d }
    }
}

// BlockMat2x2 * BlockVec2 with full trait-bound type checking
impl<A, B, C, D, U, L> Mul<BlockVec2<U, L>> for BlockMat2x2<A, B, C, D>
where
    A: Mul<U>,
    B: Mul<L>,
    C: Mul<U>,
    D: Mul<L>,
    <A as Mul<U>>::Output: Add<<B as Mul<L>>::Output>,
    <C as Mul<U>>::Output: Add<<D as Mul<L>>::Output>,
    U: Clone,
    L: Clone,
{
    type Output = BlockVec2<
        <<A as Mul<U>>::Output as Add<<B as Mul<L>>::Output>>::Output,
        <<C as Mul<U>>::Output as Add<<D as Mul<L>>::Output>>::Output,
    >;
    #[inline(always)]
    fn mul(self, rhs: BlockVec2<U, L>) -> Self::Output {
        BlockVec2 {
            upper: self.a * rhs.upper.clone() + self.b * rhs.lower.clone(),
            lower: self.c * rhs.upper + self.d * rhs.lower,
        }
    }
}

// BlockMat2x2 transpose
impl<A, B, C, D> BlockMat2x2<A, B, C, D> {
    /// Transpose: swap off-diagonal blocks and transpose each block.
    ///
    /// ```text
    /// | A  B |ᵀ   | Aᵀ  Cᵀ |
    /// | C  D |  = | Bᵀ  Dᵀ |
    /// ```
    #[inline(always)]
    pub fn transpose<AT, CT, BT, DT>(self) -> BlockMat2x2<AT, CT, BT, DT>
    where
        A: TransposeBlock<Output = AT>,
        B: TransposeBlock<Output = BT>,
        C: TransposeBlock<Output = CT>,
        D: TransposeBlock<Output = DT>,
    {
        BlockMat2x2 {
            a: self.a.block_transpose(),
            b: self.c.block_transpose(),
            c: self.b.block_transpose(),
            d: self.d.block_transpose(),
        }
    }
}

/// Trait for transposing a block element (used by BlockMat2x2::transpose).
pub trait TransposeBlock {
    type Output;
    fn block_transpose(self) -> Self::Output;
}

impl<DR, DC, const R: usize, const C: usize> TransposeBlock
    for crate::unit_mat::UnitMat<DR, DC, R, C>
{
    type Output = crate::unit_mat::UnitMat<DC, DR, C, R>;
    #[inline(always)]
    fn block_transpose(self) -> Self::Output {
        self.transpose()
    }
}

// BlockMat2x2 + BlockMat2x2
impl<A: Add, B: Add, C: Add, D: Add> Add for BlockMat2x2<A, B, C, D> {
    type Output = BlockMat2x2<A::Output, B::Output, C::Output, D::Output>;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        BlockMat2x2 {
            a: self.a + rhs.a,
            b: self.b + rhs.b,
            c: self.c + rhs.c,
            d: self.d + rhs.d,
        }
    }
}

// BlockMat2x2 * BlockMat2x2
//
// | A1 B1 |   | A2 B2 |   | A1*A2+B1*C2  A1*B2+B1*D2 |
// | C1 D1 | * | C2 D2 | = | C1*A2+D1*C2  C1*B2+D1*D2 |
impl<A1, B1, C1, D1, A2, B2, C2, D2> Mul<BlockMat2x2<A2, B2, C2, D2>>
    for BlockMat2x2<A1, B1, C1, D1>
where
    A1: Mul<A2> + Mul<B2>,
    B1: Mul<C2> + Mul<D2>,
    C1: Mul<A2> + Mul<B2>,
    D1: Mul<C2> + Mul<D2>,
    <A1 as Mul<A2>>::Output: Add<<B1 as Mul<C2>>::Output>,
    <A1 as Mul<B2>>::Output: Add<<B1 as Mul<D2>>::Output>,
    <C1 as Mul<A2>>::Output: Add<<D1 as Mul<C2>>::Output>,
    <C1 as Mul<B2>>::Output: Add<<D1 as Mul<D2>>::Output>,
    A1: Clone,
    B1: Clone,
    C1: Clone,
    D1: Clone,
    A2: Clone,
    B2: Clone,
    C2: Clone,
    D2: Clone,
{
    type Output = BlockMat2x2<
        <<A1 as Mul<A2>>::Output as Add<<B1 as Mul<C2>>::Output>>::Output,
        <<A1 as Mul<B2>>::Output as Add<<B1 as Mul<D2>>::Output>>::Output,
        <<C1 as Mul<A2>>::Output as Add<<D1 as Mul<C2>>::Output>>::Output,
        <<C1 as Mul<B2>>::Output as Add<<D1 as Mul<D2>>::Output>>::Output,
    >;
    #[inline(always)]
    fn mul(self, rhs: BlockMat2x2<A2, B2, C2, D2>) -> Self::Output {
        BlockMat2x2 {
            a: self.a.clone() * rhs.a.clone() + self.b.clone() * rhs.c.clone(),
            b: self.a * rhs.b.clone() + self.b * rhs.d.clone(),
            c: self.c.clone() * rhs.a + self.d.clone() * rhs.c,
            d: self.c * rhs.b + self.d * rhs.d,
        }
    }
}

// ---------------------------------------------------------------------------
// BlockMat3x3 — 3×3 block matrix
// ---------------------------------------------------------------------------

/// A 3×3 block matrix for larger state spaces (e.g. position + velocity + bias).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlockMat3x3<A, B, C, D, E, F, G, H, I> {
    pub a: A,
    pub b: B,
    pub c: C,
    pub d: D,
    pub e: E,
    pub f: F,
    pub g: G,
    pub h: H,
    pub i: I,
}

impl<A, B, C, D, E, F, G, H, I> BlockMat3x3<A, B, C, D, E, F, G, H, I> {
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub fn new(a: A, b: B, c: C, d: D, e: E, f: F, g: G, h: H, i: I) -> Self {
        Self {
            a, b, c, d, e, f, g, h, i,
        }
    }
}

// BlockMat3x3 * BlockVec3
impl<MA, MB, MC, MD, ME, MF, MG, MH, MI, VA, VB, VC>
    Mul<BlockVec3<VA, VB, VC>> for BlockMat3x3<MA, MB, MC, MD, ME, MF, MG, MH, MI>
where
    MA: Mul<VA>,
    MB: Mul<VB>,
    MC: Mul<VC>,
    MD: Mul<VA>,
    ME: Mul<VB>,
    MF: Mul<VC>,
    MG: Mul<VA>,
    MH: Mul<VB>,
    MI: Mul<VC>,
    <MA as Mul<VA>>::Output: Add<<MB as Mul<VB>>::Output>,
    <<MA as Mul<VA>>::Output as Add<<MB as Mul<VB>>::Output>>::Output:
        Add<<MC as Mul<VC>>::Output>,
    <MD as Mul<VA>>::Output: Add<<ME as Mul<VB>>::Output>,
    <<MD as Mul<VA>>::Output as Add<<ME as Mul<VB>>::Output>>::Output:
        Add<<MF as Mul<VC>>::Output>,
    <MG as Mul<VA>>::Output: Add<<MH as Mul<VB>>::Output>,
    <<MG as Mul<VA>>::Output as Add<<MH as Mul<VB>>::Output>>::Output:
        Add<<MI as Mul<VC>>::Output>,
    VA: Clone,
    VB: Clone,
    VC: Clone,
{
    type Output = BlockVec3<
        <<<MA as Mul<VA>>::Output as Add<<MB as Mul<VB>>::Output>>::Output as Add<
            <MC as Mul<VC>>::Output,
        >>::Output,
        <<<MD as Mul<VA>>::Output as Add<<ME as Mul<VB>>::Output>>::Output as Add<
            <MF as Mul<VC>>::Output,
        >>::Output,
        <<<MG as Mul<VA>>::Output as Add<<MH as Mul<VB>>::Output>>::Output as Add<
            <MI as Mul<VC>>::Output,
        >>::Output,
    >;
    #[inline(always)]
    fn mul(self, rhs: BlockVec3<VA, VB, VC>) -> Self::Output {
        BlockVec3 {
            a: (self.a * rhs.a.clone() + self.b * rhs.b.clone()) + self.c * rhs.c.clone(),
            b: (self.d * rhs.a.clone() + self.e * rhs.b.clone()) + self.f * rhs.c.clone(),
            c: (self.g * rhs.a + self.h * rhs.b) + self.i * rhs.c,
        }
    }
}
