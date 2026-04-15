//! Runtime dimension and prefix name generation.
//!
//! Provides traits to extract human-readable unit strings from type-level
//! dimension and prefix parameters at runtime.

use typenum::Integer;

/// Trait for types that can produce a human-readable dimension string.
///
/// Implemented for `Dim<L, M, T, I, Th, N, J>` when all exponents implement
/// `Integer` (which they always do for typenum integers).
pub trait DimName {
    /// Write the dimension string (e.g., "m·s⁻¹", "kg·m·s⁻²").
    /// Returns empty string for dimensionless.
    fn dim_name(buf: &mut DimNameBuf);
}

/// Trait for prefix types that can produce a prefix string.
pub trait PrefixName {
    /// Prefix symbol (e.g., "k", "M", "n", ""). Empty for base.
    fn prefix_symbol() -> &'static str;
}

// ---- PrefixName implementations ----

impl PrefixName for typenum::Z0 {
    fn prefix_symbol() -> &'static str { "" }
}

// Use a macro for the common prefixes
macro_rules! impl_prefix_name {
    ($ty:ty, $sym:expr) => {
        impl PrefixName for $ty {
            fn prefix_symbol() -> &'static str { $sym }
        }
    };
}

impl_prefix_name!(typenum::N9, "n");
impl_prefix_name!(typenum::N6, "μ");
impl_prefix_name!(typenum::N3, "m");
impl_prefix_name!(typenum::P3, "k");
impl_prefix_name!(typenum::P6, "M");
impl_prefix_name!(typenum::P9, "G");

// Fallback: for any Integer not explicitly listed, show 10^N
// (Can't do blanket impl due to orphan rules, so we cover common cases above)

// ---- DimName for Dim ----

/// Fixed-size buffer for dimension name (no alloc needed).
pub struct DimNameBuf {
    buf: [u8; 64],
    len: usize,
}

impl Default for DimNameBuf {
    fn default() -> Self { Self::new() }
}

impl DimNameBuf {
    pub fn new() -> Self {
        Self { buf: [0; 64], len: 0 }
    }

    fn push_str(&mut self, s: &str) {
        let bytes = s.as_bytes();
        let end = (self.len + bytes.len()).min(self.buf.len());
        let copy_len = end - self.len;
        self.buf[self.len..end].copy_from_slice(&bytes[..copy_len]);
        self.len = end;
    }

    pub fn as_str(&self) -> &str {
        core::str::from_utf8(&self.buf[..self.len]).unwrap_or("?")
    }
}

const SI_SYMBOLS: [&str; 7] = ["m", "kg", "s", "A", "K", "mol", "cd"];

fn write_dim_component(buf: &mut DimNameBuf, symbol: &str, exp: i64, first: &mut bool) {
    if exp == 0 {
        return;
    }
    if !*first {
        buf.push_str("·");
    }
    *first = false;
    buf.push_str(symbol);
    if exp != 1 {
        // Write superscript-style exponent
        if exp == -1 {
            buf.push_str("⁻¹");
        } else if exp == -2 {
            buf.push_str("⁻²");
        } else if exp == -3 {
            buf.push_str("⁻³");
        } else if exp == 2 {
            buf.push_str("²");
        } else if exp == 3 {
            buf.push_str("³");
        } else {
            buf.push_str("^");
            // Simple integer-to-string for small exponents
            if exp < 0 {
                buf.push_str("-");
                write_small_int(buf, (-exp) as u64);
            } else {
                write_small_int(buf, exp as u64);
            }
        }
    }
}

fn write_small_int(buf: &mut DimNameBuf, n: u64) {
    if n >= 10 {
        write_small_int(buf, n / 10);
    }
    let digit = b'0' + (n % 10) as u8;
    buf.buf[buf.len] = digit;
    buf.len += 1;
}

impl<L, M, T, I, Th, N, J> DimName for crate::dim::Dim<L, M, T, I, Th, N, J>
where
    L: Integer,
    M: Integer,
    T: Integer,
    I: Integer,
    Th: Integer,
    N: Integer,
    J: Integer,
{
    fn dim_name(buf: &mut DimNameBuf) {
        let exps = [
            L::to_i64(),
            M::to_i64(),
            T::to_i64(),
            I::to_i64(),
            Th::to_i64(),
            N::to_i64(),
            J::to_i64(),
        ];
        let mut first = true;
        for (i, &exp) in exps.iter().enumerate() {
            write_dim_component(buf, SI_SYMBOLS[i], exp, &mut first);
        }
    }
}
