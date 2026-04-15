//! SI prefix support as type-level power-of-10 exponents.
//!
//! Each prefix is a `typenum` integer representing the exponent of 10:
//! - `Nano` = -9 → 10⁻⁹
//! - `Micro` = -6 → 10⁻⁶
//! - `Milli` = -3 → 10⁻³
//! - `Base` = 0 → 10⁰ = 1
//! - `Kilo` = 3 → 10³
//! - `Mega` = 6 → 10⁶
//! - `Giga` = 9 → 10⁹
//!
//! When quantities with different prefixes are multiplied or divided,
//! the prefixes are added or subtracted at the type level. Addition
//! requires matching prefixes (compile-time enforced).
//!
//! # Numerical precision
//!
//! By storing values in their natural scale (e.g., 5.0 for 5 nm rather
//! than 5e-9), we keep intermediate values near 1.0, reducing
//! floating-point errors in long computation chains.

use typenum::consts::*;

/// No prefix (SI base units). 10⁰ = 1.
pub type Base = Z0;

/// Nano: 10⁻⁹
pub type Nano = N9;
/// Micro: 10⁻⁶
pub type Micro = N6;
/// Milli: 10⁻³
pub type Milli = N3;

/// Kilo: 10³
pub type Kilo = P3;
/// Mega: 10⁶
pub type Mega = P6;
/// Giga: 10⁹
pub type Giga = P9;

/// Compute 10^n for an integer exponent (no_std compatible).
#[inline(always)]
pub fn pow10_i32(n: i32) -> f64 {
    // For common SI prefix exponents, use exact constants
    match n {
        -9 => 1e-9,
        -6 => 1e-6,
        -3 => 1e-3,
        0 => 1.0,
        3 => 1e3,
        6 => 1e6,
        9 => 1e9,
        _ => {
            // General case: compute via repeated multiplication
            if n >= 0 {
                let mut result = 1.0_f64;
                for _ in 0..n {
                    result *= 10.0;
                }
                result
            } else {
                let mut result = 1.0_f64;
                for _ in 0..(-n) {
                    result /= 10.0;
                }
                result
            }
        }
    }
}

/// Compute 10^P at runtime for a typenum integer P.
#[inline(always)]
pub fn pow10<P: typenum::Integer>() -> f64 {
    pow10_i32(P::to_i64() as i32)
}
