//! Rational number type for exact fraction representation.
//!
//! This module provides a `Rational` type that represents fractions exactly,
//! avoiding floating-point precision issues. This is particularly important
//! for angle calculations in quantum computing where angles like 1/4 turn
//! (pi/2 radians) must be exact.
//!
//! # Examples
//!
//! ```
//! use zlup::rational::Rational;
//!
//! let quarter = Rational::new(1, 4);
//! let half = Rational::new(1, 2);
//! assert_eq!(quarter + quarter, half);
//! ```

use std::cmp::Ordering;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// A rational number represented as numerator/denominator.
///
/// Rationals are always stored in lowest terms with a positive denominator.
#[derive(Clone, Copy, Eq, PartialEq, Hash)]
pub struct Rational {
    /// Numerator (can be negative)
    num: i64,
    /// Denominator (always positive, never zero)
    den: u64,
}

impl Rational {
    /// Create a new rational number, automatically reducing to lowest terms.
    pub fn new(numerator: i64, denominator: i64) -> Self {
        if denominator == 0 {
            panic!("Rational denominator cannot be zero");
        }

        // Normalize sign: denominator is always positive
        let (num, den) = if denominator < 0 {
            (-numerator, (-denominator) as u64)
        } else {
            (numerator, denominator as u64)
        };

        // Reduce to lowest terms
        let g = gcd(num.unsigned_abs(), den);
        Self {
            num: num / g as i64,
            den: den / g,
        }
    }

    /// Create a rational from an integer.
    pub fn from_int(n: i64) -> Self {
        Self { num: n, den: 1 }
    }

    /// Create zero.
    pub const ZERO: Rational = Rational { num: 0, den: 1 };

    /// Create one.
    pub const ONE: Rational = Rational { num: 1, den: 1 };

    /// Create one half.
    pub const HALF: Rational = Rational { num: 1, den: 2 };

    /// Create one quarter.
    pub const QUARTER: Rational = Rational { num: 1, den: 4 };

    /// Create one eighth.
    pub const EIGHTH: Rational = Rational { num: 1, den: 8 };

    /// Get the numerator.
    pub fn numerator(&self) -> i64 {
        self.num
    }

    /// Get the denominator.
    pub fn denominator(&self) -> u64 {
        self.den
    }

    /// Check if this is zero.
    pub fn is_zero(&self) -> bool {
        self.num == 0
    }

    /// Check if this is an integer.
    pub fn is_integer(&self) -> bool {
        self.den == 1
    }

    /// Convert to an integer if exact, otherwise None.
    pub fn to_integer(&self) -> Option<i64> {
        if self.den == 1 { Some(self.num) } else { None }
    }

    /// Convert to f64.
    pub fn to_f64(&self) -> f64 {
        self.num as f64 / self.den as f64
    }

    /// Try to convert an f64 to a rational.
    ///
    /// Uses continued fraction approximation to find a rational with
    /// denominator up to `max_denominator` that approximates the float.
    pub fn from_f64(value: f64, max_denominator: u64) -> Option<Self> {
        if !value.is_finite() {
            return None;
        }

        // Handle negative values
        if value < 0.0 {
            return Self::from_f64(-value, max_denominator).map(|r| -r);
        }

        // Handle zero
        if value == 0.0 {
            return Some(Self::ZERO);
        }

        // Handle integers
        if value == value.floor() && value.abs() < i64::MAX as f64 {
            return Some(Self::from_int(value as i64));
        }

        // Continued fraction approximation
        let mut x = value;
        let a0 = x.floor() as i64;

        // Build convergents
        let mut h_prev: i64 = 1;
        let mut k_prev: u64 = 0;
        let mut h_curr: i64 = a0;
        let mut k_curr: u64 = 1;

        const MAX_ITERATIONS: usize = 50;
        const TOLERANCE: f64 = 1e-15;

        for _ in 0..MAX_ITERATIONS {
            let frac = x - x.floor();
            if frac.abs() < TOLERANCE {
                break;
            }

            x = 1.0 / frac;
            let a = x.floor() as i64;

            // Compute next convergent
            let h_next = a.saturating_mul(h_curr).saturating_add(h_prev);
            let k_next = (a as u64).saturating_mul(k_curr).saturating_add(k_prev);

            if k_next > max_denominator {
                break;
            }

            h_prev = h_curr;
            k_prev = k_curr;
            h_curr = h_next;
            k_curr = k_next;

            // Check if we've converged
            let approx = h_curr as f64 / k_curr as f64;
            if (approx - value).abs() < TOLERANCE {
                break;
            }
        }

        Some(Self::new(h_curr, k_curr as i64))
    }

    /// Try to recognize a common fraction from a float value.
    ///
    /// Returns Some if the value is very close to a common fraction
    /// like 1/2, 1/4, 1/8, 1/3, etc.
    pub fn from_f64_common(value: f64) -> Option<Self> {
        const TOLERANCE: f64 = 1e-12;

        // Common fractions to check
        static COMMON: &[(f64, i64, u64)] = &[
            (0.0, 0, 1),
            (1.0, 1, 1),
            (0.5, 1, 2),
            (0.25, 1, 4),
            (0.125, 1, 8),
            (0.0625, 1, 16),
            (0.75, 3, 4),
            (0.375, 3, 8),
            (0.625, 5, 8),
            (0.875, 7, 8),
            // Thirds
            (1.0 / 3.0, 1, 3),
            (2.0 / 3.0, 2, 3),
            // Sixths
            (1.0 / 6.0, 1, 6),
            (5.0 / 6.0, 5, 6),
            // Twelfths
            (1.0 / 12.0, 1, 12),
            (5.0 / 12.0, 5, 12),
            (7.0 / 12.0, 7, 12),
            (11.0 / 12.0, 11, 12),
        ];

        // Handle negative
        let (abs_value, sign) = if value < 0.0 {
            (-value, -1i64)
        } else {
            (value, 1i64)
        };

        // Check for common fractions
        for &(frac_val, num, den) in COMMON {
            if (abs_value - frac_val).abs() < TOLERANCE {
                return Some(Self {
                    num: sign * num,
                    den,
                });
            }
        }

        // Check for fractions with small denominators (1-16)
        for den in 1u64..=16 {
            let num = (abs_value * den as f64).round() as i64;
            if num >= 0 {
                let approx = num as f64 / den as f64;
                if (approx - abs_value).abs() < TOLERANCE {
                    return Some(Self::new(sign * num, den as i64));
                }
            }
        }

        None
    }

    /// Convert an f64 to its exact rational representation.
    ///
    /// Every IEEE 754 float is exactly representable as a rational number
    /// (specifically, a dyadic rational with power-of-2 denominator).
    /// This is similar to Python's `fractions.Fraction.from_float()`.
    ///
    /// Note: The resulting rational may have a very large denominator.
    /// Use `limit_denominator()` to find a simpler approximation.
    ///
    /// # Examples
    ///
    /// ```
    /// use zlup::rational::Rational;
    ///
    /// // Exact representations
    /// assert_eq!(Rational::from_f64_exact(0.5), Some(Rational::new(1, 2)));
    /// assert_eq!(Rational::from_f64_exact(0.25), Some(Rational::new(1, 4)));
    ///
    /// // 0.1 is not exactly representable in binary, so we get the exact
    /// // IEEE 754 representation as a rational
    /// let r = Rational::from_f64_exact(0.1).unwrap();
    /// assert_eq!(r.to_f64(), 0.1); // Round-trips exactly
    /// ```
    pub fn from_f64_exact(value: f64) -> Option<Self> {
        if !value.is_finite() {
            return None;
        }

        if value == 0.0 {
            return Some(Self::ZERO);
        }

        // Handle negative values
        let (abs_value, sign) = if value < 0.0 {
            (-value, -1i64)
        } else {
            (value, 1i64)
        };

        // Decompose the float into mantissa and exponent
        // f64 = mantissa * 2^exponent where mantissa is in [1, 2)
        // But we want the integer mantissa representation
        let bits = abs_value.to_bits();
        let exponent_bits = ((bits >> 52) & 0x7FF) as i32;
        let mantissa_bits = bits & 0x000F_FFFF_FFFF_FFFF;

        if exponent_bits == 0 {
            // Subnormal number
            // Value = mantissa_bits * 2^(-1022 - 52)
            let num = sign * (mantissa_bits as i64);
            let exp = 1022 + 52;
            // Denominator is 2^exp, which is huge for subnormals
            // We'll simplify by dividing out common factors of 2
            return Self::from_mantissa_exp(num, exp);
        }

        // Normal number
        // The implicit leading 1 bit: mantissa = 1.mantissa_bits
        // So integer mantissa = (1 << 52) | mantissa_bits
        let int_mantissa = (1u64 << 52) | mantissa_bits;
        let exponent = exponent_bits - 1023 - 52; // Subtract bias and mantissa bits

        let num = sign * (int_mantissa as i64);

        if exponent >= 0 {
            // Value = mantissa * 2^exponent (integer result)
            if exponent < 63 {
                Some(Self::from_int(num << exponent))
            } else {
                // Too large, would overflow
                None
            }
        } else {
            // Value = mantissa / 2^(-exponent)
            Self::from_mantissa_exp(num, (-exponent) as u32)
        }
    }

    /// Helper: create rational from mantissa / 2^exp, reducing common factors
    fn from_mantissa_exp(mantissa: i64, exp: u32) -> Option<Self> {
        if mantissa == 0 {
            return Some(Self::ZERO);
        }

        // Count trailing zeros in mantissa to reduce the fraction
        let trailing_zeros = (mantissa.unsigned_abs()).trailing_zeros();
        let reduced_mantissa = mantissa >> trailing_zeros;
        let reduced_exp = exp.saturating_sub(trailing_zeros);

        if reduced_exp > 62 {
            // Denominator would overflow u64
            return None;
        }

        let denominator = 1u64 << reduced_exp;
        Some(Self {
            num: reduced_mantissa,
            den: denominator,
        })
    }

    /// Find the closest rational with denominator at most `max_denominator`.
    ///
    /// Similar to Python's `Fraction.limit_denominator()`. Useful for
    /// simplifying exact float conversions to human-readable fractions.
    ///
    /// # Examples
    ///
    /// ```
    /// use zlup::rational::Rational;
    ///
    /// // The exact representation of 0.1 has a huge denominator
    /// let exact = Rational::from_f64_exact(0.1).unwrap();
    ///
    /// // Limit to denominator <= 10 gives us 1/10
    /// let simple = exact.limit_denominator(10);
    /// assert_eq!(simple, Rational::new(1, 10));
    /// ```
    pub fn limit_denominator(&self, max_denominator: u64) -> Self {
        if self.den <= max_denominator {
            return *self;
        }

        // Use continued fraction algorithm to find best approximation
        // This is the standard algorithm from Python's fractions module
        let mut p0: i64 = 0;
        let mut q0: u64 = 1;
        let mut p1: i64 = 1;
        let mut q1: u64 = 0;

        let mut n = self.num.abs();
        let mut d = self.den;

        loop {
            let a = n / d as i64;
            let q2 = q0 + (a as u64) * q1;

            if q2 > max_denominator {
                break;
            }

            let p2 = p0 + a * p1;
            p0 = p1;
            q0 = q1;
            p1 = p2;
            q1 = q2;

            let new_n = d as i64;
            d = (n % d as i64) as u64;
            n = new_n;

            if d == 0 {
                break;
            }
        }

        // Choose between p1/q1 and the mediant
        let k = (max_denominator - q0) / q1;
        let bound1 = Self::new(p0 + (k as i64) * p1, (q0 + k * q1) as i64);
        let bound2 = Self::new(p1, q1 as i64);

        let abs_self = self.abs();
        let diff1 = (abs_self - bound1).abs();
        let diff2 = (abs_self - bound2).abs();

        let result = if diff1 <= diff2 { bound1 } else { bound2 };

        if self.num < 0 { -result } else { result }
    }

    /// Smart float-to-rational conversion.
    ///
    /// Tries multiple strategies in order of preference:
    /// 1. Exact integer check
    /// 2. Common fractions (1/2, 1/3, 1/4, etc.)
    /// 3. Exact IEEE 754 conversion, limited to reasonable denominator
    /// 4. Continued fraction approximation
    ///
    /// This is the recommended method for converting floats to rationals.
    pub fn from_f64_best(value: f64, max_denominator: u64) -> Option<Self> {
        if !value.is_finite() {
            return None;
        }

        // 1. Check for zero
        if value == 0.0 {
            return Some(Self::ZERO);
        }

        // 2. Check for exact integer
        if value == value.floor() && value.abs() < i64::MAX as f64 {
            return Some(Self::from_int(value as i64));
        }

        // 3. Check for common fractions (most quantum angles)
        if let Some(r) = Self::from_f64_common(value)
            && r.den <= max_denominator
        {
            return Some(r);
        }

        // 4. Try exact conversion with limit
        if let Some(exact) = Self::from_f64_exact(value) {
            let limited = exact.limit_denominator(max_denominator);
            // Verify it's a good approximation
            if (limited.to_f64() - value).abs() < 1e-15 * value.abs().max(1.0) {
                return Some(limited);
            }
        }

        // 5. Fall back to continued fraction approximation
        Self::from_f64(value, max_denominator)
    }

    /// Return the absolute value.
    pub fn abs(&self) -> Self {
        Self {
            num: self.num.abs(),
            den: self.den,
        }
    }

    /// Return the reciprocal (1/self).
    pub fn recip(&self) -> Self {
        if self.num == 0 {
            panic!("Cannot take reciprocal of zero");
        }
        if self.num > 0 {
            Self {
                num: self.den as i64,
                den: self.num as u64,
            }
        } else {
            Self {
                num: -(self.den as i64),
                den: (-self.num) as u64,
            }
        }
    }

    /// Try to recognize a float value as a rational multiple of pi.
    ///
    /// If `value ≈ (n/d) * pi`, returns `Some((n, d))`.
    /// This is useful for converting radians to turns while preserving precision.
    pub fn from_f64_pi_multiple(value: f64) -> Option<(i64, u64)> {
        use std::f64::consts::PI;
        const TOLERANCE: f64 = 1e-12;

        if !value.is_finite() {
            return None;
        }

        let (abs_value, sign) = if value < 0.0 {
            (-value, -1i64)
        } else {
            (value, 1i64)
        };

        // Try to express as (n/d) * pi for small denominators
        for d in 1u64..=16 {
            let n_float = abs_value * d as f64 / PI;
            let n = n_float.round() as i64;

            if n > 0 {
                let expected = n as f64 * PI / d as f64;
                if (expected - abs_value).abs() < TOLERANCE * abs_value.max(1.0) {
                    let r = Self::new(sign * n, d as i64);
                    return Some((r.num, r.den));
                }
            }
        }

        None
    }

    /// Convert a radian value (as a float) to turns as a Rational.
    ///
    /// Detects if the radian value is a rational multiple of pi and preserves
    /// that precision. If `radians = (n/d) * pi`, then `turns = n / (2*d)`.
    pub fn radians_to_turns(radians: f64) -> Option<Self> {
        // First try to detect if this is a rational multiple of pi
        if let Some((n, d)) = Self::from_f64_pi_multiple(radians) {
            // (n/d) * pi radians = n / (2*d) turns
            return Some(Self::new(n, 2 * d as i64));
        }

        // Fall back to direct conversion
        let turns = radians / (2.0 * std::f64::consts::PI);
        Self::from_f64_best(turns, 1000)
    }

    /// Convert turns (as a Rational) to radians as a float.
    pub fn turns_to_radians(&self) -> f64 {
        self.to_f64() * 2.0 * std::f64::consts::PI
    }
}

impl fmt::Debug for Rational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.den == 1 {
            write!(f, "Rational({})", self.num)
        } else {
            write!(f, "Rational({}/{})", self.num, self.den)
        }
    }
}

impl fmt::Display for Rational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.den == 1 {
            write!(f, "{}", self.num)
        } else {
            write!(f, "{}/{}", self.num, self.den)
        }
    }
}

impl Default for Rational {
    fn default() -> Self {
        Self::ZERO
    }
}

impl From<i64> for Rational {
    fn from(n: i64) -> Self {
        Self::from_int(n)
    }
}

impl From<i32> for Rational {
    fn from(n: i32) -> Self {
        Self::from_int(n as i64)
    }
}

impl PartialOrd for Rational {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Rational {
    fn cmp(&self, other: &Self) -> Ordering {
        // a/b compared to c/d: compare a*d to c*b
        let lhs = self.num as i128 * other.den as i128;
        let rhs = other.num as i128 * self.den as i128;
        lhs.cmp(&rhs)
    }
}

impl Add for Rational {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        // a/b + c/d = (a*d + c*b) / (b*d)
        let num = self.num as i128 * other.den as i128 + other.num as i128 * self.den as i128;
        let den = self.den as i128 * other.den as i128;
        Self::new(num as i64, den as i64)
    }
}

impl Sub for Rational {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        self + (-other)
    }
}

impl Mul for Rational {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        // a/b * c/d = (a*c) / (b*d)
        // Cross-reduce first to avoid overflow
        let g1 = gcd(self.num.unsigned_abs(), other.den);
        let g2 = gcd(other.num.unsigned_abs(), self.den);

        let num = (self.num / g1 as i64) * (other.num / g2 as i64);
        let den = (self.den / g2) * (other.den / g1);

        Self { num, den }
    }
}

impl Div for Rational {
    type Output = Self;

    // Using multiplication by reciprocal is the standard way to implement
    // division for rationals: a/b ÷ c/d = a/b × d/c
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, other: Self) -> Self {
        self * other.recip()
    }
}

impl Neg for Rational {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            num: -self.num,
            den: self.den,
        }
    }
}

/// Greatest common divisor using Euclidean algorithm.
fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a.max(1) // Ensure we never return 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_reduces() {
        let r = Rational::new(2, 4);
        assert_eq!(r.numerator(), 1);
        assert_eq!(r.denominator(), 2);
    }

    #[test]
    fn test_negative_denominator() {
        let r = Rational::new(1, -2);
        assert_eq!(r.numerator(), -1);
        assert_eq!(r.denominator(), 2);
    }

    #[test]
    fn test_add() {
        let a = Rational::new(1, 4);
        let b = Rational::new(1, 4);
        assert_eq!(a + b, Rational::new(1, 2));
    }

    #[test]
    fn test_sub() {
        let a = Rational::new(1, 2);
        let b = Rational::new(1, 4);
        assert_eq!(a - b, Rational::new(1, 4));
    }

    #[test]
    fn test_mul() {
        let a = Rational::new(2, 3);
        let b = Rational::new(3, 4);
        assert_eq!(a * b, Rational::new(1, 2));
    }

    #[test]
    fn test_div() {
        let a = Rational::new(1, 2);
        let b = Rational::new(1, 4);
        assert_eq!(a / b, Rational::new(2, 1));
    }

    #[test]
    fn test_to_f64() {
        let r = Rational::new(1, 4);
        assert_eq!(r.to_f64(), 0.25);
    }

    #[test]
    fn test_from_f64_common() {
        assert_eq!(Rational::from_f64_common(0.25), Some(Rational::new(1, 4)));
        assert_eq!(Rational::from_f64_common(0.125), Some(Rational::new(1, 8)));
        assert_eq!(Rational::from_f64_common(0.5), Some(Rational::new(1, 2)));
        assert_eq!(
            Rational::from_f64_common(1.0 / 3.0),
            Some(Rational::new(1, 3))
        );
    }

    #[test]
    fn test_from_f64_approximation() {
        let r = Rational::from_f64(0.333333333, 100).unwrap();
        assert_eq!(r, Rational::new(1, 3));
    }

    #[test]
    fn test_comparison() {
        let a = Rational::new(1, 3);
        let b = Rational::new(1, 4);
        assert!(a > b);
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", Rational::new(1, 4)), "1/4");
        assert_eq!(format!("{}", Rational::new(3, 1)), "3");
    }

    #[test]
    fn test_from_f64_exact_dyadic() {
        // Dyadic fractions (power of 2 denominators) are exactly representable
        assert_eq!(Rational::from_f64_exact(0.5), Some(Rational::new(1, 2)));
        assert_eq!(Rational::from_f64_exact(0.25), Some(Rational::new(1, 4)));
        assert_eq!(Rational::from_f64_exact(0.125), Some(Rational::new(1, 8)));
        assert_eq!(Rational::from_f64_exact(0.0625), Some(Rational::new(1, 16)));
        assert_eq!(Rational::from_f64_exact(0.75), Some(Rational::new(3, 4)));
        assert_eq!(Rational::from_f64_exact(0.375), Some(Rational::new(3, 8)));
    }

    #[test]
    fn test_from_f64_exact_roundtrip() {
        // Any float should round-trip through exact conversion
        let values = [0.1, 0.2, 0.3, 0.7, 1.1, 3.14159, 0.123456789];
        for &v in &values {
            let r = Rational::from_f64_exact(v).unwrap();
            assert_eq!(r.to_f64(), v, "Round-trip failed for {}", v);
        }
    }

    #[test]
    fn test_from_f64_exact_negative() {
        assert_eq!(Rational::from_f64_exact(-0.5), Some(Rational::new(-1, 2)));
        assert_eq!(Rational::from_f64_exact(-0.25), Some(Rational::new(-1, 4)));
    }

    #[test]
    fn test_from_f64_exact_special() {
        assert_eq!(Rational::from_f64_exact(0.0), Some(Rational::ZERO));
        assert_eq!(Rational::from_f64_exact(f64::NAN), None);
        assert_eq!(Rational::from_f64_exact(f64::INFINITY), None);
        assert_eq!(Rational::from_f64_exact(f64::NEG_INFINITY), None);
    }

    #[test]
    fn test_limit_denominator() {
        // 0.1 exact representation has large denominator
        let exact = Rational::from_f64_exact(0.1).unwrap();
        assert!(
            exact.denominator() > 10,
            "0.1 exact should have large denominator"
        );

        // Limit to 10 should give 1/10
        let limited = exact.limit_denominator(10);
        assert_eq!(limited, Rational::new(1, 10));

        // Verify it's close enough
        assert!((limited.to_f64() - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_limit_denominator_already_small() {
        let r = Rational::new(1, 4);
        let limited = r.limit_denominator(100);
        assert_eq!(limited, Rational::new(1, 4));
    }

    #[test]
    fn test_limit_denominator_pi_approximations() {
        // Pi ≈ 3.14159...
        let pi_exact = Rational::from_f64_exact(std::f64::consts::PI).unwrap();

        // Famous approximations:
        // 22/7 ≈ 3.142857 (denominator 7)
        let approx_7 = pi_exact.limit_denominator(10);
        assert_eq!(approx_7, Rational::new(22, 7));

        // 333/106 ≈ 3.141509 (denominator 106)
        let approx_1000 = pi_exact.limit_denominator(1000);
        assert!((approx_1000.to_f64() - std::f64::consts::PI).abs() < 0.0001);
    }

    #[test]
    fn test_from_f64_best() {
        // Common fractions should be recognized
        assert_eq!(
            Rational::from_f64_best(0.25, 100),
            Some(Rational::new(1, 4))
        );
        assert_eq!(Rational::from_f64_best(0.5, 100), Some(Rational::new(1, 2)));
        assert_eq!(
            Rational::from_f64_best(1.0 / 3.0, 100),
            Some(Rational::new(1, 3))
        );

        // Integers
        assert_eq!(Rational::from_f64_best(5.0, 100), Some(Rational::new(5, 1)));

        // Arbitrary value should get best approximation
        let r = Rational::from_f64_best(0.1, 100).unwrap();
        assert_eq!(r, Rational::new(1, 10));
    }

    #[test]
    fn test_quantum_angle_fractions() {
        // Common quantum computing angles as fractions of a turn
        // T-gate: 1/8 turn
        assert_eq!(
            Rational::from_f64_best(0.125, 100),
            Some(Rational::new(1, 8))
        );

        // S-gate: 1/4 turn
        assert_eq!(
            Rational::from_f64_best(0.25, 100),
            Some(Rational::new(1, 4))
        );

        // Z-gate: 1/2 turn
        assert_eq!(Rational::from_f64_best(0.5, 100), Some(Rational::new(1, 2)));

        // T-dagger: 7/8 turn
        assert_eq!(
            Rational::from_f64_best(0.875, 100),
            Some(Rational::new(7, 8))
        );

        // S-dagger: 3/4 turn
        assert_eq!(
            Rational::from_f64_best(0.75, 100),
            Some(Rational::new(3, 4))
        );
    }
}
