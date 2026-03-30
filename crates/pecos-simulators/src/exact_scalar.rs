// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License.You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

//! Exact scalar for Clifford+T computation.
//!
//! Represents values of the form `sign * 2^{p/2} * exp(i * pi * e / 4)` where:
//! - `sign` is +1 or -1
//! - `p` is an integer (power of sqrt(2))
//! - `e` is in {0, 1, 2, ..., 7} (eighth root of unity)
//!
//! This representation is closed under multiplication and can exactly represent
//! any scalar that arises during CH-form Clifford gate updates.
//!
//! # References
//!
//! Bravyi, Browne, Calpin, Campbell, Gosset, Howard.
//! "Simulation of quantum circuits by low-rank stabilizer decompositions."
//! [arXiv:1808.00128](https://arxiv.org/abs/1808.00128) (2019).

use num_complex::Complex64;

/// Exact scalar for CH-form computation.
///
/// Represents `sign * 2^{p/2} * exp(i * pi * e / 4)`, or zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExactScalar {
    is_zero: bool,
    sign: bool,     // true = negative
    sqrt2_pow: i32, // p in 2^{p/2}
    phase8: u8,     // e in exp(i * pi * e / 4), mod 8
}

impl ExactScalar {
    /// The scalar 1.
    #[must_use]
    pub fn one() -> Self {
        Self {
            is_zero: false,
            sign: false,
            sqrt2_pow: 0,
            phase8: 0,
        }
    }

    /// The scalar 0.
    #[must_use]
    pub fn zero() -> Self {
        Self {
            is_zero: true,
            sign: false,
            sqrt2_pow: 0,
            phase8: 0,
        }
    }

    /// Create exp(i * pi * e / 4).
    #[must_use]
    pub fn from_phase(e: u8) -> Self {
        Self {
            is_zero: false,
            sign: false,
            sqrt2_pow: 0,
            phase8: e & 7,
        }
    }

    /// Whether this scalar is zero.
    #[inline]
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.is_zero
    }
    #[must_use]
    pub fn sign(&self) -> bool {
        self.sign
    }
    #[must_use]
    pub fn sqrt2_pow(&self) -> i32 {
        self.sqrt2_pow
    }
    #[must_use]
    pub fn phase8(&self) -> u8 {
        self.phase8
    }

    /// Negate: multiply by -1.
    pub fn negate(&mut self) {
        if !self.is_zero {
            self.sign = !self.sign;
        }
    }

    /// Multiply by 2^{dp/2} (shift the sqrt(2) power).
    pub fn mul_sqrt2_pow(&mut self, dp: i32) {
        if !self.is_zero {
            self.sqrt2_pow += dp;
        }
    }

    /// Multiply by exp(i * pi * de / 4) (shift the phase).
    pub fn mul_phase(&mut self, de: u8) {
        if !self.is_zero {
            self.phase8 = (self.phase8 + de) & 7;
        }
    }

    /// Multiply by i^k (= exp(i * pi * k / 2)).
    pub fn mul_i_pow(&mut self, k: u8) {
        // i^k = exp(i * pi * k / 2) = exp(i * pi * (2k) / 4)
        self.mul_phase((2 * k) & 7);
    }

    /// Multiply by another `ExactScalar`.
    #[must_use]
    pub fn mul(&self, other: &Self) -> Self {
        if self.is_zero || other.is_zero {
            return Self::zero();
        }
        Self {
            is_zero: false,
            sign: self.sign ^ other.sign,
            sqrt2_pow: self.sqrt2_pow + other.sqrt2_pow,
            phase8: (self.phase8 + other.phase8) & 7,
        }
    }

    /// Multiply in place.
    pub fn mul_assign(&mut self, other: &Self) {
        *self = self.mul(other);
    }

    /// Convert to `Complex64` for testing / inner product computation.
    #[must_use]
    pub fn to_complex(&self) -> Complex64 {
        if self.is_zero {
            return Complex64::new(0.0, 0.0);
        }

        // 2^{p/2}. Split: even p -> exact 2^{p/2}, odd p -> sqrt(2) * 2^{(p-1)/2}.
        // Using 2.0_f64.powi() avoids the expensive SQRT_2.powi() path.
        let p = self.sqrt2_pow;
        let magnitude = if p & 1 == 0 {
            2.0_f64.powi(p / 2)
        } else if p > 0 {
            std::f64::consts::SQRT_2 * 2.0_f64.powi((p - 1) / 2)
        } else {
            std::f64::consts::FRAC_1_SQRT_2 * 2.0_f64.powi((p + 1) / 2)
        };

        let sign = if self.sign { -1.0 } else { 1.0 };

        // exp(i * pi * e / 4) -- lookup table, no sincos needed.
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let phase = match self.phase8 & 7 {
            0 => Complex64::new(1.0, 0.0),
            1 => Complex64::new(s, s),
            2 => Complex64::new(0.0, 1.0),
            3 => Complex64::new(-s, s),
            4 => Complex64::new(-1.0, 0.0),
            5 => Complex64::new(-s, -s),
            6 => Complex64::new(0.0, -1.0),
            7 => Complex64::new(s, -s),
            _ => unreachable!(),
        };

        phase * sign * magnitude
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::FRAC_PI_4;

    const EPS: f64 = 1e-12;

    fn approx_eq(a: Complex64, b: Complex64) -> bool {
        (a - b).norm() < EPS
    }

    #[test]
    fn test_one() {
        let s = ExactScalar::one();
        assert!(approx_eq(s.to_complex(), Complex64::new(1.0, 0.0)));
    }

    #[test]
    fn test_zero() {
        let s = ExactScalar::zero();
        assert!(approx_eq(s.to_complex(), Complex64::new(0.0, 0.0)));
    }

    #[test]
    fn test_eighth_roots_of_unity() {
        for e in 0..8u8 {
            let s = ExactScalar::from_phase(e);
            let angle = f64::from(e) * FRAC_PI_4;
            let expected = Complex64::new(angle.cos(), angle.sin());
            assert!(
                approx_eq(s.to_complex(), expected),
                "phase {e}: got {:?}, expected {:?}",
                s.to_complex(),
                expected
            );
        }
    }

    #[test]
    fn test_i_squared_is_minus_one() {
        // i = exp(i*pi/2) = phase 2
        let i_val = ExactScalar::from_phase(2);
        let minus_one = i_val.mul(&i_val);
        assert!(approx_eq(minus_one.to_complex(), Complex64::new(-1.0, 0.0)));
    }

    #[test]
    fn test_negate() {
        let mut s = ExactScalar::one();
        s.negate();
        assert!(approx_eq(s.to_complex(), Complex64::new(-1.0, 0.0)));
    }

    #[test]
    fn test_sqrt2_pow() {
        let mut s = ExactScalar::one();
        s.mul_sqrt2_pow(2); // 2^{2/2} = 2
        assert!(approx_eq(s.to_complex(), Complex64::new(2.0, 0.0)));

        let mut s2 = ExactScalar::one();
        s2.mul_sqrt2_pow(-2); // 2^{-2/2} = 1/2
        assert!(approx_eq(s2.to_complex(), Complex64::new(0.5, 0.0)));

        let mut s3 = ExactScalar::one();
        s3.mul_sqrt2_pow(1); // 2^{1/2} = sqrt(2)
        assert!(approx_eq(
            s3.to_complex(),
            Complex64::new(std::f64::consts::SQRT_2, 0.0)
        ));
    }

    #[test]
    fn test_mul_i_pow() {
        let mut s = ExactScalar::one();
        s.mul_i_pow(1); // multiply by i
        assert!(approx_eq(s.to_complex(), Complex64::new(0.0, 1.0)));

        let mut s2 = ExactScalar::one();
        s2.mul_i_pow(2); // multiply by i^2 = -1
        assert!(approx_eq(s2.to_complex(), Complex64::new(-1.0, 0.0)));

        let mut s3 = ExactScalar::one();
        s3.mul_i_pow(3); // multiply by i^3 = -i
        assert!(approx_eq(s3.to_complex(), Complex64::new(0.0, -1.0)));
    }

    #[test]
    fn test_compound_multiplication() {
        // sqrt(2) * exp(i*pi/4) = 1 + i
        let mut s = ExactScalar::one();
        s.mul_sqrt2_pow(1);
        s.mul_phase(1);
        let c = s.to_complex();
        assert!(approx_eq(c, Complex64::new(1.0, 1.0)));
    }

    #[test]
    fn test_zero_absorbs() {
        let z = ExactScalar::zero();
        let s = ExactScalar::from_phase(3);
        assert!(z.mul(&s).is_zero());
        assert!(s.mul(&z).is_zero());
    }

    #[test]
    fn test_mul_associative() {
        let a = ExactScalar::from_phase(1);
        let b = ExactScalar::from_phase(3);
        let c = ExactScalar::from_phase(5);
        let ab_c = a.mul(&b).mul(&c);
        let a_bc = a.mul(&b.mul(&c));
        assert_eq!(ab_c, a_bc);
    }
}
