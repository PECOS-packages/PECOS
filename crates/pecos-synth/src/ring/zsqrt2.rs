//! The real quadratic integer ring `Z[sqrt(2)]`.

use std::cmp::Ordering;
use std::ops::{Add, Mul, Neg, Sub};

use num_bigint::BigInt;
use num_traits::{One, Signed, Zero};

/// An element `a + b sqrt(2)` of `Z[sqrt(2)]`.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct ZSqrt2 {
    a: BigInt,
    b: BigInt,
}

impl ZSqrt2 {
    /// Constructs `a + b sqrt(2)`.
    #[must_use]
    pub fn new(a: BigInt, b: BigInt) -> Self {
        Self { a, b }
    }

    /// Returns the rational coordinate `a`.
    #[must_use]
    pub const fn rational_part(&self) -> &BigInt {
        &self.a
    }

    /// Returns the `sqrt(2)` coordinate `b`.
    #[must_use]
    pub const fn sqrt2_part(&self) -> &BigInt {
        &self.b
    }

    /// Returns the ring element `sqrt(2)`.
    #[must_use]
    pub fn sqrt2() -> Self {
        Self::new(BigInt::zero(), BigInt::one())
    }

    /// Applies the `sqrt(2)`-conjugation `a + b sqrt(2) -> a - b sqrt(2)`.
    #[must_use]
    pub fn sqrt2_conjugate(&self) -> Self {
        Self::new(self.a.clone(), -self.b.clone())
    }

    /// Returns the algebraic norm `a^2 - 2 b^2`.
    #[must_use]
    pub fn norm(&self) -> BigInt {
        &self.a * &self.a - BigInt::from(2_u8) * &self.b * &self.b
    }

    /// Returns whether this element is non-negative in the real embedding.
    #[must_use]
    pub fn is_non_negative(&self) -> bool {
        self.cmp_zero() != Ordering::Less
    }

    fn cmp_zero(&self) -> Ordering {
        if self.a.is_zero() {
            return self.b.cmp(&BigInt::zero());
        }
        if self.b.is_zero() || self.a.is_positive() == self.b.is_positive() {
            return self.a.cmp(&BigInt::zero());
        }

        // When a and b have opposite signs, the larger of |a| and |b sqrt(2)|
        // determines the sign. Equality cannot occur for nonzero integers:
        // a^2 = 2 b^2 would make sqrt(2) = |a / b| rational.
        let a_squared = &self.a * &self.a;
        let twice_b_squared = BigInt::from(2_u8) * &self.b * &self.b;
        if a_squared > twice_b_squared {
            self.a.cmp(&BigInt::zero())
        } else {
            self.b.cmp(&BigInt::zero())
        }
    }
}

impl Ord for ZSqrt2 {
    fn cmp(&self, other: &Self) -> Ordering {
        Self::new(&self.a - &other.a, &self.b - &other.b).cmp_zero()
    }
}

impl PartialOrd for ZSqrt2 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Zero for ZSqrt2 {
    fn zero() -> Self {
        Self::new(BigInt::zero(), BigInt::zero())
    }

    fn is_zero(&self) -> bool {
        self.a.is_zero() && self.b.is_zero()
    }
}

impl One for ZSqrt2 {
    fn one() -> Self {
        Self::new(BigInt::one(), BigInt::zero())
    }
}

impl Add for ZSqrt2 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        &self + &rhs
    }
}

impl Add for &ZSqrt2 {
    type Output = ZSqrt2;

    fn add(self, rhs: Self) -> Self::Output {
        ZSqrt2::new(&self.a + &rhs.a, &self.b + &rhs.b)
    }
}

impl Sub for ZSqrt2 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        &self - &rhs
    }
}

impl Sub for &ZSqrt2 {
    type Output = ZSqrt2;

    fn sub(self, rhs: Self) -> Self::Output {
        ZSqrt2::new(&self.a - &rhs.a, &self.b - &rhs.b)
    }
}

impl Mul for ZSqrt2 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        &self * &rhs
    }
}

impl Mul for &ZSqrt2 {
    type Output = ZSqrt2;

    fn mul(self, rhs: Self) -> Self::Output {
        ZSqrt2::new(
            &self.a * &rhs.a + BigInt::from(2_u8) * &self.b * &rhs.b,
            &self.a * &rhs.b + &self.b * &rhs.a,
        )
    }
}

impl Neg for ZSqrt2 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::new(-self.a, -self.b)
    }
}

impl Neg for &ZSqrt2 {
    type Output = ZSqrt2;

    fn neg(self) -> Self::Output {
        ZSqrt2::new(-&self.a, -&self.b)
    }
}

#[cfg(test)]
mod tests {
    use num_bigint::BigInt;

    use super::ZSqrt2;
    use crate::ring::test_support::{Lcg, assert_commutative_ring};

    fn sample(rng: &mut Lcg) -> ZSqrt2 {
        ZSqrt2::new(
            BigInt::from(rng.next_i64(30)),
            BigInt::from(rng.next_i64(30)),
        )
    }

    #[test]
    fn ring_axioms() {
        let mut rng = Lcg::new(0x5a5a_1020_3040_5060);
        for _ in 0..128 {
            assert_commutative_ring(&sample(&mut rng), &sample(&mut rng), &sample(&mut rng));
        }
    }

    #[test]
    fn conjugation_is_an_involution_and_norm_is_its_product() {
        let mut rng = Lcg::new(0xb011_e700_1234_5678);
        for _ in 0..128 {
            let value = sample(&mut rng);
            assert_eq!(value.sqrt2_conjugate().sqrt2_conjugate(), value);
            let norm = &value * &value.sqrt2_conjugate();
            assert_eq!(norm.sqrt2_part(), &BigInt::from(0));
            assert_eq!(norm.rational_part(), &value.norm());
        }
    }

    #[test]
    fn exact_ordering_and_non_negative_predicate() {
        let positive_small = ZSqrt2::new(BigInt::from(3), BigInt::from(-2));
        let negative_small = ZSqrt2::new(BigInt::from(-3), BigInt::from(2));
        let positive_large = ZSqrt2::new(BigInt::from(-7), BigInt::from(5));
        let negative_large = ZSqrt2::new(BigInt::from(7), BigInt::from(-5));

        assert!(positive_small.is_non_negative());
        assert!(!negative_small.is_non_negative());
        assert!(positive_large.is_non_negative());
        assert!(!negative_large.is_non_negative());
        assert!(positive_small > positive_large);
    }
}
