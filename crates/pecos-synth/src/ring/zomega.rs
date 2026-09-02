//! The cyclotomic integer ring `Z[omega]`, where `omega = exp(i pi / 4)`.

use std::ops::{Add, Mul, Neg, Sub};

use num_bigint::BigInt;
use num_traits::{One, Zero};

use super::ZSqrt2;

/// An element of `Z[omega]` in the basis `(1, omega, omega^2, omega^3)`.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct ZOmega {
    coordinates: [BigInt; 4],
}

impl ZOmega {
    /// Constructs `x0 + x1 omega + x2 omega^2 + x3 omega^3`.
    #[must_use]
    pub fn new(x0: BigInt, x1: BigInt, x2: BigInt, x3: BigInt) -> Self {
        Self {
            coordinates: [x0, x1, x2, x3],
        }
    }

    /// Returns the coordinates in the basis `(1, omega, omega^2, omega^3)`.
    #[must_use]
    pub const fn coordinates(&self) -> &[BigInt; 4] {
        &self.coordinates
    }

    /// Returns the ring element `omega`.
    #[must_use]
    pub fn omega() -> Self {
        Self::new(
            BigInt::zero(),
            BigInt::one(),
            BigInt::zero(),
            BigInt::zero(),
        )
    }

    /// Returns the imaginary unit `i = omega^2`.
    #[must_use]
    pub fn i() -> Self {
        Self::new(
            BigInt::zero(),
            BigInt::zero(),
            BigInt::one(),
            BigInt::zero(),
        )
    }

    /// Returns `sqrt(2) = omega - omega^3`.
    #[must_use]
    pub fn sqrt2() -> Self {
        // omega = (1 + i) / sqrt(2) and omega^3 = (-1 + i) / sqrt(2),
        // so omega - omega^3 = 2 / sqrt(2) = sqrt(2).
        Self::new(
            BigInt::zero(),
            BigInt::one(),
            BigInt::zero(),
            -BigInt::one(),
        )
    }

    /// Applies the bullet automorphism `sigma_5`, which maps `omega` to `-omega`.
    ///
    /// This automorphism fixes `i` and negates `sqrt(2)`.
    #[must_use]
    pub fn sqrt2_conjugate(&self) -> Self {
        let [x0, x1, x2, x3] = &self.coordinates;
        Self::new(x0.clone(), -x1, x2.clone(), -x3)
    }

    /// Applies complex conjugation `sigma_7`, which maps `omega` to `omega^7`.
    ///
    /// This automorphism negates `i` and fixes `sqrt(2)`.
    #[must_use]
    pub fn conjugate(&self) -> Self {
        let [x0, x1, x2, x3] = &self.coordinates;
        // omega^7 = -omega^3, omega^14 = -omega^2, and omega^21 = -omega;
        // therefore [x0, x1, x2, x3] maps to [x0, -x3, -x2, -x1].
        Self::new(x0.clone(), -x3, -x2, -x1)
    }

    /// Returns `self^dagger * self` as an element of `Z[sqrt(2)]`.
    #[must_use]
    pub fn norm_squared(&self) -> ZSqrt2 {
        let product = &self.conjugate() * self;
        debug_assert!(product.coordinates[2].is_zero());
        debug_assert_eq!(product.coordinates[3], -&product.coordinates[1]);
        ZSqrt2::new(
            product.coordinates[0].clone(),
            product.coordinates[1].clone(),
        )
    }

    /// Returns whether this element is divisible by `sqrt(2)` in `Z[omega]`.
    #[must_use]
    pub fn is_divisible_by_sqrt2(&self) -> bool {
        let [y0, y1, y2, y3] = &self.coordinates;
        let two = BigInt::from(2_u8);
        ((y0 - y2) % &two).is_zero() && ((y1 - y3) % two).is_zero()
    }

    /// Divides this element exactly by `sqrt(2)`, if it is divisible.
    #[must_use]
    pub fn div_sqrt2(&self) -> Option<Self> {
        if !self.is_divisible_by_sqrt2() {
            return None;
        }

        // For x = [x0, x1, x2, x3], reduction by omega^4 = -1 gives
        // x * omega   = [-x3, x0, x1, x2]
        // x * omega^3 = [-x1, -x2, -x3, x0].
        // Since sqrt(2) = omega - omega^3,
        // x * sqrt(2) = [x1 - x3, x0 + x2, x1 + x3, x2 - x0].
        // Thus output coordinates 0 and 2 share parity, as do coordinates 1
        // and 3. Conversely those two parity congruences make every half in
        // the inverse map below integral, proving the divisibility lemma.
        let [y0, y1, y2, y3] = &self.coordinates;
        let two = BigInt::from(2_u8);
        Some(Self::new(
            (y1 - y3) / &two,
            (y0 + y2) / &two,
            (y1 + y3) / &two,
            (y2 - y0) / two,
        ))
    }

    pub(super) fn mul_sqrt2_pow(&self, exponent: u32) -> Self {
        let scalar = BigInt::from(2_u8).pow(exponent / 2);
        let scaled = Self::new(
            &self.coordinates[0] * &scalar,
            &self.coordinates[1] * &scalar,
            &self.coordinates[2] * &scalar,
            &self.coordinates[3] * scalar,
        );
        if exponent.is_multiple_of(2) {
            scaled
        } else {
            &scaled * &Self::sqrt2()
        }
    }
}

impl From<BigInt> for ZOmega {
    fn from(value: BigInt) -> Self {
        Self::new(value, BigInt::zero(), BigInt::zero(), BigInt::zero())
    }
}

impl From<i64> for ZOmega {
    fn from(value: i64) -> Self {
        Self::from(BigInt::from(value))
    }
}

impl Zero for ZOmega {
    fn zero() -> Self {
        Self::from(BigInt::zero())
    }

    fn is_zero(&self) -> bool {
        self.coordinates.iter().all(Zero::is_zero)
    }
}

impl One for ZOmega {
    fn one() -> Self {
        Self::from(BigInt::one())
    }
}

impl Add for ZOmega {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        &self + &rhs
    }
}

impl Add for &ZOmega {
    type Output = ZOmega;

    fn add(self, rhs: Self) -> Self::Output {
        ZOmega::new(
            &self.coordinates[0] + &rhs.coordinates[0],
            &self.coordinates[1] + &rhs.coordinates[1],
            &self.coordinates[2] + &rhs.coordinates[2],
            &self.coordinates[3] + &rhs.coordinates[3],
        )
    }
}

impl Sub for ZOmega {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        &self - &rhs
    }
}

impl Sub for &ZOmega {
    type Output = ZOmega;

    fn sub(self, rhs: Self) -> Self::Output {
        ZOmega::new(
            &self.coordinates[0] - &rhs.coordinates[0],
            &self.coordinates[1] - &rhs.coordinates[1],
            &self.coordinates[2] - &rhs.coordinates[2],
            &self.coordinates[3] - &rhs.coordinates[3],
        )
    }
}

impl Mul for ZOmega {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        &self * &rhs
    }
}

impl Mul for &ZOmega {
    type Output = ZOmega;

    fn mul(self, rhs: Self) -> Self::Output {
        let [a0, a1, a2, a3] = &self.coordinates;
        let [b0, b1, b2, b3] = &rhs.coordinates;
        // Terms of degree at least four acquire a minus sign because omega^4 = -1.
        ZOmega::new(
            a0 * b0 - a1 * b3 - a2 * b2 - a3 * b1,
            a0 * b1 + a1 * b0 - a2 * b3 - a3 * b2,
            a0 * b2 + a1 * b1 + a2 * b0 - a3 * b3,
            a0 * b3 + a1 * b2 + a2 * b1 + a3 * b0,
        )
    }
}

impl Neg for ZOmega {
    type Output = Self;

    fn neg(self) -> Self::Output {
        let [x0, x1, x2, x3] = self.coordinates;
        Self::new(-x0, -x1, -x2, -x3)
    }
}

impl Neg for &ZOmega {
    type Output = ZOmega;

    fn neg(self) -> Self::Output {
        ZOmega::new(
            -&self.coordinates[0],
            -&self.coordinates[1],
            -&self.coordinates[2],
            -&self.coordinates[3],
        )
    }
}

#[cfg(test)]
mod tests {
    use num_bigint::BigInt;
    use num_traits::{One, Zero};

    use super::ZOmega;
    use crate::ring::test_support::{Lcg, assert_commutative_ring};

    fn sample(rng: &mut Lcg) -> ZOmega {
        ZOmega::new(
            BigInt::from(rng.next_i64(15)),
            BigInt::from(rng.next_i64(15)),
            BigInt::from(rng.next_i64(15)),
            BigInt::from(rng.next_i64(15)),
        )
    }

    fn independent_division_candidate(value: &ZOmega) -> ZOmega {
        let [y0, y1, y2, y3] = value.coordinates();
        let two = BigInt::from(2_u8);
        ZOmega::new(
            (y1 - y3) / &two,
            (y0 + y2) / &two,
            (y1 + y3) / &two,
            (y2 - y0) / two,
        )
    }

    #[test]
    fn ring_axioms() {
        let mut rng = Lcg::new(0x0a3e_6a12_c900_4147);
        for _ in 0..128 {
            assert_commutative_ring(&sample(&mut rng), &sample(&mut rng), &sample(&mut rng));
        }
    }

    #[test]
    fn powers_and_sqrt2_identity() {
        let omega = ZOmega::omega();
        let omega2 = &omega * &omega;
        let omega4 = &omega2 * &omega2;
        let omega8 = &omega4 * &omega4;

        assert_eq!(omega2, ZOmega::i());
        assert_eq!(omega4, -ZOmega::one());
        assert_eq!(omega8, ZOmega::one());
        assert_eq!(&ZOmega::sqrt2() * &ZOmega::sqrt2(), ZOmega::from(2_i64));
    }

    #[test]
    fn automorphisms_have_the_required_element_actions() {
        assert_eq!(ZOmega::sqrt2().sqrt2_conjugate(), -ZOmega::sqrt2());
        assert_eq!(ZOmega::i().sqrt2_conjugate(), ZOmega::i());
        assert_eq!(ZOmega::i().conjugate(), -ZOmega::i());
        assert_eq!(ZOmega::sqrt2().conjugate(), ZOmega::sqrt2());
    }

    #[test]
    fn automorphisms_are_commuting_involutions() {
        let mut rng = Lcg::new(0x7711_3eaf_5c21_090d);
        for _ in 0..128 {
            let value = sample(&mut rng);
            assert_eq!(value.sqrt2_conjugate().sqrt2_conjugate(), value);
            assert_eq!(value.conjugate().conjugate(), value);
            assert_eq!(
                value.sqrt2_conjugate().conjugate(),
                value.conjugate().sqrt2_conjugate()
            );
        }
    }

    #[test]
    fn dagger_product_lands_in_zsqrt2_and_is_non_negative() {
        let mut rng = Lcg::new(0x4488_1199_aacc_3377);
        for _ in 0..256 {
            let value = sample(&mut rng);
            let product = &value.conjugate() * &value;
            let coordinates = product.coordinates();
            // A Z[sqrt(2)] element has coordinates [a, b, 0, -b]: the
            // imaginary coordinate and the residual odd-coordinate sum vanish.
            assert!(coordinates[2].is_zero());
            assert!((&coordinates[1] + &coordinates[3]).is_zero());

            let norm_squared = value.norm_squared();
            assert_eq!(norm_squared.rational_part(), &coordinates[0]);
            assert_eq!(norm_squared.sqrt2_part(), &coordinates[1]);
            assert!(norm_squared.is_non_negative());
        }
    }

    #[test]
    fn exact_sqrt2_division_matches_independent_multiplication_check() {
        let sqrt2 = ZOmega::sqrt2();
        let mut rng = Lcg::new(0xd171_51b1_e0a2_55aa);
        for _ in 0..256 {
            let value = sample(&mut rng);
            let candidate = independent_division_candidate(&value);
            let independently_divisible = &candidate * &sqrt2 == value;
            assert_eq!(value.is_divisible_by_sqrt2(), independently_divisible);

            if let Some(quotient) = value.div_sqrt2() {
                assert_eq!(&quotient * &sqrt2, value);
            } else {
                assert!(!independently_divisible);
            }

            let product = &value * &sqrt2;
            assert!(product.is_divisible_by_sqrt2());
            assert_eq!(product.div_sqrt2(), Some(value));
        }
    }
}
