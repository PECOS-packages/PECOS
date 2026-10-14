//! The cyclotomic integer ring `Z[omega]`, where `omega = exp(i pi / 4)`.

use std::ops::{Add, Mul, Neg, Sub};

use num_bigint::BigInt;
use num_traits::{One, Signed, Zero};

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

    fn field_norm(&self) -> BigInt {
        let norm = self.norm_squared().norm();
        debug_assert!(!norm.is_negative());
        norm
    }

    /// Euclidean gcd in `Z[omega]`, with a deterministic unit associate.
    pub(crate) fn gcd(&self, other: &Self) -> Option<Self> {
        let mut left = self.clone();
        let mut right = other.clone();
        let input_bits = left
            .field_norm()
            .bits()
            .saturating_add(right.field_norm().bits())
            .saturating_add(1);
        let mut remaining = input_bits.saturating_mul(32).saturating_add(256);
        while !right.is_zero() {
            if remaining == 0 {
                return None;
            }
            remaining -= 1;
            let quotient = nearest_quotient(&left, &right);
            let remainder = &left - &(&quotient * &right);
            debug_assert!(remainder.field_norm() < right.field_norm());
            left = right;
            right = remainder;
        }
        if left.is_zero() {
            None
        } else {
            canonical_associate(&left)
        }
    }
}

fn nearest_quotient(dividend: &ZOmega, divisor: &ZOmega) -> ZOmega {
    // In Q(omega), y^-1 = y^dagger y^bullet (y^bullet)^dagger / N(y).
    // Thus all four power-basis coordinates of x/y have one positive integer
    // denominator. Choosing floor or ceiling independently gives coordinate
    // errors |e_j| <= 1/2. For e=sum e_j omega^j, write
    // A=sum e_j^2 and B=e0e1-e0e3+e1e2+e2e3. Then
    // N(e)=A^2-2B^2 <= A^2 <= 1. Equality can only occur when all four
    // errors are exact halves; enumerating both choices at every half includes
    // a sign pattern with B != 0, so the minimum is strictly below one.
    let bullet = divisor.sqrt2_conjugate();
    let inverse_numerator = &(&divisor.conjugate() * &bullet) * &bullet.conjugate();
    let quotient_numerator = dividend * &inverse_numerator;
    let denominator = divisor.field_norm();
    debug_assert!(denominator.is_positive());

    let mut floors: [BigInt; 4] = std::array::from_fn(|index| {
        floor_ratio(&quotient_numerator.coordinates[index], &denominator)
    });
    let ceilings: [BigInt; 4] = std::array::from_fn(|index| {
        let numerator = &quotient_numerator.coordinates[index];
        let floor = &floors[index];
        if floor * &denominator == *numerator {
            floor.clone()
        } else {
            floor + 1_u8
        }
    });
    let mut best: Option<(BigInt, [BigInt; 4])> = None;
    for mask in 0_u8..16 {
        let coordinates: [BigInt; 4] = std::array::from_fn(|index| {
            if mask & (1_u8 << u32::try_from(index).expect("coordinate index fits u32")) == 0 {
                floors[index].clone()
            } else {
                ceilings[index].clone()
            }
        });
        let candidate = ZOmega {
            coordinates: coordinates.clone(),
        };
        let remainder = dividend - &(&candidate * divisor);
        let norm = remainder.field_norm();
        let replace = best.as_ref().is_none_or(|(best_norm, best_coordinates)| {
            norm < *best_norm || (norm == *best_norm && coordinates < *best_coordinates)
        });
        if replace {
            best = Some((norm, coordinates));
        }
    }
    let (_, coordinates) = best.expect("finite quotient candidate set is nonempty");
    debug_assert!({
        let candidate = ZOmega {
            coordinates: coordinates.clone(),
        };
        (dividend - &(&candidate * divisor)).field_norm() < divisor.field_norm()
    });
    floors = coordinates;
    ZOmega {
        coordinates: floors,
    }
}

fn floor_ratio(numerator: &BigInt, positive_denominator: &BigInt) -> BigInt {
    let mut quotient = numerator / positive_denominator;
    if numerator.is_negative() && !(numerator % positive_denominator).is_zero() {
        quotient -= 1_u8;
    }
    quotient
}

fn canonical_associate(value: &ZOmega) -> Option<ZOmega> {
    let lambda = ZSqrt2::new(BigInt::one(), BigInt::one());
    let inverse_lambda = ZSqrt2::new(BigInt::from(-1), BigInt::one());
    let lambda_squared = &lambda * &lambda;
    let lambda_fourth = &lambda_squared * &lambda_squared;
    let lambda_eighth = &lambda_fourth * &lambda_fourth;
    let embed = |scalar: &ZSqrt2| {
        ZOmega::new(
            scalar.rational_part().clone(),
            scalar.sqrt2_part().clone(),
            BigInt::zero(),
            -scalar.sqrt2_part(),
        )
    };
    let mut normalized = value.clone();
    // Field norm cannot bound this work: every lambda^k is a unit of field
    // norm one, while normalizing it takes |k| steps. Coordinate magnitude
    // does grow exponentially with |k|, so its maximum bit length gives a
    // linear bound. Exhaustion remains a fallible gcd result, never a panic.
    let coordinate_bits = value
        .coordinates
        .iter()
        .map(BigInt::bits)
        .max()
        .unwrap_or(0);
    let mut remaining = coordinate_bits.saturating_add(8).saturating_mul(8);
    while remaining != 0 {
        remaining -= 1;
        let norm = normalized.norm_squared();
        let norm_square = &norm * &norm;
        let bullet = norm.sqrt2_conjugate();
        let bullet_square = &bullet * &bullet;
        // A Z[omega] unit is omega^j lambda^k; its relative norm is
        // lambda^(2k). Consequently the absolute embedding ratio of the
        // relative norm moves by lambda^4, so its fundamental interval is
        // [1, lambda^4), whose squared upper boundary is lambda^8.
        if norm_square < bullet_square {
            normalized = &normalized * &embed(&lambda);
            continue;
        }
        if norm_square >= &lambda_eighth * &bullet_square {
            normalized = &normalized * &embed(&inverse_lambda);
            continue;
        }
        let omega = ZOmega::omega();
        let mut rotation = ZOmega::one();
        let mut best = normalized.clone();
        for _ in 1_u8..8 {
            rotation = &rotation * &omega;
            let candidate = &normalized * &rotation;
            if candidate.coordinates < best.coordinates {
                best = candidate;
            }
        }
        return Some(best);
    }
    None
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
    use num_traits::{One, Signed, Zero};

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

    fn pow(mut base: ZOmega, mut exponent: u32) -> ZOmega {
        let mut result = ZOmega::one();
        while exponent != 0 {
            if !exponent.is_multiple_of(2) {
                result = &result * &base;
            }
            exponent /= 2;
            if exponent != 0 {
                base = &base * &base;
            }
        }
        result
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

    #[test]
    fn euclidean_gcd_has_a_canonical_unit_associate() {
        let rational_prime = ZOmega::from(41_i64);
        let root_term = &ZOmega::from(9_i64) + &ZOmega::i();
        let expected = rational_prime
            .gcd(&root_term)
            .expect("cyclotomic gcd should converge");
        assert_eq!(
            root_term
                .gcd(&rational_prime)
                .expect("swapped gcd should converge"),
            expected
        );
        assert_eq!(
            (&rational_prime * &ZOmega::omega())
                .gcd(&root_term)
                .expect("unit-associated gcd should converge"),
            expected
        );
        assert_eq!(
            expected.norm_squared().norm().abs(),
            BigInt::from(41_u8).pow(2)
        );
    }

    #[test]
    fn gcd_normalizes_large_and_common_unit_associates() {
        let lambda = ZOmega::new(BigInt::one(), BigInt::one(), BigInt::zero(), -BigInt::one());
        let large_unit = pow(lambda, 160);
        let canonical_unit = ZOmega::one()
            .gcd(&ZOmega::zero())
            .expect("unit gcd should converge");
        assert_eq!(large_unit.gcd(&ZOmega::zero()), Some(canonical_unit));

        let primitive_common = &ZOmega::one() + &ZOmega::i();
        let unbalanced_common = &large_unit * &primitive_common;
        let left = &unbalanced_common * &ZOmega::from(2_i64);
        let right = &unbalanced_common * &ZOmega::from(3_i64);
        assert_eq!(left.gcd(&right), primitive_common.gcd(&ZOmega::zero()));
    }
}
