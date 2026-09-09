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

    /// Divides in `Z[sqrt(2)]`, returning `None` when the quotient is not integral.
    pub(crate) fn div_exact(&self, divisor: &Self) -> Option<Self> {
        if divisor.is_zero() {
            return None;
        }
        let denominator = divisor.norm();
        let rational_numerator = &self.a * &divisor.a - BigInt::from(2_u8) * &self.b * &divisor.b;
        let sqrt2_numerator = &self.b * &divisor.a - &self.a * &divisor.b;
        if (&rational_numerator % &denominator).is_zero()
            && (&sqrt2_numerator % &denominator).is_zero()
        {
            Some(Self::new(
                rational_numerator / &denominator,
                sqrt2_numerator / denominator,
            ))
        } else {
            None
        }
    }

    /// Returns a canonical associate and the signed power of `lambda` used.
    ///
    /// For nonzero `pi`, multiplication by `lambda = 1 + sqrt(2)` scales
    /// `|pi / pi^bullet|` by `lambda^2`. We put that absolute ratio in the
    /// half-open interval `[1, lambda^2)`. Squaring both embeddings makes the
    /// comparisons exact without computing absolute values. The final sign is
    /// selected lexically from the two coordinates.
    pub(crate) fn canonical_associate_with_exponent(&self) -> (Self, i64) {
        assert!(!self.is_zero(), "zero has no canonical associate ratio");
        let lambda = Self::new(BigInt::one(), BigInt::one());
        let inverse_lambda = Self::new(BigInt::from(-1), BigInt::one());
        let lambda_squared = &lambda * &lambda;
        let lambda_fourth = &lambda_squared * &lambda_squared;
        let mut output = self.clone();
        let mut exponent = 0_i64;

        // Each step moves the embedding ratio by the fixed factor lambda^2;
        // its distance from the fundamental domain is linear in the input
        // coordinate bit length. This guard converts a broken comparison or
        // update into an invariant failure instead of a nonterminating test.
        let mut remaining = self
            .a
            .bits()
            .saturating_add(self.b.bits())
            .saturating_add(8)
            .saturating_mul(8);
        loop {
            assert!(
                remaining != 0,
                "Z[sqrt(2)] associate normalization failed to converge"
            );
            remaining -= 1;
            let square = &output * &output;
            let conjugate = output.sqrt2_conjugate();
            let conjugate_square = &conjugate * &conjugate;
            if square < conjugate_square {
                output = &output * &lambda;
                exponent = exponent
                    .checked_add(1)
                    .expect("associate exponent exceeded i64::MAX");
                continue;
            }
            if square >= &lambda_fourth * &conjugate_square {
                output = &output * &inverse_lambda;
                exponent = exponent
                    .checked_sub(1)
                    .expect("associate exponent fell below i64::MIN");
                continue;
            }
            break;
        }

        if output.a.is_negative() || (output.a.is_zero() && output.b.is_negative()) {
            output = -output;
        }
        (output, exponent)
    }

    pub(crate) fn canonical_associate(&self) -> Self {
        self.canonical_associate_with_exponent().0
    }

    /// Euclidean gcd in `Z[sqrt(2)]`, canonicalized up to its unit group.
    pub(crate) fn gcd(&self, other: &Self) -> Option<Self> {
        let mut left = self.clone();
        let mut right = other.clone();
        let input_bits = left
            .norm()
            .bits()
            .saturating_add(right.norm().bits())
            .saturating_add(1);
        let mut remaining = input_bits.saturating_mul(16).saturating_add(128);

        while !right.is_zero() {
            if remaining == 0 {
                return None;
            }
            remaining -= 1;
            let quotient = nearest_quotient(&left, &right);
            let remainder = &left - &(&quotient * &right);
            debug_assert!(remainder.norm().abs() < right.norm().abs());
            left = right;
            right = remainder;
        }
        (!left.is_zero()).then(|| left.canonical_associate())
    }
}

fn nearest_quotient(dividend: &ZSqrt2, divisor: &ZSqrt2) -> ZSqrt2 {
    // In Q(sqrt(2)), multiply by the conjugate of the divisor. Rounding both
    // resulting rational coordinates to nearest integers leaves errors
    // |e_a|, |e_b| <= 1/2, hence
    // |N(e_a + e_b sqrt(2))| = |e_a^2 - 2 e_b^2| <= 1/2 < 1.
    // The resulting remainder therefore has strictly smaller absolute norm.
    let mut denominator = divisor.norm();
    let mut rational_numerator =
        &dividend.a * &divisor.a - BigInt::from(2_u8) * &dividend.b * &divisor.b;
    let mut sqrt2_numerator = &dividend.b * &divisor.a - &dividend.a * &divisor.b;
    if denominator.is_negative() {
        denominator = -denominator;
        rational_numerator = -rational_numerator;
        sqrt2_numerator = -sqrt2_numerator;
    }
    ZSqrt2::new(
        round_ratio(&rational_numerator, &denominator),
        round_ratio(&sqrt2_numerator, &denominator),
    )
}

fn round_ratio(numerator: &BigInt, positive_denominator: &BigInt) -> BigInt {
    let mut floor = numerator / positive_denominator;
    let mut remainder = numerator % positive_denominator;
    if remainder.is_negative() {
        floor -= 1_u8;
        remainder += positive_denominator;
    }
    if &remainder * BigInt::from(2_u8) > *positive_denominator {
        floor + 1_u8
    } else {
        // Exact halves go toward negative infinity, a deterministic choice.
        floor
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
    use num_traits::{One, Signed, Zero};

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

    #[test]
    fn exact_division_and_euclidean_gcd() {
        let left = ZSqrt2::new(BigInt::from(17), BigInt::from(6));
        let right = ZSqrt2::new(BigInt::from(5), BigInt::from(2));
        let product = &left * &right;
        assert_eq!(product.div_exact(&right), Some(left.clone()));
        assert_eq!(
            product.gcd(&left).expect("gcd should converge"),
            left.canonical_associate()
        );
        assert_eq!(left.div_exact(&ZSqrt2::sqrt2()), None);
    }

    #[test]
    fn associate_canonicalization_is_total_idempotent_and_half_open() {
        let lambda = ZSqrt2::new(BigInt::one(), BigInt::one());
        let lambda_fourth = {
            let square = &lambda * &lambda;
            &square * &square
        };
        let values = [
            ZSqrt2::sqrt2(),
            ZSqrt2::new(BigInt::from(7), BigInt::zero()),
            ZSqrt2::new(BigInt::from(3), BigInt::one()),
            ZSqrt2::new(BigInt::from(-9), BigInt::from(4)),
            ZSqrt2::new(BigInt::from(23), BigInt::from(-7)),
            &ZSqrt2::new(BigInt::from(3), BigInt::one())
                * &ZSqrt2::new(BigInt::from(5), BigInt::from(2)),
        ];
        for value in values {
            let canonical = value.canonical_associate();
            assert_eq!(canonical.canonical_associate(), canonical);
            let square = &canonical * &canonical;
            let conjugate = canonical.sqrt2_conjugate();
            let conjugate_square = &conjugate * &conjugate;
            assert!(square >= conjugate_square);
            assert!(square < &lambda_fourth * &conjugate_square);
            assert!(
                canonical.rational_part().is_positive() || {
                    canonical.rational_part().is_zero() && canonical.sqrt2_part().is_positive()
                }
            );
        }

        // The upper boundary is excluded: lambda itself normalizes to one.
        assert_eq!(lambda.canonical_associate(), ZSqrt2::one());
    }

    #[test]
    fn canonicalization_is_multiplicative_up_to_units() {
        let mut rng = Lcg::new(0xca11_0ca1_2026_020b);
        for _ in 0..128 {
            let left = sample(&mut rng);
            let right = sample(&mut rng);
            if left == ZSqrt2::new(BigInt::from(0), BigInt::from(0))
                || right == ZSqrt2::new(BigInt::from(0), BigInt::from(0))
            {
                continue;
            }
            let product = &left * &right;
            let separately = &left.canonical_associate() * &right.canonical_associate();
            assert_eq!(
                product.canonical_associate(),
                separately.canonical_associate()
            );
        }
    }
}
