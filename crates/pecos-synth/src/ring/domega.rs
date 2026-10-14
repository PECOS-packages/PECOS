//! The dyadic cyclotomic ring `D[omega] = Z[omega][1/sqrt(2)]`.

use std::ops::{Add, Mul, Neg, Sub};

use num_traits::{One, Zero};

use super::ZOmega;

/// An element `u / sqrt(2)^k` of `D[omega]` in least-denominator form.
///
/// The numerator is divisible by `sqrt(2)` only when the denominator exponent
/// is zero. Construction and every arithmetic operation preserve this invariant.
///
/// # Capacity
///
/// The denominator exponent is a `u32`: a fixed width, so results are
/// reproducible across platforms (`usize` is deliberately not used). This is a
/// capacity limit of the representation, in the same sense as a collection's
/// length, and it is far above anything reachable in practice: an exponent `k`
/// is only meaningfully distinct from its neighbours when the numerator carries
/// about `k` bits, so `u32::MAX` corresponds to a numerator of roughly half a
/// gigabyte, while synthesis to `1e-100` needs `k` in the hundreds.
///
/// Multiplication adds exponents, so it can exceed that limit only if a caller
/// constructs values near `u32::MAX` directly. That panics with an explicit
/// message rather than wrapping or silently truncating.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct DOmega {
    numerator: ZOmega,
    denominator_exponent: u32,
}

impl DOmega {
    /// Constructs `numerator / sqrt(2)^denominator_exponent` and reduces it to
    /// least-denominator form.
    #[must_use]
    pub fn new(numerator: ZOmega, denominator_exponent: u32) -> Self {
        Self::normalize(numerator, u64::from(denominator_exponent))
    }

    /// Returns the canonical numerator.
    #[must_use]
    pub const fn numerator(&self) -> &ZOmega {
        &self.numerator
    }

    /// Returns the least denominator exponent.
    #[must_use]
    pub const fn least_denominator_exponent(&self) -> u32 {
        self.denominator_exponent
    }

    /// Applies the lifted bullet automorphism `sigma_5`.
    ///
    /// This maps `omega` to `-omega` and negates `sqrt(2)`.
    #[must_use]
    pub fn sqrt2_conjugate(&self) -> Self {
        // For x = u / sqrt(2)^k, bullet(sqrt(2)) = -sqrt(2), so
        // bullet(x) = bullet(u) / (-sqrt(2))^k
        //           = (-1)^k bullet(u) / sqrt(2)^k.
        let mut numerator = self.numerator.sqrt2_conjugate();
        if !self.denominator_exponent.is_multiple_of(2) {
            numerator = -numerator;
        }
        Self::new(numerator, self.denominator_exponent)
    }

    /// Applies lifted complex conjugation `sigma_7`.
    ///
    /// Complex conjugation maps `omega` to `omega^7` and fixes `sqrt(2)`.
    #[must_use]
    pub fn conjugate(&self) -> Self {
        Self::new(self.numerator.conjugate(), self.denominator_exponent)
    }

    fn normalize(mut numerator: ZOmega, mut denominator_exponent: u64) -> Self {
        if numerator.is_zero() {
            return Self {
                numerator,
                denominator_exponent: 0,
            };
        }

        while denominator_exponent > 0 {
            let Some(quotient) = numerator.div_sqrt2() else {
                break;
            };
            numerator = quotient;
            denominator_exponent -= 1;
        }

        Self {
            numerator,
            denominator_exponent: u32::try_from(denominator_exponent).expect(
                "D[omega] denominator exponent exceeded u32::MAX; see the type's Capacity section",
            ),
        }
    }
}

impl From<ZOmega> for DOmega {
    fn from(value: ZOmega) -> Self {
        Self::new(value, 0)
    }
}

impl Zero for DOmega {
    fn zero() -> Self {
        Self::from(ZOmega::zero())
    }

    fn is_zero(&self) -> bool {
        self.numerator.is_zero()
    }
}

impl One for DOmega {
    fn one() -> Self {
        Self::from(ZOmega::one())
    }
}

impl Add for DOmega {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        &self + &rhs
    }
}

impl Add for &DOmega {
    type Output = DOmega;

    fn add(self, rhs: Self) -> Self::Output {
        let denominator_exponent = self.denominator_exponent.max(rhs.denominator_exponent);
        let lhs_numerator = self
            .numerator
            .mul_sqrt2_pow(denominator_exponent - self.denominator_exponent);
        let rhs_numerator = rhs
            .numerator
            .mul_sqrt2_pow(denominator_exponent - rhs.denominator_exponent);
        DOmega::new(lhs_numerator + rhs_numerator, denominator_exponent)
    }
}

impl Sub for DOmega {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        &self - &rhs
    }
}

impl Sub for &DOmega {
    type Output = DOmega;

    fn sub(self, rhs: Self) -> Self::Output {
        let denominator_exponent = self.denominator_exponent.max(rhs.denominator_exponent);
        let lhs_numerator = self
            .numerator
            .mul_sqrt2_pow(denominator_exponent - self.denominator_exponent);
        let rhs_numerator = rhs
            .numerator
            .mul_sqrt2_pow(denominator_exponent - rhs.denominator_exponent);
        DOmega::new(lhs_numerator - rhs_numerator, denominator_exponent)
    }
}

impl Mul for DOmega {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        &self * &rhs
    }
}

impl Mul for &DOmega {
    type Output = DOmega;

    fn mul(self, rhs: Self) -> Self::Output {
        let numerator = &self.numerator * &rhs.numerator;
        let denominator_exponent =
            u64::from(self.denominator_exponent) + u64::from(rhs.denominator_exponent);
        DOmega::normalize(numerator, denominator_exponent)
    }
}

impl Neg for DOmega {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::new(-self.numerator, self.denominator_exponent)
    }
}

impl Neg for &DOmega {
    type Output = DOmega;

    fn neg(self) -> Self::Output {
        DOmega::new(-&self.numerator, self.denominator_exponent)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    use num_bigint::BigInt;
    use num_traits::{One, Zero};

    use super::DOmega;
    use crate::ring::ZOmega;
    use crate::ring::test_support::{Lcg, assert_commutative_ring};

    fn sample_zomega(rng: &mut Lcg) -> ZOmega {
        ZOmega::new(
            BigInt::from(rng.next_i64(8)),
            BigInt::from(rng.next_i64(8)),
            BigInt::from(rng.next_i64(8)),
            BigInt::from(rng.next_i64(8)),
        )
    }

    fn sample(rng: &mut Lcg) -> DOmega {
        let exponent =
            u32::try_from(rng.next_i64(4).unsigned_abs()).expect("sampled exponent fits in u32");
        DOmega::new(sample_zomega(rng), exponent)
    }

    fn assert_canonical(value: &DOmega) {
        assert!(
            value.least_denominator_exponent() == 0 || !value.numerator().is_divisible_by_sqrt2()
        );
        if value.is_zero() {
            assert_eq!(value.least_denominator_exponent(), 0);
        }
    }

    fn hash(value: &DOmega) -> u64 {
        let mut hasher = DefaultHasher::new();
        value.hash(&mut hasher);
        hasher.finish()
    }

    #[test]
    fn ring_axioms() {
        let mut rng = Lcg::new(0xd0e6_a010_cca7_2026);
        for _ in 0..128 {
            assert_commutative_ring(&sample(&mut rng), &sample(&mut rng), &sample(&mut rng));
        }
    }

    #[test]
    fn construction_and_arithmetic_are_canonical() {
        let mut rng = Lcg::new(0xca10_11ca_1f00_0bad);
        for _ in 0..256 {
            let left = sample(&mut rng);
            let right = sample(&mut rng);

            assert_canonical(&left);
            assert_canonical(&right);
            assert_canonical(&(&left + &right));
            assert_canonical(&(&left - &right));
            assert_canonical(&(&left * &right));
            assert_canonical(&(-&left));
        }

        assert_eq!(DOmega::new(ZOmega::zero(), 19), DOmega::zero());
    }

    #[test]
    fn equivalent_unreduced_inputs_compare_and_hash_equally() {
        let mut rng = Lcg::new(0xe011_a1e5_55aa_17c0);
        let sqrt2 = ZOmega::sqrt2();
        for _ in 0..128 {
            let numerator = sample_zomega(&mut rng);
            let exponent = u32::try_from(rng.next_i64(3).unsigned_abs())
                .expect("sampled exponent fits in u32");
            let canonical_input = DOmega::new(numerator.clone(), exponent);

            let extra = 1_u32
                + u32::try_from(rng.next_i64(3).unsigned_abs())
                    .expect("sampled exponent fits in u32");
            let mut unreduced_numerator = numerator;
            for _ in 0..extra {
                unreduced_numerator = &unreduced_numerator * &sqrt2;
            }
            let unreduced_input = DOmega::new(unreduced_numerator, exponent + extra);

            assert_eq!(canonical_input, unreduced_input);
            assert_eq!(hash(&canonical_input), hash(&unreduced_input));
            assert_canonical(&unreduced_input);
        }
    }

    #[test]
    fn automorphisms_have_required_actions_and_commute() {
        let sqrt2 = DOmega::from(ZOmega::sqrt2());
        let i = DOmega::from(ZOmega::i());
        assert_eq!(sqrt2.sqrt2_conjugate(), -sqrt2.clone());
        assert_eq!(i.sqrt2_conjugate(), i.clone());
        assert_eq!(i.conjugate(), -i.clone());
        assert_eq!(sqrt2.conjugate(), sqrt2);

        let inverse_sqrt2 = DOmega::new(ZOmega::one(), 1);
        assert_eq!(inverse_sqrt2.sqrt2_conjugate(), -inverse_sqrt2.clone());

        let mut rng = Lcg::new(0xa070_40f1_ee77_8b2d);
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
    fn denominator_zero_arithmetic_matches_zomega() {
        let mut rng = Lcg::new(0x200d_0e00_cafe_5151);
        for _ in 0..128 {
            let left = sample_zomega(&mut rng);
            let right = sample_zomega(&mut rng);
            let dleft = DOmega::from(left.clone());
            let dright = DOmega::from(right.clone());

            assert_eq!(&dleft + &dright, DOmega::from(&left + &right));
            assert_eq!(&dleft - &dright, DOmega::from(&left - &right));
            assert_eq!(&dleft * &dright, DOmega::from(&left * &right));
            assert_eq!(-dleft, DOmega::from(-left));
        }
    }
}
