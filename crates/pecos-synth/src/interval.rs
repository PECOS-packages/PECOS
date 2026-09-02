//! Outward-rounded dyadic interval arithmetic used by grid synthesis.

use std::cmp::Ordering;
use std::sync::OnceLock;

use num_bigint::{BigInt, BigUint};
use num_traits::{One, Signed, Zero};

use crate::SynthError;

/// Number of fractional binary digits stored for the enclosure of pi.
const PI_BITS: u32 = 1024;
const TAYLOR_GUARD_BITS: u32 = 16;

/// The result of comparing every value of one interval with every value of
/// another.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum IntervalOrdering {
    DefinitelyLess,
    DefinitelyGreaterOrEqual,
    Straddles,
}

/// A closed interval whose endpoints are integer multiples of `2^-precision`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct DyadicInterval {
    lo: BigInt,
    hi: BigInt,
    precision: u32,
}

impl DyadicInterval {
    #[must_use]
    pub(crate) fn new(lo: BigInt, hi: BigInt, precision: u32) -> Self {
        assert!(lo <= hi, "dyadic interval endpoints must be ordered");
        Self { lo, hi, precision }
    }

    #[must_use]
    pub(crate) fn exact(numerator: BigInt, precision: u32) -> Self {
        Self::new(numerator.clone(), numerator, precision)
    }

    #[must_use]
    pub(crate) fn integer(value: impl Into<BigInt>, precision: u32) -> Self {
        let scaled = value.into() << checked_shift(precision);
        Self::exact(scaled, precision)
    }

    #[must_use]
    pub(crate) const fn precision(&self) -> u32 {
        self.precision
    }

    #[must_use]
    pub(crate) const fn lo_numerator(&self) -> &BigInt {
        &self.lo
    }

    #[must_use]
    pub(crate) const fn hi_numerator(&self) -> &BigInt {
        &self.hi
    }

    #[must_use]
    pub(crate) fn is_exact(&self) -> bool {
        self.lo == self.hi
    }

    #[must_use]
    pub(crate) fn width_numerator(&self) -> BigInt {
        &self.hi - &self.lo
    }

    /// Changes the fixed-point precision, rounding outward if bits are dropped.
    #[must_use]
    pub(crate) fn at_precision(&self, precision: u32) -> Self {
        match precision.cmp(&self.precision) {
            Ordering::Equal => self.clone(),
            Ordering::Greater => {
                let shift = precision - self.precision;
                Self::new(
                    &self.lo << checked_shift(shift),
                    &self.hi << checked_shift(shift),
                    precision,
                )
            }
            Ordering::Less => {
                let shift = self.precision - precision;
                Self::new(
                    floor_div_pow2(&self.lo, shift),
                    ceil_div_pow2(&self.hi, shift),
                    precision,
                )
            }
        }
    }

    #[must_use]
    pub(crate) fn add(&self, rhs: &Self) -> Self {
        let precision = self.precision.max(rhs.precision);
        let lhs = self.at_precision(precision);
        let rhs = rhs.at_precision(precision);
        Self::new(lhs.lo + rhs.lo, lhs.hi + rhs.hi, precision)
    }

    #[must_use]
    pub(crate) fn sub(&self, rhs: &Self) -> Self {
        self.add(&rhs.neg())
    }

    #[must_use]
    pub(crate) fn neg(&self) -> Self {
        Self::new(-&self.hi, -&self.lo, self.precision)
    }

    #[must_use]
    pub(crate) fn mul(&self, rhs: &Self) -> Self {
        let precision = self.precision.max(rhs.precision);
        self.mul_to(rhs, precision)
    }

    #[must_use]
    pub(crate) fn mul_to(&self, rhs: &Self, precision: u32) -> Self {
        let products = [
            &self.lo * &rhs.lo,
            &self.lo * &rhs.hi,
            &self.hi * &rhs.lo,
            &self.hi * &rhs.hi,
        ];
        let lo = products.iter().min().expect("four products exist");
        let hi = products.iter().max().expect("four products exist");
        let product_precision = checked_precision_add(self.precision, rhs.precision);
        round_bounds(lo, hi, product_precision, precision)
    }

    #[must_use]
    pub(crate) fn square(&self) -> Self {
        let precision = self.precision;
        let product_precision = checked_precision_add(precision, precision);
        let (lo, hi) = if self.lo.is_positive() {
            (&self.lo * &self.lo, &self.hi * &self.hi)
        } else if self.hi.is_negative() {
            (&self.hi * &self.hi, &self.lo * &self.lo)
        } else {
            let hi = (&self.lo * &self.lo).max(&self.hi * &self.hi);
            (BigInt::zero(), hi)
        };
        round_bounds(&lo, &hi, product_precision, precision)
    }

    #[must_use]
    pub(crate) fn div_positive_int(&self, divisor: &BigInt) -> Self {
        assert!(divisor.is_positive(), "interval divisor must be positive");
        Self::new(
            floor_div_positive(&self.lo, divisor),
            ceil_div_positive(&self.hi, divisor),
            self.precision,
        )
    }

    #[must_use]
    pub(crate) fn compare(&self, rhs: &Self) -> IntervalOrdering {
        let precision = self.precision.max(rhs.precision);
        let lhs = self.at_precision(precision);
        let rhs = rhs.at_precision(precision);
        if lhs.hi < rhs.lo {
            IntervalOrdering::DefinitelyLess
        } else if lhs.lo >= rhs.hi {
            IntervalOrdering::DefinitelyGreaterOrEqual
        } else {
            IntervalOrdering::Straddles
        }
    }

    #[must_use]
    pub(crate) fn contains_zero(&self) -> bool {
        self.lo <= BigInt::zero() && self.hi >= BigInt::zero()
    }

    #[must_use]
    pub(crate) fn floor_upper(&self) -> BigInt {
        floor_div_pow2(&self.hi, self.precision)
    }

    #[must_use]
    pub(crate) fn ceil_lower(&self) -> BigInt {
        ceil_div_pow2(&self.lo, self.precision)
    }
}

fn round_bounds(lo: &BigInt, hi: &BigInt, from: u32, to: u32) -> DyadicInterval {
    if to >= from {
        let shift = to - from;
        DyadicInterval::new(lo << checked_shift(shift), hi << checked_shift(shift), to)
    } else {
        let shift = from - to;
        DyadicInterval::new(floor_div_pow2(lo, shift), ceil_div_pow2(hi, shift), to)
    }
}

/// Floor division by a power of two. `BigInt`'s signed right shift is
/// arithmetic, hence it is floor division rather than truncation toward zero.
fn floor_div_pow2(value: &BigInt, exponent: u32) -> BigInt {
    value >> checked_shift(exponent)
}

/// Ceiling division by a power of two, expressed through floor division so
/// negative operands take the correct outward direction.
fn ceil_div_pow2(value: &BigInt, exponent: u32) -> BigInt {
    -floor_div_pow2(&(-value), exponent)
}

/// Signed floor division by a positive integer. Division is performed only on
/// a non-negative magnitude; the sign and non-zero remainder are then handled
/// explicitly, avoiding `BigInt`'s truncation-toward-zero trap.
fn floor_div_positive(value: &BigInt, divisor: &BigInt) -> BigInt {
    let magnitude = value.abs();
    let quotient = &magnitude / divisor;
    if value.is_negative() && &quotient * divisor != magnitude {
        -quotient - BigInt::one()
    } else if value.is_negative() {
        -quotient
    } else {
        quotient
    }
}

fn ceil_div_positive(value: &BigInt, divisor: &BigInt) -> BigInt {
    -floor_div_positive(&(-value), divisor)
}

fn checked_precision_add(left: u32, right: u32) -> u32 {
    let sum = u64::from(left) + u64::from(right);
    u32::try_from(sum).expect("dyadic precision exceeds u32::MAX")
}

fn checked_shift(shift: u32) -> usize {
    usize::try_from(u64::from(shift)).expect("shift count does not fit usize")
}

/// Retries an interval decision at geometrically increasing precision.
pub(crate) fn retry_with_precision<T, E, F>(
    base_precision: u32,
    max_precision: u32,
    mut operation: F,
) -> Result<T, E>
where
    E: From<SynthError>,
    F: FnMut(u32) -> Result<Option<T>, E>,
{
    let mut precision = base_precision.max(1);
    if precision > max_precision {
        return Err(SynthError::Inconclusive {
            precision: max_precision,
        }
        .into());
    }
    loop {
        if let Some(value) = operation(precision)? {
            return Ok(value);
        }
        if precision == max_precision {
            return Err(SynthError::Inconclusive { precision }.into());
        }
        let doubled = u64::from(precision)
            .checked_mul(2)
            .expect("u32 precision doubled in u64 cannot overflow");
        precision = u32::try_from(doubled.min(u64::from(max_precision)))
            .expect("precision was capped at u32 max");
    }
}

/// Converts an exact `Angle64::fraction()` value to the signed half-angle in turns.
/// The half-turn bit represents `+pi`, while larger fractions wrap negative.
#[must_use]
pub(crate) fn signed_half_angle_turns(angle_fraction: u64) -> (BigInt, u32) {
    let half_turn = 1_u64 << 63;
    let numerator = if angle_fraction <= half_turn {
        BigInt::from(angle_fraction)
    } else {
        BigInt::from(angle_fraction) - (BigInt::one() << 64_usize)
    };
    (numerator, 65)
}

/// Encloses `(cos(theta/2), sin(theta/2))` from an `Angle64` fraction.
pub(crate) fn half_angle_sin_cos(
    angle_fraction: u64,
    precision: u32,
) -> Result<(DyadicInterval, DyadicInterval), SynthError> {
    let (numerator, exponent) = signed_half_angle_turns(angle_fraction);
    sin_cos_turns(&numerator, exponent, precision)
}

/// Encloses `z = exp(-i theta/2)` from an `Angle64` fraction.
pub(crate) fn target_phase(
    angle_fraction: u64,
    precision: u32,
) -> Result<(DyadicInterval, DyadicInterval), SynthError> {
    let (cosine, sine) = half_angle_sin_cos(angle_fraction, precision)?;
    Ok((cosine, sine.neg()))
}

/// Encloses sine and cosine of an exact dyadic number of full turns.
pub(crate) fn sin_cos_turns(
    numerator: &BigInt,
    exponent: u32,
    precision: u32,
) -> Result<(DyadicInterval, DyadicInterval), SynthError> {
    let work_precision = u32::try_from(u64::from(precision) + u64::from(TAYLOR_GUARD_BITS))
        .map_err(|_| SynthError::Inconclusive { precision })?;
    if work_precision > PI_BITS {
        return Err(SynthError::Inconclusive { precision });
    }
    if exponent < 3 {
        let lifted = numerator << checked_shift(3 - exponent);
        return sin_cos_turns(&lifted, 3, precision);
    }

    let modulus = BigInt::one() << checked_shift(exponent);
    let quotient = floor_div_pow2(numerator, exponent);
    let fraction = numerator - quotient * &modulus;
    let octant_bits = exponent - 3;
    let octant_big = &fraction >> checked_shift(octant_bits);
    let octant = u8::try_from(octant_big).expect("the top three bits fit in u8");
    let octant_width = BigInt::one() << checked_shift(octant_bits);
    let remainder = fraction - (BigInt::from(octant) << checked_shift(octant_bits));
    let reduced = if octant.is_multiple_of(2) {
        remainder
    } else {
        &octant_width - remainder
    };

    // The first 256 hexadecimal fractional digits (1024 bits) are the
    // Bailey--Borwein--Plouffe hexadecimal expansion of pi. The stored lower
    // endpoint is floor(pi * 2^1024), and the adjacent integer is the upper
    // endpoint. Digits were generated with the BBP formula and checked against
    // the published prefix `3.243F6A8...` documented by Fabrice Bellard,
    // "Binary digits of PI" (1997).
    let pi = pi_interval(work_precision);
    let turns = DyadicInterval::exact(reduced, exponent).at_precision(work_precision);
    let radians = turns
        .mul_to(&pi, work_precision)
        .mul(&DyadicInterval::integer(2_u8, work_precision));
    let (sin_reduced, cos_reduced) = taylor_sin_cos(&radians, work_precision);

    let (sin, cos) = match octant {
        0 => (sin_reduced, cos_reduced),
        1 => (cos_reduced, sin_reduced),
        2 => (cos_reduced, sin_reduced.neg()),
        3 => (sin_reduced, cos_reduced.neg()),
        4 => (sin_reduced.neg(), cos_reduced.neg()),
        5 => (cos_reduced.neg(), sin_reduced.neg()),
        6 => (cos_reduced.neg(), sin_reduced),
        7 => (sin_reduced.neg(), cos_reduced),
        _ => unreachable!("octant is represented by three bits"),
    };
    Ok((cos.at_precision(precision), sin.at_precision(precision)))
}

fn taylor_sin_cos(x: &DyadicInterval, precision: u32) -> (DyadicInterval, DyadicInterval) {
    let one = DyadicInterval::integer(1_u8, precision);
    let mut sine = DyadicInterval::integer(0_u8, precision);
    let mut cosine = one.clone();
    let mut power = one;
    let mut factorial = BigInt::one();
    let threshold = BigInt::one();
    let mut degree = 0_u32;
    let mut sin_done = false;
    let mut cos_done = false;

    while !(sin_done && cos_done) {
        degree = degree.checked_add(1).expect("Taylor degree fits u32");
        power = power.mul_to(x, precision);
        factorial *= BigInt::from(degree);
        let term = power.div_positive_int(&factorial);

        if degree.is_multiple_of(2) {
            if !cos_done {
                let positive = degree.is_multiple_of(4);
                cosine = if positive {
                    cosine.add(&term)
                } else {
                    cosine.sub(&term)
                };
                let next_power = power.mul_to(x, precision);
                let next_factorial = &factorial * BigInt::from(degree + 1);
                let remainder = next_power.div_positive_int(&next_factorial);
                if remainder.hi <= threshold {
                    cosine = widen(&cosine, &remainder);
                    cos_done = true;
                }
            }
        } else if !sin_done {
            let positive = degree & 3 == 1;
            sine = if positive {
                sine.add(&term)
            } else {
                sine.sub(&term)
            };
            let next_power = power.mul_to(x, precision);
            let next_factorial = &factorial * BigInt::from(degree + 1);
            let remainder = next_power.div_positive_int(&next_factorial);
            if remainder.hi <= threshold {
                sine = widen(&sine, &remainder);
                sin_done = true;
            }
        }
    }
    (sine, cosine)
}

/// Taylor's theorem with the Lagrange remainder gives
/// `|R_N(x)| <= |x|^(N+1)/(N+1)!`, because every derivative of sine
/// and cosine has absolute value at most one. `remainder` encloses that
/// non-negative bound, so widening both ways is outward-safe.
fn widen(value: &DyadicInterval, remainder: &DyadicInterval) -> DyadicInterval {
    DyadicInterval::new(
        &value.lo - &remainder.hi,
        &value.hi + &remainder.hi,
        value.precision,
    )
}

/// Outward enclosure of a non-negative interval's square root.
pub(crate) fn sqrt_interval(value: &DyadicInterval) -> DyadicInterval {
    assert!(
        !value.lo.is_negative(),
        "square root requires a non-negative interval"
    );
    let precision = value.precision;
    let radicand_shift = precision;
    let lo_scaled = BigUint::try_from(&value.lo << checked_shift(radicand_shift))
        .expect("non-negative lower radicand");
    let hi_scaled = BigUint::try_from(&value.hi << checked_shift(radicand_shift))
        .expect("non-negative upper radicand");
    let lo = integer_sqrt(&lo_scaled);
    let hi_floor = integer_sqrt(&hi_scaled);
    let hi = if &hi_floor * &hi_floor == hi_scaled {
        hi_floor
    } else {
        hi_floor + BigUint::one()
    };
    DyadicInterval::new(BigInt::from(lo), BigInt::from(hi), precision)
}

fn integer_sqrt(value: &BigUint) -> BigUint {
    if value.is_zero() {
        return BigUint::zero();
    }
    let bits = value.bits();
    let shift = usize::try_from(bits.div_ceil(2)).expect("BigUint bit count fits usize");
    let mut current = BigUint::one() << shift;
    loop {
        let next = (&current + value / &current) >> 1_usize;
        if next >= current {
            return current;
        }
        current = next;
    }
}

pub(crate) fn pi_interval(precision: u32) -> DyadicInterval {
    static PI_FLOOR: OnceLock<BigInt> = OnceLock::new();
    let floor = PI_FLOOR.get_or_init(|| {
        let digits = concat!(
            "3243F6A8885A308D313198A2E03707344A4093822299F31D0082EFA98EC4E6C89",
            "452821E638D01377BE5466CF34E90C6CC0AC29B7C97C50DD3F84D5B5B5470917",
            "9216D5D98979FB1BD1310BA698DFB5AC2FFD72DBD01ADFB7B8E1AFED6A267E96",
            "BA7C9045F12C7F9924A19947B3916CF70801F2E2858EFC16636920D871574E69"
        );
        let parsed = BigInt::parse_bytes(digits.as_bytes(), 16).expect("stored pi digits are hex");
        let supplied_fractional_bits = 1024_u32;
        if supplied_fractional_bits == PI_BITS {
            parsed
        } else {
            unreachable!("the stored pi constant and PI_BITS must agree")
        }
    });
    let full = DyadicInterval::new(floor.clone(), floor + BigInt::one(), PI_BITS);
    full.at_precision(precision)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> i64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            i64::try_from((self.0 >> 32) & 0x1ff).expect("nine bits fit in i64") - 256
        }
    }

    fn contains_rational(interval: &DyadicInterval, numerator: &BigInt, exponent: u32) -> bool {
        let common = interval.precision.max(exponent);
        let value = numerator << checked_shift(common - exponent);
        let lo = interval.lo_numerator() << checked_shift(common - interval.precision);
        let hi = interval.hi_numerator() << checked_shift(common - interval.precision);
        lo <= value && value <= hi
    }

    fn contains_interval(outer: &DyadicInterval, inner: &DyadicInterval) -> bool {
        let precision = outer.precision.max(inner.precision);
        let outer = outer.at_precision(precision);
        let inner = inner.at_precision(precision);
        outer.lo <= inner.lo && inner.hi <= outer.hi
    }

    #[test]
    fn signed_rounding_is_outward_for_negative_operands() {
        assert_eq!(floor_div_pow2(&BigInt::from(-3), 1), BigInt::from(-2));
        assert_eq!(ceil_div_pow2(&BigInt::from(-3), 1), BigInt::from(-1));
        assert_eq!(
            floor_div_positive(&BigInt::from(-7), &BigInt::from(3)),
            BigInt::from(-3)
        );
        assert_eq!(
            ceil_div_positive(&BigInt::from(-7), &BigInt::from(3)),
            BigInt::from(-2)
        );
    }

    #[test]
    fn angle_principal_value_and_half_angle_are_exact() {
        assert_eq!(
            signed_half_angle_turns(1_u64 << 63),
            (BigInt::from(1_u64 << 63), 65)
        );
        assert_eq!(signed_half_angle_turns(u64::MAX), (BigInt::from(-1), 65));
    }

    #[test]
    fn sampled_arithmetic_contains_exact_rational_results() {
        let mut rng = Lcg(0x5eed_1a7e_f17e_2026);
        for _ in 0..256 {
            let left_numerator = BigInt::from(rng.next());
            let right_numerator = BigInt::from(rng.next());
            let left_precision = u32::try_from(rng.next().unsigned_abs() & 15).unwrap();
            let right_precision = u32::try_from(rng.next().unsigned_abs() & 15).unwrap();
            let left = DyadicInterval::exact(left_numerator.clone(), left_precision);
            let right = DyadicInterval::exact(right_numerator.clone(), right_precision);

            let common = left_precision.max(right_precision);
            let left_common = &left_numerator << checked_shift(common - left_precision);
            let right_common = &right_numerator << checked_shift(common - right_precision);
            assert!(contains_rational(
                &left.add(&right),
                &(&left_common + &right_common),
                common
            ));
            assert!(contains_rational(
                &left.sub(&right),
                &(&left_common - &right_common),
                common
            ));
            assert!(contains_rational(
                &left.mul(&right),
                &(&left_numerator * &right_numerator),
                checked_precision_add(left_precision, right_precision)
            ));
            assert!(contains_rational(
                &left.square(),
                &(&left_numerator * &left_numerator),
                checked_precision_add(left_precision, left_precision)
            ));
            assert!(contains_rational(
                &left.neg(),
                &(-&left_numerator),
                left_precision
            ));
        }
    }

    #[test]
    fn pinned_trigonometric_values_and_octant_boundaries() {
        for precision in [32_u32, 64, 128] {
            let (cosine, sine) = sin_cos_turns(&BigInt::zero(), 64, precision).unwrap();
            let one = DyadicInterval::integer(1_u8, precision);
            let zero = DyadicInterval::integer(0_u8, precision);
            assert_eq!(cosine, one);
            assert_eq!(sine, zero);

            let sqrt2_over_two = sqrt_interval(&DyadicInterval::integer(2_u8, precision))
                .div_positive_int(&BigInt::from(2_u8));
            let negative_one = one.neg();
            let negative_sqrt2_over_two = sqrt2_over_two.neg();
            let expected = [
                (one.clone(), zero.clone()),
                (sqrt2_over_two.clone(), sqrt2_over_two.clone()),
                (zero.clone(), one.clone()),
                (negative_sqrt2_over_two.clone(), sqrt2_over_two.clone()),
                (negative_one.clone(), zero.clone()),
                (
                    negative_sqrt2_over_two.clone(),
                    negative_sqrt2_over_two.clone(),
                ),
                (zero.clone(), negative_one),
                (sqrt2_over_two.clone(), negative_sqrt2_over_two),
            ];

            for octant in 0_u8..8 {
                let numerator = BigInt::from(octant) << 61_usize;
                let (c, s) = sin_cos_turns(&numerator, 64, precision).unwrap();
                let (expected_cosine, expected_sine) = &expected[usize::from(octant)];
                assert!(contains_interval(&c, expected_cosine));
                assert!(contains_interval(&s, expected_sine));
            }
        }

        let precision = 192;
        let (cosine, sine) = sin_cos_turns(&BigInt::one(), 3, precision).unwrap();
        let sqrt2_over_two = sqrt_interval(&DyadicInterval::integer(2_u8, precision))
            .div_positive_int(&BigInt::from(2_u8));
        assert!(cosine.lo <= sqrt2_over_two.lo && sqrt2_over_two.hi <= cosine.hi);
        assert!(sine.lo <= sqrt2_over_two.lo && sqrt2_over_two.hi <= sine.hi);
    }

    #[test]
    fn trig_width_decreases_with_precision() {
        let angle = BigInt::from(1_u8);
        let mut previous: Option<(BigInt, u32)> = None;
        for precision in [24_u32, 48, 96, 192] {
            let (cosine, _) = sin_cos_turns(&angle, 3, precision).unwrap();
            if let Some((width, old_precision)) = previous {
                let aligned = width << checked_shift(precision - old_precision);
                assert!(cosine.width_numerator() <= aligned);
            }
            previous = Some((cosine.width_numerator(), precision));
        }
    }

    #[test]
    fn retry_resolves_and_reports_the_ceiling() {
        let resolved: Result<u32, SynthError> = retry_with_precision(8, 32, |precision| {
            let third =
                DyadicInterval::integer(1_u8, precision).div_positive_int(&BigInt::from(3_u8));
            let threshold = DyadicInterval::exact(BigInt::from(86_u8), 8).at_precision(precision);
            Ok(
                (third.compare(&threshold) == IntervalOrdering::DefinitelyLess)
                    .then_some(precision),
            )
        });
        assert_eq!(resolved, Ok(16));
        let failed = retry_with_precision::<(), SynthError, _>(8, 32, |_| Ok(None));
        assert_eq!(failed, Err(SynthError::Inconclusive { precision: 32 }));
    }
}
