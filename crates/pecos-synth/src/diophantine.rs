//! Construction of solutions to `t^dagger t = xi` in `D[omega]`.

use std::cmp::Ordering;

use num_bigint::{BigInt, BigUint, Sign};
use num_traits::{One, Signed, Zero};

use crate::factor::{
    FactorError, FactorParams, PrimeEvidence, factor_integer, modular_square_root, residue_mod_u8,
};
use crate::{DOmega, ZOmega, ZSqrt2};

/// The grid layer represents exact dyadic values uniformly as `D[omega]`.
/// Norm-equation inputs are the real `D[sqrt(2)]` subset of that carrier.
pub(crate) type DyadicZSqrt2Value = DOmega;

/// Practical ceiling on norm-equation denominator scaling. As with the grid
/// tolerance cap, accepting arbitrary `u32` exponents would let a compact
/// input request enormous exact powers before a budgeted operation is reached.
const MAX_SCALE_EXPONENT: u32 = 4096;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum NormResolution {
    Solved { t: DOmega },
    Unsolvable,
    Exhausted,
}

#[derive(Clone)]
struct RelativePrimePower {
    value: ZSqrt2,
    exponent: u32,
    rational_prime: BigUint,
    rational_evidence: PrimeEvidence,
}

pub(crate) fn solve_norm_equation(xi: &DyadicZSqrt2Value, params: &FactorParams) -> NormResolution {
    if params.validate().is_err() {
        return NormResolution::Exhausted;
    }
    if xi.is_zero() {
        return verified_resolution(DOmega::zero(), xi);
    }
    let scale_exponent = xi.least_denominator_exponent();
    if scale_exponent > MAX_SCALE_EXPONENT {
        return NormResolution::Exhausted;
    }
    let Some(scaled) = real_numerator(xi) else {
        return NormResolution::Exhausted;
    };

    // Ross & Selinger, arXiv:1403.2975v3, Eq. (9), Definition C.1,
    // Lemma 6.1, and Lemma C.16: a nonzero dagger norm is STRICTLY positive
    // in both real embeddings. This is proof independent of factorization and
    // must be tested before any budgeted work.
    if !is_strictly_doubly_positive(&scaled, scale_exponent) {
        return NormResolution::Unsolvable;
    }

    // Ross & Selinger, arXiv:1403.2975v3, Theorem 6.2 and Lemma C.25:
    // for xi = scaled/sqrt(2)^ell in least-denominator form, first solve
    // s^dagger s ~ scaled and later set t = delta^-ell s up to a unit.
    // Requiring equality at this scaled stage is wrong for odd ell.
    let rational_norm = scaled.norm().abs();
    let Ok(rational_norm) = BigUint::try_from(rational_norm) else {
        return NormResolution::Exhausted;
    };
    let factors = match factor_integer(&rational_norm, params) {
        Ok(factors) => factors,
        Err(FactorError::Exhausted) => return NormResolution::Exhausted,
    };
    let prime_powers = match relative_factorization(&scaled, &factors, params) {
        Ok(powers) => powers,
        Err(FactorError::Exhausted) => return NormResolution::Exhausted,
    };

    // Ross & Selinger, arXiv:1403.2975v3, Lemmas C.19 and C.23: solve
    // pairwise-coprime prime powers independently, then multiply their roots.
    let mut associate_root = ZOmega::one();
    for power in &prime_powers {
        if !power.exponent.is_multiple_of(2) && residue_mod_u8(&power.rational_prime) == 7 {
            // Ross & Selinger, arXiv:1403.2975v3, Lemmas C.19, C.21,
            // and C.23: this is the sole prime-power obstruction. It is a
            // proof only when the underlying rational primality is proved.
            return if power.rational_evidence == PrimeEvidence::Proven {
                NormResolution::Unsolvable
            } else {
                NormResolution::Exhausted
            };
        }

        let square_part = zsqrt_pow(&power.value, power.exponent / 2);
        associate_root = &associate_root * &embed_zsqrt(&square_part);
        if !power.exponent.is_multiple_of(2) {
            let Some(prime_root) = relative_prime_root(power, params.nonresidue_trials) else {
                return NormResolution::Exhausted;
            };
            associate_root = &associate_root * &prime_root;
        }
    }

    let associate_norm = associate_root.norm_squared();
    let Some(unit_ratio) = scaled.div_exact(&associate_norm) else {
        return NormResolution::Exhausted;
    };
    // N(delta) = lambda sqrt(2), hence
    // N(delta^-ell s) = lambda^-ell sqrt(2)^-ell N(s). The exact unit still
    // needed after unscaling is therefore lambda^ell scaled/N(s).
    let lambda_scale = lambda_pow(i64::from(scale_exponent));
    let correction_unit = &lambda_scale * &unit_ratio;

    // Ross & Selinger, arXiv:1403.2975v3, Lemma C.2 and the constructive
    // proof of Lemma C.16: the leftover unit is doubly positive, hence is an
    // exact square. Extract its even lambda exponent and multiply the root.
    let Some(unit_root) = doubly_positive_unit_root(&correction_unit) else {
        return NormResolution::Exhausted;
    };
    associate_root = &associate_root * &embed_zsqrt(&unit_root);

    let Some(delta_inverse) = delta_inverse() else {
        return NormResolution::Exhausted;
    };
    let unscaling = domega_pow(&delta_inverse, scale_exponent);
    let solution = &DOmega::from(associate_root) * &unscaling;
    verified_resolution(solution, xi)
}

fn is_strictly_doubly_positive(value: &ZSqrt2, denominator_exponent: u32) -> bool {
    if value <= &ZSqrt2::zero() {
        return false;
    }
    let bullet = value.sqrt2_conjugate();
    if denominator_exponent.is_multiple_of(2) {
        bullet > ZSqrt2::zero()
    } else {
        bullet < ZSqrt2::zero()
    }
}

fn relative_factorization(
    scaled: &ZSqrt2,
    factors: &[crate::factor::PrimeFactor],
    params: &FactorParams,
) -> Result<Vec<RelativePrimePower>, FactorError> {
    let mut remaining = scaled.clone();
    let mut output = Vec::new();
    for factor in factors {
        let residue = residue_mod_u8(&factor.prime);
        if factor.prime == BigUint::from(2_u8) {
            let prime = ZSqrt2::sqrt2().canonical_associate();
            let exponent = consume_valuation(&mut remaining, &prime, factor.exponent);
            if exponent != factor.exponent {
                return Err(FactorError::Exhausted);
            }
            output.push(RelativePrimePower {
                value: prime,
                exponent,
                rational_prime: factor.prime.clone(),
                rational_evidence: factor.evidence,
            });
        } else if residue == 1 || residue == 7 {
            // Ross & Selinger, arXiv:1403.2975v3, Lemmas C.11-C.12:
            // for p = 1 or 7 (mod 8), choose x^2 = 2 (mod p) and compute
            // gcd(p, x + sqrt(2)); its bullet conjugate is the other prime.
            let sqrt_two = modular_square_root(
                &BigUint::from(2_u8),
                &factor.prime,
                params.nonresidue_trials,
            )?;
            let rational = BigInt::from_biguint(Sign::Plus, factor.prime.clone());
            let root = BigInt::from_biguint(Sign::Plus, sqrt_two);
            let split = ZSqrt2::new(rational.clone(), BigInt::zero())
                .gcd(&ZSqrt2::new(root, BigInt::one()))
                .ok_or(FactorError::Exhausted)?
                .canonical_associate();
            if split.norm().abs() != rational {
                return Err(FactorError::Exhausted);
            }
            let conjugate = split.sqrt2_conjugate().canonical_associate();
            if split == conjugate {
                return Err(FactorError::Exhausted);
            }
            let mut pair = [split, conjugate];
            pair.sort_by(compare_zsqrt_coordinates);
            let mut total_exponent = 0_u32;
            for prime in pair {
                let exponent = consume_valuation(&mut remaining, &prime, factor.exponent);
                total_exponent = total_exponent
                    .checked_add(exponent)
                    .expect("relative prime exponent exceeded u32::MAX");
                if exponent != 0 {
                    output.push(RelativePrimePower {
                        value: prime,
                        exponent,
                        rational_prime: factor.prime.clone(),
                        rational_evidence: factor.evidence,
                    });
                }
            }
            if total_exponent != factor.exponent {
                return Err(FactorError::Exhausted);
            }
        } else if residue == 3 || residue == 5 {
            // Ross & Selinger, arXiv:1403.2975v3, Lemma C.9: rational
            // primes in these two residue classes stay prime in Z[sqrt(2)].
            if !factor.exponent.is_multiple_of(2) {
                return Err(FactorError::Exhausted);
            }
            let rational = BigInt::from_biguint(Sign::Plus, factor.prime.clone());
            let prime = ZSqrt2::new(rational, BigInt::zero());
            let expected = factor.exponent / 2;
            let exponent = consume_valuation(&mut remaining, &prime, expected);
            if exponent != expected {
                return Err(FactorError::Exhausted);
            }
            output.push(RelativePrimePower {
                value: prime,
                exponent,
                rational_prime: factor.prime.clone(),
                rational_evidence: factor.evidence,
            });
        } else {
            return Err(FactorError::Exhausted);
        }
    }
    if remaining.norm().abs() != BigInt::one() {
        return Err(FactorError::Exhausted);
    }
    Ok(output)
}

fn consume_valuation(value: &mut ZSqrt2, prime: &ZSqrt2, maximum: u32) -> u32 {
    let mut exponent = 0_u32;
    while exponent < maximum {
        let Some(quotient) = value.div_exact(prime) else {
            break;
        };
        *value = quotient;
        exponent += 1;
    }
    exponent
}

fn relative_prime_root(power: &RelativePrimePower, nonresidue_trials: u32) -> Option<ZOmega> {
    if power.rational_prime == BigUint::from(2_u8) {
        // Ross & Selinger, arXiv:1403.2975v3, Lemma C.20:
        // delta = 1 + omega has delta^dagger delta = lambda sqrt(2).
        return Some(&ZOmega::one() + &ZOmega::omega());
    }
    let residue = residue_mod_u8(&power.rational_prime);
    let term = if residue == 1 || residue == 5 {
        // Ross & Selinger, arXiv:1403.2975v3, Lemma C.20 and Remark C.22:
        // u^2 = -1 mod p makes the relative root gcd(pi, u + i).
        let radicand = &power.rational_prime - BigUint::one();
        let root = modular_square_root(&radicand, &power.rational_prime, nonresidue_trials).ok()?;
        &ZOmega::from(BigInt::from_biguint(Sign::Plus, root)) + &ZOmega::i()
    } else if residue == 3 {
        // Ross & Selinger, arXiv:1403.2975v3, Lemma C.20 and Remark C.22:
        // u^2 = -2 mod p makes the relative root gcd(pi, u + i sqrt(2)).
        let radicand = &power.rational_prime - BigUint::from(2_u8);
        let root = modular_square_root(&radicand, &power.rational_prime, nonresidue_trials).ok()?;
        let imaginary_sqrt_two = &ZOmega::i() * &ZOmega::sqrt2();
        &ZOmega::from(BigInt::from_biguint(Sign::Plus, root)) + &imaginary_sqrt_two
    } else {
        return None;
    };
    let gcd = embed_zsqrt(&power.value).gcd(&term)?;
    let quotient = power.value.div_exact(&gcd.norm_squared())?;
    (quotient.norm().abs() == BigInt::one()).then_some(gcd)
}

fn doubly_positive_unit_root(unit: &ZSqrt2) -> Option<ZSqrt2> {
    if unit.norm() != BigInt::one()
        || unit <= &ZSqrt2::zero()
        || unit.sqrt2_conjugate() <= ZSqrt2::zero()
    {
        return None;
    }
    let (canonical, exponent) = unit.canonical_associate_with_exponent();
    if canonical != ZSqrt2::one() || exponent % 2 != 0 {
        return None;
    }
    // Canonicalization records `unit * lambda^exponent = 1`; Lemma C.2 says
    // double positivity makes this exponent even. Thus lambda^(-exponent/2)
    // is the constructive square root used in the proof of Lemma C.16.
    let root_exponent = exponent.checked_neg()?.checked_div(2)?;
    let root = lambda_pow(root_exponent);
    (&root * &root == *unit).then_some(root)
}

fn delta_inverse() -> Option<DOmega> {
    // Ross & Selinger, arXiv:1403.2975v3, proof of Lemma C.25 gives
    // delta^-1 = delta lambda^-1 omega^-1 / sqrt(2), with
    // lambda^-1 = -1 + sqrt(2) and omega^-1 = -omega^3.
    let delta = &ZOmega::one() + &ZOmega::omega();
    let lambda_inverse = embed_zsqrt(&ZSqrt2::new(BigInt::from(-1), BigInt::one()));
    let omega_inverse = ZOmega::new(
        BigInt::zero(),
        BigInt::zero(),
        BigInt::zero(),
        BigInt::from(-1),
    );
    let inverse = DOmega::new(&(&delta * &lambda_inverse) * &omega_inverse, 1);
    (&DOmega::from(delta) * &inverse == DOmega::one()).then_some(inverse)
}

fn verified_resolution(solution: DOmega, xi: &DyadicZSqrt2Value) -> NormResolution {
    // Resolution semantics require this exact equality unconditionally: even
    // probable factors may produce a solution, but may never bypass this gate.
    let actual = &solution.conjugate() * &solution;
    if actual != *xi {
        return NormResolution::Exhausted;
    }
    NormResolution::Solved { t: solution }
}

fn real_numerator(value: &DOmega) -> Option<ZSqrt2> {
    let [rational, sqrt2, imaginary, negative_sqrt2] = value.numerator().coordinates();
    if imaginary.is_zero() && negative_sqrt2 == &-sqrt2 {
        Some(ZSqrt2::new(rational.clone(), sqrt2.clone()))
    } else {
        None
    }
}

fn embed_zsqrt(value: &ZSqrt2) -> ZOmega {
    ZOmega::new(
        value.rational_part().clone(),
        value.sqrt2_part().clone(),
        BigInt::zero(),
        -value.sqrt2_part(),
    )
}

fn zsqrt_pow(value: &ZSqrt2, exponent: u32) -> ZSqrt2 {
    let mut result = ZSqrt2::one();
    let mut base = value.clone();
    let mut remaining = exponent;
    while remaining != 0 {
        if !remaining.is_multiple_of(2) {
            result = &result * &base;
        }
        remaining /= 2;
        if remaining != 0 {
            base = &base * &base;
        }
    }
    result
}

fn lambda_pow(exponent: i64) -> ZSqrt2 {
    let mut base = if exponent >= 0 {
        ZSqrt2::new(BigInt::one(), BigInt::one())
    } else {
        ZSqrt2::new(BigInt::from(-1), BigInt::one())
    };
    let mut result = ZSqrt2::one();
    let mut remaining = exponent.unsigned_abs();
    while remaining != 0 {
        if !remaining.is_multiple_of(2) {
            result = &result * &base;
        }
        remaining /= 2;
        if remaining != 0 {
            base = &base * &base;
        }
    }
    result
}

fn domega_pow(value: &DOmega, exponent: u32) -> DOmega {
    let mut result = DOmega::one();
    let mut base = value.clone();
    let mut remaining = exponent;
    while remaining != 0 {
        if !remaining.is_multiple_of(2) {
            result = &result * &base;
        }
        remaining /= 2;
        if remaining != 0 {
            base = &base * &base;
        }
    }
    result
}

fn compare_zsqrt_coordinates(left: &ZSqrt2, right: &ZSqrt2) -> Ordering {
    left.rational_part()
        .cmp(right.rational_part())
        .then_with(|| left.sqrt2_part().cmp(right.sqrt2_part()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> FactorParams {
        FactorParams {
            seed: 0xd10f_a7a0_5eed_020b,
            factor_attempts: 32,
            rho_steps: 100_000,
            nonresidue_trials: 64,
            primality_rounds: 24,
        }
    }

    fn dyadic_zsqrt(value: &ZSqrt2, denominator_exponent: u32) -> DOmega {
        DOmega::new(embed_zsqrt(value), denominator_exponent)
    }

    fn assert_solved_exactly(xi: &DOmega, resolution: &NormResolution) {
        let NormResolution::Solved { t } = resolution else {
            panic!("expected a solution for {xi:?}, got {resolution:?}");
        };
        assert_eq!(&t.conjugate() * t, *xi);
    }

    #[test]
    fn verified_resolution_rejects_mismatch_and_accepts_exact_root() {
        let solution = DOmega::from(&ZOmega::one() + &ZOmega::i());
        let exact_xi = &solution.conjugate() * &solution;
        let mismatched_xi = &exact_xi + &DOmega::one();

        assert_eq!(
            verified_resolution(solution.clone(), &mismatched_xi),
            NormResolution::Exhausted
        );
        assert_eq!(
            verified_resolution(solution.clone(), &exact_xi),
            NormResolution::Solved { t: solution }
        );
    }

    #[test]
    fn zero_and_strict_positivity_boundary() {
        assert_eq!(
            solve_norm_equation(&DOmega::zero(), &params()),
            NormResolution::Solved { t: DOmega::zero() }
        );
        assert_eq!(
            solve_norm_equation(&dyadic_zsqrt(&-ZSqrt2::one(), 0), &params()),
            NormResolution::Unsolvable
        );
        assert_eq!(
            solve_norm_equation(&dyadic_zsqrt(&ZSqrt2::sqrt2(), 0), &params()),
            NormResolution::Unsolvable
        );

        let composite = BigInt::from(65_537_u32) * BigInt::from(65_539_u32);
        let negative = dyadic_zsqrt(&ZSqrt2::new(-composite, BigInt::zero()), 0);
        let mut no_factoring = params();
        no_factoring.factor_attempts = 0;
        assert_eq!(
            solve_norm_equation(&negative, &no_factoring),
            NormResolution::Unsolvable
        );
    }

    #[test]
    fn invalid_parameters_and_excessive_scaling_are_exhausted() {
        let mut invalid = params();
        invalid.primality_rounds = 0;
        assert_eq!(
            solve_norm_equation(&DOmega::zero(), &invalid),
            NormResolution::Exhausted
        );

        let excessive = DOmega::new(ZOmega::one(), MAX_SCALE_EXPONENT + 1);
        assert_eq!(
            excessive.least_denominator_exponent(),
            MAX_SCALE_EXPONENT + 1
        );
        assert_eq!(
            solve_norm_equation(&excessive, &params()),
            NormResolution::Exhausted
        );
    }

    #[test]
    fn odd_scaling_counterexample_is_solved_up_to_associates() {
        let xi = dyadic_zsqrt(&ZSqrt2::new(BigInt::from(2_u8), BigInt::one()), 2);
        assert_eq!(xi.least_denominator_exponent(), 1);
        let resolution = solve_norm_equation(&xi, &params());
        assert_solved_exactly(&xi, &resolution);
    }

    #[test]
    fn several_small_norms_constructed_from_roots_are_solved() {
        let roots = [
            DOmega::one(),
            DOmega::from(&ZOmega::one() + &ZOmega::omega()),
            DOmega::from(&ZOmega::one() + &ZOmega::i()),
            DOmega::new(
                ZOmega::new(
                    BigInt::from(2_u8),
                    BigInt::one(),
                    BigInt::from(-1),
                    BigInt::zero(),
                ),
                2,
            ),
        ];
        for root in roots {
            let xi = &root.conjugate() * &root;
            let resolution = solve_norm_equation(&xi, &params());
            assert_solved_exactly(&xi, &resolution);
        }

        // Exercise every constructive odd-prime residue class used by C.20.
        for rational in [3_u8, 5, 17] {
            let xi = dyadic_zsqrt(&ZSqrt2::new(BigInt::from(rational), BigInt::zero()), 0);
            let resolution = solve_norm_equation(&xi, &params());
            assert_solved_exactly(&xi, &resolution);
        }
    }

    #[test]
    fn seven_has_the_proved_prime_power_obstruction() {
        // RS Lemma C.12 splits 7 as (3+sqrt(2))(3-sqrt(2)); both factors
        // have norm 7 and occur to odd exponent, so Lemma C.21 obstructs it.
        let xi = dyadic_zsqrt(&ZSqrt2::new(BigInt::from(7_u8), BigInt::zero()), 0);
        assert_eq!(
            solve_norm_equation(&xi, &params()),
            NormResolution::Unsolvable
        );
    }

    #[test]
    fn single_split_seven_prime_obstructs_only_at_odd_exponent() {
        let split_prime = ZSqrt2::new(BigInt::from(3_u8), BigInt::one());
        assert_eq!(split_prime.norm(), BigInt::from(7_u8));

        let odd_power = dyadic_zsqrt(&split_prime, 0);
        assert_eq!(
            solve_norm_equation(&odd_power, &params()),
            NormResolution::Unsolvable
        );

        let even_power = dyadic_zsqrt(&(&split_prime * &split_prime), 0);
        let resolution = solve_norm_equation(&even_power, &params());
        assert_solved_exactly(&even_power, &resolution);
    }

    #[test]
    fn seeded_solvable_instances_verify_independently() {
        let mut state = 0x1234_5678_9abc_def0_u64;
        for _ in 0..32 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let coordinate =
                |shift: u32| i64::try_from((state >> shift) & 7).expect("three bits fit i64") - 3;
            let numerator = ZOmega::new(
                BigInt::from(coordinate(0)),
                BigInt::from(coordinate(3)),
                BigInt::from(coordinate(6)),
                BigInt::from(coordinate(9)),
            );
            if numerator.is_zero() {
                continue;
            }
            let t = DOmega::new(
                numerator,
                u32::try_from(state & 3).expect("two bits fit u32"),
            );
            let xi = &t.conjugate() * &t;
            let resolution = solve_norm_equation(&xi, &params());
            assert_solved_exactly(&xi, &resolution);
        }
    }

    #[test]
    fn resolution_is_deterministic() {
        let xi = dyadic_zsqrt(&ZSqrt2::new(BigInt::from(41_u8), BigInt::zero()), 0);
        let first = solve_norm_equation(&xi, &params());
        let second = solve_norm_equation(&xi, &params());
        assert_eq!(first, second);
        assert_eq!(
            first,
            NormResolution::Solved {
                t: DOmega::from(ZOmega::new(
                    BigInt::zero(),
                    BigInt::from(5_u8),
                    BigInt::zero(),
                    BigInt::from(-4),
                )),
            }
        );
        assert_solved_exactly(&xi, &first);
    }

    #[test]
    fn probable_seven_mod_eight_obstruction_is_not_a_proof() {
        let rational_prime = BigUint::from(18_446_744_073_709_552_423_u128);
        assert_eq!(residue_mod_u8(&rational_prime), 7);
        let sqrt_two = modular_square_root(&BigUint::from(2_u8), &rational_prime, 64)
            .expect("two is a residue for this prime");
        let split = ZSqrt2::new(
            BigInt::from_biguint(Sign::Plus, rational_prime.clone()),
            BigInt::zero(),
        )
        .gcd(&ZSqrt2::new(
            BigInt::from_biguint(Sign::Plus, sqrt_two),
            BigInt::one(),
        ))
        .expect("split-prime gcd should converge");
        let lambda = ZSqrt2::new(BigInt::one(), BigInt::one());
        let mut positive = if split.norm().is_negative() {
            &split * &lambda
        } else {
            split
        };
        if positive < ZSqrt2::zero() {
            positive = -positive;
        }
        assert!(positive > ZSqrt2::zero());
        assert!(positive.sqrt2_conjugate() > ZSqrt2::zero());

        let xi = dyadic_zsqrt(&positive, 0);
        assert_eq!(
            solve_norm_equation(&xi, &params()),
            NormResolution::Exhausted
        );
    }

    #[test]
    fn nonresidue_budget_propagates_as_exhausted() {
        let xi = dyadic_zsqrt(&ZSqrt2::new(BigInt::from(5_u8), BigInt::zero()), 0);
        let mut exhausted = params();
        exhausted.nonresidue_trials = 0;
        assert_eq!(
            solve_norm_equation(&xi, &exhausted),
            NormResolution::Exhausted
        );
    }

    #[test]
    fn factor_attempt_budget_propagates_as_exhausted() {
        let rational = BigInt::from(65_537_u32) * BigInt::from(65_539_u32);
        let xi = dyadic_zsqrt(&ZSqrt2::new(rational, BigInt::zero()), 0);
        let mut exhausted = params();
        exhausted.factor_attempts = 0;
        assert_eq!(
            solve_norm_equation(&xi, &exhausted),
            NormResolution::Exhausted
        );
    }
}
