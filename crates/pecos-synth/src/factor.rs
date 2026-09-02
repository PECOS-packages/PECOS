//! Bounded integer factoring and modular square roots for norm equations.

use std::collections::BTreeMap;

use num_bigint::BigUint;
use num_traits::{One, ToPrimitive, Zero};
use pecos_random::nth_derived_seed;

const SMALL_PRIME_LIMIT: usize = 1 << 16;
const SIEVE_WORDS: usize = SMALL_PRIME_LIMIT / 64;
const DETERMINISTIC_BASES: [u64; 12] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];

const fn prime_sieve() -> [u64; SIEVE_WORDS] {
    let mut sieve = [u64::MAX; SIEVE_WORDS];
    sieve[0] &= !3_u64;
    let mut prime = 2;
    while prime * prime < SMALL_PRIME_LIMIT {
        if sieve[prime / 64] & (1_u64 << (prime % 64)) != 0 {
            let mut composite = prime * prime;
            while composite < SMALL_PRIME_LIMIT {
                sieve[composite / 64] &= !(1_u64 << (composite % 64));
                composite += prime;
            }
        }
        prime += 1;
    }
    sieve
}

static SMALL_PRIME_SIEVE: [u64; SIEVE_WORDS] = prime_sieve();

fn is_sieved_prime(value: usize) -> bool {
    SMALL_PRIME_SIEVE[value / 64] & (1_u64 << (value % 64)) != 0
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct FactorParams {
    pub(crate) seed: u64,
    pub(crate) factor_attempts: u32,
    pub(crate) rho_steps: u32,
    pub(crate) nonresidue_trials: u32,
    pub(crate) primality_rounds: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PrimeEvidence {
    Proven,
    Probable,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct PrimeFactor {
    pub(crate) prime: BigUint,
    pub(crate) exponent: u32,
    pub(crate) evidence: PrimeEvidence,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum FactorError {
    Exhausted,
}

#[derive(Clone, Copy)]
#[repr(u64)]
enum Purpose {
    RhoPolynomial = 0,
    RhoStart = 1,
    ProbablePrimeBase = 2,
    Reserved = 3,
}

struct SeedStreams {
    stream_seeds: [u64; 4],
    counters: [u64; 4],
}

impl SeedStreams {
    fn new(seed: u64) -> Self {
        // These nested derivations provide protocol domain separation, not
        // injectivity: purpose streams are shifted copies of one permutation,
        // so cross-purpose value collisions can occur. That is harmless here
        // because draws only parameterize algorithms and are never identifiers.
        Self {
            stream_seeds: [
                nth_derived_seed(seed, Purpose::RhoPolynomial as u64),
                nth_derived_seed(seed, Purpose::RhoStart as u64),
                nth_derived_seed(seed, Purpose::ProbablePrimeBase as u64),
                nth_derived_seed(seed, Purpose::Reserved as u64),
            ],
            counters: [0; 4],
        }
    }

    fn draw_word(&mut self, purpose: Purpose) -> u64 {
        let index = purpose as usize;
        let counter = self.counters[index];
        self.counters[index] = counter
            .checked_add(1)
            .expect("factor seed-stream counter overflowed; counters must never be reused");
        nth_derived_seed(self.stream_seeds[index], counter)
    }

    fn below(&mut self, purpose: Purpose, upper_exclusive: &BigUint) -> BigUint {
        assert!(!upper_exclusive.is_zero(), "random range must be nonempty");
        let words = upper_exclusive.bits().div_ceil(64);
        let mut bytes = Vec::with_capacity(
            usize::try_from(words.saturating_mul(8)).expect("random sample size fits usize"),
        );
        for _ in 0..words {
            // The contract deliberately expands successive 64-bit stream words
            // little-endian and then reduces modulo the range. This accepts the
            // small modulo bias in exchange for a stable cross-platform stream.
            bytes.extend_from_slice(&self.draw_word(purpose).to_le_bytes());
        }
        BigUint::from_bytes_le(&bytes) % upper_exclusive
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Primality {
    Composite,
    Prime(PrimeEvidence),
}

pub(crate) fn factor_integer(
    value: &BigUint,
    params: &FactorParams,
) -> Result<Vec<PrimeFactor>, FactorError> {
    if value.is_zero() {
        return Err(FactorError::Exhausted);
    }
    let mut streams = SeedStreams::new(params.seed);
    let mut remaining = value.clone();
    let mut factors = BTreeMap::<BigUint, (u32, PrimeEvidence)>::new();

    // A compile-time sieve supplies every prime below 2^16. We deliberately
    // traverse the whole fixed table (unless the residual becomes one), rather
    // than stopping at sqrt(n), so this stage has no input-dependent budget.
    for candidate in (2..SMALL_PRIME_LIMIT).filter(|candidate| is_sieved_prime(*candidate)) {
        if remaining.is_one() {
            break;
        }
        let prime = BigUint::from(u32::try_from(candidate).expect("sieve index fits u32"));
        let mut exponent = 0_u32;
        while (&remaining % &prime).is_zero() {
            remaining /= &prime;
            exponent = exponent
                .checked_add(1)
                .expect("prime exponent exceeded u32::MAX");
        }
        if exponent != 0 {
            factors.insert(prime, (exponent, PrimeEvidence::Proven));
        }
    }

    if !remaining.is_one() {
        factor_residual(&remaining, params, &mut streams, &mut factors)?;
    }

    Ok(factors
        .into_iter()
        .map(|(prime, (exponent, evidence))| PrimeFactor {
            prime,
            exponent,
            evidence,
        })
        .collect())
}

fn factor_residual(
    value: &BigUint,
    params: &FactorParams,
    streams: &mut SeedStreams,
    factors: &mut BTreeMap<BigUint, (u32, PrimeEvidence)>,
) -> Result<(), FactorError> {
    if value.is_one() {
        return Ok(());
    }
    if let Primality::Prime(evidence) = classify_primality(value, params, streams) {
        let entry = factors.entry(value.clone()).or_insert((0, evidence));
        entry.0 = entry
            .0
            .checked_add(1)
            .expect("prime exponent exceeded u32::MAX");
        if evidence == PrimeEvidence::Probable {
            entry.1 = PrimeEvidence::Probable;
        }
        return Ok(());
    }

    let divisor = pollard_brent(value, params, streams).ok_or(FactorError::Exhausted)?;
    let quotient = value / &divisor;
    // Canonical recursion order makes all stream consumption and output stable.
    if divisor <= quotient {
        factor_residual(&divisor, params, streams, factors)?;
        factor_residual(&quotient, params, streams, factors)
    } else {
        factor_residual(&quotient, params, streams, factors)?;
        factor_residual(&divisor, params, streams, factors)
    }
}

fn classify_primality(
    value: &BigUint,
    params: &FactorParams,
    streams: &mut SeedStreams,
) -> Primality {
    if value < &BigUint::from(2_u8) {
        return Primality::Composite;
    }
    for prime in DETERMINISTIC_BASES {
        let prime = BigUint::from(prime);
        if value == &prime {
            return Primality::Prime(PrimeEvidence::Proven);
        }
        if (value % &prime).is_zero() {
            return Primality::Composite;
        }
    }

    if value.bits() <= 64 {
        // Sorenson & Webster, arXiv:1509.00864v1, Theorem 1.1 computes
        // psi_12 = 318665857834031151167461, the least strong pseudoprime
        // to the first twelve PRIME bases. Since psi_12 > 2^64, passing
        // precisely the primes through 37 is deterministic below 2^64.
        if DETERMINISTIC_BASES
            .iter()
            .all(|base| is_strong_probable_prime(value, &BigUint::from(*base)))
        {
            Primality::Prime(PrimeEvidence::Proven)
        } else {
            Primality::Composite
        }
    } else {
        for _ in 0..params.primality_rounds {
            let span = value - BigUint::from(3_u8);
            let base = streams.below(Purpose::ProbablePrimeBase, &span) + BigUint::from(2_u8);
            if !is_strong_probable_prime(value, &base) {
                return Primality::Composite;
            }
        }
        Primality::Prime(PrimeEvidence::Probable)
    }
}

fn is_strong_probable_prime(value: &BigUint, base: &BigUint) -> bool {
    let one = BigUint::one();
    let value_minus_one = value - &one;
    let mut odd_part = value_minus_one.clone();
    let mut power_of_two = 0_u64;
    while (&odd_part & &one).is_zero() {
        odd_part >>= 1_u8;
        power_of_two += 1;
    }
    let mut x = (base % value).modpow(&odd_part, value);
    if x == one || x == value_minus_one {
        return true;
    }
    for _ in 1..power_of_two {
        x = (&x * &x) % value;
        if x == value_minus_one {
            return true;
        }
    }
    false
}

fn pollard_brent(
    value: &BigUint,
    params: &FactorParams,
    streams: &mut SeedStreams,
) -> Option<BigUint> {
    if (value & BigUint::one()).is_zero() {
        return Some(BigUint::from(2_u8));
    }
    for _ in 0..params.factor_attempts {
        if let Some(factor) = pollard_brent_attempt(value, params.rho_steps, streams) {
            return Some(factor);
        }
    }
    None
}

fn pollard_brent_attempt(
    value: &BigUint,
    step_budget: u32,
    streams: &mut SeedStreams,
) -> Option<BigUint> {
    let one = BigUint::one();
    let c = streams.below(Purpose::RhoPolynomial, &(value - &one)) + &one;
    let start_span = value - BigUint::from(3_u8);
    let mut y = streams.below(Purpose::RhoStart, &start_span) + BigUint::from(2_u8);
    let mut x = BigUint::zero();
    let mut saved_y = y.clone();
    let mut gcd = one.clone();
    let mut block_length = 1_u64;
    let mut steps = 0_u32;

    // Brent, "An improved Monte Carlo factorization algorithm" (BIT 20,
    // 1980), Sections 2, 5, and 7: compare a fixed power-of-two checkpoint
    // with a moving iterate, batch differences into one gcd, and backtrack
    // with single gcds if the batch product vanishes modulo n.
    while gcd.is_one() && steps < step_budget {
        x.clone_from(&y);
        for _ in 0..block_length {
            if steps == step_budget {
                return None;
            }
            y = iterate_polynomial(&y, &c, value);
            steps += 1;
        }

        let mut offset = 0_u64;
        while offset < block_length && gcd.is_one() {
            saved_y.clone_from(&y);
            let width = 128_u64.min(block_length - offset);
            let mut product = one.clone();
            for _ in 0..width {
                if steps == step_budget {
                    return None;
                }
                y = iterate_polynomial(&y, &c, value);
                steps += 1;
                product = (product * absolute_difference(&x, &y)) % value;
            }
            gcd = integer_gcd(product, value.clone());
            offset += width;
        }
        block_length = block_length.checked_mul(2)?;
    }

    if gcd == *value {
        loop {
            if steps == step_budget {
                return None;
            }
            saved_y = iterate_polynomial(&saved_y, &c, value);
            steps += 1;
            gcd = integer_gcd(absolute_difference(&x, &saved_y), value.clone());
            if gcd > one {
                break;
            }
        }
    }
    (gcd != *value && gcd > one).then_some(gcd)
}

fn iterate_polynomial(value: &BigUint, constant: &BigUint, modulus: &BigUint) -> BigUint {
    (value * value + constant) % modulus
}

fn absolute_difference(left: &BigUint, right: &BigUint) -> BigUint {
    if left >= right {
        left - right
    } else {
        right - left
    }
}

fn integer_gcd(mut left: BigUint, mut right: BigUint) -> BigUint {
    while !right.is_zero() {
        let remainder = &left % &right;
        left = right;
        right = remainder;
    }
    left
}

pub(crate) fn modular_square_root(
    value: &BigUint,
    prime: &BigUint,
    nonresidue_trials: u32,
) -> Result<BigUint, FactorError> {
    if prime == &BigUint::from(2_u8) {
        return Ok(value % prime);
    }
    let value = value % prime;
    if value.is_zero() {
        return Ok(BigUint::zero());
    }
    let one = BigUint::one();
    let two = BigUint::from(2_u8);
    let prime_minus_one = prime - &one;
    if value.modpow(&(&prime_minus_one / &two), prime) != one {
        return Err(FactorError::Exhausted);
    }

    let root = if (prime % BigUint::from(4_u8)) == BigUint::from(3_u8) {
        value.modpow(&((prime + &one) / BigUint::from(4_u8)), prime)
    } else {
        let mut odd_part = prime_minus_one.clone();
        let mut power = 0_u32;
        while (&odd_part & &one).is_zero() {
            odd_part >>= 1_u8;
            power = power
                .checked_add(1)
                .expect("Tonelli-Shanks two-adic valuation exceeded u32::MAX");
        }

        let mut nonresidue = None;
        for trial in 0..nonresidue_trials {
            // The stable sequential search is 2 followed by successive odd
            // integers 3, 5, 7, ... . Composite odd candidates need not be
            // skipped; testing one simply consumes its budgeted trial.
            let candidate = if trial == 0 {
                two.clone()
            } else {
                BigUint::from(2_u64 * u64::from(trial) + 1)
            };
            if candidate.modpow(&(&prime_minus_one / &two), prime) == prime_minus_one {
                nonresidue = Some(candidate);
                break;
            }
        }
        let nonresidue = nonresidue.ok_or(FactorError::Exhausted)?;
        let mut c = nonresidue.modpow(&odd_part, prime);
        let mut x = value.modpow(&((&odd_part + &one) / &two), prime);
        let mut t = value.modpow(&odd_part, prime);
        let mut m = power;

        while t != one {
            let mut i = 1_u32;
            let mut square = (&t * &t) % prime;
            while square != one && i < m {
                square = (&square * &square) % prime;
                i += 1;
            }
            if i == m {
                return Err(FactorError::Exhausted);
            }
            let exponent_shift = m
                .checked_sub(i)
                .and_then(|difference| difference.checked_sub(1))
                .expect("Tonelli-Shanks exponent is nonnegative");
            let exponent = BigUint::one()
                << usize::try_from(exponent_shift).expect("u32 shift count fits usize");
            let b = c.modpow(&exponent, prime);
            x = (&x * &b) % prime;
            let b_squared = (&b * &b) % prime;
            t = (&t * &b_squared) % prime;
            c = b_squared;
            m = i;
        }
        x
    };

    if (&root * &root) % prime != value {
        return Err(FactorError::Exhausted);
    }
    let other = prime - &root;
    Ok(if root <= other { root } else { other })
}

pub(crate) fn residue_mod_u8(value: &BigUint) -> u8 {
    (value % BigUint::from(8_u8))
        .to_u8()
        .expect("residue modulo eight fits u8")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> FactorParams {
        FactorParams {
            seed: 0x5eed_1234_9876_abcd,
            factor_attempts: 32,
            rho_steps: 100_000,
            nonresidue_trials: 64,
            primality_rounds: 24,
        }
    }

    #[test]
    fn twelve_bases_reject_classic_strong_pseudoprimes() {
        let mut streams = SeedStreams::new(7);
        // Each is a strong pseudoprime to an increasingly long prefix of the
        // usual prime-base list, but none survives all twelve prime bases.
        // The sequence through psi_8 is tabulated in Jaeschke, Math. Comp. 61
        // (1993), and Jiang & Deng, arXiv:1207.0063; their Q_11 supplies the
        // final 64-bit case below.
        for composite in [
            2_047_u64,
            1_373_653,
            25_326_001,
            3_215_031_751,
            2_152_302_898_747,
            3_474_749_660_383,
            341_550_071_728_321,
            3_825_123_056_546_413_051,
        ] {
            let composite = BigUint::from(composite);
            assert!(is_strong_probable_prime(&composite, &BigUint::from(2_u8)));
            assert!(
                !DETERMINISTIC_BASES
                    .iter()
                    .all(|base| is_strong_probable_prime(&composite, &BigUint::from(*base)))
            );
            assert_eq!(
                classify_primality(&composite, &params(), &mut streams),
                Primality::Composite
            );
        }
    }

    #[test]
    fn primality_above_u64_is_only_probable() {
        let prime = BigUint::from(18_446_744_073_709_552_423_u128);
        let mut streams = SeedStreams::new(params().seed);
        assert_eq!(
            classify_primality(&prime, &params(), &mut streams),
            Primality::Prime(PrimeEvidence::Probable)
        );
    }

    #[test]
    fn factors_are_sorted_and_reconstruct_input() {
        let input = BigUint::from(2_u8).pow(7)
            * BigUint::from(65_521_u32)
            * BigUint::from(65_537_u32)
            * BigUint::from(1_000_003_u32);
        let factors = factor_integer(&input, &params()).expect("factorization should succeed");
        assert!(factors.windows(2).all(|pair| pair[0].prime < pair[1].prime));
        let reconstructed = factors.iter().fold(BigUint::one(), |product, factor| {
            product * factor.prime.pow(factor.exponent)
        });
        assert_eq!(reconstructed, input);
        assert!(
            factors
                .iter()
                .all(|factor| factor.evidence == PrimeEvidence::Proven)
        );
    }

    #[test]
    fn rho_attempt_exhaustion_is_explicit() {
        let mut exhausted = params();
        exhausted.factor_attempts = 0;
        let composite = BigUint::from(65_537_u32) * BigUint::from(65_539_u32);
        assert_eq!(
            factor_integer(&composite, &exhausted),
            Err(FactorError::Exhausted)
        );

        exhausted.factor_attempts = 1;
        exhausted.rho_steps = 0;
        assert_eq!(
            factor_integer(&composite, &exhausted),
            Err(FactorError::Exhausted)
        );
    }

    #[test]
    fn tonelli_shanks_roots_are_canonical_in_used_residue_classes() {
        for (prime, radicands) in [
            (7_u32, vec![2_u32]),
            (11, vec![9]),
            (13, vec![3, 12]),
            (17, vec![2, 16]),
            (41, vec![2, 40]),
            (73, vec![2, 72]),
        ] {
            let prime = BigUint::from(prime);
            for radicand in radicands {
                let radicand = BigUint::from(radicand);
                let root = modular_square_root(&radicand, &prime, 64)
                    .expect("listed radicand should be a square");
                assert_eq!((&root * &root) % &prime, radicand % &prime);
                assert!(root <= &prime / BigUint::from(2_u8));
            }
        }
    }

    #[test]
    fn tonelli_nonresidue_budget_is_enforced() {
        assert_eq!(
            modular_square_root(&BigUint::from(4_u8), &BigUint::from(13_u8), 0),
            Err(FactorError::Exhausted)
        );
    }
}
