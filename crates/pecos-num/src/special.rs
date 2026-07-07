// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Special functions used by PECOS numerical routines.
//!
//! The incomplete-beta implementation uses DLMF/Abramowitz-Stegun continued
//! fractions, public asymptotic expansions, and bracketed inverse solves with
//! the reflection branch selected before forming complements.

use std::fmt;

const BETACF_MAX_ITERATIONS: usize = 10_000;
const BETACF_EPSILON: f64 = f64::EPSILON;
const BETA_POWER_SERIES_MAX_ITERATIONS: usize = 10_000;
const BETA_POWER_SERIES_EPSILON: f64 = f64::EPSILON;
const GAMMA_MAX_ITERATIONS: usize = 10_000;
const GAMMA_EPSILON: f64 = f64::EPSILON;
/// Minimum `b` for the large-b beta expansion.
///
/// The effective accuracy bound is the scale test `a*a/b <= 0.1`; this floor
/// only prevents very small-count probes from entering the asymptotic path.
/// Round-5 validation swept the scipy pocket fixtures from `n = 1e6` through
/// `n = 8_999_990` for `a <= 64.5`, plus below-floor ordered probes at
/// `b ~= 1e4` and `b ~= 4.1e4`.
const BETA_LARGE_SHAPE_MIN: f64 = 10_000.0;
const BETA_LARGE_SHAPE_MODERATE_MAX: f64 = 3_001.0;
/// Maximum beta argument for large-b expansion probes.
///
/// Jeffreys quantiles in the validated large-b branch are well below this; the
/// wider guard keeps inverse-solver boundary probes out of the reflected-CF
/// underflow zone.
const BETA_LARGE_SHAPE_X_MAX: f64 = 1.0e-2;
/// Expansion scale bound for the BGRAT large-b series.
///
/// The value was retained only after fixture and boundary sweeps covering
/// `a <= 3000.5`, `b <= 1e8`, and the Round-5 small-a pocket rows.
const BETA_LARGE_SHAPE_EXPANSION_PARAMETER_MAX: f64 = 0.1;
const BETA_LARGE_B_EXPANSION_TERMS: usize = 48;
const BETA_LARGE_B_EXPANSION_LEN: usize = BETA_LARGE_B_EXPANSION_TERMS + 1;
const BETA_GAUSS_LEGENDRE_SWITCH: f64 = 3_000.0;
const BETA_GAUSS_LEGENDRE_PANELS: usize = 2;
const BETA_GAUSS_LEGENDRE_PROBABILITY_FLOOR: f64 = 5.0e-15;
const BETA_ASYMPTOTIC_NORMALIZER_MIN: f64 = 100_000.0;
const BETA_MEDIAN_ASYMPTOTIC_MIN: f64 = 1_000_000.0;
const GAMMA_HALF_INTEGER_MAX_X: f64 = 700.0;
const GAMMA_HALF_INTEGER_MAX_STEPS: usize = 128;
/// Positive floor used by the modified Lentz continued-fraction recurrence.
///
/// Lentz's algorithm evaluates continued fractions through paired forward and
/// backward recurrences; Thompson-Barnett style implementations replace a
/// denominator whose magnitude underflows with a tiny nonzero value.
const LENTZ_TINY: f64 = f64::MIN_POSITIVE / f64::EPSILON;
const HALF_LN_TWO_PI: f64 = 0.918_938_533_204_672_8;
const LN_EPSILON: f64 = -36.043_653_389_117_15;
const LN_MIN_POSITIVE: f64 = -708.396_418_532_264_1;
const SQRT_PI: f64 = 1.772_453_850_905_516;
const GAUSS_LEGENDRE_Y: [f64; 64] = [
    0.000_347_479_132_113_914_8,
    0.001_829_941_614_022_39,
    0.004_493_314_261_627_856_5,
    0.008_331_873_057_687_012,
    0.013_336_586_105_044_512,
    0.019_495_600_173_973_116,
    0.026_794_312_570_798_617,
    0.035_215_413_934_030_215,
    0.044_738_931_460_748_59,
    0.055_342_277_002_442_93,
    0.067_000_300_922_953_56,
    0.079_685_351_873_709_79,
    0.093_367_342_438_601_23,
    0.108_013_820_528_329_31,
    0.123_590_046_369_734_03,
    0.140_059_074_914_194_56,
    0.157_381_843_472_883_36,
    0.175_517_264_372_671_34,
    0.194_422_322_413_803_36,
    0.214_052_176_898_683,
    0.234_360_267_990_052_72,
    0.255_298_427_146_473_55,
    0.276_816_991_373_268,
    0.298_864_921_018_004_2,
    0.321_389_920_831_165_9,
    0.344_338_564_004_894_5,
    0.367_656_418_895_616_3,
    0.391_288_178_129_996_44,
    0.415_177_789_788_003_57,
    0.439_268_590_351_939_7,
    0.463_503_439_106_100_5,
    0.487_824_853_668_287_76,
    0.512_175_146_331_712_2,
    0.536_496_560_893_899_5,
    0.560_731_409_648_060_2,
    0.584_822_210_211_996_4,
    0.608_711_821_870_003_5,
    0.632_343_581_104_383_7,
    0.655_661_435_995_105_5,
    0.678_610_079_168_834_1,
    0.701_135_078_981_995_8,
    0.723_183_008_626_732,
    0.744_701_572_853_526_5,
    0.765_639_732_009_947_3,
    0.785_947_823_101_317_1,
    0.805_577_677_586_196_7,
    0.824_482_735_627_328_7,
    0.842_618_156_527_116_7,
    0.859_940_925_085_805_4,
    0.876_409_953_630_266,
    0.891_986_179_471_670_7,
    0.906_632_657_561_398_8,
    0.920_314_648_126_290_3,
    0.932_999_699_077_046_4,
    0.944_657_722_997_557,
    0.955_261_068_539_251_4,
    0.964_784_586_065_969_8,
    0.973_205_687_429_201_4,
    0.980_504_399_826_026_9,
    0.986_663_413_894_955_5,
    0.991_668_126_942_313,
    0.995_506_685_738_372_1,
    0.998_170_058_385_977_6,
    0.999_652_520_867_886_1,
];
const GAUSS_LEGENDRE_W: [f64; 64] = [
    0.000_891_640_360_848_133_9,
    0.002_073_516_630_281_260_4,
    0.003_252_228_984_489_184,
    0.004_423_379_913_181_967,
    0.005_584_069_730_065_538,
    0.006_731_523_948_359_368,
    0.007_863_015_238_012_236,
    0.008_975_857_887_848_613,
    0.010_067_411_576_765_09,
    0.011_135_086_904_191_606,
    0.012_176_351_284_355_466,
    0.013_188_734_857_527_322,
    0.014_169_836_307_129_73,
    0.015_117_328_536_201_213,
    0.016_028_964_177_425_803,
    0.016_902_580_918_570_807,
    0.017_736_106_628_441_18,
    0.018_527_564_270_120_034,
    0.019_275_076_589_307_8,
    0.019_976_870_566_360_18,
    0.020_631_281_621_311_774,
    0.021_236_757_561_826_813,
    0.021_791_862_264_661_736,
    0.022_295_279_081_878_29,
    0.022_745_813_963_709_06,
    0.023_142_398_290_657_198,
    0.023_484_091_408_104_996,
    0.023_770_082_857_415_158,
    0.023_999_694_298_229_166,
    0.024_172_381_117_401_45,
    0.024_287_733_720_751_697,
    0.024_345_478_504_569_865,
    0.024_345_478_504_569_865,
    0.024_287_733_720_751_697,
    0.024_172_381_117_401_45,
    0.023_999_694_298_229_166,
    0.023_770_082_857_415_158,
    0.023_484_091_408_104_996,
    0.023_142_398_290_657_198,
    0.022_745_813_963_709_06,
    0.022_295_279_081_878_29,
    0.021_791_862_264_661_736,
    0.021_236_757_561_826_813,
    0.020_631_281_621_311_774,
    0.019_976_870_566_360_18,
    0.019_275_076_589_307_8,
    0.018_527_564_270_120_034,
    0.017_736_106_628_441_18,
    0.016_902_580_918_570_807,
    0.016_028_964_177_425_803,
    0.015_117_328_536_201_213,
    0.014_169_836_307_129_73,
    0.013_188_734_857_527_322,
    0.012_176_351_284_355_466,
    0.011_135_086_904_191_606,
    0.010_067_411_576_765_09,
    0.008_975_857_887_848_613,
    0.007_863_015_238_012_236,
    0.006_731_523_948_359_368,
    0.005_584_069_730_065_538,
    0.004_423_379_913_181_967,
    0.003_252_228_984_489_184,
    0.002_073_516_630_281_260_4,
    0.000_891_640_360_848_133_9,
];

/// Error type for special-function routines.
#[derive(Debug, Clone, PartialEq)]
pub enum SpecialError {
    /// Input was outside the mathematical domain of the requested function.
    InvalidInput {
        /// Explanation of the invalid input.
        message: String,
    },
    /// A fixed iteration budget was exhausted before convergence.
    MaxIterations {
        /// Name of the routine that exhausted its budget.
        function: &'static str,
        /// Number of iterations attempted.
        iterations: usize,
    },
    /// A non-finite or otherwise unusable intermediate value was encountered.
    NumericalIssue {
        /// Explanation of the numerical issue.
        message: String,
    },
}

impl fmt::Display for SpecialError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput { message } => write!(f, "Invalid input: {message}"),
            Self::MaxIterations {
                function,
                iterations,
            } => write!(
                f,
                "Maximum iterations ({iterations}) exceeded in {function}"
            ),
            Self::NumericalIssue { message } => write!(f, "Numerical issue: {message}"),
        }
    }
}

impl std::error::Error for SpecialError {}

/// Options for the bracket-preserving inverse regularized-beta solver.
#[derive(Debug, Clone, Copy)]
pub struct InverseBetaOptions {
    /// Maximum number of Newton/Halley iterations.
    pub max_newton: usize,
    /// Maximum number of bisection-only iterations after Newton is exhausted.
    pub max_bisect: usize,
    /// Absolute probability-scale residual tolerance.
    pub probability_tolerance: f64,
    /// Relative-x step tolerance.
    pub relative_x_tolerance: f64,
}

impl Default for InverseBetaOptions {
    fn default() -> Self {
        Self {
            max_newton: 100,
            max_bisect: 200,
            probability_tolerance: 1.0e-15,
            relative_x_tolerance: 1.0e-12,
        }
    }
}

/// Inverse-beta quantile and its complement.
///
/// For roots at or below `0.5`, [`betainc_inv`] solves directly for [`Self::x`]
/// and derives [`Self::complement`]. For roots above `0.5`, it solves the
/// swapped complement problem directly for [`Self::complement`] and derives
/// [`Self::x`]. This reconciles the design-note requirement to avoid losing the
/// well-conditioned side to `1 - x` cancellation while preserving the
/// user-facing `x`/`complement` pair.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BetaQuantile {
    /// The quantile `x` such that `I_x(a,b) = p`.
    pub x: f64,
    /// The complement `1 - x`, returned with the quantile for tail-sensitive callers.
    pub complement: f64,
}

/// Lower/upper tail probabilities for a regularized distribution function.
///
/// The pair is used for both beta `I_x(a,b)`/`1 - I_x(a,b)` and gamma
/// `P(a,x)`/`Q(a,x)` helper paths.
#[derive(Debug, Clone, Copy)]
struct TailPair {
    lower: f64,
    upper: f64,
}

/// Natural logarithm of the gamma function.
///
/// This uses `libm::lgamma` explicitly, matching the Jeffreys interval
/// determinism contract for special-function paths. The v3 design note's
/// implementation plan named a Lanczos approximation; delegating to
/// `libm::lgamma` is an intentional reconciliation with the same note's
/// mandatory libm-by-name determinism contract.
///
/// # Errors
///
/// Returns an error if `x` is not positive and finite, or if `libm::lgamma`
/// returns a non-finite value.
///
/// # Examples
///
/// ```
/// use pecos_num::special::ln_gamma;
///
/// let value = ln_gamma(5.0).unwrap();
/// assert!((value - 3.178_053_830_347_945_8).abs() < 1e-14);
/// ```
pub fn ln_gamma(x: f64) -> Result<f64, SpecialError> {
    ensure_positive("x", x)?;
    let value = libm::lgamma(x);
    if value.is_finite() {
        Ok(value)
    } else {
        Err(SpecialError::NumericalIssue {
            message: "libm::lgamma returned a non-finite value".to_string(),
        })
    }
}

/// Regularized incomplete beta function `I_x(a,b)`.
///
/// The implementation evaluates the continued fraction in the better
/// conditioned tail and uses the reflection identity
/// `I_x(a,b) = 1 - I_{1-x}(b,a)` when the complement is the stable branch.
///
/// # Errors
///
/// Returns an error if `a <= 0`, `b <= 0`, `x` is outside `[0, 1]`, a
/// non-finite value is encountered, or the continued-fraction iteration budget
/// is exhausted.
///
/// # Examples
///
/// ```
/// use pecos_num::special::betainc_reg;
///
/// let value = betainc_reg(2.0, 2.0, 0.5).unwrap();
/// assert!((value - 0.5).abs() < 1e-14);
/// ```
pub fn betainc_reg(a: f64, b: f64, x: f64) -> Result<f64, SpecialError> {
    Ok(betainc_reg_pair(a, b, x)?.lower)
}

/// Inverse regularized incomplete beta function.
///
/// Returns both `x` and `1 - x` for the solution to `I_x(a,b) = p`. When the
/// root lies above `0.5`, this solves the swapped problem for `1 - x` directly
/// and derives `x`; otherwise it solves `x` directly and derives the
/// complement. For `p = 0.5` with both shapes at least `1_000_000`, it uses the
/// Peizer-Pratt/Kerman large-shape beta-median asymptotic because the GL
/// forward CDF noise dominates further Newton refinement. See [`BetaQuantile`]
/// for the exact-side contract.
///
/// # Errors
///
/// Returns an error if the inputs are outside the function domain, a
/// non-finite value is encountered, or the fixed iteration budget is exhausted.
///
/// # Examples
///
/// ```
/// use pecos_num::special::betainc_inv;
///
/// let quantile = betainc_inv(0.5, 2.0, 2.0).unwrap();
/// assert!((quantile.x - 0.5).abs() < 1e-14);
/// ```
pub fn betainc_inv(p: f64, a: f64, b: f64) -> Result<BetaQuantile, SpecialError> {
    betainc_inv_with_options(p, a, b, InverseBetaOptions::default())
}

/// Inverse regularized incomplete beta function with explicit iteration options.
///
/// This is primarily useful for tests and diagnostics that need to assert the
/// fixed-budget error behavior.
///
/// # Errors
///
/// Returns an error if the inputs or options are invalid, a non-finite value is
/// encountered, or the configured iteration budget is exhausted.
pub fn betainc_inv_with_options(
    p: f64,
    a: f64,
    b: f64,
    options: InverseBetaOptions,
) -> Result<BetaQuantile, SpecialError> {
    ensure_unit_interval("p", p)?;
    ensure_positive("a", a)?;
    ensure_positive("b", b)?;
    ensure_positive("probability_tolerance", options.probability_tolerance)?;
    ensure_positive("relative_x_tolerance", options.relative_x_tolerance)?;

    if p <= 0.0 {
        return Ok(BetaQuantile {
            x: 0.0,
            complement: 1.0,
        });
    }
    if p >= 1.0 {
        return Ok(BetaQuantile {
            x: 1.0,
            complement: 0.0,
        });
    }
    if let Some(quantile) = large_shape_median_quantile(p, a, b) {
        return Ok(quantile);
    }

    let half = betainc_reg_pair(a, b, 0.5)?;
    if p > half.lower {
        let complement = betainc_inv_lower_tail(1.0 - p, b, a, options)?;
        return Ok(BetaQuantile {
            x: complement.complement,
            complement: complement.x,
        });
    }

    betainc_inv_lower_tail(p, a, b, options)
}

fn betainc_inv_lower_tail(
    p: f64,
    a: f64,
    b: f64,
    options: InverseBetaOptions,
) -> Result<BetaQuantile, SpecialError> {
    let total_iterations = options.max_newton + options.max_bisect;
    if total_iterations == 0 {
        return Err(SpecialError::MaxIterations {
            function: "betainc_inv",
            iterations: 0,
        });
    }

    let mut lo = 0.0;
    let mut hi = 1.0;
    let mut x = initial_betainc_inv_guess(p, a, b);
    if !is_strictly_inside(x, lo, hi) {
        x = midpoint(lo, hi);
    }

    let mut previous_x = x;
    let mut have_previous_step = false;
    let mut lo_established = false;
    let mut hi_established = false;
    let effective_probability_tolerance = effective_probability_tolerance(options, a, b);

    for iteration in 0..total_iterations {
        let residual = betainc_inv_residual(p, a, b, x)?;
        let abs_residual = residual.abs();

        if same_float(abs_residual, 0.0) {
            return Ok(direct_quantile(x));
        }

        if residual < 0.0 {
            lo = x;
            lo_established = true;
        } else {
            hi = x;
            hi_established = true;
        }

        let bracket_established = lo_established && hi_established;
        let step_ok = have_previous_step
            && (x - previous_x).abs() <= options.relative_x_tolerance * previous_x.abs();
        let bracket_limited = bracket_established && bracket_is_representation_limited(lo, hi, x);
        if abs_residual <= effective_probability_tolerance && (step_ok || bracket_limited) {
            return Ok(direct_quantile(x));
        }
        if bracket_limited {
            return Ok(direct_quantile(midpoint(lo, hi)));
        }

        let newton = (iteration < options.max_newton)
            .then(|| newton_candidate(a, b, x, residual))
            .flatten()
            .filter(|&candidate| is_strictly_inside(candidate, lo, hi));
        let mut candidate =
            newton.unwrap_or_else(|| fallback_candidate(residual, x, lo, hi, bracket_established));

        if same_float(candidate, x) {
            if abs_residual <= effective_probability_tolerance {
                return Ok(direct_quantile(x));
            }
            candidate = fallback_candidate(residual, x, lo, hi, bracket_established);
        }

        if same_float(candidate, x) && bracket_established {
            let bisected = midpoint(lo, hi);
            if same_float(bisected, x) {
                if bracket_limited {
                    return Ok(direct_quantile(bisected));
                }
                return Err(SpecialError::MaxIterations {
                    function: "betainc_inv",
                    iterations: iteration + 1,
                });
            }
            candidate = bisected;
        }

        if same_float(candidate, x) {
            return Err(SpecialError::MaxIterations {
                function: "betainc_inv",
                iterations: iteration + 1,
            });
        }

        previous_x = x;
        x = candidate;
        have_previous_step = true;
    }

    Err(SpecialError::MaxIterations {
        function: "betainc_inv",
        iterations: total_iterations,
    })
}

fn effective_probability_tolerance(options: InverseBetaOptions, a: f64, b: f64) -> f64 {
    if a > BETA_GAUSS_LEGENDRE_SWITCH && b > BETA_GAUSS_LEGENDRE_SWITCH {
        // This is a local solver tolerance, not a global bound on GL forward
        // error against scipy. Large symmetric CDF probes can differ by up to
        // roughly `(a + b) * eps` from log-difference rounding; measured probes
        // were about `0.26 * (a + b) * eps`. Newton only needs a deterministic
        // zero of this fixed forward function, and the bracket-collapse check
        // plus huge local density keep the returned x stable.
        options
            .probability_tolerance
            .max(BETA_GAUSS_LEGENDRE_PROBABILITY_FLOOR)
    } else {
        options.probability_tolerance
    }
}

fn fallback_candidate(residual: f64, x: f64, lo: f64, hi: f64, bracket_established: bool) -> f64 {
    if bracket_established {
        return midpoint(lo, hi);
    }

    if residual < 0.0 {
        let expanded = (2.0 * x).min(0.5).max(midpoint(x, 0.5));
        if is_strictly_inside(expanded, x, hi) {
            expanded
        } else {
            midpoint(x, hi)
        }
    } else {
        let contracted = 0.5 * x;
        if is_strictly_inside(contracted, lo, x) {
            contracted
        } else {
            midpoint(lo, x)
        }
    }
}

fn large_shape_median_quantile(p: f64, a: f64, b: f64) -> Option<BetaQuantile> {
    if !same_float(p, 0.5) || a < BETA_MEDIAN_ASYMPTOTIC_MIN || b < BETA_MEDIAN_ASYMPTOTIC_MIN {
        return None;
    }

    // Peizer-Pratt/Kerman large-shape beta-median asymptotic. In this regime
    // the GL forward CDF noise dominates further Newton refinement; the
    // approximation is the Boost/Temme-style asymptotic inverse used for the
    // median-only path.
    let denominator = a + b - 2.0 / 3.0;
    if a > b {
        let complement = (b - 1.0 / 3.0) / denominator;
        Some(BetaQuantile {
            x: 1.0 - complement,
            complement,
        })
    } else {
        let x = (a - 1.0 / 3.0) / denominator;
        Some(BetaQuantile {
            x,
            complement: 1.0 - x,
        })
    }
}

#[inline]
fn same_float(left: f64, right: f64) -> bool {
    left.to_bits() == right.to_bits()
}

fn betainc_reg_pair(a: f64, b: f64, x: f64) -> Result<TailPair, SpecialError> {
    ensure_positive("a", a)?;
    ensure_positive("b", b)?;
    ensure_unit_interval("x", x)?;

    if x <= 0.0 {
        return Ok(TailPair {
            lower: 0.0,
            upper: 1.0,
        });
    }
    if x >= 1.0 {
        return Ok(TailPair {
            lower: 1.0,
            upper: 0.0,
        });
    }

    if let Some(cdf) = beta_large_shape_pair(a, b, x)? {
        return Ok(cdf);
    }
    if a > BETA_GAUSS_LEGENDRE_SWITCH && b > BETA_GAUSS_LEGENDRE_SWITCH {
        return betainc_reg_gauss_legendre_pair(a, b, x);
    }

    let prefactor = beta_prefactor(a, b, x)?;
    let threshold = (a + 1.0) / (a + b + 2.0);
    if x <= threshold || (x <= 0.1 && b * x <= 5.0) {
        let lower = lower_beta_tail(a, b, x, prefactor)?;
        ensure_finite("regularized beta lower tail", lower)?;
        Ok(TailPair {
            lower,
            upper: 1.0 - lower,
        })
    } else {
        let upper = lower_beta_tail(b, a, 1.0 - x, prefactor)?;
        ensure_finite("regularized beta upper tail", upper)?;
        Ok(TailPair {
            lower: 1.0 - upper,
            upper,
        })
    }
}

fn lower_beta_tail(a: f64, b: f64, x: f64, prefactor: f64) -> Result<f64, SpecialError> {
    if b * x <= 1.0 && x <= 0.95 {
        beta_power_series(a, b, x)
    } else {
        Ok(prefactor * betacf(a, b, x)? / a)
    }
}

fn beta_prefactor(a: f64, b: f64, x: f64) -> Result<f64, SpecialError> {
    let value = libm::exp(log_beta_normalizer(a, b)? + a * libm::log(x) + b * libm::log1p(-x));
    ensure_finite("regularized beta prefactor", value)?;
    Ok(value)
}

fn beta_power_series(a: f64, b: f64, x: f64) -> Result<f64, SpecialError> {
    let reciprocal_a = 1.0 / a;
    let mut term = (1.0 - b) * x;
    let mut value = term / (a + 1.0);
    let mut sum = value;
    let tolerance = BETA_POWER_SERIES_EPSILON * reciprocal_a;
    let mut n = 2.0;

    for _iteration in 2..=BETA_POWER_SERIES_MAX_ITERATIONS {
        term *= (n - b) * x / n;
        value = term / (a + n);
        sum += value;
        if value.abs() <= tolerance {
            let log_prefactor = log_beta_normalizer(a, b)? + a * libm::log(x);
            let result = (sum + reciprocal_a) * libm::exp(log_prefactor);
            ensure_finite("regularized beta power series", result)?;
            return Ok(result);
        }
        n += 1.0;
    }

    Err(SpecialError::MaxIterations {
        function: "beta_power_series",
        iterations: BETA_POWER_SERIES_MAX_ITERATIONS,
    })
}

fn betainc_reg_gauss_legendre_pair(a: f64, b: f64, x: f64) -> Result<TailPair, SpecialError> {
    let a1 = a - 1.0;
    let b1 = b - 1.0;
    let mu = a / (a + b);
    let ln_mu = libm::log(mu);
    let ln_mu_complement = libm::log1p(-mu);
    let sigma = libm::sqrt(a * b / ((a + b) * (a + b) * (a + b + 1.0)));
    let integrate_upper = x > mu;

    let xu = if integrate_upper {
        if x >= 1.0 {
            return Ok(TailPair {
                lower: 1.0,
                upper: 0.0,
            });
        }
        (mu + 10.0 * sigma).max(x + 5.0 * sigma).min(1.0)
    } else {
        if x <= 0.0 {
            return Ok(TailPair {
                lower: 0.0,
                upper: 1.0,
            });
        }
        (mu - 10.0 * sigma).min(x - 5.0 * sigma).max(0.0)
    };

    let span = if integrate_upper { xu - x } else { x - xu };
    let panel_width = span / usize_to_f64(BETA_GAUSS_LEGENDRE_PANELS);
    let start = if integrate_upper { x } else { xu };
    let mut sum = 0.0;
    for panel in 0..BETA_GAUSS_LEGENDRE_PANELS {
        let panel_start = start + usize_to_f64(panel) * panel_width;
        for (&y, &weight) in GAUSS_LEGENDRE_Y.iter().zip(GAUSS_LEGENDRE_W.iter()) {
            let t = panel_start + panel_width * y;
            sum += weight
                * libm::exp(
                    a1 * (libm::log(t) - ln_mu) + b1 * (libm::log1p(-t) - ln_mu_complement),
                );
        }
    }

    let tail = sum
        * panel_width
        * libm::exp(a1 * ln_mu + b1 * ln_mu_complement + log_beta_normalizer(a, b)?);
    ensure_finite("large-shape beta quadrature", tail)?;
    if integrate_upper {
        Ok(TailPair {
            lower: 1.0 - tail,
            upper: tail,
        })
    } else {
        Ok(TailPair {
            lower: tail,
            upper: 1.0 - tail,
        })
    }
}

fn beta_large_shape_pair(a: f64, b: f64, x: f64) -> Result<Option<TailPair>, SpecialError> {
    if let Some(pair) = beta_large_b_saturation_pair(a, b, x)? {
        return Ok(Some(pair));
    }

    if beta_large_b_gate(a, b, x) {
        return Ok(Some(beta_large_b_asymptotic_pair(a, b, x)?));
    }

    let x_complement = 1.0 - x;
    if let Some(swapped) = beta_large_b_saturation_pair(b, a, x_complement)? {
        return Ok(Some(TailPair {
            lower: swapped.upper,
            upper: swapped.lower,
        }));
    }

    if beta_large_b_gate(b, a, x_complement) {
        let swapped = beta_large_b_asymptotic_pair(b, a, x_complement)?;
        return Ok(Some(TailPair {
            lower: swapped.upper,
            upper: swapped.lower,
        }));
    }

    Ok(None)
}

fn beta_large_b_gate(a: f64, b: f64, x: f64) -> bool {
    b >= BETA_LARGE_SHAPE_MIN
        && a <= BETA_LARGE_SHAPE_MODERATE_MAX
        && x <= BETA_LARGE_SHAPE_X_MAX
        && a * a / b <= BETA_LARGE_SHAPE_EXPANSION_PARAMETER_MAX
}

fn beta_large_b_saturation_pair(a: f64, b: f64, x: f64) -> Result<Option<TailPair>, SpecialError> {
    if b < BETA_LARGE_SHAPE_MIN
        || a > BETA_LARGE_SHAPE_MODERATE_MAX
        || a * a / b > BETA_LARGE_SHAPE_EXPANSION_PARAMETER_MAX
    {
        return Ok(None);
    }

    let w = -b * libm::log1p(-x);
    if w <= a {
        return Ok(None);
    }

    if gamma_upper_is_below_precision(a, w)? {
        Ok(Some(TailPair {
            lower: 1.0,
            upper: 0.0,
        }))
    } else {
        Ok(None)
    }
}

// DiDonato and Morris, ACM TOMS 18(3), 1992, Algorithm 708, uses the BGRAT
// large-b expansion to reduce the beta tail to incomplete-gamma tails.  With
// `w = -b log1p(-x)`, the beta integrand is a gamma density multiplied by the
// fixed series `((1 - exp(-z)) / z)^(a - 1)`, `z = w / b`. The fixed expansion
// budget is enabled only for the validated Jeffreys regime `b >= 1e4`,
// `x <= 1e-2`, `a <= 3001`, and `a*a/b <= 0.1`. The scale bound, not the fixed
// floor, is the active lower limit for the Round-5 small-a pocket rows; far-tail
// probes with gamma-Q below machine precision saturate before reaching the
// reflected continued fraction.
fn beta_large_b_asymptotic_pair(a: f64, b: f64, x: f64) -> Result<TailPair, SpecialError> {
    let w = -b * libm::log1p(-x);
    if w <= 0.0 {
        return Ok(TailPair {
            lower: 0.0,
            upper: 1.0,
        });
    }

    let coefficients = large_b_expansion_coefficients(a);
    let mut rising = 1.0;
    let mut b_power = 1.0;
    let mut lower_sum = 0.0;
    let mut upper_sum = 0.0;

    for (index, coefficient) in coefficients.iter().enumerate() {
        if index > 0 {
            rising *= a + usize_to_f64(index - 1);
            b_power *= b;
        }

        let weight = coefficient * rising / b_power;
        let gamma = regularized_gamma_pq(a + usize_to_f64(index), w)?;
        lower_sum += weight * gamma.lower;
        upper_sum += weight * gamma.upper;
    }

    let normalizer = libm::exp(a * libm::log(b) - log_gamma_delta(b, a));
    ensure_finite("large-b beta expansion normalizer", normalizer)?;
    if normalizer <= 0.0 {
        return Err(SpecialError::NumericalIssue {
            message: "large-b beta expansion normalizer was non-positive".to_string(),
        });
    }

    let lower = clean_unit_probability("large-b beta lower tail", lower_sum / normalizer)?;
    let upper = clean_unit_probability("large-b beta upper tail", upper_sum / normalizer)?;
    Ok(TailPair { lower, upper })
}

fn large_b_expansion_coefficients(a: f64) -> [f64; BETA_LARGE_B_EXPANSION_LEN] {
    let exponent = a - 1.0;
    let mut base = [0.0; BETA_LARGE_B_EXPANSION_LEN];
    let mut factorial = 1.0;
    for (index, coefficient) in base.iter_mut().enumerate().skip(1) {
        factorial *= usize_to_f64(index + 1);
        let sign = if index % 2 == 0 { 1.0 } else { -1.0 };
        *coefficient = sign / factorial;
    }

    let mut coefficients = [0.0; BETA_LARGE_B_EXPANSION_LEN];
    coefficients[0] = 1.0;
    let mut base_power = [0.0; BETA_LARGE_B_EXPANSION_LEN];
    base_power[0] = 1.0;
    let mut binomial = 1.0;

    for order in 1..BETA_LARGE_B_EXPANSION_LEN {
        base_power = multiply_series(&base_power, &base);
        binomial *= (exponent - usize_to_f64(order - 1)) / usize_to_f64(order);
        for index in order..BETA_LARGE_B_EXPANSION_LEN {
            coefficients[index] += binomial * base_power[index];
        }
    }

    coefficients
}

fn multiply_series(
    left: &[f64; BETA_LARGE_B_EXPANSION_LEN],
    right: &[f64; BETA_LARGE_B_EXPANSION_LEN],
) -> [f64; BETA_LARGE_B_EXPANSION_LEN] {
    let mut product = [0.0; BETA_LARGE_B_EXPANSION_LEN];
    for left_index in 0..BETA_LARGE_B_EXPANSION_LEN {
        for right_index in 0..(BETA_LARGE_B_EXPANSION_LEN - left_index) {
            product[left_index + right_index] += left[left_index] * right[right_index];
        }
    }
    product
}

fn regularized_gamma_pq(a: f64, x: f64) -> Result<TailPair, SpecialError> {
    ensure_positive("a", a)?;
    if x < 0.0 || !x.is_finite() {
        return Err(SpecialError::InvalidInput {
            message: "x must be non-negative and finite".to_string(),
        });
    }
    if x <= 0.0 {
        return Ok(TailPair {
            lower: 0.0,
            upper: 1.0,
        });
    }

    if let Some(pair) = regularized_gamma_half_integer_pq(a, x)? {
        return Ok(pair);
    }

    if x < a + 1.0 {
        let lower = regularized_gamma_p_series(a, x)?;
        Ok(TailPair {
            lower,
            upper: 1.0 - lower,
        })
    } else {
        if gamma_upper_is_below_precision(a, x)? {
            return Ok(TailPair {
                lower: 1.0,
                upper: 0.0,
            });
        }

        let upper = regularized_gamma_q_fraction(a, x)?;
        Ok(TailPair {
            lower: 1.0 - upper,
            upper,
        })
    }
}

fn gamma_upper_is_below_precision(a: f64, x: f64) -> Result<bool, SpecialError> {
    let log_scale = -x + a * libm::log(x) - ln_gamma(a)?;
    if log_scale <= LN_MIN_POSITIVE {
        return Ok(true);
    }

    let denominator = (x + 1.0 - a).max(1.0);
    Ok(log_scale - libm::log(denominator) <= LN_EPSILON)
}

fn regularized_gamma_half_integer_pq(a: f64, x: f64) -> Result<Option<TailPair>, SpecialError> {
    if x < a || x >= GAMMA_HALF_INTEGER_MAX_X {
        return Ok(None);
    }

    let steps_f = a - 0.5;
    if steps_f < 0.0 || !same_float(steps_f, steps_f.round()) {
        return Ok(None);
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    // Half-integer gate bounds steps to a non-negative exact integer
    let steps = steps_f as usize;
    if steps > GAMMA_HALF_INTEGER_MAX_STEPS {
        return Ok(None);
    }

    let root_x = libm::sqrt(x);
    let mut upper = libm::erfc(root_x);
    let mut term = 2.0 * root_x * libm::exp(-x) / SQRT_PI;
    let mut shape = 0.5;

    for _ in 0..steps {
        upper += term;
        term *= x / (shape + 1.0);
        shape += 1.0;
    }

    let upper = clean_unit_probability("half-integer gamma upper tail", upper)?;
    let lower = clean_unit_probability("half-integer gamma lower tail", 1.0 - upper)?;
    Ok(Some(TailPair { lower, upper }))
}

fn regularized_gamma_p_series(a: f64, x: f64) -> Result<f64, SpecialError> {
    // DLMF 8.7.1 gives the lower incomplete-gamma power series. After
    // multiplying by exp(-x) x^a / Gamma(a), the summand recurrence is
    // x / (a + n) times the previous term.
    let mut shifted_shape = a;
    let mut term = 1.0 / shifted_shape;
    let mut series = term;

    for _iteration in 1..=GAMMA_MAX_ITERATIONS {
        shifted_shape += 1.0;
        term *= x / shifted_shape;
        series += term;
        if term.abs() <= series.abs() * GAMMA_EPSILON {
            let value = series * libm::exp(-x + a * libm::log(x) - ln_gamma(a)?);
            ensure_finite("regularized gamma lower series", value)?;
            return Ok(value);
        }
    }

    Err(SpecialError::MaxIterations {
        function: "regularized_gamma_p_series",
        iterations: GAMMA_MAX_ITERATIONS,
    })
}

fn regularized_gamma_q_fraction(a: f64, x: f64) -> Result<f64, SpecialError> {
    // DLMF 8.9.2 / A&S 6.5.31 express Gamma(a, x) as
    // exp(-x) x^a times the reciprocal of a continued fraction. The reciprocal
    // is evaluated with the modified Lentz algorithm (Lentz 1976; Thompson and
    // Barnett 1986 underflow guard).
    let fraction = modified_lentz_reciprocal(
        x + 1.0 - a,
        LentzProblem {
            max_terms: GAMMA_MAX_ITERATIONS,
            reported_iterations: GAMMA_MAX_ITERATIONS,
            epsilon: GAMMA_EPSILON,
            quantity_name: "regularized gamma upper fraction",
            function_name: "regularized_gamma_q_fraction",
            convergence_checkpoint: |_| true,
            numerator: |term_index| {
                let n = usize_to_f64(term_index);
                n * (a - n)
            },
            denominator: |term_index| x + usize_to_f64(2 * term_index + 1) - a,
        },
    )?;
    let value = libm::exp(-x + a * libm::log(x) - ln_gamma(a)?) * fraction;
    ensure_finite("regularized gamma upper fraction", value)?;
    Ok(value)
}

fn betacf(a: f64, b: f64, x: f64) -> Result<f64, SpecialError> {
    // DLMF 8.17.22 writes the regularized-beta factor as the reciprocal of
    // 1 + d_1/(1 + d_2/(1 + ...)), with d_{2m} and d_{2m+1} given in DLMF
    // 8.17.23. The continued fraction is evaluated by modified Lentz updates
    // (Lentz 1976; Thompson and Barnett 1986 underflow guard).
    let even_coefficient = |m: f64| m * (b - m) * x / ((a + 2.0 * m - 1.0) * (a + 2.0 * m));
    let odd_coefficient =
        |m: f64| -((a + m) * (a + b + m) * x) / ((a + 2.0 * m) * (a + 2.0 * m + 1.0));
    modified_lentz_reciprocal(
        1.0,
        LentzProblem {
            max_terms: 2 * BETACF_MAX_ITERATIONS + 1,
            reported_iterations: BETACF_MAX_ITERATIONS,
            epsilon: BETACF_EPSILON,
            quantity_name: "regularized beta continued fraction",
            function_name: "betacf",
            convergence_checkpoint: |term_index| term_index > 1 && term_index % 2 == 1,
            numerator: |term_index| {
                if term_index % 2 == 0 {
                    even_coefficient(usize_to_f64(term_index / 2))
                } else {
                    odd_coefficient(usize_to_f64((term_index - 1) / 2))
                }
            },
            denominator: |_| 1.0,
        },
    )
}

struct LentzProblem<N, D, C> {
    max_terms: usize,
    reported_iterations: usize,
    epsilon: f64,
    quantity_name: &'static str,
    function_name: &'static str,
    convergence_checkpoint: C,
    numerator: N,
    denominator: D,
}

fn modified_lentz_reciprocal<N, D, C>(
    initial_denominator: f64,
    problem: LentzProblem<N, D, C>,
) -> Result<f64, SpecialError>
where
    N: Fn(usize) -> f64,
    D: Fn(usize) -> f64,
    C: Fn(usize) -> bool,
{
    let LentzProblem {
        max_terms,
        reported_iterations,
        epsilon,
        quantity_name,
        function_name,
        convergence_checkpoint,
        numerator,
        denominator,
    } = problem;
    let mut backward_denominator_inverse = 1.0 / lentz_nonzero(initial_denominator);
    let mut forward_denominator = 1.0 / LENTZ_TINY;
    let mut reciprocal = backward_denominator_inverse;

    for term_index in 1..=max_terms {
        let partial_numerator = numerator(term_index);
        let partial_denominator = denominator(term_index);

        let next_backward_denominator =
            partial_denominator + partial_numerator * backward_denominator_inverse;
        backward_denominator_inverse = 1.0 / lentz_nonzero(next_backward_denominator);
        forward_denominator =
            lentz_nonzero(partial_denominator + partial_numerator / forward_denominator);

        let correction = forward_denominator * backward_denominator_inverse;
        reciprocal *= correction;
        ensure_finite(quantity_name, reciprocal)?;

        if convergence_checkpoint(term_index) && (correction - 1.0).abs() <= epsilon {
            return Ok(reciprocal);
        }
    }

    Err(SpecialError::MaxIterations {
        function: function_name,
        iterations: reported_iterations,
    })
}

fn lentz_nonzero(value: f64) -> f64 {
    if value.abs() < LENTZ_TINY {
        LENTZ_TINY
    } else {
        value
    }
}

fn betainc_inv_residual(p: f64, a: f64, b: f64, x: f64) -> Result<f64, SpecialError> {
    let cdf = betainc_reg_pair(a, b, x)?;
    if p <= 0.5 {
        Ok(cdf.lower - p)
    } else {
        Ok((1.0 - p) - cdf.upper)
    }
}

fn newton_candidate(a: f64, b: f64, x: f64, residual: f64) -> Option<f64> {
    let density = beta_density(a, b, x).ok()?;
    if density <= 0.0 {
        return None;
    }

    let first_order_step = residual / density;
    let slope = (a - 1.0) / x - (b - 1.0) / (1.0 - x);
    let denominator = 1.0 - 0.5 * first_order_step * slope;
    let step = if denominator.is_finite() && denominator > 0.0 {
        first_order_step / denominator
    } else {
        first_order_step
    };
    let candidate = x - step;
    candidate.is_finite().then_some(candidate)
}

fn beta_density(a: f64, b: f64, x: f64) -> Result<f64, SpecialError> {
    let log_density =
        (a - 1.0) * libm::log(x) + (b - 1.0) * libm::log1p(-x) + log_beta_normalizer(a, b)?;
    let density = libm::exp(log_density);
    ensure_finite("beta density", density)?;
    Ok(density)
}

fn log_beta_normalizer(a: f64, b: f64) -> Result<f64, SpecialError> {
    if a >= BETA_ASYMPTOTIC_NORMALIZER_MIN && b >= BETA_ASYMPTOTIC_NORMALIZER_MIN {
        Ok(log_beta_normalizer_asymptotic(a, b))
    } else if a <= b && b >= 8.0 {
        Ok(log_gamma_delta(b, a) - log_gamma_for_normalizer(a)?)
    } else if a >= 8.0 {
        Ok(log_gamma_delta(a, b) - log_gamma_for_normalizer(b)?)
    } else {
        Ok(ln_gamma(a + b)? - ln_gamma(a)? - ln_gamma(b)?)
    }
}

fn log_beta_normalizer_asymptotic(a: f64, b: f64) -> f64 {
    let sum = a + b;
    a * libm::log1p(b / a)
        + b * libm::log1p(a / b)
        + 0.5 * (libm::log(a) + libm::log(b) - libm::log(sum))
        - HALF_LN_TWO_PI
        + stirling_correction(sum)
        - stirling_correction(a)
        - stirling_correction(b)
}

fn log_gamma_for_normalizer(x: f64) -> Result<f64, SpecialError> {
    if x >= 8.0 {
        Ok(log_gamma_stirling(x))
    } else {
        ln_gamma(x)
    }
}

fn log_gamma_stirling(x: f64) -> f64 {
    (x - 0.5) * libm::log(x) - x + HALF_LN_TWO_PI + stirling_correction(x)
}

fn log_gamma_delta(base: f64, increment: f64) -> f64 {
    let ratio = increment / base;
    increment * libm::log(base)
        + (increment - 0.5) * libm::log1p(ratio)
        + base * log1pmx(ratio)
        + stirling_correction(base + increment)
        - stirling_correction(base)
}

fn log1pmx(x: f64) -> f64 {
    if x.abs() >= 0.1 {
        return libm::log1p(x) - x;
    }

    let mut term = x * x;
    let mut sum = -0.5 * term;
    for n in 3..=40 {
        term *= x;
        let signed = if n % 2 == 0 { -term } else { term };
        sum += signed / usize_to_f64(n);
    }
    sum
}

fn stirling_correction(x: f64) -> f64 {
    let inverse = 1.0 / x;
    let inverse2 = inverse * inverse;
    let inverse3 = inverse * inverse2;
    let inverse5 = inverse3 * inverse2;
    let inverse7 = inverse5 * inverse2;
    let inverse9 = inverse7 * inverse2;
    let inverse11 = inverse9 * inverse2;
    let inverse13 = inverse11 * inverse2;

    inverse / 12.0 - inverse3 / 360.0 + inverse5 / 1_260.0 - inverse7 / 1_680.0 + inverse9 / 1_188.0
        - 691.0 * inverse11 / 360_360.0
        + inverse13 / 156.0
}

fn initial_betainc_inv_guess(p: f64, a: f64, b: f64) -> f64 {
    if a >= 1.0 && b >= 1.0 {
        // A&S 26.2.23 supplies the rational approximation to the normal
        // quantile; A&S 26.5.22 maps that deviate to an incomplete-beta
        // quantile seed before the bracketed Newton/bisection solve.
        let smaller_tail = if p < 0.5 { p } else { 1.0 - p };
        let tail_radius = libm::sqrt(-2.0 * libm::log(smaller_tail));
        let normal_correction = (2.307_53 + tail_radius * 0.270_61)
            / (1.0 + tail_radius * (0.992_29 + tail_radius * 0.044_81));
        let mut normal_deviate = normal_correction - tail_radius;
        if p < 0.5 {
            normal_deviate = -normal_deviate;
        }
        let normal_curvature = (normal_deviate * normal_deviate - 3.0) / 6.0;
        let adjusted_shape = 2.0 / (1.0 / (2.0 * a - 1.0) + 1.0 / (2.0 * b - 1.0));
        let shape_skew = 1.0 / (2.0 * b - 1.0) - 1.0 / (2.0 * a - 1.0);
        let logit_half_shift = normal_deviate * libm::sqrt(normal_curvature + adjusted_shape)
            / adjusted_shape
            - shape_skew * (normal_curvature + 5.0 / 6.0 - 2.0 / (3.0 * adjusted_shape));
        interiorize(a / (a + b * libm::exp(2.0 * logit_half_shift)))
    } else {
        // A&S 26.5.22 also gives a small-shape branch based on the endpoint
        // weights of the two beta tails.
        let total_shape = a + b;
        let left_log_share = libm::log(a / total_shape);
        let right_log_share = libm::log(b / total_shape);
        let left_endpoint_weight = libm::exp(a * left_log_share) / a;
        let right_endpoint_weight = libm::exp(b * right_log_share) / b;
        let endpoint_weight_sum = left_endpoint_weight + right_endpoint_weight;
        let x = if p < left_endpoint_weight / endpoint_weight_sum {
            libm::exp(libm::log(a * endpoint_weight_sum * p) / a)
        } else {
            1.0 - libm::exp(libm::log(b * endpoint_weight_sum * (1.0 - p)) / b)
        };
        interiorize(x)
    }
}

fn ensure_positive(name: &'static str, value: f64) -> Result<(), SpecialError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(SpecialError::InvalidInput {
            message: format!("{name} must be positive and finite"),
        })
    }
}

fn ensure_unit_interval(name: &'static str, value: f64) -> Result<(), SpecialError> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(SpecialError::InvalidInput {
            message: format!("{name} must be finite and in [0, 1]"),
        })
    }
}

fn ensure_finite(name: &'static str, value: f64) -> Result<(), SpecialError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(SpecialError::NumericalIssue {
            message: format!("{name} was non-finite"),
        })
    }
}

fn clean_unit_probability(name: &'static str, value: f64) -> Result<f64, SpecialError> {
    ensure_finite(name, value)?;
    // Snap rounding excess just outside [0, 1] back to the boundary. The snap
    // width is the crate's documented 1e-12 accuracy contract: a saturating
    // series legitimately overshoots 1.0 by tens of ULPs (~2e-14 observed from
    // the 48-term BGRAT sum), which exceeds any ULP-scale tolerance, while
    // anything outside the contract width is a genuine numerical failure and
    // must surface as a typed error, never a silent clamp.
    let boundary_tolerance = 1.0e-12;
    if (0.0..=1.0).contains(&value) {
        Ok(value)
    } else if value < 0.0 && value > -boundary_tolerance {
        Ok(0.0)
    } else if value > 1.0 && value < 1.0 + boundary_tolerance {
        Ok(1.0)
    } else {
        Err(SpecialError::NumericalIssue {
            message: format!("{name} was outside [0, 1]"),
        })
    }
}

fn is_strictly_inside(x: f64, lo: f64, hi: f64) -> bool {
    x.is_finite() && x > lo && x < hi
}

fn midpoint(lo: f64, hi: f64) -> f64 {
    lo + 0.5 * (hi - lo)
}

fn interiorize(x: f64) -> f64 {
    if !x.is_finite() {
        0.5
    } else if x <= 0.0 {
        f64::MIN_POSITIVE
    } else if x >= 1.0 {
        1.0 - f64::EPSILON
    } else {
        x
    }
}

fn direct_quantile(x: f64) -> BetaQuantile {
    BetaQuantile {
        x,
        complement: 1.0 - x,
    }
}

fn bracket_is_representation_limited(lo: f64, hi: f64, x: f64) -> bool {
    hi - lo <= 2.0 * spacing_above(x).max(f64::MIN_POSITIVE)
}

fn spacing_above(x: f64) -> f64 {
    f64::from_bits(x.to_bits() + 1) - x
}

#[allow(clippy::cast_precision_loss)] // Iteration budgets are small and exactly representable as f64
fn usize_to_f64(value: usize) -> f64 {
    value as f64
}

#[cfg(test)]
mod tests {
    use super::{betainc_inv, betainc_reg, ln_gamma};

    #[test]
    fn ln_gamma_matches_factorial_case() {
        let value = ln_gamma(6.0).unwrap();
        assert!((value - libm::log(120.0)).abs() < 1.0e-14);
    }

    #[test]
    fn betainc_reg_uses_reflection_symmetry() {
        let a = 2.5;
        let b = 7.5;
        let x = 0.8;
        let left = betainc_reg(a, b, x).unwrap();
        let right = 1.0 - betainc_reg(b, a, 1.0 - x).unwrap();
        assert!((left - right).abs() < 1.0e-13);
    }

    #[test]
    fn betainc_inv_round_trips() {
        let quantile = betainc_inv(0.025, 10.5, 90.5).unwrap();
        let probability = betainc_reg(10.5, 90.5, quantile.x).unwrap();
        assert!((probability - 0.025).abs() < 1.0e-14);
    }
}
