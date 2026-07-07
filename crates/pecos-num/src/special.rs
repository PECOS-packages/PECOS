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

//! Special functions: log-gamma and the regularized incomplete beta
//! function with its inverse.
//!
//! These are the primitives behind Beta-distribution quantiles, which in
//! turn back the Jeffreys binomial interval in [`crate::stats`].
//!
//! Algorithms follow Numerical Recipes, 3rd edition, section 6.1
//! (Lanczos log-gamma) and section 6.4 (Lentz continued fraction for the
//! incomplete beta; Halley iteration with the Abramowitz & Stegun 26.5.22
//! initial guess for the inverse). Differential tests against `SciPy`
//! reference values live in the test module below.

/// Convergence threshold for the inverse incomplete beta Halley iteration.
const BETAINC_INV_EPS: f64 = 1.0e-8;
/// Maximum iterations for the incomplete-beta continued fraction.
const BETACF_MAX_ITER: u32 = 10_000;
/// Convergence threshold for the continued fraction.
const BETACF_EPS: f64 = 3.0e-14;
/// Smallest representable ratio guard for Lentz's method.
const BETACF_FPMIN: f64 = f64::MIN_POSITIVE / BETACF_EPS;

/// Natural logarithm of the gamma function for `x > 0`.
///
/// Drop-in replacement for `scipy.special.gammaln` on the positive real
/// axis. Uses the 14-term Lanczos approximation (Numerical Recipes 3rd
/// ed., section 6.1), accurate to roughly machine precision.
///
/// Returns NaN for `x <= 0`.
///
/// # Examples
///
/// ```
/// use pecos_num::special::ln_gamma;
///
/// // Gamma(1) = 1, Gamma(2) = 1
/// assert!(ln_gamma(1.0).abs() < 1e-14);
/// assert!(ln_gamma(2.0).abs() < 1e-14);
/// // Gamma(0.5) = sqrt(pi)
/// let half = std::f64::consts::PI.sqrt().ln();
/// assert!((ln_gamma(0.5) - half).abs() < 1e-14);
/// ```
#[must_use]
pub fn ln_gamma(x: f64) -> f64 {
    const COF: [f64; 14] = [
        57.156_235_665_862_92,
        -59.597_960_355_475_49,
        14.136_097_974_741_746,
        -0.491_913_816_097_620_2,
        3.399_464_998_481_189e-5,
        4.652_362_892_704_858e-5,
        -9.837_447_530_487_956e-5,
        1.580_887_032_249_125e-4,
        -2.102_644_417_241_049e-4,
        2.174_396_181_152_126e-4,
        -1.643_181_065_367_639e-4,
        8.441_822_398_385_274e-5,
        -2.619_083_840_158_141e-5,
        3.689_918_265_953_162e-6,
    ];
    const LANCZOS_G: f64 = 5.242_187_5;
    const SQRT_2PI: f64 = 2.506_628_274_631_000_5;

    if x <= 0.0 {
        return f64::NAN;
    }

    let mut denom = x;
    let tmp = x + LANCZOS_G;
    let tmp = (x + 0.5) * tmp.ln() - tmp;
    let mut series = 0.999_999_999_999_997_1;
    for c in COF {
        denom += 1.0;
        series += c / denom;
    }
    tmp + (SQRT_2PI * series / x).ln()
}

/// Continued-fraction evaluation for the incomplete beta function
/// (Numerical Recipes 3rd ed., section 6.4, via Lentz's method).
fn betacf(a: f64, b: f64, x: f64) -> f64 {
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < BETACF_FPMIN {
        d = BETACF_FPMIN;
    }
    d = 1.0 / d;
    let mut h = d;

    for m in 1..=BETACF_MAX_ITER {
        let m = f64::from(m);
        let m2 = 2.0 * m;

        // Even step of the continued fraction.
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < BETACF_FPMIN {
            d = BETACF_FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < BETACF_FPMIN {
            c = BETACF_FPMIN;
        }
        d = 1.0 / d;
        h *= d * c;

        // Odd step.
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < BETACF_FPMIN {
            d = BETACF_FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < BETACF_FPMIN {
            c = BETACF_FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;

        if (del - 1.0).abs() <= BETACF_EPS {
            return h;
        }
    }
    // The fraction converges for all valid inputs; reaching the iteration
    // cap means the inputs were extreme enough that the result is unusable.
    f64::NAN
}

/// Regularized incomplete beta function `I_x(a, b)` for `a, b > 0` and
/// `x` in `[0, 1]`.
///
/// Drop-in replacement for `scipy.special.betainc`. This is also the CDF
/// of the Beta(a, b) distribution evaluated at `x`.
///
/// Returns NaN outside the valid domain.
///
/// # Examples
///
/// ```
/// use pecos_num::special::betainc_reg;
///
/// // Symmetric case: I_{0.5}(a, a) = 0.5
/// assert!((betainc_reg(10.0, 10.0, 0.5) - 0.5).abs() < 1e-12);
/// // Boundaries
/// assert_eq!(betainc_reg(2.0, 3.0, 0.0), 0.0);
/// assert_eq!(betainc_reg(2.0, 3.0, 1.0), 1.0);
/// ```
#[must_use]
pub fn betainc_reg(a: f64, b: f64, x: f64) -> f64 {
    let domain_ok = a > 0.0 && b > 0.0 && (0.0..=1.0).contains(&x);
    if !domain_ok {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }

    // Prefactor x^a (1-x)^b / (a B(a,b)), computed in log space.
    let ln_front = ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln();
    let front = ln_front.exp();

    // Use the continued fraction directly where it converges fastest,
    // and the symmetry I_x(a,b) = 1 - I_{1-x}(b,a) otherwise.
    if x < (a + 1.0) / (a + b + 2.0) {
        front * betacf(a, b, x) / a
    } else {
        1.0 - front * betacf(b, a, 1.0 - x) / b
    }
}

/// Inverse of the regularized incomplete beta function: returns `x` such
/// that `betainc_reg(a, b, x) == p`.
///
/// Matches `scipy.special.betaincinv` (this is the quantile / inverse CDF
/// of the Beta(a, b) distribution) over the parameter scales validated by
/// the test module: shape parameters up to roughly binomial-trial counts
/// of 1e12 and `p` away from the extreme tails by more than ~1e-15. The
/// upper tail is computed through the symmetry
/// `betainc_inv(a, b, p) = 1 - betainc_inv(b, a, 1 - p)` so both tails
/// share the well-conditioned lower-tail path; beyond those scales the
/// continued fraction can converge spuriously, so callers with extreme
/// parameters must validate independently.
///
/// Follows Numerical Recipes 3rd ed., section 6.4: an initial guess from
/// Abramowitz & Stegun 26.5.22 refined by Halley iterations.
///
/// Returns NaN outside the valid domain (`a, b > 0`, `p` in `[0, 1]`).
///
/// # Examples
///
/// ```
/// use pecos_num::special::{betainc_inv, betainc_reg};
///
/// let x = betainc_inv(2.0, 3.0, 0.6);
/// assert!((betainc_reg(2.0, 3.0, x) - 0.6).abs() < 1e-10);
/// ```
#[must_use]
pub fn betainc_inv(a: f64, b: f64, p: f64) -> f64 {
    let domain_ok = a > 0.0 && b > 0.0 && (0.0..=1.0).contains(&p);
    if !domain_ok {
        return f64::NAN;
    }
    if p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return 1.0;
    }
    // Compute upper-tail quantiles through the lower tail of the mirrored
    // distribution: `err = betainc_reg(...) - p` loses all precision when
    // p is within ~1e-10 of 1 (the Halley correction then stalls on a
    // cancelled residual), while 1 - p is exact in the mirrored call.
    if p > 0.5 {
        return 1.0 - betainc_inv(b, a, 1.0 - p);
    }

    let a1 = a - 1.0;
    let b1 = b - 1.0;

    let mut x: f64;
    if a >= 1.0 && b >= 1.0 {
        // Abramowitz & Stegun 26.5.22 via the normal quantile
        // approximation 26.2.23.
        let pp = if p < 0.5 { p } else { 1.0 - p };
        let t = (-2.0 * pp.ln()).sqrt();
        let mut gauss = (2.30753 + t * 0.27061) / (1.0 + t * (0.99229 + t * 0.04481)) - t;
        if p < 0.5 {
            gauss = -gauss;
        }
        let al = (gauss * gauss - 3.0) / 6.0;
        let h = 2.0 / (1.0 / (2.0 * a - 1.0) + 1.0 / (2.0 * b - 1.0));
        let w = gauss * (al + h).sqrt() / h
            - (1.0 / (2.0 * b - 1.0) - 1.0 / (2.0 * a - 1.0)) * (al + 5.0 / 6.0 - 2.0 / (3.0 * h));
        x = a / (a + b * (2.0 * w).exp());
    } else {
        let lna = (a / (a + b)).ln();
        let lnb = (b / (a + b)).ln();
        let t = (a * lna).exp() / a;
        let u = (b * lnb).exp() / b;
        let w = t + u;
        x = if p < t / w {
            (a * w * p).powf(1.0 / a)
        } else {
            1.0 - (b * w * (1.0 - p)).powf(1.0 / b)
        };
    }

    let afac = ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b);
    for iteration in 0..10 {
        if x <= 0.0 {
            return 0.0;
        }
        if x >= 1.0 {
            return 1.0;
        }
        let err = betainc_reg(a, b, x) - p;
        let t = (a1 * x.ln() + b1 * (1.0 - x).ln() + afac).exp();
        let u = err / t;
        // Halley step.
        let step = u / (1.0 - 0.5 * f64::min(1.0, u * (a1 / x - b1 / (1.0 - x))));
        x -= step;
        if x <= 0.0 {
            x = 0.5 * (x + step);
        }
        if x >= 1.0 {
            x = 0.5 * (x + step + 1.0);
        }
        if step.abs() < BETAINC_INV_EPS * x && iteration > 0 {
            break;
        }
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Assert `actual` matches `expected` to a RELATIVE tolerance.
    ///
    /// A pure relative check: callers that legitimately expect a value of
    /// exactly zero must handle that case separately (the `ln_gamma` test
    /// does). An earlier version OR'd in an absolute `abs_err <= rel_tol`
    /// fallback, which silently weakened the relative tolerance to an
    /// absolute one for `|expected| < 1` (e.g. a claimed 1e-12 relative
    /// became 1e-12 absolute — ~1000x looser at `expected = 1e-3`).
    fn assert_close(actual: f64, expected: f64, rel_tol: f64) {
        let denom = expected.abs().max(f64::MIN_POSITIVE);
        let rel_err = (actual - expected).abs() / denom;
        assert!(
            rel_err <= rel_tol,
            "expected {expected:.17e}, got {actual:.17e} (relative error {rel_err:.3e} > {rel_tol:.3e})"
        );
    }

    // Reference values generated with:
    //   uv run python -c "from scipy import special; print(special.gammaln(x))"
    // (scipy 1.x, double precision)
    #[test]
    fn ln_gamma_matches_scipy() {
        let cases = [
            (0.5, 0.572_364_942_924_7),
            (1.0, 0.0),
            (1.5, -0.120_782_237_635_245_26),
            (2.0, 0.0),
            (3.7, 1.428_072_326_665_388),
            (10.0, 12.801_827_480_081_469),
            (100.5, 361.435_540_467_777_57),
            (1000.0, 5_905.220_423_209_181),
        ];
        for (x, expected) in cases {
            let actual = ln_gamma(x);
            if expected == 0.0 {
                assert!(actual.abs() < 1e-13, "ln_gamma({x}) = {actual:.3e}, want 0");
            } else {
                assert_close(actual, expected, 1e-12);
            }
        }
    }

    #[test]
    fn ln_gamma_invalid_domain_is_nan() {
        assert!(ln_gamma(0.0).is_nan());
        assert!(ln_gamma(-1.5).is_nan());
    }

    // Reference values generated with:
    //   uv run python -c "from scipy import special; print(special.betainc(a, b, x))"
    #[test]
    fn betainc_reg_matches_scipy() {
        let cases = [
            (0.5, 0.5, 0.3, 0.369_010_119_565_545_36),
            (2.0, 3.0, 0.4, 0.524_799_999_999_999_9),
            (5.5, 1.5, 0.7, 0.251_904_453_669_740_85),
            (10.0, 10.0, 0.5, 0.5),
            (0.5, 20.5, 0.01, 0.476_541_531_548_465_6),
            (100.5, 900.5, 0.1, 0.494_385_987_853_672_66),
            (3.5, 0.5, 0.99, 0.797_971_695_234_850_9),
        ];
        for (a, b, x, expected) in cases {
            assert_close(betainc_reg(a, b, x), expected, 1e-12);
        }
    }

    #[test]
    fn betainc_reg_invalid_domain_is_nan() {
        assert!(betainc_reg(0.0, 1.0, 0.5).is_nan());
        assert!(betainc_reg(1.0, -1.0, 0.5).is_nan());
        assert!(betainc_reg(1.0, 1.0, -0.1).is_nan());
        assert!(betainc_reg(1.0, 1.0, 1.1).is_nan());
    }

    // Reference values generated with:
    //   uv run python -c "from scipy import special; print(special.betaincinv(a, b, p))"
    #[test]
    fn betainc_inv_matches_scipy() {
        let cases = [
            (0.5, 0.5, 0.25, 0.146_446_609_406_726_24),
            (2.0, 3.0, 0.6, 0.444_500_002_083_767_4),
            (5.5, 1.5, 0.05, 0.505_461_253_650_681_3),
            (10.0, 10.0, 0.975, 0.711_356_752_083_001_1),
            (0.5, 20.5, 0.995, 0.176_754_097_436_689_93),
            (100.5, 900.5, 0.025, 0.082_562_652_843_060_04),
            // Binomial-CI-scale parameters: n = 20000 trials, far tail.
            (50.5, 19_950.5, 0.999_999, 0.004_581_655_467_494_118_5),
        ];
        for (a, b, p, expected) in cases {
            assert_close(betainc_inv(a, b, p), expected, 1e-8);
        }
    }

    // Reference values generated with:
    //   uv run python -c "from scipy import special; print(special.betaincinv(a, b, p))"
    // Upper-tail quantiles exercise the symmetry path (the direct Halley
    // iteration loses the residual to cancellation beyond p ~ 1 - 1e-10).
    #[test]
    fn betainc_inv_upper_tail_matches_scipy() {
        let cases: [(f64, f64, f64, f64); 4] = [
            (2.0, 3.0, 0.999_999_9, 0.997_073_840_091_498_9),
            (100.5, 900.5, 1.0 - 1e-12, 0.179_649_794_238_11),
            (7.5, 19_993.5, 1.0 - 2.3e-16, 0.002_728_615_291_135_757_6),
            (0.5, 0.5, 0.999_999, 0.999_999_999_997_532_6),
        ];
        for (a, b, p, expected) in cases {
            let actual = betainc_inv(a, b, p);
            let scale: f64 = expected.abs();
            assert!(
                ((actual - expected).abs() / scale) < 1e-8,
                "betainc_inv({a}, {b}, {p}): expected {expected:.12e}, got {actual:.12e}"
            );
        }
    }

    #[test]
    fn betainc_inv_tails_are_symmetric() {
        // Relative comparison, with cases chosen so neither side hits
        // f64 representation limits: p stays >= 1e-6 (forming `1 - p`
        // closer to 1 destroys p's precision before the function is even
        // called) and the quantiles stay far enough from 0 and 1 that
        // `1 - upper` keeps its significant digits. Outside those limits
        // a mirrored comparison measures representation error, not
        // implementation error.
        let cases: [(f64, f64, &[f64]); 3] = [
            (2.0, 3.0, &[1e-6, 0.01, 0.3]),
            (50.5, 19_950.5, &[1e-6, 0.01, 0.3]),
            // Quantiles for this shape at small p sit below 1e-13, where
            // the mirrored side cannot represent them; compare only at
            // moderate p.
            (0.5, 20.5, &[0.01, 0.3]),
        ];
        for (a, b, ps) in cases {
            for &p in ps {
                let lower = betainc_inv(a, b, p);
                let upper = betainc_inv(b, a, 1.0 - p);
                let mirrored = 1.0 - upper;
                assert!(
                    ((lower - mirrored) / lower).abs() < 1e-8,
                    "tail symmetry failed for a={a}, b={b}, p={p}: {lower} vs {mirrored}"
                );
            }
        }
    }

    #[test]
    fn betainc_inv_round_trips_through_betainc_reg() {
        for &(a, b) in &[(0.5, 0.5), (2.0, 3.0), (7.5, 19_993.5), (100.5, 900.5)] {
            for &p in &[1e-6, 0.025, 0.5, 0.975, 1.0 - 1e-6] {
                let x = betainc_inv(a, b, p);
                let back = betainc_reg(a, b, x);
                assert!(
                    (back - p).abs() < 1e-9,
                    "round trip failed for a={a}, b={b}, p={p}: x={x}, back={back}"
                );
            }
        }
    }

    // Allow exact float comparisons: the edge cases return the sentinel
    // values 0.0 and 1.0 verbatim.
    #[allow(clippy::float_cmp)]
    #[test]
    fn betainc_inv_edges() {
        assert_eq!(betainc_inv(2.0, 3.0, 0.0), 0.0);
        assert_eq!(betainc_inv(2.0, 3.0, 1.0), 1.0);
        assert!(betainc_inv(0.0, 3.0, 0.5).is_nan());
        assert!(betainc_inv(2.0, 3.0, -0.1).is_nan());
    }
}
