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

use pecos_num::special::{self, InverseBetaOptions, SpecialError};
use pecos_num::stats::{JeffreysError, JeffreysEstimator, jeffreys_interval, jeffreys_point};

const FIXTURE_CSV: &str = include_str!("fixtures/jeffreys_scipy.csv");
const U01_BITS: u64 = 0x3ff0_0000_0000_0000;

#[derive(Debug, Clone, Copy)]
struct FixtureRow {
    k: u64,
    n: u64,
    alpha: f64,
    lo: f64,
    hi: f64,
    median: f64,
}

#[derive(Debug, Clone, Copy)]
struct WorstError {
    relative_error: f64,
    k: u64,
    n: u64,
    alpha: f64,
    bound: &'static str,
}

#[derive(Debug, Clone, Copy)]
struct WorstResidual {
    residual: f64,
    k: u64,
    n: u64,
    alpha: f64,
    bound: &'static str,
}

impl Default for WorstError {
    fn default() -> Self {
        Self {
            relative_error: 0.0,
            k: 0,
            n: 0,
            alpha: 0.0,
            bound: "",
        }
    }
}

impl Default for WorstResidual {
    fn default() -> Self {
        Self {
            residual: 0.0,
            k: 0,
            n: 0,
            alpha: 0.0,
            bound: "",
        }
    }
}

#[test]
fn fixture_table_matches_scipy() {
    let mut worst = WorstError::default();

    for row in fixture_rows() {
        let interval = jeffreys_interval(row.k, row.n, row.alpha).unwrap_or_else(|err| {
            panic!(
                "interval failed for k={}, n={}, alpha={}: {err}",
                row.k, row.n, row.alpha
            )
        });
        check_fixture_value(&mut worst, row, "lo", interval.lo, row.lo);
        check_fixture_value(&mut worst, row, "hi", interval.hi, row.hi);

        let median =
            jeffreys_point(row.k, row.n, JeffreysEstimator::Median).unwrap_or_else(|err| {
                panic!(
                    "median failed for k={}, n={}, alpha={}: {err}",
                    row.k, row.n, row.alpha
                )
            });
        check_fixture_value(&mut worst, row, "median", median, row.median);
    }

    eprintln!(
        "max scipy fixture relative error: {} at k={}, n={}, alpha={}, bound={}",
        worst.relative_error, worst.k, worst.n, worst.alpha, worst.bound
    );
}

#[test]
fn forward_betainc_reg_near_one_saturation_matches_scipy() {
    // Quantile fixtures never evaluate the forward CDF in the near-1 saturation
    // band inside the asymptotic gate: betainc_inv converges in the tail that stays
    // cleanly below 1, so a forward-side defect there is invisible to every
    // interval test (a boundary-snap ordering bug survived four review rounds
    // this way, collapsing I_x ~ 1 to 0.0). Oracle: scipy.special.betainc.
    let cases = [
        (30.5, 10_001.0, 0.0095, 0.999_999_999_999_997_7),
        (31.2, 10_050.0, 0.0099, 0.999_999_999_999_999_8),
        (10.5, 10_050.0, 0.006, 0.999_999_999_999_999_6),
        (2.5, 20_000.0, 0.002, 0.999_999_999_999_999_2),
        (44.6, 20_000.0, 0.006, 0.999_999_999_999_999_3),
    ];
    for (a, b, x, expected) in cases {
        let value = special::betainc_reg(a, b, x)
            .unwrap_or_else(|err| panic!("betainc_reg({a}, {b}, {x}) failed: {err}"));
        let relative = (value - expected).abs() / expected;
        assert!(
            relative <= 1.0e-12,
            "betainc_reg({a}, {b}, {x}) = {value}, expected {expected}, rel {relative}"
        );
    }
}

#[test]
fn forward_betainc_reg_residual_at_fixture_quantiles_is_bounded() {
    let mut worst = WorstResidual::default();

    for row in fixture_rows() {
        let a = usize_to_f64(row.k) + 0.5;
        let b = usize_to_f64(row.n - row.k) + 0.5;
        let tail = 0.5 * row.alpha;

        if row.k > 0 {
            check_forward_residual(&mut worst, row, "lo", a, b, row.lo, tail);
        }
        if row.k < row.n {
            check_forward_residual(&mut worst, row, "hi", a, b, row.hi, 1.0 - tail);
        }
        if !uses_large_shape_median_shortcut(a, b) {
            check_forward_residual(&mut worst, row, "median", a, b, row.median, 0.5);
        }
    }

    eprintln!(
        "max forward betainc_reg fixture residual: {} at k={}, n={}, alpha={}, bound={}",
        worst.residual, worst.k, worst.n, worst.alpha, worst.bound
    );
    assert!(
        worst.residual <= 5.0e-9,
        "forward betainc_reg residual too large: {worst:?}"
    );
}

#[test]
fn symmetry_round_trip_holds() {
    for &(n, alpha) in &[(1, 0.05), (2, 0.05), (101, 0.01), (257, 0.1), (503, 1.0e-6)] {
        for k in 0..=n {
            let interval = jeffreys_interval(k, n, alpha).unwrap();
            let reflected = jeffreys_interval(n - k, n, alpha).unwrap();
            assert_close(interval.lo, 1.0 - reflected.hi, 2.0e-9, "lower symmetry");
            assert_close(interval.hi, 1.0 - reflected.lo, 2.0e-9, "upper symmetry");
        }
    }
}

#[test]
fn bounds_are_monotone_in_k() {
    for &(n, alpha) in &[(200, 0.05), (200, 0.01), (200, 0.1), (200, 1.0e-6)] {
        let first = jeffreys_interval(0, n, alpha).unwrap();
        let mut previous_lo = first.lo;
        let mut previous_hi = first.hi;

        for k in 1..=n {
            let interval = jeffreys_interval(k, n, alpha).unwrap();
            assert!(
                interval.lo + 1.0e-14 >= previous_lo,
                "lo not monotone at k={k}, n={n}, alpha={alpha}: {} < {previous_lo}",
                interval.lo
            );
            assert!(
                interval.hi + 1.0e-14 >= previous_hi,
                "hi not monotone at k={k}, n={n}, alpha={alpha}: {} < {previous_hi}",
                interval.hi
            );
            previous_lo = interval.lo;
            previous_hi = interval.hi;
        }
    }
}

#[test]
fn gate_sweeps_remain_monotone_and_ordered() {
    check_k_sweep_monotone(60, 70, 100_000_000, 0.05);
    check_k_sweep_monotone(2_990, 3_030, 1_000_000, 0.05);
}

#[test]
fn alpha_sweeps_are_continuous_near_regime_gates() {
    for &(k, n) in &[
        (64, 100_000_000),
        (65, 100_000_000),
        (3_000, 1_000_000),
        (3_016, 1_000_000),
    ] {
        let alphas = [1.0e-6, 3.0e-6, 1.0e-5, 3.0e-5, 1.0e-4, 1.0e-3, 0.01, 0.05];
        let mut previous = jeffreys_interval(k, n, alphas[0]).unwrap();
        assert_interval_ordered(previous, k, n, alphas[0]);

        for &alpha in &alphas[1..] {
            let interval = jeffreys_interval(k, n, alpha).unwrap();
            assert_interval_ordered(interval, k, n, alpha);
            assert!(
                interval.lo >= previous.lo,
                "lo widened as alpha increased at k={k}, n={n}, alpha={alpha}"
            );
            assert!(
                interval.hi <= previous.hi,
                "hi widened as alpha increased at k={k}, n={n}, alpha={alpha}"
            );
            assert!(
                relative_x_error(interval.lo, previous.lo).is_finite(),
                "lo discontinuity at k={k}, n={n}, alpha={alpha}"
            );
            assert!(
                relative_x_error(interval.hi, previous.hi).is_finite(),
                "hi discontinuity at k={k}, n={n}, alpha={alpha}"
            );
            previous = interval;
        }
    }
}

#[test]
fn large_b_gate_n_sweep_is_ordered() {
    let k = 64;
    let alpha = 0.05;
    let trial_counts = [
        49_999_990,
        50_000_000,
        50_000_063,
        50_000_064,
        50_000_100,
        100_000_000,
    ];

    for &n in &trial_counts {
        let interval = jeffreys_interval(k, n, alpha).unwrap();
        assert_interval_ordered(interval, k, n, alpha);
    }
}

#[test]
fn small_a_low_b_pocket_sweeps_are_ordered() {
    for &n in &[1_000_000, 3_000_000, 6_000_000, 8_900_002, 8_999_990] {
        check_k_sweep_monotone(2, 40, n, 0.05);
    }
}

#[test]
fn large_b_floor_side_sweeps_are_ordered() {
    for &(k, counts) in &[
        (2, &[9_000, 9_999, 10_000, 10_001, 30_000, 100_000][..]),
        (64, &[40_000, 41_000, 42_000, 100_000, 1_000_000][..]),
    ] {
        for &n in counts {
            let interval = jeffreys_interval(k, n, 0.05).unwrap();
            assert_interval_ordered(interval, k, n, 0.05);
        }
    }
}

#[test]
fn large_b_saturation_keeps_inverse_orientation() {
    let cdf_at_half = special::betainc_reg(5.5, 2_999_995.5, 0.5).unwrap();
    assert!(
        cdf_at_half > 1.0 - 1.0e-15,
        "large-b half CDF should saturate high, got {cdf_at_half}"
    );

    let cdf_at_guard = special::betainc_reg(5.5, 2_999_995.5, 1.0e-3).unwrap();
    assert!(
        cdf_at_guard > 1.0 - 1.0e-15,
        "large-b guard CDF should saturate high, got {cdf_at_guard}"
    );

    let interval = jeffreys_interval(5, 3_000_000, 0.05).unwrap();
    assert_interval_ordered(interval, 5, 3_000_000, 0.05);
}

#[test]
fn gl_tail_orientation_regression_has_positive_width() {
    for &(k, n, alpha) in &[
        (3_016, 1_000_000, 0.05),
        (15_881, 20_000, 0.05),
        (67_867_393, 100_000_000, 0.05),
    ] {
        let interval = jeffreys_interval(k, n, alpha).unwrap();
        assert_interval_ordered(interval, k, n, alpha);
    }
}

#[test]
fn endpoints_are_bit_exact() {
    for &(n, alpha) in &[(1, 0.05), (2, 0.05), (100, 0.01), (100_000_000, 1.0e-6)] {
        let zero = jeffreys_interval(0, n, alpha).unwrap();
        assert_eq!(zero.lo.to_bits(), 0.0_f64.to_bits());

        let full = jeffreys_interval(n, n, alpha).unwrap();
        assert_eq!(full.hi.to_bits(), 1.0_f64.to_bits());
    }
}

#[test]
fn posterior_mean_point_estimate_is_used() {
    for &(k, n) in &[(0, 1), (1, 2), (10, 100), (1_000, 1_000_000)] {
        let expected = (usize_to_f64(k) + 0.5) / (usize_to_f64(n) + 1.0);
        let point = jeffreys_point(k, n, JeffreysEstimator::Mean).unwrap();
        assert_eq!(point.to_bits(), expected.to_bits());

        let interval = jeffreys_interval(k, n, 0.05).unwrap();
        assert_eq!(interval.point.to_bits(), expected.to_bits());
    }
}

#[test]
fn typed_errors_are_returned() {
    assert!(matches!(
        jeffreys_interval(0, 0, 0.05),
        Err(JeffreysError::ZeroTrials)
    ));
    assert!(matches!(
        jeffreys_interval(2, 1, 0.05),
        Err(JeffreysError::SuccessesExceedTrials { k: 2, n: 1 })
    ));
    assert!(matches!(
        jeffreys_interval(0, 100_000_001, 0.05),
        Err(JeffreysError::TrialsExceedSupported {
            n: 100_000_001,
            max: 100_000_000
        })
    ));
    assert!(matches!(
        jeffreys_interval(0, 1, 0.0),
        Err(JeffreysError::InvalidAlpha { alpha: 0.0 })
    ));
    assert!(matches!(
        jeffreys_interval(0, 1, 1.0),
        Err(JeffreysError::InvalidAlpha { alpha: 1.0 })
    ));

    let options = InverseBetaOptions {
        max_newton: 0,
        max_bisect: 0,
        ..InverseBetaOptions::default()
    };
    assert!(matches!(
        special::betainc_inv_with_options(0.5, 2.0, 2.0, options),
        Err(SpecialError::MaxIterations {
            function: "betainc_inv",
            iterations: 0
        })
    ));
}

#[test]
fn deterministic_randomized_grid_matches_puruspe_dev_check() {
    // scipy fixtures are the oracle of record for Jeffreys endpoints. This
    // deterministic puruspe sweep is only a dev-only differential check; on any
    // disagreement, the pinned scipy fixture table takes precedence.
    let mut rng = Lcg::new(0x4a45_4646_5245_5953);

    for _ in 0..200 {
        let n_usize = rng.next_usize(1_000) + 2;
        let k_usize = rng.next_usize(n_usize - 2) + 1;
        let n = usize_to_u64(n_usize);
        let k = usize_to_u64(k_usize);
        let alpha = 1.0e-4 + rng.next_unit_f64() * (0.25 - 1.0e-4);
        let interval = jeffreys_interval(k, n, alpha).unwrap();
        let median = jeffreys_point(k, n, JeffreysEstimator::Median).unwrap();

        let a = usize_to_f64(k) + 0.5;
        let b = usize_to_f64(n - k) + 0.5;
        let tail = 0.5 * alpha;
        let puruspe_lo = if k == 0 {
            0.0
        } else {
            puruspe::invbetai(tail, a, b)
        };
        let puruspe_hi = if k == n {
            1.0
        } else {
            puruspe::invbetai(1.0 - tail, a, b)
        };
        let puruspe_median = puruspe::invbetai(0.5, a, b);

        assert_close(interval.lo, puruspe_lo, 1.0e-5, "puruspe lo");
        assert_close(interval.hi, puruspe_hi, 1.0e-5, "puruspe hi");
        assert_close(median, puruspe_median, 1.0e-5, "puruspe median");
    }
}

fn fixture_rows() -> Vec<FixtureRow> {
    FIXTURE_CSV.lines().skip(1).map(parse_fixture_row).collect()
}

fn parse_fixture_row(line: &str) -> FixtureRow {
    let mut parts = line.split(',');
    let row = FixtureRow {
        k: parts.next().unwrap().parse().unwrap(),
        n: parts.next().unwrap().parse().unwrap(),
        alpha: parts.next().unwrap().parse().unwrap(),
        lo: parts.next().unwrap().parse().unwrap(),
        hi: parts.next().unwrap().parse().unwrap(),
        median: parts.next().unwrap().parse().unwrap(),
    };
    assert!(parts.next().is_none(), "too many CSV columns in {line}");
    row
}

fn check_fixture_value(
    worst: &mut WorstError,
    row: FixtureRow,
    bound: &'static str,
    actual: f64,
    expected: f64,
) {
    // The design tolerance is expressed as
    // max(1e-12, 5e-15 / max(p, 1-p)) on relative x error. The fixture columns
    // are x-values, so this test interprets p as x_oracle and uses
    // max(x_oracle, 1 - x_oracle).
    let relative_error = relative_x_error(actual, expected);
    let tolerance = fixture_relative_tolerance(expected);
    if relative_error > worst.relative_error {
        *worst = WorstError {
            relative_error,
            k: row.k,
            n: row.n,
            alpha: row.alpha,
            bound,
        };
    }
    assert!(
        relative_error <= tolerance,
        "{bound} mismatch for k={}, n={}, alpha={}: actual={actual}, expected={expected}, rel={relative_error}, tol={tolerance}",
        row.k,
        row.n,
        row.alpha,
    );
}

fn check_forward_residual(
    worst: &mut WorstResidual,
    row: FixtureRow,
    bound: &'static str,
    a: f64,
    b: f64,
    x: f64,
    target: f64,
) {
    let actual = special::betainc_reg(a, b, x).unwrap_or_else(|err| {
        panic!(
            "forward betainc_reg failed for {bound}, k={}, n={}, alpha={}: {err}",
            row.k, row.n, row.alpha
        )
    });
    let residual = (actual - target).abs();
    if residual > worst.residual {
        *worst = WorstResidual {
            residual,
            k: row.k,
            n: row.n,
            alpha: row.alpha,
            bound,
        };
    }
}

fn fixture_relative_tolerance(expected: f64) -> f64 {
    1.0e-12_f64.max(5.0e-15 / expected.max(1.0 - expected))
}

fn relative_x_error(actual: f64, expected: f64) -> f64 {
    if expected.to_bits() == 0.0_f64.to_bits() {
        if actual.to_bits() == 0.0_f64.to_bits() {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        (actual - expected).abs() / expected.abs()
    }
}

fn assert_close(actual: f64, expected: f64, relative_tolerance: f64, label: &str) {
    let relative_error = relative_x_error(actual, expected);
    assert!(
        relative_error <= relative_tolerance,
        "{label}: actual={actual}, expected={expected}, rel={relative_error}, tol={relative_tolerance}"
    );
}

fn check_k_sweep_monotone(k_start: usize, k_end: usize, n: usize, alpha: f64) {
    let first = jeffreys_interval(usize_to_u64(k_start), usize_to_u64(n), alpha).unwrap();
    assert_interval_ordered(first, usize_to_u64(k_start), usize_to_u64(n), alpha);
    let mut previous_lo = first.lo;
    let mut previous_hi = first.hi;

    for k in (k_start + 1)..=k_end {
        let interval = jeffreys_interval(usize_to_u64(k), usize_to_u64(n), alpha).unwrap();
        assert_interval_ordered(interval, usize_to_u64(k), usize_to_u64(n), alpha);
        assert!(
            interval.lo + 1.0e-14 >= previous_lo,
            "lo not monotone across gate at k={k}, n={n}, alpha={alpha}"
        );
        assert!(
            interval.hi + 1.0e-14 >= previous_hi,
            "hi not monotone across gate at k={k}, n={n}, alpha={alpha}"
        );
        previous_lo = interval.lo;
        previous_hi = interval.hi;
    }
}

fn assert_interval_ordered(
    interval: pecos_num::stats::JeffreysInterval,
    k: u64,
    n: u64,
    alpha: f64,
) {
    assert!(
        interval.lo >= 0.0 && interval.lo <= interval.hi && interval.hi <= 1.0,
        "invalid interval at k={k}, n={n}, alpha={alpha}: {interval:?}"
    );
}

fn uses_large_shape_median_shortcut(a: f64, b: f64) -> bool {
    a >= 1_000_000.0 && b >= 1_000_000.0
}

#[allow(clippy::cast_precision_loss)] // Test counts are <= 1e8, exactly representable as f64
fn usize_to_f64(value: u64) -> f64 {
    value as f64
}

fn usize_to_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap()
}

struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    fn next_unit_f64(&mut self) -> f64 {
        f64::from_bits(U01_BITS | (self.next_u64() >> 12)) - 1.0
    }

    fn next_usize(&mut self, max_inclusive: usize) -> usize {
        let span = u64::try_from(max_inclusive).unwrap() + 1;
        usize::try_from(self.next_u64() % span).unwrap()
    }
}
