// Copyright 2025 The PECOS Developers
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

//! Statistical functions for numerical analysis.
//!
//! Drop-in replacements for numpy/scipy statistical functions.
//!
//! # Functions
//!
//! ## 1D Slice Operations (Simple API)
//! - [`mean`] - Calculate mean of a 1D slice
//! - [`std`] - Calculate standard deviation of a 1D slice
//!
//! ## nD Array Operations (Idiomatic ndarray API)
//! - [`mean_axis`] - Calculate mean along an axis of an ndarray
//! - [`std_axis`] - Calculate standard deviation along an axis of an ndarray
//!
//! ## Resampling Methods
//! - [`jackknife_resamples`] - Generate leave-one-out resamples from data
//! - [`jackknife_stats`] - Compute jackknife mean and standard error from 1D estimates
//! - [`jackknife_stats_axis`] - Compute jackknife mean and standard error along axis of 2D array
//! - [`jackknife_weighted`] - Jackknife resampling for weighted/grouped data (full workflow)
//! - [`weighted_mean`] - Calculate weighted mean from (value, weight) pairs
//!
//! ## Binomial Proportions
//! - [`jeffreys_interval`] - Jeffreys credible interval for a binomial proportion
//!
//! The slice functions are fast and simple for 1D data. The axis functions
//! provide idiomatic Rust API for multi-dimensional arrays.

use crate::special::betainc_inv;
use ndarray::{Array, ArrayView, Axis, Dimension, RemoveAxis};

/// Jeffreys credible interval for a binomial proportion.
///
/// Computes the equal-tailed interval of the Beta(k + 1/2, n - k + 1/2)
/// posterior arising from the Jeffreys prior Beta(1/2, 1/2), following
/// Brown, Cai & `DasGupta`, "Interval Estimation for a Binomial
/// Proportion", Statistical Science 16(2), 2001. Per that paper's
/// standard modification, the lower bound is 0 when `successes == 0` and
/// the upper bound is 1 when `successes == trials`.
///
/// # Arguments
///
/// * `successes` - Number of observed successes (k)
/// * `trials` - Number of trials (n), must be nonzero
/// * `confidence` - Interval mass, e.g. 0.95; must be in (0, 1)
///
/// # Returns
///
/// `(lower, upper)` bounds on the proportion.
///
/// # Panics
///
/// Panics if `trials == 0`, `trials > 10^12`, `successes > trials`, or
/// `confidence` is not in (0, 1). The first three are contract
/// violations; the trials cap marks the scale beyond which the
/// underlying incomplete-beta continued fraction has not been validated
/// (it can converge spuriously for extreme shape parameters, returning a
/// nonsense interval without warning). Also panics if the computed
/// bounds come back inverted — a numeric-breakdown trip-wire that should
/// be unreachable within the supported scales.
///
/// # Examples
///
/// ```
/// use pecos_num::stats::jeffreys_interval;
///
/// let (lo, hi) = jeffreys_interval(50, 200, 0.95);
/// assert!(lo < 0.25 && 0.25 < hi);
///
/// // Zero successes: lower bound is exactly 0.
/// let (lo, hi) = jeffreys_interval(0, 100, 0.95);
/// assert_eq!(lo, 0.0);
/// assert!(hi < 0.05);
/// ```
#[must_use]
#[allow(clippy::cast_precision_loss)]
// Cast is safe: the trials cap keeps counts far below f64 mantissa precision
pub fn jeffreys_interval(successes: u64, trials: u64, confidence: f64) -> (f64, f64) {
    const MAX_TRIALS: u64 = 1_000_000_000_000;

    assert!(trials > 0, "jeffreys_interval requires trials > 0");
    assert!(
        trials <= MAX_TRIALS,
        "jeffreys_interval supports at most {MAX_TRIALS} trials (got {trials}); the \
         incomplete-beta evaluation is not validated beyond that scale"
    );
    assert!(
        successes <= trials,
        "jeffreys_interval requires successes ({successes}) <= trials ({trials})"
    );
    assert!(
        confidence > 0.0 && confidence < 1.0,
        "jeffreys_interval requires confidence in (0, 1), got {confidence}"
    );

    let alpha = 1.0 - confidence;
    let a = successes as f64 + 0.5;
    let b = (trials - successes) as f64 + 0.5;

    let lower = if successes == 0 {
        0.0
    } else {
        betainc_inv(a, b, alpha / 2.0)
    };
    let upper = if successes == trials {
        1.0
    } else {
        betainc_inv(a, b, 1.0 - alpha / 2.0)
    };
    assert!(
        lower <= upper,
        "jeffreys_interval produced inverted bounds ({lower} > {upper}) for k={successes}, \
         n={trials}; this indicates incomplete-beta breakdown and is a bug"
    );
    (lower, upper)
}

/// Calculate the arithmetic mean of a slice of values.
///
/// # Arguments
///
/// * `values` - A slice of f64 values
///
/// # Returns
///
/// The arithmetic mean as f64, or `f64::NAN` if the slice is empty
///
/// # Examples
///
/// ```
/// use pecos_num::stats::mean;
///
/// let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// assert_eq!(mean(&values), 3.0);
///
/// let values = vec![0.5, 0.3];
/// assert_eq!(mean(&values), 0.4);
/// ```
#[must_use]
#[allow(clippy::cast_precision_loss)]
// Cast is safe: array lengths in practice are much smaller than f64 mantissa precision
pub fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }

    let sum: f64 = values.iter().sum();
    sum / values.len() as f64
}

/// Calculate the standard deviation of values along an axis.
///
/// Drop-in replacement for `numpy.std()` with ddof (delta degrees of freedom) parameter.
///
/// # Arguments
///
/// * `values` - Array slice containing the data
/// * `ddof` - Degrees of freedom correction (0 for population std, 1 for sample std)
///
/// # Returns
///
/// Standard deviation of the values. Returns NaN if the array is empty or if
/// the corrected sample size (n - ddof) is <= 0.
///
/// # Examples
///
/// ```
/// use pecos_num::stats::std;
///
/// let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let population_std = std(&values, 0);  // Population std
/// let sample_std = std(&values, 1);      // Sample std
/// assert!((population_std - 1.4142135623730951).abs() < 1e-10);
/// assert!((sample_std - 1.5811388300841898).abs() < 1e-10);
/// ```
#[must_use]
#[allow(clippy::cast_precision_loss)]
// Cast is safe: array lengths in practice are much smaller than f64 mantissa precision
pub fn std(values: &[f64], ddof: usize) -> f64 {
    let n = values.len();

    if n == 0 {
        return f64::NAN;
    }

    // Check if corrected sample size is valid
    if n <= ddof {
        return f64::NAN;
    }

    let mean_val = mean(values);
    let variance: f64 = values
        .iter()
        .map(|&x| {
            let diff = x - mean_val;
            diff * diff
        })
        .sum();

    let corrected_n = (n - ddof) as f64;
    (variance / corrected_n).sqrt()
}

/// Calculate the arithmetic mean along an axis of an ndarray.
///
/// Idiomatic Rust API for multi-dimensional arrays. This is a thin wrapper
/// around ndarray's built-in `mean_axis` method.
///
/// # Arguments
///
/// * `arr` - Array view of any dimension
/// * `axis` - The axis along which to compute the mean
///
/// # Returns
///
/// `Some(Array)` with reduced dimension if successful, `None` if the axis is empty
///
/// # Examples
///
/// ```
/// use pecos_num::stats::mean_axis;
/// use ndarray::{array, Axis};
///
/// let arr = array![[1.0, 2.0], [3.0, 4.0]];
/// let mean_cols = mean_axis(&arr.view(), Axis(0)).unwrap();
/// assert_eq!(mean_cols, array![2.0, 3.0]);
///
/// let mean_rows = mean_axis(&arr.view(), Axis(1)).unwrap();
/// assert_eq!(mean_rows, array![1.5, 3.5]);
/// ```
#[must_use]
pub fn mean_axis<D>(arr: &ArrayView<f64, D>, axis: Axis) -> Option<Array<f64, D::Smaller>>
where
    D: Dimension + RemoveAxis,
{
    arr.mean_axis(axis)
}

/// Calculate the standard deviation along an axis of an ndarray.
///
/// Idiomatic Rust API for multi-dimensional arrays. This is a thin wrapper
/// around ndarray's built-in `std_axis` method.
///
/// # Arguments
///
/// * `arr` - Array view of any dimension
/// * `axis` - The axis along which to compute the standard deviation
/// * `ddof` - Delta degrees of freedom (0 for population std, 1 for sample std)
///
/// # Returns
///
/// Array with reduced dimension containing standard deviations
///
/// # Examples
///
/// ```
/// use pecos_num::stats::std_axis;
/// use ndarray::{array, Axis};
///
/// let arr = array![[1.0, 2.0], [3.0, 4.0]];
///
/// // Population std along axis 0 (down columns)
/// let std_cols = std_axis(&arr.view(), Axis(0), 0.0);
/// assert!((std_cols[0] - 1.0).abs() < 1e-10);
/// assert!((std_cols[1] - 1.0).abs() < 1e-10);
///
/// // Sample std along axis 1 (across rows)
/// let std_rows = std_axis(&arr.view(), Axis(1), 1.0);
/// assert!((std_rows[0] - 0.7071067811865476).abs() < 1e-10);
/// ```
#[must_use]
pub fn std_axis<D>(arr: &ArrayView<f64, D>, axis: Axis, ddof: f64) -> Array<f64, D::Smaller>
where
    D: Dimension + RemoveAxis,
{
    arr.std_axis(axis, ddof)
}

/// Generate jackknife resamples from a 1D data array.
///
/// Jackknife resampling generates `n` deterministic samples of size `n-1` from
/// a measured sample of size `n`. The i-th resample is created by removing the
/// i-th element from the original data.
///
/// This is a drop-in replacement for `astropy.stats.jackknife_resampling`.
///
/// # Arguments
///
/// * `data` - Original 1D sample from which jackknife resamples will be generated
///
/// # Returns
///
/// A 2D array where each row is a jackknife resample. The i-th row contains
/// the original data with the i-th measurement removed. Shape: `(n, n-1)`.
///
/// # Panics
///
/// Panics if `data` is empty.
///
/// # Examples
///
/// ```
/// use pecos_num::stats::jackknife_resamples;
///
/// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let resamples = jackknife_resamples(&data);
///
/// // resamples[0] = [2.0, 3.0, 4.0, 5.0]  (removed 1.0)
/// // resamples[1] = [1.0, 3.0, 4.0, 5.0]  (removed 2.0)
/// // resamples[2] = [1.0, 2.0, 4.0, 5.0]  (removed 3.0)
/// // resamples[3] = [1.0, 2.0, 3.0, 5.0]  (removed 4.0)
/// // resamples[4] = [1.0, 2.0, 3.0, 4.0]  (removed 5.0)
///
/// assert_eq!(resamples.shape(), &[5, 4]);
/// ```
#[must_use]
pub fn jackknife_resamples(data: &[f64]) -> Array<f64, ndarray::Ix2> {
    let n = data.len();
    assert!(n > 0, "data must contain at least one measurement");

    let mut resamples = Array::zeros((n, n - 1));

    for i in 0..n {
        // Fill the i-th row with all elements except the i-th
        let mut col = 0;
        for (j, &value) in data.iter().enumerate() {
            if j != i {
                resamples[[i, col]] = value;
                col += 1;
            }
        }
    }

    resamples
}

/// Compute jackknife statistics from leave-one-out parameter estimates.
///
/// Given a set of parameter estimates computed from jackknife resamples,
/// calculate the jackknife mean estimate and standard error.
///
/// The jackknife standard error uses the standard formula:
/// `SE = sqrt((n-1)/n * sum((θ_i - θ_mean)^2))`
///
/// where `θ_i` are the individual jackknife estimates and `θ_mean` is their mean.
///
/// # Arguments
///
/// * `estimates` - Slice of parameter estimates from each jackknife resample
///
/// # Returns
///
/// Tuple of `(mean_estimate, standard_error)`
///
/// # Panics
///
/// Panics if `estimates` is empty.
///
/// # Examples
///
/// ```
/// use pecos_num::stats::{jackknife_resamples, jackknife_stats, mean};
///
/// // Original data
/// let data = vec![1.5, 1.6, 1.4, 1.5, 1.7];
///
/// // Generate jackknife resamples
/// let resamples = jackknife_resamples(&data);
///
/// // Compute estimator (e.g., mean) for each resample
/// let mut estimates = Vec::new();
/// for i in 0..resamples.nrows() {
///     let resample = resamples.row(i).to_vec();
///     estimates.push(mean(&resample));
/// }
///
/// // Compute jackknife statistics
/// let (jack_mean, jack_se) = jackknife_stats(&estimates);
/// ```
#[must_use]
#[allow(clippy::cast_precision_loss)]
// Cast is safe: array lengths in practice are much smaller than f64 mantissa precision
pub fn jackknife_stats(estimates: &[f64]) -> (f64, f64) {
    assert!(!estimates.is_empty(), "estimates must not be empty");

    let n = estimates.len();
    let theta_mean = mean(estimates);

    // Jackknife standard error: SE = sqrt((n-1)/n * sum((θ_i - θ_mean)^2))
    let sum_sq_diff: f64 = estimates
        .iter()
        .map(|&theta_i| {
            let diff = theta_i - theta_mean;
            diff * diff
        })
        .sum();

    let n_f64 = n as f64;
    let standard_error = ((n_f64 - 1.0) / n_f64 * sum_sq_diff).sqrt();

    (theta_mean, standard_error)
}

/// Compute jackknife statistics along an axis of a 2D array.
///
/// Given a 2D array where each row contains parameter estimates from one jackknife
/// resample (with multiple parameters per resample), compute the jackknife mean
/// and standard error for each parameter.
///
/// This is useful for threshold curve fitting where you fit multiple parameters
/// (pth, v0, a, b, c, ...) for each jackknife resample and need statistics on
/// all parameters simultaneously.
///
/// # Arguments
///
/// * `estimates` - 2D array view where:
///   - `axis=0`: Each row is one jackknife resample, columns are different parameters
///   - `axis=1`: Each column is one jackknife resample, rows are different parameters
/// * `axis` - The axis along which to compute statistics:
///   - `Axis(0)`: Compute stats down columns (each column is a parameter)
///   - `Axis(1)`: Compute stats across rows (each row is a parameter)
///
/// # Returns
///
/// Tuple of `(mean_estimates, standard_errors)` where each is a 1D array with
/// one element per parameter.
///
/// # Panics
///
/// Panics if the specified axis has length 0.
///
/// # Examples
///
/// ```
/// use pecos_num::stats::jackknife_stats_axis;
/// use ndarray::{array, Axis};
///
/// // 3 jackknife resamples × 2 parameters
/// // Each row is estimates from one resample: [param1, param2]
/// let estimates = array![
///     [1.5, 10.0],  // Resample 1 estimates
///     [1.6, 10.5],  // Resample 2 estimates
///     [1.4, 9.5],   // Resample 3 estimates
/// ];
///
/// // Compute stats for each parameter (down columns)
/// let (means, stds) = jackknife_stats_axis(&estimates.view(), Axis(0));
///
/// // means[0] = jackknife mean of parameter 1
/// // means[1] = jackknife mean of parameter 2
/// // stds[0] = jackknife SE of parameter 1
/// // stds[1] = jackknife SE of parameter 2
/// ```
#[must_use]
#[allow(clippy::cast_precision_loss)]
// Cast is safe: array lengths in practice are much smaller than f64 mantissa precision
pub fn jackknife_stats_axis(
    estimates: &ArrayView<f64, ndarray::Ix2>,
    axis: Axis,
) -> (Array<f64, ndarray::Ix1>, Array<f64, ndarray::Ix1>) {
    // Check that axis is valid for 2D arrays
    assert!(axis.index() <= 1, "axis must be 0 or 1 for 2D arrays");

    let axis_len = estimates.len_of(axis);
    assert!(axis_len > 0, "axis length must be > 0");

    let n_f64 = axis_len as f64;

    // Compute along the specified axis
    match axis {
        Axis(0) => {
            // axis=0: compute stats down columns (each column is a parameter)
            let n_params = estimates.ncols();
            let mut means = Array::zeros(n_params);
            let mut stds = Array::zeros(n_params);

            for param_idx in 0..n_params {
                let param_estimates = estimates.column(param_idx);
                let theta_mean: f64 = param_estimates.sum() / n_f64;

                let sum_sq_diff: f64 = param_estimates
                    .iter()
                    .map(|&theta_i| {
                        let diff = theta_i - theta_mean;
                        diff * diff
                    })
                    .sum();

                let standard_error = ((n_f64 - 1.0) / n_f64 * sum_sq_diff).sqrt();

                means[param_idx] = theta_mean;
                stds[param_idx] = standard_error;
            }

            (means, stds)
        }
        Axis(1) => {
            // axis=1: compute stats across rows (each row is a parameter)
            let n_params = estimates.nrows();
            let mut means = Array::zeros(n_params);
            let mut stds = Array::zeros(n_params);

            for param_idx in 0..n_params {
                let param_estimates = estimates.row(param_idx);
                let theta_mean: f64 = param_estimates.sum() / n_f64;

                let sum_sq_diff: f64 = param_estimates
                    .iter()
                    .map(|&theta_i| {
                        let diff = theta_i - theta_mean;
                        diff * diff
                    })
                    .sum();

                let standard_error = ((n_f64 - 1.0) / n_f64 * sum_sq_diff).sqrt();

                means[param_idx] = theta_mean;
                stds[param_idx] = standard_error;
            }

            (means, stds)
        }
        _ => unreachable!("axis validity checked above"),
    }
}

/// Calculate weighted mean from (value, weight) pairs.
///
/// This is a drop-in replacement for the `wt_mean()` function in PECOS sampling.py.
///
/// # Arguments
///
/// * `data` - Slice of (value, weight) tuples. Weights should be positive.
///
/// # Returns
///
/// The weighted mean: `sum(value * weight) / sum(weight)`.
/// Returns `f64::NAN` if data is empty or total weight is zero.
///
/// # Examples
///
/// ```
/// use pecos_num::stats::weighted_mean;
///
/// // Fidelity measurements with shot counts
/// let data = vec![(0.98, 100.0), (0.94, 500.0), (0.96, 200.0)];
/// let avg = weighted_mean(&data);
/// // avg ≈ (0.98*100 + 0.94*500 + 0.96*200) / (100 + 500 + 200)
/// //     = (98 + 470 + 192) / 800 = 760 / 800 = 0.95
/// ```
#[must_use]
pub fn weighted_mean(data: &[(f64, f64)]) -> f64 {
    if data.is_empty() {
        return f64::NAN;
    }

    let (sum_weighted, sum_weight) = data
        .iter()
        .fold((0.0, 0.0), |(acc_val, acc_wt), &(value, weight)| {
            (acc_val + value * weight, acc_wt + weight)
        });

    if sum_weight == 0.0 {
        return f64::NAN;
    }

    sum_weighted / sum_weight
}

/// Jackknife resampling for weighted data with bias correction.
///
/// This is a drop-in replacement for the `jackknife()` function in PECOS sampling.py.
/// It handles weighted data (e.g., fidelity measurements with shot counts) and returns
/// the bias-corrected estimate and standard error.
///
/// For quantum computing applications, `data` typically contains (fidelity, `shot_count`)
/// pairs from multiple experimental runs.
///
/// # Arguments
///
/// * `data` - Slice of (value, weight) tuples. For quantum experiments, this is typically
///   `[(fidelity, shot_count), ...]`. Weights should be positive numbers
///   (shot counts can be passed as f64 even though they're integers).
///
/// # Returns
///
/// Tuple of `(corrected_estimate, standard_error)` where:
/// - `corrected_estimate` is the bias-corrected jackknife estimate
/// - `standard_error` is the jackknife standard error
///
/// # Special Cases
///
/// For a single data point, returns the binomial error estimate:
/// - Estimate = value
/// - Error = sqrt(p * (1-p) / weight) where p = 1 - value
///
/// # Panics
///
/// Panics if `data` is empty.
///
/// # Examples
///
/// ```
/// use pecos_num::stats::jackknife_weighted;
///
/// // Multiple fidelity measurements with shot counts
/// let data = vec![(0.98, 100.0), (0.94, 500.0), (0.96, 200.0)];
/// let (corrected, std_err) = jackknife_weighted(&data);
///
/// // Single measurement case (uses binomial error)
/// let single_data = vec![(0.95, 1000.0)];
/// let (estimate, error) = jackknife_weighted(&single_data);
/// // estimate = 0.95
/// // error = sqrt(0.05 * 0.95 / 1000) ≈ 0.0069
/// ```
#[must_use]
#[allow(clippy::cast_precision_loss)]
// Cast is safe: array lengths in practice are much smaller than f64 mantissa precision
pub fn jackknife_weighted(data: &[(f64, f64)]) -> (f64, f64) {
    assert!(
        !data.is_empty(),
        "data must contain at least one measurement"
    );

    let n = data.len();

    // Special case: single data point uses binomial error
    if n == 1 {
        let (value, weight) = data[0];
        let p = 1.0 - value;
        let error = (p * (1.0 - p) / weight).sqrt();
        return (value, error);
    }

    // Compute statistic on full data
    let stat_data = weighted_mean(data);

    // Generate leave-one-out resamples and compute statistic for each
    let mut jack_stats = Vec::with_capacity(n);
    for i in 0..n {
        // Create resample by excluding i-th element
        let resample: Vec<(f64, f64)> = data
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, &item)| item)
            .collect();

        jack_stats.push(weighted_mean(&resample));
    }

    // Compute mean of jackknife statistics
    let mean_jack_stat = mean(&jack_stats);

    // Bias correction: bias = (n-1) * (mean_jack_stat - stat_data)
    let n_f64 = n as f64;
    let bias = (n_f64 - 1.0) * (mean_jack_stat - stat_data);

    // Standard error: SE = sqrt((n-1) * mean((jack_stat - mean_jack_stat)^2))
    let sum_sq_diff: f64 = jack_stats
        .iter()
        .map(|&stat| {
            let diff = stat - mean_jack_stat;
            diff * diff
        })
        .sum();

    let std_err = ((n_f64 - 1.0) * sum_sq_diff / n_f64).sqrt();

    // Corrected estimate
    let corrected = stat_data - bias;

    (corrected, std_err)
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
mod tests {
    use super::*;
    use ndarray::Axis;

    // Reference values generated with:
    //   uv run python -c "from scipy import stats;
    //     print(stats.beta.ppf(q, k + 0.5, n - k + 0.5))"
    #[test]
    fn jeffreys_interval_matches_scipy_beta_quantiles() {
        let cases: [(u64, u64, f64, f64, f64); 6] = [
            (0, 100, 0.95, 0.0, 0.024_745_270_015_269_89),
            (100, 100, 0.95, 0.975_254_729_984_730_1, 1.0),
            (
                3,
                1000,
                0.95,
                0.000_845_634_801_829_834_8,
                0.007_984_367_358_403_443,
            ),
            (
                50,
                200,
                0.95,
                0.193_872_680_411_726_73,
                0.313_302_662_892_847_86,
            ),
            // High-confidence intervals at LER-study scales.
            (
                7,
                20_000,
                0.99999,
                3.838_996_822_347_517e-5,
                1.307_358_543_951_447_7e-3,
            ),
            (
                1234,
                20_000,
                0.99999,
                0.054_475_933_954_188_78,
                0.069_508_522_504_658_04,
            ),
        ];
        for (k, n, conf, lo_expected, hi_expected) in cases {
            let (lo, hi) = jeffreys_interval(k, n, conf);
            let lo_scale = lo_expected.abs().max(1e-12);
            let hi_scale = hi_expected.abs().max(1e-12);
            assert!(
                (lo - lo_expected).abs() / lo_scale < 1e-6,
                "lower bound for k={k}, n={n}: expected {lo_expected:.12e}, got {lo:.12e}"
            );
            assert!(
                (hi - hi_expected).abs() / hi_scale < 1e-6,
                "upper bound for k={k}, n={n}: expected {hi_expected:.12e}, got {hi:.12e}"
            );
        }
    }

    #[test]
    fn jeffreys_interval_brackets_the_point_estimate() {
        let (lo, hi) = jeffreys_interval(50, 200, 0.95);
        assert!(lo < 0.25 && 0.25 < hi);
        // Wider confidence gives a wider interval.
        let (lo99, hi99) = jeffreys_interval(50, 200, 0.99);
        assert!(lo99 < lo && hi < hi99);
    }

    #[test]
    #[should_panic(expected = "trials > 0")]
    fn jeffreys_interval_rejects_zero_trials() {
        let _ = jeffreys_interval(0, 0, 0.95);
    }

    #[test]
    #[should_panic(expected = "at most")]
    fn jeffreys_interval_rejects_unvalidated_trial_scales() {
        // Beyond ~1e12 trials the incomplete-beta continued fraction can
        // converge spuriously (observed at 2e15: inverted bounds).
        let _ = jeffreys_interval(1_000_000_000_000_000, 2_000_000_000_000_000, 0.95);
    }

    #[test]
    #[should_panic(expected = "successes")]
    fn jeffreys_interval_rejects_successes_above_trials() {
        let _ = jeffreys_interval(11, 10, 0.95);
    }

    #[test]
    #[should_panic(expected = "confidence")]
    fn jeffreys_interval_rejects_bad_confidence() {
        let _ = jeffreys_interval(5, 10, 1.0);
    }

    // Allow exact float comparisons in tests - we're testing mathematically exact results
    // that are exactly representable in IEEE 754 (e.g., 3.0, 42.0, 0.4)
    #[allow(clippy::float_cmp)]
    #[test]
    fn test_mean_basic() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(mean(&values), 3.0);
    }

    #[allow(clippy::float_cmp)]
    #[test]
    fn test_mean_single_value() {
        let values = vec![42.0];
        assert_eq!(mean(&values), 42.0);
    }

    #[allow(clippy::float_cmp)]
    #[test]
    fn test_mean_two_values() {
        let values = vec![0.5, 0.3];
        assert_eq!(mean(&values), 0.4);
    }

    #[test]
    fn test_mean_empty() {
        let values: Vec<f64> = vec![];
        assert!(mean(&values).is_nan());
    }

    #[allow(clippy::float_cmp)]
    #[test]
    fn test_mean_negative() {
        let values = vec![-1.0, -2.0, -3.0];
        assert_eq!(mean(&values), -2.0);
    }

    #[allow(clippy::float_cmp)]
    #[test]
    fn test_mean_mixed() {
        let values = vec![-2.0, 0.0, 2.0];
        assert_eq!(mean(&values), 0.0);
    }

    #[test]
    fn test_mean_precise() {
        // Test case from error models: averaging (0.001, 0.002)
        let values = vec![0.001, 0.002];
        let result = mean(&values);
        assert!((result - 0.0015).abs() < 1e-10);
    }

    #[test]
    fn test_mean_tuple_averaging() {
        // Simulating the p_meas tuple averaging use case
        let p_meas_tuple = vec![0.01, 0.015, 0.02];
        let avg = mean(&p_meas_tuple);
        assert!((avg - 0.015).abs() < 1e-10);
    }

    // Tests for std()

    #[test]
    fn test_std_population() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = std(&values, 0); // Population std (ddof=0)
        assert!((result - std::f64::consts::SQRT_2).abs() < 1e-10);
    }

    #[test]
    fn test_std_sample() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = std(&values, 1); // Sample std (ddof=1)
        assert!((result - 1.581_138_830_084_189_8).abs() < 1e-10);
    }

    #[test]
    fn test_std_single_value() {
        let values = vec![42.0];
        let result = std(&values, 0);
        assert!((result - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_std_empty() {
        let values: Vec<f64> = vec![];
        assert!(std(&values, 0).is_nan());
    }

    #[test]
    fn test_std_ddof_too_large() {
        let values = vec![1.0, 2.0];
        // With ddof=2, corrected n would be 0
        assert!(std(&values, 2).is_nan());
    }

    #[test]
    fn test_std_uniform_values() {
        let values = vec![5.0, 5.0, 5.0, 5.0];
        let result = std(&values, 0);
        assert!((result - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_std_negative_values() {
        let values = vec![-3.0, -1.0, 1.0, 3.0];
        let result = std(&values, 0);
        assert!((result - 2.236_067_977_499_79).abs() < 1e-10);
    }

    #[test]
    fn test_std_threshold_data() {
        // Simulating threshold analysis data: parameter estimates from jackknife
        let values = vec![1.5, 1.6, 1.4, 1.5, 1.7];
        let result = std(&values, 0);
        assert!((result - 0.101_980_390_271_855_71).abs() < 1e-10);
    }

    // Tests for mean_axis()

    #[allow(clippy::float_cmp)]
    #[test]
    fn test_mean_axis_2d_axis_0() {
        use ndarray::array;
        let arr = array![[1.0, 2.0], [3.0, 4.0]];
        let mean_cols = mean_axis(&arr.view(), Axis(0)).unwrap();
        assert_eq!(mean_cols, array![2.0, 3.0]);
    }

    #[allow(clippy::float_cmp)]
    #[test]
    fn test_mean_axis_2d_axis_1() {
        use ndarray::array;
        let arr = array![[1.0, 2.0], [3.0, 4.0]];
        let mean_rows = mean_axis(&arr.view(), Axis(1)).unwrap();
        assert_eq!(mean_rows, array![1.5, 3.5]);
    }

    #[test]
    fn test_mean_axis_3d() {
        use ndarray::array;
        // 3D array: 2x2x2
        let arr = array![[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]];

        // Mean along axis 0 (across the two 2x2 matrices)
        let mean_0 = mean_axis(&arr.view(), Axis(0)).unwrap();
        assert_eq!(mean_0, array![[3.0, 4.0], [5.0, 6.0]]);

        // Mean along axis 1 (down rows within each matrix)
        let mean_1 = mean_axis(&arr.view(), Axis(1)).unwrap();
        assert_eq!(mean_1, array![[2.0, 3.0], [6.0, 7.0]]);

        // Mean along axis 2 (across columns within each row)
        let mean_2 = mean_axis(&arr.view(), Axis(2)).unwrap();
        assert_eq!(mean_2, array![[1.5, 3.5], [5.5, 7.5]]);
    }

    #[test]
    fn test_mean_axis_empty_axis() {
        use ndarray::Array2;
        let arr: Array2<f64> = Array2::zeros((0, 5));
        let result = mean_axis(&arr.view(), Axis(0));
        assert!(result.is_none());
    }

    // Tests for std_axis()

    #[test]
    fn test_std_axis_2d_axis_0_population() {
        use ndarray::array;
        let arr = array![[1.0, 2.0], [3.0, 4.0]];
        let std_cols = std_axis(&arr.view(), Axis(0), 0.0);
        assert!((std_cols[0] - 1.0).abs() < 1e-10);
        assert!((std_cols[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_std_axis_2d_axis_1_sample() {
        use ndarray::array;
        use std::f64::consts::FRAC_1_SQRT_2;
        let arr = array![[1.0, 2.0], [3.0, 4.0]];
        let std_rows = std_axis(&arr.view(), Axis(1), 1.0);
        // Sample std with ddof=1: sqrt(0.5) = 1/sqrt(2)
        assert!((std_rows[0] - FRAC_1_SQRT_2).abs() < 1e-10);
        assert!((std_rows[1] - FRAC_1_SQRT_2).abs() < 1e-10);
    }

    #[test]
    fn test_std_axis_3d() {
        use ndarray::array;
        // 3D array with known variance patterns
        let arr = array![[[1.0, 3.0], [5.0, 7.0]], [[2.0, 4.0], [6.0, 8.0]]];

        // Std along axis 0 (population std)
        let std_0 = std_axis(&arr.view(), Axis(0), 0.0);
        // Each pair differs by 1, so std = 0.5
        assert!((std_0[[0, 0]] - 0.5).abs() < 1e-10);
        assert!((std_0[[0, 1]] - 0.5).abs() < 1e-10);
        assert!((std_0[[1, 0]] - 0.5).abs() < 1e-10);
        assert!((std_0[[1, 1]] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_std_axis_uniform_values() {
        use ndarray::Array2;
        let arr = Array2::from_elem((3, 4), 5.0);
        let std_axis_0 = std_axis(&arr.view(), Axis(0), 0.0);
        let std_axis_1 = std_axis(&arr.view(), Axis(1), 0.0);

        // All values are the same, so std should be 0
        for &val in &std_axis_0 {
            assert!((val - 0.0).abs() < 1e-10);
        }
        for &val in &std_axis_1 {
            assert!((val - 0.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_mean_and_std_axis_consistency() {
        use ndarray::array;
        // Test that mean_axis and std_axis work together correctly
        let arr = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];

        let means = mean_axis(&arr.view(), Axis(0)).unwrap();
        let stds = std_axis(&arr.view(), Axis(0), 0.0);

        // Mean of each column: [4.0, 5.0, 6.0]
        assert_eq!(means, array![4.0, 5.0, 6.0]);

        // Std of each column (population): all should be sqrt(6) ≈ 2.449
        for &std_val in &stds {
            assert!((std_val - 2.449_489_742_783_178).abs() < 1e-10);
        }
    }

    // Tests for jackknife_resamples()

    #[test]
    fn test_jackknife_resamples_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let resamples = jackknife_resamples(&data);

        // Check shape
        assert_eq!(resamples.shape(), &[5, 4]);

        // Check each resample
        assert_eq!(resamples.row(0).to_vec(), vec![2.0, 3.0, 4.0, 5.0]); // removed 1.0
        assert_eq!(resamples.row(1).to_vec(), vec![1.0, 3.0, 4.0, 5.0]); // removed 2.0
        assert_eq!(resamples.row(2).to_vec(), vec![1.0, 2.0, 4.0, 5.0]); // removed 3.0
        assert_eq!(resamples.row(3).to_vec(), vec![1.0, 2.0, 3.0, 5.0]); // removed 4.0
        assert_eq!(resamples.row(4).to_vec(), vec![1.0, 2.0, 3.0, 4.0]); // removed 5.0
    }

    #[test]
    fn test_jackknife_resamples_two_elements() {
        let data = vec![10.0, 20.0];
        let resamples = jackknife_resamples(&data);

        assert_eq!(resamples.shape(), &[2, 1]);
        assert_eq!(resamples.row(0).to_vec(), vec![20.0]);
        assert_eq!(resamples.row(1).to_vec(), vec![10.0]);
    }

    #[test]
    fn test_jackknife_resamples_single_element() {
        let data = vec![42.0];
        let resamples = jackknife_resamples(&data);

        assert_eq!(resamples.shape(), &[1, 0]);
    }

    #[test]
    #[should_panic(expected = "data must contain at least one measurement")]
    fn test_jackknife_resamples_empty() {
        let data: Vec<f64> = vec![];
        let _ = jackknife_resamples(&data);
    }

    #[test]
    fn test_jackknife_resamples_negative_values() {
        let data = vec![-3.0, -1.0, 1.0, 3.0];
        let resamples = jackknife_resamples(&data);

        assert_eq!(resamples.shape(), &[4, 3]);
        assert_eq!(resamples.row(0).to_vec(), vec![-1.0, 1.0, 3.0]);
        assert_eq!(resamples.row(1).to_vec(), vec![-3.0, 1.0, 3.0]);
        assert_eq!(resamples.row(2).to_vec(), vec![-3.0, -1.0, 3.0]);
        assert_eq!(resamples.row(3).to_vec(), vec![-3.0, -1.0, 1.0]);
    }

    // Tests for jackknife_stats()

    #[test]
    fn test_jackknife_stats_basic() {
        // Example from threshold analysis
        let estimates = vec![1.5, 1.6, 1.4, 1.5, 1.7];
        let (jack_mean, jack_se) = jackknife_stats(&estimates);

        // Mean should be 1.54
        assert!((jack_mean - 1.54).abs() < 1e-10);

        // Jackknife SE: sqrt((n-1)/n * sum((θ_i - θ_mean)^2))
        // n = 5
        // θ_mean = 1.54
        // Differences: [-0.04, 0.06, -0.14, -0.04, 0.16]
        // Sum of squares: 0.0016 + 0.0036 + 0.0196 + 0.0016 + 0.0256 = 0.052
        // SE = sqrt(4/5 * 0.052) = sqrt(0.0416) ≈ 0.204
        assert!((jack_se - 0.203_960_780_543_711_4).abs() < 1e-10);
    }

    #[test]
    fn test_jackknife_stats_uniform_estimates() {
        // All estimates the same → SE should be 0
        let estimates = vec![2.5, 2.5, 2.5, 2.5];
        let (jack_mean, jack_se) = jackknife_stats(&estimates);

        assert!((jack_mean - 2.5).abs() < 1e-10);
        assert!((jack_se - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_jackknife_stats_two_estimates() {
        let estimates = vec![1.0, 3.0];
        let (jack_mean, jack_se) = jackknife_stats(&estimates);

        // Mean = 2.0
        assert!((jack_mean - 2.0).abs() < 1e-10);

        // SE = sqrt(1/2 * ((1-2)^2 + (3-2)^2)) = sqrt(1/2 * 2) = 1.0
        assert!((jack_se - 1.0).abs() < 1e-10);
    }

    #[test]
    #[should_panic(expected = "estimates must not be empty")]
    fn test_jackknife_stats_empty() {
        let estimates: Vec<f64> = vec![];
        let _ = jackknife_stats(&estimates);
    }

    #[test]
    fn test_jackknife_resamples_and_stats_integration() {
        // Full jackknife workflow: resample data, compute estimates, get statistics
        let data = vec![1.5, 1.6, 1.4, 1.5, 1.7];

        // Generate jackknife resamples
        let resamples = jackknife_resamples(&data);

        // Compute mean for each resample
        let mut estimates = Vec::new();
        for i in 0..resamples.nrows() {
            let resample = resamples.row(i).to_vec();
            estimates.push(mean(&resample));
        }

        // Compute jackknife statistics
        let (jack_mean, jack_se) = jackknife_stats(&estimates);

        // The jackknife mean of means should be close to the original mean
        let original_mean = mean(&data);
        assert!((jack_mean - original_mean).abs() < 1e-10);

        // SE should be positive and reasonable
        assert!(jack_se > 0.0);
        assert!(jack_se < 1.0); // Sanity check for this data
    }

    // Tests for weighted_mean()

    #[test]
    fn test_weighted_mean_basic() {
        // Example from docstring
        let data = vec![(0.98, 100.0), (0.94, 500.0), (0.96, 200.0)];
        let avg = weighted_mean(&data);
        // (0.98*100 + 0.94*500 + 0.96*200) / (100 + 500 + 200)
        // = (98 + 470 + 192) / 800 = 760 / 800 = 0.95
        assert!((avg - 0.95).abs() < 1e-10);
    }

    #[test]
    fn test_weighted_mean_uniform_weights() {
        // With uniform weights, should match unweighted mean
        let data = vec![(1.0, 1.0), (2.0, 1.0), (3.0, 1.0), (4.0, 1.0), (5.0, 1.0)];
        let wt_avg = weighted_mean(&data);
        assert!((wt_avg - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_weighted_mean_single_value() {
        let data = vec![(0.95, 1000.0)];
        let avg = weighted_mean(&data);
        assert!((avg - 0.95).abs() < 1e-10);
    }

    #[test]
    fn test_weighted_mean_empty() {
        let data: Vec<(f64, f64)> = vec![];
        assert!(weighted_mean(&data).is_nan());
    }

    #[test]
    fn test_weighted_mean_zero_total_weight() {
        let data = vec![(0.5, 0.0), (0.7, 0.0)];
        assert!(weighted_mean(&data).is_nan());
    }

    #[test]
    fn test_weighted_mean_heavy_weight() {
        // One measurement has much higher weight
        let data = vec![(0.5, 10.0), (0.9, 1000.0)];
        let avg = weighted_mean(&data);
        // (0.5*10 + 0.9*1000) / (10 + 1000) = 905 / 1010 ≈ 0.896
        assert!((avg - 0.896_039_603_960_396).abs() < 1e-10);
    }

    // Tests for jackknife_weighted()

    #[test]
    fn test_jackknife_weighted_single_measurement() {
        // Single measurement should use binomial error
        let data = vec![(0.95, 1000.0)];
        let (estimate, error) = jackknife_weighted(&data);

        // Estimate should be the value itself
        assert!((estimate - 0.95).abs() < 1e-10);

        // Error = sqrt(p * (1-p) / n) where p = 1 - 0.95 = 0.05
        // error = sqrt(0.05 * 0.95 / 1000) = sqrt(0.0000475) ≈ 0.00689
        let expected_error = (0.05_f64 * 0.95 / 1000.0).sqrt();
        assert!((error - expected_error).abs() < 1e-10);
    }

    #[test]
    fn test_jackknife_weighted_multiple_measurements() {
        // Multiple measurements with different weights
        let data = vec![(0.98, 100.0), (0.94, 500.0), (0.96, 200.0)];
        let (corrected, std_err) = jackknife_weighted(&data);

        // The corrected estimate should be close to the weighted mean
        let wt_avg = weighted_mean(&data);
        // Bias correction might shift it slightly, but should be in same ballpark
        assert!((corrected - wt_avg).abs() < 0.1); // Loose check

        // Standard error should be positive
        assert!(std_err > 0.0);
        assert!(std_err < 1.0); // Sanity check
    }

    #[test]
    fn test_jackknife_weighted_uniform_weights() {
        // With uniform weights, behavior should match unweighted jackknife
        let data = vec![(1.0, 1.0), (2.0, 1.0), (3.0, 1.0), (4.0, 1.0), (5.0, 1.0)];
        let (corrected, std_err) = jackknife_weighted(&data);

        // Mean is 3.0, jackknife should be close
        assert!((corrected - 3.0).abs() < 0.1);

        // SE should be reasonable
        assert!(std_err > 0.0);
    }

    #[test]
    #[should_panic(expected = "data must contain at least one measurement")]
    fn test_jackknife_weighted_empty() {
        let data: Vec<(f64, f64)> = vec![];
        let _ = jackknife_weighted(&data);
    }

    #[test]
    fn test_jackknife_weighted_two_measurements() {
        let data = vec![(0.9, 100.0), (0.8, 200.0)];
        let (corrected, std_err) = jackknife_weighted(&data);

        // Weighted mean = (0.9*100 + 0.8*200) / 300 = 250/300 ≈ 0.833
        let wt_avg = weighted_mean(&data);
        assert!((wt_avg - 0.833_333_333_333_333_3).abs() < 1e-10);

        // Corrected should be close to weighted mean
        assert!((corrected - wt_avg).abs() < 0.1);

        // SE should be positive
        assert!(std_err > 0.0);
    }

    #[test]
    fn test_jackknife_weighted_vs_sampling_py() {
        // Test case from sampling.py docstring
        // data = [(0.98, 100), (0.94, 500), (0.96, 200)]
        let data = vec![(0.98, 100.0), (0.94, 500.0), (0.96, 200.0)];
        let (corrected, std_err) = jackknife_weighted(&data);

        // Weighted mean
        let wt_mean = weighted_mean(&data); // 0.95

        // Full jackknife calculation to verify
        let n = data.len();

        // Leave-one-out estimates
        let est_0 = weighted_mean(&[(0.94, 500.0), (0.96, 200.0)]); // removed first
        let est_1 = weighted_mean(&[(0.98, 100.0), (0.96, 200.0)]); // removed second
        let est_2 = weighted_mean(&[(0.98, 100.0), (0.94, 500.0)]); // removed third

        let jack_estimates = vec![est_0, est_1, est_2];
        let mean_jack = mean(&jack_estimates);

        // Bias = (n-1) * (mean_jack - wt_mean)
        let bias = (n as f64 - 1.0) * (mean_jack - wt_mean);
        let expected_corrected = wt_mean - bias;

        // SE = sqrt((n-1) * mean((est - mean_jack)^2))
        let sum_sq_diff: f64 = jack_estimates
            .iter()
            .map(|&e| (e - mean_jack).powi(2))
            .sum();
        let expected_se = ((n as f64 - 1.0) * sum_sq_diff / n as f64).sqrt();

        assert!((corrected - expected_corrected).abs() < 1e-10);
        assert!((std_err - expected_se).abs() < 1e-10);
    }

    // Tests for jackknife_stats_axis()

    #[test]
    fn test_jackknife_stats_axis_0_basic() {
        use ndarray::array;
        // 3 resamples × 2 parameters
        let estimates = array![[1.5, 10.0], [1.6, 10.5], [1.4, 9.5]];

        let (means, stds) = jackknife_stats_axis(&estimates.view(), Axis(0));

        // Check means
        assert!((means[0] - 1.5).abs() < 1e-10); // mean of [1.5, 1.6, 1.4]
        assert!((means[1] - 10.0).abs() < 1e-10); // mean of [10.0, 10.5, 9.5]

        // Check standard errors manually
        // For param 0: [1.5, 1.6, 1.4], mean=1.5, diffs=[0, 0.1, -0.1]
        // SE = sqrt(2/3 * (0 + 0.01 + 0.01)) = sqrt(2/3 * 0.02) = sqrt(0.0133...)
        let expected_se_0 = ((2.0 / 3.0) * 0.02_f64).sqrt();
        assert!((stds[0] - expected_se_0).abs() < 1e-10);

        // For param 1: [10.0, 10.5, 9.5], mean=10.0, diffs=[0, 0.5, -0.5]
        // SE = sqrt(2/3 * (0 + 0.25 + 0.25)) = sqrt(2/3 * 0.5)
        let expected_se_1 = ((2.0_f64 / 3.0) * 0.5).sqrt();
        assert!((stds[1] - expected_se_1).abs() < 1e-10);
    }

    #[test]
    fn test_jackknife_stats_axis_1_basic() {
        use ndarray::array;
        // 2 parameters × 3 resamples (transposed from above)
        let estimates = array![[1.5, 1.6, 1.4], [10.0, 10.5, 9.5]];

        let (means, stds) = jackknife_stats_axis(&estimates.view(), Axis(1));

        // Same expected results as axis=0 test
        assert!((means[0] - 1.5).abs() < 1e-10);
        assert!((means[1] - 10.0).abs() < 1e-10);

        let expected_se_0 = ((2.0 / 3.0) * 0.02_f64).sqrt();
        let expected_se_1 = ((2.0_f64 / 3.0) * 0.5).sqrt();
        assert!((stds[0] - expected_se_0).abs() < 1e-10);
        assert!((stds[1] - expected_se_1).abs() < 1e-10);
    }

    #[test]
    fn test_jackknife_stats_axis_uniform() {
        use ndarray::Array2;
        // All estimates the same → SE should be 0
        let estimates = Array2::from_elem((5, 3), 2.5);

        let (means, stds) = jackknife_stats_axis(&estimates.view(), Axis(0));

        for &mean_val in &means {
            assert!((mean_val - 2.5).abs() < 1e-10);
        }
        for &std_val in &stds {
            assert!((std_val - 0.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_jackknife_stats_axis_threshold_use_case() {
        use ndarray::array;
        // Simulating threshold fitting: 5 jackknife resamples, fitting 5 parameters
        // (pth, v0, a, b, c)
        let estimates = array![
            [0.101, 1.5, 0.50, 2.1, -0.3],  // Resample 1
            [0.102, 1.6, 0.51, 2.2, -0.31], // Resample 2
            [0.100, 1.4, 0.49, 2.0, -0.29], // Resample 3
            [0.101, 1.5, 0.50, 2.1, -0.30], // Resample 4
            [0.103, 1.7, 0.52, 2.3, -0.32], // Resample 5
        ];

        let (means, stds) = jackknife_stats_axis(&estimates.view(), Axis(0));

        // Should have 5 parameter means and 5 parameter SEs
        assert_eq!(means.len(), 5);
        assert_eq!(stds.len(), 5);

        // pth mean should be around 0.1014
        assert!((means[0] - 0.101_4).abs() < 1e-10);

        // All SEs should be positive
        for &se in &stds {
            assert!(se > 0.0);
        }
    }

    #[test]
    fn test_jackknife_stats_axis_vs_1d() {
        use ndarray::array;
        // Single parameter case: should match 1D jackknife_stats
        let estimates_1d = vec![1.5, 1.6, 1.4, 1.5, 1.7];
        let (mean_1d, se_1d) = jackknife_stats(&estimates_1d);

        // Same data as 2D array (5 resamples × 1 parameter)
        let estimates_2d = array![[1.5], [1.6], [1.4], [1.5], [1.7]];
        let (means_2d, stds_2d) = jackknife_stats_axis(&estimates_2d.view(), Axis(0));

        assert!((mean_1d - means_2d[0]).abs() < 1e-10);
        assert!((se_1d - stds_2d[0]).abs() < 1e-10);
    }

    #[test]
    #[should_panic(expected = "axis length must be > 0")]
    fn test_jackknife_stats_axis_empty_axis() {
        use ndarray::Array2;
        let estimates: Array2<f64> = Array2::zeros((0, 5));
        let _ = jackknife_stats_axis(&estimates.view(), Axis(0));
    }

    #[test]
    #[should_panic(expected = "axis must be 0 or 1 for 2D arrays")]
    fn test_jackknife_stats_axis_invalid_axis() {
        use ndarray::array;
        let estimates = array![[1.0, 2.0], [3.0, 4.0]];
        let _ = jackknife_stats_axis(&estimates.view(), Axis(2));
    }
}
