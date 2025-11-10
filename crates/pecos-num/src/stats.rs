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
//! This module provides drop-in replacements for numpy statistical functions.

use ndarray::{Array1, ArrayView2};

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

/// Calculate the power of a base raised to an exponent.
///
/// Drop-in replacement for `numpy.power()` for scalar values.
///
/// # Arguments
///
/// * `base` - The base value
/// * `exponent` - The exponent value
///
/// # Returns
///
/// The result of base^exponent as f64
///
/// # Examples
///
/// ```
/// use pecos_num::stats::power;
///
/// // Basic integer power
/// assert!((power(2.0, 3.0) - 8.0).abs() < 1e-10);
///
/// // Fractional power (square root)
/// assert!((power(4.0, 0.5) - 2.0).abs() < 1e-10);
///
/// // Negative power
/// assert!((power(2.0, -1.0) - 0.5).abs() < 1e-10);
///
/// // Threshold curve use case
/// let dist = 5.0;
/// let v0 = 2.0;
/// let result = power(dist, 1.0 / v0);
/// assert!((result - 2.236_067_977_499_79).abs() < 1e-10);
/// ```
#[must_use]
pub fn power(base: f64, exponent: f64) -> f64 {
    base.powf(exponent)
}

/// Calculate the square root of a value.
///
/// Drop-in replacement for `numpy.sqrt()` for scalar values.
///
/// # Arguments
///
/// * `x` - Input value
///
/// # Returns
///
/// The square root of x. Returns NaN for negative inputs.
///
/// # Examples
///
/// ```
/// use pecos_num::stats::sqrt;
///
/// assert_eq!(sqrt(4.0), 2.0);
/// assert_eq!(sqrt(9.0), 3.0);
/// assert!((sqrt(2.0) - 1.414_213_562_373_095).abs() < 1e-10);
///
/// // Variance to standard deviation use case
/// let variance = 2.0;
/// let std_dev = sqrt(variance);
/// assert!((std_dev - 1.414_213_562_373_095).abs() < 1e-10);
/// ```
#[must_use]
pub fn sqrt(x: f64) -> f64 {
    x.sqrt()
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

/// Extract the diagonal elements from a 2D array (matrix).
///
/// This is a drop-in replacement for `numpy.diag()` when extracting diagonal elements.
///
/// # Arguments
///
/// * `matrix` - A 2D array view
///
/// # Returns
///
/// A 1D array containing the diagonal elements
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use pecos_num::stats::diag;
///
/// // Extract diagonal from a square matrix
/// let matrix = array![[1.0, 2.0, 3.0],
///                     [4.0, 5.0, 6.0],
///                     [7.0, 8.0, 9.0]];
/// let diagonal = diag(matrix.view());
/// assert_eq!(diagonal, array![1.0, 5.0, 9.0]);
///
/// // Works with non-square matrices too
/// let matrix = array![[1.0, 2.0],
///                     [3.0, 4.0],
///                     [5.0, 6.0]];
/// let diagonal = diag(matrix.view());
/// assert_eq!(diagonal, array![1.0, 4.0]);
/// ```
#[must_use]
pub fn diag(matrix: ArrayView2<f64>) -> Array1<f64> {
    let (nrows, ncols) = matrix.dim();
    let diag_len = nrows.min(ncols);

    let mut diagonal = Array1::zeros(diag_len);
    for i in 0..diag_len {
        diagonal[i] = matrix[[i, i]];
    }

    diagonal
}

/// Generate evenly spaced values over a specified interval.
///
/// This is a Rust implementation of `numpy.linspace()`.
///
/// Returns `num` evenly spaced samples, calculated over the interval `[start, stop]`.
/// The endpoint of the interval can optionally be excluded.
///
/// # Arguments
///
/// * `start` - The starting value of the sequence
/// * `stop` - The end value of the sequence
/// * `num` - Number of samples to generate. Default is 50.
/// * `endpoint` - If true, `stop` is the last sample. Otherwise, it is not included. Default is true.
///
/// # Returns
///
/// Array of `num` equally spaced samples in the closed interval `[start, stop]` or
/// the half-open interval `[start, stop)` (depending on whether `endpoint` is true or false).
///
/// # Examples
///
/// ```
/// use pecos_num::stats::linspace;
///
/// // Generate 5 values from 0 to 10
/// let values = linspace(0.0, 10.0, 5, true);
/// assert_eq!(values.len(), 5);
/// assert!((values[0] - 0.0).abs() < 1e-10);
/// assert!((values[4] - 10.0).abs() < 1e-10);
///
/// // Generate 4 values from 0 to 10 (endpoint excluded)
/// let values = linspace(0.0, 10.0, 4, false);
/// assert_eq!(values.len(), 4);
/// assert!((values[0] - 0.0).abs() < 1e-10);
/// assert!((values[3] - 7.5).abs() < 1e-10);
/// ```
#[must_use]
pub fn linspace(start: f64, stop: f64, num: usize, endpoint: bool) -> Array1<f64> {
    if num == 0 {
        return Array1::zeros(0);
    }

    if num == 1 {
        return Array1::from_vec(vec![start]);
    }

    let mut result = Array1::zeros(num);

    if endpoint {
        // Include the endpoint: divide the range into (num-1) segments
        let step = (stop - start) / (num - 1) as f64;
        for i in 0..num {
            result[i] = start + step * i as f64;
        }
        // Ensure the last value is exactly stop to avoid floating point errors
        result[num - 1] = stop;
    } else {
        // Exclude the endpoint: divide the range into num segments
        let step = (stop - start) / num as f64;
        for i in 0..num {
            result[i] = start + step * i as f64;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

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

    // Tests for power()

    #[allow(clippy::float_cmp)]
    #[test]
    fn test_power_integer_exponent() {
        // Basic integer powers
        assert_eq!(power(2.0, 3.0), 8.0);
        assert_eq!(power(3.0, 2.0), 9.0);
        assert_eq!(power(10.0, 0.0), 1.0);
    }

    #[test]
    fn test_power_fractional_exponent() {
        // Fractional powers (roots)
        assert!((power(4.0, 0.5) - 2.0).abs() < 1e-10);
        assert!((power(27.0, 1.0 / 3.0) - 3.0).abs() < 1e-10);
        assert!((power(16.0, 0.25) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_power_negative_exponent() {
        // Negative powers (reciprocals)
        assert!((power(2.0, -1.0) - 0.5).abs() < 1e-10);
        assert!((power(4.0, -0.5) - 0.5).abs() < 1e-10);
        assert!((power(10.0, -2.0) - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_power_negative_base() {
        // Negative base with integer exponent
        assert!((power(-2.0, 3.0) - (-8.0)).abs() < 1e-10);
        assert!((power(-3.0, 2.0) - 9.0).abs() < 1e-10);
    }

    #[allow(clippy::float_cmp)]
    #[test]
    fn test_power_special_cases() {
        // Special cases
        assert_eq!(power(0.0, 2.0), 0.0);
        assert_eq!(power(1.0, 100.0), 1.0);
        assert_eq!(power(5.0, 0.0), 1.0);
    }

    #[test]
    fn test_power_threshold_curve_pattern() {
        // Pattern from threshold_curve.py: np.power(dist, 1.0 / v0)
        let dist = 5.0;
        let v0 = 2.0;
        let result = power(dist, 1.0 / v0);
        assert!((result - 2.236_067_977_499_79).abs() < 1e-10);
    }

    #[test]
    fn test_power_squared() {
        // Pattern from threshold_curve.py: np.power(x, 2)
        let x = 3.5;
        let result = power(x, 2.0);
        assert!((result - 12.25).abs() < 1e-10);
    }

    #[test]
    fn test_power_large_exponent() {
        // Test with larger exponents
        assert!((power(2.0, 10.0) - 1024.0).abs() < 1e-10);
        assert!((power(1.5, 5.0) - 7.59375).abs() < 1e-10);
    }

    #[allow(clippy::float_cmp)]
    #[test]
    fn test_sqrt_perfect_squares() {
        assert_eq!(sqrt(4.0), 2.0);
        assert_eq!(sqrt(9.0), 3.0);
        assert_eq!(sqrt(16.0), 4.0);
        assert_eq!(sqrt(25.0), 5.0);
        assert_eq!(sqrt(100.0), 10.0);
    }

    #[test]
    fn test_sqrt_irrational() {
        // Test irrational square roots
        assert!((sqrt(2.0) - std::f64::consts::SQRT_2).abs() < 1e-10);
        assert!((sqrt(3.0) - 1.732_050_807_568_877).abs() < 1e-10);
        assert!((sqrt(5.0) - 2.236_067_977_499_79).abs() < 1e-10);
    }

    #[allow(clippy::float_cmp)]
    #[test]
    fn test_sqrt_special_cases() {
        assert_eq!(sqrt(0.0), 0.0);
        assert_eq!(sqrt(1.0), 1.0);
        assert!(sqrt(-1.0).is_nan());
        assert!(sqrt(f64::NEG_INFINITY).is_nan());
        assert_eq!(sqrt(f64::INFINITY), f64::INFINITY);
    }

    #[test]
    fn test_sqrt_variance_to_std() {
        // Test the variance-to-standard-deviation use case
        let variance = 2.0;
        let std_dev = sqrt(variance);
        assert!((std_dev - std::f64::consts::SQRT_2).abs() < 1e-10);

        let variance = 4.0;
        let std_dev = sqrt(variance);
        assert!((std_dev - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_sqrt_small_values() {
        // Test with small fractional values
        assert!((sqrt(0.25) - 0.5).abs() < 1e-10);
        assert!((sqrt(0.01) - 0.1).abs() < 1e-10);
        assert!((sqrt(0.0001) - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_sqrt_large_values() {
        // Test with larger values
        assert!((sqrt(10_000.0) - 100.0).abs() < 1e-10);
        assert!((sqrt(1_000_000.0) - 1000.0).abs() < 1e-10);
    }

    // Tests for diag()
    #[test]
    fn test_diag_square_matrix() {
        use ndarray::array;

        // 3x3 matrix
        let matrix = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let diagonal = diag(matrix.view());

        assert_eq!(diagonal.len(), 3);
        #[allow(clippy::float_cmp)]
        {
            assert_eq!(diagonal[0], 1.0);
            assert_eq!(diagonal[1], 5.0);
            assert_eq!(diagonal[2], 9.0);
        }
    }

    #[test]
    fn test_diag_rectangular_matrix_more_rows() {
        use ndarray::array;

        // 3x2 matrix (more rows than columns)
        let matrix = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let diagonal = diag(matrix.view());

        assert_eq!(diagonal.len(), 2);
        #[allow(clippy::float_cmp)]
        {
            assert_eq!(diagonal[0], 1.0);
            assert_eq!(diagonal[1], 4.0);
        }
    }

    #[test]
    fn test_diag_rectangular_matrix_more_cols() {
        use ndarray::array;

        // 2x3 matrix (more columns than rows)
        let matrix = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let diagonal = diag(matrix.view());

        assert_eq!(diagonal.len(), 2);
        #[allow(clippy::float_cmp)]
        {
            assert_eq!(diagonal[0], 1.0);
            assert_eq!(diagonal[1], 5.0);
        }
    }

    #[test]
    fn test_diag_covariance_matrix() {
        use ndarray::array;

        // Typical covariance matrix from polyfit
        let cov_matrix = array![[0.0025, 0.0010], [0.0010, 0.0004]];
        let variances = diag(cov_matrix.view());

        assert_eq!(variances.len(), 2);
        assert!((variances[0] - 0.0025).abs() < 1e-10);
        assert!((variances[1] - 0.0004).abs() < 1e-10);
    }

    #[test]
    fn test_diag_identity_matrix() {
        use ndarray::array;

        let identity = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let diagonal = diag(identity.view());

        assert_eq!(diagonal.len(), 3);
        #[allow(clippy::float_cmp)]
        {
            assert_eq!(diagonal[0], 1.0);
            assert_eq!(diagonal[1], 1.0);
            assert_eq!(diagonal[2], 1.0);
        }
    }

    #[test]
    fn test_linspace_basic() {
        let values = linspace(0.0, 10.0, 5, true);
        assert_eq!(values.len(), 5);
        assert!((values[0] - 0.0).abs() < 1e-10);
        assert!((values[1] - 2.5).abs() < 1e-10);
        assert!((values[2] - 5.0).abs() < 1e-10);
        assert!((values[3] - 7.5).abs() < 1e-10);
        assert!((values[4] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_linspace_endpoint_false() {
        let values = linspace(0.0, 10.0, 4, false);
        assert_eq!(values.len(), 4);
        assert!((values[0] - 0.0).abs() < 1e-10);
        assert!((values[1] - 2.5).abs() < 1e-10);
        assert!((values[2] - 5.0).abs() < 1e-10);
        assert!((values[3] - 7.5).abs() < 1e-10);
    }

    #[test]
    fn test_linspace_single_value() {
        let values = linspace(5.0, 10.0, 1, true);
        assert_eq!(values.len(), 1);
        #[allow(clippy::float_cmp)]
        {
            assert_eq!(values[0], 5.0);
        }
    }

    #[test]
    fn test_linspace_empty() {
        let values = linspace(0.0, 10.0, 0, true);
        assert_eq!(values.len(), 0);
    }

    #[test]
    fn test_linspace_negative_range() {
        let values = linspace(-5.0, 5.0, 11, true);
        assert_eq!(values.len(), 11);
        assert!((values[0] - (-5.0)).abs() < 1e-10);
        assert!((values[5] - 0.0).abs() < 1e-10);
        assert!((values[10] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_linspace_large_num() {
        // Test with 1000 points (common use case for plotting)
        let values = linspace(0.0, 1.0, 1000, true);
        assert_eq!(values.len(), 1000);
        assert!((values[0] - 0.0).abs() < 1e-10);
        assert!((values[999] - 1.0).abs() < 1e-10);
        // Check spacing is uniform
        let expected_step = 1.0 / 999.0;
        assert!((values[1] - values[0] - expected_step).abs() < 1e-10);
    }
}
