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
}
