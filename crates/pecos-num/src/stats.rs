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
//! This module provides drop-in replacements for numpy/scipy statistical functions.
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
//! The slice functions are fast and simple for 1D data. The axis functions
//! provide idiomatic Rust API for multi-dimensional arrays.

use ndarray::{Array, ArrayView, Axis, Dimension, RemoveAxis};

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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Axis;

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
}
