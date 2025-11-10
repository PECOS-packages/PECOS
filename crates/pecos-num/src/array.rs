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

//! Array operations for numerical analysis.
//!
//! This module provides drop-in replacements for numpy array operations.

use ndarray::{Array1, ArrayView2};

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
/// use pecos_num::array::diag;
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
/// use pecos_num::array::linspace;
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
#[allow(clippy::cast_precision_loss)] // Intentional: converting array size to f64 for mathematical operations
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
        let delta = (stop - start) / (num - 1) as f64;
        for i in 0..num {
            result[i] = start + delta * i as f64;
        }
        // Ensure the last value is exactly stop to avoid floating point errors
        result[num - 1] = stop;
    } else {
        // Exclude the endpoint: divide the range into num segments
        let delta = (stop - start) / num as f64;
        for i in 0..num {
            result[i] = start + delta * i as f64;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

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
