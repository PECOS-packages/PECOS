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
//!
//! # Design Philosophy
//!
//! This module follows idiomatic Rust patterns:
//! - Use standard library iterator methods (`.iter().sum()`) rather than custom traits
//! - Provide simple functions for common cases
//! - Provide `_axis()` variants for multi-dimensional operations
//!
//! The polymorphism happens in the `PyO3` bindings, not in custom Rust traits.

use ndarray::{Array, Array1, ArrayBase, ArrayView2, Axis, Data, Dimension, RemoveAxis};

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

/// Return evenly spaced values within a given interval.
///
/// This is a Rust implementation of `numpy.arange()`.
///
/// Returns values in the half-open interval `[start, stop)` with the given step.
/// This function is similar to Python's built-in `range()` but returns an array
/// and can handle floating-point arguments.
///
/// # Arguments
///
/// * `start` - Start of interval (inclusive)
/// * `stop` - End of interval (exclusive)
/// * `step` - Spacing between values
///
/// # Returns
///
/// Array of evenly spaced values. For floating-point arguments, the length is
/// `ceil((stop - start) / step)`.
///
/// # Notes
///
/// - When using non-integer step sizes, floating-point precision errors can occur.
///   For such cases, consider using `linspace()` instead.
/// - The actual step value used is `stop - start` divided by the number of elements,
///   which may differ slightly from the requested `step` due to floating-point arithmetic.
///
/// # Examples
///
/// ```
/// use pecos_num::array::arange;
///
/// // Integer-like steps
/// let values = arange(0.0, 5.0, 1.0);
/// assert_eq!(values.len(), 5);
/// assert!((values[0] - 0.0).abs() < 1e-10);
/// assert!((values[4] - 4.0).abs() < 1e-10);
///
/// // Floating-point steps
/// let values = arange(0.0, 1.0, 0.25);
/// assert_eq!(values.len(), 4);
/// assert!((values[0] - 0.0).abs() < 1e-10);
/// assert!((values[1] - 0.25).abs() < 1e-10);
/// assert!((values[2] - 0.5).abs() < 1e-10);
/// assert!((values[3] - 0.75).abs() < 1e-10);
///
/// // Negative step (countdown)
/// let values = arange(5.0, 0.0, -1.0);
/// assert_eq!(values.len(), 5);
/// assert!((values[0] - 5.0).abs() < 1e-10);
/// assert!((values[4] - 1.0).abs() < 1e-10);
/// ```
///
/// # Panics
///
/// Panics if `step_size` is zero or if `step_size` has the wrong sign for the given start/stop.
#[must_use]
#[allow(clippy::cast_precision_loss)] // Intentional: converting array size to f64 for mathematical operations
#[allow(clippy::cast_possible_truncation)] // Intentional: ceil returns f64, we need usize
#[allow(clippy::cast_sign_loss)] // Intentional: we've validated that length is positive
pub fn arange(start: f64, stop: f64, step_size: f64) -> Array1<f64> {
    assert!(step_size != 0.0, "arange: step cannot be zero");

    // Calculate the number of elements
    // NumPy behavior: length = ceil((stop - start) / step_size)
    let length_f64 = ((stop - start) / step_size).ceil();

    // Handle edge cases
    if length_f64 <= 0.0 {
        // Empty array if start >= stop and step > 0, or start <= stop and step < 0
        return Array1::zeros(0);
    }

    let length = length_f64 as usize;
    let mut result = Array1::zeros(length);

    // Generate values: result[i] = start + i * step_size
    for i in 0..length {
        result[i] = start + (i as f64) * step_size;
    }

    result
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

// Note: sum() for slices removed - use values.iter().sum() directly (idiomatic Rust)
// sum_axis() below is kept for multi-dimensional operations

/// Calculate the sum of array elements along an axis.
///
/// Drop-in replacement for `numpy.sum()` with axis parameter.
///
/// # Arguments
///
/// * `arr` - Array to sum
/// * `axis` - Axis along which to sum
///
/// # Returns
///
/// Array with sums computed along the specified axis
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use pecos_num::array::sum_axis;
/// use ndarray::Axis;
///
/// // 2D array
/// let arr = array![[1.0, 2.0, 3.0],
///                  [4.0, 5.0, 6.0]];
///
/// // Sum along axis 0 (down columns)
/// let sum_cols = sum_axis(&arr.view(), Axis(0));
/// assert_eq!(sum_cols, array![5.0, 7.0, 9.0]);
///
/// // Sum along axis 1 (across rows)
/// let sum_rows = sum_axis(&arr.view(), Axis(1));
/// assert_eq!(sum_rows, array![6.0, 15.0]);
/// ```
#[must_use]
pub fn sum_axis<S, D>(arr: &ArrayBase<S, D>, axis: Axis) -> Array<f64, D::Smaller>
where
    S: Data<Elem = f64>,
    D: Dimension + RemoveAxis,
{
    arr.map_axis(axis, |lane| lane.sum())
}

/// Create a new array filled with zeros.
///
/// Drop-in replacement for `numpy.zeros()` for float arrays.
///
/// # Arguments
///
/// * `shape` - Shape of the new array (e.g., `(3, 4)` for a 3x4 matrix)
///
/// # Returns
///
/// Array filled with zeros
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use pecos_num::array::zeros;
///
/// // 1D array
/// let arr = zeros(5);
/// assert_eq!(arr, array![0.0, 0.0, 0.0, 0.0, 0.0]);
///
/// // 2D array
/// let arr2d = zeros((2, 3));
/// assert_eq!(arr2d, array![[0.0, 0.0, 0.0],
///                          [0.0, 0.0, 0.0]]);
/// ```
#[must_use]
pub fn zeros<Sh>(shape: Sh) -> Array<f64, Sh::Dim>
where
    Sh: ndarray::ShapeBuilder,
{
    Array::zeros(shape)
}

/// Create a new array filled with ones.
///
/// Drop-in replacement for `numpy.ones()` for float arrays.
///
/// # Arguments
///
/// * `shape` - Shape of the new array (e.g., `(3, 4)` for a 3x4 matrix)
///
/// # Returns
///
/// Array filled with ones
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use pecos_num::array::ones;
///
/// // 1D array
/// let arr = ones(5);
/// assert_eq!(arr, array![1.0, 1.0, 1.0, 1.0, 1.0]);
///
/// // 2D array
/// let arr2d = ones((2, 3));
/// assert_eq!(arr2d, array![[1.0, 1.0, 1.0],
///                          [1.0, 1.0, 1.0]]);
/// ```
#[must_use]
pub fn ones<Sh>(shape: Sh) -> Array<f64, Sh::Dim>
where
    Sh: ndarray::ShapeBuilder,
{
    Array::ones(shape)
}

/// Delete an element from an array at the specified index.
///
/// Drop-in replacement for `numpy.delete()` for 1D arrays with a single index.
///
/// Returns a new array with the element at the specified index removed.
/// This is particularly useful for jackknife resampling and leave-one-out analysis.
///
/// # Arguments
///
/// * `arr` - Input array (1D)
/// * `index` - Index of the element to remove
///
/// # Returns
///
/// A new array with the element at `index` removed
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use pecos_num::array::delete;
///
/// // Delete single element
/// let arr = array![1.0, 2.0, 3.0, 4.0, 5.0];
/// let result = delete(&arr, 2);
/// assert_eq!(result, array![1.0, 2.0, 4.0, 5.0]);
///
/// // Delete first element
/// let arr = array![10.0, 20.0, 30.0];
/// let result = delete(&arr, 0);
/// assert_eq!(result, array![20.0, 30.0]);
///
/// // Delete last element
/// let arr = array![10.0, 20.0, 30.0];
/// let result = delete(&arr, 2);
/// assert_eq!(result, array![10.0, 20.0]);
/// ```
///
/// # Panics
///
/// Panics if `index` is out of bounds.
#[must_use]
pub fn delete<T: Clone>(arr: &Array1<T>, index: usize) -> Array1<T> {
    assert!(
        index < arr.len(),
        "Index {} out of bounds for array of length {}",
        index,
        arr.len()
    );

    // Create result vector by concatenating elements before and after the index
    let mut result_vec = Vec::with_capacity(arr.len() - 1);

    // Add elements before the index
    result_vec.extend_from_slice(&arr.as_slice().unwrap()[..index]);

    // Add elements after the index
    result_vec.extend_from_slice(&arr.as_slice().unwrap()[(index + 1)..]);

    Array1::from_vec(result_vec)
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

    // Tests for sum() removed - use values.iter().sum() directly (stdlib functionality)

    // Tests for sum_axis()
    #[test]
    fn test_sum_axis_2d_axis0() {
        use ndarray::array;

        let arr = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let result = sum_axis(&arr.view(), Axis(0));

        assert_eq!(result.len(), 3);
        #[allow(clippy::float_cmp)]
        {
            assert_eq!(result[0], 5.0); // 1.0 + 4.0
            assert_eq!(result[1], 7.0); // 2.0 + 5.0
            assert_eq!(result[2], 9.0); // 3.0 + 6.0
        }
    }

    #[test]
    fn test_sum_axis_2d_axis1() {
        use ndarray::array;

        let arr = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let result = sum_axis(&arr.view(), Axis(1));

        assert_eq!(result.len(), 2);
        #[allow(clippy::float_cmp)]
        {
            assert_eq!(result[0], 6.0); // 1.0 + 2.0 + 3.0
            assert_eq!(result[1], 15.0); // 4.0 + 5.0 + 6.0
        }
    }

    #[test]
    fn test_sum_axis_3d() {
        use ndarray::array;

        // 2x2x3 array
        let arr = array![
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
        ];

        // Sum along axis 0 (first dimension)
        let result = sum_axis(&arr.view(), Axis(0));
        assert_eq!(result.shape(), &[2, 3]);
        #[allow(clippy::float_cmp)]
        {
            assert_eq!(result[[0, 0]], 8.0); // 1.0 + 7.0
            assert_eq!(result[[1, 2]], 18.0); // 6.0 + 12.0
        }
    }

    // Tests for delete()
    #[test]
    fn test_delete_middle() {
        use ndarray::array;

        let arr = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = delete(&arr, 2);

        assert_eq!(result.len(), 4);
        assert_eq!(result, array![1.0, 2.0, 4.0, 5.0]);
    }

    #[test]
    fn test_delete_first() {
        use ndarray::array;

        let arr = array![10.0, 20.0, 30.0];
        let result = delete(&arr, 0);

        assert_eq!(result.len(), 2);
        assert_eq!(result, array![20.0, 30.0]);
    }

    #[test]
    fn test_delete_last() {
        use ndarray::array;

        let arr = array![10.0, 20.0, 30.0];
        let result = delete(&arr, 2);

        assert_eq!(result.len(), 2);
        assert_eq!(result, array![10.0, 20.0]);
    }

    #[test]
    fn test_delete_two_elements() {
        use ndarray::array;

        let arr = array![1.0, 2.0];
        let result = delete(&arr, 0);
        assert_eq!(result, array![2.0]);

        let result2 = delete(&arr, 1);
        assert_eq!(result2, array![1.0]);
    }

    #[test]
    #[should_panic(expected = "Index 5 out of bounds for array of length 5")]
    fn test_delete_out_of_bounds() {
        use ndarray::array;

        let arr = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let _result = delete(&arr, 5);
    }

    #[test]
    fn test_delete_jackknife_use_case() {
        use ndarray::array;

        // Simulate jackknife resampling use case from threshold_curve.py
        let plist = array![0.01, 0.02, 0.03, 0.04, 0.05];

        // Leave-one-out: remove each element in turn
        for i in 0..plist.len() {
            let p_copy = delete(&plist, i);
            assert_eq!(p_copy.len(), plist.len() - 1);

            // Verify the removed element is not in the result
            #[allow(clippy::float_cmp)] // Exact comparison needed for test correctness
            for j in 0..p_copy.len() {
                assert_ne!(p_copy[j], plist[i]);
            }
        }
    }

    // Tests for arange()
    #[test]
    fn test_arange_basic() {
        let values = arange(0.0, 5.0, 1.0);
        assert_eq!(values.len(), 5);
        #[allow(clippy::float_cmp)]
        {
            assert_eq!(values[0], 0.0);
            assert_eq!(values[1], 1.0);
            assert_eq!(values[2], 2.0);
            assert_eq!(values[3], 3.0);
            assert_eq!(values[4], 4.0);
        }
    }

    #[test]
    fn test_arange_float_step() {
        let values = arange(0.0, 1.0, 0.25);
        assert_eq!(values.len(), 4);
        assert!((values[0] - 0.0).abs() < 1e-10);
        assert!((values[1] - 0.25).abs() < 1e-10);
        assert!((values[2] - 0.5).abs() < 1e-10);
        assert!((values[3] - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_arange_negative_step() {
        let values = arange(5.0, 0.0, -1.0);
        assert_eq!(values.len(), 5);
        #[allow(clippy::float_cmp)]
        {
            assert_eq!(values[0], 5.0);
            assert_eq!(values[1], 4.0);
            assert_eq!(values[2], 3.0);
            assert_eq!(values[3], 2.0);
            assert_eq!(values[4], 1.0);
        }
    }

    #[test]
    fn test_arange_empty_positive_step() {
        // start >= stop with positive step should give empty array
        let values = arange(5.0, 0.0, 1.0);
        assert_eq!(values.len(), 0);
    }

    #[test]
    fn test_arange_empty_negative_step() {
        // start <= stop with negative step should give empty array
        let values = arange(0.0, 5.0, -1.0);
        assert_eq!(values.len(), 0);
    }

    #[test]
    fn test_arange_small_step() {
        let values = arange(0.0, 0.3, 0.1);
        assert_eq!(values.len(), 3);
        assert!((values[0] - 0.0).abs() < 1e-10);
        assert!((values[1] - 0.1).abs() < 1e-10);
        assert!((values[2] - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_arange_negative_range() {
        let values = arange(-2.0, 2.0, 1.0);
        assert_eq!(values.len(), 4);
        #[allow(clippy::float_cmp)]
        {
            assert_eq!(values[0], -2.0);
            assert_eq!(values[1], -1.0);
            assert_eq!(values[2], 0.0);
            assert_eq!(values[3], 1.0);
        }
    }

    #[test]
    #[should_panic(expected = "arange: step cannot be zero")]
    fn test_arange_zero_step() {
        let _values = arange(0.0, 5.0, 0.0);
    }
}
