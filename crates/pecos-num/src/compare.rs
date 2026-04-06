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

//! Comparison and validation functions for numerical analysis.
//!
//! Trait-based comparison operations across scalars, complex numbers,
//! and arrays.

use ndarray::{Array, ArrayBase, Axis, Data, Dimension, RemoveAxis};
use num_complex::Complex64;

/// Trait for checking if values are NaN (Not a Number).
///
/// Uniform interface for NaN checking across different numeric types.
///
/// # Examples
///
/// ```
/// use pecos_num::prelude::*;
///
/// // Scalars
/// assert!(f64::NAN.isnan());
/// assert!(!5.0_f64.isnan());
///
/// // Arrays
/// let arr = array![1.0, f64::NAN, 3.0];
/// let result = arr.isnan();
/// assert_eq!(result, array![false, true, false]);
/// ```
pub trait IsNan {
    /// The output type when checking for NaN.
    type Output;

    /// Check if this value (or values) are NaN.
    fn isnan(&self) -> Self::Output;
}

/// Check if a scalar f32 value is NaN.
impl IsNan for f32 {
    type Output = bool;

    #[inline]
    fn isnan(&self) -> bool {
        f32::is_nan(*self)
    }
}

/// Check if a scalar f64 value is NaN.
impl IsNan for f64 {
    type Output = bool;

    #[inline]
    fn isnan(&self) -> bool {
        f64::is_nan(*self)
    }
}

/// Check if a complex32 scalar value is NaN.
impl IsNan for num_complex::Complex<f32> {
    type Output = bool;

    #[inline]
    fn isnan(&self) -> bool {
        self.re.is_nan() || self.im.is_nan()
    }
}

/// Check if a complex128 scalar value is NaN.
impl IsNan for Complex64 {
    type Output = bool;

    #[inline]
    fn isnan(&self) -> bool {
        self.re.is_nan() || self.im.is_nan()
    }
}

/// Check if values in an array are NaN.
///
/// This implementation works for arrays of any type that implements `IsNan`.
impl<S, D, T> IsNan for ArrayBase<S, D>
where
    S: Data<Elem = T>,
    T: IsNan<Output = bool> + Clone,
    D: Dimension,
{
    type Output = Array<bool, D>;

    #[inline]
    fn isnan(&self) -> Array<bool, D> {
        self.mapv(|x| x.isnan())
    }
}

/// Trait for checking if values are close within a tolerance.
///
/// Tolerance-based comparison across different numeric types.
///
/// The tolerance check follows `NumPy`'s convention:
/// `|a - b| <= (atol + rtol * |b|)`
///
/// # Examples
///
/// ```
/// use pecos_num::prelude::*;
///
/// // Scalars
/// assert!(1.0_f64.isclose(&1.00001, 1e-4, 1e-8));
/// assert!(!1.0_f64.isclose(&1.1, 1e-5, 1e-8));
///
/// // Arrays
/// let a = array![1.0, 2.0, 3.0];
/// let b = array![1.00001, 2.00001, 3.1];
/// let result = a.isclose(&b, 1e-4, 1e-8);
/// assert_eq!(result, array![true, true, false]);
/// ```
pub trait IsClose {
    /// The output type when checking closeness.
    type Output;

    /// Check if values are close within specified tolerances.
    ///
    /// # Arguments
    ///
    /// * `other` - The value to compare against
    /// * `rtol` - Relative tolerance (typical: 1e-5)
    /// * `atol` - Absolute tolerance (typical: 1e-8)
    fn isclose(&self, other: &Self, rtol: f64, atol: f64) -> Self::Output;
}

/// Check if two i32 values are close within tolerance.
/// For integers, converts to f64 for tolerance checking.
impl IsClose for i32 {
    type Output = bool;

    #[inline]
    fn isclose(&self, other: &i32, rtol: f64, atol: f64) -> bool {
        // Convert to f64 for tolerance calculation
        let self_f = f64::from(*self);
        let other_f = f64::from(*other);
        (self_f - other_f).abs() <= (atol + rtol * other_f.abs())
    }
}

/// Check if two i64 values are close within tolerance.
/// For integers, converts to f64 for tolerance checking.
impl IsClose for i64 {
    type Output = bool;

    #[inline]
    fn isclose(&self, other: &i64, rtol: f64, atol: f64) -> bool {
        // Convert to f64 for tolerance calculation
        #[allow(clippy::cast_precision_loss)]
        let self_f = *self as f64;
        #[allow(clippy::cast_precision_loss)]
        let other_f = *other as f64;
        (self_f - other_f).abs() <= (atol + rtol * other_f.abs())
    }
}

/// Check if two f32 values are close within tolerance.
impl IsClose for f32 {
    type Output = bool;

    #[inline]
    fn isclose(&self, other: &f32, rtol: f64, atol: f64) -> bool {
        // Handle special cases
        // Exact equality check is intentional before tolerance check
        #[allow(clippy::float_cmp)]
        if self == other {
            return true;
        }

        // Both NaN should return false (numpy behavior)
        if self.is_nan() || other.is_nan() {
            return false;
        }

        // Both infinity with same sign returns true
        if self.is_infinite() && other.is_infinite() {
            return self.signum() == other.signum();
        }

        // Check tolerance: |a - b| <= (atol + rtol * |b|)
        // Use f64 for tolerance calculation to match numpy precision
        let self_f64 = f64::from(*self);
        let other_f64 = f64::from(*other);
        (self_f64 - other_f64).abs() <= (atol + rtol * other_f64.abs())
    }
}

/// Check if two f64 values are close within tolerance.
impl IsClose for f64 {
    type Output = bool;

    #[inline]
    fn isclose(&self, other: &f64, rtol: f64, atol: f64) -> bool {
        // Handle special cases
        // Exact equality check is intentional before tolerance check
        #[allow(clippy::float_cmp)]
        if self == other {
            return true;
        }

        // Both NaN should return false (numpy behavior)
        if self.is_nan() || other.is_nan() {
            return false;
        }

        // Both infinity with same sign returns true
        if self.is_infinite() && other.is_infinite() {
            return self.signum() == other.signum();
        }

        // Check tolerance: |a - b| <= (atol + rtol * |b|)
        (self - other).abs() <= (atol + rtol * other.abs())
    }
}

/// Check if two complex values are close within tolerance.
///
/// Uses magnitude-based comparison to match `NumPy`'s behavior:
/// `|a - b| <= (atol + rtol * |b|)`
/// where `|z|` is the L2 norm (magnitude): `sqrt(real² + imag²)`
impl IsClose for Complex64 {
    type Output = bool;

    #[inline]
    fn isclose(&self, other: &Complex64, rtol: f64, atol: f64) -> bool {
        let diff = self - other;
        diff.norm() <= (atol + rtol * other.norm())
    }
}

/// Check if two arrays are element-wise close within tolerance.
///
/// This implementation works for arrays of any type that implements `IsClose`.
impl<S, D, T> IsClose for ArrayBase<S, D>
where
    S: Data<Elem = T>,
    T: IsClose<Output = bool> + Clone,
    D: Dimension,
{
    type Output = Array<bool, D>;

    #[inline]
    fn isclose(&self, other: &Self, rtol: f64, atol: f64) -> Array<bool, D> {
        ndarray::Zip::from(self)
            .and(other)
            .map_collect(|a_val, b_val| a_val.isclose(b_val, rtol, atol))
    }
}

/// Check if all elements in two arrays are close within specified tolerances.
///
/// Drop-in replacement for `numpy.allclose()`. Returns `true` if all pairs
/// of elements are close according to the tolerance check:
/// `|a - b| <= (atol + rtol * |b|)`
///
/// # Arguments
///
/// * `a` - First array or scalar
/// * `b` - Second array or scalar
/// * `rtol` - Relative tolerance (typical: 1e-5)
/// * `atol` - Absolute tolerance (typical: 1e-8)
/// * `equal_nan` - If true, NaNs in the same position are considered equal (default: false in numpy)
///
/// # Returns
///
/// Returns `true` if all elements are close, `false` otherwise.
///
/// # Examples
///
/// ```
/// use pecos_num::prelude::*;
///
/// // 1D Arrays
/// let a = array![1.0, 2.0, 3.0];
/// let b = array![1.00001, 2.00001, 3.00001];
/// assert!(allclose(&a, &b, 1e-4, 1e-8, false));
///
/// let c = array![1.0, 2.0, 10.0];
/// assert!(!allclose(&a, &c, 1e-5, 1e-8, false));
///
/// // 2D Arrays (quantum gate matrices)
/// let gate1 = array![[1.0, 0.0], [0.0, 1.0]];
/// let gate2 = array![[1.00001, 0.0], [0.0, 0.99999]];
/// assert!(allclose(&gate1, &gate2, 1e-4, 1e-8, false));
/// ```
#[must_use]
pub fn allclose<S1, S2, D, T>(
    a: &ArrayBase<S1, D>,
    b: &ArrayBase<S2, D>,
    rtol: f64,
    atol: f64,
    equal_nan: bool,
) -> bool
where
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
    T: IsClose<Output = bool> + IsNan<Output = bool> + Clone,
    D: Dimension,
{
    // Arrays must have the same shape
    if a.shape() != b.shape() {
        return false;
    }

    // Check all elements
    ndarray::Zip::from(a).and(b).all(|a_val, b_val| {
        // Handle NaN case if equal_nan is true
        if equal_nan && a_val.isnan() && b_val.isnan() {
            return true;
        }
        a_val.isclose(b_val, rtol, atol)
    })
}

/// Assert that all elements in two arrays are close within specified tolerances.
///
/// Drop-in replacement for `numpy.testing.assert_allclose()`. Panics with a detailed
/// error message if any elements are not close according to the tolerance check:
/// `|a - b| <= (atol + rtol * |b|)`
///
/// # Arguments
///
/// * `a` - First array
/// * `b` - Second array
/// * `rtol` - Relative tolerance (default: 1e-7)
/// * `atol` - Absolute tolerance (default: 0.0)
/// * `equal_nan` - If true, NaNs in the same position are considered equal (default: false)
///
/// # Panics
///
/// Panics if arrays are not close, providing detailed information about:
/// - Shape mismatches
/// - Maximum absolute difference
/// - Maximum relative difference
/// - Number of mismatched elements
///
/// # Examples
///
/// ```
/// use pecos_num::prelude::*;
///
/// // These should pass
/// let a = array![1.0, 2.0, 3.0];
/// let b = array![1.00001, 2.00001, 3.00001];
/// assert_allclose(&a, &b, 1e-4, 1e-8, false);
/// ```
///
/// ```should_panic
/// use pecos_num::prelude::*;
///
/// // This should panic with detailed error
/// let a = array![1.0, 2.0, 3.0];
/// let c = array![1.0, 2.0, 10.0];
/// assert_allclose(&a, &c, 1e-5, 1e-8, false);
/// ```
pub fn assert_allclose<S1, S2, D>(
    a: &ArrayBase<S1, D>,
    b: &ArrayBase<S2, D>,
    rtol: f64,
    atol: f64,
    equal_nan: bool,
) where
    S1: Data<Elem = f64>,
    S2: Data<Elem = f64>,
    D: Dimension,
{
    // Check shapes first
    assert!(
        a.shape() == b.shape(),
        "Arrays have different shapes: a.shape={:?}, b.shape={:?}",
        a.shape(),
        b.shape()
    );

    // Compute element-wise differences
    let mut max_abs_diff: f64 = 0.0;
    let mut max_rel_diff: f64 = 0.0;
    let mut mismatch_count: usize = 0;
    let mut first_mismatch_values: Option<(f64, f64)> = None;

    // Check all elements
    for (a_val, b_val) in a.iter().zip(b.iter()) {
        // Handle NaN case
        if equal_nan && a_val.is_nan() && b_val.is_nan() {
            continue;
        }

        // Check if values are close
        if !a_val.isclose(b_val, rtol, atol) {
            let abs_diff = (a_val - b_val).abs();
            let rel_diff = if b_val.abs() > 0.0_f64 {
                abs_diff / b_val.abs()
            } else {
                abs_diff
            };

            max_abs_diff = max_abs_diff.max(abs_diff);
            max_rel_diff = max_rel_diff.max(rel_diff);
            mismatch_count += 1;

            // Store first mismatch for detailed error message
            if first_mismatch_values.is_none() {
                first_mismatch_values = Some((*a_val, *b_val));
            }
        }
    }

    // If there are mismatches, panic with detailed error message
    if mismatch_count > 0 {
        let (first_a, first_b) = first_mismatch_values
            .expect("first_mismatch_values must be Some when mismatch_count > 0");

        panic!(
            "\nNot equal to tolerance rtol={}, atol={}\n\
             Mismatched elements: {} / {}\n\
             Max absolute difference: {}\n\
             Max relative difference: {}\n\
             First mismatch values:\n\
             \ta = {}\n\
             \tb = {}",
            rtol,
            atol,
            mismatch_count,
            a.len(),
            max_abs_diff,
            max_rel_diff,
            first_a,
            first_b
        );
    }
}

/// Check if two arrays are equal element-wise.
///
/// Drop-in replacement for `numpy.array_equal(a1, a2, equal_nan=False)`.
///
/// Returns `True` if two arrays have the same shape and all elements are equal.
/// Unlike `allclose`, this function uses exact equality (`==`) rather than tolerance-based comparison.
///
/// # Arguments
///
/// * `a` - First input array
/// * `b` - Second input array
/// * `equal_nan` - If `true`, NaNs in the same position are considered equal (default: `false`)
///
/// # Returns
///
/// `true` if arrays are equal, `false` otherwise
///
/// # Examples
///
/// ```
/// use pecos_num::compare::array_equal;
/// use ndarray::array;
///
/// // Equal arrays
/// let a = array![1.0, 2.0, 3.0];
/// let b = array![1.0, 2.0, 3.0];
/// assert!(array_equal(&a, &b, false));
///
/// // Different values
/// let c = array![1.0, 2.0, 4.0];
/// assert!(!array_equal(&a, &c, false));
///
/// // Different shapes - use a 1D array with different length
/// let d = array![1.0, 2.0];
/// assert!(!array_equal(&a.view(), &d.view(), false));
/// ```
pub fn array_equal<S1, S2, D, T>(
    a: &ArrayBase<S1, D>,
    b: &ArrayBase<S2, D>,
    equal_nan: bool,
) -> bool
where
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
    T: PartialEq + IsNan<Output = bool> + Clone,
    D: Dimension,
{
    // Arrays must have the same shape
    if a.shape() != b.shape() {
        return false;
    }

    // Check all elements for exact equality
    ndarray::Zip::from(a).and(b).all(|a_val, b_val| {
        // Handle NaN case if equal_nan is true
        if equal_nan && a_val.isnan() && b_val.isnan() {
            return true;
        }
        a_val == b_val
    })
}

/// Reduce a boolean array along an axis with `all`.
///
/// Returns an array with one fewer dimension, where each element is `true`
/// if all values along the given axis are `true`.
///
/// # Examples
///
/// ```
/// use pecos_num::compare::all_axis;
/// use ndarray::{array, Axis};
///
/// let arr = array![[true, false], [true, true]];
/// assert_eq!(all_axis(&arr, Axis(0)), array![true, false]);
/// assert_eq!(all_axis(&arr, Axis(1)), array![false, true]);
/// ```
#[must_use]
pub fn all_axis<S, D>(arr: &ArrayBase<S, D>, axis: Axis) -> Array<bool, D::Smaller>
where
    S: Data<Elem = bool>,
    D: Dimension + RemoveAxis,
{
    arr.map_axis(axis, |lane| lane.iter().all(|&x| x))
}

/// Reduce a boolean array along an axis with `any`.
///
/// Returns an array with one fewer dimension, where each element is `true`
/// if any value along the given axis is `true`.
///
/// # Examples
///
/// ```
/// use pecos_num::compare::any_axis;
/// use ndarray::{array, Axis};
///
/// let arr = array![[true, false], [false, false]];
/// assert_eq!(any_axis(&arr, Axis(0)), array![true, false]);
/// assert_eq!(any_axis(&arr, Axis(1)), array![true, false]);
/// ```
#[must_use]
pub fn any_axis<S, D>(arr: &ArrayBase<S, D>, axis: Axis) -> Array<bool, D::Smaller>
where
    S: Data<Elem = bool>,
    D: Dimension + RemoveAxis,
{
    arr.map_axis(axis, |lane| lane.iter().any(|&x| x))
}

/// Conditional selection based on a boolean condition (scalar version).
///
/// Drop-in replacement for `numpy.where(condition, x, y)` for scalar inputs.
/// Returns `x` if condition is true, otherwise returns `y`.
///
/// # Arguments
///
/// * `condition` - Boolean value determining which value to return
/// * `x` - Value to return if condition is true
/// * `y` - Value to return if condition is false
///
/// # Returns
///
/// Returns `x` if `condition` is true, otherwise returns `y`
///
/// # Examples
///
/// ```
/// use pecos_num::compare::where_;
///
/// // Scalar usage
/// assert_eq!(where_(true, 10.0, 20.0), 10.0);
/// assert_eq!(where_(false, 10.0, 20.0), 20.0);
///
/// // Typical use case: conditional computation
/// let dist = 5;
/// let result = where_(
///     dist % 2 == 1,
///     dist as f64 * 2.0,
///     dist as f64 / 2.0,
/// );
/// assert_eq!(result, 10.0); // dist is odd, so returns dist * 2.0
/// ```
#[must_use]
#[inline]
pub fn where_<T>(condition: bool, x: T, y: T) -> T {
    if condition { x } else { y }
}

/// Trait for conditional element selection - similar to `numpy.where()`.
///
/// Provides a `.where()` method that selects elements from `x` or `y` based on
/// a boolean condition. This follows the pattern of numpy's `where` function.
///
/// # Examples
///
/// ```
/// use pecos_num::prelude::*;
///
/// // Scalar bool
/// let result = true.where_(&10.0, &20.0);
/// assert_eq!(result, 10.0);
///
/// let result = false.where_(&10.0, &20.0);
/// assert_eq!(result, 20.0);
///
/// // Boolean arrays
/// let condition = array![true, false, true, false];
/// let x = array![10.0, 20.0, 30.0, 40.0];
/// let y = array![100.0, 200.0, 300.0, 400.0];
/// let result = condition.where_(&x, &y);
/// assert_eq!(result, array![10.0, 200.0, 30.0, 400.0]);
/// ```
pub trait Where<Rhs = Self> {
    /// The output type after conditional selection.
    type Output;

    /// Select elements from `x` where condition is true, otherwise from `y`.
    ///
    /// # Arguments
    ///
    /// * `x` - Values to select when condition is true
    /// * `y` - Values to select when condition is false
    fn where_(&self, x: &Rhs, y: &Rhs) -> Self::Output;
}

/// Conditional selection for scalars - simple if-else operation.
impl<T: Clone> Where<T> for bool {
    type Output = T;

    fn where_(&self, x: &T, y: &T) -> T {
        if *self { x.clone() } else { y.clone() }
    }
}

/// Conditional selection for arrays - element-wise where operation.
///
/// This implementation uses ndarray's `Zip` for efficient functional-style
/// element-wise conditional selection.
impl<S1, S2, D, T> Where<ArrayBase<S2, D>> for ArrayBase<S1, D>
where
    S1: Data<Elem = bool>,
    S2: Data<Elem = T>,
    T: Clone,
    D: Dimension,
{
    type Output = Array<T, D>;

    fn where_(&self, x: &ArrayBase<S2, D>, y: &ArrayBase<S2, D>) -> Array<T, D> {
        // Functional-style element-wise selection using Zip
        ndarray::Zip::from(self)
            .and(x)
            .and(y)
            .map_collect(
                |&cond, x_val, y_val| {
                    if cond { x_val.clone() } else { y_val.clone() }
                },
            )
    }
}

/// Check if two floating-point values are approximately equal using relative tolerance.
///
/// This matches the behavior of the `approx` crate's `relative_eq` function.
/// It checks both absolute and relative tolerance to handle near-zero values:
/// - Absolute: `|a - b| <= epsilon`
/// - Relative: `|a - b| <= epsilon * max(|a|, |b|)`
///
/// Returns true if EITHER condition is satisfied.
///
/// # Arguments
///
/// * `a` - First value
/// * `b` - Second value
/// * `epsilon` - Tolerance (used for both absolute and relative comparison)
///
/// # Returns
///
/// `true` if the values are within the tolerance
///
/// # Examples
///
/// ```
/// use pecos_num::compare::relative_eq;
///
/// assert!(relative_eq(1.0, 1.0000001, 1e-6));
/// assert!(!relative_eq(1.0, 1.1, 1e-6));
/// // Near-zero values work correctly
/// assert!(relative_eq(1e-17, 0.0, 1e-9));
/// ```
#[must_use]
#[inline]
pub fn relative_eq(a: f64, b: f64, epsilon: f64) -> bool {
    // Handle exact equality (including infinities with same sign)
    #[allow(clippy::float_cmp)]
    if a == b {
        return true;
    }

    // Handle NaN
    if a.is_nan() || b.is_nan() {
        return false;
    }

    // Handle infinities with different signs
    if a.is_infinite() || b.is_infinite() {
        return false;
    }

    let diff = (a - b).abs();

    // Check absolute tolerance first (handles near-zero case)
    if diff <= epsilon {
        return true;
    }

    // Then check relative tolerance: |a - b| <= epsilon * max(|a|, |b|)
    let max_abs = a.abs().max(b.abs());
    diff <= epsilon * max_abs
}

/// Assert that two floating-point values are approximately equal using relative tolerance.
///
/// This macro provides a drop-in replacement for `approx::assert_relative_eq!`.
///
/// # Syntax
///
/// ```text
/// assert_relative_eq!(a, b, epsilon = 1e-10);
/// ```
///
/// # Panics
///
/// Panics if `|a - b| > epsilon * max(|a|, |b|)`
///
/// # Examples
///
/// ```
/// use pecos_num::assert_relative_eq;
///
/// let a = 1.0;
/// let b = 1.0000001;
/// assert_relative_eq!(a, b, epsilon = 1e-6);
/// ```
#[macro_export]
macro_rules! assert_relative_eq {
    ($a:expr, $b:expr, epsilon = $epsilon:expr) => {{
        let a = $a;
        let b = $b;
        let epsilon = $epsilon;
        if !$crate::compare::relative_eq(a, b, epsilon) {
            panic!(
                "assertion failed: `relative_eq(left, right)`\n  left: `{:?}`\n right: `{:?}`\n epsilon: `{:?}`\n |left - right|: `{:?}`",
                a,
                b,
                epsilon,
                (a - b).abs()
            );
        }
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    // Tests for IsNan trait
    #[test]
    fn test_isnan_with_nan() {
        // Test with actual NaN value
        assert!(f64::NAN.isnan());
    }

    #[test]
    fn test_isnan_with_normal_values() {
        // Test with normal finite values
        assert!(!0.0_f64.isnan());
        assert!(!1.0_f64.isnan());
        assert!(!(-1.0_f64).isnan());
        assert!(!42.5_f64.isnan());
        assert!(!(-999.999_f64).isnan());
    }

    #[test]
    fn test_isnan_with_infinity() {
        // Test with infinity values (should return false)
        assert!(!f64::INFINITY.isnan());
        assert!(!f64::NEG_INFINITY.isnan());
    }

    #[test]
    fn test_isnan_with_zero() {
        // Test with positive and negative zero
        assert!(!0.0_f64.isnan());
        assert!(!(-0.0_f64).isnan());
    }

    #[test]
    fn test_isnan_with_computed_nan() {
        // Test with NaN constant and invalid computations
        assert!(f64::NAN.isnan());

        let inf_minus_inf = f64::INFINITY - f64::INFINITY;
        assert!(inf_minus_inf.isnan());

        let sqrt_negative = (-1.0_f64).sqrt();
        assert!(sqrt_negative.isnan());
    }

    #[test]
    fn test_isnan_validation_use_case() {
        // Error checking use case (curve fitting validation)
        let valid_variance = 0.0025_f64;
        let invalid_variance = f64::NAN;

        assert!(!valid_variance.isnan());
        assert!(invalid_variance.isnan());

        // Simulate variance validation loop
        let variances = [0.001_f64, 0.002, f64::NAN, 0.004];
        let has_nan = variances.iter().any(super::IsNan::isnan);
        assert!(has_nan);
    }

    // Tests for IsClose trait
    #[test]
    fn test_isclose_exact() {
        // Exact equality
        assert!(1.0_f64.isclose(&1.0, 1e-5, 1e-8));
        assert!(0.0_f64.isclose(&0.0, 1e-5, 1e-8));
        assert!((-1.0_f64).isclose(&-1.0, 1e-5, 1e-8));
    }

    #[test]
    fn test_isclose_within_tolerance() {
        // Within relative tolerance
        assert!(1.0_f64.isclose(&1.00001, 1e-4, 1e-8));
        assert!(100.0_f64.isclose(&100.001, 1e-4, 1e-8));

        // Within absolute tolerance
        assert!(1e-10_f64.isclose(&2e-10, 0.0, 1e-9));
        assert!(0.0_f64.isclose(&1e-9, 0.0, 1e-8));
    }

    #[test]
    fn test_isclose_outside_tolerance() {
        // Outside both tolerances
        assert!(!1.0_f64.isclose(&1.1, 1e-5, 1e-8));
        assert!(!1.0_f64.isclose(&2.0, 1e-5, 1e-8));
        assert!(!100.0_f64.isclose(&101.0, 1e-5, 1e-8));
    }

    #[test]
    fn test_isclose_quantum_gate_angles() {
        // Quantum gate angle comparison use case (from find_cliffs.py)
        let pi = std::f64::consts::PI;

        // Check if angle is exactly π/2
        let angle = pi / 2.0;
        assert!(angle.isclose(&(pi / 2.0), 0.0, 1e-12));

        // Check if angle is close to π/2 with tight tolerance
        let theta = pi / 2.0 + 1e-13;
        assert!(theta.isclose(&(pi / 2.0), 0.0, 1e-12));

        // Check if angle is NOT close to π/2
        let theta = pi / 2.0 + 1e-10;
        assert!(!theta.isclose(&(pi / 2.0), 0.0, 1e-12));
    }

    #[test]
    fn test_isclose_special_nan() {
        // NaN should not be close to anything, including itself
        assert!(!f64::NAN.isclose(&f64::NAN, 1e-5, 1e-8));
        assert!(!f64::NAN.isclose(&1.0, 1e-5, 1e-8));
        assert!(!1.0.isclose(&f64::NAN, 1e-5, 1e-8));
    }

    #[test]
    fn test_isclose_special_infinity() {
        // Infinity with same sign should be close
        assert!(f64::INFINITY.isclose(&f64::INFINITY, 1e-5, 1e-8));
        assert!(f64::NEG_INFINITY.isclose(&f64::NEG_INFINITY, 1e-5, 1e-8));

        // Infinity with different sign should not be close
        assert!(!f64::INFINITY.isclose(&f64::NEG_INFINITY, 1e-5, 1e-8));

        // Infinity and finite should not be close
        assert!(!f64::INFINITY.isclose(&1e308, 1e-5, 1e-8));
        assert!(!f64::NEG_INFINITY.isclose(&(-1e308), 1e-5, 1e-8));
    }

    #[test]
    fn test_isclose_zero_tolerance() {
        // With zero tolerances, only exact equality should pass
        assert!(1.0_f64.isclose(&1.0, 0.0, 0.0));
        assert!(!1.0_f64.isclose(&(1.0 + 1e-15), 0.0, 0.0));
    }

    #[test]
    fn test_isclose_asymmetric() {
        // Test that tolerance is relative to b, not a
        assert!(100.0_f64.isclose(&100.001, 1e-5, 0.0));
        assert!(0.1_f64.isclose(&0.10001, 1e-3, 0.0));
    }

    // Tests for where_
    #[test]
    #[allow(clippy::float_cmp)]
    fn test_where_true() {
        assert_eq!(where_(true, 10.0, 20.0), 10.0);
        assert_eq!(where_(true, "yes", "no"), "yes");
        assert_eq!(where_(true, 1, 2), 1);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_where_false() {
        assert_eq!(where_(false, 10.0, 20.0), 20.0);
        assert_eq!(where_(false, "yes", "no"), "no");
        assert_eq!(where_(false, 1, 2), 2);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_where_odd_even() {
        // Typical use case from threshold analysis
        let dist = 5;
        let result = where_(dist % 2 == 1, f64::from(dist) * 2.0, f64::from(dist) / 2.0);
        assert_eq!(result, 10.0); // dist is odd, so returns dist * 2.0

        let dist = 4;
        let result = where_(dist % 2 == 1, f64::from(dist) * 2.0, f64::from(dist) / 2.0);
        assert_eq!(result, 2.0); // dist is even, so returns dist / 2.0
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_where_computations() {
        // Test with actual computations
        let condition = 3 > 2;
        let result = where_(condition, 3.0_f64.powi(2), 2.0_f64.powi(3));
        assert_eq!(result, 9.0);

        let condition = 3 < 2;
        let result = where_(condition, 3.0_f64.powi(2), 2.0_f64.powi(3));
        assert_eq!(result, 8.0);
    }

    // Tests for Where trait
    #[test]
    #[allow(clippy::float_cmp)]
    fn test_where_trait_scalar_bool() {
        // Test bool.where_() method for scalars
        assert_eq!(true.where_(&10.0, &20.0), 10.0);
        assert_eq!(false.where_(&10.0, &20.0), 20.0);

        // Test with different types
        assert_eq!(true.where_(&"yes", &"no"), "yes");
        assert_eq!(false.where_(&1, &2), 2);
    }

    #[test]
    fn test_where_basic() {
        use crate::prelude::array;

        // Basic element-wise conditional selection
        let condition = array![true, false, true, false];
        let x = array![10.0, 20.0, 30.0, 40.0];
        let y = array![100.0, 200.0, 300.0, 400.0];

        let result = condition.where_(&x, &y);
        assert_eq!(result, array![10.0, 200.0, 30.0, 400.0]);
    }

    #[test]
    fn test_where_all_true() {
        use crate::prelude::array;

        // All conditions true - should select all from x
        let condition = array![true, true, true];
        let x = array![1.0, 2.0, 3.0];
        let y = array![10.0, 20.0, 30.0];

        let result = condition.where_(&x, &y);
        assert_eq!(result, array![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_where_all_false() {
        use crate::prelude::array;

        // All conditions false - should select all from y
        let condition = array![false, false, false];
        let x = array![1.0, 2.0, 3.0];
        let y = array![10.0, 20.0, 30.0];

        let result = condition.where_(&x, &y);
        assert_eq!(result, array![10.0, 20.0, 30.0]);
    }

    #[test]
    fn test_where_2d() {
        use crate::prelude::array;

        // 2D array selection
        let condition = array![[true, false], [false, true]];
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![[10.0, 20.0], [30.0, 40.0]];

        let result = condition.where_(&x, &y);
        assert_eq!(result, array![[1.0, 20.0], [30.0, 4.0]]);
    }

    #[test]
    fn test_where_integers() {
        use crate::prelude::array;

        // Test with integer types
        let condition = array![true, false, true];
        let x = array![1, 2, 3];
        let y = array![10, 20, 30];

        let result = condition.where_(&x, &y);
        assert_eq!(result, array![1, 20, 3]);
    }

    // Tests for allclose()
    #[test]
    fn test_allclose_identical_arrays() {
        use crate::prelude::array;

        let a = array![1.0, 2.0, 3.0];
        let b = array![1.0, 2.0, 3.0];
        assert!(super::allclose(&a, &b, 1e-5, 1e-8, false));
    }

    #[test]
    fn test_allclose_within_tolerance() {
        use crate::prelude::array;

        // 1D arrays
        let a = array![1.0, 2.0, 3.0];
        let b = array![1.00001, 2.00001, 3.00001];
        assert!(super::allclose(&a, &b, 1e-4, 1e-8, false));

        // 2D arrays (quantum gate matrices)
        let gate1 = array![[1.0, 0.0], [0.0, 1.0]];
        let gate2 = array![[1.00001, 0.0], [0.0, 0.99999]];
        assert!(super::allclose(&gate1, &gate2, 1e-4, 1e-8, false));
    }

    #[test]
    fn test_allclose_outside_tolerance() {
        use crate::prelude::array;

        let a = array![1.0, 2.0, 3.0];
        let b = array![1.0, 2.0, 10.0]; // Last element too far
        assert!(!super::allclose(&a, &b, 1e-5, 1e-8, false));
    }

    #[test]
    fn test_allclose_different_shapes() {
        use crate::prelude::array;

        let a = array![1.0, 2.0, 3.0];
        let b = array![1.0, 2.0];
        assert!(!super::allclose(&a, &b, 1e-5, 1e-8, false));
    }

    #[test]
    fn test_allclose_with_nan() {
        use crate::prelude::array;

        let a = array![1.0, f64::NAN, 3.0];
        let b = array![1.0, f64::NAN, 3.0];

        // Without equal_nan, should return false
        assert!(!super::allclose(&a, &b, 1e-5, 1e-8, false));

        // With equal_nan=true, should return true
        assert!(super::allclose(&a, &b, 1e-5, 1e-8, true));
    }

    #[test]
    fn test_allclose_quantum_gate_matrices() {
        use crate::prelude::array;

        // Identity gate
        let identity1 = array![[1.0, 0.0], [0.0, 1.0]];
        let identity2 = array![[0.99999, 0.0], [0.0, 1.00001]];
        assert!(super::allclose(&identity1, &identity2, 1e-4, 1e-8, false));

        // Pauli X gate
        let x_gate1 = array![[0.0, 1.0], [1.0, 0.0]];
        let x_gate2 = array![[0.0, 0.99999], [1.00001, 0.0]];
        assert!(super::allclose(&x_gate1, &x_gate2, 1e-4, 1e-8, false));

        // Different gates should not be close
        assert!(!super::allclose(&identity1, &x_gate1, 1e-5, 1e-8, false));
    }

    #[test]
    fn test_allclose_complex_arrays() {
        use crate::prelude::array;
        use num_complex::Complex64;

        let a = array![Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0),];
        let b = array![Complex64::new(1.00001, 0.0), Complex64::new(0.0, 1.00001),];
        assert!(super::allclose(&a, &b, 1e-4, 1e-8, false));

        let c = array![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 2.0), // Significantly different
        ];
        assert!(!super::allclose(&a, &c, 1e-5, 1e-8, false));
    }

    #[test]
    fn test_array_equal_same_arrays() {
        use crate::prelude::array;

        let a = array![1.0, 2.0, 3.0];
        let b = array![1.0, 2.0, 3.0];
        assert!(super::array_equal(&a, &b, false));
    }

    #[test]
    fn test_array_equal_different_values() {
        use crate::prelude::array;

        let a = array![1.0, 2.0, 3.0];
        let b = array![1.0, 2.0, 4.0];
        assert!(!super::array_equal(&a, &b, false));
    }

    #[test]
    fn test_array_equal_different_shapes() {
        use crate::prelude::array;

        let a = array![1.0, 2.0, 3.0, 4.0];
        let b = array![1.0, 2.0, 3.0];
        // Different lengths, should be false
        assert!(!super::array_equal(&a, &b, false));
    }

    #[test]
    fn test_array_equal_with_nan_default() {
        use crate::prelude::array;

        let a = array![1.0, f64::NAN, 3.0];
        let b = array![1.0, f64::NAN, 3.0];
        // With equal_nan=false, NaN != NaN
        assert!(!super::array_equal(&a, &b, false));
    }

    #[test]
    fn test_array_equal_with_nan_true() {
        use crate::prelude::array;

        let a = array![1.0, f64::NAN, 3.0];
        let b = array![1.0, f64::NAN, 3.0];
        // With equal_nan=true, NaN == NaN
        assert!(super::array_equal(&a, &b, true));
    }

    #[test]
    fn test_array_equal_complex() {
        use crate::prelude::array;
        use num_complex::Complex64;

        let a = array![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)];
        let b = array![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)];
        assert!(super::array_equal(&a, &b, false));

        let c = array![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.1)];
        assert!(!super::array_equal(&a, &c, false));
    }

    // Note: Integer arrays don't support equal_nan parameter since integers can't be NaN
    // For integers, use direct comparison or allclose instead

    #[test]
    fn test_array_equal_2d() {
        use crate::prelude::array;

        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[1.0, 2.0], [3.0, 4.0]];
        assert!(super::array_equal(&a, &b, false));

        let c = array![[1.0, 2.0], [3.0, 5.0]];
        assert!(!super::array_equal(&a, &c, false));
    }

    // Tests for relative_eq
    #[test]
    fn test_relative_eq_exact() {
        assert!(super::relative_eq(1.0, 1.0, 1e-10));
        assert!(super::relative_eq(0.0, 0.0, 1e-10));
        assert!(super::relative_eq(-5.0, -5.0, 1e-10));
    }

    #[test]
    #[allow(clippy::unreadable_literal)] // Test values - exact digits matter
    fn test_relative_eq_within_tolerance() {
        assert!(super::relative_eq(1.0, 1.0000001, 1e-6));
        assert!(super::relative_eq(100.0, 100.0001, 1e-5));
        assert!(super::relative_eq(0.5, 0.50000001, 1e-6));
    }

    #[test]
    fn test_relative_eq_outside_tolerance() {
        assert!(!super::relative_eq(1.0, 1.1, 1e-6));
        assert!(!super::relative_eq(1.0, 2.0, 1e-6));
        assert!(!super::relative_eq(100.0, 101.0, 1e-5));
    }

    #[test]
    #[allow(clippy::unreadable_literal)] // Test values - exact digits matter
    fn test_relative_eq_near_zero() {
        // Near zero, relative comparison can be tricky
        // Our implementation handles this by checking absolute tolerance first
        assert!(super::relative_eq(1e-10, 1e-10, 1e-6));
        assert!(super::relative_eq(0.0, 0.0, 1e-10));
        // These are the cases that failed before - tiny floating point errors near zero
        assert!(super::relative_eq(6.123233995736766e-17, 0.0, 1e-9));
        assert!(super::relative_eq(-5.551115123125783e-17, 0.0, 1e-9));
        assert!(super::relative_eq(1e-17, 0.0, 1e-9));
    }

    #[test]
    fn test_relative_eq_nan() {
        assert!(!super::relative_eq(f64::NAN, f64::NAN, 1e-6));
        assert!(!super::relative_eq(f64::NAN, 1.0, 1e-6));
        assert!(!super::relative_eq(1.0, f64::NAN, 1e-6));
    }

    #[test]
    fn test_relative_eq_infinity() {
        assert!(super::relative_eq(f64::INFINITY, f64::INFINITY, 1e-6));
        assert!(super::relative_eq(
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
            1e-6
        ));
        assert!(!super::relative_eq(f64::INFINITY, f64::NEG_INFINITY, 1e-6));
        assert!(!super::relative_eq(f64::INFINITY, 1e308, 1e-6));
    }

    #[test]
    #[allow(clippy::unreadable_literal)] // Test values - exact digits matter
    fn test_assert_relative_eq_macro() {
        crate::assert_relative_eq!(1.0, 1.0000001, epsilon = 1e-6);
        crate::assert_relative_eq!(100.0, 100.0001, epsilon = 1e-5);
        crate::assert_relative_eq!(0.5, 0.5, epsilon = 1e-10);
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn test_assert_relative_eq_macro_fails() {
        crate::assert_relative_eq!(1.0, 2.0, epsilon = 1e-6);
    }

    // Tests for all_axis / any_axis
    #[test]
    fn test_all_axis_0() {
        use crate::prelude::array;
        let arr = array![[true, false], [true, true]];
        assert_eq!(all_axis(&arr, Axis(0)), array![true, false]);
    }

    #[test]
    fn test_all_axis_1() {
        use crate::prelude::array;
        let arr = array![[true, false], [true, true]];
        assert_eq!(all_axis(&arr, Axis(1)), array![false, true]);
    }

    #[test]
    fn test_any_axis_0() {
        use crate::prelude::array;
        let arr = array![[true, false], [false, false]];
        assert_eq!(any_axis(&arr, Axis(0)), array![true, false]);
    }

    #[test]
    fn test_any_axis_1() {
        use crate::prelude::array;
        let arr = array![[true, false], [false, false]];
        assert_eq!(any_axis(&arr, Axis(1)), array![true, false]);
    }

    #[test]
    fn test_all_axis_all_true() {
        use crate::prelude::array;
        let arr = array![[true, true], [true, true]];
        assert_eq!(all_axis(&arr, Axis(0)), array![true, true]);
        assert_eq!(all_axis(&arr, Axis(1)), array![true, true]);
    }

    #[test]
    fn test_all_axis_all_false() {
        use crate::prelude::array;
        let arr = array![[false, false], [false, false]];
        assert_eq!(all_axis(&arr, Axis(0)), array![false, false]);
        assert_eq!(all_axis(&arr, Axis(1)), array![false, false]);
    }

    #[test]
    fn test_any_axis_all_true() {
        use crate::prelude::array;
        let arr = array![[true, true], [true, true]];
        assert_eq!(any_axis(&arr, Axis(0)), array![true, true]);
        assert_eq!(any_axis(&arr, Axis(1)), array![true, true]);
    }

    #[test]
    fn test_any_axis_all_false() {
        use crate::prelude::array;
        let arr = array![[false, false], [false, false]];
        assert_eq!(any_axis(&arr, Axis(0)), array![false, false]);
        assert_eq!(any_axis(&arr, Axis(1)), array![false, false]);
    }
}
