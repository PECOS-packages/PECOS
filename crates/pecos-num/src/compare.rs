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
//! This module provides trait-based comparison operations that work
//! across scalars, complex numbers, and arrays.

use ndarray::{Array, ArrayBase, Data, Dimension};
use num_complex::Complex64;

/// Trait for checking if values are NaN (Not a Number).
///
/// This trait provides a uniform interface for NaN checking across
/// different numeric types.
///
/// # Examples
///
/// ```
/// use pecos_num::prelude::*;
///
/// // Scalars
/// assert!(f64::NAN.isnan());
/// assert!(!5.0.isnan());
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

/// Check if a scalar f64 value is NaN.
impl IsNan for f64 {
    type Output = bool;

    #[inline]
    fn isnan(&self) -> bool {
        f64::is_nan(*self)
    }
}

/// Check if a complex scalar value is NaN.
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
/// This trait provides a uniform interface for tolerance-based comparison
/// across different numeric types. The tolerance check follows `NumPy`'s convention:
/// `|a - b| <= (atol + rtol * |b|)`
///
/// # Examples
///
/// ```
/// use pecos_num::prelude::*;
///
/// // Scalars
/// assert!(1.0.isclose(&1.00001, 1e-4, 1e-8));
/// assert!(!1.0.isclose(&1.1, 1e-5, 1e-8));
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
        assert!(!0.0.isnan());
        assert!(!1.0.isnan());
        assert!(!(-1.0).isnan());
        assert!(!42.5.isnan());
        assert!(!(-999.999).isnan());
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
        assert!(!0.0.isnan());
        assert!(!(-0.0).isnan());
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
        let valid_variance = 0.0025;
        let invalid_variance = f64::NAN;

        assert!(!valid_variance.isnan());
        assert!(invalid_variance.isnan());

        // Simulate variance validation loop
        let variances = [0.001, 0.002, f64::NAN, 0.004];
        let has_nan = variances.iter().any(super::IsNan::isnan);
        assert!(has_nan);
    }

    // Tests for IsClose trait
    #[test]
    fn test_isclose_exact() {
        // Exact equality
        assert!(1.0.isclose(&1.0, 1e-5, 1e-8));
        assert!(0.0.isclose(&0.0, 1e-5, 1e-8));
        assert!((-1.0).isclose(&-1.0, 1e-5, 1e-8));
    }

    #[test]
    fn test_isclose_within_tolerance() {
        // Within relative tolerance
        assert!(1.0.isclose(&1.00001, 1e-4, 1e-8));
        assert!(100.0.isclose(&100.001, 1e-4, 1e-8));

        // Within absolute tolerance
        assert!(1e-10.isclose(&2e-10, 0.0, 1e-9));
        assert!(0.0.isclose(&1e-9, 0.0, 1e-8));
    }

    #[test]
    fn test_isclose_outside_tolerance() {
        // Outside both tolerances
        assert!(!1.0.isclose(&1.1, 1e-5, 1e-8));
        assert!(!1.0.isclose(&2.0, 1e-5, 1e-8));
        assert!(!100.0.isclose(&101.0, 1e-5, 1e-8));
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
        assert!(1.0.isclose(&1.0, 0.0, 0.0));
        assert!(!1.0.isclose(&(1.0 + 1e-15), 0.0, 0.0));
    }

    #[test]
    fn test_isclose_asymmetric() {
        // Test that tolerance is relative to b, not a
        assert!(100.0.isclose(&100.001, 1e-5, 0.0));
        assert!(0.1.isclose(&0.10001, 1e-3, 0.0));
    }
}
