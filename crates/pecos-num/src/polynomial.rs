// Copyright 2024 The PECOS Developers
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

//! Polynomial fitting and evaluation.
//!
//! This module provides implementations of polynomial operations,
//! compatible with numpy.polyfit and numpy.poly1d API.
//!
//! Uses Peroxide for linear algebra (SVD solving).

use ndarray::{Array1, ArrayView1};
use peroxide::fuga::{Col, LU, LinearAlgebra, MatrixTrait, Row, matrix};

/// Error type for polynomial operations.
#[derive(Debug, Clone)]
pub enum PolynomialError {
    /// Insufficient data points for the requested degree
    InsufficientData { num_points: usize, degree: usize },
    /// Numerical issues during fitting
    NumericalIssue { message: String },
    /// Linear algebra error
    LinAlgError { message: String },
}

impl std::fmt::Display for PolynomialError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InsufficientData { num_points, degree } => {
                write!(
                    f,
                    "Insufficient data: need at least {} points for degree {}, got {}",
                    degree + 1,
                    degree,
                    num_points
                )
            }
            Self::NumericalIssue { message } => write!(f, "Numerical issue: {message}"),
            Self::LinAlgError { message } => write!(f, "Linear algebra error: {message}"),
        }
    }
}

impl std::error::Error for PolynomialError {}

/// Fit a polynomial of given degree to data points.
///
/// This is a Rust implementation of numpy.polyfit.
///
/// # Arguments
///
/// * `x` - x-coordinates of data points
/// * `y` - y-coordinates of data points
/// * `deg` - Degree of the polynomial fit
///
/// # Returns
///
/// Returns the polynomial coefficients in decreasing order of degree.
/// For example, for degree 2: [c0, c1, c2] where y = c0*x^2 + c1*x + c2
///
/// # Errors
///
/// Returns an error if:
/// - Not enough data points for the requested degree
/// - Numerical issues during fitting
///
/// # Examples
///
/// ```
/// use pecos_num::polynomial::polyfit;
/// use ndarray::array;
///
/// // Fit y = 2x + 1
/// let x = array![0.0, 1.0, 2.0, 3.0];
/// let y = array![1.0, 3.0, 5.0, 7.0];
/// let coeffs = polyfit(x.view(), y.view(), 1).unwrap();
/// assert!((coeffs[0] - 2.0).abs() < 1e-10);  // slope
/// assert!((coeffs[1] - 1.0).abs() < 1e-10);  // intercept
/// ```
pub fn polyfit(
    x: ArrayView1<f64>,
    y: ArrayView1<f64>,
    deg: usize,
) -> Result<Array1<f64>, PolynomialError> {
    let n = x.len();

    if n != y.len() {
        return Err(PolynomialError::NumericalIssue {
            message: format!("x and y must have same length: x={n}, y={}", y.len()),
        });
    }

    if n < deg + 1 {
        return Err(PolynomialError::InsufficientData {
            num_points: n,
            degree: deg,
        });
    }

    // Build Vandermonde matrix using Peroxide
    // For degree 2: [[x0^2, x0, 1], [x1^2, x1, 1], ...]
    // Flatten to 1D vec for Peroxide's matrix constructor
    let mut vandermonde_data = Vec::with_capacity(n * (deg + 1));
    for &xi in x {
        for j in 0..=deg {
            // Cast is safe: polynomial degrees are always << i32::MAX
            #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
            let power = (deg - j) as i32;
            vandermonde_data.push(xi.powi(power));
        }
    }
    let vandermonde = matrix(vandermonde_data, n, deg + 1, Row);

    // Convert y to vector and then to column matrix
    let y_vec: Vec<f64> = y.iter().copied().collect();
    let y_mat = matrix(y_vec.clone(), n, 1, Col);

    // Solve least squares: coeffs = (A^T A)^{-1} A^T y
    // where A is the Vandermonde matrix
    let at = vandermonde.t(); // A^T
    let gram_matrix = &at * &vandermonde; // A^T A (Gram matrix)
    let at_y = &at * &y_mat; // A^T y

    // Solve the normal equations
    let at_y_vec: Vec<f64> = at_y.data.clone();
    let coeffs_vec = gram_matrix.solve(&at_y_vec, LU);

    // Convert back to ndarray
    let coeffs = Array1::from_vec(coeffs_vec);

    log::debug!("polyfit: fitted polynomial of degree {deg} with coeffs: {coeffs:?}");

    Ok(coeffs)
}

/// Polynomial class for evaluation.
///
/// This is a Rust implementation of numpy.poly1d functionality.
#[derive(Debug, Clone)]
pub struct Poly1d {
    /// Polynomial coefficients in decreasing order of degree
    /// For [c0, c1, c2]: y = c0*x^2 + c1*x + c2
    coeffs: Array1<f64>,
}

impl Poly1d {
    /// Create a new polynomial from coefficients.
    ///
    /// # Arguments
    ///
    /// * `coeffs` - Coefficients in decreasing order of degree
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_num::polynomial::Poly1d;
    /// use ndarray::array;
    ///
    /// // Create polynomial: 2x^2 + 3x + 1
    /// let p = Poly1d::new(array![2.0, 3.0, 1.0]);
    /// assert_eq!(p.eval(0.0), 1.0);  // p(0) = 1
    /// assert_eq!(p.eval(1.0), 6.0);  // p(1) = 2 + 3 + 1 = 6
    /// ```
    #[must_use]
    pub fn new(coeffs: Array1<f64>) -> Self {
        Self { coeffs }
    }

    /// Evaluate the polynomial at a given value.
    ///
    /// Uses Horner's method for efficient evaluation.
    ///
    /// # Arguments
    ///
    /// * `x` - Value at which to evaluate the polynomial
    ///
    /// # Returns
    ///
    /// The value of the polynomial at x
    ///
    /// # Panics
    ///
    /// Panics if the coefficient array is not in standard layout (contiguous in memory).
    #[must_use]
    pub fn eval(&self, x: f64) -> f64 {
        if self.coeffs.is_empty() {
            return 0.0;
        }

        // Horner's method: a0 + x(a1 + x(a2 + x(...)))
        let mut result = self.coeffs[0];
        for &coeff in &self.coeffs.as_slice().unwrap()[1..] {
            result = result * x + coeff;
        }
        result
    }

    /// Get the degree of the polynomial.
    #[must_use]
    pub fn degree(&self) -> usize {
        if self.coeffs.is_empty() {
            0
        } else {
            self.coeffs.len() - 1
        }
    }

    /// Get the coefficients.
    #[must_use]
    pub fn coefficients(&self) -> &Array1<f64> {
        &self.coeffs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_polyfit_linear() {
        // Fit y = 2x + 1
        let x = array![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = array![1.0, 3.0, 5.0, 7.0, 9.0];

        let coeffs = polyfit(x.view(), y.view(), 1).unwrap();

        assert_eq!(coeffs.len(), 2);
        assert!((coeffs[0] - 2.0).abs() < 1e-10); // slope
        assert!((coeffs[1] - 1.0).abs() < 1e-10); // intercept
    }

    #[test]
    fn test_polyfit_quadratic() {
        // Fit y = x^2 + 2x + 3
        let x = array![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = array![3.0, 6.0, 11.0, 18.0, 27.0];

        let coeffs = polyfit(x.view(), y.view(), 2).unwrap();

        assert_eq!(coeffs.len(), 3);
        assert!((coeffs[0] - 1.0).abs() < 1e-10); // x^2
        assert!((coeffs[1] - 2.0).abs() < 1e-10); // x
        assert!((coeffs[2] - 3.0).abs() < 1e-10); // constant
    }

    #[test]
    fn test_poly1d_eval() {
        // Test polynomial: 2x^2 + 3x + 1
        let p = Poly1d::new(array![2.0, 3.0, 1.0]);

        // Allow exact float comparison for simple polynomial evaluations with integer coefficients
        #[allow(clippy::float_cmp)]
        {
            assert_eq!(p.eval(0.0), 1.0); // p(0) = 1
            assert_eq!(p.eval(1.0), 6.0); // p(1) = 2 + 3 + 1 = 6
            assert_eq!(p.eval(2.0), 15.0); // p(2) = 8 + 6 + 1 = 15
            assert_eq!(p.eval(-1.0), 0.0); // p(-1) = 2 - 3 + 1 = 0
        }
    }

    #[test]
    fn test_polyfit_and_eval() {
        // Fit a polynomial and check evaluation
        let x = array![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = array![1.0, 3.0, 5.0, 7.0, 9.0];

        let coeffs = polyfit(x.view(), y.view(), 1).unwrap();
        let p = Poly1d::new(coeffs);

        // Check that polynomial evaluates correctly at training points
        for (xi, yi) in x.iter().zip(y.iter()) {
            assert!((p.eval(*xi) - yi).abs() < 1e-10);
        }
    }

    #[test]
    fn test_polyfit_insufficient_data() {
        let x = array![0.0, 1.0];
        let y = array![1.0, 2.0];

        // Try to fit degree 3 polynomial with only 2 points
        let result = polyfit(x.view(), y.view(), 3);
        assert!(matches!(
            result,
            Err(PolynomialError::InsufficientData { .. })
        ));
    }
}
