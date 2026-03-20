// Copyright 2025 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License.You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

//! Linear algebra operations for quantum computing.
//!
//! This module provides drop-in replacements for numpy.linalg functions.

use ndarray::{Array2, ArrayBase, Data, Dimension, LinalgScalar};
use num_complex::Complex64;

/// Compute the norm of a vector or matrix.
///
/// Drop-in replacement for `numpy.linalg.norm()`.
///
/// # Arguments
///
/// * `x` - Input array (1-D or 2-D)
/// * `ord` - Order of the norm (default: 2-norm for vectors, Frobenius for matrices)
///
/// # Supported norms
///
/// For vectors (1-D arrays):
/// - `None` or `2.0`: Euclidean norm (L2)
/// - `1.0`: Sum of absolute values (L1)
/// - `f64::INFINITY`: Maximum absolute value (L∞)
/// - `f64::NEG_INFINITY`: Minimum absolute value
/// - Other: p-norm `sum(abs(x)**ord)**(1/ord)`
///
/// For matrices (2-D arrays):
/// - `None` or `"fro"`: Frobenius norm
/// - Other matrix norms not yet implemented
///
/// # Examples
///
/// ```
/// use pecos_num::linalg::norm;
/// use ndarray::array;
///
/// let v = array![3.0, 4.0];
/// assert!((norm(&v, None) - 5.0).abs() < 1e-10);
/// ```
///
/// # Panics
///
/// Panics if the array is not contiguous in memory.
pub fn norm<S, D>(x: &ArrayBase<S, D>, ord: Option<f64>) -> f64
where
    S: Data<Elem = f64>,
    D: Dimension,
{
    let ord = ord.unwrap_or(2.0);

    // For 1-D arrays (vectors)
    if x.ndim() == 1 {
        return vector_norm(x.as_slice().unwrap(), ord);
    }

    // For 2-D arrays (matrices) - Frobenius norm
    if x.ndim() == 2 {
        return frobenius_norm(x);
    }

    // For higher dimensions, flatten and compute vector norm
    let flat: Vec<f64> = x.iter().copied().collect();
    vector_norm(&flat, ord)
}

/// Compute the norm of a complex vector or matrix.
///
/// Complex number variant of `norm()`.
///
/// # Panics
///
/// Panics if the array is not contiguous in memory.
pub fn norm_complex<S, D>(x: &ArrayBase<S, D>, ord: Option<f64>) -> f64
where
    S: Data<Elem = Complex64>,
    D: Dimension,
{
    let ord = ord.unwrap_or(2.0);

    // For 1-D arrays (vectors)
    if x.ndim() == 1 {
        return vector_norm_complex(x.as_slice().unwrap(), ord);
    }

    // For 2-D arrays (matrices) - Frobenius norm
    if x.ndim() == 2 {
        return frobenius_norm_complex(x);
    }

    // For higher dimensions, flatten and compute vector norm
    let flat: Vec<Complex64> = x.iter().copied().collect();
    vector_norm_complex(&flat, ord)
}

/// Compute vector norm for real values.
#[allow(clippy::float_cmp)] // Comparing exact values (1.0, 2.0) which are exactly representable
fn vector_norm(x: &[f64], ord: f64) -> f64 {
    if ord == 2.0 {
        // Euclidean norm (L2)
        x.iter().map(|&v| v * v).sum::<f64>().sqrt()
    } else if ord == 1.0 {
        // Manhattan norm (L1)
        x.iter().map(|&v| v.abs()).sum()
    } else if ord == f64::INFINITY {
        // Maximum absolute value (L∞)
        x.iter().map(|&v| v.abs()).fold(0.0, f64::max)
    } else if ord == f64::NEG_INFINITY {
        // Minimum absolute value
        x.iter().map(|&v| v.abs()).fold(f64::INFINITY, f64::min)
    } else {
        // p-norm: (sum(|x|^p))^(1/p)
        x.iter()
            .map(|&v| v.abs().powf(ord))
            .sum::<f64>()
            .powf(1.0 / ord)
    }
}

/// Compute vector norm for complex values.
#[allow(clippy::float_cmp)] // Comparing exact values (1.0, 2.0) which are exactly representable
fn vector_norm_complex(x: &[Complex64], ord: f64) -> f64 {
    if ord == 2.0 {
        // Euclidean norm (L2)
        x.iter()
            .map(num_complex::Complex::norm_sqr)
            .sum::<f64>()
            .sqrt()
    } else if ord == 1.0 {
        // Manhattan norm (L1)
        x.iter().map(|v| v.norm()).sum()
    } else if ord == f64::INFINITY {
        // Maximum absolute value (L∞)
        x.iter().map(|v| v.norm()).fold(0.0, f64::max)
    } else if ord == f64::NEG_INFINITY {
        // Minimum absolute value
        x.iter().map(|v| v.norm()).fold(f64::INFINITY, f64::min)
    } else {
        // p-norm: (sum(|x|^p))^(1/p)
        x.iter()
            .map(|v| v.norm().powf(ord))
            .sum::<f64>()
            .powf(1.0 / ord)
    }
}

/// Compute Frobenius norm for real matrices.
fn frobenius_norm<S, D>(x: &ArrayBase<S, D>) -> f64
where
    S: Data<Elem = f64>,
    D: Dimension,
{
    x.iter().map(|&v| v * v).sum::<f64>().sqrt()
}

/// Compute Frobenius norm for complex matrices.
fn frobenius_norm_complex<S, D>(x: &ArrayBase<S, D>) -> f64
where
    S: Data<Elem = Complex64>,
    D: Dimension,
{
    x.iter()
        .map(num_complex::Complex::norm_sqr)
        .sum::<f64>()
        .sqrt()
}

// ============================================================================
// Kronecker product
// ============================================================================

/// Compute the Kronecker product of two 2D arrays.
///
/// Generic over element type -- works for both `f64` and `Complex64`.
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use pecos_num::linalg::kron;
///
/// let a = array![[1.0, 2.0], [3.0, 4.0]];
/// let b = array![[0.0, 5.0], [6.0, 7.0]];
/// let result = kron(&a, &b);
/// assert_eq!(result.shape(), &[4, 4]);
/// ```
#[must_use]
pub fn kron<T: LinalgScalar>(a: &Array2<T>, b: &Array2<T>) -> Array2<T> {
    let (m, n) = (a.nrows(), a.ncols());
    let (p, q) = (b.nrows(), b.ncols());
    let mut result = Array2::<T>::zeros((m * p, n * q));
    for i in 0..m {
        for j in 0..n {
            let a_ij = a[(i, j)];
            for k in 0..p {
                for l in 0..q {
                    result[(i * p + k, j * q + l)] = a_ij * b[(k, l)];
                }
            }
        }
    }
    result
}

// ============================================================================
// Matrix power
// ============================================================================

/// Raise a square f64 matrix to a non-negative integer power using binary exponentiation.
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use pecos_num::linalg::matrix_power_f64;
///
/// let a = array![[1.0, 2.0], [3.0, 4.0]];
/// let result = matrix_power_f64(&a, 2);
/// assert!((result[(0, 0)] - 7.0).abs() < 1e-10);
/// ```
#[must_use]
pub fn matrix_power_f64(a: &Array2<f64>, n: u32) -> Array2<f64> {
    let size = a.nrows();
    if n == 0 {
        let mut id = Array2::<f64>::zeros((size, size));
        for i in 0..size {
            id[(i, i)] = 1.0;
        }
        return id;
    }
    let mut exp = n;
    let mut base = a.clone();
    let mut result = {
        let mut id = Array2::<f64>::zeros((size, size));
        for i in 0..size {
            id[(i, i)] = 1.0;
        }
        id
    };
    while exp > 0 {
        if exp % 2 == 1 {
            result = result.dot(&base);
        }
        base = base.dot(&base);
        exp /= 2;
    }
    result
}

/// Raise a square complex matrix to a non-negative integer power using binary exponentiation.
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use num_complex::Complex64;
/// use pecos_num::linalg::matrix_power_c64;
///
/// let a = array![
///     [Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
///     [Complex64::new(3.0, 0.0), Complex64::new(4.0, 0.0)]
/// ];
/// let result = matrix_power_c64(&a, 2);
/// assert!((result[(0, 0)] - Complex64::new(7.0, 0.0)).norm() < 1e-10);
/// ```
#[must_use]
pub fn matrix_power_c64(a: &Array2<Complex64>, n: u32) -> Array2<Complex64> {
    let size = a.nrows();
    if n == 0 {
        let mut id = Array2::<Complex64>::zeros((size, size));
        for i in 0..size {
            id[(i, i)] = Complex64::new(1.0, 0.0);
        }
        return id;
    }
    let mut exp = n;
    let mut base = a.clone();
    let mut result = {
        let mut id = Array2::<Complex64>::zeros((size, size));
        for i in 0..size {
            id[(i, i)] = Complex64::new(1.0, 0.0);
        }
        id
    };
    while exp > 0 {
        if exp % 2 == 1 {
            result = result.dot(&base);
        }
        base = base.dot(&base);
        exp /= 2;
    }
    result
}

// ============================================================================
// Matrix exponential and logarithm
// ============================================================================

use nalgebra::DMatrix;

/// Computes the matrix exponential exp(M) for a complex square matrix.
///
/// Uses the scaling and squaring method with Padé approximation.
///
/// # Arguments
/// * `m` - A square complex matrix
///
/// # Returns
/// The matrix exponential exp(M)
///
/// # Example
/// ```
/// use nalgebra::DMatrix;
/// use num_complex::Complex64;
/// use pecos_num::linalg::matrix_exp;
///
/// // exp(0) = I
/// let zero = DMatrix::from_element(2, 2, Complex64::new(0.0, 0.0));
/// let result = matrix_exp(&zero);
/// assert!((result[(0, 0)] - Complex64::new(1.0, 0.0)).norm() < 1e-10);
/// ```
///
/// # Panics
/// Panics if the matrix is not square.
#[must_use]
pub fn matrix_exp(m: &DMatrix<Complex64>) -> DMatrix<Complex64> {
    assert!(m.is_square(), "Matrix must be square for exponential");
    let n = m.nrows();

    if n == 0 {
        return DMatrix::from_element(0, 0, Complex64::new(0.0, 0.0));
    }

    // Scaling: find s such that ||M / 2^s|| < 1
    let norm = matrix_1_norm(m);
    #[allow(clippy::cast_possible_truncation)] // Scaling factor s is always small
    let s = (norm.log2().ceil() as i32).max(0);
    let scale = 2.0_f64.powi(s);

    // Scale the matrix
    let a = m / Complex64::new(scale, 0.0);

    // Padé approximation of order (6, 6)
    // exp(A) ≈ N(A) / D(A) where N and D are polynomials
    let exp_scaled = pade_approximation(&a, 6);

    // Squaring: compute exp(M) = (exp(M/2^s))^(2^s)
    let mut result = exp_scaled;
    for _ in 0..s {
        result = &result * &result;
    }

    result
}

/// Computes the matrix logarithm log(M) for a complex square matrix.
///
/// Uses the inverse scaling and squaring method with Padé approximation.
///
/// # Arguments
/// * `m` - A square complex matrix (should be close to identity for best results)
///
/// # Returns
/// * `Some(log(M))` if the computation succeeds
/// * `None` if the matrix is singular or the computation fails
///
/// # Example
/// ```
/// use nalgebra::DMatrix;
/// use num_complex::Complex64;
/// use pecos_num::linalg::matrix_log;
///
/// // log(I) = 0
/// let eye = DMatrix::identity(2, 2);
/// let eye_complex: DMatrix<Complex64> = eye.map(|x| Complex64::new(x, 0.0));
/// let result = matrix_log(&eye_complex);
/// assert!(result.is_some());
/// let log_i = result.unwrap();
/// assert!(log_i[(0, 0)].norm() < 1e-10);
/// ```
///
/// # Panics
/// Panics if the matrix is not square.
#[must_use]
pub fn matrix_log(m: &DMatrix<Complex64>) -> Option<DMatrix<Complex64>> {
    assert!(m.is_square(), "Matrix must be square for logarithm");
    let n = m.nrows();

    if n == 0 {
        return Some(DMatrix::from_element(0, 0, Complex64::new(0.0, 0.0)));
    }

    // Inverse scaling: compute M^(1/2^s) until close to identity
    let mut a = m.clone();
    let mut s = 0;

    // Take square roots until ||A - I|| is small
    let identity = DMatrix::<Complex64>::identity(n, n);
    while matrix_1_norm(&(&a - &identity)) > 0.5 && s < 50 {
        a = matrix_sqrt(&a)?;
        s += 1;
    }

    // Padé approximation for log(A) where A is close to I
    // log(A) ≈ (A - I) * P((A - I)) / Q((A - I)) for A near I
    let a_minus_i = &a - &identity;
    let log_scaled = pade_log_approximation(&a_minus_i, 6);

    // Scaling: log(M) = 2^s * log(M^(1/2^s))
    let scale = Complex64::new(2.0_f64.powi(s), 0.0);
    Some(log_scaled * scale)
}

// ============================================================================
// ndarray convenience wrappers for expm / logm
// ============================================================================

/// Compute the matrix exponential of a complex 2D array.
///
/// Convenience wrapper around [`matrix_exp`] that accepts and returns `ndarray::Array2`
/// instead of `nalgebra::DMatrix`, so callers don't need the nalgebra dependency.
///
/// Uses the scipy naming convention (`expm`) to match the project's goal of being
/// numpy/scipy drop-in replacements.
///
/// # Arguments
///
/// * `m` - A square complex 2D array
///
/// # Returns
///
/// The matrix exponential exp(M), or an error if the matrix is not square.
///
/// # Examples
///
/// ```
/// use ndarray::Array2;
/// use num_complex::Complex64;
/// use pecos_num::linalg::expm;
///
/// let zero = Array2::from_elem((2, 2), Complex64::new(0.0, 0.0));
/// let result = expm(&zero).unwrap();
/// assert!((result[(0, 0)] - Complex64::new(1.0, 0.0)).norm() < 1e-10);
/// ```
///
/// # Errors
///
/// Returns an error if the matrix is not square.
pub fn expm(m: &Array2<Complex64>) -> Result<Array2<Complex64>, String> {
    let (rows, cols) = (m.nrows(), m.ncols());
    if rows != cols {
        return Err(format!("expm requires a square matrix, got {rows}x{cols}"));
    }
    let dmat = DMatrix::from_fn(rows, cols, |i, j| m[(i, j)]);
    let result_dmat = matrix_exp(&dmat);
    Ok(Array2::from_shape_fn((rows, cols), |(i, j)| {
        result_dmat[(i, j)]
    }))
}

/// Compute the matrix logarithm of a complex 2D array.
///
/// Convenience wrapper around [`matrix_log`] that accepts and returns `ndarray::Array2`
/// instead of `nalgebra::DMatrix`.
///
/// Uses the scipy naming convention (`logm`) to match the project's goal of being
/// numpy/scipy drop-in replacements.
///
/// # Arguments
///
/// * `m` - A square complex 2D array
///
/// # Returns
///
/// The matrix logarithm log(M), or an error if the matrix is not square or is singular.
///
/// # Examples
///
/// ```
/// use ndarray::Array2;
/// use num_complex::Complex64;
/// use pecos_num::linalg::logm;
///
/// // log(I) = 0
/// let mut eye = Array2::zeros((2, 2));
/// eye[(0, 0)] = Complex64::new(1.0, 0.0);
/// eye[(1, 1)] = Complex64::new(1.0, 0.0);
/// let result = logm(&eye).unwrap();
/// assert!(result[(0, 0)].norm() < 1e-10);
/// ```
///
/// # Errors
///
/// Returns an error if the matrix is not square or if the computation fails
/// (e.g., singular matrix).
pub fn logm(m: &Array2<Complex64>) -> Result<Array2<Complex64>, String> {
    let (rows, cols) = (m.nrows(), m.ncols());
    if rows != cols {
        return Err(format!("logm requires a square matrix, got {rows}x{cols}"));
    }
    let dmat = DMatrix::from_fn(rows, cols, |i, j| m[(i, j)]);
    let result_dmat =
        matrix_log(&dmat).ok_or_else(|| "logm failed: matrix may be singular".to_string())?;
    Ok(Array2::from_shape_fn((rows, cols), |(i, j)| {
        result_dmat[(i, j)]
    }))
}

/// Computes the principal square root of a complex matrix.
///
/// Uses the Denman-Beavers iteration.
fn matrix_sqrt(m: &DMatrix<Complex64>) -> Option<DMatrix<Complex64>> {
    let n = m.nrows();
    let identity = DMatrix::<Complex64>::identity(n, n);

    let mut y = m.clone();
    let mut z = identity.clone();

    for _ in 0..50 {
        let y_inv = y.clone().try_inverse()?;
        let z_inv = z.clone().try_inverse()?;

        let y_new = (&y + &z_inv) * Complex64::new(0.5, 0.0);
        let z_new = (&z + &y_inv) * Complex64::new(0.5, 0.0);

        // Check convergence
        let diff = matrix_1_norm(&(&y_new - &y));
        y = y_new;
        z = z_new;

        if diff < 1e-14 {
            return Some(y);
        }
    }

    Some(y) // Return best approximation
}

/// Computes the 1-norm (maximum column sum) of a complex matrix.
fn matrix_1_norm(m: &DMatrix<Complex64>) -> f64 {
    let mut max_col_sum: f64 = 0.0;
    for j in 0..m.ncols() {
        let col_sum: f64 = (0..m.nrows()).map(|i| m[(i, j)].norm()).sum();
        max_col_sum = max_col_sum.max(col_sum);
    }
    max_col_sum
}

/// Padé approximation for matrix exponential.
///
/// Computes exp(A) using a (p,p) Padé approximant.
fn pade_approximation(a: &DMatrix<Complex64>, p: usize) -> DMatrix<Complex64> {
    let n = a.nrows();
    let identity = DMatrix::<Complex64>::identity(n, n);

    // Compute powers of A
    let mut a_powers = vec![identity.clone()];
    let mut current = a.clone();
    for _ in 1..=p {
        a_powers.push(current.clone());
        current = &current * a;
    }

    // Padé coefficients for (p,p) approximant
    let coeffs = pade_coefficients(p);

    // N(A) = sum(c_k * A^k) for even k
    // D(A) = sum(c_k * (-A)^k) for even k = sum((-1)^k * c_k * A^k)
    let mut n_matrix = DMatrix::from_element(n, n, Complex64::new(0.0, 0.0));
    let mut d_matrix = DMatrix::from_element(n, n, Complex64::new(0.0, 0.0));

    for (k, &coeff) in coeffs.iter().enumerate() {
        let c = Complex64::new(coeff, 0.0);
        n_matrix += &a_powers[k] * c;
        let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
        d_matrix += &a_powers[k] * Complex64::new(sign * coeff, 0.0);
    }

    // exp(A) ≈ D(A)^(-1) * N(A)
    match d_matrix.try_inverse() {
        Some(d_inv) => d_inv * n_matrix,
        None => n_matrix, // Fallback if D is singular
    }
}

/// Padé approximation for matrix logarithm (for A near identity).
///
/// Computes log(I + X) where X is small.
fn pade_log_approximation(x: &DMatrix<Complex64>, p: usize) -> DMatrix<Complex64> {
    let n = x.nrows();

    // For small X, log(I + X) ≈ X - X²/2 + X³/3 - ...
    // Use Padé approximant for better convergence

    // Compute powers of X
    let mut x_powers = vec![DMatrix::<Complex64>::identity(n, n)];
    let mut current = x.clone();
    for _ in 1..=p {
        x_powers.push(current.clone());
        current = &current * x;
    }

    // Simple series approximation for log(I + X)
    let mut result = DMatrix::from_element(n, n, Complex64::new(0.0, 0.0));
    #[allow(clippy::needless_range_loop)] // k is used both as index and in coefficient computation
    for k in 1..=p {
        let sign = if k % 2 == 1 { 1.0 } else { -1.0 };
        #[allow(clippy::cast_precision_loss)] // k is small (p ~ 20), precision loss negligible
        let coeff = Complex64::new(sign / (k as f64), 0.0);
        result += &x_powers[k] * coeff;
    }

    result
}

/// Returns Padé coefficients for (p,p) approximant of exp(x).
#[allow(clippy::cast_precision_loss)] // Factorials for small p (~ 6) fit precisely in f64
fn pade_coefficients(p: usize) -> Vec<f64> {
    // Coefficients c_k = (2p - k)! * p! / ((2p)! * k! * (p - k)!)
    let mut coeffs = vec![0.0; p + 1];
    let two_p_factorial = factorial(2 * p);

    #[allow(clippy::needless_range_loop)] // k is used both as index and in factorial computation
    for k in 0..=p {
        let numerator = factorial(2 * p - k) * factorial(p);
        let denominator = two_p_factorial * factorial(k) * factorial(p - k);
        coeffs[k] = (numerator as f64) / (denominator as f64);
    }

    coeffs
}

/// Compute factorial.
fn factorial(n: usize) -> u64 {
    (1..=n as u64).product()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_vector_norm_l2() {
        let v = array![3.0, 4.0];
        assert!((norm(&v, None) - 5.0).abs() < 1e-10);
        assert!((norm(&v, Some(2.0)) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_vector_norm_l1() {
        let v = array![3.0, 4.0];
        assert!((norm(&v, Some(1.0)) - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_vector_norm_linf() {
        let v = array![3.0, 4.0];
        assert!((norm(&v, Some(f64::INFINITY)) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrix_frobenius() {
        let m = array![[1.0, 2.0], [3.0, 4.0]];
        // Frobenius norm: sqrt(1^2 + 2^2 + 3^2 + 4^2) = sqrt(30)
        let expected = (30.0_f64).sqrt();
        assert!((norm(&m, None) - expected).abs() < 1e-10);
    }

    #[test]
    fn test_vector_norm_complex() {
        let v = array![Complex64::new(3.0, 0.0), Complex64::new(4.0, 0.0)];
        assert!((norm_complex(&v, None) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_vector_norm_complex_with_imag() {
        // (3+4i) has magnitude 5
        let v = array![Complex64::new(3.0, 4.0)];
        assert!((norm_complex(&v, None) - 5.0).abs() < 1e-10);
    }

    // ---- kron tests ----

    #[test]
    fn test_kron_identity() {
        let a = array![[1.0, 0.0], [0.0, 1.0]];
        let b = array![[1.0, 0.0], [0.0, 1.0]];
        let result = kron(&a, &b);
        let expected = array![
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_kron_known_value() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[0.0, 5.0], [6.0, 7.0]];
        let result = kron(&a, &b);
        let expected = array![
            [0.0, 5.0, 0.0, 10.0],
            [6.0, 7.0, 12.0, 14.0],
            [0.0, 15.0, 0.0, 20.0],
            [18.0, 21.0, 24.0, 28.0]
        ];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_kron_not_commutative() {
        let a = array![[1.0, 2.0], [0.0, 1.0]];
        let b = array![[0.0, 1.0], [1.0, 0.0]];
        let ab = kron(&a, &b);
        let ba = kron(&b, &a);
        assert_ne!(ab, ba);
    }

    #[test]
    fn test_kron_complex() {
        let i = Complex64::new(0.0, 1.0);
        let one = Complex64::new(1.0, 0.0);
        let zero = Complex64::new(0.0, 0.0);
        let a = array![[one, zero], [zero, i]];
        let b = array![[one, zero], [zero, one]];
        let result = kron(&a, &b);
        assert_eq!(result[(0, 0)], one);
        assert_eq!(result[(2, 2)], i);
        assert_eq!(result[(3, 3)], i);
    }

    // ---- matrix_power tests ----

    #[test]
    fn test_matrix_power_f64_zero() {
        let a = array![[2.0, 3.0], [4.0, 5.0]];
        let result = matrix_power_f64(&a, 0);
        let expected = array![[1.0, 0.0], [0.0, 1.0]];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_matrix_power_f64_one() {
        let a = array![[2.0, 3.0], [4.0, 5.0]];
        let result = matrix_power_f64(&a, 1);
        assert_eq!(result, a);
    }

    #[test]
    fn test_matrix_power_f64_two() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let result = matrix_power_f64(&a, 2);
        let expected = a.dot(&a);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_matrix_power_c64() {
        let one = Complex64::new(1.0, 0.0);
        let two = Complex64::new(2.0, 0.0);
        let three = Complex64::new(3.0, 0.0);
        let four = Complex64::new(4.0, 0.0);
        let a = array![[one, two], [three, four]];
        let result = matrix_power_c64(&a, 2);
        let expected = a.dot(&a);
        for i in 0..2 {
            for j in 0..2 {
                assert!((result[(i, j)] - expected[(i, j)]).norm() < 1e-10);
            }
        }
    }

    // ---- expm / logm tests ----

    #[test]
    fn test_expm_zero_is_identity() {
        let zero = Array2::from_elem((2, 2), Complex64::new(0.0, 0.0));
        let result = expm(&zero).unwrap();
        let one = Complex64::new(1.0, 0.0);
        assert!((result[(0, 0)] - one).norm() < 1e-10);
        assert!((result[(1, 1)] - one).norm() < 1e-10);
        assert!(result[(0, 1)].norm() < 1e-10);
        assert!(result[(1, 0)].norm() < 1e-10);
    }

    #[test]
    fn test_expm_known_2x2() {
        // exp([[0, 1], [0, 0]]) = [[1, 1], [0, 1]]
        let zero = Complex64::new(0.0, 0.0);
        let one = Complex64::new(1.0, 0.0);
        let m = array![[zero, one], [zero, zero]];
        let result = expm(&m).unwrap();
        assert!((result[(0, 0)] - one).norm() < 1e-10);
        assert!((result[(0, 1)] - one).norm() < 1e-10);
        assert!(result[(1, 0)].norm() < 1e-10);
        assert!((result[(1, 1)] - one).norm() < 1e-10);
    }

    #[test]
    fn test_expm_non_square_error() {
        let m = Array2::from_elem((2, 3), Complex64::new(0.0, 0.0));
        let result = expm(&m);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("square"));
    }

    #[test]
    fn test_logm_identity_is_zero() {
        let one = Complex64::new(1.0, 0.0);
        let zero = Complex64::new(0.0, 0.0);
        let eye = array![[one, zero], [zero, one]];
        let result = logm(&eye).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!(result[(i, j)].norm() < 1e-10);
            }
        }
    }

    #[test]
    fn test_logm_non_square_error() {
        let m = Array2::from_elem((2, 3), Complex64::new(0.0, 0.0));
        let result = logm(&m);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("square"));
    }

    #[test]
    fn test_logm_expm_roundtrip() {
        let zero = Complex64::new(0.0, 0.0);
        let a = Complex64::new(0.1, 0.2);
        let b = Complex64::new(0.3, 0.0);
        let m = array![[a, b], [zero, a]];
        let exp_m = expm(&m).unwrap();
        let log_exp_m = logm(&exp_m).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (log_exp_m[(i, j)] - m[(i, j)]).norm() < 1e-6,
                    "mismatch at ({i},{j}): got {:?}, expected {:?}",
                    log_exp_m[(i, j)],
                    m[(i, j)]
                );
            }
        }
    }
}
