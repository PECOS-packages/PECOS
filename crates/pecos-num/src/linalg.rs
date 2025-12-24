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

use ndarray::{ArrayBase, Data, Dimension};
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
}
