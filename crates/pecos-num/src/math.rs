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

//! Mathematical functions for numerical analysis.
//!
//! This module provides trait-based mathematical operations that work
//! across scalars, complex numbers, and arrays.

use ndarray::{Array, ArrayBase, Data, Dimension};
use num_complex::Complex64;

// ============================================================================
// Trait Definitions
// ============================================================================

/// Trait for calculating exponential (e^x).
///
/// This trait provides a uniform interface for exponential operations across
/// different numeric types.
///
/// # Examples
///
/// ```
/// use pecos_num::prelude::*;
///
/// // Scalars
/// assert!((1.0.exp() - std::f64::consts::E).abs() < 1e-10);
///
/// // Complex numbers
/// let z = Complex64::new(0.0, std::f64::consts::PI);
/// let result = z.exp();
/// assert!((result.re - (-1.0)).abs() < 1e-10);
///
/// // Arrays
/// let arr = array![0.0, 1.0, 2.0];
/// let result = arr.exp();
/// assert!((result[1] - std::f64::consts::E).abs() < 1e-10);
/// ```
pub trait Exp {
    /// The output type when calculating exponential.
    type Output;

    /// Calculate e^self.
    fn exp(&self) -> Self::Output;
}

/// Trait for calculating square root.
///
/// This trait provides a uniform interface for square root operations.
///
/// # Examples
///
/// ```
/// use pecos_num::prelude::*;
///
/// // Scalars
/// assert_eq!(4.0.sqrt(), 2.0);
///
/// // Arrays
/// let arr = array![4.0, 9.0, 16.0];
/// assert_eq!(arr.sqrt(), array![2.0, 3.0, 4.0]);
/// ```
pub trait Sqrt {
    /// The output type when calculating square root.
    type Output;

    /// Calculate √self.
    fn sqrt(&self) -> Self::Output;
}

/// Trait for calculating power (base^exponent).
///
/// This trait provides a uniform interface for power operations.
///
/// # Examples
///
/// ```
/// use pecos_num::prelude::*;
///
/// // Scalars
/// assert!((2.0.power(3.0) - 8.0).abs() < 1e-10);
///
/// // Arrays
/// let arr = array![2.0, 3.0, 4.0];
/// let result = arr.power(2.0);
/// assert_eq!(result, array![4.0, 9.0, 16.0]);
/// ```
pub trait Power {
    /// The output type when calculating power.
    type Output;

    /// Calculate self^exponent.
    fn power(&self, exponent: f64) -> Self::Output;
}

/// Trait for calculating cosine.
///
/// This trait provides a uniform interface for cosine operations.
///
/// # Examples
///
/// ```
/// use pecos_num::prelude::*;
///
/// // Scalars
/// assert!((0.0_f64.cos() - 1.0).abs() < 1e-10);
///
/// // Arrays
/// let arr = array![0.0, PI / 2.0, PI];
/// let result = arr.cos();
/// assert!((result[0] - 1.0).abs() < 1e-10);
/// ```
pub trait Cos {
    /// The output type when calculating cosine.
    type Output;

    /// Calculate cos(self) where self is in radians.
    fn cos(&self) -> Self::Output;
}

/// Trait for calculating sine.
///
/// This trait provides a uniform interface for sine operations.
///
/// # Examples
///
/// ```
/// use pecos_num::prelude::*;
///
/// // Scalars
/// assert!((0.0_f64.sin()).abs() < 1e-10);
///
/// // Arrays
/// let arr = array![0.0, PI / 2.0, PI];
/// let result = arr.sin();
/// assert!((result[1] - 1.0).abs() < 1e-10);
/// ```
pub trait Sin {
    /// The output type when calculating sine.
    type Output;

    /// Calculate sin(self) where self is in radians.
    fn sin(&self) -> Self::Output;
}

// ============================================================================
// Scalar Implementations
// ============================================================================

/// Calculate exponential for f64 scalars.
impl Exp for f64 {
    type Output = f64;

    #[inline]
    fn exp(&self) -> f64 {
        f64::exp(*self)
    }
}

/// Calculate exponential for complex scalars.
impl Exp for Complex64 {
    type Output = Complex64;

    #[inline]
    fn exp(&self) -> Complex64 {
        Complex64::exp(*self)
    }
}

/// Calculate square root for f64 scalars.
impl Sqrt for f64 {
    type Output = f64;

    #[inline]
    fn sqrt(&self) -> f64 {
        f64::sqrt(*self)
    }
}

/// Calculate power for f64 scalars.
impl Power for f64 {
    type Output = f64;

    #[inline]
    fn power(&self, exponent: f64) -> f64 {
        self.powf(exponent)
    }
}

/// Calculate cosine for f64 scalars.
impl Cos for f64 {
    type Output = f64;

    #[inline]
    fn cos(&self) -> f64 {
        f64::cos(*self)
    }
}

/// Calculate sine for f64 scalars.
impl Sin for f64 {
    type Output = f64;

    #[inline]
    fn sin(&self) -> f64 {
        f64::sin(*self)
    }
}

// ============================================================================
// Array Implementations
// ============================================================================

/// Calculate exponential element-wise for arrays.
///
/// This generic implementation works for any element type that implements Exp.
impl<S, D, T> Exp for ArrayBase<S, D>
where
    S: Data<Elem = T>,
    T: Exp<Output = T> + Clone,
    D: Dimension,
{
    type Output = Array<T, D>;

    #[inline]
    fn exp(&self) -> Array<T, D> {
        self.mapv(|x| x.exp())
    }
}

/// Calculate square root element-wise for arrays.
///
/// This generic implementation works for any element type that implements Sqrt.
impl<S, D, T> Sqrt for ArrayBase<S, D>
where
    S: Data<Elem = T>,
    T: Sqrt<Output = T> + Clone,
    D: Dimension,
{
    type Output = Array<T, D>;

    #[inline]
    fn sqrt(&self) -> Array<T, D> {
        self.mapv(|x| x.sqrt())
    }
}

/// Calculate power element-wise for arrays.
///
/// This generic implementation works for any element type that implements Power.
impl<S, D, T> Power for ArrayBase<S, D>
where
    S: Data<Elem = T>,
    T: Power<Output = T> + Clone,
    D: Dimension,
{
    type Output = Array<T, D>;

    #[inline]
    fn power(&self, exponent: f64) -> Array<T, D> {
        self.mapv(|x| x.power(exponent))
    }
}

/// Calculate cosine element-wise for arrays.
///
/// This generic implementation works for any element type that implements Cos.
impl<S, D, T> Cos for ArrayBase<S, D>
where
    S: Data<Elem = T>,
    T: Cos<Output = T> + Clone,
    D: Dimension,
{
    type Output = Array<T, D>;

    #[inline]
    fn cos(&self) -> Array<T, D> {
        self.mapv(|x| x.cos())
    }
}

/// Calculate sine element-wise for arrays.
///
/// This generic implementation works for any element type that implements Sin.
impl<S, D, T> Sin for ArrayBase<S, D>
where
    S: Data<Elem = T>,
    T: Sin<Output = T> + Clone,
    D: Dimension,
{
    type Output = Array<T, D>;

    #[inline]
    fn sin(&self) -> Array<T, D> {
        self.mapv(|x| x.sin())
    }
}

// ============================================================================
// Legacy Functions (for backward compatibility during transition)
// ============================================================================

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
/// use pecos_num::math::power;
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
/// use pecos_num::math::sqrt;
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

/// Calculate the exponential (e^x) of a value.
///
/// Drop-in replacement for `numpy.exp()` for scalar values.
///
/// # Arguments
///
/// * `x` - Input value (exponent)
///
/// # Returns
///
/// e raised to the power of x (e^x), where e is Euler's number (≈2.71828).
///
/// # Examples
///
/// ```
/// use pecos_num::math::exp;
///
/// assert!((exp(0.0) - 1.0).abs() < 1e-10);
/// assert!((exp(1.0) - std::f64::consts::E).abs() < 1e-10);
/// assert!((exp(2.0) - 7.389_056_098_930_650).abs() < 1e-10);
/// assert!((exp(-1.0) - 0.367_879_441_171_442_3).abs() < 1e-10);
///
/// // Exponential decay use case (threshold analysis)
/// let decay_rate = 0.5;
/// let time = 2.0;
/// let amplitude = exp(-decay_rate * time);
/// assert!((amplitude - 0.367_879_441_171_442_3).abs() < 1e-10);
/// ```
#[must_use]
pub fn exp(x: f64) -> f64 {
    x.exp()
}

/// Calculate the exponential of a complex number.
///
/// Drop-in replacement for `numpy.exp()` for complex values.
/// Uses the num-complex crate for robust complex number arithmetic.
///
/// # Arguments
///
/// * `z` - Complex number input
///
/// # Returns
///
/// Complex64 result of e^z
///
/// # Examples
///
/// ```
/// use pecos_num::math::exp_complex;
/// use num_complex::Complex64;
/// use std::f64::consts::PI;
///
/// // e^(i*π) = -1 + 0i (Euler's identity)
/// let z = Complex64::new(0.0, PI);
/// let result = exp_complex(z);
/// assert!((result.re - (-1.0)).abs() < 1e-10);
/// assert!(result.im.abs() < 1e-10);
///
/// // e^(1+0i) = e + 0i
/// let z = Complex64::new(1.0, 0.0);
/// let result = exp_complex(z);
/// assert!((result.re - std::f64::consts::E).abs() < 1e-10);
/// assert!(result.im.abs() < 1e-10);
///
/// // Quantum gate phase: e^(i*π/2) = i
/// let z = Complex64::new(0.0, PI / 2.0);
/// let result = exp_complex(z);
/// assert!(result.re.abs() < 1e-10);
/// assert!((result.im - 1.0).abs() < 1e-10);
/// ```
#[must_use]
pub fn exp_complex(z: Complex64) -> Complex64 {
    z.exp()
}

/// Calculate the cosine of a value (in radians).
///
/// Drop-in replacement for `numpy.cos()` for scalar values.
///
/// # Arguments
///
/// * `x` - Input value in radians
///
/// # Returns
///
/// The cosine of x.
///
/// # Examples
///
/// ```
/// use pecos_num::math::cos;
///
/// assert!((cos(0.0) - 1.0).abs() < 1e-10);
/// assert!((cos(std::f64::consts::PI) - (-1.0)).abs() < 1e-10);
/// assert!((cos(std::f64::consts::PI / 2.0)).abs() < 1e-10);
/// assert!((cos(std::f64::consts::PI / 4.0) - 0.707_106_781_186_547_5).abs() < 1e-10);
///
/// // Quantum gate construction use case
/// let theta = std::f64::consts::PI / 3.0;
/// let c = cos(theta * 0.5);
/// assert!((c - 0.866_025_403_784_438_7).abs() < 1e-10);
/// ```
#[must_use]
pub fn cos(x: f64) -> f64 {
    x.cos()
}

/// Calculate the sine of a value (in radians).
///
/// Drop-in replacement for `numpy.sin()` for scalar values.
///
/// # Arguments
///
/// * `x` - Input value in radians
///
/// # Returns
///
/// The sine of x.
///
/// # Examples
///
/// ```
/// use pecos_num::math::sin;
///
/// assert!((sin(0.0)).abs() < 1e-10);
/// assert!((sin(std::f64::consts::PI)).abs() < 1e-10);
/// assert!((sin(std::f64::consts::PI / 2.0) - 1.0).abs() < 1e-10);
/// assert!((sin(std::f64::consts::PI / 4.0) - 0.707_106_781_186_547_5).abs() < 1e-10);
///
/// // Quantum gate construction use case
/// let theta = std::f64::consts::PI / 3.0;
/// let s = sin(theta * 0.5);
/// assert!((s - 0.5).abs() < 1e-10);
/// ```
#[must_use]
pub fn sin(x: f64) -> f64 {
    x.sin()
}

/// Return the floor of x as a float, the largest integer value less than or equal to x.
///
/// Drop-in replacement for `numpy.floor()` for scalar values.
///
/// # Arguments
///
/// * `x` - Input value
///
/// # Returns
///
/// The floor of x.
///
/// # Examples
///
/// ```
/// use pecos_num::math::floor;
///
/// assert_eq!(floor(3.7), 3.0);
/// assert_eq!(floor(-3.7), -4.0);
/// assert_eq!(floor(0.0), 0.0);
/// assert_eq!(floor(-0.0), -0.0);
///
/// // Fault tolerance threshold calculation use case
/// let t = floor((5.0 - 1.0) / 2.0);
/// assert_eq!(t, 2.0);
/// ```
#[must_use]
pub fn floor(x: f64) -> f64 {
    x.floor()
}

/// Return the ceiling of x as a float, the smallest integer value greater than or equal to x.
///
/// Drop-in replacement for `numpy.ceil()` for scalar values.
///
/// # Arguments
///
/// * `x` - Input value
///
/// # Returns
///
/// The ceiling of x.
///
/// # Examples
///
/// ```
/// use pecos_num::math::ceil;
///
/// assert_eq!(ceil(3.2), 4.0);
/// assert_eq!(ceil(-3.2), -3.0);
/// assert_eq!(ceil(0.0), 0.0);
/// assert_eq!(ceil(-0.0), -0.0);
/// ```
#[must_use]
pub fn ceil(x: f64) -> f64 {
    x.ceil()
}

/// Round a number to the nearest integer as a float.
///
/// Drop-in replacement for `numpy.round()` for scalar values (with default decimals=0).
/// Uses the "round half to even" strategy (banker's rounding) to match numpy behavior.
///
/// # Arguments
///
/// * `x` - Input value
///
/// # Returns
///
/// The rounded value.
///
/// # Examples
///
/// ```
/// use pecos_num::math::round;
///
/// assert_eq!(round(3.7), 4.0);
/// assert_eq!(round(3.2), 3.0);
/// assert_eq!(round(-3.7), -4.0);
/// assert_eq!(round(-3.2), -3.0);
/// assert_eq!(round(0.0), 0.0);
///
/// // Round half to even (banker's rounding)
/// assert_eq!(round(2.5), 2.0);
/// assert_eq!(round(3.5), 4.0);
/// ```
#[must_use]
pub fn round(x: f64) -> f64 {
    // Implement "round half to even" (banker's rounding) to match numpy
    // Rust's f64::round() uses "round half away from zero" which differs from numpy

    // Handle special values
    if !x.is_finite() {
        return x;
    }

    let floor_val = x.floor();
    let frac = x - floor_val;

    // If fractional part is exactly 0.5, round to even
    #[allow(clippy::float_cmp)]
    if frac == 0.5 {
        // Check if floor_val is even
        #[allow(clippy::cast_possible_truncation)]
        let floor_int = floor_val as i64;
        if floor_int % 2 == 0 {
            floor_val
        } else {
            floor_val + 1.0
        }
    } else if frac > 0.5 {
        floor_val + 1.0
    } else {
        floor_val
    }
}

// ============================================================================
// Mathematical Constants
// ============================================================================
//
// These constants provide drop-in replacements for numpy.pi, math.pi, etc.
// Using Rust's compile-time constants ensures maximum performance.

/// Archimedes' constant (π)
///
/// Drop-in replacement for `numpy.pi` and `math.pi`.
///
/// # Value
///
/// π ≈ 3.14159265358979323846264338327950288
pub const PI: f64 = std::f64::consts::PI;

/// The full circle constant (τ)
///
/// τ = 2π ≈ 6.28318530717958647692528676655900577
pub const TAU: f64 = std::f64::consts::TAU;

/// Euler's number (e)
///
/// Drop-in replacement for `numpy.e` and `math.e`.
///
/// e ≈ 2.71828182845904523536028747135266250
pub const E: f64 = std::f64::consts::E;

/// π/2 ≈ 1.57079632679489661923132169163975144
pub const FRAC_PI_2: f64 = std::f64::consts::FRAC_PI_2;

/// π/3 ≈ 1.04719755119659774615421446109316763
pub const FRAC_PI_3: f64 = std::f64::consts::FRAC_PI_3;

/// π/4 ≈ 0.78539816339744830961566084581987572
pub const FRAC_PI_4: f64 = std::f64::consts::FRAC_PI_4;

/// π/6 ≈ 0.52359877559829887307710723054658381
pub const FRAC_PI_6: f64 = std::f64::consts::FRAC_PI_6;

/// π/8 ≈ 0.39269908169872415480783042290993786
pub const FRAC_PI_8: f64 = std::f64::consts::FRAC_PI_8;

/// 1/π ≈ 0.31830988618379067153776752674502872
pub const FRAC_1_PI: f64 = std::f64::consts::FRAC_1_PI;

/// 2/π ≈ 0.63661977236758134307553505349005744
pub const FRAC_2_PI: f64 = std::f64::consts::FRAC_2_PI;

/// 2/√π ≈ 1.12837916709551257389615890312154517
pub const FRAC_2_SQRT_PI: f64 = std::f64::consts::FRAC_2_SQRT_PI;

/// √2 ≈ 1.41421356237309504880168872420969808
pub const SQRT_2: f64 = std::f64::consts::SQRT_2;

/// 1/√2 ≈ 0.70710678118654752440084436210484904
pub const FRAC_1_SQRT_2: f64 = std::f64::consts::FRAC_1_SQRT_2;

/// ln(2) ≈ 0.69314718055994530941723212145817657
pub const LN_2: f64 = std::f64::consts::LN_2;

/// ln(10) ≈ 2.30258509299404568401799145468436421
pub const LN_10: f64 = std::f64::consts::LN_10;

/// log₂(e) ≈ 1.44269504088896340735992468100189214
pub const LOG2_E: f64 = std::f64::consts::LOG2_E;

/// log₁₀(e) ≈ 0.43429448190325182765112891891660508
pub const LOG10_E: f64 = std::f64::consts::LOG10_E;

// ============================================================================
// Array Functions
// ============================================================================

/// Calculate the power of a base raised to an exponent element-wise for arrays.
///
/// Drop-in replacement for `numpy.power()` for arrays.
/// Broadcasts scalar exponent across all elements of the base array.
///
/// # Arguments
///
/// * `base` - The base array
/// * `exponent` - The exponent (scalar broadcast to all elements)
///
/// # Returns
///
/// Array where each element is base[i]^exponent
///
/// # Examples
///
/// ```
/// use pecos_num::math::power_array;
/// use ndarray::array;
///
/// let base = array![2.0, 3.0, 4.0];
/// let result = power_array(&base, 2.0);
/// assert_eq!(result, array![4.0, 9.0, 16.0]);
/// ```
#[must_use]
pub fn power_array<S, D>(base: &ArrayBase<S, D>, exponent: f64) -> Array<f64, D>
where
    S: Data<Elem = f64>,
    D: Dimension,
{
    base.mapv(|x| x.powf(exponent))
}

/// Calculate the square root element-wise for arrays.
///
/// Drop-in replacement for `numpy.sqrt()` for arrays.
///
/// # Arguments
///
/// * `arr` - Input array
///
/// # Returns
///
/// Array where each element is the square root of the input element.
///
/// # Examples
///
/// ```
/// use pecos_num::math::sqrt_array;
/// use ndarray::array;
///
/// let arr = array![4.0, 9.0, 16.0];
/// let result = sqrt_array(&arr);
/// assert_eq!(result, array![2.0, 3.0, 4.0]);
/// ```
#[must_use]
pub fn sqrt_array<S, D>(arr: &ArrayBase<S, D>) -> Array<f64, D>
where
    S: Data<Elem = f64>,
    D: Dimension,
{
    arr.mapv(f64::sqrt)
}

/// Calculate the exponential (e^x) element-wise for arrays.
///
/// Drop-in replacement for `numpy.exp()` for arrays.
///
/// # Arguments
///
/// * `arr` - Input array
///
/// # Returns
///
/// Array where each element is e^(input element).
///
/// # Examples
///
/// ```
/// use pecos_num::math::exp_array;
/// use ndarray::array;
///
/// let arr = array![0.0, 1.0, 2.0];
/// let result = exp_array(&arr);
/// assert!((result[0] - 1.0).abs() < 1e-10);
/// assert!((result[1] - std::f64::consts::E).abs() < 1e-10);
/// ```
#[must_use]
pub fn exp_array<S, D>(arr: &ArrayBase<S, D>) -> Array<f64, D>
where
    S: Data<Elem = f64>,
    D: Dimension,
{
    arr.mapv(f64::exp)
}

/// Calculate the exponential of complex numbers element-wise for arrays.
///
/// Drop-in replacement for `numpy.exp()` for complex arrays.
///
/// # Arguments
///
/// * `arr` - Complex input array
///
/// # Returns
///
/// Complex array where each element is e^(input element).
///
/// # Examples
///
/// ```
/// use pecos_num::math::exp_complex_array;
/// use num_complex::Complex64;
/// use ndarray::array;
///
/// let arr = array![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)];
/// let result = exp_complex_array(&arr);
/// assert!((result[0].re - 1.0).abs() < 1e-10);
/// assert!((result[1].re - std::f64::consts::E).abs() < 1e-10);
/// ```
#[must_use]
pub fn exp_complex_array<S, D>(arr: &ArrayBase<S, D>) -> Array<Complex64, D>
where
    S: Data<Elem = Complex64>,
    D: Dimension,
{
    arr.mapv(num_complex::Complex::exp)
}

/// Calculate the cosine element-wise for arrays.
///
/// Drop-in replacement for `numpy.cos()` for arrays.
///
/// # Arguments
///
/// * `arr` - Input array (angles in radians)
///
/// # Returns
///
/// Array where each element is the cosine of the input element.
///
/// # Examples
///
/// ```
/// use pecos_num::math::cos_array;
/// use ndarray::array;
///
/// let arr = array![0.0, std::f64::consts::PI];
/// let result = cos_array(&arr);
/// assert!((result[0] - 1.0).abs() < 1e-10);
/// assert!((result[1] - (-1.0)).abs() < 1e-10);
/// ```
#[must_use]
pub fn cos_array<S, D>(arr: &ArrayBase<S, D>) -> Array<f64, D>
where
    S: Data<Elem = f64>,
    D: Dimension,
{
    arr.mapv(f64::cos)
}

/// Calculate the sine element-wise for arrays.
///
/// Drop-in replacement for `numpy.sin()` for arrays.
///
/// # Arguments
///
/// * `arr` - Input array (angles in radians)
///
/// # Returns
///
/// Array where each element is the sine of the input element.
///
/// # Examples
///
/// ```
/// use pecos_num::math::sin_array;
/// use ndarray::array;
///
/// let arr = array![0.0, std::f64::consts::PI / 2.0];
/// let result = sin_array(&arr);
/// assert!((result[0]).abs() < 1e-10);
/// assert!((result[1] - 1.0).abs() < 1e-10);
/// ```
#[must_use]
pub fn sin_array<S, D>(arr: &ArrayBase<S, D>) -> Array<f64, D>
where
    S: Data<Elem = f64>,
    D: Dimension,
{
    arr.mapv(f64::sin)
}

#[cfg(test)]
mod tests {
    use super::*;

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

    // Tests for exp()
    #[test]
    fn test_exp_zero() {
        // exp(0) should be 1
        assert!((exp(0.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_exp_one() {
        // exp(1) should be e
        assert!((exp(1.0) - std::f64::consts::E).abs() < 1e-10);
    }

    #[test]
    fn test_exp_positive_values() {
        // Test with various positive values
        assert!((exp(2.0) - 7.389_056_098_930_65).abs() < 1e-10);
        assert!((exp(0.5) - 1.648_721_270_700_128).abs() < 1e-10);
        assert!((exp(5.0) - 148.413_159_102_576_6).abs() < 1e-8);
    }

    #[test]
    fn test_exp_negative_values() {
        // Test with negative values (exponential decay)
        assert!((exp(-1.0) - 0.367_879_441_171_442_3).abs() < 1e-10);
        assert!((exp(-2.0) - 0.135_335_283_236_612_7).abs() < 1e-10);
        assert!((exp(-0.5) - 0.606_530_659_712_633_4).abs() < 1e-10);
    }

    #[test]
    fn test_exp_decay_use_case() {
        // Exponential decay modeling (threshold analysis use case)
        let decay_rate = 0.5;
        let time = 2.0;
        let amplitude = exp(-decay_rate * time);
        assert!((amplitude - 0.367_879_441_171_442_3).abs() < 1e-10);
    }

    #[test]
    fn test_exp_large_values() {
        // Test with larger values
        assert!((exp(10.0) - 22_026.465_794_806_718).abs() < 1e-6);
        // Very large values approach infinity
        assert!(exp(100.0).is_finite());
        assert!(exp(700.0) > 1e300);
    }

    #[test]
    fn test_exp_special_cases() {
        // Test special values
        assert!(exp(f64::NEG_INFINITY) == 0.0);
        assert!(exp(f64::INFINITY).is_infinite());
        assert!(exp(f64::NAN).is_nan());
    }

    // Tests for exp_complex()
    #[test]
    fn test_exp_complex_euler_identity() {
        // e^(i*π) = -1 + 0i (Euler's identity)
        let pi = std::f64::consts::PI;
        let z = Complex64::new(0.0, pi);
        let result = exp_complex(z);
        assert!((result.re - (-1.0)).abs() < 1e-10);
        assert!(result.im.abs() < 1e-10);
    }

    #[test]
    fn test_exp_complex_real_only() {
        // e^(1+0i) = e + 0i
        let z = Complex64::new(1.0, 0.0);
        let result = exp_complex(z);
        assert!((result.re - std::f64::consts::E).abs() < 1e-10);
        assert!(result.im.abs() < 1e-10);
    }

    #[test]
    fn test_exp_complex_imaginary_only() {
        // e^(0+i*π/2) = 0 + i
        let pi = std::f64::consts::PI;
        let z = Complex64::new(0.0, pi / 2.0);
        let result = exp_complex(z);
        assert!(result.re.abs() < 1e-10);
        assert!((result.im - 1.0).abs() < 1e-10);

        // e^(0+i*π) = -1 + 0i
        let z = Complex64::new(0.0, pi);
        let result = exp_complex(z);
        assert!((result.re - (-1.0)).abs() < 1e-10);
        assert!(result.im.abs() < 1e-10);

        // e^(0+i*3π/2) = 0 - i
        let z = Complex64::new(0.0, 3.0 * pi / 2.0);
        let result = exp_complex(z);
        assert!(result.re.abs() < 1e-10);
        assert!((result.im - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_exp_complex_quantum_gate_use_case() {
        // Quantum gate matrix elements use exp(-i*phi) and exp(i*phi)
        let pi = std::f64::consts::PI;
        let phi = pi / 4.0; // 45 degrees

        // e^(-i*π/4)
        let z = Complex64::new(0.0, -phi);
        let result = exp_complex(z);
        let expected_val = 1.0 / 2.0_f64.sqrt(); // cos(π/4) = sin(π/4) = 1/√2
        assert!((result.re - expected_val).abs() < 1e-10);
        assert!((result.im - (-expected_val)).abs() < 1e-10);

        // e^(i*π/4)
        let z = Complex64::new(0.0, phi);
        let result = exp_complex(z);
        assert!((result.re - expected_val).abs() < 1e-10);
        assert!((result.im - expected_val).abs() < 1e-10);
    }

    #[test]
    fn test_exp_complex_general() {
        // e^(1+i*π/2) = e*(0 + i) = 0 + e*i
        let pi = std::f64::consts::PI;
        let e = std::f64::consts::E;
        let z = Complex64::new(1.0, pi / 2.0);
        let result = exp_complex(z);
        assert!(result.re.abs() < 1e-10);
        assert!((result.im - e).abs() < 1e-10);
    }

    #[test]
    fn test_exp_complex_rz_gate() {
        // RZ gate uses exp(-i*theta/2) and exp(i*theta/2)
        let pi = std::f64::consts::PI;
        let theta = pi / 2.0;

        let z1 = Complex64::new(0.0, -theta / 2.0);
        let result1 = exp_complex(z1);
        let z2 = Complex64::new(0.0, theta / 2.0);
        let result2 = exp_complex(z2);

        // exp(-i*π/4) should give (1/√2, -1/√2)
        let val = 1.0 / 2.0_f64.sqrt();
        assert!((result1.re - val).abs() < 1e-10);
        assert!((result1.im - (-val)).abs() < 1e-10);

        // exp(i*π/4) should give (1/√2, 1/√2)
        assert!((result2.re - val).abs() < 1e-10);
        assert!((result2.im - val).abs() < 1e-10);
    }

    // Tests for cos()
    #[test]
    fn test_cos_zero() {
        // cos(0) should be 1
        assert!((cos(0.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cos_key_angles() {
        // Test with key angles
        assert!((cos(std::f64::consts::PI) - (-1.0)).abs() < 1e-10);
        assert!((cos(std::f64::consts::PI / 2.0)).abs() < 1e-10); // Should be ~0
        assert!((cos(std::f64::consts::PI / 4.0) - 0.707_106_781_186_547_5).abs() < 1e-10);
        assert!((cos(std::f64::consts::PI / 3.0) - 0.5).abs() < 1e-10);
        assert!((cos(std::f64::consts::PI / 6.0) - 0.866_025_403_784_438_6).abs() < 1e-10);
    }

    #[test]
    fn test_cos_negative_angles() {
        // cos is an even function: cos(-x) = cos(x)
        assert!((cos(-std::f64::consts::PI / 4.0) - cos(std::f64::consts::PI / 4.0)).abs() < 1e-10);
        assert!((cos(-std::f64::consts::PI / 3.0) - cos(std::f64::consts::PI / 3.0)).abs() < 1e-10);
    }

    #[test]
    fn test_cos_periodicity() {
        // cos is periodic with period 2π
        let angle = std::f64::consts::PI / 6.0;
        assert!((cos(angle) - cos(angle + 2.0 * std::f64::consts::PI)).abs() < 1e-10);
    }

    #[test]
    fn test_cos_quantum_gate_use_case() {
        // Quantum gate construction use case: theta = π/3, so theta/2 = π/6
        let theta = std::f64::consts::PI / 3.0;
        let c = cos(theta * 0.5);
        // cos(π/6) = √3/2 ≈ 0.866025403784439
        assert!((c - 0.866_025_403_784_439).abs() < 1e-10);
    }

    // Tests for sin()
    #[test]
    fn test_sin_zero() {
        // sin(0) should be 0
        assert!((sin(0.0)).abs() < 1e-10);
    }

    #[test]
    fn test_sin_key_angles() {
        // Test with key angles
        assert!((sin(std::f64::consts::PI)).abs() < 1e-10); // Should be ~0
        assert!((sin(std::f64::consts::PI / 2.0) - 1.0).abs() < 1e-10);
        assert!((sin(std::f64::consts::PI / 4.0) - 0.707_106_781_186_547_5).abs() < 1e-10);
        assert!((sin(std::f64::consts::PI / 3.0) - 0.866_025_403_784_438_6).abs() < 1e-10);
        assert!((sin(std::f64::consts::PI / 6.0) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_sin_negative_angles() {
        // sin is an odd function: sin(-x) = -sin(x)
        assert!((sin(-std::f64::consts::PI / 4.0) + sin(std::f64::consts::PI / 4.0)).abs() < 1e-10);
        assert!((sin(-std::f64::consts::PI / 3.0) + sin(std::f64::consts::PI / 3.0)).abs() < 1e-10);
    }

    #[test]
    fn test_sin_periodicity() {
        // sin is periodic with period 2π
        let angle = std::f64::consts::PI / 6.0;
        assert!((sin(angle) - sin(angle + 2.0 * std::f64::consts::PI)).abs() < 1e-10);
    }

    #[test]
    fn test_sin_quantum_gate_use_case() {
        // Quantum gate construction use case: theta = π/3, so theta/2 = π/6
        let theta = std::f64::consts::PI / 3.0;
        let s = sin(theta * 0.5);
        // sin(π/6) = 1/2 = 0.5
        assert!((s - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_sin_cos_pythagorean_identity() {
        // Test the Pythagorean identity: sin²(x) + cos²(x) = 1
        let angles = vec![
            0.0,
            std::f64::consts::PI / 6.0,
            std::f64::consts::PI / 4.0,
            std::f64::consts::PI / 3.0,
            std::f64::consts::PI / 2.0,
            std::f64::consts::PI,
        ];

        for angle in angles {
            let sin_val = sin(angle);
            let cos_val = cos(angle);
            assert!((sin_val * sin_val + cos_val * cos_val - 1.0).abs() < 1e-10);
        }
    }

    // Tests for floor()
    #[allow(clippy::float_cmp)]
    #[test]
    fn test_floor_positive() {
        assert_eq!(floor(3.7), 3.0);
        assert_eq!(floor(3.0), 3.0);
        assert_eq!(floor(3.1), 3.0);
        assert_eq!(floor(3.9), 3.0);
    }

    #[allow(clippy::float_cmp)]
    #[test]
    fn test_floor_negative() {
        assert_eq!(floor(-3.7), -4.0);
        assert_eq!(floor(-3.0), -3.0);
        assert_eq!(floor(-3.1), -4.0);
        assert_eq!(floor(-3.9), -4.0);
    }

    #[allow(clippy::float_cmp)]
    #[test]
    fn test_floor_zero() {
        assert_eq!(floor(0.0), 0.0);
        assert_eq!(floor(-0.0), -0.0);
    }

    #[allow(clippy::float_cmp)]
    #[test]
    fn test_floor_special_values() {
        assert!(floor(f64::NAN).is_nan());
        assert_eq!(floor(f64::INFINITY), f64::INFINITY);
        assert_eq!(floor(f64::NEG_INFINITY), f64::NEG_INFINITY);
    }

    #[allow(clippy::float_cmp)]
    #[test]
    fn test_floor_fault_tolerance_use_case() {
        // Calculating error correction parameter t from distance d
        // t = floor((d - 1) / 2)
        let d = 5.0;
        let t = floor((d - 1.0) / 2.0);
        assert_eq!(t, 2.0);

        let d = 7.0;
        let t = floor((d - 1.0) / 2.0);
        assert_eq!(t, 3.0);
    }

    // Tests for ceil()
    #[allow(clippy::float_cmp)]
    #[test]
    fn test_ceil_positive() {
        assert_eq!(ceil(3.2), 4.0);
        assert_eq!(ceil(3.0), 3.0);
        assert_eq!(ceil(3.1), 4.0);
        assert_eq!(ceil(3.9), 4.0);
    }

    #[allow(clippy::float_cmp)]
    #[test]
    fn test_ceil_negative() {
        assert_eq!(ceil(-3.2), -3.0);
        assert_eq!(ceil(-3.0), -3.0);
        assert_eq!(ceil(-3.9), -3.0);
        assert_eq!(ceil(-3.1), -3.0);
    }

    #[allow(clippy::float_cmp)]
    #[test]
    fn test_ceil_zero() {
        assert_eq!(ceil(0.0), 0.0);
        assert_eq!(ceil(-0.0), -0.0);
    }

    #[allow(clippy::float_cmp)]
    #[test]
    fn test_ceil_special_values() {
        assert!(ceil(f64::NAN).is_nan());
        assert_eq!(ceil(f64::INFINITY), f64::INFINITY);
        assert_eq!(ceil(f64::NEG_INFINITY), f64::NEG_INFINITY);
    }

    // Tests for round()
    #[allow(clippy::float_cmp)]
    #[test]
    fn test_round_positive() {
        assert_eq!(round(3.7), 4.0);
        assert_eq!(round(3.2), 3.0);
        assert_eq!(round(3.0), 3.0);
        assert_eq!(round(3.5), 4.0);
    }

    #[allow(clippy::float_cmp)]
    #[test]
    fn test_round_negative() {
        assert_eq!(round(-3.7), -4.0);
        assert_eq!(round(-3.2), -3.0);
        assert_eq!(round(-3.0), -3.0);
        assert_eq!(round(-3.5), -4.0);
    }

    #[allow(clippy::float_cmp)]
    #[test]
    fn test_round_zero() {
        assert_eq!(round(0.0), 0.0);
        assert_eq!(round(-0.0), -0.0);
    }

    #[allow(clippy::float_cmp)]
    #[test]
    fn test_round_half_to_even() {
        // Test "round half to even" (banker's rounding) to match numpy
        assert_eq!(round(2.5), 2.0); // Even
        assert_eq!(round(3.5), 4.0); // Even
        assert_eq!(round(4.5), 4.0); // Even
        assert_eq!(round(5.5), 6.0); // Even

        // Test negative half values
        assert_eq!(round(-2.5), -2.0); // Even
        assert_eq!(round(-3.5), -4.0); // Even
        assert_eq!(round(-4.5), -4.0); // Even
        assert_eq!(round(-5.5), -6.0); // Even
    }

    #[allow(clippy::float_cmp)]
    #[test]
    fn test_round_special_values() {
        assert!(round(f64::NAN).is_nan());
        assert_eq!(round(f64::INFINITY), f64::INFINITY);
        assert_eq!(round(f64::NEG_INFINITY), f64::NEG_INFINITY);
    }
}
