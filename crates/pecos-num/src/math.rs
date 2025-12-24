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
use num_complex::{Complex, Complex32, Complex64};

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

/// Trait for calculating tangent.
///
/// This trait provides a uniform interface for tangent operations.
///
/// # Examples
///
/// ```
/// use pecos_num::prelude::*;
///
/// // Scalars
/// assert!((0.0_f64.tan()).abs() < 1e-10);
///
/// // Arrays
/// let arr = array![0.0, PI / 4.0];
/// let result = arr.tan();
/// assert!((result[0]).abs() < 1e-10);
/// ```
pub trait Tan {
    /// The output type when calculating tangent.
    type Output;

    /// Calculate tan(self) where self is in radians.
    fn tan(&self) -> Self::Output;
}

/// Trait for calculating hyperbolic sine.
///
/// This trait provides a uniform interface for hyperbolic sine operations.
///
/// # Examples
///
/// ```
/// use pecos_num::prelude::*;
///
/// // Scalars
/// assert!((0.0_f64.sinh()).abs() < 1e-10);
/// ```
pub trait Sinh {
    /// The output type when calculating hyperbolic sine.
    type Output;

    /// Calculate sinh(self).
    fn sinh(&self) -> Self::Output;
}

/// Trait for calculating hyperbolic cosine.
///
/// This trait provides a uniform interface for hyperbolic cosine operations.
///
/// # Examples
///
/// ```
/// use pecos_num::prelude::*;
///
/// // Scalars
/// assert!((0.0_f64.cosh() - 1.0).abs() < 1e-10);
/// ```
pub trait Cosh {
    /// The output type when calculating hyperbolic cosine.
    type Output;

    /// Calculate cosh(self).
    fn cosh(&self) -> Self::Output;
}

/// Trait for calculating hyperbolic tangent.
///
/// This trait provides a uniform interface for hyperbolic tangent operations.
///
/// # Examples
///
/// ```
/// use pecos_num::prelude::*;
///
/// // Scalars
/// assert!((0.0_f64.tanh()).abs() < 1e-10);
///
/// // Arrays
/// let arr = array![0.0, 1.0, -1.0];
/// let result = arr.tanh();
/// assert!((result[0]).abs() < 1e-10);
/// ```
pub trait Tanh {
    /// The output type when calculating hyperbolic tangent.
    type Output;

    /// Calculate tanh(self).
    fn tanh(&self) -> Self::Output;
}

/// Trait for calculating arcsine (inverse sine).
///
/// Drop-in replacement for `numpy.arcsin()` and `math.asin()`.
///
/// # Examples
///
/// ```
/// use pecos_num::prelude::*;
///
/// // Scalar
/// let x = 0.5_f64;
/// assert!((x.asin() - std::f64::consts::FRAC_PI_6).abs() < 1e-10);
///
/// // Array
/// let arr = array![0.0, 0.5, 1.0];
/// let result = arr.asin();
/// assert!(result[0].abs() < 1e-10);
/// assert!((result[2] - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
/// ```
pub trait Asin {
    /// The output type when calculating arcsine.
    type Output;

    /// Calculate arcsin(self).
    fn asin(&self) -> Self::Output;
}

/// Trait for calculating arccosine (inverse cosine).
///
/// Drop-in replacement for `numpy.arccos()` and `math.acos()`.
///
/// # Examples
///
/// ```
/// use pecos_num::prelude::*;
///
/// // Scalar
/// let x = 0.5_f64;
/// assert!((x.acos() - std::f64::consts::FRAC_PI_3).abs() < 1e-10);
///
/// // Array
/// let arr = array![0.0, 0.5, 1.0];
/// let result = arr.acos();
/// assert!((result[0] - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
/// assert!(result[2].abs() < 1e-10);
/// ```
pub trait Acos {
    /// The output type when calculating arccosine.
    type Output;

    /// Calculate arccos(self).
    fn acos(&self) -> Self::Output;
}

/// Trait for calculating arctangent (inverse tangent).
///
/// Drop-in replacement for `numpy.arctan()` and `math.atan()`.
///
/// # Examples
///
/// ```
/// use pecos_num::prelude::*;
///
/// // Scalar
/// let x = 1.0_f64;
/// assert!((x.atan() - std::f64::consts::FRAC_PI_4).abs() < 1e-10);
///
/// // Array
/// let arr = array![0.0, 1.0, -1.0];
/// let result = arr.atan();
/// assert!(result[0].abs() < 1e-10);
/// assert!((result[1] - std::f64::consts::FRAC_PI_4).abs() < 1e-10);
/// ```
pub trait Atan {
    /// The output type when calculating arctangent.
    type Output;

    /// Calculate arctan(self).
    fn atan(&self) -> Self::Output;
}

/// Trait for calculating inverse hyperbolic sine.
///
/// Drop-in replacement for `numpy.arcsinh()` and `math.asinh()`.
///
/// # Examples
///
/// ```
/// use pecos_num::prelude::*;
///
/// // Scalar
/// let x = 1.0_f64;
/// assert!((x.asinh() - 0.881_373_587_019_543).abs() < 1e-10);
///
/// // Array
/// let arr = array![0.0, 1.0, -1.0];
/// let result = arr.asinh();
/// assert!(result[0].abs() < 1e-10);
/// assert!((result[1] - 0.881_373_587_019_543).abs() < 1e-10);
/// ```
pub trait Asinh {
    /// The output type when calculating inverse hyperbolic sine.
    type Output;

    /// Calculate arcsinh(self).
    fn asinh(&self) -> Self::Output;
}

/// Trait for calculating inverse hyperbolic cosine.
///
/// Drop-in replacement for `numpy.arccosh()` and `math.acosh()`.
///
/// # Examples
///
/// ```
/// use pecos_num::prelude::*;
///
/// // Scalar
/// let x = 2.0_f64;
/// assert!((x.acosh() - 1.316_957_896_924_817).abs() < 1e-10);
///
/// // Array
/// let arr = array![1.0, 2.0, 3.0];
/// let result = arr.acosh();
/// assert!(result[0].abs() < 1e-10);
/// assert!((result[1] - 1.316_957_896_924_817).abs() < 1e-10);
/// ```
pub trait Acosh {
    /// The output type when calculating inverse hyperbolic cosine.
    type Output;

    /// Calculate arccosh(self).
    fn acosh(&self) -> Self::Output;
}

/// Trait for calculating inverse hyperbolic tangent.
///
/// Drop-in replacement for `numpy.arctanh()` and `math.atanh()`.
///
/// # Examples
///
/// ```
/// use pecos_num::prelude::*;
///
/// // Scalar
/// let x = 0.5_f64;
/// assert!((x.atanh() - 0.549_306_144_334_055).abs() < 1e-10);
///
/// // Array
/// let arr = array![0.0, 0.5, -0.5];
/// let result = arr.atanh();
/// assert!(result[0].abs() < 1e-10);
/// assert!((result[1] - 0.549_306_144_334_055).abs() < 1e-10);
/// ```
pub trait Atanh {
    /// The output type when calculating inverse hyperbolic tangent.
    type Output;

    /// Calculate arctanh(self).
    fn atanh(&self) -> Self::Output;
}

/// Trait for calculating two-argument arctangent with quadrant handling.
///
/// Drop-in replacement for `numpy.arctan2()` and `math.atan2()`.
///
/// Computes the angle θ in radians such that `x = r cos(θ)` and `y = r sin(θ)`,
/// where `r = sqrt(x² + y²)`. The result is in the range `[-π, π]`.
///
/// # Examples
///
/// ```
/// use pecos_num::prelude::*;
/// use std::f64::consts::{PI, FRAC_PI_4};
///
/// // Scalar - first quadrant
/// let y = 1.0_f64;
/// assert!((y.atan2(1.0) - FRAC_PI_4).abs() < 1e-10);
///
/// // Scalar - second quadrant
/// assert!((y.atan2(-1.0) - 3.0 * FRAC_PI_4).abs() < 1e-10);
/// ```
pub trait Atan2<Rhs = Self> {
    /// The output type when calculating atan2.
    type Output;

    /// Calculate atan2(self, x) - the angle in radians in the range [-π, π].
    ///
    /// # Arguments
    ///
    /// * `x` - The x-coordinate
    ///
    /// # Returns
    ///
    /// The angle θ such that `x_input = r cos(θ)` and `self = r sin(θ)`
    fn atan2(&self, x: Rhs) -> Self::Output;
}

/// Trait for calculating natural logarithm (base e) for arrays.
///
/// This trait extends `.ln()` support to Complex64 arrays for consistency with f64 arrays.
/// ndarray provides `.ln()` for Float arrays, but not for Complex arrays.
///
/// Note: For f64 scalars and arrays, use the stdlib/ndarray `.ln()` method directly.
/// For Complex64 scalars, use the num-complex `.ln()` method directly.
/// This trait is only needed for Complex64 arrays.
///
/// Drop-in replacement for `numpy.log()` on arrays.
///
/// # Examples
///
/// ```
/// use pecos_num::prelude::*;
///
/// // Float arrays use ndarray's built-in .ln()
/// let arr = array![1.0, E, E * E];
/// let result = arr.ln();
/// assert!((result[0]).abs() < 1e-10);
/// assert!((result[1] - 1.0).abs() < 1e-10);
/// assert!((result[2] - 2.0).abs() < 1e-10);
///
/// // Complex arrays use this trait's .ln()
/// use pecos_num::math::Ln;
/// let arr = array![Complex64::new(1.0, 0.0), Complex64::new(E, 0.0)];
/// let result = arr.ln();
/// assert!(result[0].re.abs() < 1e-10);
/// assert!((result[1].re - 1.0).abs() < 1e-10);
/// ```
pub trait Ln {
    /// The output type when calculating natural logarithm.
    type Output;

    /// Calculate natural logarithm (base e) of self.
    fn ln(&self) -> Self::Output;
}

/// Trait for calculating logarithm with custom base for arrays.
///
/// This trait extends `.log(base)` support to Complex64 arrays for consistency with f64 arrays.
/// ndarray provides `.log(base)` for Float arrays, but not for Complex arrays.
///
/// Note: For f64 scalars and arrays, use the stdlib/ndarray `.log(base)` method directly.
/// For Complex64 scalars, use the num-complex `.log(base)` method directly.
/// This trait is only needed for Complex64 arrays.
///
/// # Examples
///
/// ```
/// use pecos_num::prelude::*;
///
/// // Float arrays use ndarray's built-in .log(base)
/// let arr = array![10.0, 100.0, 1000.0];
/// let result = arr.log(10.0);
/// assert!((result[0] - 1.0).abs() < 1e-10);
/// assert!((result[1] - 2.0).abs() < 1e-10);
/// assert!((result[2] - 3.0).abs() < 1e-10);
///
/// // Complex arrays use this trait's .log(base)
/// use pecos_num::math::LogBase;
/// let arr = array![Complex64::new(10.0, 0.0), Complex64::new(100.0, 0.0)];
/// let result = arr.log(10.0);
/// assert!((result[0].re - 1.0).abs() < 1e-10);
/// assert!((result[1].re - 2.0).abs() < 1e-10);
/// ```
pub trait LogBase {
    /// The output type when calculating logarithm.
    type Output;

    /// Calculate logarithm with given base.
    fn log(&self, base: f64) -> Self::Output;
}

/// Trait for calculating absolute value.
///
/// This trait provides a uniform interface for absolute value operations
/// across different numeric types.
///
/// # Examples
///
/// ```
/// use pecos_num::prelude::*;
///
/// // Scalars
/// assert_eq!((-5.0).abs(), 5.0);
///
/// // Complex numbers (returns magnitude)
/// let z = Complex64::new(3.0, 4.0);
/// assert!((z.abs() - 5.0).abs() < 1e-10);
///
/// // Arrays
/// let arr = array![-1.0, -2.0, 3.0];
/// let result = arr.abs();
/// assert_eq!(result, array![1.0, 2.0, 3.0]);
/// ```
pub trait Abs {
    /// The output type when calculating absolute value.
    /// For complex numbers, this returns f64 (the magnitude).
    type Output;

    /// Calculate |self| (absolute value or magnitude).
    fn abs(&self) -> Self::Output;
}

/// Trait for calculating floor for arrays.
///
/// Note: For f32/f64 scalars, use the stdlib `.floor()` method directly.
/// This trait is primarily for array operations.
///
/// Drop-in replacement for `numpy.floor()` on arrays.
pub trait Floor {
    /// Output type (same as input for arrays).
    type Output;

    /// Calculate floor element-wise.
    fn floor(&self) -> Self::Output;
}

/// Trait for calculating ceiling for arrays.
///
/// Note: For f32/f64 scalars, use the stdlib `.ceil()` method directly.
/// This trait is primarily for array operations.
///
/// Drop-in replacement for `numpy.ceil()` on arrays.
pub trait Ceil {
    /// Output type (same as input for arrays).
    type Output;

    /// Calculate ceiling element-wise.
    fn ceil(&self) -> Self::Output;
}

/// Extension trait for rounding using "round half to even" (banker's rounding).
///
/// This trait extends `.round_ties_even()` to types not covered by Rust stdlib,
/// specifically complex numbers. For f32/f64 scalars, use the stdlib method directly.
///
/// For complex numbers, both real and imaginary parts are rounded independently,
/// matching `NumPy`'s behavior.
pub trait RoundTiesEven {
    /// Round using "round half to even" (banker's rounding).
    #[must_use]
    fn round_ties_even(&self) -> Self;
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

/// Extend `.round_ties_even()` to Complex64.
///
/// Rounds real and imaginary parts independently using "round half to even",
/// matching `NumPy`'s behavior for complex number rounding.
impl RoundTiesEven for Complex64 {
    #[inline]
    fn round_ties_even(&self) -> Self {
        Complex64::new(self.re.round_ties_even(), self.im.round_ties_even())
    }
}

/// Extend `.round_ties_even()` to Complex<f32>.
///
/// Rounds real and imaginary parts independently using "round half to even",
/// matching `NumPy`'s behavior for complex number rounding.
impl RoundTiesEven for Complex<f32> {
    #[inline]
    fn round_ties_even(&self) -> Self {
        Complex::new(self.re.round_ties_even(), self.im.round_ties_even())
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

/// Calculate tangent for f64 scalars.
impl Tan for f64 {
    type Output = f64;

    #[inline]
    fn tan(&self) -> f64 {
        f64::tan(*self)
    }
}

/// Calculate tangent for Complex64 scalars.
impl Tan for Complex64 {
    type Output = Complex64;

    #[inline]
    fn tan(&self) -> Complex64 {
        Complex64::tan(*self)
    }
}

/// Calculate hyperbolic sine for f64 scalars.
impl Sinh for f64 {
    type Output = f64;

    #[inline]
    fn sinh(&self) -> f64 {
        f64::sinh(*self)
    }
}

/// Calculate hyperbolic sine for Complex64 scalars.
impl Sinh for Complex64 {
    type Output = Complex64;

    #[inline]
    fn sinh(&self) -> Complex64 {
        Complex64::sinh(*self)
    }
}

/// Calculate hyperbolic cosine for f64 scalars.
impl Cosh for f64 {
    type Output = f64;

    #[inline]
    fn cosh(&self) -> f64 {
        f64::cosh(*self)
    }
}

/// Calculate hyperbolic cosine for Complex64 scalars.
impl Cosh for Complex64 {
    type Output = Complex64;

    #[inline]
    fn cosh(&self) -> Complex64 {
        Complex64::cosh(*self)
    }
}

/// Calculate hyperbolic tangent for f64 scalars.
impl Tanh for f64 {
    type Output = f64;

    #[inline]
    fn tanh(&self) -> f64 {
        f64::tanh(*self)
    }
}

/// Calculate hyperbolic tangent for Complex64 scalars.
impl Tanh for Complex64 {
    type Output = Complex64;

    #[inline]
    fn tanh(&self) -> Complex64 {
        Complex64::tanh(*self)
    }
}

/// Calculate arcsine for f64 scalars.
impl Asin for f64 {
    type Output = f64;

    #[inline]
    fn asin(&self) -> f64 {
        f64::asin(*self)
    }
}

/// Calculate arcsine for Complex64 scalars.
impl Asin for Complex64 {
    type Output = Complex64;

    #[inline]
    fn asin(&self) -> Complex64 {
        Complex64::asin(*self)
    }
}

/// Calculate arccosine for f64 scalars.
impl Acos for f64 {
    type Output = f64;

    #[inline]
    fn acos(&self) -> f64 {
        f64::acos(*self)
    }
}

/// Calculate arccosine for Complex64 scalars.
impl Acos for Complex64 {
    type Output = Complex64;

    #[inline]
    fn acos(&self) -> Complex64 {
        Complex64::acos(*self)
    }
}

/// Calculate arctangent for f64 scalars.
impl Atan for f64 {
    type Output = f64;

    #[inline]
    fn atan(&self) -> f64 {
        f64::atan(*self)
    }
}

/// Calculate arctangent for Complex64 scalars.
impl Atan for Complex64 {
    type Output = Complex64;

    #[inline]
    fn atan(&self) -> Complex64 {
        Complex64::atan(*self)
    }
}

/// Calculate inverse hyperbolic sine for f64 scalars.
impl Asinh for f64 {
    type Output = f64;

    #[inline]
    fn asinh(&self) -> f64 {
        f64::asinh(*self)
    }
}

/// Calculate inverse hyperbolic sine for Complex64 scalars.
impl Asinh for Complex64 {
    type Output = Complex64;

    #[inline]
    fn asinh(&self) -> Complex64 {
        Complex64::asinh(*self)
    }
}

/// Calculate inverse hyperbolic cosine for f64 scalars.
impl Acosh for f64 {
    type Output = f64;

    #[inline]
    fn acosh(&self) -> f64 {
        f64::acosh(*self)
    }
}

/// Calculate inverse hyperbolic cosine for Complex64 scalars.
impl Acosh for Complex64 {
    type Output = Complex64;

    #[inline]
    fn acosh(&self) -> Complex64 {
        Complex64::acosh(*self)
    }
}

/// Calculate inverse hyperbolic tangent for f64 scalars.
impl Atanh for f64 {
    type Output = f64;

    #[inline]
    fn atanh(&self) -> f64 {
        f64::atanh(*self)
    }
}

/// Calculate inverse hyperbolic tangent for Complex64 scalars.
impl Atanh for Complex64 {
    type Output = Complex64;

    #[inline]
    fn atanh(&self) -> Complex64 {
        Complex64::atanh(*self)
    }
}

/// Calculate two-argument arctangent for f64 scalars.
///
/// Returns the angle θ in radians such that `x = r cos(θ)` and `self = r sin(θ)`,
/// where `r = sqrt(x² + self²)`. The result is in the range `[-π, π]`.
impl Atan2 for f64 {
    type Output = f64;

    #[inline]
    fn atan2(&self, x: f64) -> f64 {
        f64::atan2(*self, x)
    }
}

/// Calculate two-argument arctangent for Complex64 scalars.
///
/// For complex numbers, atan2(y, x) is computed as:
/// atan2(y, x) = -i * ln((x + i*y) / sqrt(x² + y²))
///
/// This provides a complex extension of the real atan2 function.
impl Atan2 for Complex64 {
    type Output = Complex64;

    #[inline]
    fn atan2(&self, x: Complex64) -> Complex64 {
        // atan2(y, x) = -i * ln((x + i*y) / sqrt(x² + y²))
        let i = Complex64::new(0.0, 1.0);
        let numerator = x + i * self;
        let denominator = (x * x + self * self).sqrt();
        -i * (numerator / denominator).ln()
    }
}

/// Calculate absolute value for f64 scalars.
impl Abs for f64 {
    type Output = f64;

    #[inline]
    fn abs(&self) -> f64 {
        f64::abs(*self)
    }
}

/// Calculate absolute value (magnitude) for Complex64 scalars.
impl Abs for Complex64 {
    type Output = f64;

    #[inline]
    fn abs(&self) -> f64 {
        Complex64::norm(*self)
    }
}

/// Calculate absolute value (magnitude) for Complex32 scalars.
impl Abs for Complex32 {
    type Output = f32;

    #[inline]
    fn abs(&self) -> f32 {
        Complex32::norm(*self)
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

/// Calculate tangent element-wise for arrays.
///
/// This generic implementation works for any element type that implements Tan.
impl<S, D, T> Tan for ArrayBase<S, D>
where
    S: Data<Elem = T>,
    T: Tan<Output = T> + Clone,
    D: Dimension,
{
    type Output = Array<T, D>;

    #[inline]
    fn tan(&self) -> Array<T, D> {
        self.mapv(|x| x.tan())
    }
}

/// Calculate hyperbolic sine element-wise for arrays.
///
/// This generic implementation works for any element type that implements Sinh.
impl<S, D, T> Sinh for ArrayBase<S, D>
where
    S: Data<Elem = T>,
    T: Sinh<Output = T> + Clone,
    D: Dimension,
{
    type Output = Array<T, D>;

    #[inline]
    fn sinh(&self) -> Array<T, D> {
        self.mapv(|x| x.sinh())
    }
}

/// Calculate hyperbolic cosine element-wise for arrays.
///
/// This generic implementation works for any element type that implements Cosh.
impl<S, D, T> Cosh for ArrayBase<S, D>
where
    S: Data<Elem = T>,
    T: Cosh<Output = T> + Clone,
    D: Dimension,
{
    type Output = Array<T, D>;

    #[inline]
    fn cosh(&self) -> Array<T, D> {
        self.mapv(|x| x.cosh())
    }
}

/// Calculate hyperbolic tangent element-wise for arrays.
///
/// This generic implementation works for any element type that implements Tanh.
impl<S, D, T> Tanh for ArrayBase<S, D>
where
    S: Data<Elem = T>,
    T: Tanh<Output = T> + Clone,
    D: Dimension,
{
    type Output = Array<T, D>;

    #[inline]
    fn tanh(&self) -> Array<T, D> {
        self.mapv(|x| x.tanh())
    }
}

/// Calculate arcsine element-wise for arrays.
///
/// This generic implementation works for any element type that implements Asin.
impl<S, D, T> Asin for ArrayBase<S, D>
where
    S: Data<Elem = T>,
    T: Asin<Output = T> + Clone,
    D: Dimension,
{
    type Output = Array<T, D>;

    #[inline]
    fn asin(&self) -> Array<T, D> {
        self.mapv(|x| x.asin())
    }
}

/// Calculate arccosine element-wise for arrays.
///
/// This generic implementation works for any element type that implements Acos.
impl<S, D, T> Acos for ArrayBase<S, D>
where
    S: Data<Elem = T>,
    T: Acos<Output = T> + Clone,
    D: Dimension,
{
    type Output = Array<T, D>;

    #[inline]
    fn acos(&self) -> Array<T, D> {
        self.mapv(|x| x.acos())
    }
}

/// Calculate arctangent element-wise for arrays.
///
/// This generic implementation works for any element type that implements Atan.
impl<S, D, T> Atan for ArrayBase<S, D>
where
    S: Data<Elem = T>,
    T: Atan<Output = T> + Clone,
    D: Dimension,
{
    type Output = Array<T, D>;

    #[inline]
    fn atan(&self) -> Array<T, D> {
        self.mapv(|x| x.atan())
    }
}

/// Calculate inverse hyperbolic sine element-wise for arrays.
///
/// This generic implementation works for any element type that implements Asinh.
impl<S, D, T> Asinh for ArrayBase<S, D>
where
    S: Data<Elem = T>,
    T: Asinh<Output = T> + Clone,
    D: Dimension,
{
    type Output = Array<T, D>;

    #[inline]
    fn asinh(&self) -> Array<T, D> {
        self.mapv(|x| x.asinh())
    }
}

/// Calculate inverse hyperbolic cosine element-wise for arrays.
///
/// This generic implementation works for any element type that implements Acosh.
impl<S, D, T> Acosh for ArrayBase<S, D>
where
    S: Data<Elem = T>,
    T: Acosh<Output = T> + Clone,
    D: Dimension,
{
    type Output = Array<T, D>;

    #[inline]
    fn acosh(&self) -> Array<T, D> {
        self.mapv(|x| x.acosh())
    }
}

/// Calculate inverse hyperbolic tangent element-wise for arrays.
///
/// This generic implementation works for any element type that implements Atanh.
impl<S, D, T> Atanh for ArrayBase<S, D>
where
    S: Data<Elem = T>,
    T: Atanh<Output = T> + Clone,
    D: Dimension,
{
    type Output = Array<T, D>;

    #[inline]
    fn atanh(&self) -> Array<T, D> {
        self.mapv(|x| x.atanh())
    }
}

/// Calculate two-argument arctangent element-wise for arrays with scalar second argument.
///
/// Computes `atan2(array_elem`, scalar) for each element in the array.
impl<S, D, T> Atan2<T> for ArrayBase<S, D>
where
    S: Data<Elem = T>,
    T: Atan2<T, Output = T> + Clone,
    D: Dimension,
{
    type Output = Array<T, D>;

    #[inline]
    fn atan2(&self, x: T) -> Array<T, D> {
        self.mapv(|y| y.atan2(x.clone()))
    }
}

/// Calculate natural logarithm element-wise for Complex64 arrays.
///
/// Provides `.ln()` for Complex64 arrays for consistency with f64 arrays.
/// (ndarray only provides `.ln()` for Float types, not Complex types)
impl<S, D> Ln for ArrayBase<S, D>
where
    S: Data<Elem = Complex64>,
    D: Dimension,
{
    type Output = Array<Complex64, D>;

    #[inline]
    fn ln(&self) -> Array<Complex64, D> {
        self.mapv(num_complex::Complex::ln)
    }
}

/// Calculate logarithm with custom base element-wise for Complex64 arrays.
///
/// Provides `.log(base)` for Complex64 arrays for consistency with f64 arrays.
/// (ndarray only provides `.log(base)` for Float types, not Complex types)
impl<S, D> LogBase for ArrayBase<S, D>
where
    S: Data<Elem = Complex64>,
    D: Dimension,
{
    type Output = Array<Complex64, D>;

    #[inline]
    fn log(&self, base: f64) -> Array<Complex64, D> {
        self.mapv(|x| x.log(base))
    }
}

/// Calculate absolute value element-wise for arrays.
///
/// This generic implementation works for any element type that implements Abs.
/// For arrays of floats, returns array of floats. For arrays of complex numbers,
/// returns array of magnitudes (f64/f32).
///
/// Note: For complex arrays, this implementation explicitly uses the `Abs` trait
/// implementations for Complex64/Complex32, which correctly use `.norm()` to compute
/// the magnitude of each complex number element.
impl<S, D, T> Abs for ArrayBase<S, D>
where
    S: Data<Elem = T>,
    T: Abs + Clone,
    D: Dimension,
{
    type Output = Array<T::Output, D>;

    #[inline]
    fn abs(&self) -> Array<T::Output, D> {
        self.mapv(|x| Abs::abs(&x))
    }
}

/// Calculate floor element-wise for arrays.
///
/// This implementation delegates to the stdlib `floor()` method for each element.
/// Works for f32 and f64 arrays.
impl<S, D, T> Floor for ArrayBase<S, D>
where
    S: Data<Elem = T>,
    T: num_traits::Float + Clone,
    D: Dimension,
{
    type Output = Array<T, D>;

    #[inline]
    fn floor(&self) -> Array<T, D> {
        self.mapv(num_traits::Float::floor)
    }
}

/// Calculate ceiling element-wise for arrays.
///
/// This implementation delegates to the stdlib `ceil()` method for each element.
/// Works for f32 and f64 arrays.
impl<S, D, T> Ceil for ArrayBase<S, D>
where
    S: Data<Elem = T>,
    T: num_traits::Float + Clone,
    D: Dimension,
{
    type Output = Array<T, D>;

    #[inline]
    fn ceil(&self) -> Array<T, D> {
        self.mapv(num_traits::Float::ceil)
    }
}

// ============================================================================
// Scalar Functions (Python Bindings API)
// ============================================================================
//
// These functions are primarily intended as entry points for the PyO3 bindings
// and provide a NumPy-compatible scalar API.
//
// For Rust code working with arrays, prefer the trait-based approach:
//   - Use `arr.exp()` instead of calling `exp()` on each element
//   - Use `arr.sqrt()` instead of manually mapping
//   - Use `arr.power(2.0)` for element-wise power operations
//
// This is more idiomatic Rust and avoids manual iteration patterns.

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

/// Calculate the tangent of a value (in radians).
///
/// Drop-in replacement for `numpy.tan()` for scalar values.
///
/// # Arguments
///
/// * `x` - Input value in radians
///
/// # Returns
///
/// The tangent of x.
///
/// # Examples
///
/// ```
/// use pecos_num::math::tan;
///
/// assert!((tan(0.0)).abs() < 1e-10);
/// assert!((tan(std::f64::consts::PI)).abs() < 1e-10);
/// assert!((tan(std::f64::consts::PI / 4.0) - 1.0).abs() < 1e-10);
/// assert!((tan(-std::f64::consts::PI / 4.0) + 1.0).abs() < 1e-10);
///
/// // Quantum gate construction use case
/// let theta = std::f64::consts::PI / 6.0;
/// let t = tan(theta);
/// assert!((t - 0.577_350_269_189_625_8).abs() < 1e-10);
/// ```
#[must_use]
pub fn tan(x: f64) -> f64 {
    x.tan()
}

/// Calculate the hyperbolic tangent of a value.
///
/// Drop-in replacement for `numpy.tanh()` for scalar values.
///
/// # Arguments
///
/// * `x` - Input value
///
/// # Returns
///
/// The hyperbolic tangent of x.
///
/// # Examples
///
/// ```
/// use pecos_num::math::tanh;
///
/// assert!((tanh(0.0)).abs() < 1e-10);
/// assert!((tanh(1.0) - 0.761_594_155_955_764_9).abs() < 1e-10);
/// assert!((tanh(-1.0) + 0.761_594_155_955_764_9).abs() < 1e-10);
/// assert!((tanh(f64::INFINITY) - 1.0).abs() < 1e-10);
/// assert!((tanh(f64::NEG_INFINITY) + 1.0).abs() < 1e-10);
///
/// // Activation function use case (quantum machine learning)
/// let x = 0.5;
/// let activation = tanh(x);
/// assert!((activation - 0.462_117_157_260_009_8).abs() < 1e-10);
/// ```
#[must_use]
pub fn tanh(x: f64) -> f64 {
    x.tanh()
}

/// Calculate the hyperbolic sine of a value.
///
/// Drop-in replacement for `numpy.sinh()` for scalar values.
///
/// # Arguments
///
/// * `x` - Input value
///
/// # Returns
///
/// The hyperbolic sine of x.
///
/// # Examples
///
/// ```
/// use pecos_num::math::sinh;
///
/// assert!((sinh(0.0)).abs() < 1e-10);
/// assert!((sinh(1.0) - 1.175_201_193_643_801_4).abs() < 1e-10);
/// assert!((sinh(-1.0) + 1.175_201_193_643_801_4).abs() < 1e-10);
/// ```
#[must_use]
pub fn sinh(x: f64) -> f64 {
    x.sinh()
}

/// Calculate the hyperbolic cosine of a value.
///
/// Drop-in replacement for `numpy.cosh()` for scalar values.
///
/// # Arguments
///
/// * `x` - Input value
///
/// # Returns
///
/// The hyperbolic cosine of x.
///
/// # Examples
///
/// ```
/// use pecos_num::math::cosh;
///
/// assert!((cosh(0.0) - 1.0).abs() < 1e-10);
/// assert!((cosh(1.0) - 1.543_080_634_815_243_7).abs() < 1e-10);
/// assert!((cosh(-1.0) - 1.543_080_634_815_243_7).abs() < 1e-10);
/// ```
#[must_use]
pub fn cosh(x: f64) -> f64 {
    x.cosh()
}

/// Calculate the arcsine (inverse sine) of a value.
///
/// Drop-in replacement for `numpy.arcsin()` / `numpy.asin()` for scalar values.
///
/// # Arguments
///
/// * `x` - Input value (must be in range [-1, 1])
///
/// # Returns
///
/// The arcsine of x in radians, in the range [-π/2, π/2].
///
/// # Examples
///
/// ```
/// use pecos_num::math::asin;
/// use std::f64::consts::{FRAC_PI_2, FRAC_PI_6};
///
/// assert!((asin(0.0)).abs() < 1e-10);
/// assert!((asin(1.0) - FRAC_PI_2).abs() < 1e-10);
/// assert!((asin(-1.0) + FRAC_PI_2).abs() < 1e-10);
/// assert!((asin(0.5) - FRAC_PI_6).abs() < 1e-10);
/// ```
#[must_use]
pub fn asin(x: f64) -> f64 {
    x.asin()
}

/// Calculate the arccosine (inverse cosine) of a value.
///
/// Drop-in replacement for `numpy.arccos()` / `numpy.acos()` for scalar values.
///
/// # Arguments
///
/// * `x` - Input value (must be in range [-1, 1])
///
/// # Returns
///
/// The arccosine of x in radians, in the range [0, π].
///
/// # Examples
///
/// ```
/// use pecos_num::math::acos;
/// use std::f64::consts::{PI, FRAC_PI_2, FRAC_PI_3};
///
/// assert!((acos(1.0)).abs() < 1e-10);
/// assert!((acos(-1.0) - PI).abs() < 1e-10);
/// assert!((acos(0.0) - FRAC_PI_2).abs() < 1e-10);
/// assert!((acos(0.5) - FRAC_PI_3).abs() < 1e-10);
/// ```
#[must_use]
pub fn acos(x: f64) -> f64 {
    x.acos()
}

/// Calculate the arctangent (inverse tangent) of a value.
///
/// Drop-in replacement for `numpy.arctan()` / `numpy.atan()` for scalar values.
///
/// # Arguments
///
/// * `x` - Input value
///
/// # Returns
///
/// The arctangent of x in radians, in the range [-π/2, π/2].
///
/// # Examples
///
/// ```
/// use pecos_num::math::atan;
/// use std::f64::consts::FRAC_PI_4;
///
/// assert!((atan(0.0)).abs() < 1e-10);
/// assert!((atan(1.0) - FRAC_PI_4).abs() < 1e-10);
/// assert!((atan(-1.0) + FRAC_PI_4).abs() < 1e-10);
/// ```
#[must_use]
pub fn atan(x: f64) -> f64 {
    x.atan()
}

/// Calculate the inverse hyperbolic sine of a value.
///
/// Drop-in replacement for `numpy.arcsinh()` / `numpy.asinh()` for scalar values.
///
/// # Arguments
///
/// * `x` - Input value
///
/// # Returns
///
/// The inverse hyperbolic sine of x.
///
/// # Examples
///
/// ```
/// use pecos_num::math::asinh;
///
/// assert!((asinh(0.0)).abs() < 1e-10);
/// assert!((asinh(1.0) - 0.881_373_587_019_543).abs() < 1e-10);
/// assert!((asinh(-1.0) + 0.881_373_587_019_543).abs() < 1e-10);
/// ```
#[must_use]
pub fn asinh(x: f64) -> f64 {
    x.asinh()
}

/// Calculate the inverse hyperbolic cosine of a value.
///
/// Drop-in replacement for `numpy.arccosh()` / `numpy.acosh()` for scalar values.
///
/// # Arguments
///
/// * `x` - Input value (must be >= 1)
///
/// # Returns
///
/// The inverse hyperbolic cosine of x.
///
/// # Examples
///
/// ```
/// use pecos_num::math::acosh;
///
/// assert!((acosh(1.0)).abs() < 1e-10);
/// assert!((acosh(2.0) - 1.316_957_896_924_817).abs() < 1e-10);
/// assert!((acosh(3.0) - 1.762_747_174_039_086_1).abs() < 1e-10);
/// ```
#[must_use]
pub fn acosh(x: f64) -> f64 {
    x.acosh()
}

/// Calculate the inverse hyperbolic tangent of a value.
///
/// Drop-in replacement for `numpy.arctanh()` / `numpy.atanh()` for scalar values.
///
/// # Arguments
///
/// * `x` - Input value (must be in range (-1, 1))
///
/// # Returns
///
/// The inverse hyperbolic tangent of x.
///
/// # Examples
///
/// ```
/// use pecos_num::math::atanh;
///
/// assert!((atanh(0.0)).abs() < 1e-10);
/// assert!((atanh(0.5) - 0.549_306_144_334_055).abs() < 1e-10);
/// assert!((atanh(-0.5) + 0.549_306_144_334_055).abs() < 1e-10);
/// ```
#[must_use]
pub fn atanh(x: f64) -> f64 {
    x.atanh()
}

/// Calculate the arctangent of y/x with correct quadrant handling.
///
/// Drop-in replacement for `numpy.arctan2()` / `numpy.atan2()`.
///
/// Returns the angle in radians between the positive x-axis and the point (x, y).
/// The result is in the range [-π, π].
///
/// This is a convenience wrapper around the `Atan2` trait method.
/// For polymorphic usage, prefer using the trait method directly: `y.atan2(x)`.
///
/// # Arguments
///
/// * `y` - y-coordinate (can be scalar or array)
/// * `x` - x-coordinate (can be scalar or array)
///
/// # Returns
///
/// The angle in radians, in the range [-π, π].
///
/// # Examples
///
/// ```
/// use pecos_num::prelude::*;
/// use std::f64::consts::{PI, FRAC_PI_2, FRAC_PI_4};
///
/// // Scalar usage
/// assert!((atan2(1.0, 1.0) - FRAC_PI_4).abs() < 1e-10);
/// assert!((atan2(1.0, -1.0) - 3.0 * FRAC_PI_4).abs() < 1e-10);
///
/// // Array usage
/// let y_arr = array![1.0, 1.0, -1.0];
/// let x_val = 1.0;
/// let result = atan2(y_arr, x_val);
/// assert!((result[0] - FRAC_PI_4).abs() < 1e-10);
/// ```
#[must_use]
#[allow(clippy::needless_pass_by_value)] // Generic trait-based design requires ownership
pub fn atan2<Y, X>(y: Y, x: X) -> Y::Output
where
    Y: Atan2<X>,
{
    y.atan2(x)
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
// f32 Mathematical Constants
// ============================================================================
//
// Single-precision (32-bit) floating point constants from Rust's std library.
// These provide precise f32 values for users who need single-precision constants.
//
// Usage: pc.f32.pi, pc.f32.frac_pi_2, etc.

/// Archimedes' constant (π) - f32 precision
///
/// π ≈ 3.14159265 (32-bit precision)
pub const PI_F32: f32 = std::f32::consts::PI;

/// The full circle constant (τ) - f32 precision
///
/// τ = 2π ≈ 6.28318530 (32-bit precision)
pub const TAU_F32: f32 = std::f32::consts::TAU;

/// Euler's number (e) - f32 precision
///
/// e ≈ 2.71828182 (32-bit precision)
pub const E_F32: f32 = std::f32::consts::E;

/// π/2 ≈ 1.57079632 (32-bit precision)
pub const FRAC_PI_2_F32: f32 = std::f32::consts::FRAC_PI_2;

/// π/3 ≈ 1.04719755 (32-bit precision)
pub const FRAC_PI_3_F32: f32 = std::f32::consts::FRAC_PI_3;

/// π/4 ≈ 0.78539816 (32-bit precision)
pub const FRAC_PI_4_F32: f32 = std::f32::consts::FRAC_PI_4;

/// π/6 ≈ 0.52359877 (32-bit precision)
pub const FRAC_PI_6_F32: f32 = std::f32::consts::FRAC_PI_6;

/// π/8 ≈ 0.39269908 (32-bit precision)
pub const FRAC_PI_8_F32: f32 = std::f32::consts::FRAC_PI_8;

/// 1/π ≈ 0.31830988 (32-bit precision)
pub const FRAC_1_PI_F32: f32 = std::f32::consts::FRAC_1_PI;

/// 2/π ≈ 0.63661977 (32-bit precision)
pub const FRAC_2_PI_F32: f32 = std::f32::consts::FRAC_2_PI;

/// 2/√π ≈ 1.12837916 (32-bit precision)
pub const FRAC_2_SQRT_PI_F32: f32 = std::f32::consts::FRAC_2_SQRT_PI;

/// √2 ≈ 1.41421356 (32-bit precision)
pub const SQRT_2_F32: f32 = std::f32::consts::SQRT_2;

/// 1/√2 ≈ 0.70710678 (32-bit precision)
pub const FRAC_1_SQRT_2_F32: f32 = std::f32::consts::FRAC_1_SQRT_2;

/// ln(2) ≈ 0.69314718 (32-bit precision)
pub const LN_2_F32: f32 = std::f32::consts::LN_2;

/// ln(10) ≈ 2.30258509 (32-bit precision)
pub const LN_10_F32: f32 = std::f32::consts::LN_10;

/// log₂(e) ≈ 1.44269504 (32-bit precision)
pub const LOG2_E_F32: f32 = std::f32::consts::LOG2_E;

/// log₁₀(e) ≈ 0.43429448 (32-bit precision)
pub const LOG10_E_F32: f32 = std::f32::consts::LOG10_E;

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

    // Tests for .round_ties_even() method
    #[allow(clippy::float_cmp)]
    #[test]
    fn test_round_positive() {
        assert_eq!(3.7_f64.round_ties_even(), 4.0);
        assert_eq!(3.2_f64.round_ties_even(), 3.0);
        assert_eq!(3.0_f64.round_ties_even(), 3.0);
        assert_eq!(3.5_f64.round_ties_even(), 4.0);
    }

    #[allow(clippy::float_cmp)]
    #[test]
    fn test_round_negative() {
        assert_eq!((-3.7_f64).round_ties_even(), -4.0);
        assert_eq!((-3.2_f64).round_ties_even(), -3.0);
        assert_eq!((-3.0_f64).round_ties_even(), -3.0);
        assert_eq!((-3.5_f64).round_ties_even(), -4.0);
    }

    #[allow(clippy::float_cmp)]
    #[test]
    fn test_round_zero() {
        assert_eq!(0.0_f64.round_ties_even(), 0.0);
        assert_eq!((-0.0_f64).round_ties_even(), -0.0);
    }

    #[allow(clippy::float_cmp)]
    #[test]
    fn test_round_half_to_even() {
        // Test "round half to even" (banker's rounding) to match numpy
        assert_eq!(2.5_f64.round_ties_even(), 2.0); // Even
        assert_eq!(3.5_f64.round_ties_even(), 4.0); // Even
        assert_eq!(4.5_f64.round_ties_even(), 4.0); // Even
        assert_eq!(5.5_f64.round_ties_even(), 6.0); // Even

        // Test negative half values
        assert_eq!((-2.5_f64).round_ties_even(), -2.0); // Even
        assert_eq!((-3.5_f64).round_ties_even(), -4.0); // Even
        assert_eq!((-4.5_f64).round_ties_even(), -4.0); // Even
        assert_eq!((-5.5_f64).round_ties_even(), -6.0); // Even
    }

    #[allow(clippy::float_cmp)]
    #[test]
    fn test_round_special_values() {
        assert!(f64::NAN.round_ties_even().is_nan());
        assert_eq!(f64::INFINITY.round_ties_even(), f64::INFINITY);
        assert_eq!(f64::NEG_INFINITY.round_ties_even(), f64::NEG_INFINITY);
    }

    // Tests for complex .round_ties_even() extension
    #[allow(clippy::float_cmp)]
    #[test]
    fn test_round_ties_even_complex64() {
        use crate::math::RoundTiesEven;

        // Test basic rounding
        let z = Complex64::new(2.5, 3.5);
        let rounded = z.round_ties_even();
        assert_eq!(rounded.re, 2.0); // 2.5 rounds to 2 (even)
        assert_eq!(rounded.im, 4.0); // 3.5 rounds to 4 (even)

        // Test negative values
        let z = Complex64::new(-2.5, -3.5);
        let rounded = z.round_ties_even();
        assert_eq!(rounded.re, -2.0); // -2.5 rounds to -2 (even)
        assert_eq!(rounded.im, -4.0); // -3.5 rounds to -4 (even)

        // Test non-half values
        let z = Complex64::new(2.3, 3.7);
        let rounded = z.round_ties_even();
        assert_eq!(rounded.re, 2.0);
        assert_eq!(rounded.im, 4.0);

        // Test mixed signs
        let z = Complex64::new(4.5, -4.5);
        let rounded = z.round_ties_even();
        assert_eq!(rounded.re, 4.0); // 4.5 rounds to 4 (even)
        assert_eq!(rounded.im, -4.0); // -4.5 rounds to -4 (even)
    }

    #[allow(clippy::float_cmp)]
    #[test]
    fn test_round_ties_even_complex32() {
        use crate::math::RoundTiesEven;
        use num_complex::Complex;

        // Test basic rounding with f32
        let z = Complex::<f32>::new(2.5, 3.5);
        let rounded = z.round_ties_even();
        assert_eq!(rounded.re, 2.0); // 2.5 rounds to 2 (even)
        assert_eq!(rounded.im, 4.0); // 3.5 rounds to 4 (even)

        // Test negative values
        let z = Complex::<f32>::new(-2.5, -3.5);
        let rounded = z.round_ties_even();
        assert_eq!(rounded.re, -2.0);
        assert_eq!(rounded.im, -4.0);
    }

    // Tests for ln() - NumPy log() uses .ln() (natural logarithm)

    #[test]
    fn test_ln_basic() {
        use std::f64::consts::E;

        // ln(1) = 0
        assert!((1.0_f64.ln() - 0.0).abs() < 1e-10);

        // ln(e) = 1
        assert!((E.ln() - 1.0).abs() < 1e-10);

        // ln(e^2) = 2
        assert!(((E * E).ln() - 2.0).abs() < 1e-10);

        // ln(e^3) = 3
        assert!(((E * E * E).ln() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_ln_powers_of_ten() {
        use std::f64::consts::LN_10;

        // ln(10) ≈ 2.302585
        assert!((10.0_f64.ln() - LN_10).abs() < 1e-10);

        // ln(100) = 2 * ln(10)
        assert!((100.0_f64.ln() - 2.0 * LN_10).abs() < 1e-10);
    }

    #[test]
    fn test_ln_fractions() {
        use std::f64::consts::{E, LN_2};

        // ln(1/e) = -1
        assert!(((1.0 / E).ln() - (-1.0)).abs() < 1e-10);

        // ln(0.5) = -ln(2)
        assert!((0.5_f64.ln() + LN_2).abs() < 1e-10);
    }

    #[test]
    fn test_ln_array() {
        use std::f64::consts::E;

        // Float arrays use ndarray's built-in .ln()
        let arr = crate::prelude::array![1.0, E, E * E, E * E * E];
        let result = arr.ln();

        assert!((result[0] - 0.0).abs() < 1e-10);
        assert!((result[1] - 1.0).abs() < 1e-10);
        assert!((result[2] - 2.0).abs() < 1e-10);
        assert!((result[3] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_ln_complex() {
        use std::f64::consts::E;

        // Complex64 scalars use num-complex .ln()
        // ln(e + 0i) = 1 + 0i
        let z = Complex64::new(E, 0.0);
        let result = z.ln();
        assert!((result.re - 1.0).abs() < 1e-10);
        assert!(result.im.abs() < 1e-10);

        // ln(1 + 0i) = 0 + 0i
        let z = Complex64::new(1.0, 0.0);
        let result = z.ln();
        assert!(result.re.abs() < 1e-10);
        assert!(result.im.abs() < 1e-10);
    }

    #[test]
    fn test_ln_complex_array() {
        use crate::math::Ln;
        use std::f64::consts::E;

        // Complex64 arrays use our Ln trait
        let arr = crate::prelude::array![Complex64::new(1.0, 0.0), Complex64::new(E, 0.0)];
        let result = arr.ln();

        assert!(result[0].re.abs() < 1e-10);
        assert!(result[0].im.abs() < 1e-10);
        assert!((result[1].re - 1.0).abs() < 1e-10);
        assert!(result[1].im.abs() < 1e-10);
    }

    #[test]
    fn test_log_base_complex_array() {
        use crate::math::LogBase;

        // Complex64 arrays use our LogBase trait for .log(base)
        let arr = crate::prelude::array![Complex64::new(10.0, 0.0), Complex64::new(100.0, 0.0)];
        let result = arr.log(10.0);

        assert!((result[0].re - 1.0).abs() < 1e-10);
        assert!(result[0].im.abs() < 1e-10);
        assert!((result[1].re - 2.0).abs() < 1e-10);
        assert!(result[1].im.abs() < 1e-10);
    }
}
