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

//! Root finding and optimization algorithms.
//!
//! This module provides implementations of common numerical optimization
//! algorithms, compatible with scipy.optimize API.
//!
//! Uses Peroxide for Newton's method implementation, with scipy-compatible
//! functional wrappers.

use peroxide::fuga::{NewtonMethod, RootFinder, RootFindingProblem, anyhow};
use std::fmt;

/// Error type for optimization functions.
#[derive(Debug, Clone)]
pub enum OptimizeError {
    /// Function values at interval endpoints have the same sign
    SameSigns { fa: f64, fb: f64 },
    /// Maximum iterations exceeded without convergence
    MaxIterations { iterations: usize },
    /// Derivative is zero or near-zero
    ZeroDerivative { x: f64, derivative: f64 },
    /// Numerical issues (NaN, Inf encountered)
    NumericalIssue { message: String },
    /// Convergence criterion not met
    ConvergenceFailed { message: String },
}

impl fmt::Display for OptimizeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SameSigns { fa, fb } => {
                write!(
                    f,
                    "f(a) and f(b) must have opposite signs. Got f(a)={fa}, f(b)={fb}"
                )
            }
            Self::MaxIterations { iterations } => {
                write!(f, "Maximum iterations ({iterations}) exceeded")
            }
            Self::ZeroDerivative { x, derivative } => {
                write!(f, "Derivative is zero at x={x} (derivative={derivative})")
            }
            Self::NumericalIssue { message } => {
                write!(f, "Numerical issue: {message}")
            }
            Self::ConvergenceFailed { message } => {
                write!(f, "Convergence failed: {message}")
            }
        }
    }
}

impl std::error::Error for OptimizeError {}

/// Options for Brent's method root finding.
#[derive(Debug, Clone)]
pub struct BrentqOptions {
    /// Absolute tolerance for root finding
    pub xtol: f64,
    /// Relative tolerance for root finding
    pub rtol: f64,
    /// Maximum number of iterations
    pub maxiter: usize,
}

impl Default for BrentqOptions {
    fn default() -> Self {
        Self {
            xtol: 2e-12,
            rtol: 8.881_784_197_001_252e-16, // scipy default
            maxiter: 100,
        }
    }
}

/// Options for Newton-Raphson method.
#[derive(Debug, Clone)]
pub struct NewtonOptions {
    /// Absolute tolerance for convergence
    pub tol: f64,
    /// Maximum number of iterations
    pub maxiter: usize,
    /// Step size for numerical derivative (if fprime not provided)
    pub eps: f64,
}

impl Default for NewtonOptions {
    fn default() -> Self {
        Self {
            tol: 1.48e-8, // scipy default
            maxiter: 50,
            eps: 1e-8,
        }
    }
}

/// Find root of a function using Brent's method.
///
/// This is a Rust implementation of scipy.optimize.brentq.
///
/// Brent's method combines root bracketing, bisection, and inverse quadratic
/// interpolation. It is generally considered one of the best methods for
/// finding roots of a continuous function.
///
/// # Arguments
///
/// * `f` - Function for which to find root
/// * `a` - Lower bound of interval
/// * `b` - Upper bound of interval
/// * `options` - Optional configuration parameters
///
/// # Returns
///
/// Returns the root of the function within the interval [a, b].
///
/// # Errors
///
/// Returns an error if:
/// - f(a) and f(b) have the same sign
/// - Maximum iterations exceeded
/// - Numerical issues encountered
///
/// # Examples
///
/// ```
/// use pecos_num::optimize::{brentq, BrentqOptions};
///
/// // Find root of f(x) = x^2 - 2 (should be sqrt(2))
/// let root = brentq(|x| x * x - 2.0, 0.0, 2.0, None).unwrap();
/// assert!((root - 2f64.sqrt()).abs() < 1e-10);
/// ```
pub fn brentq<F>(f: F, a: f64, b: f64, options: Option<BrentqOptions>) -> Result<f64, OptimizeError>
where
    F: Fn(f64) -> f64,
{
    let opts = options.unwrap_or_default();

    // Use roots crate for Brent's method
    let mut convergency = roots::SimpleConvergency {
        eps: opts.xtol,
        max_iter: opts.maxiter,
    };

    let result = roots::find_root_brent(a, b, &f, &mut convergency);

    match result {
        Ok(root) => {
            log::debug!("brentq converged to root={root}");
            Ok(root)
        }
        Err(e) => {
            log::warn!("brentq failed: {e:?}");
            // Check if it's a sign issue
            let fa = f(a);
            let fb = f(b);
            if fa * fb > 0.0 {
                Err(OptimizeError::SameSigns { fa, fb })
            } else {
                Err(OptimizeError::MaxIterations {
                    iterations: opts.maxiter,
                })
            }
        }
    }
}

/// Internal wrapper for Newton's method using Peroxide.
struct NewtonProblem<F, G>
where
    F: Fn(f64) -> f64,
    G: Fn(f64) -> f64,
{
    f: F,
    fprime: Option<G>,
    eps: f64,
    x0: f64,
}

impl<F, G> RootFindingProblem<1, 1, f64> for NewtonProblem<F, G>
where
    F: Fn(f64) -> f64,
    G: Fn(f64) -> f64,
{
    fn function(&self, x: [f64; 1]) -> Result<[f64; 1], anyhow::Error> {
        Ok([(self.f)(x[0])])
    }

    fn derivative(&self, x: [f64; 1]) -> Result<[[f64; 1]; 1], anyhow::Error> {
        let fprime_x = if let Some(ref fprime_fn) = self.fprime {
            (fprime_fn)(x[0])
        } else {
            // Numerical derivative using finite differences
            let h = self.eps;
            let fx = (self.f)(x[0]);
            let fx_plus_h = (self.f)(x[0] + h);
            (fx_plus_h - fx) / h
        };

        Ok([[fprime_x]])
    }

    fn initial_guess(&self) -> f64 {
        self.x0
    }
}

/// Find root using Newton-Raphson method.
///
/// This is a scipy.optimize.newton-compatible wrapper around Peroxide's Newton implementation.
///
/// Newton's method uses the function value and its derivative to iteratively
/// converge to a root. It typically converges quickly when close to the root,
/// but may fail if the initial guess is poor or the derivative is zero.
///
/// # Arguments
///
/// * `f` - Function for which to find root
/// * `x0` - Initial guess
/// * `fprime` - Optional derivative function. If None, uses numerical derivative.
/// * `options` - Optional configuration parameters
///
/// # Returns
///
/// Returns the root of the function.
///
/// # Errors
///
/// Returns an error if:
/// - Maximum iterations exceeded
/// - Derivative is zero or near-zero
/// - Numerical issues encountered
///
/// # Examples
///
/// ```
/// use pecos_num::optimize::{newton, NewtonOptions};
///
/// // Find root of f(x) = x^2 - 2 (should be sqrt(2))
/// // With derivative f'(x) = 2x
/// let root = newton(
///     |x| x * x - 2.0,
///     1.0,  // initial guess
///     Some(|x| 2.0 * x),  // derivative
///     None
/// ).unwrap();
/// assert!((root - 2f64.sqrt()).abs() < 1e-10);
/// ```
pub fn newton<F, G>(
    f: F,
    x0: f64,
    fprime: Option<G>,
    options: Option<NewtonOptions>,
) -> Result<f64, OptimizeError>
where
    F: Fn(f64) -> f64,
    G: Fn(f64) -> f64,
{
    let opts = options.unwrap_or_default();

    log::debug!("newton starting from x0={x0}");

    // Create Peroxide problem
    let problem = NewtonProblem {
        f,
        fprime,
        eps: opts.eps,
        x0,
    };

    // Create Peroxide Newton method
    let method = NewtonMethod {
        max_iter: opts.maxiter,
        tol: opts.tol,
    };

    // Solve using Peroxide
    let result = method.find(&problem);

    match result {
        Ok(root) => {
            log::debug!("newton converged to root={}", root[0]);
            Ok(root[0])
        }
        Err(e) => {
            log::warn!("newton failed: {e:?}");
            Err(OptimizeError::ConvergenceFailed {
                message: format!("{e:?}"),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_brentq_sqrt2() {
        // Find sqrt(2) by solving x^2 - 2 = 0
        let root = brentq(|x| x * x - 2.0, 0.0, 2.0, None).unwrap();
        assert!((root - 2f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_brentq_cubic() {
        // Find root of x^3 - x - 2 = 0 (root is approximately 1.52138)
        let root = brentq(|x| x.powi(3) - x - 2.0, 1.0, 2.0, None).unwrap();
        let expected = 1.521_379_706_804_567_6;
        assert!((root - expected).abs() < 1e-10);
    }

    #[test]
    fn test_brentq_same_signs() {
        // Should fail when f(a) and f(b) have same sign
        let result = brentq(|x| x * x + 1.0, -1.0, 1.0, None);
        assert!(matches!(result, Err(OptimizeError::SameSigns { .. })));
    }

    #[test]
    fn test_newton_sqrt2() {
        // Find sqrt(2) using Newton's method with derivative
        let root = newton(|x| x * x - 2.0, 1.0, Some(|x: f64| 2.0 * x), None).unwrap();
        assert!((root - 2f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_newton_numerical_derivative() {
        // Find sqrt(2) using Newton's method with numerical derivative
        let root = newton(|x| x * x - 2.0, 1.0, None::<fn(f64) -> f64>, None).unwrap();
        assert!((root - 2f64.sqrt()).abs() < 1e-8);
    }

    #[test]
    fn test_newton_cubic() {
        // Find root of x^3 - x - 2 = 0
        let root = newton(
            |x| x.powi(3) - x - 2.0,
            1.5,
            Some(|x: f64| 3.0 * x.powi(2) - 1.0),
            None,
        )
        .unwrap();
        let expected = 1.521_379_706_804_567_6;
        assert!((root - expected).abs() < 1e-10);
    }

    #[test]
    fn test_newton_polynomial_root() {
        // Find root of (x-3)(x-5) = x^2 - 8x + 15 = 0
        // Should find root near initial guess (close to 5)
        let root = newton(
            |x| x * x - 8.0 * x + 15.0,
            4.5, // Start at 4.5, not 4.0 (which has zero derivative)
            Some(|x: f64| 2.0 * x - 8.0),
            None,
        )
        .unwrap();
        // Should converge to 5 since we start at 4.5
        assert!((root - 5.0).abs() < 1e-10);
    }
}
