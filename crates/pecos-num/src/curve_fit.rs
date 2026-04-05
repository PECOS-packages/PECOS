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

//! Non-linear curve fitting using Levenberg-Marquardt algorithm.
//!
//! Rust implementation of `scipy.optimize.curve_fit`
//! using the well-tested `levenberg-marquardt` crate.
//!
//! Note: We use `levenberg-marquardt` instead of Peroxide's optimizer because
//! Peroxide requires AD (automatic differentiation) types, while `scipy.optimize.curve_fit`
//! uses simple float functions. The levenberg-marquardt crate provides a better API match.

use levenberg_marquardt::{LeastSquaresProblem, LevenbergMarquardt};
use nalgebra::{DMatrix, DVector, Dyn, Owned};
use ndarray::{Array1, Array2, ArrayView1};

/// Error type for curve fitting operations.
#[derive(Debug, Clone)]
pub enum CurveFitError {
    /// Optimization failed to converge
    ConvergenceError { message: String },
    /// Invalid input data
    InvalidInput { message: String },
    /// Numerical issues during fitting
    NumericalIssue { message: String },
}

impl std::fmt::Display for CurveFitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ConvergenceError { message } => write!(f, "Convergence error: {message}"),
            Self::InvalidInput { message } => write!(f, "Invalid input: {message}"),
            Self::NumericalIssue { message } => write!(f, "Numerical issue: {message}"),
        }
    }
}

impl std::error::Error for CurveFitError {}

/// Problem struct for Levenberg-Marquardt optimization.
struct CurveFitProblem<F>
where
    F: Fn(f64, &[f64]) -> f64,
{
    func: F,
    xdata: Vec<f64>,
    ydata: Vec<f64>,
    params: DVector<f64>,
}

impl<F> LeastSquaresProblem<f64, Dyn, Dyn> for CurveFitProblem<F>
where
    F: Fn(f64, &[f64]) -> f64,
{
    type ParameterStorage = Owned<f64, Dyn>;
    type ResidualStorage = Owned<f64, Dyn>;
    type JacobianStorage = Owned<f64, Dyn, Dyn>;

    fn set_params(&mut self, p: &DVector<f64>) {
        self.params.copy_from(p);
    }

    fn params(&self) -> DVector<f64> {
        self.params.clone()
    }

    fn residuals(&self) -> Option<DVector<f64>> {
        let n = self.xdata.len();
        let mut residuals = DVector::zeros(n);
        let param_slice = self.params.as_slice();

        for (i, (&x, &y)) in self.xdata.iter().zip(self.ydata.iter()).enumerate() {
            residuals[i] = (self.func)(x, param_slice) - y;
        }

        Some(residuals)
    }

    fn jacobian(&self) -> Option<DMatrix<f64>> {
        let n = self.xdata.len();
        let n_params = self.params.len();
        let mut jacobian = DMatrix::zeros(n, n_params);

        let eps = 1e-8;
        let residuals = self.residuals()?;
        let param_slice = self.params.as_slice();

        for j in 0..n_params {
            let step = eps * (1.0 + param_slice[j].abs()).max(eps);

            // Create perturbed parameters
            let mut params_plus = self.params.clone();
            params_plus[j] += step;
            let params_plus_slice = params_plus.as_slice();

            // Compute residuals with perturbed parameters
            for (i, &x) in self.xdata.iter().enumerate() {
                let residual_plus = (self.func)(x, params_plus_slice) - self.ydata[i];
                jacobian[(i, j)] = (residual_plus - residuals[i]) / step;
            }
        }

        Some(jacobian)
    }
}

/// Options for curve fitting.
#[derive(Debug, Clone)]
pub struct CurveFitOptions {
    /// Maximum number of iterations
    pub maxfev: usize,
    /// Tolerance for parameter changes
    pub xtol: f64,
    /// Tolerance for cost changes
    pub ftol: f64,
    /// Initial damping parameter (ignored, using crate defaults)
    pub lambda: f64,
}

impl Default for CurveFitOptions {
    fn default() -> Self {
        Self {
            maxfev: 1000,
            xtol: 1e-8,
            ftol: 1e-8,
            lambda: 0.01,
        }
    }
}

/// Result from curve fitting.
#[derive(Debug, Clone)]
pub struct CurveFitResult {
    /// Optimal parameters
    pub params: Array1<f64>,
    /// Covariance matrix (if available)
    pub pcov: Option<Array2<f64>>,
    /// Number of function evaluations
    pub nfev: usize,
    /// Final cost value
    pub cost: f64,
}

// No problem struct needed - we'll use a closure directly

/// Fit a non-linear function to data using Levenberg-Marquardt.
///
/// This is a Rust implementation of `scipy.optimize.curve_fit` using the
/// `levenberg-marquardt` crate for robust, well-tested optimization.
///
/// # Arguments
///
/// * `func` - Model function: f(x, params) -> y
/// * `xdata` - Independent variable data
/// * `ydata` - Dependent variable data
/// * `p0` - Initial guess for parameters
/// * `options` - Optional fitting options
///
/// # Returns
///
/// Returns the optimal parameters and covariance matrix.
///
/// # Errors
///
/// Returns an error if:
/// - Data arrays have different lengths
/// - Optimization fails to converge
/// - Numerical issues during fitting
///
/// # Examples
///
/// ```
/// use pecos_num::curve_fit::{curve_fit, CurveFitOptions};
/// use ndarray::array;
///
/// // Fit linear: y = a * x + b
/// fn linear(x: f64, params: &[f64]) -> f64 {
///     params[0] * x + params[1]
/// }
///
/// let xdata = array![0.0, 1.0, 2.0, 3.0, 4.0];
/// let ydata = array![1.0, 3.0, 5.0, 7.0, 9.0];
/// let p0 = array![1.0, 0.0];
///
/// let result = curve_fit(linear, xdata.view(), ydata.view(), p0.view(), None).unwrap();
/// // result.params ≈ [2.0, 1.0] (for y = 2*x + 1)
/// ```
#[allow(clippy::needless_pass_by_value)] // ArrayView is a borrowed view, designed to be passed by value
pub fn curve_fit<F>(
    func: F,
    xdata: ArrayView1<f64>,
    ydata: ArrayView1<f64>,
    p0: ArrayView1<f64>,
    options: Option<CurveFitOptions>,
) -> Result<CurveFitResult, CurveFitError>
where
    F: Fn(f64, &[f64]) -> f64,
{
    let n = xdata.len();

    if n != ydata.len() {
        return Err(CurveFitError::InvalidInput {
            message: format!(
                "xdata and ydata must have same length: x={n}, y={}",
                ydata.len()
            ),
        });
    }

    if n < p0.len() {
        return Err(CurveFitError::InvalidInput {
            message: format!(
                "Need at least {} data points for {} parameters, got {n}",
                p0.len(),
                p0.len()
            ),
        });
    }

    let opts = options.unwrap_or_default();

    // Create problem for Levenberg-Marquardt
    let problem = CurveFitProblem {
        func,
        xdata: xdata.to_vec(),
        ydata: ydata.to_vec(),
        params: DVector::from_vec(p0.to_vec()),
    };

    // Run Levenberg-Marquardt optimization
    let (result, report) = LevenbergMarquardt::new()
        .with_stepbound(100.0)
        .with_patience(opts.maxfev)
        .minimize(problem);

    // Check convergence
    if !report.termination.was_successful() {
        return Err(CurveFitError::ConvergenceError {
            message: format!("Optimization did not converge: {:?}", report.termination),
        });
    }

    // Get final parameters and residuals
    let final_params = result.params();
    let final_residuals = result
        .residuals()
        .ok_or_else(|| CurveFitError::NumericalIssue {
            message: "Failed to compute final residuals".to_string(),
        })?;
    let cost = final_residuals.dot(&final_residuals);

    // Get Jacobian at solution
    let jacobian = result
        .jacobian()
        .ok_or_else(|| CurveFitError::NumericalIssue {
            message: "Failed to compute Jacobian".to_string(),
        })?;

    // Compute covariance matrix: (J^T * J)^-1 * variance
    let jt_j = jacobian.transpose() * &jacobian;
    let pcov = match jt_j.svd(true, true).pseudo_inverse(1e-15) {
        Ok(inv) => {
            let n_params = final_params.len();
            // Cast to f64 is safe for reasonable dataset sizes (< 2^53 points)
            #[allow(clippy::cast_precision_loss)]
            let dof = (n as f64 - n_params as f64).max(1.0);
            let variance = cost / dof;
            let cov_mat = inv * variance;

            // Convert to ndarray
            let mut pcov_array = Array2::zeros((n_params, n_params));
            for i in 0..n_params {
                for j in 0..n_params {
                    pcov_array[[i, j]] = cov_mat[(i, j)];
                }
            }
            Some(pcov_array)
        }
        Err(_) => None,
    };

    log::debug!(
        "curve_fit: converged after {} evaluations with cost={:.6e}",
        report.number_of_evaluations,
        cost
    );

    Ok(CurveFitResult {
        params: Array1::from_vec(final_params.as_slice().to_vec()),
        pcov,
        nfev: report.number_of_evaluations,
        cost,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_curve_fit_linear() {
        // Fit y = a*x + b
        fn linear(x: f64, params: &[f64]) -> f64 {
            params[0] * x + params[1]
        }

        let xdata = array![0.0, 1.0, 2.0, 3.0, 4.0];
        let ydata = array![1.0, 3.0, 5.0, 7.0, 9.0]; // y = 2*x + 1

        let p0 = array![1.0, 0.0];

        let result = curve_fit(linear, xdata.view(), ydata.view(), p0.view(), None).unwrap();

        assert!(
            (result.params[0] - 2.0).abs() < 1e-6,
            "slope should be 2.0, got {}",
            result.params[0]
        );
        assert!(
            (result.params[1] - 1.0).abs() < 1e-6,
            "intercept should be 1.0, got {}",
            result.params[1]
        );
    }

    #[test]
    fn test_curve_fit_exponential() {
        // Fit y = a * exp(b * x)
        fn exponential(x: f64, params: &[f64]) -> f64 {
            params[0] * (params[1] * x).exp()
        }

        let xdata = array![0.0, 1.0, 2.0, 3.0, 4.0];
        // y = e^x: use std::f64::consts::E for accurate test data
        let ydata = array![
            1.0_f64.exp(),
            std::f64::consts::E,
            (2.0_f64).exp(),
            (3.0_f64).exp(),
            (4.0_f64).exp()
        ];

        let p0 = array![1.0, 1.0];

        let result = curve_fit(exponential, xdata.view(), ydata.view(), p0.view(), None).unwrap();

        assert!(
            (result.params[0] - 1.0).abs() < 0.05,
            "coefficient should be ~1.0, got {}",
            result.params[0]
        );
        assert!(
            (result.params[1] - 1.0).abs() < 0.05,
            "exponent should be ~1.0, got {}",
            result.params[1]
        );
    }

    #[test]
    fn test_curve_fit_quadratic() {
        // Fit y = a*x^2 + b*x + c
        fn quadratic(x: f64, params: &[f64]) -> f64 {
            params[0] * x * x + params[1] * x + params[2]
        }

        let xdata = array![0.0, 1.0, 2.0, 3.0, 4.0];
        let ydata = array![3.0, 6.0, 11.0, 18.0, 27.0]; // y = x^2 + 2*x + 3

        let p0 = array![1.0, 1.0, 1.0];

        let result = curve_fit(quadratic, xdata.view(), ydata.view(), p0.view(), None).unwrap();

        assert!(
            (result.params[0] - 1.0).abs() < 1e-6,
            "x^2 coef should be 1.0, got {}",
            result.params[0]
        );
        assert!(
            (result.params[1] - 2.0).abs() < 1e-6,
            "x coef should be 2.0, got {}",
            result.params[1]
        );
        assert!(
            (result.params[2] - 3.0).abs() < 1e-6,
            "constant should be 3.0, got {}",
            result.params[2]
        );
    }

    #[test]
    fn test_curve_fit_insufficient_data() {
        fn linear(x: f64, params: &[f64]) -> f64 {
            params[0] * x + params[1]
        }

        let xdata = array![0.0];
        let ydata = array![1.0];
        let p0 = array![1.0, 0.0];

        let result = curve_fit(linear, xdata.view(), ydata.view(), p0.view(), None);
        assert!(matches!(result, Err(CurveFitError::InvalidInput { .. })));
    }
}
