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

//! Non-linear curve fitting using a damped least-squares algorithm.
//!
//! Rust implementation of `scipy.optimize.curve_fit` for simple float model
//! functions.

use nalgebra::{DMatrix, DVector};
use ndarray::{Array1, Array2, ArrayView1};

const DIFF_STEP: f64 = 1e-8;
const MIN_LAMBDA: f64 = 1e-15;
const MAX_LAMBDA: f64 = 1e15;
const MAX_DAMPING_ATTEMPTS: usize = 16;

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

/// Options for curve fitting.
#[derive(Debug, Clone)]
pub struct CurveFitOptions {
    /// Maximum number of iterations
    pub maxfev: usize,
    /// Tolerance for parameter changes
    pub xtol: f64,
    /// Tolerance for cost changes
    pub ftol: f64,
    /// Initial damping parameter
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

struct SolverState {
    params: DVector<f64>,
    jacobian: DMatrix<f64>,
    cost: f64,
    nfev: usize,
}

fn all_finite(values: &DVector<f64>) -> bool {
    values.iter().all(|value| value.is_finite())
}

fn max_abs(values: &DVector<f64>) -> f64 {
    values
        .iter()
        .fold(0.0_f64, |max_value, value| max_value.max(value.abs()))
}

fn evaluate_residuals<F>(
    func: &F,
    xdata: &[f64],
    ydata: &[f64],
    params: &DVector<f64>,
) -> Result<DVector<f64>, CurveFitError>
where
    F: Fn(f64, &[f64]) -> f64,
{
    let mut residuals = DVector::zeros(xdata.len());
    let param_slice = params.as_slice();

    for (i, (&x, &y)) in xdata.iter().zip(ydata.iter()).enumerate() {
        let value = func(x, param_slice);
        let residual = value - y;
        if !residual.is_finite() {
            return Err(CurveFitError::NumericalIssue {
                message: "Model function returned a non-finite residual".to_string(),
            });
        }
        residuals[i] = residual;
    }

    Ok(residuals)
}

fn finite_difference_jacobian<F>(
    func: &F,
    xdata: &[f64],
    ydata: &[f64],
    params: &DVector<f64>,
    residuals: &DVector<f64>,
    nfev: &mut usize,
) -> Result<DMatrix<f64>, CurveFitError>
where
    F: Fn(f64, &[f64]) -> f64,
{
    let n = xdata.len();
    let n_params = params.len();
    let mut jacobian = DMatrix::zeros(n, n_params);

    for j in 0..n_params {
        let step = DIFF_STEP * (1.0 + params[j].abs());
        if !step.is_finite() || step <= 0.0 {
            return Err(CurveFitError::NumericalIssue {
                message: "Failed to choose a finite-difference step".to_string(),
            });
        }

        let mut params_plus = params.clone();
        params_plus[j] += step;
        let residuals_plus = evaluate_residuals(func, xdata, ydata, &params_plus)?;
        *nfev += 1;

        for i in 0..n {
            jacobian[(i, j)] = (residuals_plus[i] - residuals[i]) / step;
        }
    }

    Ok(jacobian)
}

fn solve_damped_step(
    jt_j: &DMatrix<f64>,
    gradient: &DVector<f64>,
    lambda: f64,
) -> Option<DVector<f64>> {
    let mut lhs = jt_j.clone();
    for i in 0..lhs.nrows() {
        lhs[(i, i)] += lambda * jt_j[(i, i)].abs().max(1.0);
    }

    let rhs = -gradient;

    if let Some(cholesky) = lhs.clone().cholesky() {
        let delta = cholesky.solve(&rhs);
        if all_finite(&delta) {
            return Some(delta);
        }
    }

    if let Some(delta) = lhs.clone().lu().solve(&rhs)
        && all_finite(&delta)
    {
        return Some(delta);
    }

    match lhs.svd(true, true).solve(&rhs, 1e-12) {
        Ok(delta) if all_finite(&delta) => Some(delta),
        _ => None,
    }
}

fn increase_lambda(lambda: f64) -> f64 {
    (lambda * 10.0).min(MAX_LAMBDA)
}

fn decrease_lambda(lambda: f64) -> f64 {
    (lambda * 0.1).max(MIN_LAMBDA)
}

fn solve_least_squares<F>(
    func: &F,
    xdata: &[f64],
    ydata: &[f64],
    p0: DVector<f64>,
    opts: &CurveFitOptions,
) -> Result<SolverState, CurveFitError>
where
    F: Fn(f64, &[f64]) -> f64,
{
    let maxfev = opts.maxfev.max(1);
    let xtol = if opts.xtol.is_finite() && opts.xtol > 0.0 {
        opts.xtol
    } else {
        CurveFitOptions::default().xtol
    };
    let ftol = if opts.ftol.is_finite() && opts.ftol > 0.0 {
        opts.ftol
    } else {
        CurveFitOptions::default().ftol
    };
    let mut lambda = if opts.lambda.is_finite() && opts.lambda > 0.0 {
        opts.lambda
    } else {
        CurveFitOptions::default().lambda
    }
    .clamp(MIN_LAMBDA, MAX_LAMBDA);

    let mut params = p0;
    let mut residuals = evaluate_residuals(func, xdata, ydata, &params)?;
    let mut cost = residuals.dot(&residuals);
    let mut nfev = 1;

    while nfev < maxfev {
        let jacobian =
            finite_difference_jacobian(func, xdata, ydata, &params, &residuals, &mut nfev)?;
        let jt = jacobian.transpose();
        let jt_j = &jt * &jacobian;
        let gradient = &jt * &residuals;
        let gradient_norm = max_abs(&gradient);

        if gradient_norm <= ftol || cost <= ftol * ftol {
            return Ok(SolverState {
                params,
                jacobian,
                cost,
                nfev,
            });
        }

        let mut accepted = false;

        for _ in 0..MAX_DAMPING_ATTEMPTS {
            if nfev >= maxfev {
                break;
            }

            let Some(delta) = solve_damped_step(&jt_j, &gradient, lambda) else {
                lambda = increase_lambda(lambda);
                continue;
            };

            let step_norm = delta.norm();
            let step_converged = step_norm <= xtol * (xtol + params.norm());

            let trial_params = &params + &delta;
            if !all_finite(&trial_params) {
                lambda = increase_lambda(lambda);
                continue;
            }

            let trial_residuals = evaluate_residuals(func, xdata, ydata, &trial_params)?;
            nfev += 1;
            let trial_cost = trial_residuals.dot(&trial_residuals);

            if trial_cost.is_finite() && trial_cost < cost {
                params = trial_params;
                residuals = trial_residuals;
                cost = trial_cost;
                lambda = decrease_lambda(lambda);
                accepted = true;

                if step_converged || cost <= ftol * ftol {
                    let jacobian = finite_difference_jacobian(
                        func, xdata, ydata, &params, &residuals, &mut nfev,
                    )?;
                    return Ok(SolverState {
                        params,
                        jacobian,
                        cost,
                        nfev,
                    });
                }

                break;
            }

            lambda = increase_lambda(lambda);
        }

        if !accepted {
            return Err(CurveFitError::ConvergenceError {
                message: format!(
                    "Optimization did not converge: no improving step found after {nfev} evaluations"
                ),
            });
        }
    }

    Err(CurveFitError::ConvergenceError {
        message: format!("Optimization did not converge within {maxfev} evaluations"),
    })
}

/// Fit a non-linear function to data using damped least-squares.
///
/// This is a Rust implementation of `scipy.optimize.curve_fit` for simple
/// float model functions.
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

    let xdata_vec = xdata.to_vec();
    let ydata_vec = ydata.to_vec();
    let state = solve_least_squares(
        &func,
        &xdata_vec,
        &ydata_vec,
        DVector::from_vec(p0.to_vec()),
        &opts,
    )?;

    // Compute covariance matrix: (J^T * J)^-1 * variance
    let jt_j = state.jacobian.transpose() * &state.jacobian;
    let pcov = match jt_j.svd(true, true).pseudo_inverse(1e-15) {
        Ok(inv) => {
            let n_params = state.params.len();
            // Cast to f64 is safe for reasonable dataset sizes (< 2^53 points)
            #[allow(clippy::cast_precision_loss)]
            let dof = (n as f64 - n_params as f64).max(1.0);
            let variance = state.cost / dof;
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
        state.nfev,
        state.cost
    );

    Ok(CurveFitResult {
        params: Array1::from_vec(state.params.as_slice().to_vec()),
        pcov,
        nfev: state.nfev,
        cost: state.cost,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, array};

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
    fn test_curve_fit_gaussian() {
        fn gaussian(x: f64, params: &[f64]) -> f64 {
            let amp = params[0];
            let mu = params[1];
            let sigma = params[2];
            amp * (-((x - mu).powi(2)) / (2.0 * sigma * sigma)).exp()
        }

        let xdata = Array1::linspace(-5.0, 5.0, 50);
        let ydata = xdata.mapv(|x| gaussian(x, &[2.0, 1.0, 1.5]));
        let p0 = array![1.0, 0.0, 1.0];
        let options = CurveFitOptions {
            maxfev: 5000,
            ..CurveFitOptions::default()
        };

        let result = curve_fit(
            gaussian,
            xdata.view(),
            ydata.view(),
            p0.view(),
            Some(options),
        )
        .unwrap();

        assert!(
            (result.params[0] - 2.0).abs() < 1e-5,
            "amplitude should be 2.0, got {}",
            result.params[0]
        );
        assert!(
            (result.params[1] - 1.0).abs() < 1e-5,
            "mean should be 1.0, got {}",
            result.params[1]
        );
        assert!(
            (result.params[2] - 1.5).abs() < 1e-5,
            "sigma should be 1.5, got {}",
            result.params[2]
        );
    }

    #[test]
    fn test_curve_fit_sine() {
        fn sine(x: f64, params: &[f64]) -> f64 {
            params[0] * (2.0 * std::f64::consts::PI * params[1] * x + params[2]).sin()
        }

        let xdata = Array1::linspace(0.0, 2.0, 100);
        let ydata = xdata.mapv(|x| sine(x, &[1.5, 2.0, 0.5]));
        let p0 = array![1.0, 2.0, 0.0];
        let options = CurveFitOptions {
            maxfev: 5000,
            ..CurveFitOptions::default()
        };

        let result = curve_fit(sine, xdata.view(), ydata.view(), p0.view(), Some(options)).unwrap();

        assert!(
            (result.params[0] - 1.5).abs() < 1e-5,
            "amplitude should be 1.5, got {}",
            result.params[0]
        );
        assert!(
            (result.params[1] - 2.0).abs() < 1e-5,
            "frequency should be 2.0, got {}",
            result.params[1]
        );
        assert!(
            (result.params[2] - 0.5).abs() < 1e-5,
            "phase should be 0.5, got {}",
            result.params[2]
        );
    }

    #[test]
    fn test_curve_fit_non_finite_residual() {
        fn non_finite(x: f64, params: &[f64]) -> f64 {
            params[0] / x
        }

        let xdata = array![0.0, 1.0, 2.0];
        let ydata = array![1.0, 2.0, 3.0];
        let p0 = array![1.0];

        let result = curve_fit(non_finite, xdata.view(), ydata.view(), p0.view(), None);
        assert!(matches!(result, Err(CurveFitError::NumericalIssue { .. })));
    }

    #[test]
    fn test_curve_fit_maxfev_too_small() {
        fn linear(x: f64, params: &[f64]) -> f64 {
            params[0] * x + params[1]
        }

        let xdata = array![0.0, 1.0, 2.0, 3.0];
        let ydata = array![1.0, 3.0, 5.0, 7.0];
        let p0 = array![1.0, 0.0];
        let options = CurveFitOptions {
            maxfev: 1,
            ..CurveFitOptions::default()
        };

        let result = curve_fit(linear, xdata.view(), ydata.view(), p0.view(), Some(options));
        assert!(matches!(
            result,
            Err(CurveFitError::ConvergenceError { .. })
        ));
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
