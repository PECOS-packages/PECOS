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

//! Python bindings for pecos-num numerical computing functions.
//!
//! This module provides drop-in replacements for scipy.optimize functions,
//! implemented in Rust for better performance and easier deployment.

use numpy::ndarray::Array1;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

// Import numerical computing types from pecos prelude
// Functions are accessed via pecos::prelude module
use pecos::prelude::{
    BrentqOptions, CurveFitError, CurveFitOptions, NewtonOptions, Poly1d as RustPoly1d,
};

/// Helper function to convert `CurveFitError` to appropriate Python exception.
///
/// Maps Rust errors to Python exceptions following `scipy.optimize.curve_fit` conventions:
/// - `ConvergenceError` -> `RuntimeError` (scipy raises `RuntimeError` for convergence failures)
/// - `InvalidInput` -> `ValueError` (standard Python convention for invalid inputs)
/// - `NumericalIssue` -> `RuntimeError` (similar to convergence issues)
fn map_curve_fit_error(error: CurveFitError) -> PyErr {
    match error {
        CurveFitError::InvalidInput { message } => {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("curve_fit failed: {message}"))
        }
        CurveFitError::ConvergenceError { message } | CurveFitError::NumericalIssue { message } => {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "curve_fit failed: {message}"
            ))
        }
    }
}

/// Find root of a function using Brent's method.
///
/// This is a drop-in replacement for scipy.optimize.brentq.
///
/// Args:
///     f: Callable[[float], float] - Function for which to find root
///     a: float - Lower bound of interval
///     b: float - Upper bound of interval
///     xtol: float - Absolute tolerance (default: 2e-12)
///     rtol: float - Relative tolerance (default: 8.881784197001252e-16)
///     maxiter: int - Maximum iterations (default: 100)
///
/// Returns:
///     float: The root of the function
///
/// Raises:
///     `ValueError`: If f(a) and f(b) have the same sign
///     `RuntimeError`: If maximum iterations exceeded
///
/// Examples:
///     >>> from `pecos_rslib.num` import brentq
///     >>> # Find sqrt(2) by solving x^2 - 2 = 0
///     >>> root = brentq(lambda x: x**2 - 2, 0, 2)
///     >>> abs(root - 2**0.5) < 1e-10
///     True
#[pyfunction]
#[pyo3(signature = (f, a, b, xtol=None, rtol=None, maxiter=None))]
#[allow(clippy::needless_pass_by_value)] // Py<PyAny> is a cheap ref-counted pointer; closure needs ownership
fn brentq(
    _py: Python<'_>,
    f: Py<PyAny>,
    a: f64,
    b: f64,
    xtol: Option<f64>,
    rtol: Option<f64>,
    maxiter: Option<usize>,
) -> PyResult<f64> {
    // Create closure that calls Python function
    let func = |x: f64| -> f64 {
        Python::attach(|py| {
            f.call1(py, (x,))
                .and_then(|result| result.extract::<f64>(py))
                .unwrap_or(f64::NAN)
        })
    };

    // Configure options
    let opts = BrentqOptions {
        xtol: xtol.unwrap_or(2e-12),
        rtol: rtol.unwrap_or(8.881_784_197_001_252e-16),
        maxiter: maxiter.unwrap_or(100),
    };

    // Call Rust implementation
    pecos::prelude::brentq(func, a, b, Some(opts))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("brentq failed: {e}")))
}

/// Find root using Newton-Raphson method.
///
/// This is a drop-in replacement for scipy.optimize.newton.
///
/// Args:
///     func: Callable[[float], float] - Function for which to find root
///     x0: float - Initial guess
///     fprime: Optional[Callable[[float], float]] - Derivative function (default: None uses numerical derivative)
///     tol: float - Convergence tolerance (default: 1.48e-8)
///     maxiter: int - Maximum iterations (default: 50)
///
/// Returns:
///     float: The root of the function
///
/// Raises:
///     `ValueError`: If derivative is zero
///     `RuntimeError`: If maximum iterations exceeded or convergence fails
///
/// Examples:
///     >>> from `pecos_rslib.num` import newton
///     >>> # Find sqrt(2) by solving x^2 - 2 = 0
///     >>> root = newton(lambda x: x**2 - 2, x0=1.0, fprime=lambda x: 2*x)
///     >>> abs(root - 2**0.5) < 1e-10
///     True
#[pyfunction]
#[pyo3(signature = (func, x0, fprime=None, tol=None, maxiter=None))]
#[allow(clippy::needless_pass_by_value)] // Py<PyAny> is a cheap ref-counted pointer; closures need ownership
fn newton(
    _py: Python<'_>,
    func: Py<PyAny>,
    x0: f64,
    fprime: Option<Py<PyAny>>,
    tol: Option<f64>,
    maxiter: Option<usize>,
) -> PyResult<f64> {
    // Create closure for function
    let f = |x: f64| -> f64 {
        Python::attach(|py| {
            func.call1(py, (x,))
                .and_then(|result| result.extract::<f64>(py))
                .unwrap_or(f64::NAN)
        })
    };

    // Configure options
    let opts = NewtonOptions {
        tol: tol.unwrap_or(1.48e-8),
        maxiter: maxiter.unwrap_or(50),
        eps: 1e-8,
    };

    // Call Rust implementation
    let result = if let Some(fprime_fn) = fprime {
        // Use provided derivative
        let fprime_closure = |x: f64| -> f64 {
            Python::attach(|py| {
                fprime_fn
                    .call1(py, (x,))
                    .and_then(|result| result.extract::<f64>(py))
                    .unwrap_or(f64::NAN)
            })
        };
        pecos::prelude::newton(f, x0, Some(fprime_closure), Some(opts))
    } else {
        // Use numerical derivative
        pecos::prelude::newton(f, x0, None::<fn(f64) -> f64>, Some(opts))
    };

    result.map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("newton failed: {e}"))
    })
}

/// Fit a polynomial of given degree to data points.
///
/// This is a drop-in replacement for numpy.polyfit.
///
/// Args:
///     x: `array_like` - x-coordinates of data points
///     y: `array_like` - y-coordinates of data points
///     deg: int - Degree of the polynomial fit
///
/// Returns:
///     ndarray: Polynomial coefficients in decreasing order of degree
///              For example, for degree 2: [c0, c1, c2] where y = c0*x^2 + c1*x + c2
///
/// Raises:
///     `ValueError`: If not enough data points for the requested degree
///     `RuntimeError`: If numerical issues during fitting
///
/// Examples:
///     >>> from `pecos_rslib.num` import polyfit
///     >>> import numpy as np
///     >>> # Fit y = 2x + 1
///     >>> x = np.array([0.0, 1.0, 2.0, 3.0])
///     >>> y = np.array([1.0, 3.0, 5.0, 7.0])
///     >>> coeffs = polyfit(x, y, 1)
///     >>> # coeffs ≈ [2.0, 1.0] (slope, intercept)
#[pyfunction]
#[allow(clippy::needless_pass_by_value)] // PyReadonlyArray1 is a lightweight wrapper
fn polyfit(
    py: Python<'_>,
    x: PyReadonlyArray1<f64>,
    y: PyReadonlyArray1<f64>,
    deg: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    let x_view = x.as_array();
    let y_view = y.as_array();

    let coeffs = pecos::prelude::polyfit(x_view, y_view, deg).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("polyfit failed: {e}"))
    })?;

    Ok(PyArray1::from_array(py, &coeffs).unbind())
}

/// Polynomial class for evaluation.
///
/// This is a drop-in replacement for numpy.poly1d.
///
/// Examples:
///     >>> from `pecos_rslib.num` import Poly1d
///     >>> import numpy as np
///     >>> # Create polynomial: 2x^2 + 3x + 1
///     >>> p = Poly1d(np.array([2.0, 3.0, 1.0]))
///     >>> p.eval(0.0)  # p(0) = 1
///     1.0
///     >>> p.eval(1.0)  # p(1) = 2 + 3 + 1 = 6
///     6.0
#[pyclass]
struct Poly1d {
    inner: RustPoly1d,
}

#[pymethods]
impl Poly1d {
    /// Create a new polynomial from coefficients.
    ///
    /// Args:
    ///     coeffs: `array_like` - Coefficients in decreasing order of degree
    #[new]
    #[allow(clippy::needless_pass_by_value)] // PyReadonlyArray1 is a lightweight wrapper
    fn new(coeffs: PyReadonlyArray1<f64>) -> Self {
        let coeffs_array = coeffs.as_array().to_owned();
        Self {
            inner: RustPoly1d::new(coeffs_array),
        }
    }

    /// Evaluate the polynomial at a given value.
    ///
    /// Args:
    ///     x: float - Value at which to evaluate the polynomial
    ///
    /// Returns:
    ///     float: The value of the polynomial at x
    fn eval(&self, x: f64) -> f64 {
        self.inner.eval(x)
    }

    /// Get the degree of the polynomial.
    ///
    /// Returns:
    ///     int: Degree of the polynomial
    fn degree(&self) -> usize {
        self.inner.degree()
    }

    /// Get the polynomial coefficients.
    ///
    /// Returns:
    ///     ndarray: Coefficients in decreasing order of degree
    fn coefficients(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        PyArray1::from_array(py, self.inner.coefficients()).unbind()
    }

    /// Call the polynomial (same as eval).
    fn __call__(&self, x: f64) -> f64 {
        self.inner.eval(x)
    }

    /// String representation of the polynomial.
    fn __repr__(&self) -> String {
        format!("Poly1d(coefficients={:?})", self.inner.coefficients())
    }
}

/// Fit a non-linear function to data using Levenberg-Marquardt.
///
/// This is a drop-in replacement for `scipy.optimize.curve_fit`.
///
/// Args:
///     f: Callable[[float, array], float] - Model function f(x, params) or f((x1, x2, ...), params)
///     xdata: `array_like` or tuple of arrays - Independent variable data (can be single array or tuple of arrays)
///     ydata: `array_like` - Dependent variable data
///     p0: `array_like` - Initial guess for parameters
///     maxfev: int - Maximum function evaluations (default: 1000)
///     xtol: float - Parameter tolerance (default: 1e-8)
///     ftol: float - Cost tolerance (default: 1e-8)
///
/// Returns:
///     tuple: (popt, pcov) - Optimal parameters and covariance matrix
///
/// Raises:
///     `ValueError`: If data arrays have different lengths
///     `RuntimeError`: If optimization fails to converge
///
/// Examples:
///     >>> from `pecos_rslib.num` import `curve_fit`
///     >>> import numpy as np
///     >>> # Example 1: Single independent variable
///     >>> def func(x, a, b):
///     ...     return a * x + b
///     >>> xdata = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
///     >>> ydata = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
///     >>> p0 = np.array([1.0, 0.0])
///     >>> popt, pcov = `curve_fit(func`, xdata, ydata, p0)
///     >>> # popt ≈ [2.0, 1.0]
///     >>>
///     >>> # Example 2: Multiple independent variables (tuple of arrays)
///     >>> def func2(x, a, b):
///     ...     p, d = x  # Unpack tuple
///     ...     return a * p ** (b / d)
///     >>> pdata = np.array([0.1, 0.2, 0.3])
///     >>> ddata = np.array([3.0, 3.0, 3.0])
///     >>> ydata2 = np.array([0.5, 0.7, 0.9])
///     >>> popt2, pcov2 = `curve_fit(func2`, (pdata, ddata), ydata2, np.array([1.0, 1.0]))
#[pyfunction]
#[pyo3(signature = (f, xdata, ydata, p0, maxfev=None, xtol=None, ftol=None))]
#[allow(clippy::type_complexity)] // Complex return type required for scipy compatibility
#[allow(clippy::too_many_arguments)] // scipy.optimize.curve_fit has many parameters
#[allow(clippy::needless_pass_by_value)] // PyReadonlyArray1 is a lightweight wrapper
fn curve_fit<'py>(
    py: Python<'py>,
    f: Py<PyAny>,
    xdata: &Bound<'py, PyAny>,
    ydata: PyReadonlyArray1<f64>,
    p0: &Bound<'py, PyAny>,
    maxfev: Option<usize>,
    xtol: Option<f64>,
    ftol: Option<f64>,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray2<f64>>)> {
    // Convert p0 to array (accept array, tuple, or list)
    let p0_array = if let Ok(array) = p0.extract::<PyReadonlyArray1<f64>>() {
        array
    } else if let Ok(tuple) = p0.cast() {
        // Convert tuple to array
        let values: Vec<f64> = tuple.extract()?;
        let np = py.import("numpy")?;
        let array = np.call_method1("array", (values,))?;
        array.extract::<PyReadonlyArray1<f64>>()?
    } else if let Ok(list) = p0.extract::<Vec<f64>>() {
        // Convert list to array
        let np = py.import("numpy")?;
        let array = np.call_method1("array", (list,))?;
        array.extract::<PyReadonlyArray1<f64>>()?
    } else {
        return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "p0 must be an array, tuple, or list",
        ));
    };

    // Check if xdata is a tuple or a single array
    if let Ok(tuple) = xdata.cast() {
        // Handle tuple case (multiple independent variables)
        curve_fit_tuple(py, f, tuple, ydata, p0_array, maxfev, xtol, ftol)
    } else if let Ok(array) = xdata.extract::<PyReadonlyArray1<f64>>() {
        // Handle single array case
        curve_fit_array(py, f, array, ydata, p0_array, maxfev, xtol, ftol)
    } else {
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "xdata must be an array or tuple of arrays",
        ))
    }
}

/// Helper function for `curve_fit` with single array xdata.
#[allow(clippy::type_complexity)] // Complex return type required for scipy compatibility
#[allow(clippy::too_many_arguments)] // Matches scipy.optimize.curve_fit parameters
#[allow(clippy::needless_pass_by_value)] // PyReadonlyArray1 is a lightweight wrapper
fn curve_fit_array(
    py: Python<'_>,
    f: Py<PyAny>,
    xdata: PyReadonlyArray1<f64>,
    ydata: PyReadonlyArray1<f64>,
    p0: PyReadonlyArray1<f64>,
    maxfev: Option<usize>,
    xtol: Option<f64>,
    ftol: Option<f64>,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray2<f64>>)> {
    let xdata_view = xdata.as_array();
    let ydata_view = ydata.as_array();
    let p0_view = p0.as_array();

    // Create closure that calls Python function
    // The Python function signature is f(x, *params)
    let func = move |x: f64, params: &[f64]| -> f64 {
        Python::attach(|py| {
            // Build arguments tuple: (x, *params)
            let mut args_vec = Vec::with_capacity(1 + params.len());
            args_vec.push(x);
            args_vec.extend_from_slice(params);

            let Ok(tuple) = pyo3::types::PyTuple::new(py, &args_vec) else {
                return f64::NAN;
            };

            match f.call1(py, tuple) {
                Ok(result) => result.extract::<f64>(py).unwrap_or(f64::NAN),
                Err(_) => f64::NAN,
            }
        })
    };

    // Configure options
    let opts = CurveFitOptions {
        maxfev: maxfev.unwrap_or(1000),
        xtol: xtol.unwrap_or(1e-8),
        ftol: ftol.unwrap_or(1e-8),
        lambda: 0.01,
    };

    // Call Rust implementation
    let result = pecos::prelude::curve_fit(func, xdata_view, ydata_view, p0_view, Some(opts))
        .map_err(map_curve_fit_error)?;

    // Convert results to Python arrays
    let popt = PyArray1::from_array(py, &result.params).unbind();

    // If covariance is available, return it; otherwise create identity matrix
    let pcov = if let Some(cov) = result.pcov {
        PyArray2::from_array(py, &cov).unbind()
    } else {
        // Return identity matrix if covariance not available
        let n = result.params.len();
        let mut cov_array = vec![vec![0.0; n]; n];
        for (i, row) in cov_array.iter_mut().enumerate().take(n) {
            row[i] = 1.0;
        }
        PyArray2::from_vec2(py, &cov_array).unwrap().unbind()
    };

    Ok((popt, pcov))
}

/// Helper function for `curve_fit` with tuple of arrays as xdata.
///
/// This handles the scipy behavior where xdata can be a tuple of arrays,
/// and the function f receives tuples of x values.
#[allow(clippy::type_complexity)] // Complex return type required for scipy compatibility
#[allow(clippy::too_many_arguments)] // Matches scipy.optimize.curve_fit parameters
#[allow(clippy::too_many_lines)] // Complex scipy compatibility logic required
#[allow(clippy::needless_pass_by_value)] // PyReadonlyArray1 is a lightweight wrapper
fn curve_fit_tuple<'py>(
    py: Python<'py>,
    f: Py<PyAny>,
    xdata_tuple: &Bound<'py, PyTuple>,
    ydata: PyReadonlyArray1<f64>,
    p0: PyReadonlyArray1<f64>,
    maxfev: Option<usize>,
    xtol: Option<f64>,
    ftol: Option<f64>,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray2<f64>>)> {
    // Extract arrays from tuple
    let mut xdata_arrays: Vec<Array1<f64>> = Vec::new();
    for item in xdata_tuple.iter() {
        // Try to extract as f64 array first
        if let Ok(array) = item.extract::<PyReadonlyArray1<f64>>() {
            xdata_arrays.push(array.as_array().to_owned());
        } else if let Ok(int_array) = item.extract::<PyReadonlyArray1<i64>>() {
            // Handle integer arrays by converting to f64
            #[allow(clippy::cast_precision_loss)]
            // Accepting precision loss for large integers in scientific data
            let float_array: Array1<f64> = int_array.as_array().mapv(|x| x as f64);
            xdata_arrays.push(float_array);
        } else if let Ok(int_array) = item.extract::<PyReadonlyArray1<i32>>() {
            // Handle i32 arrays
            let float_array: Array1<f64> = int_array.as_array().mapv(f64::from);
            xdata_arrays.push(float_array);
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Each element in xdata tuple must be a numeric array (int or float)",
            ));
        }
    }

    if xdata_arrays.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "xdata tuple must contain at least one array",
        ));
    }

    // Verify all arrays have the same length
    let n = xdata_arrays[0].len();
    for (i, arr) in xdata_arrays.iter().enumerate().skip(1) {
        if arr.len() != n {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "All xdata arrays must have the same length. Array 0 has length {}, array {} has length {}",
                n,
                i,
                arr.len()
            )));
        }
    }

    let ydata_view = ydata.as_array();
    if ydata_view.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "xdata and ydata must have the same length: xdata has {}, ydata has {}",
            n,
            ydata_view.len()
        )));
    }

    // Create a "virtual" xdata that's just indices, and modify the function wrapper
    // to look up the actual values from the tuple of arrays
    #[allow(clippy::cast_precision_loss)] // Array indices are always small enough for f64
    let xdata_indices: Array1<f64> = Array1::from_iter((0..n).map(|i| i as f64));

    // Clone the arrays for use in closure
    let xdata_arrays_clone = xdata_arrays.clone();

    // Create closure that calls Python function with tuple of x values
    // The Python function signature is f((x1, x2, ...), *params)
    let func = move |idx: f64, params: &[f64]| -> f64 {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let i = idx as usize; // idx is always a valid non-negative array index

        Python::attach(|py| {
            // Build tuple of x values at index i
            let x_values: Vec<f64> = xdata_arrays_clone.iter().map(|arr| arr[i]).collect();

            // Create Python tuple for x values
            let Ok(x_tuple) = PyTuple::new(py, &x_values) else {
                return f64::NAN;
            };

            // Build complete arguments: First create a Vec of all arguments
            // Then convert to PyTuple
            // Arguments are: (x_tuple, *params)

            // Create Python list to build arguments
            let Ok(list_module) = py.import("builtins") else {
                return f64::NAN;
            };

            let py_list = match list_module.getattr("list") {
                Ok(list_func) => match list_func.call0() {
                    Ok(l) => l,
                    Err(_) => return f64::NAN,
                },
                Err(_) => return f64::NAN,
            };

            // Append x_tuple as first element
            if py_list.call_method1("append", (x_tuple,)).is_err() {
                return f64::NAN;
            }

            // Append each param
            for &param in params {
                if py_list.call_method1("append", (param,)).is_err() {
                    return f64::NAN;
                }
            }

            // Convert list to tuple
            let Ok(tuple_func) = list_module.getattr("tuple") else {
                return f64::NAN;
            };

            let Ok(args_tuple) = tuple_func.call1((py_list,)) else {
                return f64::NAN;
            };

            // Downcast to PyTuple
            let Ok(args_as_tuple) = args_tuple.cast() else {
                return f64::NAN;
            };

            // Call function with arguments
            match f.call1(py, args_as_tuple) {
                Ok(result) => result.extract::<f64>(py).unwrap_or(f64::NAN),
                Err(e) => {
                    let () = e.print(py);
                    f64::NAN
                }
            }
        })
    };

    // Configure options
    let opts = CurveFitOptions {
        maxfev: maxfev.unwrap_or(1000),
        xtol: xtol.unwrap_or(1e-8),
        ftol: ftol.unwrap_or(1e-8),
        lambda: 0.01,
    };

    let p0_view = p0.as_array();

    // Call Rust implementation with index-based xdata
    let result =
        pecos::prelude::curve_fit(func, xdata_indices.view(), ydata_view, p0_view, Some(opts))
            .map_err(map_curve_fit_error)?;

    // Convert results to Python arrays
    let popt = PyArray1::from_array(py, &result.params).unbind();

    // If covariance is available, return it; otherwise create identity matrix
    let pcov = if let Some(cov) = result.pcov {
        PyArray2::from_array(py, &cov).unbind()
    } else {
        // Return identity matrix if covariance not available
        let n = result.params.len();
        let mut cov_array = vec![vec![0.0; n]; n];
        for (i, row) in cov_array.iter_mut().enumerate().take(n) {
            row[i] = 1.0;
        }
        PyArray2::from_vec2(py, &cov_array).unwrap().unbind()
    };

    Ok((popt, pcov))
}

// ============================================================================
// Random Number Generation - NumPy drop-in replacements
// ============================================================================

/// Generate random floats from a uniform distribution over [0.0, 1.0).
///
/// This is a drop-in replacement for `numpy.random.random(size)`.
///
/// Args:
///     size: int - Number of random values to generate
///
/// Returns:
///     ndarray: Array of random floats in [0.0, 1.0)
///
/// Examples:
///     >>> from `pecos_rslib.num.random` import random
///     >>> values = random(5)
///     >>> len(values)
///     5
#[pyfunction]
fn random(py: Python<'_>, size: usize) -> Py<PyArray1<f64>> {
    let result = pecos::prelude::random::random(size);
    PyArray1::from_array(py, &result).unbind()
}

/// Generate random integers from a uniform distribution.
///
/// This is a drop-in replacement for `numpy.random.randint(low, high, size)`.
///
/// Args:
///     low: int - Lowest integer to be drawn (or upper bound if high is None)
///     high: Optional[int] - If provided, one above the largest integer to be drawn
///     size: Optional[int] - Number of random integers to generate. If None, returns a single integer.
///
/// Returns:
///     int | ndarray: Single integer or array of random integers
///
/// Examples:
///     >>> from `pecos_rslib.num.random` import randint
///     >>> # Single random integer in [0, 10)
///     >>> val = randint(10)
///     >>> 0 <= val < 10
///     True
///     >>> # Array of random integers in [5, 15)
///     >>> vals = randint(5, 15, 100)
///     >>> len(vals)
///     100
#[pyfunction]
#[pyo3(signature = (low, high=None, size=None))]
#[allow(clippy::needless_pass_by_value)] // Python object requires ownership
fn randint(
    py: Python<'_>,
    low: i64,
    high: Option<i64>,
    size: Option<usize>,
) -> PyResult<Py<PyAny>> {
    use pyo3::IntoPyObject;

    if let Some(n) = size {
        // Return array
        let result = pecos::prelude::random::randint(low, high, n);
        Ok(PyArray1::from_array(py, &result).into())
    } else {
        // Return scalar
        let result = pecos::prelude::random::randint_scalar(low, high);
        Ok(result.into_pyobject(py)?.into_any().unbind())
    }
}

/// Set the random seed for reproducible results.
///
/// This is a drop-in replacement for `numpy.random.seed(seed)`.
///
/// Sets a thread-local seed for all subsequent random number generation.
/// This ensures reproducibility for scientific computing and testing.
///
/// Args:
///     `seed_value`: int - The seed value (will be cast to u64)
///
/// Examples:
///     >>> from `pecos_rslib.num.random` import seed, random
///     >>> seed(42)
///     >>> values1 = random(5)
///     >>> seed(42)
///     >>> values2 = random(5)
///     >>> # values1 and values2 are identical
///     >>> import numpy as np
///     >>> `np.array_equal(values1`, values2)
///     True
#[pyfunction]
fn seed(seed_value: u64) {
    pecos::prelude::random::seed(seed_value);
}

/// Generate a random sample from a given array.
///
/// This is a drop-in replacement for `numpy.random.choice(a, size, replace=True)`.
///
/// Args:
///     a: list | ndarray - Array to sample from
///     size: Optional[int] - Number of samples to draw. If None, returns a single sample.
///     replace: bool - Whether to sample with replacement (default: True)
///
/// Returns:
///     Any | list: Single sample or list of samples
///
/// Examples:
///     >>> from pecos_rslib.num.random import choice
///     >>> items = ["X", "Y", "Z"]  # Quotes are Python syntax, not Rust links
///     >>> # Single sample
///     >>> sample = choice(items)
///     >>> sample in items
///     True
///     >>> # Multiple samples with replacement
///     >>> samples = choice(items, 5, True)
///     >>> len(samples)
///     5
///
/// Note: This is Python example code, not Rust documentation links
#[allow(clippy::doc_link_with_quotes, clippy::doc_markdown)]
#[pyfunction]
#[pyo3(signature = (a, size=None, replace=true))]
#[allow(clippy::needless_pass_by_value)] // Py<PyAny> is a cheap ref-counted pointer
fn choice(py: Python<'_>, a: Py<PyAny>, size: Option<usize>, replace: bool) -> PyResult<Py<PyAny>> {
    // Convert Python array/list to Vec<Py<PyAny>>
    let array = Python::attach(|py| {
        let obj = a.bind(py);

        // First try to handle numpy arrays by converting to list
        if let Ok(to_list_method) = obj.getattr("tolist")
            && let Ok(list_obj) = to_list_method.call0()
        {
            let seq = list_obj.cast::<pyo3::types::PySequence>()?;
            let len = seq.len()?;
            let mut items = Vec::with_capacity(len);
            for i in 0..len {
                items.push(seq.get_item(i)?.unbind());
            }
            return Ok::<Vec<Py<PyAny>>, PyErr>(items);
        }

        // Fall back to treating as sequence
        let seq = obj.cast::<pyo3::types::PySequence>()?;
        let len = seq.len()?;

        let mut items = Vec::with_capacity(len);
        for i in 0..len {
            items.push(seq.get_item(i)?.unbind());
        }

        Ok::<Vec<Py<PyAny>>, PyErr>(items)
    })?;

    if array.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Cannot sample from empty array",
        ));
    }

    // Validate size for sampling without replacement
    if let Some(n) = size
        && !replace
        && n > array.len()
    {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Cannot take larger sample ({}) than population ({}) when replace=False",
            n,
            array.len()
        )));
    }

    // Optimize by sampling indices instead of cloning Python objects
    // This avoids expensive Python::attach() and clone_ref() calls
    let indices: Vec<usize> = (0..array.len()).collect();

    if let Some(n) = size {
        // Sample indices instead of objects
        let sampled_indices = pecos::prelude::random::choice(&indices, n, replace);

        // Build result list by indexing array once per sample
        let py_list = pyo3::types::PyList::empty(py);
        for &idx in &sampled_indices {
            py_list.append(&array[idx])?;
        }
        Ok(py_list.into())
    } else {
        // Return single sample
        let idx = pecos::prelude::random::choice_scalar(&indices);
        Ok(array[idx].clone_ref(py))
    }
}

/// Fused operation: Check if any random value is less than threshold.
///
/// This is a high-performance fused version of `np.any(np.random.random(size) < threshold)`.
///
/// # Arguments
///
/// * `size` - Number of random values to potentially generate
/// * `threshold` - Threshold to compare against
///
/// # Returns
///
/// Returns `True` if any generated random value is less than `threshold`, `False` otherwise.
///
/// # Performance
///
/// Expected 2-3x speedup over numpy due to:
/// - No array allocation
/// - Short-circuit evaluation
/// - Reduced Python overhead
///
/// # Examples
///
/// ```python
/// from pecos_rslib.num import random
///
/// # Seed for reproducibility
/// random.seed(42)
///
/// # Check if any of 100 qubits have errors (1% error rate)
/// has_error = random.compare_any(100, 0.01)
/// ```
#[pyfunction]
fn compare_any(size: usize, threshold: f64) -> bool {
    pecos::prelude::random::compare_any(size, threshold)
}

/// Fused operation: Get indices where random values are less than threshold.
///
/// This is a high-performance fused version of:
/// ```python
/// rand_nums = np.random.random(size) < threshold
/// indices = [i for i, r in enumerate(rand_nums) if r]
/// ```
///
/// # Arguments
///
/// * `size` - Number of random values to generate
/// * `threshold` - Threshold to compare against
///
/// # Returns
///
/// Returns a list of indices where the random value was less than `threshold`.
///
/// # Performance
///
/// Expected 1.5-2x speedup over numpy due to:
/// - No intermediate boolean array allocation
/// - Direct collection of matching indices
/// - Reduced Python overhead
///
/// # Examples
///
/// ```python
/// from pecos_rslib.num import random
///
/// # Seed for reproducibility
/// random.seed(42)
///
/// # Get indices of qubits with errors (1% error rate)
/// error_indices = random.compare_indices(100, 0.01)
/// for idx in error_indices:
///     apply_error(qubits[idx])
/// ```
#[pyfunction]
fn compare_indices(py: Python<'_>, size: usize, threshold: f64) -> PyResult<Py<PyAny>> {
    let indices = pecos::prelude::random::compare_indices(size, threshold);

    // Convert Vec<usize> to Python list
    let py_list = pyo3::types::PyList::empty(py);
    for idx in indices {
        py_list.append(idx)?;
    }
    Ok(py_list.into())
}

/// Calculate the arithmetic mean of a sequence of values.
///
/// Drop-in replacement for `numpy.mean()` for 1D arrays without axis parameter.
///
/// # Arguments
///
/// * `values` - A Python list or sequence of numeric values
///
/// # Returns
///
/// The arithmetic mean as f64, or `NaN` if the sequence is empty
///
/// # Examples
///
/// ```python
/// from pecos_rslib.num import mean
///
/// # Calculate mean of a list
/// values = [1.0, 2.0, 3.0, 4.0, 5.0]
/// avg = mean(values)  # Returns 3.0
///
/// # Error model use case: average measurement error rates
/// p_meas = (0.01, 0.015, 0.02)
/// avg_p_meas = mean(p_meas)  # Returns 0.015
/// ```
// PyO3 requires Vec<T> for automatic extraction from Python sequences.
// The Vec is consumed but this is unavoidable with current PyO3 API.
#[allow(clippy::needless_pass_by_value)]
#[pyfunction]
fn mean(values: Vec<f64>) -> f64 {
    pecos::prelude::mean(&values)
}

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
/// ```python
/// from pecos_rslib.num import power
///
/// # Basic integer power
/// result = power(2.0, 3.0)  # Returns 8.0
///
/// # Fractional power (square root)
/// result = power(4.0, 0.5)  # Returns 2.0
///
/// # Threshold curve use case
/// dist = 5.0
/// v0 = 2.0
/// result = power(dist, 1.0 / v0)
/// ```
#[pyfunction]
fn power(base: f64, exponent: f64) -> f64 {
    pecos::prelude::power(base, exponent)
}

/// Calculate the standard deviation of values.
///
/// Drop-in replacement for `numpy.std()` for 1D arrays without axis parameter.
///
/// # Arguments
///
/// * `values` - A Python list or sequence of numeric values
/// * `ddof` - Delta degrees of freedom (0 for population std, 1 for sample std)
///
/// # Returns
///
/// The standard deviation as f64, or `NaN` if the sequence is empty or if n <= ddof
///
/// # Examples
///
/// ```python
/// from pecos_rslib.num import std
///
/// # Calculate population standard deviation
/// values = [1.0, 2.0, 3.0, 4.0, 5.0]
/// population_std = std(values, 0)  # Returns ~1.414
///
/// # Calculate sample standard deviation
/// sample_std = std(values, 1)  # Returns ~1.581
///
/// # Jackknife analysis use case
/// parameter_estimates = [1.5, 1.6, 1.4, 1.5, 1.7]
/// uncertainty = std(parameter_estimates, 0)
/// ```
// PyO3 requires Vec<T> for automatic extraction from Python sequences.
// The Vec is consumed but this is unavoidable with current PyO3 API.
#[allow(clippy::needless_pass_by_value)]
#[pyfunction]
fn std(values: Vec<f64>, ddof: usize) -> f64 {
    pecos::prelude::std(&values, ddof)
}

/// Register the num submodule with Python bindings.
pub fn register_num_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let num_module = PyModule::new(m.py(), "num")?;

    // Add optimization functions
    num_module.add_function(wrap_pyfunction!(brentq, &num_module)?)?;
    num_module.add_function(wrap_pyfunction!(newton, &num_module)?)?;
    num_module.add_function(wrap_pyfunction!(polyfit, &num_module)?)?;
    num_module.add_function(wrap_pyfunction!(curve_fit, &num_module)?)?;
    num_module.add_class::<Poly1d>()?;

    // Add statistical functions
    num_module.add_function(wrap_pyfunction!(mean, &num_module)?)?;
    num_module.add_function(wrap_pyfunction!(power, &num_module)?)?;
    num_module.add_function(wrap_pyfunction!(self::std, &num_module)?)?;

    // Create random submodule
    let random_module = PyModule::new(m.py(), "random")?;
    random_module.add_function(wrap_pyfunction!(seed, &random_module)?)?;
    random_module.add_function(wrap_pyfunction!(random, &random_module)?)?;
    random_module.add_function(wrap_pyfunction!(randint, &random_module)?)?;
    random_module.add_function(wrap_pyfunction!(choice, &random_module)?)?;
    random_module.add_function(wrap_pyfunction!(compare_any, &random_module)?)?;
    random_module.add_function(wrap_pyfunction!(compare_indices, &random_module)?)?;
    num_module.add_submodule(&random_module)?;

    m.add_submodule(&num_module)?;
    Ok(())
}
