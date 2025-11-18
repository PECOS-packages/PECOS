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

// Allow Clippy pedantic lints that are not applicable to PyO3 bindings
#![allow(clippy::similar_names)] // Similar parameter names are intentional (e.g., start/stop/step)
#![allow(clippy::too_many_lines)] // Large module with many function bindings
#![allow(clippy::needless_pass_by_value)] // PyO3 requires passing Bound by value
#![allow(clippy::unnecessary_wraps)] // PyResult is required for Python error handling
#![allow(clippy::cast_possible_truncation)] // Intentional truncation for dtype conversions
#![allow(clippy::cast_possible_wrap)] // Intentional wrap for Python-style indexing
#![allow(clippy::cast_sign_loss)] // Intentional sign loss for Python-style indexing
#![allow(clippy::cast_precision_loss)] // Expected precision loss in numeric conversions

use num_complex::Complex64;
use numpy::ndarray::{Array as NdArray, Array1, ArrayD, Axis, IxDyn};
use numpy::{
    IntoPyArray, PyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArrayDyn,
};
use pyo3::conversion::IntoPyObjectExt;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

// Import Array and ArrayData from pecos_array module for migration from numpy.ndarray to Array
use crate::pecos_array::{Array, ArrayData};

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
#[pyo3(signature = (x, y, deg, cov=None))]
#[allow(clippy::needless_pass_by_value)] // PyReadonlyArray1 is a lightweight wrapper
fn polyfit(
    py: Python<'_>,
    x: PyReadonlyArray1<f64>,
    y: PyReadonlyArray1<f64>,
    deg: usize,
    cov: Option<bool>,
) -> PyResult<Py<PyAny>> {
    let x_view = x.as_array();
    let y_view = y.as_array();

    let return_cov = cov.unwrap_or(false);

    if return_cov {
        // Call polyfit_with_cov and return tuple (coeffs, cov_matrix)
        let (coeffs, cov_matrix) =
            pecos::prelude::polyfit_with_cov(x_view, y_view, deg).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("polyfit failed: {e}"))
            })?;

        let coeffs_py = PyArray1::from_array(py, &coeffs).unbind();
        let cov_py = PyArray2::from_array(py, &cov_matrix).unbind();

        let tuple_items: Vec<Py<PyAny>> = vec![coeffs_py.into(), cov_py.into()];
        Ok(PyTuple::new(py, &tuple_items)?.into())
    } else {
        // Call regular polyfit and return just coefficients
        let coeffs = pecos::prelude::polyfit(x_view, y_view, deg).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("polyfit failed: {e}"))
        })?;

        Ok(PyArray1::from_array(py, &coeffs).unbind().into())
    }
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

/// Check if a value is NaN (Not a Number).
///
/// Drop-in replacement for `numpy.isnan()` for scalar values.
///
/// Args:
///     x (float): Input value to check
///
/// Returns:
///     bool: True if x is NaN, False otherwise
///
/// Examples:
///     >>> from `pecos_rslib`._`pecos_rslib` import num
///     >>> num.isnan(float('nan'))
///     True
///     >>> num.isnan(0.0)
///     False
///     >>> num.isnan(1.0)
///     False
///     >>> num.isnan(float('inf'))
///     False
///
/// # Example: Error checking (curve fitting validation)
/// ```python
/// result = 0.0 / 0.0  # NaN
/// if num.isnan(result):
///     print("Invalid computation")
/// ```
#[pyfunction]
#[allow(clippy::needless_pass_by_value)] // Bound is designed to be passed by value (PyO3 convention)
fn isnan(py: Python<'_>, x: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    use pecos::prelude::IsNan;

    // Try scalar float
    if let Ok(val) = x.extract::<f64>() {
        let result = val.isnan();
        return Ok(result.into_py_any(py).unwrap());
    }

    // Try complex scalar
    if let Ok(val) = x.extract::<Complex64>() {
        let result = val.isnan();
        return Ok(result.into_py_any(py).unwrap());
    }

    // Try float array
    if let Ok(arr) = x.cast::<PyArray<f64, IxDyn>>() {
        let result = arr.try_readonly()?.as_array().isnan();
        return Ok(PyArray::from_owned_array(py, result).into_any().unbind());
    }

    // Try complex array
    if let Ok(arr) = x.cast::<PyArray<Complex64, IxDyn>>() {
        let result = arr.try_readonly()?.as_array().isnan();
        return Ok(PyArray::from_owned_array(py, result).into_any().unbind());
    }

    Err(PyTypeError::new_err(
        "isnan() argument must be float, complex, or numpy array of float/complex",
    ))
}

/// Return the floor of x as a float.
///
/// Drop-in replacement for `numpy.floor()` for scalar values.
///
/// # Arguments
///
/// * `x` - Input value
///
/// # Returns
///
/// The largest integer value less than or equal to x, as f64
///
/// # Examples
///
/// ```python
/// from pecos_rslib.num import floor
///
/// # Basic usage
/// floor(3.7)   # Returns 3.0
/// floor(-3.7)  # Returns -4.0
///
/// # Fault tolerance threshold calculation
/// d = 5
/// t = floor((d - 1) / 2)  # Returns 2.0
/// ```
#[pyfunction]
fn floor(x: f64) -> f64 {
    pecos::prelude::floor(x)
}

/// Return the ceiling of x as a float.
///
/// Drop-in replacement for `numpy.ceil()` for scalar values.
///
/// # Arguments
///
/// * `x` - Input value
///
/// # Returns
///
/// The smallest integer value greater than or equal to x, as f64
///
/// # Examples
///
/// ```python
/// from pecos_rslib.num import ceil
///
/// # Basic usage
/// ceil(3.2)   # Returns 4.0
/// ceil(-3.2)  # Returns -3.0
/// ```
#[pyfunction]
fn ceil(x: f64) -> f64 {
    pecos::prelude::ceil(x)
}

/// Round a number to the nearest integer as a float.
///
/// Drop-in replacement for `numpy.round()` for scalar values (with default decimals=0).
/// Uses "round half to even" (banker's rounding) to match numpy behavior exactly.
///
/// # Arguments
///
/// * `x` - Input value
///
/// # Returns
///
/// The rounded value, as f64
///
/// # Examples
///
/// ```python
/// from pecos_rslib.num import round
///
/// # Basic usage
/// round(3.7)   # Returns 4.0
/// round(3.2)   # Returns 3.0
///
/// # Round half to even (banker's rounding)
/// round(2.5)   # Returns 2.0 (even)
/// round(3.5)   # Returns 4.0 (even)
/// ```
#[pyfunction]
fn round(x: f64) -> f64 {
    // Use stdlib .round_ties_even() for NumPy-compatible "round half to even" behavior
    x.round_ties_even()
}

/// Returns True if two values are element-wise equal within a tolerance.
///
/// Drop-in replacement for `numpy.isclose()` for scalar values.
///
/// # Arguments
///
/// * `a` - First input value
/// * `b` - Second input value
/// * `rtol` - Relative tolerance parameter (default: 1e-5)
/// * `atol` - Absolute tolerance parameter (default: 1e-8)
///
/// # Returns
///
/// True if the values are close within the specified tolerances, False otherwise
///
/// # Examples
///
/// ```python
/// from pecos_rslib.num import isclose
///
/// # Basic usage with defaults
/// isclose(1.0, 1.0)                           # Returns True (uses default tolerances)
/// isclose(1.0, 1.00001)                       # Returns True (within default tolerance)
/// isclose(1.0, 1.1)                           # Returns False
///
/// # Custom tolerances
/// isclose(1.0, 1.00001, rtol=1e-4, atol=1e-8) # Returns True
/// isclose(1.0, 1.1, rtol=1e-5, atol=1e-8)     # Returns False
///
/// # Quantum gate angle comparison (tight tolerance)
/// import math
/// theta = math.pi / 2.0
/// isclose(theta, math.pi / 2.0, rtol=0.0, atol=1e-12)  # Returns True
/// ```
#[pyfunction]
#[pyo3(signature = (a, b, rtol=1e-5, atol=1e-8))]
#[allow(clippy::needless_pass_by_value)] // Bound is designed to be passed by value (PyO3 convention)
fn isclose(
    py: Python<'_>,
    a: Bound<'_, PyAny>,
    b: Bound<'_, PyAny>,
    rtol: f64,
    atol: f64,
) -> PyResult<Py<PyAny>> {
    use pecos::prelude::IsClose;

    // Try scalar floats
    if let (Ok(a_val), Ok(b_val)) = (a.extract::<f64>(), b.extract::<f64>()) {
        let result = a_val.isclose(&b_val, rtol, atol);
        return Ok(result.into_py_any(py).unwrap());
    }

    // Try complex scalars (both complex)
    if let (Ok(a_val), Ok(b_val)) = (a.extract::<Complex64>(), b.extract::<Complex64>()) {
        let result = a_val.isclose(&b_val, rtol, atol);
        return Ok(result.into_py_any(py).unwrap());
    }

    // Handle mixed complex/float scalars - promote float to complex
    if let (Ok(a_val), Ok(b_val)) = (a.extract::<Complex64>(), b.extract::<f64>()) {
        let b_complex = Complex64::new(b_val, 0.0);
        let result = a_val.isclose(&b_complex, rtol, atol);
        return Ok(result.into_py_any(py).unwrap());
    }
    if let (Ok(a_val), Ok(b_val)) = (a.extract::<f64>(), b.extract::<Complex64>()) {
        let a_complex = Complex64::new(a_val, 0.0);
        let result = a_complex.isclose(&b_val, rtol, atol);
        return Ok(result.into_py_any(py).unwrap());
    }

    // Try float arrays
    if let (Ok(a_arr), Ok(b_arr)) = (
        a.cast::<PyArray<f64, IxDyn>>(),
        b.cast::<PyArray<f64, IxDyn>>(),
    ) {
        let a_readonly = a_arr.try_readonly()?;
        let b_readonly = b_arr.try_readonly()?;
        let result = a_readonly
            .as_array()
            .isclose(&b_readonly.as_array(), rtol, atol);
        return Ok(PyArray::from_owned_array(py, result).into_any().unbind());
    }

    // Try complex arrays
    if let (Ok(a_arr), Ok(b_arr)) = (
        a.cast::<PyArray<Complex64, IxDyn>>(),
        b.cast::<PyArray<Complex64, IxDyn>>(),
    ) {
        let a_readonly = a_arr.try_readonly()?;
        let b_readonly = b_arr.try_readonly()?;
        let result = a_readonly
            .as_array()
            .isclose(&b_readonly.as_array(), rtol, atol);
        return Ok(PyArray::from_owned_array(py, result).into_any().unbind());
    }

    // Handle mixed array types: complex array vs float array
    if let (Ok(a_arr), Ok(b_arr)) = (
        a.cast::<PyArray<Complex64, IxDyn>>(),
        b.cast::<PyArray<f64, IxDyn>>(),
    ) {
        let a_readonly = a_arr.try_readonly()?;
        let b_readonly = b_arr.try_readonly()?;

        // Convert float array to complex
        let b_complex = b_readonly.as_array().mapv(|x| Complex64::new(x, 0.0));
        let result = a_readonly.as_array().isclose(&b_complex.view(), rtol, atol);
        return Ok(PyArray::from_owned_array(py, result).into_any().unbind());
    }

    // Handle mixed array types: float array vs complex array
    if let (Ok(a_arr), Ok(b_arr)) = (
        a.cast::<PyArray<f64, IxDyn>>(),
        b.cast::<PyArray<Complex64, IxDyn>>(),
    ) {
        let a_readonly = a_arr.try_readonly()?;
        let b_readonly = b_arr.try_readonly()?;

        // Convert float array to complex
        let a_complex = a_readonly.as_array().mapv(|x| Complex64::new(x, 0.0));
        let result = a_complex.view().isclose(&b_readonly.as_array(), rtol, atol);
        return Ok(PyArray::from_owned_array(py, result).into_any().unbind());
    }

    Err(PyTypeError::new_err(
        "isclose() arguments must be float, complex, or numpy arrays of float/complex",
    ))
}

/// Check if all elements in two arrays are close within specified tolerances.
///
/// Drop-in replacement for `numpy.allclose()`. Returns `True` if all pairs
/// of elements are close according to the tolerance check:
/// `|a - b| <= (atol + rtol * |b|)`
///
/// # Arguments
///
/// * `a` - First array
/// * `b` - Second array
/// * `rtol` - Relative tolerance (default: 1e-5)
/// * `atol` - Absolute tolerance (default: 1e-8)
/// * `equal_nan` - If true, NaNs in the same position are considered equal (default: false)
///
/// # Returns
///
/// Returns `True` if all elements are close, `False` otherwise.
///
/// # Examples
///
/// ```python
/// import numpy as np
/// from pecos_rslib import allclose
///
/// # 1D Arrays
/// a = np.array([1.0, 2.0, 3.0])
/// b = np.array([1.00001, 2.00001, 3.00001])
/// allclose(a, b, rtol=1e-4, atol=1e-8)  # Returns True
///
/// # 2D Arrays (quantum gate matrices)
/// gate1 = np.array([[1.0, 0.0], [0.0, 1.0]])
/// gate2 = np.array([[1.00001, 0.0], [0.0, 0.99999]])
/// allclose(gate1, gate2, rtol=1e-4, atol=1e-8)  # Returns True
///
/// # With NaN handling
/// a = np.array([1.0, np.nan, 3.0])
/// b = np.array([1.0, np.nan, 3.0])
/// allclose(a, b, equal_nan=True)  # Returns True
/// ```
#[pyfunction]
#[pyo3(signature = (a, b, rtol=1e-5, atol=1e-8, equal_nan=false))]
#[allow(clippy::needless_pass_by_value)] // Bound is designed to be passed by value (PyO3 convention)
fn allclose(
    a: Bound<'_, PyAny>,
    b: Bound<'_, PyAny>,
    rtol: f64,
    atol: f64,
    equal_nan: bool,
) -> PyResult<bool> {
    use pecos::prelude::allclose as rust_allclose;

    // Try float arrays
    if let (Ok(a_arr), Ok(b_arr)) = (
        a.cast::<PyArray<f64, IxDyn>>(),
        b.cast::<PyArray<f64, IxDyn>>(),
    ) {
        let a_readonly = a_arr.try_readonly()?;
        let b_readonly = b_arr.try_readonly()?;
        return Ok(rust_allclose(
            &a_readonly.as_array(),
            &b_readonly.as_array(),
            rtol,
            atol,
            equal_nan,
        ));
    }

    // Try complex arrays
    if let (Ok(a_arr), Ok(b_arr)) = (
        a.cast::<PyArray<Complex64, IxDyn>>(),
        b.cast::<PyArray<Complex64, IxDyn>>(),
    ) {
        let a_readonly = a_arr.try_readonly()?;
        let b_readonly = b_arr.try_readonly()?;
        return Ok(rust_allclose(
            &a_readonly.as_array(),
            &b_readonly.as_array(),
            rtol,
            atol,
            equal_nan,
        ));
    }

    // Handle mixed array types: complex array vs float array
    if let (Ok(a_arr), Ok(b_arr)) = (
        a.cast::<PyArray<Complex64, IxDyn>>(),
        b.cast::<PyArray<f64, IxDyn>>(),
    ) {
        let a_readonly = a_arr.try_readonly()?;
        let b_readonly = b_arr.try_readonly()?;

        // Convert float array to complex
        let b_complex = b_readonly.as_array().mapv(|x| Complex64::new(x, 0.0));
        return Ok(rust_allclose(
            &a_readonly.as_array(),
            &b_complex.view(),
            rtol,
            atol,
            equal_nan,
        ));
    }

    // Handle mixed array types: float array vs complex array
    if let (Ok(a_arr), Ok(b_arr)) = (
        a.cast::<PyArray<f64, IxDyn>>(),
        b.cast::<PyArray<Complex64, IxDyn>>(),
    ) {
        let a_readonly = a_arr.try_readonly()?;
        let b_readonly = b_arr.try_readonly()?;

        // Convert float array to complex
        let a_complex = a_readonly.as_array().mapv(|x| Complex64::new(x, 0.0));
        return Ok(rust_allclose(
            &a_complex.view(),
            &b_readonly.as_array(),
            rtol,
            atol,
            equal_nan,
        ));
    }

    // Try Python list/tuple conversions for both arguments - convert directly to ndarray

    // Try both as float lists
    if let (Ok(a_values), Ok(b_values)) = (a.extract::<Vec<f64>>(), b.extract::<Vec<f64>>()) {
        let a_arr = ArrayD::from_shape_vec(IxDyn(&[a_values.len()]), a_values).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Failed to create array: {e}"))
        })?;
        let b_arr = ArrayD::from_shape_vec(IxDyn(&[b_values.len()]), b_values).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Failed to create array: {e}"))
        })?;
        return Ok(rust_allclose(
            &a_arr.view(),
            &b_arr.view(),
            rtol,
            atol,
            equal_nan,
        ));
    }

    // Try both as complex lists
    if let (Ok(a_values), Ok(b_values)) =
        (a.extract::<Vec<Complex64>>(), b.extract::<Vec<Complex64>>())
    {
        let a_arr = ArrayD::from_shape_vec(IxDyn(&[a_values.len()]), a_values).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Failed to create array: {e}"))
        })?;
        let b_arr = ArrayD::from_shape_vec(IxDyn(&[b_values.len()]), b_values).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Failed to create array: {e}"))
        })?;
        return Ok(rust_allclose(
            &a_arr.view(),
            &b_arr.view(),
            rtol,
            atol,
            equal_nan,
        ));
    }

    // Try mixed: float array/list vs complex array/list
    // Extract a as float (array or list)
    let a_float_opt = a
        .extract::<PyReadonlyArrayDyn<f64>>()
        .ok()
        .map(|arr| arr.as_array().to_owned())
        .or_else(|| {
            a.extract::<Vec<f64>>()
                .ok()
                .map(|values| ArrayD::from_shape_vec(IxDyn(&[values.len()]), values).unwrap())
        });

    // Extract b as complex (array or list)
    let b_complex_opt = b
        .extract::<PyReadonlyArrayDyn<Complex64>>()
        .ok()
        .map(|arr| arr.as_array().to_owned())
        .or_else(|| {
            b.extract::<Vec<Complex64>>()
                .ok()
                .map(|values| ArrayD::from_shape_vec(IxDyn(&[values.len()]), values).unwrap())
        });

    if let (Some(a_float), Some(b_complex)) = (a_float_opt, b_complex_opt) {
        let a_complex = a_float.mapv(|x| Complex64::new(x, 0.0));
        return Ok(rust_allclose(
            &a_complex.view(),
            &b_complex.view(),
            rtol,
            atol,
            equal_nan,
        ));
    }

    // Try mixed: complex array/list vs float array/list
    let a_complex_opt = a
        .extract::<PyReadonlyArrayDyn<Complex64>>()
        .ok()
        .map(|arr| arr.as_array().to_owned())
        .or_else(|| {
            a.extract::<Vec<Complex64>>()
                .ok()
                .map(|values| ArrayD::from_shape_vec(IxDyn(&[values.len()]), values).unwrap())
        });

    let b_float_opt = b
        .extract::<PyReadonlyArrayDyn<f64>>()
        .ok()
        .map(|arr| arr.as_array().to_owned())
        .or_else(|| {
            b.extract::<Vec<f64>>()
                .ok()
                .map(|values| ArrayD::from_shape_vec(IxDyn(&[values.len()]), values).unwrap())
        });

    if let (Some(a_complex), Some(b_float)) = (a_complex_opt, b_float_opt) {
        let b_complex = b_float.mapv(|x| Complex64::new(x, 0.0));
        return Ok(rust_allclose(
            &a_complex.view(),
            &b_complex.view(),
            rtol,
            atol,
            equal_nan,
        ));
    }

    Err(PyTypeError::new_err(
        "allclose() arguments must be numeric arrays or lists",
    ))
}

/// Check if two arrays are equal element-wise.
///
/// Drop-in replacement for `numpy.array_equal(a1, a2, equal_nan=False)`.
///
/// Returns `True` if two arrays have the same shape and all elements are equal.
/// Unlike `allclose`, this function uses exact equality (`==`) rather than tolerance-based comparison.
///
/// # Arguments
///
/// * `a` - First input array
/// * `b` - Second input array
/// * `equal_nan` - If `true`, NaNs in the same position are considered equal (default: `false`)
///
/// # Returns
///
/// `true` if arrays are equal, `false` otherwise
///
/// # Examples
///
/// ```python
/// import numpy as np
/// from pecos_rslib.num import array_equal
///
/// # Equal arrays
/// a = np.array([1.0, 2.0, 3.0])
/// b = np.array([1.0, 2.0, 3.0])
/// assert array_equal(a, b)
///
/// # Different values
/// c = np.array([1.0, 2.0, 4.0])
/// assert not array_equal(a, c)
///
/// # NaN handling
/// d = np.array([1.0, np.nan, 3.0])
/// e = np.array([1.0, np.nan, 3.0])
/// assert not array_equal(d, e)  # NaN != NaN by default
/// assert array_equal(d, e, equal_nan=True)  # With equal_nan=True
/// ```
#[pyfunction]
#[pyo3(signature = (a, b, equal_nan=false))]
fn array_equal(a: Bound<'_, PyAny>, b: Bound<'_, PyAny>, equal_nan: bool) -> PyResult<bool> {
    use pecos::prelude::array_equal as rust_array_equal;

    // Try bool arrays (for isnan/isclose return values)
    if let (Ok(a_arr), Ok(b_arr)) = (
        a.cast::<PyArray<bool, IxDyn>>(),
        b.cast::<PyArray<bool, IxDyn>>(),
    ) {
        let a_readonly = a_arr.try_readonly()?;
        let b_readonly = b_arr.try_readonly()?;
        let a_view = a_readonly.as_array();
        let b_view = b_readonly.as_array();

        // For booleans, just check shape and exact equality
        if a_view.shape() != b_view.shape() {
            return Ok(false);
        }
        // Check if all elements are equal
        return Ok(a_view.iter().zip(b_view.iter()).all(|(a, b)| a == b));
    }

    // Try integer arrays (for randint return values)
    if let (Ok(a_arr), Ok(b_arr)) = (
        a.cast::<PyArray<i64, IxDyn>>(),
        b.cast::<PyArray<i64, IxDyn>>(),
    ) {
        let a_readonly = a_arr.try_readonly()?;
        let b_readonly = b_arr.try_readonly()?;
        let a_view = a_readonly.as_array();
        let b_view = b_readonly.as_array();

        // For integers, just check shape and exact equality
        if a_view.shape() != b_view.shape() {
            return Ok(false);
        }
        // Check if all elements are equal
        return Ok(a_view.iter().zip(b_view.iter()).all(|(a, b)| a == b));
    }

    // Try float arrays
    if let (Ok(a_arr), Ok(b_arr)) = (
        a.cast::<PyArray<f64, IxDyn>>(),
        b.cast::<PyArray<f64, IxDyn>>(),
    ) {
        let a_readonly = a_arr.try_readonly()?;
        let b_readonly = b_arr.try_readonly()?;
        return Ok(rust_array_equal(
            &a_readonly.as_array(),
            &b_readonly.as_array(),
            equal_nan,
        ));
    }

    // Try complex arrays
    if let (Ok(a_arr), Ok(b_arr)) = (
        a.cast::<PyArray<Complex64, IxDyn>>(),
        b.cast::<PyArray<Complex64, IxDyn>>(),
    ) {
        let a_readonly = a_arr.try_readonly()?;
        let b_readonly = b_arr.try_readonly()?;
        return Ok(rust_array_equal(
            &a_readonly.as_array(),
            &b_readonly.as_array(),
            equal_nan,
        ));
    }

    // Handle mixed array types: complex array vs float array
    if let (Ok(a_arr), Ok(b_arr)) = (
        a.cast::<PyArray<Complex64, IxDyn>>(),
        a.cast::<PyArray<f64, IxDyn>>(),
    ) {
        let a_readonly = a_arr.try_readonly()?;
        let b_readonly = b_arr.try_readonly()?;

        // Convert float array to complex
        let b_complex = b_readonly.as_array().mapv(|x| Complex64::new(x, 0.0));
        return Ok(rust_array_equal(
            &a_readonly.as_array(),
            &b_complex.view(),
            equal_nan,
        ));
    }

    // Handle mixed array types: float array vs complex array
    if let (Ok(a_arr), Ok(b_arr)) = (
        a.cast::<PyArray<f64, IxDyn>>(),
        b.cast::<PyArray<Complex64, IxDyn>>(),
    ) {
        let a_readonly = a_arr.try_readonly()?;
        let b_readonly = b_arr.try_readonly()?;

        // Convert float array to complex
        let a_complex = a_readonly.as_array().mapv(|x| Complex64::new(x, 0.0));
        return Ok(rust_array_equal(
            &a_complex.view(),
            &b_readonly.as_array(),
            equal_nan,
        ));
    }

    Err(PyTypeError::new_err(
        "array_equal() arguments must be numpy arrays of bool, int, float, or complex",
    ))
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

/// Extract the diagonal elements from a 2D array.
///
/// This is a drop-in replacement for `numpy.diag()` when extracting diagonal elements.
///
/// # Arguments
///
/// * `matrix` - A 2D array
///
/// # Returns
///
/// A 1D array containing the diagonal elements
///
/// # Examples
///
/// ```python
/// import numpy as np
/// from pecos_rslib.num import diag
///
/// # Extract diagonal from covariance matrix
/// cov_matrix = np.array([[0.0025, 0.0010], [0.0010, 0.0004]])
/// variances = diag(cov_matrix)
/// print(variances)  # [0.0025, 0.0004]
/// ```
#[pyfunction]
#[allow(clippy::needless_pass_by_value)] // PyReadonlyArray2 is a lightweight wrapper
fn diag(py: Python<'_>, matrix: PyReadonlyArray2<f64>) -> Py<PyArray1<f64>> {
    let matrix_view = matrix.as_array();
    let diagonal = pecos::prelude::diag(matrix_view);
    PyArray1::from_array(py, &diagonal).unbind()
}

/// Generate evenly spaced values over a specified interval.
///
/// This is a drop-in replacement for `numpy.linspace()`.
///
/// # Arguments
///
/// * `start` - The starting value of the sequence
/// * `stop` - The end value of the sequence
/// * `num` - Number of samples to generate. Default is 50.
/// * `endpoint` - If true, stop is the last sample. Otherwise, it is not included. Default is true.
///
/// # Returns
///
/// Array of evenly spaced samples
///
/// # Examples
///
/// ```python
/// from pecos_rslib.num import linspace
///
/// # Generate 1000 points for plotting
/// x = linspace(0.0, 1.0, 1000)
/// print(len(x))  # 1000
/// print(x[0])    # 0.0
/// print(x[-1])   # 1.0
/// ```
#[pyfunction]
#[pyo3(signature = (start, stop, num=50, endpoint=true))]
fn linspace(
    py: Python<'_>,
    start: f64,
    stop: f64,
    num: usize,
    endpoint: bool,
) -> PyResult<Py<Array>> {
    let result = pecos::prelude::linspace(start, stop, num, endpoint);
    Py::new(py, Array::from_array_f64(result.into_dyn()))
}

/// Return evenly spaced values within a given interval.
///
/// Drop-in replacement for `numpy.arange()` with automatic dtype inference.
///
/// Returns values in the half-open interval `[start, stop)` with the given step.
/// This function matches `NumPy`'s dtype inference behavior:
/// - If all arguments are Python integers (not bool), returns int64 array
/// - If any argument is a float, returns float64 array
///
/// # Arguments
///
/// * `start` - Start of interval (inclusive). Can be int or float.
/// * `stop` - End of interval (exclusive). Can be int or float. Optional - if omitted, start becomes stop and start is set to 0.
/// * `step` - Spacing between values (default: 1). Can be int or float.
///
/// # Returns
///
/// Array of evenly spaced values with dtype matching `NumPy`'s inference rules
///
/// # Examples
///
/// ```python
/// from pecos_rslib.num import arange
/// import numpy as np
///
/// # All integers → int64 array (matches NumPy)
/// x = arange(0, 10, 1)
/// print(x.dtype)  # int64
/// print(x)  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
///
/// # Any float → float64 array (matches NumPy)
/// x = arange(0.0, 10, 1)
/// print(x.dtype)  # float64
///
/// # Float step
/// x = arange(0, 1, 0.1)
/// print(x)  # [0., 0.1, 0.2, ..., 0.9]
///
/// # Negative step with integers → int64
/// x = arange(10, 0, -1)
/// print(x.dtype)  # int64
/// print(x)  # [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
///
/// # Single argument form
/// x = arange(5)  # equivalent to arange(0, 5, 1)
/// print(x)  # [0, 1, 2, 3, 4]
/// ```
#[pyfunction]
#[pyo3(signature = (start, stop=None, step=None))]
fn arange(
    py: Python<'_>,
    start: Bound<'_, PyAny>,
    stop: Option<Bound<'_, PyAny>>,
    step: Option<Bound<'_, PyAny>>,
) -> PyResult<Py<Array>> {
    // Handle single-argument case: arange(stop) → arange(0, stop, 1)
    let (start_param, stop_param, step_param) = if let Some(stop_val) = stop {
        (
            start,
            stop_val,
            step.unwrap_or_else(|| 1_i64.into_pyobject(py).unwrap().into_any()),
        )
    } else {
        // arange(n) case - start becomes stop, actual start is 0
        // Use Python int (not float) for defaults to preserve dtype inference
        (
            0_i64.into_pyobject(py)?.into_any(),
            start,
            step.unwrap_or_else(|| 1_i64.into_pyobject(py).unwrap().into_any()),
        )
    };

    // Check if each parameter is a Python integer (excluding bool)
    // This matches NumPy's dtype inference: all ints → int64, any float → float64
    let is_int = |obj: &Bound<'_, PyAny>| -> bool {
        // Check if it's an int but NOT a bool (in Python, bool is a subclass of int)
        obj.is_instance_of::<pyo3::types::PyInt>() && !obj.is_instance_of::<pyo3::types::PyBool>()
    };

    let all_ints = is_int(&start_param) && is_int(&stop_param) && is_int(&step_param);

    // Extract float values for computation
    let start_f64: f64 = start_param.extract()?;
    let stop_f64: f64 = stop_param.extract()?;
    let step_f64: f64 = step_param.extract()?;

    // Generate the range using Rust implementation
    let result_f64 = pecos::prelude::arange(start_f64, stop_f64, step_f64);

    // Return appropriate dtype based on inference
    if all_ints {
        // Convert to int64 array
        #[allow(clippy::cast_possible_truncation)] // Intentional truncation for int array
        let result_i64: Array1<i64> = result_f64.mapv(|x| x as i64);
        Py::new(py, Array::from_array_i64(result_i64.into_dyn()))
    } else {
        // Return as float64 array
        Py::new(py, Array::from_array_f64(result_f64.into_dyn()))
    }
}

/// Create a new array filled with zeros.
///
/// Drop-in replacement for `numpy.zeros()`.
///
/// # Arguments
///
/// * `shape` - Shape of the array as integer (1D) or tuple of integers (multi-D)
/// * `dtype` - Optional data type ('float64', 'complex128', 'int64'). Default is 'float64'.
///
/// # Returns
///
/// Array filled with zeros of the specified shape and dtype
///
/// # Examples
///
/// ```python
/// from pecos_rslib.num import zeros
///
/// # 1D array
/// arr = zeros(5)  # [0.0, 0.0, 0.0, 0.0, 0.0]
///
/// # 2D array
/// arr2d = zeros((2, 3))  # [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
///
/// # Integer dtype
/// arr_int = zeros(5, dtype='int64')  # [0, 0, 0, 0, 0]
///
/// # Complex dtype
/// arr_complex = zeros(3, dtype='complex128')  # [0+0j, 0+0j, 0+0j]
/// ```
#[pyfunction]
#[pyo3(signature = (shape, dtype=None))]
fn zeros(
    py: Python<'_>,
    shape: Bound<'_, PyAny>,
    dtype: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<Array>> {
    use crate::dtypes::DType;
    use num_complex::Complex64;

    // Parse shape - can be int or tuple
    let shape_vec: Vec<usize> = if let Ok(n) = shape.extract::<usize>() {
        vec![n]
    } else if let Ok(tuple) = shape.extract::<Vec<usize>>() {
        tuple
    } else {
        return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "shape must be an integer or tuple of integers",
        ));
    };

    // Convert dtype to string - accept both DType enum and string, default to "float64"
    let dtype_str = if let Some(dt) = dtype {
        // dtype was provided
        if let Ok(enum_dt) = dt.extract::<DType>() {
            enum_dt.to_numpy_str()
        } else if let Ok(s) = dt.extract::<&str>() {
            s
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "dtype must be a string or DType enum",
            ));
        }
    } else {
        // dtype not provided, use default
        "float64"
    };

    match dtype_str {
        "float64" | "float" => {
            let arr = match shape_vec.len() {
                1 => pecos::prelude::zeros(shape_vec[0]).into_dyn(),
                2 => pecos::prelude::zeros((shape_vec[0], shape_vec[1])).into_dyn(),
                3 => pecos::prelude::zeros((shape_vec[0], shape_vec[1], shape_vec[2])).into_dyn(),
                _ => {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "only 1D, 2D, and 3D arrays are currently supported",
                    ));
                }
            };
            Py::new(py, Array::from_array_f64(arr))
        }
        "complex128" | "complex" => {
            let arr = match shape_vec.len() {
                1 => NdArray::from_elem(shape_vec[0], Complex64::new(0.0, 0.0)).into_dyn(),
                2 => NdArray::from_elem((shape_vec[0], shape_vec[1]), Complex64::new(0.0, 0.0))
                    .into_dyn(),
                3 => NdArray::from_elem(
                    (shape_vec[0], shape_vec[1], shape_vec[2]),
                    Complex64::new(0.0, 0.0),
                )
                .into_dyn(),
                _ => {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "only 1D, 2D, and 3D arrays are currently supported",
                    ));
                }
            };
            Py::new(py, Array::from_array_c128(arr))
        }
        "int64" | "int" => {
            let arr = match shape_vec.len() {
                1 => NdArray::from_elem(shape_vec[0], 0i64).into_dyn(),
                2 => NdArray::from_elem((shape_vec[0], shape_vec[1]), 0i64).into_dyn(),
                3 => {
                    NdArray::from_elem((shape_vec[0], shape_vec[1], shape_vec[2]), 0i64).into_dyn()
                }
                _ => {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "only 1D, 2D, and 3D arrays are currently supported",
                    ));
                }
            };
            Py::new(py, Array::from_array_i64(arr))
        }
        "float32" | "f32" => {
            let arr = match shape_vec.len() {
                1 => NdArray::from_elem(shape_vec[0], 0.0f32).into_dyn(),
                2 => NdArray::from_elem((shape_vec[0], shape_vec[1]), 0.0f32).into_dyn(),
                3 => NdArray::from_elem((shape_vec[0], shape_vec[1], shape_vec[2]), 0.0f32)
                    .into_dyn(),
                _ => {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "only 1D, 2D, and 3D arrays are currently supported",
                    ));
                }
            };
            Py::new(py, Array::from_array_f32(arr))
        }
        "int32" | "i32" => {
            let arr = match shape_vec.len() {
                1 => NdArray::from_elem(shape_vec[0], 0i32).into_dyn(),
                2 => NdArray::from_elem((shape_vec[0], shape_vec[1]), 0i32).into_dyn(),
                3 => {
                    NdArray::from_elem((shape_vec[0], shape_vec[1], shape_vec[2]), 0i32).into_dyn()
                }
                _ => {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "only 1D, 2D, and 3D arrays are currently supported",
                    ));
                }
            };
            Py::new(py, Array::from_array_i32(arr))
        }
        "int16" | "i16" => {
            let arr = match shape_vec.len() {
                1 => NdArray::from_elem(shape_vec[0], 0i16).into_dyn(),
                2 => NdArray::from_elem((shape_vec[0], shape_vec[1]), 0i16).into_dyn(),
                3 => {
                    NdArray::from_elem((shape_vec[0], shape_vec[1], shape_vec[2]), 0i16).into_dyn()
                }
                _ => {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "only 1D, 2D, and 3D arrays are currently supported",
                    ));
                }
            };
            Py::new(py, Array::from_array_i16(arr))
        }
        "int8" | "i8" => {
            let arr = match shape_vec.len() {
                1 => NdArray::from_elem(shape_vec[0], 0i8).into_dyn(),
                2 => NdArray::from_elem((shape_vec[0], shape_vec[1]), 0i8).into_dyn(),
                3 => NdArray::from_elem((shape_vec[0], shape_vec[1], shape_vec[2]), 0i8).into_dyn(),
                _ => {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "only 1D, 2D, and 3D arrays are currently supported",
                    ));
                }
            };
            Py::new(py, Array::from_array_i8(arr))
        }
        "bool" => {
            let arr = match shape_vec.len() {
                1 => NdArray::from_elem(shape_vec[0], false).into_dyn(),
                2 => NdArray::from_elem((shape_vec[0], shape_vec[1]), false).into_dyn(),
                3 => {
                    NdArray::from_elem((shape_vec[0], shape_vec[1], shape_vec[2]), false).into_dyn()
                }
                _ => {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "only 1D, 2D, and 3D arrays are currently supported",
                    ));
                }
            };
            Py::new(py, Array::from_array_bool(arr))
        }
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "unsupported dtype: {dtype_str}. Supported: 'float64', 'float32', 'complex128', 'int64', 'int32', 'int16', 'int8', 'bool'"
        ))),
    }
}

/// Create a new array filled with ones.
///
/// Drop-in replacement for `numpy.ones()`.
///
/// # Arguments
///
/// * `shape` - Shape of the array as integer (1D) or tuple of integers (multi-D)
/// * `dtype` - Optional data type ('float64', 'complex128', 'int64'). Default is 'float64'.
///
/// # Returns
///
/// Array filled with ones of the specified shape and dtype
///
/// # Examples
///
/// ```python
/// from pecos_rslib.num import ones
///
/// # 1D array
/// arr = ones(5)  # [1.0, 1.0, 1.0, 1.0, 1.0]
///
/// # 2D array
/// arr2d = ones((2, 3))  # [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
///
/// # Integer dtype
/// arr_int = ones(5, dtype='int64')  # [1, 1, 1, 1, 1]
///
/// # Complex dtype
/// arr_complex = ones(3, dtype='complex128')  # [1+0j, 1+0j, 1+0j]
/// ```
#[pyfunction]
#[pyo3(signature = (shape, dtype=None))]
fn ones(
    py: Python<'_>,
    shape: Bound<'_, PyAny>,
    dtype: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<Array>> {
    use crate::dtypes::DType;
    use num_complex::Complex64;

    // Parse shape - can be int or tuple
    let shape_vec: Vec<usize> = if let Ok(n) = shape.extract::<usize>() {
        vec![n]
    } else if let Ok(tuple) = shape.extract::<Vec<usize>>() {
        tuple
    } else {
        return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "shape must be an integer or tuple of integers",
        ));
    };

    // Convert dtype to string - accept both DType enum and string, default to "float64"
    let dtype_str = if let Some(dt) = dtype {
        // dtype was provided
        if let Ok(enum_dt) = dt.extract::<DType>() {
            enum_dt.to_numpy_str()
        } else if let Ok(s) = dt.extract::<&str>() {
            s
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "dtype must be a string or DType enum",
            ));
        }
    } else {
        // dtype not provided, use default
        "float64"
    };

    match dtype_str {
        "float64" | "float" => {
            let arr = match shape_vec.len() {
                1 => pecos::prelude::ones(shape_vec[0]).into_dyn(),
                2 => pecos::prelude::ones((shape_vec[0], shape_vec[1])).into_dyn(),
                3 => pecos::prelude::ones((shape_vec[0], shape_vec[1], shape_vec[2])).into_dyn(),
                _ => {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "only 1D, 2D, and 3D arrays are currently supported",
                    ));
                }
            };
            Py::new(py, Array::from_array_f64(arr))
        }
        "complex128" | "complex" => {
            let arr = match shape_vec.len() {
                1 => NdArray::from_elem(shape_vec[0], Complex64::new(1.0, 0.0)).into_dyn(),
                2 => NdArray::from_elem((shape_vec[0], shape_vec[1]), Complex64::new(1.0, 0.0))
                    .into_dyn(),
                3 => NdArray::from_elem(
                    (shape_vec[0], shape_vec[1], shape_vec[2]),
                    Complex64::new(1.0, 0.0),
                )
                .into_dyn(),
                _ => {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "only 1D, 2D, and 3D arrays are currently supported",
                    ));
                }
            };
            Py::new(py, Array::from_array_c128(arr))
        }
        "int64" | "int" => {
            let arr = match shape_vec.len() {
                1 => NdArray::from_elem(shape_vec[0], 1i64).into_dyn(),
                2 => NdArray::from_elem((shape_vec[0], shape_vec[1]), 1i64).into_dyn(),
                3 => {
                    NdArray::from_elem((shape_vec[0], shape_vec[1], shape_vec[2]), 1i64).into_dyn()
                }
                _ => {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "only 1D, 2D, and 3D arrays are currently supported",
                    ));
                }
            };
            Py::new(py, Array::from_array_i64(arr))
        }
        "float32" | "f32" => {
            let arr = match shape_vec.len() {
                1 => NdArray::from_elem(shape_vec[0], 1.0f32).into_dyn(),
                2 => NdArray::from_elem((shape_vec[0], shape_vec[1]), 1.0f32).into_dyn(),
                3 => NdArray::from_elem((shape_vec[0], shape_vec[1], shape_vec[2]), 1.0f32)
                    .into_dyn(),
                _ => {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "only 1D, 2D, and 3D arrays are currently supported",
                    ));
                }
            };
            Py::new(py, Array::from_array_f32(arr))
        }
        "int32" | "i32" => {
            let arr = match shape_vec.len() {
                1 => NdArray::from_elem(shape_vec[0], 1i32).into_dyn(),
                2 => NdArray::from_elem((shape_vec[0], shape_vec[1]), 1i32).into_dyn(),
                3 => {
                    NdArray::from_elem((shape_vec[0], shape_vec[1], shape_vec[2]), 1i32).into_dyn()
                }
                _ => {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "only 1D, 2D, and 3D arrays are currently supported",
                    ));
                }
            };
            Py::new(py, Array::from_array_i32(arr))
        }
        "int16" | "i16" => {
            let arr = match shape_vec.len() {
                1 => NdArray::from_elem(shape_vec[0], 1i16).into_dyn(),
                2 => NdArray::from_elem((shape_vec[0], shape_vec[1]), 1i16).into_dyn(),
                3 => {
                    NdArray::from_elem((shape_vec[0], shape_vec[1], shape_vec[2]), 1i16).into_dyn()
                }
                _ => {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "only 1D, 2D, and 3D arrays are currently supported",
                    ));
                }
            };
            Py::new(py, Array::from_array_i16(arr))
        }
        "int8" | "i8" => {
            let arr = match shape_vec.len() {
                1 => NdArray::from_elem(shape_vec[0], 1i8).into_dyn(),
                2 => NdArray::from_elem((shape_vec[0], shape_vec[1]), 1i8).into_dyn(),
                3 => NdArray::from_elem((shape_vec[0], shape_vec[1], shape_vec[2]), 1i8).into_dyn(),
                _ => {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "only 1D, 2D, and 3D arrays are currently supported",
                    ));
                }
            };
            Py::new(py, Array::from_array_i8(arr))
        }
        "bool" => {
            let arr = match shape_vec.len() {
                1 => NdArray::from_elem(shape_vec[0], true).into_dyn(),
                2 => NdArray::from_elem((shape_vec[0], shape_vec[1]), true).into_dyn(),
                3 => {
                    NdArray::from_elem((shape_vec[0], shape_vec[1], shape_vec[2]), true).into_dyn()
                }
                _ => {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "only 1D, 2D, and 3D arrays are currently supported",
                    ));
                }
            };
            Py::new(py, Array::from_array_bool(arr))
        }
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "unsupported dtype: {dtype_str}. Supported: 'float64', 'float32', 'complex128', 'int64', 'int32', 'int16', 'int8', 'bool'"
        ))),
    }
}

/// Delete elements from an array at specified index.
///
/// Drop-in replacement for `numpy.delete()` for 1D arrays with single index.
///
/// This function is particularly useful for jackknife resampling and leave-one-out
/// cross-validation, which are common operations in threshold curve fitting.
///
/// # Arguments
///
/// * `arr` - Input array (1D numpy array or array-like)
/// * `index` - Index of the element to remove (integer)
///
/// # Returns
///
/// A new array with the element at `index` removed
///
/// # Examples
///
/// Create a numpy array from a Python list, tuple, or iterable.
///
/// Drop-in replacement for `numpy.array()`.
///
/// # Arguments
///
/// * `obj` - Python object (list, tuple, or iterable) to convert to array
/// * `dtype` - Optional data type ('float64', 'complex128', 'int64', or `DType` enum). If not specified, dtype is inferred.
///
/// # Returns
///
/// Numpy array with the specified or inferred dtype
///
/// # Examples
///
/// ```python
/// from pecos_rslib.num import array
/// from pecos_rslib import dtypes
///
/// # Create float array (dtype inferred)
/// arr = array([1.0, 2.0, 3.0])  # dtype: float64
///
/// # Create complex array (dtype inferred)
/// arr_complex = array([1+2j, 3+4j])  # dtype: complex128
///
/// # Create int array (dtype inferred)
/// arr_int = array([1, 2, 3])  # dtype: int64
///
/// # Explicitly specify dtype (string or DType enum)
/// arr_float = array([1, 2, 3], dtype='float64')  # [1.0, 2.0, 3.0]
/// arr_complex = array([1.0, 2.0], dtype=dtypes.complex128)  # [1+0j, 2+0j]
///
/// # Multi-dimensional arrays
/// arr_2d = array([[1.0, 2.0], [3.0, 4.0]])  # 2D array
/// arr_3d = array([[[1.0, 2.0]], [[3.0, 4.0]]])  # 3D array
///
/// ```
#[pyfunction]
#[pyo3(signature = (obj, dtype=None))]
fn array(
    py: Python<'_>,
    obj: Bound<'_, PyAny>,
    dtype: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<Array>> {
    use crate::dtypes::DType;
    use numpy::PyArrayMethods;

    // Check if obj is already an Array - if so, handle dtype conversion or copy
    if let Ok(existing_array) = obj.extract::<PyRef<'_, Array>>() {
        // Parse dtype parameter if provided
        let target_dtype = if let Some(dt) = dtype {
            Some(if let Ok(enum_dt) = dt.extract::<DType>() {
                enum_dt
            } else if let Ok(s) = dt.extract::<&str>() {
                DType::from_str(s)?
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "dtype must be a string or DType enum",
                ));
            })
        } else {
            None
        };

        // Get current dtype
        let current_dtype = existing_array.dtype();

        // Determine if we need to create a new array
        let needs_conversion = target_dtype.is_some() && target_dtype.unwrap() != current_dtype;

        if needs_conversion {
            // Perform dtype conversion using the pure Rust astype() method
            let converted_array = existing_array.astype(target_dtype.unwrap());
            return Py::new(py, converted_array);
        }

        // No dtype conversion needed - always create a copy
        let copied_array = existing_array.copy();
        return Py::new(py, copied_array);
    }

    // Convert input to NumPy array first, then use buffer protocol
    // This allows us to support arbitrary N-dimensional arrays
    // Get NumPy module and call numpy.array() to convert input
    let numpy_mod = py.import("numpy")?;

    // Build kwargs for numpy.array() call
    let kwargs = if let Some(dt) = dtype {
        // dtype was provided - convert DType enum to NumPy-compatible string
        let dict = pyo3::types::PyDict::new(py);

        // Check if dt is a DType enum - if so, convert to numpy string
        if let Ok(dtype_enum) = dt.extract::<DType>() {
            // It's our DType enum - convert to numpy-compatible string
            let numpy_str = dtype_enum.to_numpy_str();
            dict.set_item("dtype", numpy_str)?;
        } else {
            // It's already a string or numpy dtype - pass through directly
            dict.set_item("dtype", dt)?;
        }

        Some(dict)
    } else {
        None
    };

    // Call numpy.array(obj, dtype=dtype) to get a NumPy array
    let np_array = if let Some(kw) = kwargs {
        numpy_mod.call_method("array", (obj,), Some(&kw))?
    } else {
        numpy_mod.call_method("array", (obj,), None)?
    };

    // Now use buffer protocol to extract the array data
    // Try each dtype in order
    if let Ok(arr) = np_array.cast::<numpy::PyArrayDyn<f64>>() {
        let ndarray = arr.to_owned_array();
        return Py::new(
            py,
            Array {
                data: ArrayData::Float64(ndarray),
            },
        );
    }

    if let Ok(arr) = np_array.cast::<numpy::PyArrayDyn<i64>>() {
        let ndarray = arr.to_owned_array();
        return Py::new(
            py,
            Array {
                data: ArrayData::Int64(ndarray),
            },
        );
    }

    if let Ok(arr) = np_array.cast::<numpy::PyArrayDyn<num_complex::Complex<f64>>>() {
        let ndarray = arr.to_owned_array();
        return Py::new(
            py,
            Array {
                data: ArrayData::Complex128(ndarray),
            },
        );
    }

    if let Ok(arr) = np_array.cast::<numpy::PyArrayDyn<f32>>() {
        let ndarray = arr.to_owned_array();
        return Py::new(
            py,
            Array {
                data: ArrayData::Float32(ndarray),
            },
        );
    }

    if let Ok(arr) = np_array.cast::<numpy::PyArrayDyn<i32>>() {
        let ndarray = arr.to_owned_array();
        return Py::new(
            py,
            Array {
                data: ArrayData::Int32(ndarray),
            },
        );
    }

    if let Ok(arr) = np_array.cast::<numpy::PyArrayDyn<i16>>() {
        let ndarray = arr.to_owned_array();
        return Py::new(
            py,
            Array {
                data: ArrayData::Int16(ndarray),
            },
        );
    }

    if let Ok(arr) = np_array.cast::<numpy::PyArrayDyn<i8>>() {
        let ndarray = arr.to_owned_array();
        return Py::new(
            py,
            Array {
                data: ArrayData::Int8(ndarray),
            },
        );
    }

    if let Ok(arr) = np_array.cast::<numpy::PyArrayDyn<bool>>() {
        let ndarray = arr.to_owned_array();
        return Py::new(
            py,
            Array {
                data: ArrayData::Bool(ndarray),
            },
        );
    }

    if let Ok(arr) = np_array.cast::<numpy::PyArrayDyn<num_complex::Complex<f32>>>() {
        let ndarray = arr.to_owned_array();
        return Py::new(
            py,
            Array {
                data: ArrayData::Complex64(ndarray),
            },
        );
    }

    // If we get here, dtype is not supported
    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
        "Unsupported dtype - array() failed to convert input",
    ))
}

/// Delete an element at a specific index from a 1D array.
///
/// Drop-in replacement for `numpy.delete(arr, index)` for 1D arrays.
///
/// This is particularly useful for jackknife resampling (leave-one-out cross-validation)
/// and other statistical techniques that require creating copies with one element removed.
///
/// # Arguments
///
/// * `arr` - Input array
/// * `index` - Index of element to delete
///
/// # Returns
///
/// New array with the specified element removed
///
/// # Examples
///
/// ```python
/// from pecos_rslib.num import delete
///
/// # Delete from float array
/// arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
/// result = delete(arr, 2)  # [1.0, 2.0, 4.0, 5.0]
///
/// # Delete from complex array
/// arr_complex = np.array([1+2j, 3+4j, 5+6j])
/// result = delete(arr_complex, 1)  # [1+2j, 5+6j]
///
/// # Jackknife resampling (leave-one-out)
/// plist = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
/// for i in range(len(plist)):
///     p_copy = delete(plist, i)  # Remove i-th element
///     # ... perform analysis on p_copy ...
/// ```
#[pyfunction]
fn delete(py: Python<'_>, arr: Bound<'_, PyAny>, index: usize) -> PyResult<Py<PyAny>> {
    // Check if it's already a numpy array and get its dtype
    if let Ok(dtype_attr) = arr.getattr("dtype")
        && let Ok(kind_attr) = dtype_attr.getattr("kind")
    {
        let kind: String = kind_attr.extract()?;
        match kind.as_str() {
            "f" => {
                // Float array
                let arr_read = arr.extract::<numpy::PyReadonlyArray1<f64>>()?;
                let arr_rust = arr_read.as_array();

                if index >= arr_rust.len() {
                    return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                        "index {} is out of bounds for array of length {}",
                        index,
                        arr_rust.len()
                    )));
                }

                let result = pecos::prelude::delete(&arr_rust.to_owned(), index);
                return Ok(PyArray::from_owned_array(py, result).into_any().unbind());
            }
            "c" => {
                // Complex array
                let arr_read = arr.extract::<numpy::PyReadonlyArray1<Complex64>>()?;
                let arr_rust = arr_read.as_array();

                if index >= arr_rust.len() {
                    return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                        "index {} is out of bounds for array of length {}",
                        index,
                        arr_rust.len()
                    )));
                }

                let result = pecos::prelude::delete(&arr_rust.to_owned(), index);
                return Ok(PyArray::from_owned_array(py, result).into_any().unbind());
            }
            "i" | "u" => {
                // Integer array
                let arr_read = arr.extract::<numpy::PyReadonlyArray1<i64>>()?;
                let arr_rust = arr_read.as_array();

                if index >= arr_rust.len() {
                    return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                        "index {} is out of bounds for array of length {}",
                        index,
                        arr_rust.len()
                    )));
                }

                let result = pecos::prelude::delete(&arr_rust.to_owned(), index);
                return Ok(PyArray::from_owned_array(py, result).into_any().unbind());
            }
            _ => {
                return Err(PyTypeError::new_err(format!(
                    "Unsupported dtype kind: {kind}"
                )));
            }
        }
    }

    // If not a numpy array, try to convert it to one first
    let numpy = py.import("numpy")?;
    let np_array = numpy.call_method1("array", (arr,))?;

    // Recursively call with the converted array
    delete(py, np_array, index)
}

/// Calculate the sum of array elements.
///
/// Drop-in replacement for `numpy.sum()` with full polymorphism and axis support.
/// Handles lists, tuples, numpy arrays (float and complex), and axis parameter.
///
/// # Arguments
///
/// * `a` - Array-like input (list, tuple, numpy array of floats or complex)
/// * `axis` - Optional axis along which to sum. If None, sum all elements (default).
///
/// # Returns
///
/// Sum of elements. Returns scalar if axis=None, otherwise returns array.
/// Type is f64 for float inputs, Complex64 for complex inputs.
///
/// # Examples
///
/// ```python
/// from pecos_rslib.num import sum
/// import numpy as np
///
/// # List/tuple - sum all elements
/// assert sum([1.0, 2.0, 3.0]) == 6.0
/// assert sum((1.0, 2.0, 3.0)) == 6.0
///
/// # Numpy array - sum all elements
/// assert sum(np.array([1.0, 2.0, 3.0])) == 6.0
///
/// # Complex numbers
/// arr = np.array([1+2j, 3+4j])
/// assert sum(arr) == 4+6j
///
/// # 2D array with axis parameter
/// arr = np.array([[1.0, 2.0], [3.0, 4.0]])
/// # Sum along axis 0 (down columns)
/// result = sum(arr, axis=0)  # [4.0, 6.0]
/// # Sum along axis 1 (across rows)
/// result = sum(arr, axis=1)  # [3.0, 7.0]
/// ```
#[pyfunction]
#[pyo3(signature = (a, axis=None))]
#[allow(clippy::needless_pass_by_value)] // Bound is designed to be passed by value (PyO3 convention)
fn sum(py: Python<'_>, a: Bound<'_, PyAny>, axis: Option<isize>) -> PyResult<Py<PyAny>> {
    use num_complex::Complex64;

    // Handle axis=None case: sum all elements
    if axis.is_none() {
        // Check if it's a numpy array by checking for 'dtype' attribute
        if let Ok(dtype_attr) = a.getattr("dtype") {
            // It's a numpy array - check its dtype.kind
            if let Ok(kind_attr) = dtype_attr.getattr("kind") {
                let kind: String = kind_attr.extract()?;

                match kind.as_str() {
                    "i" | "u" => {
                        // Integer array
                        let arr = a.extract::<numpy::PyReadonlyArrayDyn<i64>>()?;
                        let result: i64 = arr.as_array().iter().sum();
                        return Ok(result.into_py_any(py).unwrap());
                    }
                    "f" => {
                        // Float array
                        let arr = a.extract::<numpy::PyReadonlyArrayDyn<f64>>()?;
                        let result: f64 = arr.as_array().iter().sum();
                        return Ok(result.into_py_any(py).unwrap());
                    }
                    "c" => {
                        // Complex array
                        let arr = a.extract::<numpy::PyReadonlyArrayDyn<Complex64>>()?;
                        let result: Complex64 = arr.as_array().iter().copied().sum();
                        return result.into_py_any(py);
                    }
                    _ => {
                        return Err(PyTypeError::new_err(format!(
                            "Unsupported dtype kind: {kind}"
                        )));
                    }
                }
            }
        }

        // Not a numpy array - try lists/tuples
        // Try integer list/tuple first
        if let Ok(values) = a.extract::<Vec<i64>>() {
            let result: i64 = values.iter().sum();
            return Ok(result.into_py_any(py).unwrap());
        }

        // Try float list/tuple (before complex, since floats can convert to complex!)
        if let Ok(values) = a.extract::<Vec<f64>>() {
            let result: f64 = values.iter().sum();
            return Ok(result.into_py_any(py).unwrap());
        }

        // Try complex list/tuple
        if let Ok(values) = a.extract::<Vec<Complex64>>() {
            let result: Complex64 = values.iter().copied().sum();
            return result.into_py_any(py);
        }

        return Err(PyTypeError::new_err(
            "sum() argument must be a list, tuple, or numpy array of numbers",
        ));
    }

    // Handle axis parameter case: sum along specific axis
    let axis_val = axis.unwrap();

    // Convert Python lists/tuples to numpy arrays for axis operations
    // If it's not already a numpy array, try to convert it
    let np_array = if a.extract::<numpy::PyReadonlyArrayDyn<f64>>().is_err()
        && a.extract::<numpy::PyReadonlyArrayDyn<Complex64>>().is_err()
        && a.extract::<numpy::PyReadonlyArrayDyn<i64>>().is_err()
    {
        // Not a numpy array - convert to numpy array using numpy.array()
        let numpy = py.import("numpy")?;
        numpy.call_method1("array", (a,))?
    } else {
        // Already a numpy array
        a
    };

    // Try integer array with axis FIRST (before complex/float to avoid unwanted casting)
    if let Ok(arr) = np_array.extract::<numpy::PyReadonlyArrayDyn<i64>>() {
        let array = arr.as_array();
        let ndim = array.ndim();

        // Convert negative axis to positive
        let normalized_axis = if axis_val < 0 {
            (ndim as isize + axis_val) as usize
        } else {
            axis_val as usize
        };

        if normalized_axis >= ndim {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "axis {axis_val} is out of bounds for array of dimension {ndim}"
            )));
        }

        // Sum along the specified axis
        let result = array.sum_axis(Axis(normalized_axis));
        return Ok(PyArray::from_owned_array(py, result).into_any().unbind());
    }

    // Try complex array with axis (before float, to avoid unwanted casting)
    if let Ok(arr) = np_array.extract::<numpy::PyReadonlyArrayDyn<Complex64>>() {
        let array = arr.as_array();
        let ndim = array.ndim();

        // Convert negative axis to positive
        let normalized_axis = if axis_val < 0 {
            (ndim as isize + axis_val) as usize
        } else {
            axis_val as usize
        };

        if normalized_axis >= ndim {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "axis {axis_val} is out of bounds for array of dimension {ndim}"
            )));
        }

        // Sum along the specified axis
        let result = array.sum_axis(Axis(normalized_axis));
        return Ok(PyArray::from_owned_array(py, result).into_any().unbind());
    }

    // Try float array with axis
    if let Ok(arr) = np_array.extract::<numpy::PyReadonlyArrayDyn<f64>>() {
        let array = arr.as_array();
        let ndim = array.ndim();

        // Convert negative axis to positive
        let normalized_axis = if axis_val < 0 {
            (ndim as isize + axis_val) as usize
        } else {
            axis_val as usize
        };

        if normalized_axis >= ndim {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "axis {axis_val} is out of bounds for array of dimension {ndim}"
            )));
        }

        // Sum along the specified axis using ndarray's sum_axis
        let result = array.sum_axis(Axis(normalized_axis));
        return Ok(PyArray::from_owned_array(py, result).into_any().unbind());
    }

    Err(PyTypeError::new_err(
        "sum() with axis requires a numpy array of numbers",
    ))
}

// ============================================================================
// Array and Complex Number Support
// ============================================================================

// ============================================================================
// Math Functions (polymorphic - handle scalars, complex, and arrays)
// ============================================================================

/// Calculate exponential (e^x).
///
/// Handles scalars (float), complex numbers, and arrays automatically.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)] // Bound is designed to be passed by value (PyO3 convention)
fn exp(py: Python<'_>, x: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    use pecos::prelude::Exp;

    if let Ok(val) = x.extract::<f64>() {
        return Ok(val.exp().into_py_any(py).unwrap());
    }
    if let Ok(val) = x.extract::<Complex64>() {
        return Ok(val.exp().into_py_any(py).unwrap());
    }
    if let Ok(arr) = x.cast::<PyArray<f64, IxDyn>>() {
        let result = arr.try_readonly()?.as_array().exp();
        return Ok(PyArray::from_owned_array(py, result).into_any().unbind());
    }
    if let Ok(arr) = x.cast::<PyArray<Complex64, IxDyn>>() {
        let result = arr.try_readonly()?.as_array().exp();
        return Ok(PyArray::from_owned_array(py, result).into_any().unbind());
    }
    Err(PyTypeError::new_err(
        "exp() argument must be float, complex, or array",
    ))
}

/// Calculate natural logarithm (base e).
///
/// More explicit than `numpy.log()` - uses `ln()` instead of `log()` for clarity.
/// Handles scalars (float, complex) and arrays automatically.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)] // Bound is designed to be passed by value (PyO3 convention)
fn ln(py: Python<'_>, x: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    // Use .ln() - natural logarithm is standard in Rust stdlib, ndarray, and num-complex
    // For consistency, we provide an Ln trait for Complex64 arrays

    // Import Ln trait for Complex64 array support
    #[allow(unused_imports)]
    use pecos::prelude::Ln;

    // Scalars: use stdlib .ln() method
    if let Ok(val) = x.extract::<f64>() {
        return Ok(val.ln().into_py_any(py).unwrap());
    }
    if let Ok(val) = x.extract::<Complex64>() {
        return Ok(val.ln().into_py_any(py).unwrap());
    }

    // Arrays: use .ln() method uniformly
    // - f64 arrays: ndarray provides .ln() built-in
    // - Complex64 arrays: our Ln trait provides .ln()
    if let Ok(arr) = x.cast::<PyArray<f64, IxDyn>>() {
        let result = arr.try_readonly()?.as_array().ln();
        return Ok(PyArray::from_owned_array(py, result).into_any().unbind());
    }
    if let Ok(arr) = x.cast::<PyArray<Complex64, IxDyn>>() {
        let result = arr.try_readonly()?.as_array().ln();
        return Ok(PyArray::from_owned_array(py, result).into_any().unbind());
    }

    Err(PyTypeError::new_err(
        "ln() argument must be float, complex, or array",
    ))
}

/// Calculate logarithm with custom base.
///
/// More general than natural logarithm - log(x, base) returns `log_base(x)`.
/// Handles scalars (float, complex) and arrays automatically.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)] // Bound is designed to be passed by value (PyO3 convention)
fn log(py: Python<'_>, x: Bound<'_, PyAny>, base: f64) -> PyResult<Py<PyAny>> {
    // Use .log(base) - logarithm with custom base
    // For consistency, we provide a LogBase trait for Complex64 arrays

    // Import LogBase trait for Complex64 array support
    #[allow(unused_imports)]
    use pecos::prelude::LogBase;

    // Scalars: use stdlib .log(base) method
    if let Ok(val) = x.extract::<f64>() {
        return Ok(val.log(base).into_py_any(py).unwrap());
    }
    if let Ok(val) = x.extract::<Complex64>() {
        return Ok(val.log(base).into_py_any(py).unwrap());
    }

    // Arrays: use .log(base) method uniformly
    // - f64 arrays: ndarray provides .log(base) built-in
    // - Complex64 arrays: our LogBase trait provides .log(base)
    if let Ok(arr) = x.cast::<PyArray<f64, IxDyn>>() {
        let result = arr.try_readonly()?.as_array().log(base);
        return Ok(PyArray::from_owned_array(py, result).into_any().unbind());
    }
    if let Ok(arr) = x.cast::<PyArray<Complex64, IxDyn>>() {
        let result = arr.try_readonly()?.as_array().log(base);
        return Ok(PyArray::from_owned_array(py, result).into_any().unbind());
    }

    Err(PyTypeError::new_err(
        "log() argument must be float, complex, or array",
    ))
}

/// Test whether all array elements evaluate to True.
///
/// Drop-in replacement for `numpy.all()`.
/// Returns True if all elements are truthy (non-zero for numbers, True for bools).
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn all(_py: Python<'_>, a: Bound<'_, PyAny>) -> PyResult<bool> {
    // Handle boolean arrays
    if let Ok(arr) = a.extract::<PyReadonlyArrayDyn<bool>>() {
        return Ok(arr.as_array().iter().all(|&x| x));
    }

    // Handle float arrays (non-zero is truthy)
    if let Ok(arr) = a.extract::<PyReadonlyArrayDyn<f64>>() {
        return Ok(arr.as_array().iter().all(|&x| x != 0.0));
    }

    // Handle integer arrays
    if let Ok(arr) = a.extract::<PyReadonlyArrayDyn<i64>>() {
        return Ok(arr.as_array().iter().all(|&x| x != 0));
    }

    // Handle boolean scalar
    if let Ok(val) = a.extract::<bool>() {
        return Ok(val);
    }

    // Handle float scalar
    if let Ok(val) = a.extract::<f64>() {
        return Ok(val != 0.0);
    }

    // Handle integer scalar
    if let Ok(val) = a.extract::<i64>() {
        return Ok(val != 0);
    }

    Err(PyTypeError::new_err(
        "all() argument must be bool, numeric scalar, or array",
    ))
}

/// Test whether any array element evaluates to True.
///
/// Drop-in replacement for `numpy.any()`.
/// Returns True if any element is truthy (non-zero for numbers, True for bools).
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn any(_py: Python<'_>, a: Bound<'_, PyAny>) -> PyResult<bool> {
    // Handle boolean arrays
    if let Ok(arr) = a.extract::<PyReadonlyArrayDyn<bool>>() {
        return Ok(arr.as_array().iter().any(|&x| x));
    }

    // Handle float arrays (non-zero is truthy)
    if let Ok(arr) = a.extract::<PyReadonlyArrayDyn<f64>>() {
        return Ok(arr.as_array().iter().any(|&x| x != 0.0));
    }

    // Handle integer arrays
    if let Ok(arr) = a.extract::<PyReadonlyArrayDyn<i64>>() {
        return Ok(arr.as_array().iter().any(|&x| x != 0));
    }

    // Handle boolean scalar
    if let Ok(val) = a.extract::<bool>() {
        return Ok(val);
    }

    // Handle float scalar
    if let Ok(val) = a.extract::<f64>() {
        return Ok(val != 0.0);
    }

    // Handle integer scalar
    if let Ok(val) = a.extract::<i64>() {
        return Ok(val != 0);
    }

    Err(PyTypeError::new_err(
        "any() argument must be bool, numeric scalar, or array",
    ))
}

/// Compute the norm of a vector or matrix.
///
/// Drop-in replacement for `numpy.linalg.norm()`.
///
/// # Arguments
///
/// * `x` - Input array (1-D or 2-D), including Array
/// * `ord` - Order of the norm (default: 2 for vectors, Frobenius for matrices)
///
/// Returns the norm as a float.
#[pyfunction]
#[pyo3(signature = (x, ord=None))]
#[allow(clippy::needless_pass_by_value)]
fn norm(_py: Python<'_>, x: Bound<'_, PyAny>, ord: Option<f64>) -> PyResult<f64> {
    use crate::pecos_array::{Array, ArrayData};
    use pecos::prelude::{norm as norm_fn, norm_complex};

    // Try Array first - extract underlying data directly
    if let Ok(pecos_arr) = x.cast::<Array>() {
        let pecos_arr_ref = pecos_arr.borrow();
        // Access the internal data field and match on its type
        return match &pecos_arr_ref.data {
            ArrayData::Bool(_) => Err(pyo3::exceptions::PyTypeError::new_err(
                "norm() operation not supported on boolean arrays",
            )),
            ArrayData::Float64(arr) => Ok(norm_fn(arr, ord)),
            ArrayData::Float32(arr) => {
                // Convert f32 to f64 for norm calculation
                let arr_f64 = arr.mapv(f64::from);
                Ok(norm_fn(&arr_f64, ord))
            }
            ArrayData::Complex128(arr) => Ok(norm_complex(arr, ord)),
            ArrayData::Complex64(arr) => {
                // Convert Complex<f32> to Complex<f64>
                let arr_c128 = arr.mapv(|v| Complex64::new(f64::from(v.re), f64::from(v.im)));
                Ok(norm_complex(&arr_c128, ord))
            }
            ArrayData::Int64(arr) => {
                // Convert int to float for norm
                let arr_f64 = arr.mapv(|v| v as f64);
                Ok(norm_fn(&arr_f64, ord))
            }
            ArrayData::Int32(arr) => {
                let arr_f64 = arr.mapv(f64::from);
                Ok(norm_fn(&arr_f64, ord))
            }
            ArrayData::Int16(arr) => {
                let arr_f64 = arr.mapv(f64::from);
                Ok(norm_fn(&arr_f64, ord))
            }
            ArrayData::Int8(arr) => {
                let arr_f64 = arr.mapv(f64::from);
                Ok(norm_fn(&arr_f64, ord))
            }
        };
    }

    // Try f64 arrays (numpy arrays)
    if let Ok(arr) = x.extract::<PyReadonlyArrayDyn<f64>>() {
        return Ok(norm_fn(&arr.as_array(), ord));
    }

    // Try Complex64 arrays (numpy arrays)
    if let Ok(arr) = x.extract::<PyReadonlyArrayDyn<Complex64>>() {
        return Ok(norm_complex(&arr.as_array(), ord));
    }

    // Try Python list/tuple of floats - convert directly to ndarray
    if let Ok(values) = x.extract::<Vec<f64>>() {
        let arr = Array1::from(values);
        return Ok(norm_fn(&arr.view(), ord));
    }

    // Try Python list/tuple of complex - convert directly to ndarray
    if let Ok(values) = x.extract::<Vec<Complex64>>() {
        let arr = Array1::from(values);
        return Ok(norm_complex(&arr.view(), ord));
    }

    Err(PyTypeError::new_err(
        "norm() argument must be a numeric array or list",
    ))
}

/// Calculate square root.
///
/// Handles scalars (float) and arrays automatically.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)] // Bound is designed to be passed by value (PyO3 convention)
fn sqrt(py: Python<'_>, x: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    // Import trait to enable .sqrt() method
    #[allow(unused_imports)]
    use pecos::prelude::Sqrt;

    // Try scalar first
    if let Ok(val) = x.extract::<f64>() {
        return Ok(val.sqrt().into_py_any(py).unwrap());
    }

    // Try numpy array
    if let Ok(arr) = x.cast::<PyArray<f64, IxDyn>>() {
        let result = arr.try_readonly()?.as_array().sqrt();
        return Ok(PyArray::from_owned_array(py, result).into_any().unbind());
    }

    // Try Python sequence (list, tuple, etc.) - convert to 1D array
    if let Ok(vec) = x.extract::<Vec<f64>>() {
        let arr = Array1::from(vec);
        let result = arr.sqrt();
        return Ok(PyArray::from_owned_array(py, result).into_any().unbind());
    }

    // Try 2D Python sequence (nested lists) - convert to numpy first
    // PyO3's extract doesn't handle Vec<Vec<f64>> well, so use numpy module
    if let Ok(numpy) = py.import("numpy")
        && let Ok(np_array) = numpy.call_method1("array", (x,))
        && let Ok(arr) = np_array.cast::<PyArray<f64, IxDyn>>()
    {
        let result = arr.try_readonly()?.as_array().sqrt();
        return Ok(PyArray::from_owned_array(py, result).into_any().unbind());
    }

    Err(PyTypeError::new_err(
        "sqrt() argument must be float, array, or sequence",
    ))
}

/// Calculate base raised to exponent.
///
/// Handles scalars (float) and arrays automatically.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)] // Bound is designed to be passed by value (PyO3 convention)
fn power(
    py: Python<'_>,
    base: Bound<'_, PyAny>,
    exponent: Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    use pecos::prelude::{Array1, Power};

    // Try to extract exponent as scalar first (most common case)
    if let Ok(exp_val) = exponent.extract::<f64>() {
        // Scalar exponent - use Power trait

        // Try scalar base
        if let Ok(val) = base.extract::<f64>() {
            return Ok(val.power(exp_val).into_py_any(py).unwrap());
        }

        // Try numpy array base
        if let Ok(arr) = base.cast::<PyArray<f64, IxDyn>>() {
            let result = arr.try_readonly()?.as_array().power(exp_val);
            return Ok(PyArray::from_owned_array(py, result).into_any().unbind());
        }

        // Try Python sequence base (list, tuple, etc.) - 1D
        if let Ok(vec) = base.extract::<Vec<f64>>() {
            let arr = Array1::from(vec);
            let result = arr.power(exp_val);
            return Ok(PyArray::from_owned_array(py, result).into_any().unbind());
        }

        // Try 2D Python sequence (nested lists) - convert to numpy first
        if let Ok(numpy) = py.import("numpy")
            && let Ok(np_array) = numpy.call_method1("array", (base,))
            && let Ok(arr) = np_array.cast::<PyArray<f64, IxDyn>>()
        {
            let result = arr.try_readonly()?.as_array().power(exp_val);
            return Ok(PyArray::from_owned_array(py, result).into_any().unbind());
        }

        return Err(PyTypeError::new_err(
            "power() base must be float, array, or sequence",
        ));
    }

    // Array exponent - need element-wise power using std::f64::powf
    // Get base as scalar
    if let Ok(base_val) = base.extract::<f64>() {
        // Try numpy array exponent
        if let Ok(exp_arr) = exponent.cast::<PyArray<f64, IxDyn>>() {
            let exp_readonly = exp_arr.try_readonly()?;
            let exp_view = exp_readonly.as_array();
            let result = exp_view.mapv(|e| base_val.powf(e));
            return Ok(PyArray::from_owned_array(py, result).into_any().unbind());
        }

        // Try Python sequence exponent
        if let Ok(exp_vec) = exponent.extract::<Vec<f64>>() {
            let result: Vec<f64> = exp_vec.iter().map(|&e| base_val.powf(e)).collect();
            let arr = Array1::from(result);
            return Ok(PyArray::from_owned_array(py, arr).into_any().unbind());
        }
    }

    Err(PyTypeError::new_err(
        "power() requires scalar exponent or scalar base with array exponent",
    ))
}

/// Calculate cosine (input in radians).
///
/// Handles scalars (float) and arrays automatically.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)] // Bound is designed to be passed by value (PyO3 convention)
fn cos(py: Python<'_>, x: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    // Import trait to enable .cos() method
    #[allow(unused_imports)]
    use pecos::prelude::Cos;

    if let Ok(val) = x.extract::<f64>() {
        return Ok(val.cos().into_py_any(py).unwrap());
    }
    if let Ok(arr) = x.cast::<PyArray<f64, IxDyn>>() {
        let result = arr.try_readonly()?.as_array().cos();
        return Ok(PyArray::from_owned_array(py, result).into_any().unbind());
    }
    Err(PyTypeError::new_err(
        "cos() argument must be float or array",
    ))
}

/// Calculate sine (input in radians).
///
/// Handles scalars (float) and arrays automatically.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)] // Bound is designed to be passed by value (PyO3 convention)
fn sin(py: Python<'_>, x: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    // Import trait to enable .sin() method
    #[allow(unused_imports)]
    use pecos::prelude::Sin;

    if let Ok(val) = x.extract::<f64>() {
        return Ok(val.sin().into_py_any(py).unwrap());
    }
    if let Ok(arr) = x.cast::<PyArray<f64, IxDyn>>() {
        let result = arr.try_readonly()?.as_array().sin();
        return Ok(PyArray::from_owned_array(py, result).into_any().unbind());
    }
    Err(PyTypeError::new_err(
        "sin() argument must be float or array",
    ))
}

/// Calculate tangent (input in radians).
///
/// Drop-in replacement for `numpy.tan()`.
/// Handles scalars (float, complex) and arrays automatically.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)] // Bound is designed to be passed by value (PyO3 convention)
fn tan(py: Python<'_>, x: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    // Import trait to enable .tan() method
    #[allow(unused_imports)]
    use pecos::prelude::Tan;

    // Try scalar float
    if let Ok(val) = x.extract::<f64>() {
        return Ok(val.tan().into_py_any(py).unwrap());
    }
    // Try scalar complex
    if let Ok(val) = x.extract::<Complex64>() {
        return Ok(val.tan().into_py_any(py).unwrap());
    }
    // Try float array
    if let Ok(arr) = x.cast::<PyArray<f64, IxDyn>>() {
        let result = arr.try_readonly()?.as_array().tan();
        return Ok(PyArray::from_owned_array(py, result).into_any().unbind());
    }
    // Try complex array
    if let Ok(arr) = x.cast::<PyArray<Complex64, IxDyn>>() {
        let result = arr.try_readonly()?.as_array().tan();
        return Ok(PyArray::from_owned_array(py, result).into_any().unbind());
    }
    Err(PyTypeError::new_err(
        "tan() argument must be float, complex, or array",
    ))
}

/// Calculate hyperbolic sine.
///
/// Drop-in replacement for `numpy.sinh()`.
/// Handles scalars (float, complex) and arrays automatically.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn sinh(py: Python<'_>, x: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    use pecos::prelude::Sinh;

    if let Ok(val) = x.extract::<f64>() {
        return Ok(val.sinh().into_py_any(py).unwrap());
    }
    if let Ok(val) = x.extract::<Complex64>() {
        return Ok(val.sinh().into_py_any(py).unwrap());
    }
    if let Ok(arr) = x.cast::<PyArray<f64, IxDyn>>() {
        let result = arr.try_readonly()?.as_array().sinh();
        return Ok(PyArray::from_owned_array(py, result).into_any().unbind());
    }
    if let Ok(arr) = x.cast::<PyArray<Complex64, IxDyn>>() {
        let result = arr.try_readonly()?.as_array().sinh();
        return Ok(PyArray::from_owned_array(py, result).into_any().unbind());
    }
    Err(PyTypeError::new_err(
        "sinh() argument must be float, complex, or array",
    ))
}

/// Calculate hyperbolic cosine.
///
/// Drop-in replacement for `numpy.cosh()`.
/// Handles scalars (float, complex) and arrays automatically.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn cosh(py: Python<'_>, x: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    use pecos::prelude::Cosh;

    if let Ok(val) = x.extract::<f64>() {
        return Ok(val.cosh().into_py_any(py).unwrap());
    }
    if let Ok(val) = x.extract::<Complex64>() {
        return Ok(val.cosh().into_py_any(py).unwrap());
    }
    if let Ok(arr) = x.cast::<PyArray<f64, IxDyn>>() {
        let result = arr.try_readonly()?.as_array().cosh();
        return Ok(PyArray::from_owned_array(py, result).into_any().unbind());
    }
    if let Ok(arr) = x.cast::<PyArray<Complex64, IxDyn>>() {
        let result = arr.try_readonly()?.as_array().cosh();
        return Ok(PyArray::from_owned_array(py, result).into_any().unbind());
    }
    Err(PyTypeError::new_err(
        "cosh() argument must be float, complex, or array",
    ))
}

/// Calculate hyperbolic tangent.
///
/// Drop-in replacement for `numpy.tanh()`.
/// Handles scalars (float, complex) and arrays automatically.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)] // Bound is designed to be passed by value (PyO3 convention)
fn tanh(py: Python<'_>, x: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    // Import trait to enable .tanh() method
    #[allow(unused_imports)]
    use pecos::prelude::Tanh;

    // Try scalar float
    if let Ok(val) = x.extract::<f64>() {
        return Ok(val.tanh().into_py_any(py).unwrap());
    }
    // Try scalar complex
    if let Ok(val) = x.extract::<Complex64>() {
        return Ok(val.tanh().into_py_any(py).unwrap());
    }
    // Try float array
    if let Ok(arr) = x.cast::<PyArray<f64, IxDyn>>() {
        let result = arr.try_readonly()?.as_array().tanh();
        return Ok(PyArray::from_owned_array(py, result).into_any().unbind());
    }
    // Try complex array
    if let Ok(arr) = x.cast::<PyArray<Complex64, IxDyn>>() {
        let result = arr.try_readonly()?.as_array().tanh();
        return Ok(PyArray::from_owned_array(py, result).into_any().unbind());
    }
    Err(PyTypeError::new_err(
        "tanh() argument must be float, complex, or array",
    ))
}

/// Calculate arcsine (inverse sine).
///
/// Drop-in replacement for `numpy.arcsin()` / `numpy.asin()`.
/// Handles scalars (float, complex) and arrays automatically.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn asin(py: Python<'_>, x: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    use pecos::prelude::Asin;

    if let Ok(val) = x.extract::<f64>() {
        return Ok(val.asin().into_py_any(py).unwrap());
    }
    if let Ok(val) = x.extract::<Complex64>() {
        return Ok(val.asin().into_py_any(py).unwrap());
    }
    if let Ok(arr) = x.cast::<PyArray<f64, IxDyn>>() {
        let result = arr.try_readonly()?.as_array().asin();
        return Ok(PyArray::from_owned_array(py, result).into_any().unbind());
    }
    if let Ok(arr) = x.cast::<PyArray<Complex64, IxDyn>>() {
        let result = arr.try_readonly()?.as_array().asin();
        return Ok(PyArray::from_owned_array(py, result).into_any().unbind());
    }
    Err(PyTypeError::new_err(
        "asin() argument must be float, complex, or array",
    ))
}

/// Calculate arccosine (inverse cosine).
///
/// Drop-in replacement for `numpy.arccos()` / `numpy.acos()`.
/// Handles scalars (float, complex) and arrays automatically.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn acos(py: Python<'_>, x: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    use pecos::prelude::Acos;

    if let Ok(val) = x.extract::<f64>() {
        return Ok(val.acos().into_py_any(py).unwrap());
    }
    if let Ok(val) = x.extract::<Complex64>() {
        return Ok(val.acos().into_py_any(py).unwrap());
    }
    if let Ok(arr) = x.cast::<PyArray<f64, IxDyn>>() {
        let result = arr.try_readonly()?.as_array().acos();
        return Ok(PyArray::from_owned_array(py, result).into_any().unbind());
    }
    if let Ok(arr) = x.cast::<PyArray<Complex64, IxDyn>>() {
        let result = arr.try_readonly()?.as_array().acos();
        return Ok(PyArray::from_owned_array(py, result).into_any().unbind());
    }
    Err(PyTypeError::new_err(
        "acos() argument must be float, complex, or array",
    ))
}

/// Calculate arctangent (inverse tangent).
///
/// Drop-in replacement for `numpy.arctan()` / `numpy.atan()`.
/// Handles scalars (float, complex) and arrays automatically.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn atan(py: Python<'_>, x: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    use pecos::prelude::Atan;

    if let Ok(val) = x.extract::<f64>() {
        return Ok(val.atan().into_py_any(py).unwrap());
    }
    if let Ok(val) = x.extract::<Complex64>() {
        return Ok(val.atan().into_py_any(py).unwrap());
    }
    if let Ok(arr) = x.cast::<PyArray<f64, IxDyn>>() {
        let result = arr.try_readonly()?.as_array().atan();
        return Ok(PyArray::from_owned_array(py, result).into_any().unbind());
    }
    if let Ok(arr) = x.cast::<PyArray<Complex64, IxDyn>>() {
        let result = arr.try_readonly()?.as_array().atan();
        return Ok(PyArray::from_owned_array(py, result).into_any().unbind());
    }
    Err(PyTypeError::new_err(
        "atan() argument must be float, complex, or array",
    ))
}

/// Calculate inverse hyperbolic sine.
///
/// Drop-in replacement for `numpy.arcsinh()` / `numpy.asinh()`.
/// Handles scalars (float, complex) and arrays automatically.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn asinh(py: Python<'_>, x: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    use pecos::prelude::Asinh;

    if let Ok(val) = x.extract::<f64>() {
        return Ok(val.asinh().into_py_any(py).unwrap());
    }
    if let Ok(val) = x.extract::<Complex64>() {
        return Ok(val.asinh().into_py_any(py).unwrap());
    }
    if let Ok(arr) = x.cast::<PyArray<f64, IxDyn>>() {
        let result = arr.try_readonly()?.as_array().asinh();
        return Ok(PyArray::from_owned_array(py, result).into_any().unbind());
    }
    if let Ok(arr) = x.cast::<PyArray<Complex64, IxDyn>>() {
        let result = arr.try_readonly()?.as_array().asinh();
        return Ok(PyArray::from_owned_array(py, result).into_any().unbind());
    }
    Err(PyTypeError::new_err(
        "asinh() argument must be float, complex, or array",
    ))
}

/// Calculate inverse hyperbolic cosine.
///
/// Drop-in replacement for `numpy.arccosh()` / `numpy.acosh()`.
/// Handles scalars (float, complex) and arrays automatically.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn acosh(py: Python<'_>, x: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    use pecos::prelude::Acosh;

    if let Ok(val) = x.extract::<f64>() {
        return Ok(val.acosh().into_py_any(py).unwrap());
    }
    if let Ok(val) = x.extract::<Complex64>() {
        return Ok(val.acosh().into_py_any(py).unwrap());
    }
    if let Ok(arr) = x.cast::<PyArray<f64, IxDyn>>() {
        let result = arr.try_readonly()?.as_array().acosh();
        return Ok(PyArray::from_owned_array(py, result).into_any().unbind());
    }
    if let Ok(arr) = x.cast::<PyArray<Complex64, IxDyn>>() {
        let result = arr.try_readonly()?.as_array().acosh();
        return Ok(PyArray::from_owned_array(py, result).into_any().unbind());
    }
    Err(PyTypeError::new_err(
        "acosh() argument must be float, complex, or array",
    ))
}

/// Calculate inverse hyperbolic tangent.
///
/// Drop-in replacement for `numpy.arctanh()` / `numpy.atanh()`.
/// Handles scalars (float, complex) and arrays automatically.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn atanh(py: Python<'_>, x: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    use pecos::prelude::Atanh;

    if let Ok(val) = x.extract::<f64>() {
        return Ok(val.atanh().into_py_any(py).unwrap());
    }
    if let Ok(val) = x.extract::<Complex64>() {
        return Ok(val.atanh().into_py_any(py).unwrap());
    }
    if let Ok(arr) = x.cast::<PyArray<f64, IxDyn>>() {
        let result = arr.try_readonly()?.as_array().atanh();
        return Ok(PyArray::from_owned_array(py, result).into_any().unbind());
    }
    if let Ok(arr) = x.cast::<PyArray<Complex64, IxDyn>>() {
        let result = arr.try_readonly()?.as_array().atanh();
        return Ok(PyArray::from_owned_array(py, result).into_any().unbind());
    }
    Err(PyTypeError::new_err(
        "atanh() argument must be float, complex, or array",
    ))
}

/// Calculate arctangent of y/x with correct quadrant handling.
///
/// Drop-in replacement for `numpy.arctan2()` / `numpy.atan2()`.
/// Handles scalars and arrays.
///
/// Returns the angle in radians between the positive x-axis and the point (x, y).
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn atan2(py: Python<'_>, y: Bound<'_, PyAny>, x: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    use pecos::prelude::Atan2;

    // Scalar-scalar case: f64, f64 -> f64
    if let (Ok(y_val), Ok(x_val)) = (y.extract::<f64>(), x.extract::<f64>()) {
        return Ok(y_val.atan2(x_val).into_py_any(py).unwrap());
    }

    // Scalar-scalar case: Complex64, Complex64 -> Complex64
    if let (Ok(y_val), Ok(x_val)) = (y.extract::<Complex64>(), x.extract::<Complex64>()) {
        return Ok(y_val.atan2(x_val).into_py_any(py).unwrap());
    }

    // Array-scalar case: f64 array, f64 scalar -> f64 array
    if let (Ok(y_arr), Ok(x_val)) = (y.cast::<PyArray<f64, IxDyn>>(), x.extract::<f64>()) {
        let result = y_arr.try_readonly()?.as_array().atan2(x_val);
        return Ok(PyArray::from_owned_array(py, result).into_any().unbind());
    }

    // Array-scalar case: Complex64 array, Complex64 scalar -> Complex64 array
    if let (Ok(y_arr), Ok(x_val)) = (
        y.cast::<PyArray<Complex64, IxDyn>>(),
        x.extract::<Complex64>(),
    ) {
        let result = y_arr.try_readonly()?.as_array().atan2(x_val);
        return Ok(PyArray::from_owned_array(py, result).into_any().unbind());
    }

    Err(PyTypeError::new_err(
        "atan2() arguments must be (float, float), (complex, complex), or (array, scalar)",
    ))
}

/// Calculate absolute value.
///
/// Drop-in replacement for `numpy.abs()`.
/// Handles scalars (float, complex) and arrays automatically.
/// For complex numbers, returns the magnitude (modulus).
#[pyfunction]
#[allow(clippy::needless_pass_by_value)] // Bound is designed to be passed by value (PyO3 convention)
fn abs(py: Python<'_>, x: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    // Import trait to enable .abs() method
    #[allow(unused_imports)]
    use pecos::prelude::Abs;

    // Try f64 array first (includes numpy float scalars which are 0-dim arrays)
    if let Ok(arr) = x.cast::<PyArray<f64, IxDyn>>() {
        let result = arr.try_readonly()?.as_array().abs();
        // If it's a 0-dimensional array (numpy scalar), extract the single value
        if result.ndim() == 0
            && let Some(&val) = result.first()
        {
            return Ok(val.into_py_any(py).unwrap());
        }
        return Ok(PyArray::from_owned_array(py, result).into_any().unbind());
    }
    // Try Complex64 array (includes numpy complex scalars which are 0-dim arrays)
    if let Ok(arr) = x.cast::<PyArray<Complex64, IxDyn>>() {
        let result = arr.try_readonly()?.as_array().abs();
        // If it's a 0-dimensional array (numpy scalar), extract the single value
        if result.ndim() == 0
            && let Some(&val) = result.first()
        {
            return Ok(val.into_py_any(py).unwrap());
        }
        return Ok(PyArray::from_owned_array(py, result).into_any().unbind());
    }

    // For numpy scalars that couldn't be cast above (e.g., np.complex128 when Complex64 cast fails),
    // try using Python's abs() built-in which will call __abs__()
    if x.hasattr("__abs__")? && x.hasattr("dtype")? {
        // This is likely a numpy scalar - use Python's abs()
        if let Ok(builtins) = py.import("builtins")
            && let Ok(abs_fn) = builtins.getattr("abs")
            && let Ok(result) = abs_fn.call1((&x,))
        {
            return Ok(result.unbind());
        }
    }

    // Try f64 scalar (pure Python float)
    if let Ok(val) = x.extract::<f64>() {
        return Ok(val.abs().into_py_any(py).unwrap());
    }

    // Try Complex64 scalar (pure Python complex)
    // First attempt direct extraction
    if let Ok(val) = x.extract::<Complex64>() {
        return Ok(val.abs().into_py_any(py).unwrap());
    }

    // For numpy scalars (np.complex128, etc.), we need to convert to Python complex first
    // by calling the `complex()` built-in, which will use __complex__()
    if let Ok(builtins) = py.import("builtins")
        && let Ok(complex_fn) = builtins.getattr("complex")
        && let Ok(py_complex) = complex_fn.call1((&x,))
        && let Ok(val) = py_complex.extract::<Complex64>()
    {
        return Ok(val.abs().into_py_any(py).unwrap());
    }

    // Try Array type (our custom array wrapper)
    if let Ok(arr) = x.extract::<Py<Array>>() {
        use crate::pecos_array::ArrayData;
        let arr_ref = arr.bind(py).borrow();
        match &arr_ref.data {
            ArrayData::Bool(_) => {
                return Err(PyTypeError::new_err(
                    "abs() operation not supported on boolean arrays",
                ));
            }
            // Float types -> use Abs trait (returns f64/f32 arrays)
            ArrayData::Float64(a) => {
                let result = a.abs(); // Uses Abs trait
                return Ok(Py::new(py, Array::from_array_f64(result))?.into_any());
            }
            ArrayData::Float32(a) => {
                // abs() returns Array<f32, D>, convert to f64
                let result = a.mapv(|v| f64::from(v.abs()));
                return Ok(Py::new(py, Array::from_array_f64(result))?.into_any());
            }
            // Integer types -> use stdlib abs() for each element
            ArrayData::Int64(a) => {
                let result = a.mapv(i64::abs);
                return Ok(Py::new(py, Array::from_array_i64(result))?.into_any());
            }
            ArrayData::Int32(a) => {
                let result = a.mapv(|v| i64::from(v.abs()));
                return Ok(Py::new(py, Array::from_array_i64(result))?.into_any());
            }
            ArrayData::Int16(a) => {
                let result = a.mapv(|v| i64::from(v.abs()));
                return Ok(Py::new(py, Array::from_array_i64(result))?.into_any());
            }
            ArrayData::Int8(a) => {
                let result = a.mapv(|v| i64::from(v.abs()));
                return Ok(Py::new(py, Array::from_array_i64(result))?.into_any());
            }
            // Complex types -> use Abs trait (returns f64/f32 magnitudes)
            ArrayData::Complex128(a) => {
                let result = a.abs(); // Uses Abs trait, returns Array<f64, D>
                return Ok(Py::new(py, Array::from_array_f64(result))?.into_any());
            }
            ArrayData::Complex64(a) => {
                // abs() returns Array<f32, D>, convert to f64
                let result = a.mapv(|v| f64::from(v.norm()));
                return Ok(Py::new(py, Array::from_array_f64(result))?.into_any());
            }
        }
    }

    Err(PyTypeError::new_err(
        "abs() argument must be float, complex, or array",
    ))
}

/// Conditional selection: return x if condition is True, otherwise return y (scalar version).
///
/// Drop-in replacement for numpy.where(condition, x, y) for scalar conditions.
/// This is a simple ternary operator: `x if condition else y`
///
/// # Arguments
///
/// * `condition` - Boolean condition
/// * `x` - Value to return if condition is True
/// * `y` - Value to return if condition is False
///
/// # Returns
///
/// Returns x if condition is True, otherwise returns y
///
/// # Examples
///
/// ```python
/// from pecos_rslib.num import where
///
/// # Simple scalar usage
/// result = where(True, 10.0, 20.0)  # Returns 10.0
/// result = where(False, 10.0, 20.0)  # Returns 20.0
///
/// # Conditional computation (avoids computing both branches)
/// dist = 5
/// result = where(bool(dist % 2), dist * 2.0, dist / 2.0)  # Returns 10.0
/// ```
#[pyfunction]
fn where_(condition: bool, x: f64, y: f64) -> f64 {
    pecos::prelude::where_(condition, x, y)
}

/// Conditional selection with full broadcasting support.
///
/// Drop-in replacement for numpy.where(condition, x, y) with full broadcasting.
/// Handles all combinations of scalars and arrays for condition, x, and y parameters.
///
/// # Arguments
///
/// * `condition` - Boolean scalar or array determining which values to select
/// * `x` - Scalar or array of values to select when condition is True
/// * `y` - Scalar or array of values to select when condition is False
///
/// # Returns
///
/// Scalar if all inputs are scalars, otherwise array with broadcasting applied
///
/// # Examples
///
/// ```python
/// import numpy as np
/// from pecos_rslib.num import where_array
///
/// # All arrays, same shape
/// condition = np.array([True, False, True, False])
/// x = np.array([10.0, 20.0, 30.0, 40.0])
/// y = np.array([100.0, 200.0, 300.0, 400.0])
/// result = where_array(condition, x, y)
/// # Returns: array([10.0, 200.0, 30.0, 400.0])
///
/// # Scalar condition, array values (broadcasting)
/// result = where_array(True, np.array([1.0, 2.0, 3.0]), np.array([10.0, 20.0, 30.0]))
/// # Returns: array([1.0, 2.0, 3.0])
///
/// # Array condition, scalar values (broadcasting)
/// result = where_array(np.array([True, False, True]), 100.0, -100.0)
/// # Returns: array([100.0, -100.0, 100.0])
/// ```
#[pyfunction]
fn where_array<'py>(
    py: Python<'py>,
    condition: &Bound<'py, PyAny>,
    x: &Bound<'py, PyAny>,
    y: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    use numpy::ndarray::{Array, ArrayD, IxDyn};
    use pecos::prelude::Where;
    use pyo3::conversion::IntoPyObjectExt;

    // Helper to convert PyAny to either scalar or dynamic array
    fn to_array_or_scalar(obj: &Bound<'_, PyAny>) -> PyResult<ArrayD<f64>> {
        // Try to extract as scalar first
        if let Ok(scalar) = obj.extract::<f64>() {
            // Return 0-dimensional array
            return Ok(Array::from_elem(IxDyn(&[]), scalar));
        }

        // Try as PyArray with dynamic dimensions
        if let Ok(arr) = obj.extract::<PyReadonlyArrayDyn<f64>>() {
            return Ok(arr.as_array().to_owned());
        }

        // Convert via numpy asarray
        let py = obj.py();
        let np = py.import("numpy")?;
        let asarray = np.getattr("asarray")?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("dtype", "float64")?;
        let converted = asarray.call((obj,), Some(&kwargs))?;
        let arr = converted.extract::<PyReadonlyArrayDyn<f64>>()?;
        Ok(arr.as_array().to_owned())
    }

    fn to_bool_array_or_scalar(obj: &Bound<'_, PyAny>) -> PyResult<ArrayD<bool>> {
        // Try to extract as scalar bool first
        if let Ok(scalar) = obj.extract::<bool>() {
            return Ok(Array::from_elem(IxDyn(&[]), scalar));
        }

        // Try as PyArray with dynamic dimensions
        if let Ok(arr) = obj.extract::<PyReadonlyArrayDyn<bool>>() {
            return Ok(arr.as_array().to_owned());
        }

        // Convert via numpy asarray
        let py = obj.py();
        let np = py.import("numpy")?;
        let asarray = np.getattr("asarray")?;
        let converted = asarray.call1((obj,))?;
        let arr = converted.extract::<PyReadonlyArrayDyn<bool>>()?;
        Ok(arr.as_array().to_owned())
    }

    // Convert inputs to arrays (0-dim for scalars)
    let cond_arr = to_bool_array_or_scalar(condition)?;
    let x_arr = to_array_or_scalar(x)?;
    let y_arr = to_array_or_scalar(y)?;

    // All scalars case (all 0-dimensional)
    if cond_arr.ndim() == 0 && x_arr.ndim() == 0 && y_arr.ndim() == 0 {
        let cond_scalar = cond_arr[[]];
        let x_scalar = x_arr[[]];
        let y_scalar = y_arr[[]];
        let result = cond_scalar.where_(&x_scalar, &y_scalar);
        return result.into_py_any(py);
    }

    // Need to broadcast - determine output shape
    let shapes = vec![cond_arr.shape(), x_arr.shape(), y_arr.shape()];
    let result_shape = broadcast_shapes(&shapes)?;

    // Broadcast each array to result shape
    let cond_broadcast = broadcast_to(cond_arr.view(), &result_shape)?;
    let x_broadcast = broadcast_to(x_arr.view(), &result_shape)?;
    let y_broadcast = broadcast_to(y_arr.view(), &result_shape)?;

    // Apply where operation element-wise
    let result = cond_broadcast.where_(&x_broadcast, &y_broadcast);

    // Convert to Python array
    Ok(result.into_pyarray(py).into())
}

// Helper function to compute broadcast shape
fn broadcast_shapes(shapes: &[&[usize]]) -> PyResult<Vec<usize>> {
    let max_ndim = shapes.iter().map(|s| s.len()).max().unwrap_or(0);
    let mut result_shape = vec![1; max_ndim];

    for shape in shapes {
        // Align to the right (numpy broadcasting rule)
        let offset = max_ndim - shape.len();
        for (i, &dim) in shape.iter().enumerate() {
            let result_idx = offset + i;
            if result_shape[result_idx] == 1 {
                result_shape[result_idx] = dim;
            } else if dim != 1 && dim != result_shape[result_idx] {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "operands could not be broadcast together with shapes {shapes:?}"
                )));
            }
        }
    }

    Ok(result_shape)
}

// Helper function to broadcast array to target shape
#[allow(clippy::needless_pass_by_value)] // ArrayViewD is designed to be passed by value
fn broadcast_to<T: Clone>(
    arr: numpy::ndarray::ArrayViewD<'_, T>,
    target_shape: &[usize],
) -> PyResult<ArrayD<T>> {
    use numpy::ndarray::IxDyn;

    // If already the right shape, return owned copy
    if arr.shape() == target_shape {
        return Ok(arr.to_owned());
    }

    // Use ndarray's broadcast functionality
    let broadcast_view = arr.broadcast(IxDyn(target_shape)).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "cannot broadcast shape {:?} to {:?}",
            arr.shape(),
            target_shape
        ))
    })?;

    Ok(broadcast_view.to_owned())
}

/// Register the num submodule with Python bindings.
#[allow(clippy::too_many_lines)] // Registration function naturally has many lines
pub fn register_num_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let num_module = PyModule::new(m.py(), "num")?;

    // Create stats submodule
    let stats_module = PyModule::new(m.py(), "stats")?;
    stats_module.add_function(wrap_pyfunction!(mean, &stats_module)?)?;
    stats_module.add_function(wrap_pyfunction!(self::std, &stats_module)?)?;
    num_module.add_submodule(&stats_module)?;

    // Create math submodule
    let math_module = PyModule::new(m.py(), "math")?;

    // Math functions (polymorphic - handle scalars, complex, and arrays automatically)
    math_module.add_function(wrap_pyfunction!(exp, &math_module)?)?;
    math_module.add_function(wrap_pyfunction!(ln, &math_module)?)?;
    math_module.add_function(wrap_pyfunction!(self::log, &math_module)?)?;
    math_module.add_function(wrap_pyfunction!(sqrt, &math_module)?)?;
    math_module.add_function(wrap_pyfunction!(power, &math_module)?)?;
    math_module.add_function(wrap_pyfunction!(cos, &math_module)?)?;
    math_module.add_function(wrap_pyfunction!(sin, &math_module)?)?;
    math_module.add_function(wrap_pyfunction!(tan, &math_module)?)?;
    math_module.add_function(wrap_pyfunction!(sinh, &math_module)?)?;
    math_module.add_function(wrap_pyfunction!(cosh, &math_module)?)?;
    math_module.add_function(wrap_pyfunction!(tanh, &math_module)?)?;
    math_module.add_function(wrap_pyfunction!(asin, &math_module)?)?;
    math_module.add_function(wrap_pyfunction!(acos, &math_module)?)?;
    math_module.add_function(wrap_pyfunction!(atan, &math_module)?)?;
    math_module.add_function(wrap_pyfunction!(asinh, &math_module)?)?;
    math_module.add_function(wrap_pyfunction!(acosh, &math_module)?)?;
    math_module.add_function(wrap_pyfunction!(atanh, &math_module)?)?;
    math_module.add_function(wrap_pyfunction!(atan2, &math_module)?)?;
    math_module.add_function(wrap_pyfunction!(abs, &math_module)?)?;

    // Scalar-only functions
    math_module.add_function(wrap_pyfunction!(floor, &math_module)?)?;
    math_module.add_function(wrap_pyfunction!(ceil, &math_module)?)?;
    math_module.add_function(wrap_pyfunction!(round, &math_module)?)?;

    // Add mathematical constants to math submodule
    math_module.add("pi", pecos::prelude::PI)?;
    math_module.add("tau", pecos::prelude::TAU)?;
    math_module.add("e", pecos::prelude::E)?;
    math_module.add("inf", f64::INFINITY)?;
    math_module.add("nan", f64::NAN)?;
    math_module.add("FRAC_PI_2", pecos::prelude::FRAC_PI_2)?;
    math_module.add("FRAC_PI_3", pecos::prelude::FRAC_PI_3)?;
    math_module.add("FRAC_PI_4", pecos::prelude::FRAC_PI_4)?;
    math_module.add("FRAC_PI_6", pecos::prelude::FRAC_PI_6)?;
    math_module.add("FRAC_PI_8", pecos::prelude::FRAC_PI_8)?;
    math_module.add("FRAC_1_PI", pecos::prelude::FRAC_1_PI)?;
    math_module.add("FRAC_2_PI", pecos::prelude::FRAC_2_PI)?;
    math_module.add("FRAC_2_SQRT_PI", pecos::prelude::FRAC_2_SQRT_PI)?;
    math_module.add("SQRT_2", pecos::prelude::SQRT_2)?;
    math_module.add("FRAC_1_SQRT_2", pecos::prelude::FRAC_1_SQRT_2)?;
    math_module.add("LN_2", pecos::prelude::LN_2)?;
    math_module.add("LN_10", pecos::prelude::LN_10)?;
    math_module.add("LOG2_E", pecos::prelude::LOG2_E)?;
    math_module.add("LOG10_E", pecos::prelude::LOG10_E)?;
    num_module.add_submodule(&math_module)?;

    // Create compare submodule
    let compare_module = PyModule::new(m.py(), "compare")?;
    compare_module.add_function(wrap_pyfunction!(isnan, &compare_module)?)?;
    compare_module.add_function(wrap_pyfunction!(isclose, &compare_module)?)?;
    compare_module.add_function(wrap_pyfunction!(allclose, &compare_module)?)?;
    compare_module.add_function(wrap_pyfunction!(array_equal, &compare_module)?)?;
    compare_module.add_function(wrap_pyfunction!(all, &compare_module)?)?;
    compare_module.add_function(wrap_pyfunction!(any, &compare_module)?)?;
    compare_module.add_function(wrap_pyfunction!(where_, &compare_module)?)?;
    compare_module.add_function(wrap_pyfunction!(where_array, &compare_module)?)?;
    // Old separate functions removed - now using polymorphic isnan/isclose
    num_module.add_submodule(&compare_module)?;

    // Create array submodule
    let array_module = PyModule::new(m.py(), "array")?;
    array_module.add_function(wrap_pyfunction!(diag, &array_module)?)?;
    array_module.add_function(wrap_pyfunction!(linspace, &array_module)?)?;
    array_module.add_function(wrap_pyfunction!(arange, &array_module)?)?;
    array_module.add_function(wrap_pyfunction!(zeros, &array_module)?)?;
    array_module.add_function(wrap_pyfunction!(ones, &array_module)?)?;
    array_module.add_function(wrap_pyfunction!(delete, &array_module)?)?;
    array_module.add_function(wrap_pyfunction!(sum, &array_module)?)?;
    num_module.add_submodule(&array_module)?;

    // Create optimize submodule
    let optimize_module = PyModule::new(m.py(), "optimize")?;
    optimize_module.add_function(wrap_pyfunction!(brentq, &optimize_module)?)?;
    optimize_module.add_function(wrap_pyfunction!(newton, &optimize_module)?)?;
    num_module.add_submodule(&optimize_module)?;

    // Create polynomial submodule
    let polynomial_module = PyModule::new(m.py(), "polynomial")?;
    polynomial_module.add_function(wrap_pyfunction!(polyfit, &polynomial_module)?)?;
    polynomial_module.add_class::<Poly1d>()?;
    num_module.add_submodule(&polynomial_module)?;

    // Create curve_fit submodule
    let curve_fit_module = PyModule::new(m.py(), "curve_fit")?;
    curve_fit_module.add_function(wrap_pyfunction!(curve_fit, &curve_fit_module)?)?;
    num_module.add_submodule(&curve_fit_module)?;

    // Create linalg submodule
    let linalg_module = PyModule::new(m.py(), "linalg")?;
    linalg_module.add_function(wrap_pyfunction!(norm, &linalg_module)?)?;
    num_module.add_submodule(&linalg_module)?;

    // Create random submodule
    let random_module = PyModule::new(m.py(), "random")?;
    random_module.add_function(wrap_pyfunction!(seed, &random_module)?)?;
    random_module.add_function(wrap_pyfunction!(random, &random_module)?)?;
    random_module.add_function(wrap_pyfunction!(randint, &random_module)?)?;
    random_module.add_function(wrap_pyfunction!(choice, &random_module)?)?;
    random_module.add_function(wrap_pyfunction!(compare_any, &random_module)?)?;
    random_module.add_function(wrap_pyfunction!(compare_indices, &random_module)?)?;
    num_module.add_submodule(&random_module)?;

    // Expose all functions at the top level
    // Stats functions
    num_module.add_function(wrap_pyfunction!(mean, &num_module)?)?;
    num_module.add_function(wrap_pyfunction!(self::std, &num_module)?)?;

    // Math functions (polymorphic - handle scalars, complex, and arrays automatically)
    num_module.add_function(wrap_pyfunction!(exp, &num_module)?)?;
    num_module.add_function(wrap_pyfunction!(sqrt, &num_module)?)?;
    num_module.add_function(wrap_pyfunction!(power, &num_module)?)?;
    num_module.add_function(wrap_pyfunction!(cos, &num_module)?)?;
    num_module.add_function(wrap_pyfunction!(sin, &num_module)?)?;
    num_module.add_function(wrap_pyfunction!(tan, &num_module)?)?;
    num_module.add_function(wrap_pyfunction!(sinh, &num_module)?)?;
    num_module.add_function(wrap_pyfunction!(cosh, &num_module)?)?;
    num_module.add_function(wrap_pyfunction!(tanh, &num_module)?)?;
    num_module.add_function(wrap_pyfunction!(asin, &num_module)?)?;
    num_module.add_function(wrap_pyfunction!(acos, &num_module)?)?;
    num_module.add_function(wrap_pyfunction!(atan, &num_module)?)?;
    num_module.add_function(wrap_pyfunction!(asinh, &num_module)?)?;
    num_module.add_function(wrap_pyfunction!(acosh, &num_module)?)?;
    num_module.add_function(wrap_pyfunction!(atanh, &num_module)?)?;
    num_module.add_function(wrap_pyfunction!(atan2, &num_module)?)?;

    // Scalar-only math functions
    num_module.add_function(wrap_pyfunction!(floor, &num_module)?)?;
    num_module.add_function(wrap_pyfunction!(ceil, &num_module)?)?;
    num_module.add_function(wrap_pyfunction!(round, &num_module)?)?;

    // Comparison functions (polymorphic)
    num_module.add_function(wrap_pyfunction!(isnan, &num_module)?)?;
    num_module.add_function(wrap_pyfunction!(isclose, &num_module)?)?;
    num_module.add_function(wrap_pyfunction!(array_equal, &num_module)?)?;
    num_module.add_function(wrap_pyfunction!(where_, &num_module)?)?;

    // Array functions (polymorphic)
    num_module.add_function(wrap_pyfunction!(sum, &num_module)?)?;
    num_module.add_function(wrap_pyfunction!(diag, &num_module)?)?;
    num_module.add_function(wrap_pyfunction!(linspace, &num_module)?)?;
    num_module.add_function(wrap_pyfunction!(arange, &num_module)?)?;
    num_module.add_function(wrap_pyfunction!(zeros, &num_module)?)?;
    num_module.add_function(wrap_pyfunction!(ones, &num_module)?)?;
    num_module.add_function(wrap_pyfunction!(array, &num_module)?)?;
    num_module.add_function(wrap_pyfunction!(delete, &num_module)?)?;

    // Optimization functions
    num_module.add_function(wrap_pyfunction!(brentq, &num_module)?)?;
    num_module.add_function(wrap_pyfunction!(newton, &num_module)?)?;

    // Polynomial functions
    num_module.add_function(wrap_pyfunction!(polyfit, &num_module)?)?;
    num_module.add_class::<Poly1d>()?;

    // Curve fitting
    num_module.add_function(wrap_pyfunction!(curve_fit, &num_module)?)?;

    // Also expose constants at top level
    num_module.add("pi", pecos::prelude::PI)?;
    num_module.add("tau", pecos::prelude::TAU)?;
    num_module.add("e", pecos::prelude::E)?;
    num_module.add("inf", f64::INFINITY)?;
    num_module.add("nan", f64::NAN)?;
    num_module.add("FRAC_PI_2", pecos::prelude::FRAC_PI_2)?;
    num_module.add("FRAC_PI_3", pecos::prelude::FRAC_PI_3)?;
    num_module.add("FRAC_PI_4", pecos::prelude::FRAC_PI_4)?;
    num_module.add("FRAC_PI_6", pecos::prelude::FRAC_PI_6)?;
    num_module.add("FRAC_PI_8", pecos::prelude::FRAC_PI_8)?;
    num_module.add("FRAC_1_PI", pecos::prelude::FRAC_1_PI)?;
    num_module.add("FRAC_2_PI", pecos::prelude::FRAC_2_PI)?;
    num_module.add("FRAC_2_SQRT_PI", pecos::prelude::FRAC_2_SQRT_PI)?;
    num_module.add("SQRT_2", pecos::prelude::SQRT_2)?;
    num_module.add("FRAC_1_SQRT_2", pecos::prelude::FRAC_1_SQRT_2)?;
    num_module.add("LN_2", pecos::prelude::LN_2)?;
    num_module.add("LN_10", pecos::prelude::LN_10)?;
    num_module.add("LOG2_E", pecos::prelude::LOG2_E)?;
    num_module.add("LOG10_E", pecos::prelude::LOG10_E)?;

    m.add_submodule(&num_module)?;
    Ok(())
}
