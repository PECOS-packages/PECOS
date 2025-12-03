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
#![allow(clippy::needless_question_mark)] // PyO3 error handling patterns
#![allow(clippy::redundant_closure_for_method_calls)] // Closures more readable for complex operations

use ndarray::{Array as NdArray, Array1, ArrayD, Axis, IxDyn};
use num_complex::Complex64;
// REMOVED: use numpy::{
//     IntoPyArray, PyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
// };
use pyo3::conversion::IntoPyObjectExt;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

// Import Array and ArrayData from pecos_array module for migration from numpy.ndarray to Array
use crate::pecos_array::{Array, ArrayData};

// Import array_buffer module for NumPy interop (replacing rust-numpy)
use crate::array_buffer;

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
///     >>> from `_pecos_rslib.num` import brentq
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
///     >>> from `_pecos_rslib.num` import newton
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
///     >>> from `_pecos_rslib.num` import polyfit
///     >>> import numpy as np
///     >>> # Fit y = 2x + 1
///     >>> x = np.array([0.0, 1.0, 2.0, 3.0])
///     >>> y = np.array([1.0, 3.0, 5.0, 7.0])
///     >>> coeffs = polyfit(x, y, 1)
///     >>> # coeffs ≈ [2.0, 1.0] (slope, intercept)
#[pyfunction]
#[pyo3(signature = (x, y, deg, cov=None))]
fn polyfit(
    py: Python<'_>,
    x: Bound<'_, PyAny>,
    y: Bound<'_, PyAny>,
    deg: usize,
    cov: Option<bool>,
) -> PyResult<Py<PyAny>> {
    let x_array = array_buffer::extract_f64_array(&x)?;
    let y_array = array_buffer::extract_f64_array(&y)?;

    // Convert to 1D arrays (polyfit expects 1D)
    let x_view = x_array
        .view()
        .into_dimensionality::<ndarray::Ix1>()
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("x must be 1D array: {e}"))
        })?;
    let y_view = y_array
        .view()
        .into_dimensionality::<ndarray::Ix1>()
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("y must be 1D array: {e}"))
        })?;

    let return_cov = cov.unwrap_or(false);

    if return_cov {
        // Call polyfit_with_cov and return tuple (coeffs, cov_matrix)
        let (coeffs, cov_matrix) =
            pecos::prelude::polyfit_with_cov(x_view, y_view, deg).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("polyfit failed: {e}"))
            })?;

        let coeffs_py = Py::new(py, Array::from_array_f64(coeffs.into_dyn()))?;
        let cov_py = Py::new(py, Array::from_array_f64(cov_matrix.into_dyn()))?;

        let tuple_items: Vec<Py<PyAny>> = vec![coeffs_py.into_any(), cov_py.into_any()];
        Ok(PyTuple::new(py, &tuple_items)?.into())
    } else {
        // Call regular polyfit and return just coefficients
        let coeffs = pecos::prelude::polyfit(x_view, y_view, deg).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("polyfit failed: {e}"))
        })?;

        Ok(Py::new(py, Array::from_array_f64(coeffs.into_dyn()))?.into_any())
    }
}

/// Polynomial class for evaluation.
///
/// This is a drop-in replacement for numpy.poly1d.
///
/// Examples:
///     >>> from `_pecos_rslib.num` import Poly1d
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
    fn new(coeffs: Bound<'_, PyAny>) -> PyResult<Self> {
        let coeffs_array = array_buffer::extract_f64_array(&coeffs)?;
        // Convert to 1D array (Poly1d expects 1D)
        let coeffs_1d = coeffs_array
            .into_dimensionality::<ndarray::Ix1>()
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "coeffs must be 1D array: {e}"
                ))
            })?;
        Ok(Self {
            inner: RustPoly1d::new(coeffs_1d),
        })
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
    fn coefficients(&self, py: Python<'_>) -> Py<crate::array_buffer::F64ArrayView> {
        array_buffer::f64_array_to_py(py, self.inner.coefficients())
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
///     >>> from `_pecos_rslib.num` import `curve_fit`
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
fn curve_fit<'py>(
    py: Python<'py>,
    f: Py<PyAny>,
    xdata: &Bound<'py, PyAny>,
    ydata: &Bound<'py, PyAny>,
    p0: &Bound<'py, PyAny>,
    maxfev: Option<usize>,
    xtol: Option<f64>,
    ftol: Option<f64>,
) -> PyResult<(
    Py<crate::array_buffer::F64ArrayView>,
    Py<crate::array_buffer::F64ArrayView>,
)> {
    // Convert ydata to ndarray - handle both NumPy arrays and PECOS Arrays
    let ydata_array = array_buffer::extract_f64_array(ydata)?;

    // Convert p0 to array (accept array, tuple, or list)
    let p0_array = if let Ok(list) = p0.extract::<Vec<f64>>() {
        ArrayD::from_shape_vec(IxDyn(&[list.len()]), list).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Failed to convert p0 to array: {e}"
            ))
        })?
    } else {
        array_buffer::extract_f64_array(p0)?
    };

    // Check if xdata is a tuple or a single array
    if let Ok(tuple) = xdata.cast() {
        // Handle tuple case (multiple independent variables)
        curve_fit_tuple(py, f, tuple, ydata_array, p0_array, maxfev, xtol, ftol)
    } else {
        // Handle single array case
        let xdata_array = array_buffer::extract_f64_array(xdata)?;
        curve_fit_array(
            py,
            f,
            xdata_array,
            ydata_array,
            p0_array,
            maxfev,
            xtol,
            ftol,
        )
    }
}

/// Helper function for `curve_fit` with single array xdata.
#[allow(clippy::type_complexity)] // Complex return type required for scipy compatibility
#[allow(clippy::too_many_arguments)] // Matches scipy.optimize.curve_fit parameters
fn curve_fit_array(
    py: Python<'_>,
    f: Py<PyAny>,
    xdata: ArrayD<f64>,
    ydata: ArrayD<f64>,
    p0: ArrayD<f64>,
    maxfev: Option<usize>,
    xtol: Option<f64>,
    ftol: Option<f64>,
) -> PyResult<(
    Py<crate::array_buffer::F64ArrayView>,
    Py<crate::array_buffer::F64ArrayView>,
)> {
    // Convert to 1D arrays (curve_fit expects 1D)
    let xdata_view = xdata
        .view()
        .into_dimensionality::<ndarray::Ix1>()
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("xdata must be 1D array: {e}"))
        })?;
    let ydata_view = ydata
        .view()
        .into_dimensionality::<ndarray::Ix1>()
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("ydata must be 1D array: {e}"))
        })?;
    let p0_view = p0
        .view()
        .into_dimensionality::<ndarray::Ix1>()
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("p0 must be 1D array: {e}"))
        })?;

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
    let popt = array_buffer::f64_array_to_py(py, &result.params);

    // If covariance is available, return it; otherwise create identity matrix
    let pcov = if let Some(cov) = result.pcov {
        array_buffer::f64_array_to_py(py, &cov)
    } else {
        // Return identity matrix if covariance not available
        let n = result.params.len();
        let identity = Array1::from_shape_fn(n * n, |i| if i / n == i % n { 1.0 } else { 0.0 })
            .into_shape_with_order((n, n))
            .unwrap()
            .into_dyn();
        array_buffer::f64_array_to_py(py, &identity)
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
fn curve_fit_tuple<'py>(
    py: Python<'py>,
    f: Py<PyAny>,
    xdata_tuple: &Bound<'py, PyTuple>,
    ydata: ArrayD<f64>,
    p0: ArrayD<f64>,
    maxfev: Option<usize>,
    xtol: Option<f64>,
    ftol: Option<f64>,
) -> PyResult<(
    Py<crate::array_buffer::F64ArrayView>,
    Py<crate::array_buffer::F64ArrayView>,
)> {
    // Extract arrays from tuple using ensure_f64_array for numpy-compatible conversion
    let mut xdata_arrays: Vec<Array1<f64>> = Vec::new();

    for (i, item) in xdata_tuple.iter().enumerate() {
        // Use ensure_f64_array for comprehensive type handling and good error messages
        let arr = array_buffer::ensure_f64_array(&item, &format!("xdata[{i}]"))?;

        // Convert to 1D if needed
        let arr_1d = if arr.ndim() == 1 {
            arr.into_dimensionality::<ndarray::Ix1>().unwrap()
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "xdata[{}] must be a 1D array, got {}D array with shape {:?}",
                i,
                arr.ndim(),
                arr.shape()
            )));
        };
        xdata_arrays.push(arr_1d);
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

    // Convert to 1D array (curve_fit expects 1D)
    let ydata_view = ydata
        .view()
        .into_dimensionality::<ndarray::Ix1>()
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("ydata must be 1D array: {e}"))
        })?;
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

    // Convert to 1D array (curve_fit expects 1D)
    let p0_view = p0
        .view()
        .into_dimensionality::<ndarray::Ix1>()
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("p0 must be 1D array: {e}"))
        })?;

    // Call Rust implementation with index-based xdata
    let result =
        pecos::prelude::curve_fit(func, xdata_indices.view(), ydata_view, p0_view, Some(opts))
            .map_err(map_curve_fit_error)?;

    // Convert results to Python arrays
    let popt = array_buffer::f64_array_to_py(py, &result.params);

    // If covariance is available, return it; otherwise create identity matrix
    let pcov = if let Some(cov) = result.pcov {
        array_buffer::f64_array_to_py(py, &cov)
    } else {
        // Return identity matrix if covariance not available
        let n = result.params.len();
        let identity = Array1::from_shape_fn(n * n, |i| if i / n == i % n { 1.0 } else { 0.0 })
            .into_shape_with_order((n, n))
            .unwrap()
            .into_dyn();
        array_buffer::f64_array_to_py(py, &identity)
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
///     >>> from `_pecos_rslib.num.random` import random
///     >>> values = random(5)
///     >>> len(values)
///     5
#[pyfunction]
fn random(py: Python<'_>, size: usize) -> PyResult<Py<Array>> {
    let result = pecos::prelude::random::random(size);
    Ok(Py::new(py, Array::from_array_f64(result.into_dyn()))?)
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
///     >>> from `_pecos_rslib.num.random` import randint
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
fn randint(
    py: Python<'_>,
    low: i64,
    high: Option<i64>,
    size: Option<usize>,
) -> PyResult<Py<PyAny>> {
    use pyo3::IntoPyObject;

    if let Some(n) = size {
        // Return array
        // Match NumPy's platform-dependent dtype behavior:
        // - Windows: int32 (C long is 32-bit on Windows even on 64-bit systems)
        // - Unix: int64 (C long is 64-bit on 64-bit Unix systems)
        #[cfg(target_os = "windows")]
        {
            // On Windows, check bounds to ensure values fit in i32
            let low_i32 = i32::try_from(low).map_err(|_| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "low value {low} out of range for int32"
                ))
            })?;
            let high_i32 = if let Some(h) = high {
                Some(i32::try_from(h).map_err(|_| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "high value {h} out of range for int32"
                    ))
                })?)
            } else {
                None
            };
            let result = pecos::prelude::random::randint(low_i32, high_i32, n);
            Ok(Py::new(py, Array::from_array_i32(result.into_dyn()))?.into_any())
        }
        #[cfg(not(target_os = "windows"))]
        {
            let result = pecos::prelude::random::randint(low, high, n);
            Ok(Py::new(py, Array::from_array_i64(result.into_dyn()))?.into_any())
        }
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
///     >>> from `_pecos_rslib.num.random` import seed, random
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
///     >>> from __pecos_rslib.num.random import choice
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

        // First try to handle Array objects
        if let Ok(arr) = obj.cast::<crate::pecos_array::Array>() {
            let len = arr.len()?;
            let mut items = Vec::with_capacity(len);
            for i in 0..len {
                items.push(arr.get_item(i)?.unbind());
            }
            return Ok::<Vec<Py<PyAny>>, PyErr>(items);
        }

        // Next try to handle numpy arrays by converting to list
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
/// from __pecos_rslib.num import random
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
/// from __pecos_rslib.num import random
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
/// from __pecos_rslib.num import mean
///
/// # Calculate mean of a list
/// values = [1.0, 2.0, 3.0, 4.0, 5.0]
/// avg = mean(values)  # Returns 3.0
///
/// # Error model use case: average measurement error rates
/// p_meas = (0.01, 0.015, 0.02)
/// avg_p_meas = mean(p_meas)  # Returns 0.015
///
/// # 2D array - mean over all elements
/// arr = [[1.0, 2.0], [3.0, 4.0]]
/// mean(arr)  # Returns 2.5
///
/// # 2D array - mean along axis 0 (down columns)
/// mean(arr, axis=0)  # Returns [2.0, 3.0]
///
/// # 2D array - mean along axis 1 (across rows)
/// mean(arr, axis=1)  # Returns [1.5, 3.5]
/// ```
#[pyfunction]
#[pyo3(signature = (a, axis=None))]
fn mean(py: Python<'_>, a: &Bound<'_, PyAny>, axis: Option<isize>) -> PyResult<Py<PyAny>> {
    // Use ensure_f64_array which handles PECOS Arrays, numpy arrays, and Python sequences
    let array = array_buffer::ensure_f64_array(a, "a")?;

    match axis {
        None => {
            // No axis specified - compute mean of flattened array
            let flat: Vec<f64> = array.iter().copied().collect();
            if flat.is_empty() {
                return Ok(f64::NAN.into_pyobject(py)?.into_any().unbind());
            }
            let result = pecos::prelude::mean(&flat);
            Ok(result.into_pyobject(py)?.into_any().unbind())
        }
        Some(axis_val) => {
            // Axis specified - use mean_axis logic
            let ndim = array.ndim();

            // Convert negative axis to positive
            let axis_usize = if axis_val < 0 {
                let pos = (ndim as isize + axis_val) as usize;
                if pos >= ndim {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "axis {axis_val} is out of bounds for array of dimension {ndim}"
                    )));
                }
                pos
            } else {
                let axis_usize = axis_val as usize;
                if axis_usize >= ndim {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "axis {axis_val} is out of bounds for array of dimension {ndim}"
                    )));
                }
                axis_usize
            };

            // Call Rust implementation
            let result =
                pecos::prelude::mean_axis(&array.view(), Axis(axis_usize)).ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "mean_axis returned None - array may be empty along the specified axis",
                    )
                })?;

            // Convert back to Python Array
            Ok(Py::new(py, Array::from_array_f64(result))?.into_any())
        }
    }
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
///     >>> from `_pecos_rslib`._`pecos_rslib` import num
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
    if let Ok(arr) = array_buffer::extract_f64_array(&x) {
        let result = arr.isnan();
        return Ok(Py::new(py, Array::from_array_bool(result.to_owned().into_dyn()))?.into_any());
    }

    // Try complex array
    if let Ok(arr) = array_buffer::extract_complex64_array(&x) {
        let result = arr.isnan();
        return Ok(Py::new(py, Array::from_array_bool(result.to_owned().into_dyn()))?.into_any());
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
/// from __pecos_rslib.num import floor
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
/// from __pecos_rslib.num import ceil
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
/// from __pecos_rslib.num import round
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
/// from __pecos_rslib.num import isclose
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
    use crate::pecos_array::ArrayData;
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

    // Try to convert inputs to PECOS Arrays if they're not already
    // This handles NumPy arrays at the boundary by converting them to PECOS Arrays
    let a_pecos = if let Ok(arr) = a.extract::<Py<Array>>() {
        arr
    } else {
        // Call the Array Python class to create PECOS Array from NumPy array/list
        let array_class = py.get_type::<Array>();
        array_class.call1((&a,))?.extract()?
    };

    let b_pecos = if let Ok(arr) = b.extract::<Py<Array>>() {
        arr
    } else {
        // Call the Array Python class to create PECOS Array from NumPy array/list
        let array_class = py.get_type::<Array>();
        array_class.call1((&b,))?.extract()?
    };

    // Now work only with PECOS Arrays
    let a_ref = a_pecos.bind(py).borrow();
    let b_ref = b_pecos.bind(py).borrow();

    match (&a_ref.data, &b_ref.data) {
        (ArrayData::F64(a_data), ArrayData::F64(b_data)) => {
            let result = a_data.isclose(b_data, rtol, atol);
            return Ok(Py::new(
                py,
                Array {
                    data: ArrayData::Bool(result),
                },
            )?
            .into_any());
        }
        (ArrayData::Complex128(a_data), ArrayData::Complex128(b_data)) => {
            let result = a_data.isclose(b_data, rtol, atol);
            return Ok(Py::new(
                py,
                Array {
                    data: ArrayData::Bool(result),
                },
            )?
            .into_any());
        }
        (ArrayData::F64(a_data), ArrayData::Complex128(b_data)) => {
            // Convert float to complex
            let a_complex = a_data.mapv(|x| Complex64::new(x, 0.0));
            let result = a_complex.isclose(b_data, rtol, atol);
            return Ok(Py::new(
                py,
                Array {
                    data: ArrayData::Bool(result),
                },
            )?
            .into_any());
        }
        (ArrayData::Complex128(a_data), ArrayData::F64(b_data)) => {
            // Convert float to complex
            let b_complex = b_data.mapv(|x| Complex64::new(x, 0.0));
            let result = a_data.isclose(&b_complex, rtol, atol);
            return Ok(Py::new(
                py,
                Array {
                    data: ArrayData::Bool(result),
                },
            )?
            .into_any());
        }
        _ => {
            // Unsupported dtype combination
        }
    }

    Err(PyTypeError::new_err(
        "isclose() arguments must be float, complex, or PECOS Arrays of float/complex",
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
/// from _pecos_rslib import allclose
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
    use crate::pecos_array::ArrayData;
    use pecos::prelude::allclose as rust_allclose;

    // Try to convert inputs to PECOS Arrays if they're not already
    // This handles NumPy arrays, lists, etc. at the boundary by converting them to PECOS Arrays
    let a_pecos = if let Ok(arr) = a.extract::<Py<Array>>() {
        arr
    } else {
        // Call the Array Python class to create PECOS Array from NumPy array/list
        let array_class = a.py().get_type::<Array>();
        array_class.call1((&a,))?.extract()?
    };

    let b_pecos = if let Ok(arr) = b.extract::<Py<Array>>() {
        arr
    } else {
        // Call the Array Python class to create PECOS Array from NumPy array/list
        let array_class = b.py().get_type::<Array>();
        array_class.call1((&b,))?.extract()?
    };

    // Now work only with PECOS Arrays
    let a_ref = a_pecos.bind(a.py()).borrow();
    let b_ref = b_pecos.bind(b.py()).borrow();

    match (&a_ref.data, &b_ref.data) {
        (ArrayData::F64(a_data), ArrayData::F64(b_data)) => {
            return Ok(rust_allclose(a_data, b_data, rtol, atol, equal_nan));
        }
        (ArrayData::Complex128(a_data), ArrayData::Complex128(b_data)) => {
            return Ok(rust_allclose(a_data, b_data, rtol, atol, equal_nan));
        }
        (ArrayData::F64(a_data), ArrayData::Complex128(b_data)) => {
            // Convert float to complex
            let a_complex = a_data.mapv(|x| Complex64::new(x, 0.0));
            return Ok(rust_allclose(&a_complex, b_data, rtol, atol, equal_nan));
        }
        (ArrayData::Complex128(a_data), ArrayData::F64(b_data)) => {
            // Convert float to complex
            let b_complex = b_data.mapv(|x| Complex64::new(x, 0.0));
            return Ok(rust_allclose(a_data, &b_complex, rtol, atol, equal_nan));
        }
        _ => {
            // Unsupported dtype combination
        }
    }

    Err(PyTypeError::new_err(
        "allclose() arguments must be PECOS Arrays of compatible dtypes (float64, complex128)",
    ))
}

/// Assert that all elements in two arrays are close within specified tolerances.
///
/// Drop-in replacement for `numpy.testing.assert_allclose()`. Panics with a detailed
/// error message if any elements are not close according to the tolerance check:
/// `|a - b| <= (atol + rtol * |b|)`
///
/// # Arguments
///
/// * `a` - First input array
/// * `b` - Second input array
/// * `rtol` - Relative tolerance parameter (default: 1e-5)
/// * `atol` - Absolute tolerance parameter (default: 1e-8)
/// * `equal_nan` - If `true`, NaNs in the same position are considered equal (default: `false`)
///
/// # Panics
///
/// Panics with a detailed error message showing:
/// - Shape mismatch (if shapes differ)
/// - Number of mismatched elements
/// - Maximum absolute difference
/// - Maximum relative difference
/// - Location and values of first mismatch
///
/// # Examples
///
/// ```python
/// import pecos as pc
///
/// # These pass without error
/// a = pc.array([1.0, 2.0, 3.0])
/// b = pc.array([1.00001, 2.00001, 3.00001])
/// pc.assert_allclose(a, b, rtol=1e-4, atol=1e-8)
///
/// # This panics with detailed error message
/// c = pc.array([1.0, 2.0, 4.0])
/// try:
///     pc.assert_allclose(a, c, rtol=1e-5, atol=1e-8)
/// except AssertionError as e:
///     print(e)  # Shows mismatch details
/// ```
#[pyfunction]
#[pyo3(signature = (a, b, rtol=1e-5, atol=1e-8, equal_nan=false))]
#[allow(clippy::needless_pass_by_value)] // Bound is designed to be passed by value (PyO3 convention)
fn assert_allclose(
    a: Bound<'_, PyAny>,
    b: Bound<'_, PyAny>,
    rtol: f64,
    atol: f64,
    equal_nan: bool,
) -> PyResult<()> {
    use pecos::prelude::assert_allclose as rust_assert_allclose;

    // Try to convert inputs to PECOS Arrays if they're not already
    let a_pecos = if let Ok(arr) = a.extract::<Py<Array>>() {
        arr
    } else {
        let array_class = a.py().get_type::<Array>();
        array_class.call1((&a,))?.extract()?
    };

    let b_pecos = if let Ok(arr) = b.extract::<Py<Array>>() {
        arr
    } else {
        let array_class = b.py().get_type::<Array>();
        array_class.call1((&b,))?.extract()?
    };

    // Now work only with PECOS Arrays
    let a_ref = a_pecos.bind(a.py()).borrow();
    let b_ref = b_pecos.bind(b.py()).borrow();

    // assert_allclose panics on mismatch, so we catch the panic and convert to PyAssertionError
    let result = ::std::panic::catch_unwind(::std::panic::AssertUnwindSafe(|| {
        match (&a_ref.data, &b_ref.data) {
            (ArrayData::F64(a_data), ArrayData::F64(b_data)) => {
                rust_assert_allclose(a_data, b_data, rtol, atol, equal_nan);
            }
            (ArrayData::Complex128(a_data), ArrayData::Complex128(b_data)) => {
                // Convert complex to f64 magnitude for comparison
                // This requires special handling since assert_allclose expects f64
                // For now, we'll extract real and imaginary parts separately
                let a_real = a_data.mapv(|x| x.re);
                let b_real = b_data.mapv(|x| x.re);
                let a_imag = a_data.mapv(|x| x.im);
                let b_imag = b_data.mapv(|x| x.im);

                // Check both real and imaginary parts
                rust_assert_allclose(&a_real, &b_real, rtol, atol, equal_nan);
                rust_assert_allclose(&a_imag, &b_imag, rtol, atol, equal_nan);
            }
            (ArrayData::F64(a_data), ArrayData::Complex128(b_data)) => {
                // Convert float to complex
                let a_complex = a_data.mapv(|x| Complex64::new(x, 0.0));
                let a_real = a_complex.mapv(|x| x.re);
                let b_real = b_data.mapv(|x| x.re);
                let a_imag = a_complex.mapv(|x| x.im);
                let b_imag = b_data.mapv(|x| x.im);

                rust_assert_allclose(&a_real, &b_real, rtol, atol, equal_nan);
                rust_assert_allclose(&a_imag, &b_imag, rtol, atol, equal_nan);
            }
            (ArrayData::Complex128(a_data), ArrayData::F64(b_data)) => {
                // Convert float to complex
                let b_complex = b_data.mapv(|x| Complex64::new(x, 0.0));
                let a_real = a_data.mapv(|x| x.re);
                let b_real = b_complex.mapv(|x| x.re);
                let a_imag = a_data.mapv(|x| x.im);
                let b_imag = b_complex.mapv(|x| x.im);

                rust_assert_allclose(&a_real, &b_real, rtol, atol, equal_nan);
                rust_assert_allclose(&a_imag, &b_imag, rtol, atol, equal_nan);
            }
            _ => {
                panic!(
                    "assert_allclose() arguments must be PECOS Arrays of compatible dtypes (float64, complex128)"
                );
            }
        }
    }));

    // Convert panic to PyAssertionError
    if let Err(panic_err) = result {
        if let Some(msg) = panic_err.downcast_ref::<String>() {
            return Err(pyo3::exceptions::PyAssertionError::new_err(msg.clone()));
        } else if let Some(msg) = panic_err.downcast_ref::<&str>() {
            return Err(pyo3::exceptions::PyAssertionError::new_err(*msg));
        }
        return Err(pyo3::exceptions::PyAssertionError::new_err(
            "Assertion failed in assert_allclose",
        ));
    }

    Ok(())
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
/// from __pecos_rslib.num import array_equal
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
    use crate::pecos_array::ArrayData;
    use pecos::prelude::array_equal as rust_array_equal;

    // First try PECOS Array objects
    if let (Ok(a_arr), Ok(b_arr)) = (a.extract::<Py<Array>>(), b.extract::<Py<Array>>()) {
        let a_ref = a_arr.bind(a.py()).borrow();
        let b_ref = b_arr.bind(b.py()).borrow();

        match (&a_ref.data, &b_ref.data) {
            (ArrayData::Bool(a_data), ArrayData::Bool(b_data)) => {
                // For booleans, just check shape and exact equality
                if a_data.shape() != b_data.shape() {
                    return Ok(false);
                }
                return Ok(a_data.iter().zip(b_data.iter()).all(|(a, b)| a == b));
            }
            (ArrayData::I64(a_data), ArrayData::I64(b_data)) => {
                // For integers, just check shape and exact equality
                if a_data.shape() != b_data.shape() {
                    return Ok(false);
                }
                return Ok(a_data.iter().zip(b_data.iter()).all(|(a, b)| a == b));
            }
            (ArrayData::I32(a_data), ArrayData::I32(b_data)) => {
                // For integers, just check shape and exact equality
                if a_data.shape() != b_data.shape() {
                    return Ok(false);
                }
                return Ok(a_data.iter().zip(b_data.iter()).all(|(a, b)| a == b));
            }
            (ArrayData::F64(a_data), ArrayData::F64(b_data)) => {
                return Ok(rust_array_equal(a_data, b_data, equal_nan));
            }
            (ArrayData::Complex128(a_data), ArrayData::Complex128(b_data)) => {
                return Ok(rust_array_equal(a_data, b_data, equal_nan));
            }
            (ArrayData::F64(a_data), ArrayData::Complex128(b_data)) => {
                // Convert float to complex
                let a_complex = a_data.mapv(|x| Complex64::new(x, 0.0));
                return Ok(rust_array_equal(&a_complex.view(), b_data, equal_nan));
            }
            (ArrayData::Complex128(a_data), ArrayData::F64(b_data)) => {
                // Convert float to complex
                let b_complex = b_data.mapv(|x| Complex64::new(x, 0.0));
                return Ok(rust_array_equal(a_data, &b_complex.view(), equal_nan));
            }
            _ => {
                // Unsupported dtype combination, fall through to error
            }
        }
    }

    // Try mixed: PECOS Array and NumPy array
    // Check if one is a PECOS Array and the other is NumPy
    if let Ok(a_pecos) = a.extract::<Py<Array>>() {
        let a_ref = a_pecos.bind(a.py()).borrow();

        // Try to match with NumPy bool array
        if let Ok(b_array) = array_buffer::extract_bool_array(&b)
            && let ArrayData::Bool(a_data) = &a_ref.data
        {
            let b_view = b_array.view();
            if a_data.shape() != b_view.shape() {
                return Ok(false);
            }
            return Ok(a_data.iter().zip(b_view.iter()).all(|(a, b)| a == b));
        }

        // Try to match with NumPy int64 array
        if let Ok(b_array) = array_buffer::extract_i64_array(&b)
            && let ArrayData::I64(a_data) = &a_ref.data
        {
            let b_view = b_array.view();
            if a_data.shape() != b_view.shape() {
                return Ok(false);
            }
            return Ok(a_data.iter().zip(b_view.iter()).all(|(a, b)| a == b));
        }

        // Try to match with NumPy int32 array
        if let Ok(b_array) = array_buffer::extract_i32_array(&b)
            && let ArrayData::I32(a_data) = &a_ref.data
        {
            let b_view = b_array.view();
            if a_data.shape() != b_view.shape() {
                return Ok(false);
            }
            return Ok(a_data.iter().zip(b_view.iter()).all(|(a, b)| a == b));
        }

        // Try to match with NumPy float array
        if let Ok(b_array) = array_buffer::extract_f64_array(&b)
            && let ArrayData::F64(a_data) = &a_ref.data
        {
            return Ok(rust_array_equal(a_data, &b_array.view(), equal_nan));
        }

        // Try to match with NumPy complex array
        if let Ok(b_array) = array_buffer::extract_complex64_array(&b)
            && let ArrayData::Complex128(a_data) = &a_ref.data
        {
            return Ok(rust_array_equal(a_data, &b_array.view(), equal_nan));
        }
    }

    // Try the reverse: NumPy array first, PECOS Array second
    if let Ok(b_pecos) = b.extract::<Py<Array>>() {
        let b_ref = b_pecos.bind(b.py()).borrow();

        // Try to match with NumPy bool array
        if let Ok(a_array) = array_buffer::extract_bool_array(&a)
            && let ArrayData::Bool(b_data) = &b_ref.data
        {
            let a_view = a_array.view();
            if a_view.shape() != b_data.shape() {
                return Ok(false);
            }
            return Ok(a_view.iter().zip(b_data.iter()).all(|(a, b)| a == b));
        }

        // Try to match with NumPy int64 array
        if let Ok(a_array) = array_buffer::extract_i64_array(&a)
            && let ArrayData::I64(b_data) = &b_ref.data
        {
            let a_view = a_array.view();
            if a_view.shape() != b_data.shape() {
                return Ok(false);
            }
            return Ok(a_view.iter().zip(b_data.iter()).all(|(a, b)| a == b));
        }

        // Try to match with NumPy int32 array
        if let Ok(a_array) = array_buffer::extract_i32_array(&a)
            && let ArrayData::I32(b_data) = &b_ref.data
        {
            let a_view = a_array.view();
            if a_view.shape() != b_data.shape() {
                return Ok(false);
            }
            return Ok(a_view.iter().zip(b_data.iter()).all(|(a, b)| a == b));
        }

        // Try to match with NumPy float array
        if let Ok(a_array) = array_buffer::extract_f64_array(&a)
            && let ArrayData::F64(b_data) = &b_ref.data
        {
            return Ok(rust_array_equal(&a_array.view(), b_data, equal_nan));
        }

        // Try to match with NumPy complex array
        if let Ok(a_array) = array_buffer::extract_complex64_array(&a)
            && let ArrayData::Complex128(b_data) = &b_ref.data
        {
            return Ok(rust_array_equal(&a_array.view(), b_data, equal_nan));
        }
    }

    // Try bool arrays (for isnan/isclose return values)
    if let (Ok(a_array), Ok(b_array)) = (
        array_buffer::extract_bool_array(&a),
        array_buffer::extract_bool_array(&b),
    ) {
        let a_view = a_array.view();
        let b_view = b_array.view();

        // For booleans, just check shape and exact equality
        if a_view.shape() != b_view.shape() {
            return Ok(false);
        }
        // Check if all elements are equal
        return Ok(a_view.iter().zip(b_view.iter()).all(|(a, b)| a == b));
    }

    // Try integer arrays (for randint return values)
    if let (Ok(a_array), Ok(b_array)) = (
        array_buffer::extract_i64_array(&a),
        array_buffer::extract_i64_array(&b),
    ) {
        let a_view = a_array.view();
        let b_view = b_array.view();

        // For integers, just check shape and exact equality
        if a_view.shape() != b_view.shape() {
            return Ok(false);
        }
        // Check if all elements are equal
        return Ok(a_view.iter().zip(b_view.iter()).all(|(a, b)| a == b));
    }

    // Try float arrays
    if let (Ok(a_array), Ok(b_array)) = (
        array_buffer::extract_f64_array(&a),
        array_buffer::extract_f64_array(&b),
    ) {
        return Ok(rust_array_equal(
            &a_array.view(),
            &b_array.view(),
            equal_nan,
        ));
    }

    // Try complex arrays
    if let (Ok(a_array), Ok(b_array)) = (
        array_buffer::extract_complex64_array(&a),
        array_buffer::extract_complex64_array(&b),
    ) {
        return Ok(rust_array_equal(
            &a_array.view(),
            &b_array.view(),
            equal_nan,
        ));
    }

    // Handle mixed array types: complex array vs float array
    if let (Ok(a_array), Ok(b_array)) = (
        array_buffer::extract_complex64_array(&a),
        array_buffer::extract_f64_array(&b),
    ) {
        // Convert float array to complex
        let b_complex = b_array.view().mapv(|x| Complex64::new(x, 0.0));
        return Ok(rust_array_equal(
            &a_array.view(),
            &b_complex.view(),
            equal_nan,
        ));
    }

    // Handle mixed array types: float array vs complex array
    if let (Ok(a_array), Ok(b_array)) = (
        array_buffer::extract_f64_array(&a),
        array_buffer::extract_complex64_array(&b),
    ) {
        // Convert float array to complex
        let a_complex = a_array.view().mapv(|x| Complex64::new(x, 0.0));
        return Ok(rust_array_equal(
            &a_complex.view(),
            &b_array.view(),
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
/// from __pecos_rslib.num import std
///
/// # Calculate population standard deviation
/// values = [1.0, 2.0, 3.0, 4.0, 5.0]
/// population_std = std(values)  # Returns ~1.414 (ddof=0 default)
///
/// # Calculate sample standard deviation
/// sample_std = std(values, ddof=1)  # Returns ~1.581
///
/// # 2D array - std over all elements
/// arr = [[1.0, 2.0], [3.0, 4.0]]
/// std(arr)  # Returns std of flattened array
///
/// # 2D array - std along axis 0 (down columns)
/// std(arr, axis=0)  # Returns [1.0, 1.0]
///
/// # 2D array - std along axis 1 (across rows)
/// std(arr, axis=1)  # Returns [0.5, 0.5]
///
/// # Jackknife analysis use case
/// parameter_estimates = [1.5, 1.6, 1.4, 1.5, 1.7]
/// uncertainty = std(parameter_estimates, ddof=0)
/// ```
#[pyfunction]
#[pyo3(signature = (a, axis=None, ddof=0))]
fn std(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    axis: Option<isize>,
    ddof: usize,
) -> PyResult<Py<PyAny>> {
    // Use ensure_f64_array which handles PECOS Arrays, numpy arrays, and Python sequences
    let array = array_buffer::ensure_f64_array(a, "a")?;

    match axis {
        None => {
            // No axis specified - compute std of flattened array
            let flat: Vec<f64> = array.iter().copied().collect();
            if flat.is_empty() || flat.len() <= ddof {
                return Ok(f64::NAN.into_pyobject(py)?.into_any().unbind());
            }
            let result = pecos::prelude::std(&flat, ddof);
            Ok(result.into_pyobject(py)?.into_any().unbind())
        }
        Some(axis_val) => {
            // Axis specified - use std_axis logic
            let ndim = array.ndim();

            // Convert negative axis to positive
            let axis_usize = if axis_val < 0 {
                let pos = (ndim as isize + axis_val) as usize;
                if pos >= ndim {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "axis {axis_val} is out of bounds for array of dimension {ndim}"
                    )));
                }
                pos
            } else {
                let axis_usize = axis_val as usize;
                if axis_usize >= ndim {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "axis {axis_val} is out of bounds for array of dimension {ndim}"
                    )));
                }
                axis_usize
            };

            // Call Rust implementation (ddof is usize, function expects f64)
            let result = pecos::prelude::std_axis(&array.view(), Axis(axis_usize), ddof as f64);

            // Convert back to Python Array
            Ok(Py::new(py, Array::from_array_f64(result))?.into_any())
        }
    }
}

/// Calculate mean along a specified axis.
///
/// Drop-in replacement for numpy.mean with axis parameter.
///
/// # Arguments
///
/// * `arr` - Input array
/// * `axis` - Axis along which to compute the mean
///
/// # Returns
///
/// Array with one fewer dimension than the input
///
/// # Examples
///
/// ```python
/// from __pecos_rslib.num import mean_axis
/// import numpy as np
///
/// # 2D array
/// data = np.array([[1.0, 2.0, 3.0],
///                  [4.0, 5.0, 6.0]])
///
/// # Mean along axis 0 (columns)
/// result = mean_axis(data, 0)  # Returns [2.5, 3.5, 4.5]
///
/// # Mean along axis 1 (rows)
/// result = mean_axis(data, 1)  # Returns [2.0, 5.0]
/// ```
#[pyfunction]
fn mean_axis(py: Python<'_>, arr: &Bound<'_, PyAny>, axis: isize) -> PyResult<Py<PyAny>> {
    // Extract array from Python
    let array = array_buffer::extract_f64_array(arr)?;

    // Convert negative axis to positive
    let ndim = array.ndim();
    let axis_usize = if axis < 0 {
        let pos = (ndim as isize + axis) as usize;
        if pos >= ndim {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "axis {axis} is out of bounds for array of dimension {ndim}"
            )));
        }
        pos
    } else {
        let axis_usize = axis as usize;
        if axis_usize >= ndim {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "axis {axis} is out of bounds for array of dimension {ndim}"
            )));
        }
        axis_usize
    };

    // Call Rust implementation
    let result = pecos::prelude::mean_axis(&array.view(), Axis(axis_usize)).ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "mean_axis returned None - array may be empty along the specified axis",
        )
    })?;

    // Convert back to Python
    Ok(Py::new(py, Array::from_array_f64(result))?.into_any())
}

/// Calculate standard deviation along a specified axis.
///
/// Drop-in replacement for numpy.std with axis parameter.
///
/// # Arguments
///
/// * `arr` - Input array
/// * `axis` - Axis along which to compute the standard deviation
/// * `ddof` - Delta degrees of freedom (default: 0)
///
/// # Returns
///
/// Array with one fewer dimension than the input
///
/// # Examples
///
/// ```python
/// from __pecos_rslib.num import std_axis
/// import numpy as np
///
/// # 2D array
/// data = np.array([[1.0, 2.0, 3.0],
///                  [4.0, 5.0, 6.0]])
///
/// # Std along axis 0 (columns)
/// result = std_axis(data, 0, 0)  # Population std
///
/// # Std along axis 1 (rows) with sample correction
/// result = std_axis(data, 1, 1)  # Sample std
/// ```
#[pyfunction]
#[pyo3(signature = (arr, axis, ddof=0))]
fn std_axis(
    py: Python<'_>,
    arr: &Bound<'_, PyAny>,
    axis: isize,
    ddof: usize,
) -> PyResult<Py<PyAny>> {
    // Extract array from Python
    let array = array_buffer::extract_f64_array(arr)?;

    // Convert negative axis to positive
    let ndim = array.ndim();
    let axis_usize = if axis < 0 {
        let pos = (ndim as isize + axis) as usize;
        if pos >= ndim {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "axis {axis} is out of bounds for array of dimension {ndim}"
            )));
        }
        pos
    } else {
        let axis_usize = axis as usize;
        if axis_usize >= ndim {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "axis {axis} is out of bounds for array of dimension {ndim}"
            )));
        }
        axis_usize
    };

    // Call Rust implementation
    let result = pecos::prelude::std_axis(&array.view(), Axis(axis_usize), ddof as f64);

    // Convert back to Python
    Ok(Py::new(py, Array::from_array_f64(result))?.into_any())
}

/// Calculate weighted mean from (value, weight) pairs.
///
/// Drop-in replacement for the `wt_mean()` function in PECOS sampling.py.
///
/// # Arguments
///
/// * `data` - List of (value, weight) tuples
///
/// # Returns
///
/// The weighted mean: `sum(value * weight) / sum(weight)`.
/// Returns `NaN` if data is empty or total weight is zero.
///
/// # Examples
///
/// ```python
/// from __pecos_rslib.num import weighted_mean
///
/// # Fidelity measurements with shot counts
/// data = [(0.98, 100.0), (0.94, 500.0), (0.96, 200.0)]
/// avg = weighted_mean(data)  # Returns 0.95
/// ```
#[allow(clippy::needless_pass_by_value)]
#[pyfunction]
fn weighted_mean(data: Vec<(f64, f64)>) -> f64 {
    pecos::prelude::weighted_mean(&data)
}

/// Generate jackknife resamples from 1D data.
///
/// Drop-in replacement for `astropy.stats.jackknife_resampling`.
/// Generates n deterministic samples of size n-1 by leaving out one observation at a time.
///
/// # Arguments
///
/// * `data` - Original 1D sample
///
/// # Returns
///
/// 2D array where each row is a jackknife resample (shape: n × n-1)
///
/// # Examples
///
/// ```python
/// from __pecos_rslib.num import jackknife_resamples
///
/// data = [1.0, 2.0, 3.0, 4.0, 5.0]
/// resamples = jackknife_resamples(data)
/// # resamples[0] = [2.0, 3.0, 4.0, 5.0]  (removed 1.0)
/// # resamples[1] = [1.0, 3.0, 4.0, 5.0]  (removed 2.0)
/// # ...
/// ```
#[pyfunction]
fn jackknife_resamples(py: Python<'_>, data: Vec<f64>) -> PyResult<Py<Array>> {
    let resamples = pecos::prelude::jackknife_resamples(&data);
    Ok(Py::new(py, Array::from_array_f64(resamples.into_dyn()))?)
}

/// Compute jackknife statistics from leave-one-out estimates.
///
/// Given parameter estimates from jackknife resamples, calculate the mean and standard error.
///
/// # Arguments
///
/// * `estimates` - Parameter estimates from each jackknife resample
///
/// # Returns
///
/// Tuple of (`mean_estimate`, `standard_error`)
///
/// # Examples
///
/// ```python
/// from __pecos_rslib.num import jackknife_resamples, jackknife_stats
/// import numpy as np
///
/// data = [1.5, 1.6, 1.4, 1.5, 1.7]
/// resamples = jackknife_resamples(data)
/// estimates = [np.mean(resamples[i]) for i in range(len(resamples))]
/// jack_mean, jack_se = jackknife_stats(estimates)
/// ```
#[allow(clippy::needless_pass_by_value)]
#[pyfunction]
fn jackknife_stats(estimates: Vec<f64>) -> (f64, f64) {
    pecos::prelude::jackknife_stats(&estimates)
}

/// Compute jackknife statistics along an axis of a 2D array.
///
/// Given a 2D array where each row contains parameter estimates from one jackknife
/// resample (with multiple parameters per resample), compute the jackknife mean
/// and standard error for each parameter.
///
/// This is useful for threshold curve fitting where you fit multiple parameters
/// (pth, v0, a, b, c, ...) for each jackknife resample and need statistics on
/// all parameters simultaneously.
///
/// # Arguments
///
/// * `estimates` - 2D array where:
///   - `axis=0`: Each row is one jackknife resample, columns are different parameters
///   - `axis=1`: Each column is one jackknife resample, rows are different parameters
/// * `axis` - The axis along which to compute statistics (0 or 1)
///
/// # Returns
///
/// Tuple of (`mean_estimates`, `standard_errors`) where each is a 1D array with
/// one element per parameter.
///
/// # Examples
///
/// ```python
/// from __pecos_rslib.num import jackknife_stats_axis
/// import numpy as np
///
/// # 3 jackknife resamples × 2 parameters
/// # Each row is estimates from one resample: [param1, param2]
/// estimates = np.array([
///     [1.5, 10.0],  # Resample 1 estimates
///     [1.6, 10.5],  # Resample 2 estimates
///     [1.4, 9.5],   # Resample 3 estimates
/// ])
///
/// # Compute stats for each parameter (down columns)
/// means, stds = jackknife_stats_axis(estimates, axis=0)
/// # means[0] = jackknife mean of parameter 1
/// # means[1] = jackknife mean of parameter 2
/// ```
#[allow(clippy::needless_pass_by_value, clippy::type_complexity)]
#[pyfunction]
fn jackknife_stats_axis(
    py: Python<'_>,
    estimates: &Bound<'_, PyAny>,
    axis: usize,
) -> PyResult<(
    Py<crate::array_buffer::F64ArrayView>,
    Py<crate::array_buffer::F64ArrayView>,
)> {
    let estimates_array = array_buffer::extract_f64_array(estimates)?;
    // Convert to 2D array (jackknife_stats_axis expects 2D)
    let estimates_view = estimates_array
        .view()
        .into_dimensionality::<ndarray::Ix2>()
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "estimates must be 2D array: {e}"
            ))
        })?;
    let (means, stds) = pecos::prelude::jackknife_stats_axis(&estimates_view, Axis(axis));
    Ok((
        array_buffer::f64_array_to_py(py, &means),
        array_buffer::f64_array_to_py(py, &stds),
    ))
}

/// Jackknife resampling for weighted data with bias correction.
///
/// Drop-in replacement for the `jackknife()` function in PECOS sampling.py.
/// Handles weighted data (e.g., fidelity measurements with shot counts).
///
/// # Arguments
///
/// * `data` - List of (value, weight) tuples (e.g., [(fidelity, `shot_count`), ...])
///
/// # Returns
///
/// Tuple of (`corrected_estimate`, `standard_error`)
///
/// # Special Cases
///
/// For a single data point, returns binomial error estimate:
/// - Estimate = value
/// - Error = sqrt(p * (1-p) / weight) where p = 1 - value
///
/// # Examples
///
/// ```python
/// from __pecos_rslib.num import jackknife_weighted
///
/// # Multiple fidelity measurements with shot counts
/// data = [(0.98, 100.0), (0.94, 500.0), (0.96, 200.0)]
/// corrected, std_err = jackknife_weighted(data)
///
/// # Single measurement (uses binomial error)
/// single = [(0.95, 1000.0)]
/// estimate, error = jackknife_weighted(single)
/// ```
#[allow(clippy::needless_pass_by_value)]
#[pyfunction]
fn jackknife_weighted(data: Vec<(f64, f64)>) -> (f64, f64) {
    pecos::prelude::jackknife_weighted(&data)
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
/// from __pecos_rslib.num import diag
///
/// # Extract diagonal from covariance matrix
/// cov_matrix = np.array([[0.0025, 0.0010], [0.0010, 0.0004]])
/// variances = diag(cov_matrix)
/// print(variances)  # [0.0025, 0.0004]
/// ```
#[pyfunction]
fn diag(
    py: Python<'_>,
    matrix: Bound<'_, PyAny>,
) -> PyResult<Py<crate::array_buffer::F64ArrayView>> {
    let matrix_array = array_buffer::extract_f64_array(&matrix)?;
    // Convert to 2D array (diag expects 2D)
    let matrix_view = matrix_array
        .view()
        .into_dimensionality::<ndarray::Ix2>()
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("matrix must be 2D array: {e}"))
        })?;
    let diagonal = pecos::prelude::diag(matrix_view);
    Ok(array_buffer::f64_array_to_py(py, &diagonal))
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
/// from __pecos_rslib.num import linspace
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
/// from __pecos_rslib.num import arange
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
/// from __pecos_rslib.num import zeros
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
/// from __pecos_rslib.num import ones
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
/// from __pecos_rslib.num import array
/// from _pecos_rslib import dtypes
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

    // Now use __array_interface__ protocol to extract the array data
    // Get the dtype string from __array_interface__
    let array_iface = np_array.getattr("__array_interface__")?;
    let interface = array_iface.cast::<pyo3::types::PyDict>()?;
    let typestr = interface.get_item("typestr")?.ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("Missing 'typestr' in __array_interface__")
    })?;
    let typestr_str: &str = typestr.extract()?;

    // Match on dtype string and use appropriate extraction function
    match typestr_str {
        "<f8" | ">f8" | "=f8" => {
            let ndarray = array_buffer::extract_f64_array(&np_array)?;
            Py::new(
                py,
                Array {
                    data: ArrayData::F64(ndarray),
                },
            )
        }
        "<i8" | ">i8" | "=i8" => {
            let ndarray = array_buffer::extract_i64_array(&np_array)?;
            Py::new(
                py,
                Array {
                    data: ArrayData::I64(ndarray),
                },
            )
        }
        "<c16" | ">c16" | "=c16" => {
            let ndarray = array_buffer::extract_complex64_array(&np_array)?;
            Py::new(
                py,
                Array {
                    data: ArrayData::Complex128(ndarray),
                },
            )
        }
        "<f4" | ">f4" | "=f4" => {
            let ndarray = array_buffer::extract_f32_array(&np_array)?;
            Py::new(
                py,
                Array {
                    data: ArrayData::F32(ndarray),
                },
            )
        }
        "<i4" | ">i4" | "=i4" => {
            let ndarray = array_buffer::extract_i32_array(&np_array)?;
            Py::new(
                py,
                Array {
                    data: ArrayData::I32(ndarray),
                },
            )
        }
        "<i2" | ">i2" | "=i2" => {
            let ndarray = array_buffer::extract_i16_array(&np_array)?;
            Py::new(
                py,
                Array {
                    data: ArrayData::I16(ndarray),
                },
            )
        }
        "i1" | "|i1" => {
            let ndarray = array_buffer::extract_i8_array(&np_array)?;
            Py::new(
                py,
                Array {
                    data: ArrayData::I8(ndarray),
                },
            )
        }
        "|b1" => {
            let ndarray = array_buffer::extract_bool_array(&np_array)?;
            Py::new(
                py,
                Array {
                    data: ArrayData::Bool(ndarray),
                },
            )
        }
        "<c8" | ">c8" | "=c8" => {
            let ndarray = array_buffer::extract_complex32_array(&np_array)?;
            Py::new(
                py,
                Array {
                    data: ArrayData::Complex64(ndarray),
                },
            )
        }
        _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
            "Unsupported dtype '{typestr_str}' in array()"
        ))),
    }
}

/// Convert the input to an array, avoiding copies when possible.
///
/// Drop-in replacement for `numpy.asarray()`. Unlike `array()`, this function
/// returns the input array unchanged if it's already an Array with the correct dtype.
/// Only creates a copy when:
/// 1. Input is not an Array (e.g., list, tuple, scalar)
/// 2. dtype parameter is provided and differs from the input array's dtype
///
/// # Arguments
///
/// * `obj` - Input object (Array, list, tuple, scalar, etc.)
/// * `dtype` - Optional target dtype (string or `DType` enum)
///
/// # Returns
///
/// An Array, possibly without copying if the input is already suitable
///
/// # Examples
///
/// ```python
/// import pecos as pc
///
/// # No copy - input is already an Array
/// arr1 = pc.array([1.0, 2.0, 3.0])
/// arr2 = pc.asarray(arr1)  # arr2 is arr1 (same object)
///
/// # Creates copy - dtype conversion needed
/// arr3 = pc.asarray(arr1, dtype="i64")  # Converts to int64
///
/// # Creates Array - input is not an Array
/// arr4 = pc.asarray([1, 2, 3])  # Converts list to Array
/// ```
#[pyfunction]
#[pyo3(signature = (obj, dtype=None))]
fn asarray(
    py: Python<'_>,
    obj: Bound<'_, PyAny>,
    dtype: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<Array>> {
    use crate::dtypes::DType;

    // Check if obj is already an Array
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

        // No conversion needed - return the same object (no copy!)
        return Ok(obj.extract::<Py<Array>>()?);
    }

    // Input is not an Array - delegate to array() which will create one
    array(py, obj, dtype)
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
/// from __pecos_rslib.num import delete
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
    // Try to extract as different types using array_buffer
    if let Ok(arr_f64) = array_buffer::extract_f64_array(&arr) {
        // Float array
        if index >= arr_f64.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "index {} is out of bounds for array of length {}",
                index,
                arr_f64.len()
            )));
        }

        // Convert to 1D for delete operation
        let arr_1d = if arr_f64.ndim() == 1 {
            arr_f64.into_dimensionality::<ndarray::Ix1>().unwrap()
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "delete only supports 1D arrays",
            ));
        };

        let result = pecos::prelude::delete(&arr_1d, index);
        return Ok(Py::new(py, Array::from_array_f64(result.into_dyn()))?.into_any());
    }

    if let Ok(arr_c64) = array_buffer::extract_complex64_array(&arr) {
        // Complex array
        if index >= arr_c64.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "index {} is out of bounds for array of length {}",
                index,
                arr_c64.len()
            )));
        }

        // Convert to 1D for delete operation
        let arr_1d = if arr_c64.ndim() == 1 {
            arr_c64.into_dimensionality::<ndarray::Ix1>().unwrap()
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "delete only supports 1D arrays",
            ));
        };

        let result = pecos::prelude::delete(&arr_1d, index);
        return Ok(Py::new(py, Array::from_array_c128(result.into_dyn()))?.into_any());
    }

    // Try integer extraction via extract_i64_array if it exists, otherwise error
    if let Ok(arr_i64) = array_buffer::extract_i64_array(&arr) {
        // Integer array
        if index >= arr_i64.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "index {} is out of bounds for array of length {}",
                index,
                arr_i64.len()
            )));
        }

        // Convert to 1D for delete operation
        let arr_1d = if arr_i64.ndim() == 1 {
            arr_i64.into_dimensionality::<ndarray::Ix1>().unwrap()
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "delete only supports 1D arrays",
            ));
        };

        let result = pecos::prelude::delete(&arr_1d, index);
        return Ok(Py::new(py, Array::from_array_i64(result.into_dyn()))?.into_any());
    }

    Err(PyTypeError::new_err("Unsupported array type for delete"))
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
/// from __pecos_rslib.num import sum
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
                    "b" => {
                        // Boolean array - sum treats True=1, False=0
                        let arr = array_buffer::extract_bool_array(&a)?;
                        let result: i64 = arr.iter().map(|&b| i64::from(b)).sum();
                        return Ok(result.into_py_any(py).unwrap());
                    }
                    "i" | "u" => {
                        // Integer array
                        let arr = array_buffer::extract_i64_array(&a)?;
                        let result: i64 = arr.iter().sum();
                        return Ok(result.into_py_any(py).unwrap());
                    }
                    "f" => {
                        // Float array
                        let arr = array_buffer::extract_f64_array(&a)?;
                        let result: f64 = arr.iter().sum();
                        return Ok(result.into_py_any(py).unwrap());
                    }
                    "c" => {
                        // Complex array
                        let arr = array_buffer::extract_complex64_array(&a)?;
                        let result: Complex64 = arr.iter().copied().sum();
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
    let np_array = if array_buffer::extract_f64_array(&a).is_err()
        && array_buffer::extract_complex64_array(&a).is_err()
        && array_buffer::extract_i64_array(&a).is_err()
        && array_buffer::extract_bool_array(&a).is_err()
    {
        // Not a numpy array - convert to numpy array using numpy.array()
        let numpy = py.import("numpy")?;
        numpy.call_method1("array", (a,))?
    } else {
        // Already a numpy array
        a
    };

    // Try boolean array with axis FIRST - convert to i64 for sum
    if let Ok(arr) = array_buffer::extract_bool_array(&np_array) {
        let array = arr;
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

        // Convert boolean array to i64 array, then sum along the specified axis
        let i64_array = array.mapv(i64::from);
        let result = i64_array.sum_axis(Axis(normalized_axis));
        return Ok(array_buffer::i64_array_to_py(py, &result).into());
    }

    // Try integer array with axis (before complex/float to avoid unwanted casting)
    if let Ok(arr) = array_buffer::extract_i64_array(&np_array) {
        let array = arr;
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
        return Ok(array_buffer::i64_array_to_py(py, &result).into());
    }

    // Try complex array with axis (before float, to avoid unwanted casting)
    if let Ok(arr) = array_buffer::extract_complex64_array(&np_array) {
        let array = arr;
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
        return Ok(array_buffer::complex64_array_to_py(py, &result).into());
    }

    // Try float array with axis
    if let Ok(arr) = array_buffer::extract_f64_array(&np_array) {
        let array = arr;
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
        return Ok(array_buffer::f64_array_to_py(py, &result).into());
    }

    Err(PyTypeError::new_err(
        "sum() with axis requires a numpy array of numbers",
    ))
}

/// Return the maximum value along an array.
///
/// Drop-in replacement for `numpy.max()` or `numpy.amax()`.
/// Returns the maximum value of an array, or along an axis.
#[pyfunction]
#[pyo3(signature = (a, axis=None))]
#[allow(clippy::needless_pass_by_value)]
fn max(py: Python<'_>, a: Bound<'_, PyAny>, axis: Option<isize>) -> PyResult<Py<PyAny>> {
    // Handle axis=None case: find global maximum
    if axis.is_none() {
        // Check if it's a numpy array by checking for 'dtype' attribute
        if let Ok(dtype_attr) = a.getattr("dtype") {
            // It's a numpy array - check its dtype.kind
            if let Ok(kind_attr) = dtype_attr.getattr("kind") {
                let kind: String = kind_attr.extract()?;

                match kind.as_str() {
                    "b" => {
                        // Boolean array - max treats True=1, False=0
                        let arr = array_buffer::extract_bool_array(&a)?;
                        let result = arr.iter().any(|&x| x);
                        return Ok(result.into_py_any(py).unwrap());
                    }
                    "i" | "u" => {
                        // Integer array
                        let arr = array_buffer::extract_i64_array(&a)?;
                        let array_view = &arr;
                        let result = array_view.iter().max().ok_or_else(|| {
                            PyErr::new::<pyo3::exceptions::PyValueError, _>("max() of empty array")
                        })?;
                        return Ok((*result).into_py_any(py).unwrap());
                    }
                    "f" => {
                        // Float array
                        let arr = array_buffer::extract_f64_array(&a)?;
                        let array_view = &arr;
                        let result = array_view
                            .iter()
                            .max_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal))
                            .ok_or_else(|| {
                                PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                    "max() of empty array",
                                )
                            })?;
                        return Ok((*result).into_py_any(py).unwrap());
                    }
                    "c" => {
                        // Complex array - can't directly compare, need magnitude
                        return Err(PyTypeError::new_err(
                            "max() is not supported for complex arrays (use abs() first for magnitude comparison)",
                        ));
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
            let result = values.iter().max().ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>("max() of empty sequence")
            })?;
            return Ok((*result).into_py_any(py).unwrap());
        }

        // Try float list/tuple
        if let Ok(values) = a.extract::<Vec<f64>>() {
            let result = values
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal))
                .ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>("max() of empty sequence")
                })?;
            return Ok((*result).into_py_any(py).unwrap());
        }

        return Err(PyTypeError::new_err(
            "max() argument must be a list, tuple, or numpy array of numbers",
        ));
    }

    // Handle axis parameter case: find max along specific axis
    // Note: ndarray doesn't have a built-in max_axis for floats, so we'll fold along the axis
    let axis_val = axis.unwrap();

    // Integer array with axis
    if let Ok(arr) = array_buffer::extract_i64_array(&a) {
        let array = arr;
        let ndim = array.ndim();

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

        // Use fold_axis to find max along axis
        let result = array.fold_axis(Axis(normalized_axis), i64::MIN, |&max_val, &x| {
            if x > max_val { x } else { max_val }
        });
        return Ok(array_buffer::i64_array_to_py(py, &result).into());
    }

    // Float array with axis
    if let Ok(arr) = array_buffer::extract_f64_array(&a) {
        let array = arr;
        let ndim = array.ndim();

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

        let result = array.fold_axis(Axis(normalized_axis), f64::NEG_INFINITY, |&max_val, &x| {
            if x > max_val { x } else { max_val }
        });
        return Ok(array_buffer::f64_array_to_py(py, &result).into());
    }

    Err(PyTypeError::new_err(
        "max() with axis requires a numpy array of numbers",
    ))
}

/// Return the minimum value along an array.
///
/// Drop-in replacement for `numpy.min()` or `numpy.amin()`.
/// Returns the minimum value of an array, or along an axis.
#[pyfunction]
#[pyo3(signature = (a, axis=None))]
#[allow(clippy::needless_pass_by_value)]
fn min(py: Python<'_>, a: Bound<'_, PyAny>, axis: Option<isize>) -> PyResult<Py<PyAny>> {
    // Handle axis=None case: find global minimum
    if axis.is_none() {
        // Check if it's a numpy array by checking for 'dtype' attribute
        if let Ok(dtype_attr) = a.getattr("dtype") {
            // It's a numpy array - check its dtype.kind
            if let Ok(kind_attr) = dtype_attr.getattr("kind") {
                let kind: String = kind_attr.extract()?;

                match kind.as_str() {
                    "b" => {
                        // Boolean array - min treats True=1, False=0
                        let arr = array_buffer::extract_bool_array(&a)?;
                        let result = !arr.iter().all(|&x| x);
                        return Ok(result.into_py_any(py).unwrap());
                    }
                    "i" | "u" => {
                        // Integer array
                        let arr = array_buffer::extract_i64_array(&a)?;
                        let array_view = &arr;
                        let result = array_view.iter().min().ok_or_else(|| {
                            PyErr::new::<pyo3::exceptions::PyValueError, _>("min() of empty array")
                        })?;
                        return Ok((*result).into_py_any(py).unwrap());
                    }
                    "f" => {
                        // Float array
                        let arr = array_buffer::extract_f64_array(&a)?;
                        let array_view = &arr;
                        let result = array_view
                            .iter()
                            .min_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal))
                            .ok_or_else(|| {
                                PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                    "min() of empty array",
                                )
                            })?;
                        return Ok((*result).into_py_any(py).unwrap());
                    }
                    "c" => {
                        // Complex array - can't directly compare, need magnitude
                        return Err(PyTypeError::new_err(
                            "min() is not supported for complex arrays (use abs() first for magnitude comparison)",
                        ));
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
            let result = values.iter().min().ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>("min() of empty sequence")
            })?;
            return Ok((*result).into_py_any(py).unwrap());
        }

        // Try float list/tuple
        if let Ok(values) = a.extract::<Vec<f64>>() {
            let result = values
                .iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal))
                .ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>("min() of empty sequence")
                })?;
            return Ok((*result).into_py_any(py).unwrap());
        }

        return Err(PyTypeError::new_err(
            "min() argument must be a list, tuple, or numpy array of numbers",
        ));
    }

    // Handle axis parameter case: find min along specific axis
    let axis_val = axis.unwrap();

    // Integer array with axis
    if let Ok(arr) = array_buffer::extract_i64_array(&a) {
        let array = arr;
        let ndim = array.ndim();

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

        let result = array.fold_axis(Axis(normalized_axis), i64::MAX, |&min_val, &x| {
            if x < min_val { x } else { min_val }
        });
        return Ok(array_buffer::i64_array_to_py(py, &result).into());
    }

    // Float array with axis
    if let Ok(arr) = array_buffer::extract_f64_array(&a) {
        let array = arr;
        let ndim = array.ndim();

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

        let result = array.fold_axis(Axis(normalized_axis), f64::INFINITY, |&min_val, &x| {
            if x < min_val { x } else { min_val }
        });
        return Ok(array_buffer::f64_array_to_py(py, &result).into());
    }

    Err(PyTypeError::new_err(
        "min() with axis requires a numpy array of numbers",
    ))
}

// ============================================================================
// Array and Complex Number Support
// ============================================================================

// ============================================================================
// Math Functions (polymorphic - handle scalars, complex, and arrays)
// ============================================================================

/// Macro to apply a unary function with proper type conversion.
///
/// This macro implements the type-checking pattern that preserves dtype information
/// and avoids `ComplexWarning` when passing `NumPy` scalars to PECOS functions.
///
/// # Type Checking Order (Critical!)
///
/// The order of type checks is critical to avoid `ComplexWarning`:
/// 1. Array types (PECOS Array wrapper) - checked first
/// 2. `NumPy` scalars and array-like objects - preserves dtype
/// 3. Python scalar float - only for Python literals
/// 4. Python scalar complex - only for Python complex literals
///
/// `NumPy` scalars (np.float64, np.complex128, etc.) implement `__array_interface__`
/// and must be converted via `Array::from_python_value()` to preserve their dtype.
/// If we extract them as f64 first, complex types lose their imaginary part and
/// trigger `ComplexWarning`.
///
/// # Parameters
/// - `$fn_name`: Name of the function (for error messages)
/// - `$py`: Python interpreter reference
/// - `$x`: Input value to convert
/// - `$f64_op`: Operation to apply to f64 values (e.g., `sqrt()`)
/// - `$complex_op`: Operation to apply to complex values (e.g., `ComplexFloat::sqrt()`)
/// - `$self_fn`: Recursive function to call for arrays (e.g., `sqrt`)
macro_rules! apply_unary_math_fn {
    ($fn_name:expr, $py:expr, $x:expr, $f64_op:expr, $complex_op:expr, $self_fn:ident) => {{
        // Try Array type first (our custom array wrapper)
        if let Ok(arr) = $x.extract::<Py<Array>>() {
            use crate::pecos_array::ArrayData;
            let arr_ref = arr.bind($py).borrow();
            match &arr_ref.data {
                ArrayData::F64(a) => {
                    let result = a.mapv($f64_op);
                    return Ok(Py::new($py, Array::from_array_f64(result))?.into_any());
                }
                ArrayData::F32(a) => {
                    let result = a.mapv(|v| $f64_op(f64::from(v)));
                    return Ok(Py::new($py, Array::from_array_f64(result))?.into_any());
                }
                ArrayData::Complex128(a) => {
                    let result = a.mapv($complex_op);
                    return Ok(Py::new($py, Array::from_array_c128(result))?.into_any());
                }
                ArrayData::Complex64(a) => {
                    use num_complex::Complex;
                    let result = a.mapv(|c| {
                        let c128 = Complex::new(f64::from(c.re), f64::from(c.im));
                        $complex_op(c128)
                    });
                    return Ok(Py::new($py, Array::from_array_c128(result))?.into_any());
                }
                _ => {
                    return Err(PyTypeError::new_err(format!(
                        "{}() requires float or complex array",
                        $fn_name
                    )));
                }
            }
        }

        // Try NumPy scalars and array-like objects (handles np.float64, np.complex128, etc.)
        // This must come before scalar extraction to preserve dtype information
        if let Ok(arr) = Array::from_python_value(&$x, None) {
            let arr_py = Py::new($py, arr)?;
            return $self_fn($py, arr_py.bind($py).as_any().clone());
        }

        // Try scalar f64 (Python float or literal)
        if let Ok(val) = $x.extract::<f64>() {
            return Ok($f64_op(val).into_py_any($py).unwrap());
        }

        // Try scalar complex (Python complex literal)
        if $x.is_exact_instance_of::<pyo3::types::PyComplex>() {
            let py_complex = $x.clone().cast_into::<pyo3::types::PyComplex>().unwrap();
            if let Ok(val) = py_complex.extract::<Complex64>() {
                return Ok($complex_op(val).into_py_any($py).unwrap());
            }
        }

        Err(PyTypeError::new_err(format!(
            "{}() argument must be float, complex, or array-like",
            $fn_name
        )))
    }};
}

/// Macro to apply a unary function using `array_buffer` extraction (simpler pattern).
///
/// This macro implements the type-checking pattern for functions that use the
/// `array_buffer` module for extraction, which handles `NumPy` array conversion automatically.
///
/// The key difference from `apply_unary_math_fn` is that this pattern uses
/// `array_buffer::extract_*_array()` which internally handles `NumPy` scalars correctly.
///
/// # Parameters
/// - `$fn_name`: Name of the function (for error messages)
/// - `$py`: Python interpreter reference
/// - `$x`: Input value to convert
/// - `$trait_name`: Name of the trait to import (e.g., `Sinh`)
/// - `$f64_method`: Method to call on f64 values (e.g., `sinh`)
/// - `$complex_method`: Method to call on complex values (e.g., `sinh`)
macro_rules! apply_buffer_math_fn {
    ($fn_name:expr, $py:expr, $x:expr, $trait_name:ident, $f64_method:ident, $complex_method:ident) => {{
        use pecos::prelude::$trait_name;

        // Try arrays first (handles NumPy scalars and arrays)
        // This must come before scalar extraction to preserve dtype information
        if let Ok(arr) = array_buffer::extract_f64_array(&$x) {
            let result = arr.$f64_method();
            return Ok(array_buffer::f64_array_to_py($py, &result).into());
        }
        if let Ok(arr) = array_buffer::extract_complex64_array(&$x) {
            let result = arr.$complex_method();
            return Ok(array_buffer::complex64_array_to_py($py, &result).into());
        }
        // Try scalar float (Python float or literal)
        if let Ok(val) = $x.extract::<f64>() {
            return Ok(val.$f64_method().into_py_any($py).unwrap());
        }
        // Try scalar complex (Python complex literal)
        if let Ok(val) = $x.extract::<Complex64>() {
            return Ok(val.$complex_method().into_py_any($py).unwrap());
        }
        Err(PyTypeError::new_err(format!(
            "{}() argument must be float, complex, or array",
            $fn_name
        )))
    }};
}

/// Calculate exponential (e^x).
///
/// Handles scalars (float), complex numbers, and arrays automatically.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)] // Bound is designed to be passed by value (PyO3 convention)
fn exp(py: Python<'_>, x: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    apply_unary_math_fn!("exp", py, x, |v: f64| v.exp(), |c: Complex64| c.exp(), exp)
}

/// Calculate natural logarithm (base e).
///
/// More explicit than `numpy.log()` - uses `ln()` instead of `log()` for clarity.
/// Handles scalars (float, complex) and arrays automatically.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)] // Bound is designed to be passed by value (PyO3 convention)
fn ln(py: Python<'_>, x: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    // Try Array type first (our custom array wrapper) - return Array
    if let Ok(arr) = x.extract::<Py<Array>>() {
        use crate::pecos_array::ArrayData;
        let arr_ref = arr.bind(py).borrow();
        match &arr_ref.data {
            ArrayData::F64(a) => {
                let result = a.ln();
                return Ok(Py::new(py, Array::from_array_f64(result))?.into_any());
            }
            ArrayData::F32(a) => {
                let result_f32 = a.ln();
                let result = result_f32.mapv(f64::from);
                return Ok(Py::new(py, Array::from_array_f64(result))?.into_any());
            }
            ArrayData::Complex128(a) => {
                let result = a.mapv(|c| c.ln());
                return Ok(Py::new(py, Array::from_array_c128(result))?.into_any());
            }
            ArrayData::Complex64(a) => {
                let result = a.mapv(|c| {
                    let ln_result = c.ln();
                    Complex64::new(f64::from(ln_result.re), f64::from(ln_result.im))
                });
                return Ok(Py::new(py, Array::from_array_c128(result))?.into_any());
            }
            _ => {
                return Err(PyTypeError::new_err("ln() requires float or complex array"));
            }
        }
    }

    // Try scalar f64
    if let Ok(val) = x.extract::<f64>() {
        return Ok(val.ln().into_py_any(py).unwrap());
    }

    // Try scalar complex
    if let Ok(py_complex) = x.clone().cast_into::<pyo3::types::PyComplex>()
        && let Ok(val) = py_complex.extract::<Complex64>()
    {
        return Ok(val.ln().into_py_any(py).unwrap());
    }

    // Fallback: Try to convert input to Array (handles NumPy, lists, etc.)
    if let Ok(arr) = Array::from_python_value(&x, None) {
        let arr_py = Py::new(py, arr)?;
        return ln(py, arr_py.bind(py).as_any().clone());
    }

    Err(PyTypeError::new_err(
        "ln() argument must be float, complex, or array-like",
    ))
}

/// Calculate logarithm with custom base.
///
/// More general than natural logarithm - log(x, base) returns `log_base(x)`.
/// Handles scalars (float, complex) and arrays automatically.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)] // Bound is designed to be passed by value (PyO3 convention)
fn log(py: Python<'_>, x: Bound<'_, PyAny>, base: f64) -> PyResult<Py<PyAny>> {
    use pecos::prelude::LogBase;

    // Try Array type first (our custom array wrapper) - return Array
    if let Ok(arr) = x.extract::<Py<Array>>() {
        use crate::pecos_array::ArrayData;
        let arr_ref = arr.bind(py).borrow();
        match &arr_ref.data {
            ArrayData::F64(a) => {
                let result = a.log(base);
                return Ok(Py::new(py, Array::from_array_f64(result))?.into_any());
            }
            ArrayData::F32(a) => {
                let result_f32 = a.log(base as f32);
                let result = result_f32.mapv(f64::from);
                return Ok(Py::new(py, Array::from_array_f64(result))?.into_any());
            }
            ArrayData::Complex128(a) => {
                let result = a.log(base);
                return Ok(Py::new(py, Array::from_array_c128(result))?.into_any());
            }
            ArrayData::Complex64(a) => {
                let result = a.mapv(|c| {
                    let log_result = c.log(base as f32);
                    Complex64::new(f64::from(log_result.re), f64::from(log_result.im))
                });
                return Ok(Py::new(py, Array::from_array_c128(result))?.into_any());
            }
            _ => {
                return Err(PyTypeError::new_err(
                    "log() requires float or complex array",
                ));
            }
        }
    }

    // Try scalar f64
    if let Ok(val) = x.extract::<f64>() {
        return Ok(val.log(base).into_py_any(py).unwrap());
    }

    // Try scalar complex
    if let Ok(py_complex) = x.clone().cast_into::<pyo3::types::PyComplex>()
        && let Ok(val) = py_complex.extract::<Complex64>()
    {
        return Ok(val.log(base).into_py_any(py).unwrap());
    }

    // Fallback: Try to convert input to Array (handles NumPy, lists, etc.)
    if let Ok(arr) = Array::from_python_value(&x, None) {
        let arr_py = Py::new(py, arr)?;
        return log(py, arr_py.bind(py).as_any().clone(), base);
    }

    Err(PyTypeError::new_err(
        "log() argument must be float, complex, or array-like",
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
    if let Ok(arr) = array_buffer::extract_bool_array(&a) {
        return Ok(arr.iter().all(|&x| x));
    }

    // Handle float arrays (non-zero is truthy)
    if let Ok(arr) = array_buffer::extract_f64_array(&a) {
        return Ok(arr.iter().all(|&x| x != 0.0));
    }

    // Handle integer arrays
    if let Ok(arr) = array_buffer::extract_i64_array(&a) {
        return Ok(arr.iter().all(|&x| x != 0));
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
    if let Ok(arr) = array_buffer::extract_bool_array(&a) {
        return Ok(arr.iter().any(|&x| x));
    }

    // Handle float arrays (non-zero is truthy)
    if let Ok(arr) = array_buffer::extract_f64_array(&a) {
        return Ok(arr.iter().any(|&x| x != 0.0));
    }

    // Handle integer arrays
    if let Ok(arr) = array_buffer::extract_i64_array(&a) {
        return Ok(arr.iter().any(|&x| x != 0));
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
            ArrayData::F64(arr) => Ok(norm_fn(arr, ord)),
            ArrayData::F32(arr) => {
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
            ArrayData::I64(arr) => {
                // Convert int to float for norm
                let arr_f64 = arr.mapv(|v| v as f64);
                Ok(norm_fn(&arr_f64, ord))
            }
            ArrayData::I32(arr) => {
                let arr_f64 = arr.mapv(f64::from);
                Ok(norm_fn(&arr_f64, ord))
            }
            ArrayData::I16(arr) => {
                let arr_f64 = arr.mapv(f64::from);
                Ok(norm_fn(&arr_f64, ord))
            }
            ArrayData::I8(arr) => {
                let arr_f64 = arr.mapv(f64::from);
                Ok(norm_fn(&arr_f64, ord))
            }
            ArrayData::U64(arr) => {
                let arr_f64 = arr.mapv(|v| v as f64);
                Ok(norm_fn(&arr_f64, ord))
            }
            ArrayData::U32(arr) => {
                let arr_f64 = arr.mapv(f64::from);
                Ok(norm_fn(&arr_f64, ord))
            }
            ArrayData::U16(arr) => {
                let arr_f64 = arr.mapv(f64::from);
                Ok(norm_fn(&arr_f64, ord))
            }
            ArrayData::U8(arr) => {
                let arr_f64 = arr.mapv(f64::from);
                Ok(norm_fn(&arr_f64, ord))
            }
            ArrayData::Pauli(_) => Err(pyo3::exceptions::PyTypeError::new_err(
                "norm() operation not supported on Pauli arrays",
            )),
            ArrayData::PauliString(_) => Err(pyo3::exceptions::PyTypeError::new_err(
                "norm() operation not supported on PauliString arrays",
            )),
        };
    }

    // Try f64 arrays (numpy arrays)
    if let Ok(arr) = array_buffer::extract_f64_array(&x) {
        return Ok(norm_fn(&arr.view(), ord));
    }

    // Try Complex64 arrays (numpy arrays)
    if let Ok(arr) = array_buffer::extract_complex64_array(&x) {
        return Ok(norm_complex(&arr.view(), ord));
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
    apply_unary_math_fn!(
        "sqrt",
        py,
        x,
        |v: f64| v.sqrt(),
        |c: Complex64| c.sqrt(),
        sqrt
    )
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
        if let Ok(arr) = array_buffer::extract_f64_array(&base) {
            let result = arr.power(exp_val);
            return Ok(array_buffer::f64_array_to_py(py, &result).into());
        }

        // Try Python sequence base (list, tuple, etc.) - 1D
        if let Ok(vec) = base.extract::<Vec<f64>>() {
            let arr = Array1::from(vec);
            let result = arr.power(exp_val);
            return Ok(array_buffer::f64_array_to_py(py, &result).into());
        }

        // Try 2D Python sequence (nested lists) - convert to numpy first
        if let Ok(numpy) = py.import("numpy")
            && let Ok(np_array) = numpy.call_method1("array", (base,))
            && let Ok(arr) = array_buffer::extract_f64_array(&np_array)
        {
            let result = arr.power(exp_val);
            return Ok(array_buffer::f64_array_to_py(py, &result).into());
        }

        return Err(PyTypeError::new_err(
            "power() base must be float, array, or sequence",
        ));
    }

    // Array exponent - need element-wise power using std::f64::powf
    // Get base as scalar
    if let Ok(base_val) = base.extract::<f64>() {
        // Try numpy array exponent
        if let Ok(exp_arr) = array_buffer::extract_f64_array(&exponent) {
            let result = exp_arr.mapv(|e| base_val.powf(e));
            return Ok(array_buffer::f64_array_to_py(py, &result).into());
        }

        // Try Python sequence exponent
        if let Ok(exp_vec) = exponent.extract::<Vec<f64>>() {
            let result: Vec<f64> = exp_vec.iter().map(|&e| base_val.powf(e)).collect();
            let arr = Array1::from(result);
            return Ok(array_buffer::f64_array_to_py(py, &arr).into());
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
    apply_unary_math_fn!("cos", py, x, |v: f64| v.cos(), |c: Complex64| c.cos(), cos)
}

/// Calculate sine (input in radians).
///
/// Handles scalars (float) and arrays automatically.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)] // Bound is designed to be passed by value (PyO3 convention)
fn sin(py: Python<'_>, x: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    apply_unary_math_fn!("sin", py, x, |v: f64| v.sin(), |c: Complex64| c.sin(), sin)
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

    // Try Array type first (our custom array wrapper) - return Array
    if let Ok(arr) = x.extract::<Py<Array>>() {
        use crate::pecos_array::ArrayData;
        let arr_ref = arr.bind(py).borrow();
        match &arr_ref.data {
            ArrayData::F64(a) => {
                let result = a.tan();
                return Ok(Py::new(py, Array::from_array_f64(result))?.into_any());
            }
            ArrayData::F32(a) => {
                let result_f32 = a.tan();
                let result = result_f32.mapv(f64::from);
                return Ok(Py::new(py, Array::from_array_f64(result))?.into_any());
            }
            ArrayData::Complex128(a) => {
                let result = a.mapv(|c| c.tan());
                return Ok(Py::new(py, Array::from_array_c128(result))?.into_any());
            }
            ArrayData::Complex64(a) => {
                let result = a.mapv(|c| {
                    let tan_result = c.tan();
                    Complex64::new(f64::from(tan_result.re), f64::from(tan_result.im))
                });
                return Ok(Py::new(py, Array::from_array_c128(result))?.into_any());
            }
            _ => {
                return Err(PyTypeError::new_err(
                    "tan() requires float or complex array",
                ));
            }
        }
    }

    // Try scalar f64
    if let Ok(val) = x.extract::<f64>() {
        return Ok(val.tan().into_py_any(py).unwrap());
    }

    // Try scalar complex
    if let Ok(py_complex) = x.clone().cast_into::<pyo3::types::PyComplex>()
        && let Ok(val) = py_complex.extract::<Complex64>()
    {
        return Ok(val.tan().into_py_any(py).unwrap());
    }

    // Fallback: Try to convert input to Array (handles NumPy, lists, etc.) and return Array
    if let Ok(arr) = Array::from_python_value(&x, None) {
        let arr_py = Py::new(py, arr)?;
        // Recursively call tan() with the converted Array
        return tan(py, arr_py.bind(py).as_any().clone());
    }

    Err(PyTypeError::new_err(
        "tan() argument must be float, complex, or array-like",
    ))
}

/// Calculate hyperbolic sine.
///
/// Drop-in replacement for `numpy.sinh()`.
/// Handles scalars (float, complex) and arrays automatically.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn sinh(py: Python<'_>, x: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    apply_buffer_math_fn!("sinh", py, x, Sinh, sinh, sinh)
}

/// Calculate hyperbolic cosine.
///
/// Drop-in replacement for `numpy.cosh()`.
/// Handles scalars (float, complex) and arrays automatically.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn cosh(py: Python<'_>, x: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    apply_buffer_math_fn!("cosh", py, x, Cosh, cosh, cosh)
}

/// Calculate hyperbolic tangent.
///
/// Drop-in replacement for `numpy.tanh()`.
/// Handles scalars (float, complex) and arrays automatically.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)] // Bound is designed to be passed by value (PyO3 convention)
fn tanh(py: Python<'_>, x: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    apply_buffer_math_fn!("tanh", py, x, Tanh, tanh, tanh)
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
    if let Ok(arr) = array_buffer::extract_f64_array(&x) {
        let result = arr.asin();
        return Ok(array_buffer::f64_array_to_py(py, &result).into());
    }
    if let Ok(arr) = array_buffer::extract_complex64_array(&x) {
        let result = arr.asin();
        return Ok(array_buffer::complex64_array_to_py(py, &result).into());
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
    if let Ok(arr) = array_buffer::extract_f64_array(&x) {
        let result = arr.acos();
        return Ok(array_buffer::f64_array_to_py(py, &result).into());
    }
    if let Ok(arr) = array_buffer::extract_complex64_array(&x) {
        let result = arr.acos();
        return Ok(array_buffer::complex64_array_to_py(py, &result).into());
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
    if let Ok(arr) = array_buffer::extract_f64_array(&x) {
        let result = arr.atan();
        return Ok(array_buffer::f64_array_to_py(py, &result).into());
    }
    if let Ok(arr) = array_buffer::extract_complex64_array(&x) {
        let result = arr.atan();
        return Ok(array_buffer::complex64_array_to_py(py, &result).into());
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
    if let Ok(arr) = array_buffer::extract_f64_array(&x) {
        let result = arr.asinh();
        return Ok(array_buffer::f64_array_to_py(py, &result).into());
    }
    if let Ok(arr) = array_buffer::extract_complex64_array(&x) {
        let result = arr.asinh();
        return Ok(array_buffer::complex64_array_to_py(py, &result).into());
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
    if let Ok(arr) = array_buffer::extract_f64_array(&x) {
        let result = arr.acosh();
        return Ok(array_buffer::f64_array_to_py(py, &result).into());
    }
    if let Ok(arr) = array_buffer::extract_complex64_array(&x) {
        let result = arr.acosh();
        return Ok(array_buffer::complex64_array_to_py(py, &result).into());
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
    if let Ok(arr) = array_buffer::extract_f64_array(&x) {
        let result = arr.atanh();
        return Ok(array_buffer::f64_array_to_py(py, &result).into());
    }
    if let Ok(arr) = array_buffer::extract_complex64_array(&x) {
        let result = arr.atanh();
        return Ok(array_buffer::complex64_array_to_py(py, &result).into());
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
    if let (Ok(y_arr), Ok(x_val)) = (array_buffer::extract_f64_array(&y), x.extract::<f64>()) {
        let result = y_arr.atan2(x_val);
        return Ok(array_buffer::f64_array_to_py(py, &result).into());
    }

    // Array-scalar case: Complex64 array, Complex64 scalar -> Complex64 array
    if let (Ok(y_arr), Ok(x_val)) = (
        array_buffer::extract_complex64_array(&y),
        x.extract::<Complex64>(),
    ) {
        let result = y_arr.atan2(x_val);
        return Ok(array_buffer::complex64_array_to_py(py, &result).into());
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
    if let Ok(arr) = array_buffer::extract_f64_array(&x) {
        let result = arr.abs();
        // If it's a 0-dimensional array (numpy scalar), extract the single value
        if result.ndim() == 0
            && let Some(&val) = result.first()
        {
            return Ok(val.into_py_any(py).unwrap());
        }
        return Ok(array_buffer::f64_array_to_py(py, &result).into());
    }
    // Try Complex64 array (includes numpy complex scalars which are 0-dim arrays)
    if let Ok(arr) = array_buffer::extract_complex64_array(&x) {
        let result = Abs::abs(&arr); // Explicitly call the Abs trait method
        // If it's a 0-dimensional array (numpy scalar), extract the single value
        if result.ndim() == 0
            && let Some(&val) = result.first()
        {
            return Ok(val.into_py_any(py).unwrap());
        }
        return Ok(array_buffer::f64_array_to_py(py, &result).into());
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
            ArrayData::F64(a) => {
                let result = a.abs(); // Uses Abs trait
                return Ok(Py::new(py, Array::from_array_f64(result))?.into_any());
            }
            ArrayData::F32(a) => {
                // abs() returns Array<f32, D>, convert to f64
                let result = a.mapv(|v| f64::from(v.abs()));
                return Ok(Py::new(py, Array::from_array_f64(result))?.into_any());
            }
            // Integer types -> use stdlib abs() for each element
            ArrayData::I64(a) => {
                let result = a.mapv(i64::abs);
                return Ok(Py::new(py, Array::from_array_i64(result))?.into_any());
            }
            ArrayData::I32(a) => {
                let result = a.mapv(|v| i64::from(v.abs()));
                return Ok(Py::new(py, Array::from_array_i64(result))?.into_any());
            }
            ArrayData::I16(a) => {
                let result = a.mapv(|v| i64::from(v.abs()));
                return Ok(Py::new(py, Array::from_array_i64(result))?.into_any());
            }
            ArrayData::I8(a) => {
                let result = a.mapv(|v| i64::from(v.abs()));
                return Ok(Py::new(py, Array::from_array_i64(result))?.into_any());
            }
            // Unsigned types -> already positive, just convert to u64
            ArrayData::U64(a) => {
                return Ok(Py::new(py, Array::from_array_u64(a.clone()))?.into_any());
            }
            ArrayData::U32(a) => {
                let result = a.mapv(u64::from);
                return Ok(Py::new(py, Array::from_array_u64(result))?.into_any());
            }
            ArrayData::U16(a) => {
                let result = a.mapv(u64::from);
                return Ok(Py::new(py, Array::from_array_u64(result))?.into_any());
            }
            ArrayData::U8(a) => {
                let result = a.mapv(u64::from);
                return Ok(Py::new(py, Array::from_array_u64(result))?.into_any());
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
            ArrayData::Pauli(_) => {
                return Err(PyTypeError::new_err(
                    "abs() operation not supported on Pauli arrays",
                ));
            }
            ArrayData::PauliString(_) => {
                return Err(PyTypeError::new_err(
                    "abs() operation not supported on PauliString arrays",
                ));
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
/// from __pecos_rslib.num import where
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
/// from __pecos_rslib.num import where_array
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
    use ndarray::{Array, ArrayD, IxDyn};
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
        if let Ok(arr) = array_buffer::extract_f64_array(obj) {
            return Ok(arr);
        }

        // Convert via numpy asarray
        let py = obj.py();
        let np = py.import("numpy")?;
        let asarray = np.getattr("asarray")?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("dtype", "float64")?;
        let converted = asarray.call((obj,), Some(&kwargs))?;
        array_buffer::extract_f64_array(&converted)
    }

    fn to_bool_array_or_scalar(obj: &Bound<'_, PyAny>) -> PyResult<ArrayD<bool>> {
        // Try to extract as scalar bool first
        if let Ok(scalar) = obj.extract::<bool>() {
            return Ok(Array::from_elem(IxDyn(&[]), scalar));
        }

        // Try as PyArray with dynamic dimensions
        if let Ok(arr) = array_buffer::extract_bool_array(obj) {
            return Ok(arr);
        }

        // Convert via numpy asarray
        let py = obj.py();
        let np = py.import("numpy")?;
        let asarray = np.getattr("asarray")?;
        let converted = asarray.call1((obj,))?;
        array_buffer::extract_bool_array(&converted)
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
    Ok(array_buffer::f64_array_to_py(py, &result).into())
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
    arr: ndarray::ArrayViewD<'_, T>,
    target_shape: &[usize],
) -> PyResult<ArrayD<T>> {
    use ndarray::IxDyn;

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
    stats_module.add_function(wrap_pyfunction!(weighted_mean, &stats_module)?)?;
    stats_module.add_function(wrap_pyfunction!(jackknife_resamples, &stats_module)?)?;
    stats_module.add_function(wrap_pyfunction!(jackknife_stats, &stats_module)?)?;
    stats_module.add_function(wrap_pyfunction!(jackknife_stats_axis, &stats_module)?)?;
    stats_module.add_function(wrap_pyfunction!(jackknife_weighted, &stats_module)?)?;
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
    compare_module.add_function(wrap_pyfunction!(assert_allclose, &compare_module)?)?;
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
    array_module.add_function(wrap_pyfunction!(max, &array_module)?)?;
    array_module.add_function(wrap_pyfunction!(min, &array_module)?)?;
    array_module.add_function(wrap_pyfunction!(asarray, &array_module)?)?;
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
    num_module.add_function(wrap_pyfunction!(mean_axis, &num_module)?)?;
    num_module.add_function(wrap_pyfunction!(std_axis, &num_module)?)?;

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
    num_module.add_function(wrap_pyfunction!(allclose, &num_module)?)?;
    num_module.add_function(wrap_pyfunction!(assert_allclose, &num_module)?)?;
    num_module.add_function(wrap_pyfunction!(array_equal, &num_module)?)?;
    num_module.add_function(wrap_pyfunction!(where_, &num_module)?)?;

    // Array functions (polymorphic)
    num_module.add_function(wrap_pyfunction!(sum, &num_module)?)?;
    num_module.add_function(wrap_pyfunction!(max, &num_module)?)?;
    num_module.add_function(wrap_pyfunction!(min, &num_module)?)?;
    num_module.add_function(wrap_pyfunction!(diag, &num_module)?)?;
    num_module.add_function(wrap_pyfunction!(linspace, &num_module)?)?;
    num_module.add_function(wrap_pyfunction!(arange, &num_module)?)?;
    num_module.add_function(wrap_pyfunction!(zeros, &num_module)?)?;
    num_module.add_function(wrap_pyfunction!(ones, &num_module)?)?;
    num_module.add_function(wrap_pyfunction!(array, &num_module)?)?;
    num_module.add_function(wrap_pyfunction!(asarray, &num_module)?)?;
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

    // f32 precision constants
    num_module.add("pi_f32", pecos::prelude::PI_F32)?;
    num_module.add("tau_f32", pecos::prelude::TAU_F32)?;
    num_module.add("e_f32", pecos::prelude::E_F32)?;
    num_module.add("inf_f32", f32::INFINITY)?;
    num_module.add("nan_f32", f32::NAN)?;
    num_module.add("FRAC_PI_2_F32", pecos::prelude::FRAC_PI_2_F32)?;
    num_module.add("FRAC_PI_3_F32", pecos::prelude::FRAC_PI_3_F32)?;
    num_module.add("FRAC_PI_4_F32", pecos::prelude::FRAC_PI_4_F32)?;
    num_module.add("FRAC_PI_6_F32", pecos::prelude::FRAC_PI_6_F32)?;
    num_module.add("FRAC_PI_8_F32", pecos::prelude::FRAC_PI_8_F32)?;
    num_module.add("FRAC_1_PI_F32", pecos::prelude::FRAC_1_PI_F32)?;
    num_module.add("FRAC_2_PI_F32", pecos::prelude::FRAC_2_PI_F32)?;
    num_module.add("FRAC_2_SQRT_PI_F32", pecos::prelude::FRAC_2_SQRT_PI_F32)?;
    num_module.add("SQRT_2_F32", pecos::prelude::SQRT_2_F32)?;
    num_module.add("FRAC_1_SQRT_2_F32", pecos::prelude::FRAC_1_SQRT_2_F32)?;
    num_module.add("LN_2_F32", pecos::prelude::LN_2_F32)?;
    num_module.add("LN_10_F32", pecos::prelude::LN_10_F32)?;
    num_module.add("LOG2_E_F32", pecos::prelude::LOG2_E_F32)?;
    num_module.add("LOG10_E_F32", pecos::prelude::LOG10_E_F32)?;

    // Add missing functions at top level
    num_module.add_function(wrap_pyfunction!(ln, &num_module)?)?;
    num_module.add_function(wrap_pyfunction!(self::log, &num_module)?)?;
    num_module.add_function(wrap_pyfunction!(abs, &num_module)?)?;
    num_module.add_function(wrap_pyfunction!(all, &num_module)?)?;
    num_module.add_function(wrap_pyfunction!(any, &num_module)?)?;
    num_module.add_function(wrap_pyfunction!(where_array, &num_module)?)?;

    m.add_submodule(&num_module)?;

    // Register num module and all submodules in sys.modules
    let py = m.py();
    let sys = py.import("sys")?;
    let modules = sys.getattr("modules")?;

    modules.set_item("_pecos_rslib.num", &num_module)?;
    modules.set_item("_pecos_rslib.num.stats", num_module.getattr("stats")?)?;
    modules.set_item("_pecos_rslib.num.math", num_module.getattr("math")?)?;
    modules.set_item("_pecos_rslib.num.compare", num_module.getattr("compare")?)?;
    modules.set_item("_pecos_rslib.num.array", num_module.getattr("array")?)?;
    modules.set_item("_pecos_rslib.num.optimize", num_module.getattr("optimize")?)?;
    modules.set_item(
        "_pecos_rslib.num.polynomial",
        num_module.getattr("polynomial")?,
    )?;
    modules.set_item(
        "_pecos_rslib.num.curve_fit",
        num_module.getattr("curve_fit")?,
    )?;
    modules.set_item("_pecos_rslib.num.random", num_module.getattr("random")?)?;

    // Add 'where' alias for where_
    num_module.setattr("where", num_module.getattr("where_")?)?;

    Ok(())
}
