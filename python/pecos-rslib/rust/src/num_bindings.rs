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

/// Register the num submodule with Python bindings.
pub fn register_num_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let num_module = PyModule::new(m.py(), "num")?;
    num_module.add_function(wrap_pyfunction!(brentq, &num_module)?)?;
    num_module.add_function(wrap_pyfunction!(newton, &num_module)?)?;
    num_module.add_function(wrap_pyfunction!(polyfit, &num_module)?)?;
    num_module.add_function(wrap_pyfunction!(curve_fit, &num_module)?)?;
    num_module.add_class::<Poly1d>()?;
    m.add_submodule(&num_module)?;
    Ok(())
}
