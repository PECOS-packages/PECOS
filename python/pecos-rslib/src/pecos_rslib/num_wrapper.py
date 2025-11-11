"""Enhanced numerical functions with numpy compatibility.

This module provides Python wrappers around Rust-implemented numerical functions,
adding numpy-compatible features like axis parameter support.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pecos_rslib._pecos_rslib import num as _num_core

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


def mean(a: ArrayLike, axis: int | None = None) -> float | np.ndarray:
    """Calculate the arithmetic mean along the specified axis.

    Drop-in replacement for `numpy.mean()` supporting axis parameter.

    Args:
        a: Array-like input data
        axis: Axis or axes along which the means are computed. If None,
              compute the mean of the flattened array (default).

    Returns:
        Mean value(s). If axis is None, returns a scalar. Otherwise returns
        an array of means.

    Examples:
        >>> from pecos.num import mean
        >>>
        >>> # 1D array - simple mean
        >>> mean([1.0, 2.0, 3.0, 4.0, 5.0])
        3.0
        >>>
        >>> # Tuple averaging (error model use case)
        >>> p_meas = (0.01, 0.015, 0.02)
        >>> mean(p_meas)
        0.015
        >>>
        >>> # 2D array - mean over all elements
        >>> arr = [[1.0, 2.0], [3.0, 4.0]]
        >>> mean(arr)
        2.5
        >>>
        >>> # 2D array - mean along axis 0 (down columns)
        >>> mean(arr, axis=0)
        array([2., 3.])
        >>>
        >>> # 2D array - mean along axis 1 (across rows)
        >>> mean(arr, axis=1)
        array([1.5, 3.5])
    """
    # Convert to numpy array if not already
    arr = np.asarray(a, dtype=np.float64)

    # If axis is None, compute mean of flattened array
    if axis is None:
        flat = arr.ravel()
        return _num_core.mean(flat.tolist())

    # For specified axis, use numpy's axis handling
    # Move the specified axis to the end, then compute mean for each
    arr_moved = np.moveaxis(arr, axis, -1)
    original_shape = arr_moved.shape

    # Reshape to 2D: (all other dims, axis dim)
    arr_2d = arr_moved.reshape(-1, original_shape[-1])

    # Compute mean for each row using our Rust implementation
    means = np.array([_num_core.mean(row.tolist()) for row in arr_2d])

    # Reshape back to original shape (minus the averaged axis)
    result_shape = original_shape[:-1]
    if result_shape:
        means = means.reshape(result_shape)
    else:
        # If result is scalar, return as float
        means = float(means)

    return means


def std(a: ArrayLike, axis: int | None = None, ddof: int = 0) -> float | np.ndarray:
    """Calculate the standard deviation along the specified axis.

    Drop-in replacement for `numpy.std()` supporting axis and ddof parameters.

    Args:
        a: Array-like input data
        axis: Axis or axes along which the standard deviations are computed.
              If None, compute the std of the flattened array (default).
        ddof: Delta degrees of freedom. The divisor used in calculation is
              N - ddof, where N is the number of elements. Default is 0
              (population std). Use ddof=1 for sample std.

    Returns:
        Standard deviation value(s). If axis is None, returns a scalar.
        Otherwise returns an array of standard deviations.

    Examples:
        >>> from pecos.num import std
        >>>
        >>> # 1D array - population std
        >>> values = [1.0, 2.0, 3.0, 4.0, 5.0]
        >>> std(values, ddof=0)
        1.4142135623730951
        >>>
        >>> # 1D array - sample std
        >>> std(values, ddof=1)
        1.5811388300841898
        >>>
        >>> # 2D array - std over all elements
        >>> arr = [[1.0, 2.0], [3.0, 4.0]]
        >>> std(arr)
        1.118033988749895
        >>>
        >>> # 2D array - std along axis 0 (down columns)
        >>> std(arr, axis=0)
        array([1., 1.])
        >>>
        >>> # 2D array - std along axis 1 (across rows)
        >>> std(arr, axis=1)
        array([0.5, 0.5])
        >>>
        >>> # Jackknife analysis use case
        >>> parameter_estimates = [1.5, 1.6, 1.4, 1.5, 1.7]
        >>> uncertainty = std(parameter_estimates, ddof=0)
    """
    # Convert to numpy array if not already
    arr = np.asarray(a, dtype=np.float64)

    # If axis is None, compute std of flattened array
    if axis is None:
        flat = arr.ravel()
        return _num_core.std(flat.tolist(), ddof)

    # For specified axis, use numpy's axis handling
    # Move the specified axis to the end, then compute std for each
    arr_moved = np.moveaxis(arr, axis, -1)
    original_shape = arr_moved.shape

    # Reshape to 2D: (all other dims, axis dim)
    arr_2d = arr_moved.reshape(-1, original_shape[-1])

    # Compute std for each row using our Rust implementation
    stds = np.array([_num_core.std(row.tolist(), ddof) for row in arr_2d])

    # Reshape back to original shape (minus the averaged axis)
    result_shape = original_shape[:-1]
    if result_shape:
        stds = stds.reshape(result_shape)
    else:
        # If result is scalar, return as float
        stds = float(stds)

    return stds


# sum() is now fully polymorphic in Rust - no Python wrapper needed!
# Import directly from the Rust module
sum = _num_core.sum


# power() and sqrt() are fully polymorphic in Rust - no Python wrapper needed!
# Import directly from the Rust module
power = _num_core.math.power
sqrt = _num_core.math.sqrt


def where(
    condition: bool | ArrayLike, x: float | ArrayLike, y: float | ArrayLike
) -> float | np.ndarray:
    """Conditional selection based on a boolean condition.

    Drop-in replacement for `numpy.where(condition, x, y)`.
    Returns x if condition is true, otherwise returns y.
    Supports both scalar and array inputs with full broadcasting.

    Args:
        condition: Boolean condition (scalar or array)
        x: Value(s) to return if condition is true
        y: Value(s) to return if condition is false

    Returns:
        Selected value(s). If all inputs are scalars, returns a scalar.
        Otherwise returns an array with broadcasting applied.

    Examples:
        >>> from pecos.num import where
        >>>
        >>> # Scalar usage
        >>> where(True, 10.0, 20.0)
        10.0
        >>> where(False, 10.0, 20.0)
        20.0
        >>>
        >>> # Array usage
        >>> import numpy as np
        >>> cond = np.array([True, False, True, False])
        >>> x_arr = np.array([10.0, 20.0, 30.0, 40.0])
        >>> y_arr = np.array([100.0, 200.0, 300.0, 400.0])
        >>> where(cond, x_arr, y_arr)
        array([10.0, 200.0, 30.0, 400.0])
        >>>
        >>> # Broadcasting: scalar condition, array values
        >>> where(True, np.array([1.0, 2.0, 3.0]), np.array([10.0, 20.0, 30.0]))
        array([1., 2., 3.])
        >>>
        >>> # Broadcasting: array condition, scalar values
        >>> where(np.array([True, False, True]), 100.0, -100.0)
        array([100., -100., 100.])
    """
    # Rust handles all broadcasting - just pass through
    return _num_core.compare.where_array(condition, x, y)


# Expose all other num functions directly
brentq = _num_core.brentq
newton = _num_core.newton
polyfit = _num_core.polyfit
curve_fit = _num_core.curve_fit
# Direct imports from Rust for functions that don't need Python wrappers
# (These are already fully polymorphic in Rust)

# Direct exports that don't need wrappers
Poly1d = _num_core.Poly1d
diag = _num_core.diag
linspace = _num_core.linspace
# Note: arange has a wrapper function below for dtype inference
zeros = _num_core.zeros
ones = _num_core.ones
delete = _num_core.delete
floor = _num_core.floor
ceil = _num_core.ceil
round = _num_core.round


def arange(start, stop=None, step=1):
    """Return evenly spaced values within a given interval.

    Drop-in replacement for `numpy.arange()`.

    Values are generated in the half-open interval `[start, stop)` with the given step.
    This function mimics NumPy's dtype inference behavior:
    - If all arguments are integers, returns int64 array
    - If any argument is a float, returns float64 array

    Args:
        start: Start of interval (inclusive). If `stop` is None, this becomes the stop
               and start is set to 0.
        stop: End of interval (exclusive). Optional.
        step: Spacing between values. Default is 1.

    Returns:
        Array of evenly spaced values with dtype matching NumPy's behavior.

    Examples:
        >>> from pecos.num import arange
        >>>
        >>> # All integers → int64
        >>> arange(0, 10)
        array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])  # dtype: int64
        >>>
        >>> # Any float → float64
        >>> arange(0.0, 10)
        array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])  # dtype: float64
        >>>
        >>> # Single argument (like range)
        >>> arange(5)
        array([0, 1, 2, 3, 4])  # dtype: int64
    """
    # Handle single-argument case (like range)
    if stop is None:
        stop = start
        start = 0

    # Check if all arguments are integers to match NumPy's dtype inference
    # NumPy rule: all integers → int64, any float → float64
    # Use Python's built-in int/float types only (no numpy types)
    all_ints = (
        isinstance(start, int)
        and not isinstance(start, bool)
        and isinstance(stop, int)
        and not isinstance(stop, bool)
        and isinstance(step, int)
        and not isinstance(step, bool)
    )

    # Convert to floats for Rust function (which expects f64)
    result = _num_core.arange(float(start), float(stop), float(step))

    # Apply NumPy's dtype inference rule
    if all_ints:
        # Convert to int64 to match NumPy behavior
        #  We use the __array__() interface which returns a numpy array
        arr = np.asarray(result)
        return arr.astype(np.int64)
    else:
        # Keep as float64
        return result


# Re-export functions that are fully polymorphic in Rust (no wrappers needed)
exp = _num_core.math.exp
ln = _num_core.math.ln  # Natural logarithm (clearer than log for scientific community)
log = _num_core.math.log  # Logarithm with custom base: log(x, base)
cos = _num_core.math.cos
sin = _num_core.math.sin
tan = _num_core.math.tan
sinh = _num_core.math.sinh
cosh = _num_core.math.cosh
tanh = _num_core.math.tanh
asin = _num_core.math.asin
acos = _num_core.math.acos
atan = _num_core.math.atan
asinh = _num_core.math.asinh
acosh = _num_core.math.acosh
atanh = _num_core.math.atanh
atan2 = _num_core.math.atan2
abs = _num_core.math.abs
isnan = _num_core.compare.isnan
isclose = _num_core.compare.isclose
allclose = _num_core.compare.allclose
array_equal = _num_core.compare.array_equal
all = _num_core.compare.all  # Test if all elements are truthy
any = _num_core.compare.any  # Test if any element is truthy
# Note: where() has a Python wrapper function defined above for additional logic

# Mathematical constants - drop-in replacements for numpy.pi, math.pi, etc.
pi = _num_core.pi
tau = _num_core.tau
e = _num_core.e
inf = _num_core.inf
nan = _num_core.nan
FRAC_PI_2 = _num_core.FRAC_PI_2
FRAC_PI_3 = _num_core.FRAC_PI_3
FRAC_PI_4 = _num_core.FRAC_PI_4
FRAC_PI_6 = _num_core.FRAC_PI_6
FRAC_PI_8 = _num_core.FRAC_PI_8
FRAC_1_PI = _num_core.FRAC_1_PI
FRAC_2_PI = _num_core.FRAC_2_PI
FRAC_2_SQRT_PI = _num_core.FRAC_2_SQRT_PI
SQRT_2 = _num_core.SQRT_2
FRAC_1_SQRT_2 = _num_core.FRAC_1_SQRT_2
LN_2 = _num_core.LN_2
LN_10 = _num_core.LN_10
LOG2_E = _num_core.LOG2_E
LOG10_E = _num_core.LOG10_E

# Re-export submodules
random = _num_core.random
linalg = _num_core.linalg


__all__ = [
    # Functions
    "mean",
    "sum",
    "power",
    "sqrt",
    "exp",
    "ln",  # Natural logarithm
    "log",  # Logarithm with base
    "abs",
    "isnan",
    "cos",
    "sin",
    "tan",
    "sinh",
    "cosh",
    "tanh",
    "asin",
    "acos",
    "atan",
    "asinh",
    "acosh",
    "atanh",
    "atan2",
    "floor",
    "ceil",
    "round",
    "isclose",
    "allclose",
    "array_equal",
    "all",  # Boolean AND reduction
    "any",  # Boolean OR reduction
    "where",
    "std",
    "brentq",
    "newton",
    "polyfit",
    "curve_fit",
    "Poly1d",
    "diag",
    "linspace",
    "arange",
    "zeros",
    "ones",
    "delete",
    # Constants
    "pi",
    "tau",
    "e",
    "inf",
    "nan",
    "FRAC_PI_2",
    "FRAC_PI_3",
    "FRAC_PI_4",
    "FRAC_PI_6",
    "FRAC_PI_8",
    "FRAC_1_PI",
    "FRAC_2_PI",
    "FRAC_2_SQRT_PI",
    "SQRT_2",
    "FRAC_1_SQRT_2",
    "LN_2",
    "LN_10",
    "LOG2_E",
    "LOG10_E",
    # Submodules
    "random",
    "linalg",
]
