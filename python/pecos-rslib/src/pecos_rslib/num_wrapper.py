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

    # Use Rust's native axis implementation for massive performance improvement
    result = _num_core.mean_axis(arr, axis)

    # Convert to numpy array for consistency with numpy.mean behavior
    return np.asarray(result)


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

    # Use Rust's native axis implementation for massive performance improvement
    result = _num_core.std_axis(arr, axis, ddof)

    # Convert to numpy array for consistency with numpy.std behavior
    return np.asarray(result)


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
# arange() now handles dtype inference in Rust - no Python wrapper needed!
# The Rust implementation checks if all arguments are integers and returns
# int64 or float64 accordingly, matching NumPy's behavior exactly.
arange = _num_core.arange


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

# Statistical functions - drop-in replacements for numpy.max(), numpy.min()
max = _num_core.max  # Maximum value
min = _num_core.min  # Minimum value

# Mathematical constants - drop-in replacements for numpy.pi, math.pi, etc.
# Following NumPy/SciPy convention: lowercase names for constants
pi = _num_core.pi
tau = _num_core.tau
e = _num_core.e
inf = _num_core.inf
nan = _num_core.nan
# Pi fractions - lowercase for NumPy-style consistency
frac_pi_2 = _num_core.FRAC_PI_2
frac_pi_3 = _num_core.FRAC_PI_3
frac_pi_4 = _num_core.FRAC_PI_4
frac_pi_6 = _num_core.FRAC_PI_6
frac_pi_8 = _num_core.FRAC_PI_8
frac_1_pi = _num_core.FRAC_1_PI
frac_2_pi = _num_core.FRAC_2_PI
frac_2_sqrt_pi = _num_core.FRAC_2_SQRT_PI
# Square root constants
sqrt_2 = _num_core.SQRT_2
frac_1_sqrt_2 = _num_core.FRAC_1_SQRT_2
# Logarithmic constants
ln_2 = _num_core.LN_2
ln_10 = _num_core.LN_10
log2_e = _num_core.LOG2_E
log10_e = _num_core.LOG10_E

# f32 Mathematical constants - single precision
# Following NumPy/SciPy convention: lowercase names for constants
pi_f32 = _num_core.pi_f32
tau_f32 = _num_core.tau_f32
e_f32 = _num_core.e_f32
inf_f32 = _num_core.inf_f32
nan_f32 = _num_core.nan_f32
# Pi fractions - lowercase for NumPy-style consistency
frac_pi_2_f32 = _num_core.FRAC_PI_2_F32
frac_pi_3_f32 = _num_core.FRAC_PI_3_F32
frac_pi_4_f32 = _num_core.FRAC_PI_4_F32
frac_pi_6_f32 = _num_core.FRAC_PI_6_F32
frac_pi_8_f32 = _num_core.FRAC_PI_8_F32
frac_1_pi_f32 = _num_core.FRAC_1_PI_F32
frac_2_pi_f32 = _num_core.FRAC_2_PI_F32
frac_2_sqrt_pi_f32 = _num_core.FRAC_2_SQRT_PI_F32
# Square root constants
sqrt_2_f32 = _num_core.SQRT_2_F32
frac_1_sqrt_2_f32 = _num_core.FRAC_1_SQRT_2_F32
# Logarithmic constants
ln_2_f32 = _num_core.LN_2_F32
ln_10_f32 = _num_core.LN_10_F32
log2_e_f32 = _num_core.LOG2_E_F32
log10_e_f32 = _num_core.LOG10_E_F32

# Re-export submodules
random = _num_core.random
linalg = _num_core.linalg

# Jackknife resampling functions - direct imports from Rust
# These don't need Python wrappers - Rust handles PECOS Arrays directly
weighted_mean = _num_core.stats.weighted_mean
jackknife_resamples = _num_core.stats.jackknife_resamples
jackknife_stats = _num_core.stats.jackknife_stats
jackknife_stats_axis = _num_core.stats.jackknife_stats_axis
jackknife_weighted = _num_core.stats.jackknife_weighted

# Re-export stats module directly - no wrapper needed
stats = _num_core.stats


__all__ = [
    # Functions
    "mean",
    "sum",
    "max",
    "min",
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
    "weighted_mean",
    "jackknife_resamples",
    "jackknife_stats",
    "jackknife_stats_axis",
    "jackknife_weighted",
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
    # Constants (lowercase for NumPy compatibility)
    "pi",
    "tau",
    "e",
    "inf",
    "nan",
    "frac_pi_2",
    "frac_pi_3",
    "frac_pi_4",
    "frac_pi_6",
    "frac_pi_8",
    "frac_1_pi",
    "frac_2_pi",
    "frac_2_sqrt_pi",
    "sqrt_2",
    "frac_1_sqrt_2",
    "ln_2",
    "ln_10",
    "log2_e",
    "log10_e",
    # Submodules
    "random",
    "linalg",
    "stats",
]
