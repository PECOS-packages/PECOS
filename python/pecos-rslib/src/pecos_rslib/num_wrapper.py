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
        >>> from pecos_rslib.num import mean
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
        >>> from pecos_rslib.num import std
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


def power(x1: ArrayLike, x2: ArrayLike) -> float | np.ndarray:
    """Calculate the power of x1 raised to x2, element-wise.

    Drop-in replacement for `numpy.power()` supporting broadcasting.

    Args:
        x1: The bases (array-like)
        x2: The exponents (array-like)

    Returns:
        Element-wise power x1**x2. If both inputs are scalars, returns a scalar.
        Otherwise returns an array with broadcasting applied.

    Examples:
        >>> from pecos_rslib.num import power
        >>>
        >>> # Scalar inputs
        >>> power(2.0, 3.0)
        8.0
        >>>
        >>> # Array base, scalar exponent
        >>> power([1.0, 2.0, 3.0], 2.0)
        array([1., 4., 9.])
        >>>
        >>> # Scalar base, array exponent
        >>> power(2.0, [1.0, 2.0, 3.0])
        array([2., 4., 8.])
        >>>
        >>> # Threshold curve use case
        >>> dist = 5.0
        >>> v0 = 2.0
        >>> power(dist, 1.0 / v0)
        2.23606797749979
    """
    # Convert to numpy arrays
    arr1 = np.asarray(x1, dtype=np.float64)
    arr2 = np.asarray(x2, dtype=np.float64)

    # Check if both are scalars
    if arr1.ndim == 0 and arr2.ndim == 0:
        return _num_core.power(float(arr1), float(arr2))

    # Use numpy broadcasting for array operations
    # For arrays, use our Rust implementation element-wise
    result_shape = np.broadcast_shapes(arr1.shape, arr2.shape)

    # Broadcast arrays to common shape
    arr1_broadcast = np.broadcast_to(arr1, result_shape)
    arr2_broadcast = np.broadcast_to(arr2, result_shape)

    # Flatten and compute element-wise
    flat1 = arr1_broadcast.ravel()
    flat2 = arr2_broadcast.ravel()

    # Compute using Rust implementation
    result = np.array(
        [
            _num_core.power(float(b), float(e))
            for b, e in zip(flat1, flat2, strict=False)
        ]
    )

    # Reshape to result shape
    return result.reshape(result_shape) if result_shape else float(result)


# Expose all other num functions directly
brentq = _num_core.brentq
newton = _num_core.newton
polyfit = _num_core.polyfit
curve_fit = _num_core.curve_fit
Poly1d = _num_core.Poly1d

# Re-export the random submodule
random = _num_core.random


__all__ = [
    "mean",
    "power",
    "std",
    "brentq",
    "newton",
    "polyfit",
    "curve_fit",
    "Poly1d",
    "random",
]
