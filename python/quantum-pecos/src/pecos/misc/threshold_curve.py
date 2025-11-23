# Copyright 2018 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract
# DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Threshold curve analysis and fitting for quantum error correction.

This module provides utilities for analyzing and fitting threshold curves
in quantum error correction, including error rate scaling and critical
threshold determination.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pecos as pc

if TYPE_CHECKING:
    from collections.abc import Callable

    from pecos import (
        Array,
        f64,
    )


def func(
    x: tuple[Array[f64], Array[f64]],
    pth: float,
    v0: float,
    a: float,
    b: float,
    c: float,
) -> float | Array[f64]:
    """Fit error rates to determine threshold using polynomial expansion.

    Function that represents the curve to fit error rates to in order to determine the threshold. (see:
    arXiv:quant-ph/0207088).

    Probabilities are fine as long as p > 1/(4*distance). See paper by Watson and Barrett (arXiv:1312.5213).

    Args:
    ----
        x: Tuple of (p, dist) where p is the physical error rate and dist is the code distance.
        pth: Threshold error rate parameter to be fitted.
        v0: Critical exponent parameter for scaling behavior.
        a: Constant term coefficient in the polynomial expansion.
        b: Linear term coefficient in the polynomial expansion.
        c: Quadratic term coefficient in the polynomial expansion.

    """
    p, dist = x

    x = (p - pth) * pc.power(dist, 1.0 / v0)

    return a + b * x + c * pc.power(x, 2)


def func2(
    x: tuple[Array[f64], Array[f64]],
    pth: float,
    v0: float,
    a: float,
    b: float,
    c: float,
    d: float,
    u: float,
) -> float | Array[f64]:
    """Fit error rates with finite-size correction to determine threshold.

    Function that represents the curve to fit error rates to in order to determine the threshold. (see:
    arXiv:quant-ph/0207088).

    Probabilities are fine as long as p > 1/(4*distance). See paper by Watson and Barrett (arXiv:1312.5213).

    Args:
    ----
        x: Tuple of (p, dist) where p is the physical error rate and dist is the code distance.
        pth: Threshold error rate parameter to be fitted.
        v0: Critical exponent parameter for scaling behavior.
        a: Constant term coefficient in the polynomial expansion.
        b: Linear term coefficient in the polynomial expansion.
        c: Quadratic term coefficient in the polynomial expansion.
        d: Coefficient for the finite-size correction term.
        u: Exponent parameter for the finite-size correction term.

    """
    p, dist = x

    x = (p - pth) * pc.power(dist, 1.0 / v0)

    z = a + b * x + c * pc.power(x, 2)

    z += d * pc.power(dist, -1.0 / u)

    return z


def func3(
    x: tuple[Array[f64], Array[f64]],
    pth: float,
    v0: float,
    a: float,
    b: float,
    c: float,
    d: float,
    uodd: float,
    ueven: float,
) -> float | Array[f64]:
    """Fit error rates with odd/even distance corrections to determine threshold.

    Function that represents the curve to fit error rates to in order to determine the threshold. (see:
    arXiv:quant-ph/0207088).

    Probabilities are fine as long as p > 1/(4*distance). See paper by Watson and Barrett (arXiv:1312.5213).

    Args:
    ----
        x: Tuple of (p, dist) where p is the physical error rate and dist is the code distance.
        pth: Threshold error rate parameter to be fitted.
        v0: Critical exponent parameter for scaling behavior.
        a: Constant term coefficient in the polynomial expansion.
        b: Linear term coefficient in the polynomial expansion.
        c: Quadratic term coefficient in the polynomial expansion.
        d: Coefficient for the finite-size correction term.
        uodd: Exponent parameter for finite-size corrections at odd distances.
        ueven: Exponent parameter for finite-size corrections at even distances.

    """
    p, dist = x

    x = (p - pth) * pc.power(dist, 1.0 / v0)

    z = pc.where(
        bool(dist % 2),
        d * pc.power(dist, -1.0 / uodd),
        d * pc.power(dist, -1.0 / ueven),
    )

    z += a + b * x + c * pc.power(x, 2)

    return z


def func4(
    x: tuple[Array[f64], Array[f64]],
    pth: float,
    v0: float,
    a: float,
    b: float,
) -> float | Array[f64]:
    """Fit error rates using exponential decay to determine threshold.

    Function that represents the curve to fit error rates to in order to determine the threshold. (see:
    arXiv:quant-ph/0207088).

    Probabilities are fine as long as p > 1/(4*distance). See paper by Watson and Barrett (arXiv:1312.5213).

    Args:
    ----
        x: Tuple of (p, dist) where p is the physical error rate and dist is the code distance.
        pth: Threshold error rate parameter to be fitted.
        v0: Critical exponent parameter for scaling behavior.
        a: Amplitude coefficient for the exponential decay.
        b: Decay rate coefficient in the exponential function.

    """
    p, dist = x

    x = (p - pth) * pc.power(dist, 1.0 / v0)

    return a * pc.exp(-b * pc.power(x, v0))


def func5(
    x: tuple[Array[f64], Array[f64]],
    pth: float,
    v0: float,
    a: float,
    b: float,
    c: float,
    d: float,
) -> float | Array[f64]:
    """Fit error rates using cubic polynomial to determine threshold.

    Function that represents the curve to fit error rates to in order to determine the threshold. (see:
    arXiv:quant-ph/0207088).

    Probabilities are fine as long as p > 1/(4*distance). See paper by Watson and Barrett (arXiv:1312.5213).

    Args:
    ----
        x: Tuple of (p, dist) where p is the physical error rate and dist is the code distance.
        pth: Threshold error rate parameter to be fitted.
        v0: Critical exponent parameter for scaling behavior.
        a: Constant term coefficient in the polynomial expansion.
        b: Linear term coefficient in the polynomial expansion.
        c: Quadratic term coefficient in the polynomial expansion.
        d: Cubic term coefficient in the polynomial expansion.

    """
    p, dist = x

    x = (p - pth) * pc.power(dist, 1.0 / v0)

    return a + b * x + c * pc.power(x, 2) + d * pc.power(x, 3)


def func6(
    x: tuple[Array[f64], Array[f64]],
    a: float,
    pth: float,
) -> float | Array[f64]:
    """Fit error rates using power law relationship to determine threshold.

    Function that represents the curve to fit error rates to in order to determine the threshold. (see:
    arXiv:quant-ph/0207088).

    Probabilities are fine as long as p > 1/(4*distance). See paper by Watson and Barrett (arXiv:1312.5213).

    Args:
    ----
        x: Tuple of (p, dist) where p is the physical error rate and dist is the code distance.
        a: Amplitude coefficient for the power law relationship.
        pth: Threshold error rate parameter.

    """
    p, dist = x

    return a * pc.power(p / pth, dist / 2)


def threshold_fit(
    plist: Array[f64] | list[float],
    dlist: Array[f64] | list[float],
    plog: Array[f64] | list[float],
    func: Callable[..., float | Array[f64]],
    p0: Array[f64] | list[float],
    maxfev: int = 100000,
    **kwargs: float | bool | str | None,
) -> tuple[Array[f64], Array[f64]]:
    """Fit threshold curve to logical error rate data.

    Args:
    ----
        plist: List of ps.
        dlist: List of distances.
        plog: List of logical error rates.
        func: Function to fit to.
        p0: Initial guess for the parameters.
        maxfev: Maximum number of function evaluations.
        **kwargs: Additional keyword arguments passed to curve_fit.

    """
    popt, pcov = pc.curve_fit(func, (plist, dlist), plog, p0, maxfev=maxfev, **kwargs)

    var = pc.diag(pcov)
    stdev = pc.sqrt(var)

    return popt, stdev


def _jackknife_threshold_core(
    plist: Array[f64] | list[float],
    dlist: Array[f64] | list[float],
    plog: Array[f64] | list[float],
    func: Callable[..., float | Array[f64]],
    p0: Array[f64] | list[float],
    maxfev: int,
    resample_indices: list[list[int]],
    *,
    verbose: bool = True,
    verbose_labels: list[str] | None = None,
) -> tuple[Array[f64], Array[f64]]:
    """Core jackknife resampling implementation for threshold fitting.

    Args:
        plist: List of probability values.
        dlist: List of distance values.
        plog: List of logical error probabilities.
        func: Fitting function to use.
        p0: Initial parameter guess.
        maxfev: Maximum function evaluations.
        resample_indices: List of index lists, each specifying which indices to include in that resample.
        verbose: If True, print progress information.
        verbose_labels: Optional labels for verbose output.

    Returns:
        Tuple of (mean_parameters, std_parameters).
    """
    opt_list = []

    for i, indices in enumerate(resample_indices):
        p_copy = plist[indices]
        plog_copy = plog[indices]
        dlist_copy = dlist[indices]

        result = threshold_fit(p_copy, dlist_copy, plog_copy, func, p0, maxfev)
        opt_list.append(result[0].tolist())

        if verbose and verbose_labels:
            print(verbose_labels[i])
            print("parameter values:", result[0])
            print(f"parameter stds: {result[1]}\n")

    # Convert to PECOS array for jackknife_stats_axis
    opt_array = pc.array(opt_list)

    # Use pecos-num jackknife_stats_axis to compute stats for all parameters at once
    # axis=0 means compute stats down columns (each column is a parameter)
    means, stds = pc.stats.jackknife_stats_axis(opt_array, axis=0)

    print(f"Mean: {means}")
    print(f"Std: {stds}")

    return means, stds


def jackknife_pd(
    plist: Array[f64] | list[float],
    dlist: Array[f64] | list[float],
    plog: Array[f64] | list[float],
    func: Callable[..., float | Array[f64]],
    p0: Array[f64] | list[float],
    maxfev: int = 100000,
    *,
    verbose: bool = True,
) -> tuple[Array[f64], Array[f64]]:
    """Perform jackknife resampling for parameter and distance data.

    Uses leave-one-out resampling where each data point (p, d, plog) is removed in turn.

    Args:
        plist: List of probability values.
        dlist: List of distance values.
        plog: List of logical error probabilities.
        func: Fitting function to use.
        p0: Initial parameter guess.
        maxfev: Maximum function evaluations.
        verbose: If True, print progress information.

    Returns:
        Tuple of (mean_parameters, std_parameters).
    """
    n = len(plog)
    plist = pc.array(plist)
    dlist = pc.array(dlist)
    plog = pc.array(plog)

    # Generate leave-one-out resample indices
    resample_indices = [list(range(i)) + list(range(i + 1, n)) for i in range(n)]

    # Generate verbose labels
    verbose_labels = [
        f"removed index: {i}\np = {plist[i]}, d = {dlist[i]}" for i in range(n)
    ]

    return _jackknife_threshold_core(
        plist,
        dlist,
        plog,
        func,
        p0,
        maxfev,
        resample_indices,
        verbose=verbose,
        verbose_labels=verbose_labels if verbose else None,
    )


def jackknife_p(
    plist: Array[f64] | list[float],
    dlist: Array[f64] | list[float],
    plog: Array[f64] | list[float],
    func: Callable[..., float | Array[f64]],
    p0: Array[f64] | list[float],
    maxfev: int = 100000,
    *,
    verbose: bool = True,
) -> tuple[Array[f64], Array[f64]]:
    """Perform jackknife resampling by removing each unique probability value.

    Args:
        plist: List of probability values.
        dlist: List of distance values.
        plog: List of logical error probabilities.
        func: Fitting function to use.
        p0: Initial parameter guess.
        maxfev: Maximum function evaluations.
        verbose: If True, print progress information.

    Returns:
        Tuple of (mean_parameters, std_parameters).
    """
    plist = pc.array(plist)
    dlist = pc.array(dlist)
    plog = pc.array(plog)

    uplist = sorted(set(plist.tolist()))

    # Generate resample indices for each unique p value
    resample_indices = []
    verbose_labels = []

    for p_val in uplist:
        mask = plist != p_val
        indices = pc.where(mask)[0].tolist()
        resample_indices.append(indices)
        verbose_labels.append(f"removed p: {p_val}")

    return _jackknife_threshold_core(
        plist,
        dlist,
        plog,
        func,
        p0,
        maxfev,
        resample_indices,
        verbose=verbose,
        verbose_labels=verbose_labels if verbose else None,
    )


def jackknife_d(
    plist: Array[f64] | list[float],
    dlist: Array[f64] | list[float],
    plog: Array[f64] | list[float],
    func: Callable[..., float | Array[f64]],
    p0: Array[f64] | list[float],
    maxfev: int = 100000,
    *,
    verbose: bool = True,
) -> tuple[Array[f64], Array[f64]]:
    """Perform jackknife resampling by removing each unique distance value.

    Args:
        plist: List of probability values.
        dlist: List of distance values.
        plog: List of logical error probabilities.
        func: Fitting function to use.
        p0: Initial parameter guess.
        maxfev: Maximum function evaluations.
        verbose: If True, print progress information.

    Returns:
        Tuple of (mean_parameters, std_parameters).
    """
    plist = pc.array(plist)
    dlist = pc.array(dlist)
    plog = pc.array(plog)

    udlist = sorted(set(dlist.tolist()))

    # Generate resample indices for each unique d value
    resample_indices = []
    verbose_labels = []

    for d_val in udlist:
        mask = dlist != d_val
        indices = pc.where(mask)[0].tolist()
        resample_indices.append(indices)
        verbose_labels.append(f"removed d: {d_val}")

    return _jackknife_threshold_core(
        plist,
        dlist,
        plog,
        func,
        p0,
        maxfev,
        resample_indices,
        verbose=verbose,
        verbose_labels=verbose_labels if verbose else None,
    )


def get_est(
    value_is: list[float],
    label: str,
    *,
    verbose: bool = True,
) -> tuple[float, float]:
    """Calculate mean and standard deviation estimate.

    Args:
        value_is: List of values to analyze.
        label: Label for verbose output.
        verbose: If True, print the results.

    Returns:
        Tuple of (mean, standard_deviation).
    """
    v_est = sum(value_is) / len(value_is)
    v_est_std = pc.std(value_is)

    if verbose:
        print(f"{label}_est: {v_est} (mean) +- {v_est_std} (std)")

    return v_est, v_est_std


def get_i(
    result: dict[str, tuple[float, float]],
    symbol: str,
    value_list: list[float],
    *,
    verbose: bool = True,
) -> None:
    """Extract and append a value from results dictionary.

    Args:
        result: Dictionary containing parameter estimates.
        symbol: Key to extract from the results.
        value_list: List to append the extracted value to.
        verbose: If True, print the extracted value.
    """
    value_i = result[symbol][0]
    value_list.append(value_i)

    if verbose:
        print(f"{symbol}_i = {value_i}")
