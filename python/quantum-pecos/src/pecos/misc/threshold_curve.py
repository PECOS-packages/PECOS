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

import numpy as np
from pecos_rslib.num import curve_fit

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


def func(
    x: tuple[NDArray[np.float64], NDArray[np.float64]],
    pth: float,
    v0: float,
    a: float,
    b: float,
    c: float,
) -> float | NDArray[np.float64]:
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

    x = (p - pth) * np.power(dist, 1.0 / v0)

    return a + b * x + c * np.power(x, 2)


def func2(
    x: tuple[NDArray[np.float64], NDArray[np.float64]],
    pth: float,
    v0: float,
    a: float,
    b: float,
    c: float,
    d: float,
    u: float,
) -> float | NDArray[np.float64]:
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

    x = (p - pth) * np.power(dist, 1.0 / v0)

    z = a + b * x + c * np.power(x, 2)

    z += d * np.power(dist, -1.0 / u)

    return z


def func3(
    x: tuple[NDArray[np.float64], NDArray[np.float64]],
    pth: float,
    v0: float,
    a: float,
    b: float,
    c: float,
    d: float,
    uodd: float,
    ueven: float,
) -> float | NDArray[np.float64]:
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

    x = (p - pth) * np.power(dist, 1.0 / v0)

    z = np.where(
        bool(dist % 2),
        d * np.power(dist, -1.0 / uodd),
        d * np.power(dist, -1.0 / ueven),
    )

    z += a + b * x + c * np.power(x, 2)

    return z


def func4(
    x: tuple[NDArray[np.float64], NDArray[np.float64]],
    pth: float,
    v0: float,
    a: float,
    b: float,
) -> float | NDArray[np.float64]:
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

    x = (p - pth) * np.power(dist, 1.0 / v0)

    return a * np.exp(-b * np.power(x, v0))


def func5(
    x: tuple[NDArray[np.float64], NDArray[np.float64]],
    pth: float,
    v0: float,
    a: float,
    b: float,
    c: float,
    d: float,
) -> float | NDArray[np.float64]:
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

    x = (p - pth) * np.power(dist, 1.0 / v0)

    return a + b * x + c * np.power(x, 2) + d * np.power(x, 3)


def func6(
    x: tuple[NDArray[np.float64], NDArray[np.float64]],
    a: float,
    pth: float,
) -> float | NDArray[np.float64]:
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

    return a * np.power(p / pth, dist / 2)


def threshold_fit(
    plist: NDArray[np.float64] | list[float],
    dlist: NDArray[np.float64] | list[float],
    plog: NDArray[np.float64] | list[float],
    func: Callable[..., float | NDArray[np.float64]],
    p0: NDArray[np.float64] | list[float],
    maxfev: int = 100000,
    **kwargs: float | bool | str | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
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
    popt, pcov = curve_fit(func, (plist, dlist), plog, p0, maxfev=maxfev, **kwargs)

    var = np.diag(pcov)
    stdev = np.sqrt(var)

    return popt, stdev


def jackknife_pd(
    plist: NDArray[np.float64] | list[float],
    dlist: NDArray[np.float64] | list[float],
    plog: NDArray[np.float64] | list[float],
    func: Callable[..., float | NDArray[np.float64]],
    p0: NDArray[np.float64] | list[float],
    maxfev: int = 100000,
    *,
    verbose: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Perform jackknife resampling for parameter and distance data.

    Args:
        plist: List of probability values.
        dlist: List of distance values.
        plog: List of logical error probabilities.
        func: Fitting function to use.
        p0: Initial parameter guess.
        maxfev: Maximum function evaluations.
        verbose: If True, print progress information.

    Returns:
        Tuple of (optimized_parameters, covariance_matrices).
    """
    opt_list = []
    cov_list = []
    for i in range(len(plog)):
        p_copy = np.delete(plist, i)
        plog_copy = np.delete(plog, i)
        dlist_copy = np.delete(dlist, i)

        result = threshold_fit(p_copy, dlist_copy, plog_copy, func, p0, maxfev)
        opt_list.append(result[0])
        cov_list.append(result[1])

        if verbose:
            print(f"removed index: {i}")
            print(f"p = {plist[i]}, d = {dlist[i]}")
            print("parameter values:", result[0])
            print(f"parameter stds: {result[1]}\n")

    est = np.mean(opt_list, axis=0)
    std = np.std(opt_list, axis=0)

    print(f"Mean: {est}")
    print(f"Std: {std}")

    return est, std


def jackknife_p(
    plist: NDArray[np.float64] | list[float],
    dlist: NDArray[np.float64] | list[float],
    plog: NDArray[np.float64] | list[float],
    func: Callable[..., float | NDArray[np.float64]],
    p0: NDArray[np.float64] | list[float],
    maxfev: int = 100000,
    *,
    verbose: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
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
    opt_list = []
    cov_list = []
    uplist = sorted(set(plist))
    for p in uplist:
        mask = plist != p
        p_copy = plist[mask]
        plog_copy = plog[mask]
        dlist_copy = dlist[mask]

        result = threshold_fit(p_copy, dlist_copy, plog_copy, func, p0, maxfev)
        opt_list.append(result[0])
        cov_list.append(result[1])

        if verbose:
            print(f"removed p: {p}")
            print("parameter values:", result[0])
            print(f"parameter stds: {result[1]}\n")

    est = np.mean(opt_list, axis=0)
    std = np.std(opt_list, axis=0)

    print(f"Mean: {est}")
    print(f"Std: {std}")

    return est, std


def jackknife_d(
    plist: NDArray[np.float64] | list[float],
    dlist: NDArray[np.float64] | list[float],
    plog: NDArray[np.float64] | list[float],
    func: Callable[..., float | NDArray[np.float64]],
    p0: NDArray[np.float64] | list[float],
    maxfev: int = 100000,
    *,
    verbose: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
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
    opt_list = []
    cov_list = []

    udlist = sorted(set(dlist))
    for d in udlist:
        mask = dlist != d
        p_copy = plist[mask]
        plog_copy = plog[mask]
        dlist_copy = dlist[mask]

        result = threshold_fit(p_copy, dlist_copy, plog_copy, func, p0, maxfev)
        opt_list.append(result[0])
        cov_list.append(result[1])

        if verbose:
            print(f"removed d: {d}")
            print("parameter values:", result[0])
            print(f"parameter stds: {result[1]}\n")

    est = np.mean(opt_list, axis=0)
    std = np.std(opt_list, axis=0)

    print(f"Mean: {est}")
    print(f"Std: {std}")

    return est, std


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
    v_est_std = np.std(value_is)

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
