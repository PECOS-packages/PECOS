"""Pseudo-threshold analysis tools for quantum error correction.

This module provides tools for analyzing pseudo-thresholds in quantum
error correction codes, including curve fitting, data analysis, and
threshold estimation for various error models and code parameters.
"""

# Copyright 2018 The PECOS Developers
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

from __future__ import annotations

from typing import TYPE_CHECKING

import pecos as pc
from pecos.decoders import MWPM2D
from pecos.engines import circuit_runners
from pecos.error_models import XModel
from pecos.misc.threshold_curve import func
from pecos.qeccs import Surface4444
from pecos.tools.threshold_tools import (
    codecapacity_logical_rate,
    codecapacity_logical_rate2,
    codecapacity_logical_rate3,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import TypedDict

    from pecos import (
        Array,
        f64,
    )
    from pecos.engines.circuit_runners import Standard
    from pecos.protocols import Decoder, ErrorGenerator, QECCProtocol

    class PseudoThresholdResult(TypedDict):
        """Result from pseudo threshold calculations."""

        ps: Array[f64]
        distance: int
        plog: Array[f64]


def pseudo_threshold_code_capacity(
    ps: Sequence[float],
    distance: int,
    runs: int,
    qecc_class: type[QECCProtocol] | None = None,
    error_gen: type[ErrorGenerator] | None = None,
    decoder_class: type[Decoder] | None = None,
    *,
    verbose: bool = True,
    mode: int = 1,
    deg: int = 2,
    circuit_runner: Standard | None = None,
    plotting: bool = False,
    basis: str | None = None,
) -> PseudoThresholdResult:
    """Function that generates p_logical values given a list of physical errors (ps) and distance (ds).

    Args:
    ----
        ps: List of physical error probabilities to test.
        distance: The code distance to use for the simulation.
        runs: Number of Monte Carlo runs to perform for each error probability.
        qecc_class: The quantum error correcting code class to use (default: Surface4444).
        error_gen: The error generator for creating noise models (default: XModel).
        decoder_class: The decoder class to use for error correction (default: MWPM2D).
        verbose: If True, prints detailed progress and results.
        mode: The mode for logical rate calculation (1, 2, or 3).
        deg: Degree of the polynomial fit for pseudo-threshold calculation.
        circuit_runner: The circuit runner to use for simulations.
        plotting: If True, generates plots of the results.
        basis: The basis for logical measurements (e.g., 'X' or 'Z').
    """
    if circuit_runner is None:
        circuit_runner = circuit_runners.Standard()

    if error_gen is None:
        error_gen = XModel(model_level="code_capacity")

    if qecc_class is None:
        qecc_class = Surface4444

    if decoder_class is None:
        decoder_class = MWPM2D

    if mode == 1:
        determine_rate = codecapacity_logical_rate
    elif mode == 2:
        determine_rate = codecapacity_logical_rate2
    elif mode == 3:
        determine_rate = codecapacity_logical_rate3
    else:
        msg = f'Mode "{mode}" is not handled!'
        raise Exception(msg)

    ps = pc.array(ps)

    plog = []

    qecc = qecc_class(distance=distance)
    decoder = decoder_class(qecc)

    for p in ps:
        logical_error_rate, time = determine_rate(
            runs,
            qecc,
            distance,
            error_gen,
            error_params={"p": p},
            decoder=decoder,
            verbose=verbose,
            circuit_runner=circuit_runner,
            basis=basis,
        )
        if verbose and time:
            print(f"Runtime: {time} s")

        if verbose:
            print("----")

        plog.append(logical_error_rate)

    plog = pc.array(plog)

    if verbose:
        print("ps=", ps)
        print("plog=", plog)

    if plotting:
        find_polyfit(ps, plog, deg, verbose)

        plot(ps, plog, deg)

    # return {'plist': plist, 'distance': distance, 'plog': plog, 'opt': popt, 'std': stdev,
    # 'pseudo_threshold': pseudo_thr}
    return {"ps": ps, "distance": distance, "plog": plog}


def find_polyfit(
    ps: Sequence[float] | Array[f64],
    plog: Sequence[float] | Array[f64],
    deg: int,
    *,
    verbose: bool = True,
) -> tuple[float, Array, Array]:
    """Find polynomial fit for pseudo-threshold analysis.

    Performs polynomial fitting on error probability data to determine
    pseudo-threshold values for quantum error correction performance.

    Args:
        ps: Physical error probabilities.
        plog: Logarithm of logical error probabilities.
        deg: Degree of polynomial fit.
        verbose: Whether to print fitting results.

    Returns:
        Tuple of pseudo-threshold, fitted parameters, and covariance matrix.
    """
    plist = pc.array(ps)

    popt, pcov = pc.polyfit(ps, plog, deg=deg, cov=True)

    var = pc.diag(pcov)
    stdev = pc.sqrt(var)

    if verbose:
        print("params=", popt)
        print("std=", stdev)

    pseudo_thr = find_pseudo(plist, plog, deg)

    if verbose:
        print(f"Pseudo-threshold: {pseudo_thr}")

    return pseudo_thr, popt, pcov


def find_uniscalefit(
    ps: list[float] | Array,
    plog: list[float] | Array,
    distance: int,
    p0: list[float] | Array | None = None,
    maxfev: int = 1000000,
    *,
    verbose: bool = True,
    **kwargs: float | bool | str | None,
) -> tuple[float, float, float, float, Array, Array]:
    """Find universal scaling fit for pseudo-threshold analysis.

    Performs universal scaling function fitting to extract pseudo-threshold
    and critical exponent from quantum error correction data.

    Args:
        ps: Physical error probabilities.
        plog: Logarithm of logical error probabilities.
        distance: Code distance for scaling analysis.
        p0: Initial parameter guess for fitting.
        maxfev: Maximum function evaluations for curve fitting.
        verbose: Whether to print fitting results.
        **kwargs: Additional arguments for curve fitting.

    Returns:
        Tuple of pseudo-threshold, its standard deviation, critical exponent,
        its standard deviation, fitted parameters, and covariance matrix.

    Raises:
        Exception: If fitting fails to converge.
    """
    plist = pc.array(ps)
    dlist = ns2nsfit(distance, len(plist))

    popt, pcov = pc.curve_fit(func, (plist, dlist), plog, p0, maxfev=maxfev, **kwargs)

    var = pc.diag(pcov)
    stdev = pc.sqrt(var)

    for v in var:
        if pc.isnan(v):
            msg = "Was not able to find a good fit. Suggestion: Use `p0` to specify parameter guess."
            raise Exception(msg)

    pseudo_thr = popt[0]
    v0 = popt[1]
    pseudo_thr_std = stdev[0]
    v0_std = stdev[1]

    if verbose:
        print(f"pseudo-threshold: {pseudo_thr} +- {pseudo_thr_std} (1 stdev)")
        print(f"v0: {v0} +- {v0_std} (1 stdev)")

    return pseudo_thr, pseudo_thr_std, v0, v0_std, popt, pcov


def ns2nsfit(ns: Sequence[int], num: int) -> list[int]:
    """Returns a list of distances or ps for performing fits.

    If ds == 5 and num == 3:
        -> [5, 5, 5]

    If ds == [3, 5, 7] and num == 3:
        -> [3, 3, 3, 5, 5, 5, 7, 7, 7]

    Likewise for ps.

    Args:
    ----
        ns: Either a single integer or a list of integers (distances or probabilities).
        num: Number of times to repeat each element in the output list.
    """
    if isinstance(ns, int):
        return [ns] * num

    new_list = []

    for i in ns:
        new_list.extend([i] * num)
    return new_list


def find_pseudo(
    plist: Sequence[float] | Array[f64],
    plog: Sequence[float] | Array[f64],
    deg: int,
) -> float:
    """Determines the pseudo threshold from list of ps and plogs.

    Args:
    ----
        plist: List of physical error probabilities.
        plog: List of logical error probabilities corresponding to plist.
        deg: Degree of the polynomial fit to use.

    Returns:
    -------
        float: The value of the pseudo-threshold.

    """
    popt = pc.polyfit(plist, plog, deg=deg)
    poly = pc.Poly1d(popt)

    def fnc(x: float) -> float:
        return poly(x) - x

    try:
        pseudo_thr = pc.brentq(fnc, 0, 1)
    except ValueError:
        pseudo_thr = pc.newton(fnc, 0.05)

    return pseudo_thr


def plot(
    plist: Sequence[float] | Array[f64],
    plog: Sequence[float] | Array[f64],
    deg: int = 2,
    figsize: tuple[int, int] = (10, 10),
    p_start: float | None = None,
    p_end: float | None = None,
) -> None:
    """Plot pseudo-threshold curve with polynomial fit.

    Args:
    ----
        plist: List of physical error rates.
        plog: List of logical error rates.
        deg(int): Degree of polynomial fit.
        figsize(tuple of int): Figure size for the plot.
        p_start(float): Starting point for the plot axes. If None, automatically determined.
        p_end(float): Ending point for the plot axes. If None, automatically determined.
    """
    import matplotlib.pyplot as plt

    if p_start is None:
        p_start = min(plog) * 0.9

    if p_end is None:
        p_end = max(plog) * 1.1

    pseudo_thr = find_pseudo(plist, plog, deg)

    popt, _ = pc.polyfit(
        plist,
        plog,
        deg,
        cov=True,
    )  # C_z is estimated covariance matrix

    axis_start = p_start
    axis_end = p_end

    x = pc.linspace(axis_start, axis_end, 1000)

    poly = pc.Poly1d(popt)
    yi = poly(x)

    # Do the plotting:
    fg, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_title(f"Pseudothreshold from Polynomial Fit of Degree {deg}", size=20)

    ax.plot(x, yi, "-")
    ax.plot(plist, plog, "ro")
    ax.axis("tight")

    y = x
    plt.plot(x, y, "k-", alpha=0.30)

    ax.set_ylim(axis_start, axis_end)
    ax.set_xlim(axis_start, axis_end)

    plt.xlabel("Physical error rate", size=18)
    plt.ylabel("Logical error rate", size=18)

    pth = pseudo_thr
    plt.axvline(
        pth,
        color="green",
        linewidth=2,
        linestyle="dashed",
        label=f"Pseudo-threshold ({pth})",
    )
    plt.legend(fontsize=16)

    fg.canvas.draw()
    plt.show()
