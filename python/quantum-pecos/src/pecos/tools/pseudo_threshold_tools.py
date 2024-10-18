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

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq, curve_fit, newton

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


def pseudo_threshold_code_capacity(
    ps,
    distance,
    runs,
    qecc_class=None,
    error_gen=None,
    decoder_class=None,
    verbose=True,
    mode=1,
    deg=2,
    circuit_runner=None,
    plotting=False,
    basis=None,
):
    """Function that generates p_logical values given a list of physical errors (ps) and distance (ds).

    Args:
    ----
        ps:
        distance:
        runs:
        qecc_class:
        error_gen:
        decoder_class:
        verbose:
        mode:
        deg:
        circuit_runner:

    Returns:
    -------

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
        raise Exception('Mode "%s" is not handled!' % mode)

    ps = np.array(ps)

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
            print("Runtime: %s s" % time)

        if verbose:
            print("----")

        plog.append(logical_error_rate)

    plog = np.array(plog)

    if verbose:
        print("ps=", ps)
        print("plog=", plog)

    if plotting:
        find_polyfit(ps, plog, deg, verbose)

        plot(ps, plog, deg)

    # return {'plist': plist, 'distance': distance, 'plog': plog, 'opt': popt, 'std': stdev,
    # 'pseudo_threshold': pseudo_thr}
    return {"ps": ps, "distance": distance, "plog": plog}


def find_polyfit(ps, plog, deg, verbose=True):
    plist = np.array(ps)

    popt, pcov = np.polyfit(ps, plog, deg=deg, cov=True)

    var = np.diag(pcov)
    stdev = np.sqrt(var)

    if verbose:
        print("params=", popt)
        print("std=", stdev)

    pseudo_thr = find_pseudo(plist, plog, deg)

    if verbose:
        print("Pseudo-threshold: %s" % pseudo_thr)

    return pseudo_thr, popt, pcov


def find_uniscalefit(
    ps,
    plog,
    distance,
    p0=None,
    maxfev=1000000,
    verbose=True,
    **kwargs,
):
    plist = np.array(ps)
    dlist = ns2nsfit(distance, len(plist))

    popt, pcov = curve_fit(func, (plist, dlist), plog, p0, maxfev=maxfev, **kwargs)

    var = np.diag(pcov)
    stdev = np.sqrt(var)

    for v in var:
        if np.isnan(v):
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


def ns2nsfit(ns, num):
    """Returns a list of distances or ps for performing fits.

    If ds == 5 and num == 3:
        -> [5, 5, 5]

    If ds == [3, 5, 7] and num == 3:
        -> [3, 3, 3, 5, 5, 5, 7, 7, 7]

    Likewise for ps.

    Args:
    ----
        ds:
        num:

    Returns:
    -------

    """
    if isinstance(ns, int):
        return [ns] * num

    else:
        new_list = []

        for i in ns:
            new_list.extend([i] * num)
        return new_list


def find_pseudo(plist, plog, deg):
    """Determines the pseudo threshold from list of ps and plogs.

    Args:
    ----
        plist:
        plog:
        deg:

    Returns:
    -------
        float: The value of the pseudo-threshold.

    """
    popt = np.polyfit(plist, plog, deg=deg)
    poly = np.poly1d(popt)

    def fnc(x):
        return poly(x) - x

    try:
        pseudo_thr = brentq(fnc, 0, 1)
    except ValueError:
        pseudo_thr = newton(fnc, 0.05)

    return pseudo_thr


def plot(plist, plog, deg=2, figsize=(10, 10), p_start=None, p_end=None):
    """Args:
    ----
        plist:
        plog:
        deg(int): Degree of polynomial fit.
        figsize(tuple of int):
        axis_start(float): Where the x and y axes begin.
        axis_end(float): Where the x and y axes end.

    Returns:
    -------

    """
    if p_start is None:
        p_start = min(plog) * 0.9

    if p_end is None:
        p_end = max(plog) * 1.1

    pseudo_thr = find_pseudo(plist, plog, deg)

    popt, _ = np.polyfit(
        plist,
        plog,
        deg,
        cov=True,
    )  # C_z is estimated covariance matrix

    axis_start = p_start
    axis_end = p_end

    x = np.linspace(axis_start, axis_end, 1000)

    poly = np.poly1d(popt)
    yi = poly(x)

    # Do the plotting:
    fg, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_title("Pseudothreshold from Polynomial Fit of Degree %s" % deg, size=20)

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
        label="Pseudo-threshold (%s)" % pth,
    )
    plt.legend(fontsize=16)

    fg.canvas.draw()
    plt.show()
