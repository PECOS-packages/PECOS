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

import numpy as np
from scipy.optimize import curve_fit


def func(x, pth, v0, a, b, c):
    """Function that represents the curve to fit error rates to in order to determine the threshold. (see:
    arXiv:quant-ph/0207088).

    Probabilities are fine as long as p > 1/(4*distance). See paper by Watson and Barrett (arXiv:1312.5213).

    Args:
    ----
        x:
        a:
        b:
        c:
        pth:
        v0:

    Returns:
    -------

    """
    p, dist = x

    x = (p - pth) * np.power(dist, 1.0 / v0)

    return a + b * x + c * np.power(x, 2)


def func2(x, pth, v0, a, b, c, d, u):
    """Function that represents the curve to fit error rates to in order to determine the threshold. (see:
    arXiv:quant-ph/0207088).

    Probabilities are fine as long as p > 1/(4*distance). See paper by Watson and Barrett (arXiv:1312.5213).

    Args:
    ----
        x:
        a:
        b:
        c:
        pth:
        v0:

    Returns:
    -------

    """
    p, dist = x

    x = (p - pth) * np.power(dist, 1.0 / v0)

    z = a + b * x + c * np.power(x, 2)

    z += d * np.power(dist, -1.0 / u)

    return z


def func3(x, pth, v0, a, b, c, d, uodd, ueven):
    """Function that represents the curve to fit error rates to in order to determine the threshold. (see:
    arXiv:quant-ph/0207088).

    Probabilities are fine as long as p > 1/(4*distance). See paper by Watson and Barrett (arXiv:1312.5213).

    Args:
    ----
        x:
        a:
        b:
        c:
        pth:
        v0:

    Returns:
    -------

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


def func4(x, pth, v0, a, b):
    """Function that represents the curve to fit error rates to in order to determine the threshold. (see:
    arXiv:quant-ph/0207088).

    Probabilities are fine as long as p > 1/(4*distance). See paper by Watson and Barrett (arXiv:1312.5213).

    Args:
    ----
        x:
        a:
        b:
        c:
        pth:
        v0:

    Returns:
    -------

    """
    p, dist = x

    x = (p - pth) * np.power(dist, 1.0 / v0)

    return a * np.exp(-b * np.power(x, v0))


def func5(x, pth, v0, a, b, c, d):
    """Function that represents the curve to fit error rates to in order to determine the threshold. (see:
    arXiv:quant-ph/0207088).

    Probabilities are fine as long as p > 1/(4*distance). See paper by Watson and Barrett (arXiv:1312.5213).

    Args:
    ----
        x:
        a:
        b:
        c:
        pth:
        v0:

    Returns:
    -------

    """
    p, dist = x

    x = (p - pth) * np.power(dist, 1.0 / v0)

    return a + b * x + c * np.power(x, 2) + d * np.power(x, 3)


def func6(x, a, pth):
    """Function that represents the curve to fit error rates to in order to determine the threshold. (see:
    arXiv:quant-ph/0207088).

    Probabilities are fine as long as p > 1/(4*distance). See paper by Watson and Barrett (arXiv:1312.5213).

    Args:
    ----
        x:
        a:
        b:
        c:
        pth:
        v0:

    Returns:
    -------

    """
    p, dist = x

    return a * np.power(p / pth, dist / 2)


def threshold_fit(plist, dlist, plog, func, p0, maxfev=100000, **kwargs):
    """Args:
    ----
        plist: List of ps.
        dlist: List of distances.
        plog: List of logical error rates.
        func: Function to fit to.
        maxfev:

    Returns:
    -------

    """
    popt, pcov = curve_fit(func, (plist, dlist), plog, p0, maxfev=maxfev, **kwargs)

    var = np.diag(pcov)
    stdev = np.sqrt(var)

    return popt, stdev


def jackknife_pd(plist, dlist, plog, func, p0, maxfev=100000, verbose=True):
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
            print("removed index: %s" % i)
            print(f"p = {plist[i]}, d = {dlist[i]}")
            print("parameter values:", result[0])
            print("parameter stds: %s\n" % result[1])

    est = np.mean(opt_list, axis=0)
    std = np.std(opt_list, axis=0)

    print("Mean: %s" % est)
    print("Std: %s" % std)

    return est, std


def jackknife_p(plist, dlist, plog, p0, maxfev=100000, verbose=True):
    opt_list = []
    cov_list = []
    uplist = sorted(set(plist))
    for p in uplist:
        mask = plist != p
        p_copy = plist[mask]
        plog_copy = plog[mask]
        dlist_copy = dlist[mask]

        result = threshold_fit(p_copy, dlist_copy, plog_copy, p0, maxfev)
        opt_list.append(result[0])
        cov_list.append(result[1])

        if verbose:
            print("removed p: %s" % p)
            print("parameter values:", result[0])
            print("parameter stds: %s\n" % result[1])

    est = np.mean(opt_list, axis=0)
    std = np.std(opt_list, axis=0)

    print("Mean: %s" % est)
    print("Std: %s" % std)

    return est, std


def jackknife_d(plist, dlist, plog, p0, maxfev=100000, verbose=True):
    opt_list = []
    cov_list = []

    udlist = sorted(set(dlist))
    for d in udlist:
        mask = dlist != d
        p_copy = plist[mask]
        plog_copy = plog[mask]
        dlist_copy = dlist[mask]

        result = threshold_fit(p_copy, dlist_copy, plog_copy, p0, maxfev)
        opt_list.append(result[0])
        cov_list.append(result[1])

        if verbose:
            print("removed d: %s" % d)
            print("parameter values:", result[0])
            print("parameter stds: %s\n" % result[1])

    est = np.mean(opt_list, axis=0)
    std = np.std(opt_list, axis=0)

    print("Mean: %s" % est)
    print("Std: %s" % std)

    return est, std


def get_est(value_is, label, verbose=True):
    v_est = sum(value_is) / len(value_is)
    v_est_std = np.std(value_is)

    if verbose:
        print(f"{label}_est: {v_est} (mean) +- {v_est_std} (std)")

    return v_est, v_est_std


def get_i(result, symbol, value_list, verbose=True):
    value_i = result[symbol][0]
    value_list.append(value_i)

    if verbose:
        print(f"{symbol}_i = {value_i}")
