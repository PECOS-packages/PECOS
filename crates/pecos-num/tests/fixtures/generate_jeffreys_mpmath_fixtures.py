# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain a
# copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.
"""Offline generator for the mpmath second-oracle Jeffreys fixture table.

Independent of the primary scipy oracle by construction: mpmath evaluates the
regularized incomplete beta in arbitrary precision (30 significant digits
here) rather than through the TOMS-708 lineage scipy and R share, so an
agreement between both oracles and pecos-num rules out a shared-lineage bug.
scipy is used ONLY to seed the root find; every returned quantile is verified
by an mpmath residual check at full precision.

Run offline with:

    uv run --with mpmath --with scipy python generate_jeffreys_mpmath_fixtures.py \
        > jeffreys_mpmath.csv

The case grid is a fixed-seed randomized sweep over the full supported regime
(n in [1, 1e8], k in [0, n], alpha in [1e-6, 0.5]) with structured k patterns
(endpoints, near-endpoints, typical-LER counts, midpoints) so the table keeps
covering the regime-gate seams between the hand-picked scipy rows.
"""

from __future__ import annotations

import random
import sys

from mpmath import mp, mpf
from scipy.stats import beta as scipy_beta

mp.dps = 30
RESIDUAL_LIMIT = mpf("1e-24")


def log_density(a: mpf, b: mpf, x: mpf) -> mpf:
    """Log of the Beta(a, b) density at x, in arbitrary precision."""
    return (a - 1) * mp.log(x) + (b - 1) * mp.log1p(-x) - mp.log(mp.beta(a, b))


def beta_cdf(a: mpf, b: mpf, x: mpf) -> mpf:
    """Regularized incomplete beta by arbitrary-precision quadrature ONLY.

    mpmath's betainc was found to return silently inaccurate values (no
    exception) for large-shape near-saturation inputs, which disqualifies it
    as an oracle. Direct quadrature of the density -- the definition itself,
    no series machinery -- with the interval split at the mode, integrating
    whichever tail is smaller and complementing.
    """
    if x <= 0:
        return mpf(0)
    if x >= 1:
        return mpf(1)
    density = lambda t: mp.exp(log_density(a, b, t))  # noqa: E731
    mode = (a - 1) / (a + b - 2) if a + b > 2 else mpf("0.5")
    if not 0 < mode < 1:
        mode = mpf("0.5")
    if x <= mode:
        return mp.quad(density, [mpf(0), x])
    return 1 - mp.quad(density, [x, mpf(1)])


def quantile_lower_side(a: float, b: float, p: mpf, seed: float) -> mpf:
    """Solve I_x(a, b) = p for x on the well-conditioned lower side."""
    # Bracket-guarded Newton on the quadrature CDF: the float64 scipy seed is
    # already ~1e-16 relative, so 2-3 quadratically-converging steps reach far
    # below RESIDUAL_LIMIT with only a handful of expensive quadrature calls.
    # The bracket (updated from every evaluation's sign) catches any bad step.
    am, bm = mpf(a), mpf(b)
    lo, hi = mpf(0), mpf(1)
    x = mpf(seed) if 0.0 < seed < 1.0 else mpf("0.5")
    residual = None
    for _ in range(8):
        f = beta_cdf(am, bm, x) - p
        residual = abs(f)
        if f < 0:
            lo = x
        else:
            hi = x
        if residual <= RESIDUAL_LIMIT:
            return x
        step = f * mp.exp(-log_density(am, bm, x))
        candidate = x - step
        if not lo < candidate < hi:
            candidate = (lo + hi) / 2
        x = candidate
    msg = f"residual {residual} too large for a={a}, b={b}, p={p}"
    raise RuntimeError(msg)


def quantile(a: float, b: float, p: float) -> mpf:
    """Beta(a, b) quantile at p, solved on whichever side is well conditioned."""
    # Root-find on whichever side of the distribution is well conditioned:
    # near x = 1 the CDF derivative vanishes, so solve the mirrored problem
    # I_y(b, a) = 1 - p for the small complement y instead (same complement
    # strategy the Rust implementation uses).
    seed = float(scipy_beta.ppf(p, a, b))
    if seed > 0.5:
        return 1 - quantile_lower_side(b, a, 1 - mpf(p), 1.0 - seed)
    return quantile_lower_side(a, b, mpf(p), seed)


def emit(k: int, n: int, alpha: float) -> str:
    """Format one CSV row of mpmath-oracle interval bounds and median."""
    a = k + 0.5
    b = n - k + 0.5
    lo = mpf(0) if k == 0 else quantile(a, b, alpha / 2)
    hi = mpf(1) if k == n else quantile(a, b, 1 - alpha / 2)
    median = quantile(a, b, 0.5)
    return f"{k},{n},{alpha!r},{float(lo)!r},{float(hi)!r},{float(median)!r}"


def k_patterns(rng: random.Random, n: int) -> list[int]:
    """Return the structured + random success counts probed for one n."""
    picks = {0, n, min(1, n), max(n - 1, 0), n // 2}
    picks.add(max(1, round(n * 0.001)))  # typical logical-error-rate count
    picks.add(rng.randint(0, n))
    return sorted(picks)


def main() -> None:
    """Print the randomized second-oracle CSV to stdout."""
    rng = random.Random(20260707)
    print("k,n,alpha,lo,hi,median")
    seen: set[tuple[int, int, float]] = set()
    for _ in range(40):
        n = round(10 ** rng.uniform(0.0, 8.0))
        n = max(1, min(n, 100_000_000))
        alpha = 10 ** rng.uniform(-6.0, -0.301)  # alpha in [1e-6, 0.5]
        for k in k_patterns(rng, n):
            case = (k, n, alpha)
            if case not in seen:
                seen.add(case)
                print(emit(k, n, alpha))


if __name__ == "__main__":
    sys.exit(main())
