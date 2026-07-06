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
"""Offline generator for the Jeffreys-interval oracle fixture table.

scipy.stats.beta.ppf (TOMS-708 lineage) is the oracle of record per
design/jeffreys-pecos-num.md in the pecos-docs vault. Run offline with:

    uv run --with scipy python generate_jeffreys_fixtures.py > jeffreys_scipy.csv

scipy is never a runtime dependency; this script exists so the pinned CSV is
reproducible. The adversarial case grid follows the design note's v3 fixture
table (endpoint boundaries, n=1e8 band, typical-LER regime, alpha extremes
within the supported regime alpha in [1e-6, 0.5]).
"""

from __future__ import annotations

import sys

from scipy.stats import beta


def jeffreys_row(k: int, n: int, alpha: float) -> str:
    a = k + 0.5
    b = n - k + 0.5
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, a, b))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, a, b))
    median = float(beta.ppf(0.5, a, b))
    return f"{k},{n},{alpha!r},{lo!r},{hi!r},{median!r}"


def cases() -> list[tuple[int, int, float]]:
    out: list[tuple[int, int, float]] = []
    # Minimum and smallest non-trivial experiments
    for n in (1, 2):
        for k in range(n + 1):
            out.append((k, n, 0.05))
    # Endpoint boundaries k=0 and k=n across the full supported regime
    for n in (100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000):
        for alpha in (0.01, 0.05, 0.1):
            out.append((0, n, alpha))
            out.append((n, n, alpha))
    # Near-boundary shapes a=1.5 / b=1.5 at large n
    for n in (1_000_000, 100_000_000):
        out.append((1, n, 0.05))
        out.append((n - 1, n, 0.05))
    # n=1e8 boundary band (v3 addition), all in-regime alphas
    n8 = 100_000_000
    for k in (0, 1, 10, n8 - 10, n8 - 1, n8):
        for alpha in (0.01, 0.05, 0.1, 1e-6):
            out.append((k, n8, alpha))
    # Typical logical-error-rate working regime
    for n in (1_000, 10_000, 100_000, 1_000_000):
        out.append((max(1, n // 1000), n, 0.05))
    # Wide-CI alpha extreme and best-conditioned symmetric peak
    for n in (1_000, 1_000_000):
        out.append((n // 2, n, 1e-6))
        out.append((n // 2, n, 0.05))
    # Asymmetric Gauss-Legendre-region shapes (both Beta shapes > 3000), including
    # the k=3016, n=1e6 tail-orientation regression and a large-n asymmetric case
    for k in (3000, 3016, 3100, 5000, 10_000):
        for alpha in (0.01, 0.05, 1e-6):
            out.append((k, 1_000_000, alpha))
    out.append((15_881, 20_000, 0.05))
    out.append((10_000, 100_000_000, 0.05))
    out.append((67_867_393, 100_000_000, 0.05))
    # Continued-fraction accuracy band: moderate a with huge b, straddling the
    # asymptotic-branch gate at a = 64.5
    for k in (63, 64, 65, 100, 300, 1_000, 3_000):
        for alpha in (0.01, 0.05):
            out.append((k, 100_000_000, alpha))
    # Deduplicate, preserving order
    seen: set[tuple[int, int, float]] = set()
    unique = []
    for case in out:
        if case not in seen:
            seen.add(case)
            unique.append(case)
    return unique


def main() -> None:
    print("k,n,alpha,lo,hi,median")
    for k, n, alpha in cases():
        print(jeffreys_row(k, n, alpha))


if __name__ == "__main__":
    sys.exit(main())
