# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.

"""Release-build performance guard for the native numeric layer (PECOS #505).

The migration campaign fixed several order-of-magnitude performance defects
(``tolist`` ~843x, float reductions ~3,000x -- see #503/#506). These guards keep
them fixed. Ceilings are absolute wall-clock with roughly 10-100x headroom over
measured release times, so they catch the order-of-magnitude regression class
while staying immune to runner jitter. They are meaningless on debug builds and
run only under ``-m performance``.

The #505 array-layer paths use ceilings near 10x their measured release medians,
matching the order-of-magnitude-tripwire philosophy above.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np
import pytest

import pecos
from pecos import asarray, dtypes, zeros

N = 1_000_000


def _best_of(fn: Callable[[], Any], repeats: int = 3) -> float:
    fn()  # warm-up
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return min(samples)


CEILINGS: list[tuple[str, float]] = [
    # (case name, ceiling in seconds) -- release medians in comments
    ("zeros_uint8", 0.002),  # ~0.013 ms
    ("sum_uint8", 0.005),  # ~0.21 ms; beats NumPy
    ("sum_float64", 0.010),  # ~0.46 ms post-#506 (was ~465 ms)
    ("max_float64", 0.020),  # ~0.91 ms post-#506
    ("min_float64", 0.020),  # ~0.93 ms post-#506
    ("elementwise_ne", 0.002),  # ~0.025 ms
    ("astype_u8_i64", 0.006),  # ~0.50 ms post-#505
    ("astype_f64_f32", 0.006),  # ~0.53 ms post-#505
    ("tolist_uint8", 0.030),  # ~2.7 ms post-#503 (was ~2,250 ms)
    ("array_int_list", 0.065),  # ~6.0 ms post-#505
    ("array_float_list", 0.060),  # ~5.0 ms post-#505
    ("flatten_1000x1000", 0.005),  # ~0.44 ms post-#505
    ("boolean_mask_read", 0.050),  # ~5.2 ms post-#505 (was ~50 ms)
    ("fancy_read", 0.015),  # ~1.5 ms post-#505 (was ~33 ms)
    ("binomial_vectorised", 0.100),  # vectorised sampling path from #487
]


@pytest.fixture(scope="module")
def cases() -> dict[str, Callable[[], Any]]:
    uint8_array = asarray(np.random.default_rng(0).integers(0, 256, N, dtype=np.uint8), dtype=dtypes.uint8)
    float64_array = asarray(np.random.default_rng(1).random(N), dtype=dtypes.float64)
    mask = asarray(
        np.random.default_rng(2).integers(0, 2, N, dtype=np.uint8).astype(np.bool_),
        dtype=dtypes.bool,
    )
    fancy = list(range(0, N, 7))
    integer_list = list(range(N))
    float_list = [float(value) for value in range(N)]
    n_arr = asarray(np.full(64, 1000, dtype=np.int64), dtype=dtypes.int64)
    p_arr = asarray(np.full(64, 0.01), dtype=dtypes.float64)
    square = asarray(np.random.default_rng(3).random((1000, 1000)), dtype=dtypes.float64)
    return {
        "zeros_uint8": lambda: zeros(N, dtype=dtypes.uint8),
        "sum_uint8": lambda: pecos.sum(uint8_array),
        "sum_float64": lambda: pecos.sum(float64_array),
        "max_float64": lambda: pecos.max(float64_array),
        "min_float64": lambda: pecos.min(float64_array),
        "elementwise_ne": lambda: uint8_array != uint8_array,
        "astype_u8_i64": lambda: uint8_array.astype(dtypes.int64),
        "astype_f64_f32": lambda: float64_array.astype(dtypes.float32),
        "tolist_uint8": lambda: uint8_array.tolist(),
        "array_int_list": lambda: pecos.array(integer_list),
        "array_float_list": lambda: pecos.array(float_list),
        "flatten_1000x1000": lambda: square.flatten(),
        "boolean_mask_read": lambda: float64_array[mask],
        "fancy_read": lambda: float64_array[fancy],
        "binomial_vectorised": lambda: pecos.random.binomial(n_arr, p_arr, size=(100, 64)),
    }


@pytest.mark.performance
@pytest.mark.parametrize(("case", "ceiling"), CEILINGS)
def test_native_layer_stays_within_release_ceiling(
    case: str, ceiling: float, cases: dict[str, Callable[[], Any]]
) -> None:
    """Each op must stay under an order-of-magnitude ceiling on release builds."""
    elapsed = _best_of(cases[case])
    assert elapsed < ceiling, (
        f"{case} took {elapsed * 1e3:.1f} ms against a {ceiling * 1e3:.0f} ms ceiling -- "
        f"an order-of-magnitude performance regression (see PECOS #505)"
    )
