# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the
# License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
# either express or implied. See the License for the specific language governing permissions and
# limitations under the License.

"""Python coverage for randomized code-distance upper bounds."""

from pecos.qec import (
    FaultDistanceUpperBoundConfig,
    FaultDistanceUpperBoundResult,
    randomized_code_distance_upper_bound,
)
from pecos.quantum import ParityCheckMatrix


def _config(samples: int, seed: int) -> FaultDistanceUpperBoundConfig:
    return FaultDistanceUpperBoundConfig(
        samples=samples,
        seed=seed,
        observable_subset_strategy="each_single_then_random",
        error_rate=0.1,
        max_iterations=100,
        bp_method="product_sum",
        bp_schedule="parallel",
        min_sum_scaling_factor=1.0,
        osd_method="osd_0",
        osd_order=0,
        omp_threads=1,
    )


def test_repetition_code_upper_bound_is_qubit_indexed_sound_and_tight() -> None:
    h = ParityCheckMatrix([[1, 1, 0], [0, 1, 1]])
    logicals = ParityCheckMatrix([[1, 1, 1]])

    result = randomized_code_distance_upper_bound(h, logicals, _config(1, 17))

    assert isinstance(result, FaultDistanceUpperBoundResult)
    assert result.weight >= 3
    assert result.weight == 3
    assert result.mechanism_indices == [0, 1, 2]
    assert result.bound_kind == "upper_bound"


def test_code_upper_bound_is_deterministic_and_zero_samples_return_none() -> None:
    h = ParityCheckMatrix([[1, 1, 0], [0, 1, 1]])
    logicals = ParityCheckMatrix([[1, 1, 1]])
    config = _config(8, 991)

    first = randomized_code_distance_upper_bound(h, logicals, config)
    second = randomized_code_distance_upper_bound(h, logicals, config)

    assert first is not None
    assert second is not None
    assert first.weight == second.weight
    assert first.mechanism_indices == second.mechanism_indices
    assert randomized_code_distance_upper_bound(h, logicals, _config(0, 1)) is None
