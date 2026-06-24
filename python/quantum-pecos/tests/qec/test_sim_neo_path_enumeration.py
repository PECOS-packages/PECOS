# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Path enumeration through the sim_neo Python bindings.

Mirrors the Rust tool API: each distinct realized measurement-branch path is
one result row, with its exact probability in `result.weights`.
"""

from __future__ import annotations

import pytest

pecos_rslib_exp = pytest.importorskip("pecos_rslib_exp")

from pecos.quantum import TickCircuit  # noqa: E402
from pecos_rslib_exp import (  # noqa: E402
    depolarizing,
    monte_carlo,
    path_enumeration,
    sim_neo,
    stabilizer,
)


def bell_circuit() -> TickCircuit:
    tc = TickCircuit()
    tc.tick().h([0])
    tc.tick().cx([(0, 1)])
    tc.tick().mz([0, 1])
    return tc


def deterministic_circuit() -> TickCircuit:
    tc = TickCircuit()
    tc.tick().x([0])
    tc.tick().mz([0])
    return tc


def test_bell_pair_enumerates_two_correlated_paths() -> None:
    result = sim_neo(bell_circuit()).quantum(stabilizer()).sampling(path_enumeration(1)).run()

    assert result.num_shots == 2
    weights = result.weights
    assert weights is not None
    assert abs(sum(weights) - 1.0) < 1e-12
    for row, weight in zip(result, weights, strict=True):
        assert abs(weight - 0.5) < 1e-12
        assert row[0] == row[1], "Bell pair outcomes must be correlated"


def test_deterministic_circuit_dedupes_to_one_path() -> None:
    result = sim_neo(deterministic_circuit()).auto().sampling(path_enumeration(2)).run()

    assert result.num_shots == 1
    assert result.weights == [1.0]
    assert list(result[0]) == [1]


def test_monte_carlo_results_have_no_weights() -> None:
    result = (
        sim_neo(deterministic_circuit())
        .quantum(stabilizer())
        .sampling(monte_carlo(3))
        .seed(1)
        .run()
    )
    assert result.weights is None


def test_path_enumeration_rejects_noise() -> None:
    with pytest.raises(ValueError, match=r"remove \.noise\(\)"):
        sim_neo(bell_circuit()).quantum(stabilizer()).noise(depolarizing().p1(0.01)).sampling(
            path_enumeration(1),
        ).run()


def test_path_enumeration_rejects_huge_enumeration() -> None:
    with pytest.raises(ValueError, match="more than 16M paths"):
        sim_neo(bell_circuit()).quantum(stabilizer()).sampling(path_enumeration(25)).run()
