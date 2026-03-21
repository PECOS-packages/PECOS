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

"""Tests for exposing simulator seeding on stabilizer backends."""

import pytest

from pecos_rslib import SparseSim, Stab


def _measurement_sequence(sim_cls, *, seed=None, reseed=None, rounds=32):
    sim = sim_cls(1, seed=seed) if seed is not None else sim_cls(1)
    if reseed is not None:
        sim.set_seed(reseed)

    outcomes = []
    for _ in range(rounds):
        sim.reset()
        sim.run_1q_gate("H", 0)
        outcomes.append(sim.run_1q_gate("MZ", 0))
    return outcomes


@pytest.mark.parametrize("sim_cls", [SparseSim, Stab])
def test_seeded_constructor_repeats_measurement_sequence(sim_cls) -> None:
    assert _measurement_sequence(sim_cls, seed=42) == _measurement_sequence(sim_cls, seed=42)


@pytest.mark.parametrize("sim_cls", [SparseSim, Stab])
def test_set_seed_repeats_measurement_sequence(sim_cls) -> None:
    assert _measurement_sequence(sim_cls, reseed=42) == _measurement_sequence(sim_cls, reseed=42)


@pytest.mark.parametrize("sim_cls", [SparseSim, Stab])
def test_different_seeds_change_measurement_sequence(sim_cls) -> None:
    assert _measurement_sequence(sim_cls, seed=42) != _measurement_sequence(sim_cls, seed=43)
