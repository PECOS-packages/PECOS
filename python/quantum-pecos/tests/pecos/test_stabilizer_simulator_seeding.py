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

"""High-level tests for seeded stabilizer simulator re-exports."""

import pytest
from pecos.simulators import SparseSim, Stab


def _measurement_sequence(
    sim_cls: type,
    *,
    seed: int | None = None,
    reseed: int | None = None,
    rounds: int = 16,
) -> list:
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
def test_high_level_simulators_accept_seed_and_set_seed(sim_cls: type) -> None:
    """Verify that seeded stabilizer simulators produce reproducible results."""
    assert _measurement_sequence(sim_cls, seed=42) == _measurement_sequence(sim_cls, seed=42)
    assert _measurement_sequence(sim_cls, reseed=42) == _measurement_sequence(sim_cls, reseed=42)
