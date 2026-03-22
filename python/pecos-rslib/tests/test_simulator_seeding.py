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

"""Tests for simulator seeding via the pecos_rslib bindings."""

import pytest
from pecos_rslib.simulators import SparseSim, Stab


def _measure_sequence(sim_cls: type, *, seed: int, rounds: int = 16) -> list:
    """Create a seeded simulator, apply H then MZ repeatedly, return outcomes."""
    sim = sim_cls(1, seed=seed)
    outcomes = []
    for _ in range(rounds):
        sim.reset()
        sim.run_1q_gate("H", 0)
        outcomes.append(sim.run_1q_gate("MZ", 0))
    return outcomes


@pytest.mark.parametrize("sim_cls", [SparseSim, Stab])
class TestSimulatorSeeding:
    """Verify that seeded stabilizer simulators produce reproducible results."""

    def test_constructor_seed_is_reproducible(self, sim_cls: type) -> None:
        """Same seed in constructor gives same measurement sequence."""
        assert _measure_sequence(sim_cls, seed=42) == _measure_sequence(sim_cls, seed=42)

    def test_different_seeds_differ(self, sim_cls: type) -> None:
        """Different seeds give different measurement sequences."""
        assert _measure_sequence(sim_cls, seed=42) != _measure_sequence(sim_cls, seed=99)

    def test_set_seed_is_reproducible(self, sim_cls: type) -> None:
        """Calling set_seed after construction gives reproducible results."""
        sim_a = sim_cls(1)
        sim_a.set_seed(42)
        outcomes_a = [sim_a.run_1q_gate("H", 0) for _ in range(8)]

        sim_b = sim_cls(1)
        sim_b.set_seed(42)
        outcomes_b = [sim_b.run_1q_gate("H", 0) for _ in range(8)]

        assert outcomes_a == outcomes_b
