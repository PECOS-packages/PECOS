# Copyright 2025 The PECOS Developers
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

"""Tests for the PECOS Qulacs Selene plugin."""

import pytest
from guppylang import guppy
from guppylang.std.angles import pi
from guppylang.std.builtins import result
from guppylang.std.quantum import cx, discard, h, measure, qubit, rz
from pecos_selene_qulacs import QulacsPlugin
from selene_sim.build import build


class TestQulacsBasic:
    """Basic functionality tests for the Qulacs plugin."""

    def test_single_qubit_discard(self) -> None:
        """Test that a qubit can be created and discarded."""

        @guppy
        def main() -> None:
            q = qubit()
            discard(q)

        runner = build(main.compile())
        simulator = QulacsPlugin(random_seed=42)

        # Run a single shot
        results = list(runner.run(simulator, n_qubits=1))
        assert len(results) == 0  # No results expected since no measurements

    def test_single_qubit_identity(self) -> None:
        """Test that a qubit without operations measures to 0."""

        @guppy
        def main() -> None:
            q = qubit()
            bit = measure(q)
            result("outcome", bit)

        runner = build(main.compile())
        simulator = QulacsPlugin(random_seed=42)

        # Run a single shot
        results = list(runner.run(simulator, n_qubits=1))
        assert dict(results)["outcome"] == 0

    def test_hadamard_measurement(self) -> None:
        """Test that H gate creates superposition."""

        @guppy
        def main() -> None:
            q = qubit()
            h(q)
            bit = measure(q)
            result("outcome", bit)

        runner = build(main.compile())
        simulator = QulacsPlugin(random_seed=123)

        # Run a single shot
        results = list(runner.run(simulator, n_qubits=1))
        # The result should be either 0 or 1
        assert dict(results)["outcome"] in [0, 1]


class TestQulacsBellState:
    """Tests involving Bell states and entanglement."""

    def test_bell_state_correlation(self) -> None:
        """Test that Bell state measurements are correlated."""

        @guppy
        def main() -> None:
            q0 = qubit()
            q1 = qubit()
            h(q0)
            cx(q0, q1)
            b0 = measure(q0)
            b1 = measure(q1)
            result("q0", b0)
            result("q1", b1)

        runner = build(main.compile())
        simulator = QulacsPlugin(random_seed=999)

        # Run a single shot
        results = list(runner.run(simulator, n_qubits=2))
        d = dict(results)
        # Both qubits should always have the same outcome in a Bell state
        assert d["q0"] == d["q1"], f"Bell state correlation failed: {d}"


class TestQulacsArbitraryRotations:
    """Tests for arbitrary rotation angles (non-Clifford)."""

    def test_t_gate_like_rotation(self) -> None:
        """Test that a T-gate-like rotation (pi/4) works."""

        @guppy
        def main() -> None:
            q = qubit()
            h(q)
            # T gate is Rz(pi/4)
            rz(q, pi / 4)
            h(q)
            bit = measure(q)
            result("outcome", bit)

        runner = build(main.compile())
        simulator = QulacsPlugin(random_seed=42)

        # Run multiple shots to verify it works
        for _ in range(5):
            results = list(runner.run(simulator, n_qubits=1))
            # Just check it doesn't crash - the rotation is valid
            assert dict(results)["outcome"] in [0, 1]

    def test_arbitrary_rz_angle(self) -> None:
        """Test an arbitrary Rz rotation angle."""

        @guppy
        def main() -> None:
            q = qubit()
            h(q)
            # Non-Clifford angle (pi/8 is a common non-Clifford angle)
            rz(q, pi / 8)
            h(q)
            bit = measure(q)
            result("outcome", bit)

        runner = build(main.compile())
        simulator = QulacsPlugin(random_seed=42)

        # This should work without error (unlike stabilizer simulators)
        results = list(runner.run(simulator, n_qubits=1))
        assert dict(results)["outcome"] in [0, 1]


class TestQulacsPlugin:
    """Tests for the plugin interface."""

    def test_library_file_exists(self) -> None:
        """Test that the library file property returns a valid path."""
        plugin = QulacsPlugin()
        lib_path = plugin.library_file

        # The path should be a Path object pointing to the expected location
        assert lib_path.name.startswith(
            "libpecos_selene_qulacs",
        ) or lib_path.name.startswith(
            "pecos_selene_qulacs",
        )

    def test_init_args_default(self) -> None:
        """Test that init args contain default mode."""
        plugin = QulacsPlugin()
        args = plugin.get_init_args()

        assert args == ["--mode=state_vector"]
