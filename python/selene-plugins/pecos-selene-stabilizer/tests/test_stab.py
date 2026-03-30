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

"""Tests for the PECOS Stab Selene plugin."""

import pytest
from guppylang import guppy
from guppylang.std.builtins import result
from guppylang.std.quantum import cx, discard, h, measure, qubit
from pecos_selene_stabilizer import StabPlugin
from selene_sim.build import build


class TestStabBasic:
    """Basic functionality tests for the Stab plugin."""

    def test_single_qubit_discard(self) -> None:
        """Test that a qubit can be created and discarded."""

        @guppy
        def main() -> None:
            q = qubit()
            discard(q)

        runner = build(main.compile())
        simulator = StabPlugin(random_seed=42)

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
        simulator = StabPlugin(random_seed=42)

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
        simulator = StabPlugin(random_seed=123)

        # Run a single shot
        results = list(runner.run(simulator, n_qubits=1))
        # The result should be either 0 or 1
        assert dict(results)["outcome"] in [0, 1]


class TestStabBellState:
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
        simulator = StabPlugin(random_seed=999)

        # Run a single shot
        results = list(runner.run(simulator, n_qubits=2))
        d = dict(results)
        # Both qubits should always have the same outcome in a Bell state
        assert d["q0"] == d["q1"], f"Bell state correlation failed: {d}"


class TestStabAngleValidation:
    """Tests for angle threshold and Clifford validation."""

    def test_invalid_angle_threshold(self) -> None:
        """Test that angle_threshold must be positive."""
        with pytest.raises(ValueError, match="greater than zero"):
            StabPlugin(angle_threshold=0)

        with pytest.raises(ValueError, match="greater than zero"):
            StabPlugin(angle_threshold=-0.1)

    def test_valid_angle_threshold(self) -> None:
        """Test that valid angle thresholds are accepted."""
        plugin = StabPlugin(angle_threshold=1e-6)
        assert plugin.angle_threshold == 1e-6

        plugin = StabPlugin(angle_threshold=0.1)
        assert plugin.angle_threshold == 0.1


class TestStabPlugin:
    """Tests for the plugin interface."""

    def test_library_file_exists(self) -> None:
        """Test that the library file property returns a valid path."""
        plugin = StabPlugin()
        lib_path = plugin.library_file

        # The path should be a Path object pointing to the expected location
        assert lib_path.name.startswith(
            "libpecos_selene_stabilizer",
        ) or lib_path.name.startswith(
            "pecos_selene_stabilizer",
        )

    def test_init_args(self) -> None:
        """Test that init args are correctly formatted."""
        plugin = StabPlugin(angle_threshold=1e-5)
        args = plugin.get_init_args()

        assert len(args) == 1
        assert args[0] == "--angle-threshold=1e-05"

    def test_default_init_args(self) -> None:
        """Test default init args."""
        plugin = StabPlugin()
        args = plugin.get_init_args()

        assert len(args) == 1
        assert "--angle-threshold=" in args[0]
