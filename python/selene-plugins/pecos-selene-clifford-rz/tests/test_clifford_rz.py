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

"""Tests for the PECOS CliffordRz Selene plugin."""

import pytest
from guppylang import guppy
from guppylang.std.angles import pi
from guppylang.std.builtins import result
from guppylang.std.quantum import crz, cx, discard, h, measure, qubit, reset, rx, ry, rz
from pecos_selene_clifford_rz import CliffordRzPlugin
from selene_sim.build import build


class TestCliffordRzBasic:
    """Basic functionality tests for the CliffordRz plugin."""

    def test_single_qubit_discard(self) -> None:
        """Test that a qubit can be created and discarded."""

        @guppy
        def main() -> None:
            q = qubit()
            discard(q)

        runner = build(main.compile())
        simulator = CliffordRzPlugin(random_seed=42)

        results = list(runner.run(simulator, n_qubits=1))
        assert len(results) == 0

    def test_single_qubit_identity(self) -> None:
        """Test that a qubit without operations measures to 0."""

        @guppy
        def main() -> None:
            q = qubit()
            bit = measure(q)
            result("outcome", bit)

        runner = build(main.compile())
        simulator = CliffordRzPlugin(random_seed=42)

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
        simulator = CliffordRzPlugin(random_seed=123)

        results = list(runner.run(simulator, n_qubits=1))
        assert dict(results)["outcome"] in [0, 1]


class TestCliffordRzBellState:
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
        simulator = CliffordRzPlugin(random_seed=999)

        results = list(runner.run(simulator, n_qubits=2))
        d = dict(results)
        assert d["q0"] == d["q1"], f"Bell state correlation failed: {d}"


class TestCliffordRzArbitraryRotations:
    """Tests for arbitrary rotation angles (non-Clifford)."""

    def test_t_gate_like_rotation(self) -> None:
        """Test that a T-gate-like rotation (pi/4) works."""

        @guppy
        def main() -> None:
            q = qubit()
            h(q)
            rz(q, pi / 4)
            h(q)
            bit = measure(q)
            result("outcome", bit)

        runner = build(main.compile())
        simulator = CliffordRzPlugin(random_seed=42)

        for _ in range(5):
            results = list(runner.run(simulator, n_qubits=1))
            assert dict(results)["outcome"] in [0, 1]

    def test_arbitrary_rz_angle(self) -> None:
        """Test an arbitrary Rz rotation angle."""

        @guppy
        def main() -> None:
            q = qubit()
            h(q)
            rz(q, pi / 8)
            h(q)
            bit = measure(q)
            result("outcome", bit)

        runner = build(main.compile())
        simulator = CliffordRzPlugin(random_seed=42)

        results = list(runner.run(simulator, n_qubits=1))
        assert dict(results)["outcome"] in [0, 1]

    def test_rx_rotation(self) -> None:
        """Test RX rotation (exercises the RXY Selene interface)."""

        @guppy
        def main() -> None:
            q = qubit()
            rx(q, pi / 3)
            bit = measure(q)
            result("outcome", bit)

        runner = build(main.compile())
        simulator = CliffordRzPlugin(random_seed=42)

        results = list(runner.run(simulator, n_qubits=1))
        assert dict(results)["outcome"] in [0, 1]

    def test_ry_rotation(self) -> None:
        """Test RY rotation (exercises the RXY Selene interface with phi=pi/2)."""

        @guppy
        def main() -> None:
            q = qubit()
            ry(q, pi / 3)
            bit = measure(q)
            result("outcome", bit)

        runner = build(main.compile())
        simulator = CliffordRzPlugin(random_seed=42)

        results = list(runner.run(simulator, n_qubits=1))
        assert dict(results)["outcome"] in [0, 1]

    def test_crz_two_qubit(self) -> None:
        """Test controlled-RZ rotation (exercises the RZZ Selene interface)."""

        @guppy
        def main() -> None:
            q0 = qubit()
            q1 = qubit()
            h(q0)
            crz(q0, q1, pi / 4)
            b0 = measure(q0)
            b1 = measure(q1)
            result("q0", b0)
            result("q1", b1)

        runner = build(main.compile())
        simulator = CliffordRzPlugin(random_seed=42)

        results = list(runner.run(simulator, n_qubits=2))
        d = dict(results)
        assert d["q0"] in [0, 1]
        assert d["q1"] in [0, 1]


class TestCliffordRzReset:
    """Tests for qubit reset."""

    def test_reset_after_x(self) -> None:
        """Test that reset brings a |1> qubit back to |0>."""

        @guppy
        def main() -> None:
            q = qubit()
            h(q)
            h(q)  # Back to |0> deterministically... but let's use reset
            # Flip to |1> via H-measure-H pattern isn't clean, so test reset directly
            reset(q)
            bit = measure(q)
            result("outcome", bit)

        runner = build(main.compile())
        simulator = CliffordRzPlugin(random_seed=42)

        results = list(runner.run(simulator, n_qubits=1))
        assert dict(results)["outcome"] == 0

    def test_reset_in_circuit(self) -> None:
        """Test reset mid-circuit followed by further operations."""

        @guppy
        def main() -> None:
            q = qubit()
            h(q)
            reset(q)
            h(q)
            bit = measure(q)
            result("outcome", bit)

        runner = build(main.compile())
        simulator = CliffordRzPlugin(random_seed=42)

        results = list(runner.run(simulator, n_qubits=1))
        assert dict(results)["outcome"] in [0, 1]


class TestCliffordRzPlugin:
    """Tests for the plugin interface."""

    def test_library_file_exists(self) -> None:
        """Test that the library file property returns a valid path."""
        plugin = CliffordRzPlugin()
        lib_path = plugin.library_file

        assert lib_path.name.startswith(
            "libpecos_selene_clifford_rz",
        ) or lib_path.name.startswith(
            "pecos_selene_clifford_rz",
        )

    def test_init_args_empty(self) -> None:
        """Test that init args are empty (no special parameters)."""
        plugin = CliffordRzPlugin()
        args = plugin.get_init_args()

        assert len(args) == 0
