# Copyright 2025 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Unit tests for Qulacs gate operations."""

import pytest

pytest.importorskip("pecos_rslib", reason="pecos_rslib required for qulacs tests")

import pecos as pc
from pecos.simulators.qulacs import Qulacs


class TestQulacsGateBindings:
    """Test individual gate operations and their bindings."""

    def test_identity_gate(self) -> None:
        """Test identity gate does nothing."""
        sim = Qulacs(1)
        initial_state = sim.vector.copy()

        sim.bindings["I"](sim, 0)

        assert pc.allclose(sim.vector, initial_state)

    def test_gate_parameter_passing(self) -> None:
        """Test gates that require parameters work correctly."""
        sim = Qulacs(1)

        # Test parameterized rotation gates
        angles_to_test = [0, pc.f64.frac_pi_4, pc.f64.frac_pi_2, pc.f64.pi, pc.f64.tau]

        for angle in angles_to_test:
            sim.reset()
            sim.bindings["RX"](sim, 0, angle=angle)

            # Verify state is normalized
            norm = pc.sum(pc.abs(sim.vector) ** 2)
            assert pc.isclose(norm, 1.0, rtol=1e-5, atol=1e-8)

    def test_square_root_gates(self) -> None:
        """Test square root gates (SX, SY, SZ)."""
        sim = Qulacs(1)

        # SX applied twice should equal X
        sim.bindings["SX"](sim, 0)
        sim.bindings["SX"](sim, 0)
        expected_x = pc.array([0, 1], dtype="complex")
        assert pc.allclose(sim.vector, expected_x)

        # Test SX and SXdg are inverses
        sim.reset()
        sim.bindings["SX"](sim, 0)
        sim.bindings["SXdg"](sim, 0)
        expected_identity = pc.array([1, 0], dtype="complex")
        assert pc.allclose(sim.vector, expected_identity, atol=1e-10)

    def test_dagger_gates(self) -> None:
        """Test that dagger gates are proper inverses."""
        sim = Qulacs(1)

        # Test T and Tdg
        sim.bindings["T"](sim, 0)
        sim.bindings["Tdg"](sim, 0)
        expected = pc.array([1, 0], dtype="complex")
        assert pc.allclose(sim.vector, expected, atol=1e-10)

        # Test SZ and SZdg
        sim.reset()
        sim.bindings["SZ"](sim, 0)
        sim.bindings["SZdg"](sim, 0)
        assert pc.allclose(sim.vector, expected, atol=1e-10)

    def test_all_single_qubit_gates_exist(self) -> None:
        """Test all expected single-qubit gates are in bindings."""
        sim = Qulacs(1)

        single_qubit_gates = [
            "I",
            "X",
            "Y",
            "Z",
            "H",
            "SX",
            "SXdg",
            "SY",
            "SYdg",
            "SZ",
            "SZdg",
            "T",
            "Tdg",
            "RX",
            "RY",
            "RZ",
        ]

        for gate in single_qubit_gates:
            assert gate in sim.bindings, f"Gate {gate} missing from bindings"

    def test_all_two_qubit_gates_exist(self) -> None:
        """Test all expected two-qubit gates are in bindings."""
        sim = Qulacs(2)

        two_qubit_gates = [
            "CX",
            "CY",
            "CZ",
            "SWAP",
            "RXX",
            "RYY",
            "RZZ",
        ]

        for gate in two_qubit_gates:
            assert gate in sim.bindings, f"Gate {gate} missing from bindings"

    def test_gate_aliases(self) -> None:
        """Test that gate aliases work correctly."""
        sim = Qulacs(2)

        # Test CNOT alias for CX
        sim.bindings["X"](sim, 0)  # |10⟩
        sim.bindings["CNOT"](sim, 0, 1)  # Should become |11⟩

        expected = pc.zeros(4, dtype="complex")
        expected[3] = 1.0  # |11⟩
        assert pc.allclose(sim.vector, expected)

        # Test S alias for SZ
        sim2 = Qulacs(1)
        sim2.bindings["H"](sim2, 0)
        sim2.bindings["S"](sim2, 0)  # Should be same as SZ

        sim3 = Qulacs(1)
        sim3.bindings["H"](sim3, 0)
        sim3.bindings["SZ"](sim3, 0)

        assert pc.allclose(sim2.vector, sim3.vector)

    def test_measurement_and_init_gates(self) -> None:
        """Test measurement and initialization gates."""
        sim = Qulacs(1, seed=42)

        # Test init gates
        sim.bindings["Init"](sim, 0)  # Should initialize to |0⟩
        expected = pc.array([1, 0], dtype="complex")
        assert pc.allclose(sim.vector, expected)

        # Test measurement
        result = sim.bindings["Measure"](sim, 0)
        assert result in [0, 1]

    def test_single_qubit_initialization(self) -> None:
        """Test single-qubit initialization doesn't affect other qubits."""
        # Test with 3-qubit system
        sim = Qulacs(3)

        # Initialize to a specific state: |101⟩
        sim.bindings["X"](sim, 0)  # qubit 0 -> |1⟩
        sim.bindings["I"](sim, 1)  # qubit 1 -> |0⟩ (already initialized)
        sim.bindings["X"](sim, 2)  # qubit 2 -> |1⟩

        # Expected state: |101⟩ = [0, 1, 0, 0, 0, 0, 0, 0] in computational basis
        # But with MSB-first ordering it's |101⟩ -> index 5 (binary: 101₂ = 5₁₀)
        expected_before = pc.zeros(8, dtype="complex")
        expected_before[5] = 1.0
        assert pc.allclose(
            sim.vector,
            expected_before,
        ), f"Initial state incorrect: {sim.vector}"

        # Reset qubit 1 to |0⟩ (should be no change since it's already |0⟩)
        sim.bindings["init |0>"](sim, 1)
        assert pc.allclose(
            sim.vector,
            expected_before,
        ), f"Reset qubit 1 to |0⟩ changed other qubits: {sim.vector}"

        # Reset qubit 1 to |1⟩ (should change state to |111⟩)
        sim.bindings["init |1>"](sim, 1)
        expected_after_init_one = pc.zeros(8, dtype="complex")
        expected_after_init_one[7] = 1.0  # |111⟩ -> index 7
        assert pc.allclose(
            sim.vector,
            expected_after_init_one,
        ), f"Init qubit 1 to |1⟩ incorrect: {sim.vector}"

        # Reset qubit 0 to |0⟩ (should change state to |011⟩)
        sim.bindings["init |0>"](sim, 0)
        expected_after_reset_0 = pc.zeros(8, dtype="complex")
        expected_after_reset_0[3] = 1.0  # |011⟩ -> index 3
        assert pc.allclose(
            sim.vector,
            expected_after_reset_0,
        ), f"Reset qubit 0 to |0⟩ incorrect: {sim.vector}"

        # Reset qubit 2 to |0⟩ (should change state to |010⟩)
        sim.bindings["init |0>"](sim, 2)
        expected_final = pc.zeros(8, dtype="complex")
        expected_final[2] = 1.0  # |010⟩ -> index 2
        assert pc.allclose(
            sim.vector,
            expected_final,
        ), f"Reset qubit 2 to |0⟩ incorrect: {sim.vector}"


class TestQulacsThreadSafety:
    """Test thread safety aspects of the simulator."""

    def test_independent_simulators(self) -> None:
        """Test that different simulator instances are independent."""
        sim1 = Qulacs(2, seed=42)
        sim2 = Qulacs(2, seed=42)

        # Apply different operations to each
        sim1.bindings["X"](sim1, 0)
        sim2.bindings["H"](sim2, 1)

        # States should be different
        assert not pc.allclose(sim1.vector, sim2.vector)

    def test_simulator_cloning_behavior(self) -> None:
        """Test that simulators with same seed produce same results."""
        sim1 = Qulacs(2, seed=123)
        sim2 = Qulacs(2, seed=123)

        # Apply same operations
        operations = [
            ("H", 0),
            ("CX", (0, 1)),
            ("RZ", 0, {"angle": pc.f64.pi / 3}),
        ]

        for op in operations:
            if len(op) == 2:
                # Single-qubit gate without parameters or two-qubit gate
                if isinstance(op[1], tuple):
                    # Two-qubit gate
                    sim1.bindings[op[0]](sim1, op[1][0], op[1][1])
                    sim2.bindings[op[0]](sim2, op[1][0], op[1][1])
                else:
                    # Single-qubit gate
                    sim1.bindings[op[0]](sim1, op[1])
                    sim2.bindings[op[0]](sim2, op[1])
            elif len(op) == 3:
                # Parameterized gate
                sim1.bindings[op[0]](sim1, op[1], **op[2])
                sim2.bindings[op[0]](sim2, op[1], **op[2])

        # Results should be identical
        assert pc.allclose(sim1.vector, sim2.vector)


class TestQulacsErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_qubit_indices(self) -> None:
        """Test behavior with invalid qubit indices."""
        sim = Qulacs(2)

        # Should raise an IndexError for out-of-bounds qubit index
        with pytest.raises(IndexError):
            sim.bindings["X"](sim, 5)  # Invalid qubit index

    def test_missing_parameters(self) -> None:
        """Test behavior when required parameters are missing."""
        sim = Qulacs(1)

        # RX gate requires angle parameter
        with pytest.raises(TypeError):
            sim.bindings["RX"](sim, 0)  # Missing angle parameter
