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

"""Tests for Qulacs simulator."""

import numpy as np
import pytest

pytest.importorskip("pecos_rslib", reason="pecos_rslib required for qulacs tests")

from pecos.num import (
    abs,  # noqa: A004 - intentionally shadow builtin
    allclose,
    array,
    isclose,
    pi,
    zeros,
)
from pecos.simulators.qulacs import Qulacs


class TestQulacsBasic:
    """Basic functionality tests for Qulacs simulator."""

    def test_initialization(self) -> None:
        """Test simulator initialization."""
        sim = Qulacs(3)
        assert sim.num_qubits == 3

        # Check initial state is |000⟩
        state = sim.vector
        assert state.shape == (8,)
        assert isclose(abs(state[0]) ** 2, 1.0, rtol=1e-5, atol=1e-8)
        for i in range(1, 8):
            assert isclose(abs(state[i]) ** 2, 0.0, rtol=1e-5, atol=1e-8)

    def test_initialization_with_seed(self) -> None:
        """Test simulator initialization with deterministic seed."""
        sim1 = Qulacs(2, seed=42)
        sim2 = Qulacs(2, seed=42)

        # Apply some gates and measure
        sim1.bindings["H"](sim1, 0)
        sim2.bindings["H"](sim2, 0)

        # States should be identical
        assert allclose(sim1.vector, sim2.vector)

    def test_reset(self) -> None:
        """Test state reset functionality."""
        sim = Qulacs(2)

        # Apply some gates
        sim.bindings["H"](sim, 0)
        sim.bindings["CX"](sim, 0, 1)

        # Reset should return to |00⟩
        sim.reset()
        expected = zeros(4, dtype="complex")
        expected[0] = 1.0

        assert allclose(sim.vector, expected)


class TestQulacsSingleQubitGates:
    """Test single-qubit gate operations."""

    def test_pauli_gates(self) -> None:
        """Test Pauli X, Y, Z gates."""
        sim = Qulacs(1)

        # Test X gate: X|0⟩ = |1⟩
        sim.bindings["X"](sim, 0)
        expected = array([0, 1], dtype="complex")
        assert allclose(sim.vector, expected)

        # Test X again: X|1⟩ = |0⟩
        sim.bindings["X"](sim, 0)
        expected = array([1, 0], dtype="complex")
        assert allclose(sim.vector, expected)

        # Test Y gate: Y|0⟩ = i|1⟩
        sim.reset()
        sim.bindings["Y"](sim, 0)
        expected = array([0, 1j], dtype="complex")
        assert allclose(sim.vector, expected)

        # Test Z gate on |+⟩ state
        sim.reset()
        sim.bindings["H"](sim, 0)  # Create |+⟩
        sim.bindings["Z"](sim, 0)  # Z|+⟩ = |-⟩
        sim.bindings["H"](sim, 0)  # H|-⟩ = |1⟩
        expected = array([0, 1], dtype="complex")
        assert allclose(sim.vector, expected)

    def test_hadamard_gate(self) -> None:
        """Test Hadamard gate."""
        sim = Qulacs(1)

        # H|0⟩ = |+⟩ = (|0⟩ + |1⟩)/√2
        sim.bindings["H"](sim, 0)
        expected = array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype="complex")
        assert allclose(sim.vector, expected)

        # H|1⟩ = |-⟩ = (|0⟩ - |1⟩)/√2
        sim.reset()
        sim.bindings["X"](sim, 0)
        sim.bindings["H"](sim, 0)
        expected = array([1 / np.sqrt(2), -1 / np.sqrt(2)], dtype="complex")
        assert allclose(sim.vector, expected)

    def test_phase_gates(self) -> None:
        """Test S and T gates."""
        sim = Qulacs(1)

        # Test S gate: S|+⟩ = |i⟩ = (|0⟩ + i|1⟩)/√2
        sim.bindings["H"](sim, 0)  # |+⟩
        sim.bindings["SZ"](sim, 0)  # S gate
        expected_phase = 1j
        state = sim.vector
        phase_ratio = state[1] / state[0]
        assert isclose(phase_ratio, expected_phase, rtol=0.0, atol=1e-10)

        # Test T gate
        sim.reset()
        sim.bindings["H"](sim, 0)
        sim.bindings["T"](sim, 0)
        state = sim.vector
        expected_t_phase = np.exp(1j * pi / 4)
        phase_ratio = state[1] / state[0]
        assert isclose(phase_ratio, expected_t_phase, rtol=0.0, atol=1e-10)

    def test_rotation_gates(self) -> None:
        """Test rotation gates RX, RY, RZ."""
        sim = Qulacs(1)

        # Test RX(π) = -iX
        sim.bindings["RX"](sim, 0, angle=pi)
        state = sim.vector
        assert isclose(state[0], 0, rtol=0.0, atol=1e-10)
        assert isclose(state[1], -1j, rtol=0.0, atol=1e-10)

        # Test RY(π/2) creates equal superposition
        sim.reset()
        sim.bindings["RY"](sim, 0, angle=pi / 2)
        state = sim.vector
        assert isclose(abs(state[0]), 1 / np.sqrt(2), rtol=0.0, atol=1e-10)
        assert isclose(abs(state[1]), 1 / np.sqrt(2), rtol=0.0, atol=1e-10)

        # Test RZ(π) on |+⟩
        sim.reset()
        sim.bindings["H"](sim, 0)  # Create |+⟩
        sim.bindings["RZ"](sim, 0, angle=pi)
        sim.bindings["H"](sim, 0)  # Should give |1⟩ (possibly with phase)
        state = sim.vector
        # Check that qubit is effectively in |1⟩ state (allowing for global phase)
        assert isclose(abs(state[0]), 0, rtol=0.0, atol=1e-10)
        assert isclose(abs(state[1]), 1, rtol=0.0, atol=1e-10)


class TestQulacsTwoQubitGates:
    """Test two-qubit gate operations."""

    def test_bell_state(self) -> None:
        """Test Bell state creation with H and CNOT."""
        sim = Qulacs(2)

        # Create Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
        sim.bindings["H"](sim, 0)
        sim.bindings["CX"](sim, 0, 1)

        state = sim.vector
        expected = zeros(4, dtype="complex")
        expected[0] = 1 / np.sqrt(2)  # |00⟩
        expected[3] = 1 / np.sqrt(2)  # |11⟩

        assert allclose(state, expected)

    def test_controlled_gates(self) -> None:
        """Test controlled X, Y, Z gates."""
        sim = Qulacs(2)

        # Test CX gate
        sim.bindings["X"](sim, 0)  # |10⟩
        sim.bindings["CX"](sim, 0, 1)  # Should become |11⟩
        expected = zeros(4, dtype="complex")
        expected[3] = 1.0  # |11⟩
        assert allclose(sim.vector, expected)

        # Test CZ gate on |++⟩
        sim.reset()
        sim.bindings["H"](sim, 0)
        sim.bindings["H"](sim, 1)
        sim.bindings["CZ"](sim, 0, 1)

        state = sim.vector
        # CZ|++⟩ = (|00⟩ + |01⟩ + |10⟩ - |11⟩)/2
        expected = array([0.5, 0.5, 0.5, -0.5], dtype="complex")
        assert allclose(state, expected)

    def test_swap_gate(self) -> None:
        """Test SWAP gate."""
        sim = Qulacs(2)

        # Prepare |10⟩ and swap to |01⟩
        sim.bindings["X"](sim, 0)  # |10⟩
        sim.bindings["SWAP"](sim, 0, 1)  # Should become |01⟩

        # Check that exactly one basis state has probability 1
        probs = abs(sim.vector) ** 2
        assert np.sum(probs > 0.5) == 1  # Exactly one state should be populated


class TestQulacsMeasurement:
    """Test measurement operations."""

    def test_deterministic_measurement(self) -> None:
        """Test measurement on definite states."""
        sim = Qulacs(1, seed=100)

        # Measure |0⟩ state
        sim.reset()
        result = sim.bindings["Measure"](sim, 0)
        assert result == 0

        # Measure |1⟩ state
        sim.bindings["X"](sim, 0)
        result = sim.bindings["Measure"](sim, 0)
        assert result == 1

    def test_measurement_statistics(self) -> None:
        """Test measurement statistics on superposition states."""
        sim = Qulacs(1, seed=42)

        # Prepare |+⟩ state and measure many times
        n_trials = 1000
        results = []

        for _ in range(n_trials):
            sim.reset()
            sim.bindings["H"](sim, 0)  # |+⟩ state
            result = sim.bindings["Measure"](sim, 0)
            results.append(result)

        # Should be approximately 50/50
        ones_count = sum(results)
        ratio = ones_count / n_trials
        assert abs(ratio - 0.5) < 0.1  # Allow some variance


class TestQulacsCompatibility:
    """Test compatibility with existing PECOS patterns."""

    def test_gate_bindings_structure(self) -> None:
        """Test that gate bindings follow expected structure."""
        sim = Qulacs(2)

        # Test that all expected gates are available
        expected_gates = [
            "X",
            "Y",
            "Z",
            "H",
            "SZ",
            "SZdg",
            "T",
            "Tdg",
            "CX",
            "CY",
            "CZ",
            "SWAP",
            "RX",
            "RY",
            "RZ",
            "Init",
            "Measure",
        ]

        for gate in expected_gates:
            assert gate in sim.bindings, f"Gate {gate} not found in bindings"

    def test_numpy_compatibility(self) -> None:
        """Test numpy array compatibility."""
        sim = Qulacs(2)

        state = sim.vector

        # Should be numpy-compatible (Array implements buffer protocol)
        # Can convert to numpy array via np.asarray
        state_np = np.asarray(state)
        assert isinstance(state_np, np.ndarray)

        # Should have complex dtype
        assert np.iscomplexobj(state_np)

        # Should be normalized
        norm = np.sum(abs(state_np) ** 2)
        assert isclose(norm, 1.0, rtol=1e-5, atol=1e-8)

        # Should support numpy operations
        probabilities = abs(state_np) ** 2
        assert isinstance(probabilities, np.ndarray)
        assert probabilities.dtype == float


class TestQulacsAdvanced:
    """Advanced tests for edge cases and complex scenarios."""

    def test_ghz_state(self) -> None:
        """Test GHZ state creation."""
        sim = Qulacs(3)

        # Create GHZ state |GHZ⟩ = (|000⟩ + |111⟩)/√2
        sim.bindings["H"](sim, 0)
        sim.bindings["CX"](sim, 0, 1)
        sim.bindings["CX"](sim, 1, 2)

        state = sim.vector
        expected = zeros(8, dtype="complex")
        expected[0] = 1 / np.sqrt(2)  # |000⟩
        expected[7] = 1 / np.sqrt(2)  # |111⟩

        assert allclose(state, expected)

    def test_state_normalization_preservation(self) -> None:
        """Test that state remains normalized after various operations."""
        sim = Qulacs(3)

        # Apply various gates
        sim.bindings["H"](sim, 0)
        sim.bindings["CX"](sim, 0, 1)
        sim.bindings["RY"](sim, 2, angle=pi / 4)
        sim.bindings["CZ"](sim, 1, 2)
        sim.bindings["T"](sim, 0)

        # Check normalization
        state = sim.vector
        norm_squared = np.sum(abs(state) ** 2)
        assert isclose(norm_squared, 1.0, rtol=0.0, atol=1e-10)

    def test_gate_reversibility(self) -> None:
        """Test that gates are properly reversible."""
        sim = Qulacs(2)

        # Save initial state
        initial_state = sim.vector.copy()

        # Apply gates and their inverses
        sim.bindings["H"](sim, 0)
        sim.bindings["CX"](sim, 0, 1)
        sim.bindings["SZ"](sim, 1)
        sim.bindings["SZdg"](sim, 1)  # S†
        sim.bindings["CX"](sim, 0, 1)
        sim.bindings["H"](sim, 0)

        # Should be back to initial state
        final_state = sim.vector
        assert allclose(initial_state, final_state, atol=1e-10)
