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

"""Integration tests for C++ sparse simulator via Rust bindings."""

import pytest
from pecos.simulators import SparseSimCpp


def test_basic_gates() -> None:
    """Test basic gate operations without checking tableaus."""
    sim = SparseSimCpp(2)

    # Test single qubit gates
    sim.run_gate("X", {0})
    sim.run_gate("Y", {1})
    sim.run_gate("Z", {0})
    sim.run_gate("H", {0})

    # Test two qubit gates
    sim.run_gate("CX", {(0, 1)})
    sim.run_gate("CZ", {(0, 1)})

    # Reset and test again
    sim.reset()
    sim.run_gate("H", {0})
    sim.run_gate("CX", {(0, 1)})


def test_measurements() -> None:
    """Test measurement operations."""
    sim = SparseSimCpp(3)

    # Measure in computational basis (should get 0)
    result = sim.run_gate("MZ", {0})
    assert result[0] == 0

    # Apply X then measure (should get 1)
    sim.run_gate("X", {1})
    result = sim.run_gate("MZ", {1})
    assert result[1] == 1

    # Test measurement after H (random but deterministic with fixed seed)
    sim.reset()
    sim.run_gate("H", {0})
    result = sim.run_gate("MZ", {0})
    assert result[0] in [0, 1]


def test_bell_state() -> None:
    """Test creating and measuring Bell states."""
    sim = SparseSimCpp(2)

    # Create |00> + |11> Bell state
    sim.run_gate("H", {0})
    sim.run_gate("CX", {(0, 1)})

    # Measure both qubits - they should be correlated
    result0 = sim.run_gate("MZ", {0})
    result1 = sim.run_gate("MZ", {1})
    assert result0[0] == result1[1]


def test_ghz_state() -> None:
    """Test creating and measuring GHZ states."""
    sim = SparseSimCpp(3)

    # Create |000> + |111> GHZ state
    sim.run_gate("H", {0})
    sim.run_gate("CX", {(0, 1)})
    sim.run_gate("CX", {(0, 2)})

    # Measure all qubits - they should all be the same
    result0 = sim.run_gate("MZ", {0})
    result1 = sim.run_gate("MZ", {1})
    result2 = sim.run_gate("MZ", {2})
    assert result0[0] == result1[1] == result2[2]


def test_circuit_execution() -> None:
    """Test running a simple circuit."""
    sim = SparseSimCpp(4)

    # Define a simple circuit
    circuit = [
        ("H", {0, 1}),
        ("CX", {(0, 2)}),
        ("CX", {(1, 3)}),
        ("H", {2, 3}),
    ]

    # Run the circuit
    for gate, qubits in circuit:
        sim.run_gate(gate, qubits)

    # The circuit should execute without errors
    # We're not checking the state, just that it runs


def test_reset() -> None:
    """Test reset functionality."""
    sim = SparseSimCpp(2)

    # Apply some gates
    sim.run_gate("X", {0})
    sim.run_gate("H", {1})
    sim.run_gate("CX", {(0, 1)})

    # Reset
    sim.reset()

    # After reset, measuring should give |00>
    result0 = sim.run_gate("MZ", {0})
    result1 = sim.run_gate("MZ", {1})
    assert result0[0] == 0
    assert result1[1] == 0


@pytest.mark.parametrize("num_qubits", [1, 2, 3, 5, 10])
def test_various_sizes(num_qubits: int) -> None:
    """Test simulator with various numbers of qubits."""
    sim = SparseSimCpp(num_qubits)

    # Apply some gates
    for i in range(num_qubits):
        sim.run_gate("H", {i})

    # Apply CNOT chain
    for i in range(num_qubits - 1):
        sim.run_gate("CX", {(i, i + 1)})

    # Measure all qubits
    results = {}
    for i in range(num_qubits):
        result = sim.run_gate("MZ", {i})
        results.update(result)

    # Check all were measured
    assert len(results) == num_qubits

    # Check that all results are valid (0 or 1)
    for i in range(num_qubits):
        assert results[i] in [0, 1]
