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

"""Test the PauliProp with Rust backend."""

import pytest
from pecos.circuits import QuantumCircuit
from pecos.simulators import PauliProp


def test_pauli_fault_prop_basic() -> None:
    """Test basic functionality of PauliProp with Rust backend."""
    state = PauliProp(num_qubits=4, track_sign=True)

    # Initially empty
    assert state.faults == {"X": set(), "Y": set(), "Z": set()}
    assert state.fault_wt() == 0
    assert state.sign == 0
    assert state.img == 0

    # Add faults via circuit
    qc = QuantumCircuit()
    qc.append({"X": {0, 1}, "Z": {2}, "Y": {3}})
    state.add_faults(qc)

    # Check faults were added
    assert 0 in state.faults["X"]
    assert 1 in state.faults["X"]
    assert 2 in state.faults["Z"]
    assert 3 in state.faults["Y"]
    assert state.fault_wt() == 4

    # Check string representations
    assert state.get_str() == "+XXZY"
    assert state.fault_str_operator() == "XXZY"


def test_pauli_fault_prop_composition() -> None:
    """Test Pauli composition with the Rust backend."""
    state = PauliProp(num_qubits=3, track_sign=True)

    # Add X to qubit 0
    qc1 = QuantumCircuit()
    qc1.append({"X": {0}})
    state.add_faults(qc1)

    # Add Z to qubit 0 (X * Z = -iY)
    qc2 = QuantumCircuit()
    qc2.append({"Z": {0}})
    state.add_faults(qc2)

    # Should now have Y on qubit 0
    assert 0 in state.faults["Y"]
    assert 0 not in state.faults["X"]
    assert 0 not in state.faults["Z"]

    # Check phase tracking
    assert state.img == 1  # Should have i factor
    assert state.sign == 1  # Should have negative sign


def test_pauli_fault_prop_sign_tracking() -> None:
    """Test sign and phase tracking."""
    state = PauliProp(num_qubits=2, track_sign=True)

    # Test sign flipping
    assert state.sign == 0
    state.flip_sign()
    assert state.sign == 1
    state.flip_sign()
    assert state.sign == 0

    # Test imaginary component
    assert state.img == 0
    state.flip_img(1)
    assert state.img == 1
    state.flip_img(2)
    assert state.img == 1  # 1 + 2 = 3 % 2 = 1, with sign flip
    assert state.sign == 1  # Should have flipped sign


def test_pauli_fault_prop_setters() -> None:
    """Test property setters."""
    state = PauliProp(num_qubits=3, track_sign=True)

    # Set faults directly
    state.faults = {"X": {0}, "Y": {1}, "Z": {2}}
    assert state.faults["X"] == {0}
    assert state.faults["Y"] == {1}
    assert state.faults["Z"] == {2}

    # Set sign directly
    state.sign = 1
    assert state.sign == 1
    state.sign = 0
    assert state.sign == 0

    # Set img directly
    state.img = 1
    assert state.img == 1
    state.img = 0
    assert state.img == 0


def test_pauli_fault_prop_string_methods() -> None:
    """Test various string representation methods."""
    state = PauliProp(num_qubits=4, track_sign=True)

    # Add some faults
    qc = QuantumCircuit()
    qc.append({"X": {0}, "Y": {1}, "Z": {2}})
    state.add_faults(qc)

    # Test string representations
    assert state.get_str() == "+XYZI"
    assert state.fault_str_operator() == "XYZI"
    assert state.fault_str_sign(strip=False) in ["+ ", "+", "+ "]
    assert state.fault_str_sign(strip=True) == "+"
    assert state.fault_string() in ["+ XYZI", "+XYZI"]

    # Test with negative sign
    state.flip_sign()
    assert state.get_str() == "-XYZI"
    assert state.fault_str_sign(strip=True) == "-"

    # Test __str__ method
    str_repr = str(state)
    assert "{'X': {0}, 'Y': {1}, 'Z': {2}}" in str_repr


def test_pauli_fault_prop_with_minus() -> None:
    """Test add_faults with minus parameter."""
    state = PauliProp(num_qubits=2, track_sign=True)

    # Add faults with minus=True
    qc = QuantumCircuit()
    qc.append({"X": {0}})
    state.add_faults(qc, minus=True)

    assert state.sign == 1  # Should have negative sign
    assert 0 in state.faults["X"]


def test_pauli_fault_prop_invalid_operations() -> None:
    """Test error handling for invalid operations."""
    state = PauliProp(num_qubits=2, track_sign=False)

    # Try to add non-Pauli operation
    qc = QuantumCircuit()
    qc.append({"H": {0}})  # Hadamard is not a Pauli

    with pytest.raises(Exception, match="Can only handle Pauli errors"):
        state.add_faults(qc)
