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

"""Test the Rust PauliProp integration."""

from pecos.circuits import QuantumCircuit
from pecos.simulators import PauliProp
from pecos_rslib.simulators import PauliProp as PauliPropRs


def test_rust_pauli_prop_basic() -> None:
    """Test basic functionality of Rust PauliProp."""
    sim = PauliPropRs(num_qubits=4, track_sign=True)

    # Add Pauli operators
    sim.add_x(0)
    sim.add_z(2)
    sim.add_y(3)

    assert sim.weight() == 3
    assert sim.contains_x(0)
    assert sim.contains_z(2)
    assert sim.contains_y(3)

    # Test string representations
    assert sim.to_dense_string() == "+XIZY"

    # Test Hadamard gate (X -> Z)
    sim.h(0)
    assert sim.contains_z(0)
    assert not sim.contains_x(0)


def test_rust_pauli_prop_composition() -> None:
    """Test Pauli composition with phase tracking."""
    sim = PauliPropRs(num_qubits=3, track_sign=True)

    # X * Z = -iY (applying Z after X)
    sim.add_x(0)
    sim.add_paulis({"Z": {0}})

    assert sim.contains_y(0)
    assert sim.sign_string() == "-i"  # X*Z = -iY

    # Y * Y = I
    sim.add_y(0)
    assert not sim.contains_x(0)
    assert not sim.contains_z(0)
    assert not sim.contains_y(0)


def test_rust_pauli_prop_gates() -> None:
    """Test Clifford gates."""
    sim = PauliPropRs(num_qubits=3, track_sign=False)

    # Test CX propagation
    sim.add_x(0)  # X on control
    sim.cx(0, 1)  # Should propagate to target
    assert sim.contains_x(0)
    assert sim.contains_x(1)

    # Test CZ propagation
    sim2 = PauliPropRs(num_qubits=3, track_sign=False)
    sim2.add_z(1)  # Z on target
    sim2.cx(0, 1)  # Should propagate to control
    assert sim2.contains_z(0)
    assert sim2.contains_z(1)


def test_rust_vs_python_consistency() -> None:
    """Test that Rust and Python implementations give same results."""
    # Create both simulators
    rust_sim = PauliPropRs(num_qubits=4, track_sign=True)
    py_sim = PauliProp(num_qubits=4, track_sign=True)

    # Add same faults using appropriate APIs
    qc = QuantumCircuit()
    qc.append({"X": {0, 1}, "Z": {2}, "Y": {3}})
    py_sim.add_faults(qc)

    rust_sim.add_x(0)
    rust_sim.add_x(1)
    rust_sim.add_z(2)
    rust_sim.add_y(3)

    # Check weights match
    assert rust_sim.weight() == py_sim.fault_wt()

    # Check string representations match
    assert rust_sim.to_dense_string() == py_sim.get_str()

    # Test composition
    qc2 = QuantumCircuit()
    qc2.append({"Z": {0}})  # Add Z to qubit with X -> Y
    py_sim.add_faults(qc2)

    rust_sim.add_paulis({"Z": {0}})

    # Both should now have Y on qubit 0
    assert 0 in py_sim.faults["Y"]
    assert rust_sim.contains_y(0)

    # Check phase tracking
    assert rust_sim.sign_string() == py_sim.fault_str_sign().strip()


def test_rust_pauli_prop_weight() -> None:
    """Test weight calculation."""
    sim = PauliPropRs(num_qubits=5, track_sign=False)

    assert sim.weight() == 0

    sim.add_x(0)
    assert sim.weight() == 1

    sim.add_z(1)
    assert sim.weight() == 2

    sim.add_y(2)
    assert sim.weight() == 3

    # Adding X to qubit with Z makes Y (still weight 3)
    sim.add_x(1)
    assert sim.weight() == 3
    assert sim.contains_y(1)
