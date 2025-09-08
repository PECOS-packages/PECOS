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

"""Rust-based Pauli propagation simulator for PECOS.

This module provides a Python interface to the high-performance Rust implementation of a Pauli
propagation simulator. The simulator efficiently tracks how Pauli operators transform under
Clifford operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pecos_rslib._pecos_rslib import PauliProp as RustPauliProp

if TYPE_CHECKING:
    pass


class PauliPropRs:
    """Rust-based Pauli propagation simulator.

    A high-performance simulator for tracking Pauli operator propagation through
    Clifford circuits. Useful for fault propagation and stabilizer simulations.
    """

    def __init__(self, num_qubits: int | None = None, track_sign: bool = False):
        """Initialize the Rust-backed Pauli propagation simulator.

        Args:
            num_qubits: Optional number of qubits (for string representation).
            track_sign: Whether to track sign and phase.
        """
        self._sim = RustPauliProp(num_qubits, track_sign)
        self.num_qubits = num_qubits
        self.track_sign = track_sign

    def reset(self) -> PauliPropRs:
        """Reset the simulator state.

        Returns:
            Self for method chaining.
        """
        self._sim.reset()
        return self

    @property
    def faults(self) -> dict[str, set[int]]:
        """Get the current faults as a dictionary.

        Returns:
            Dictionary with keys "X", "Y", "Z" mapping to sets of qubit indices.
        """
        return self._sim.get_faults()

    def contains_x(self, qubit: int) -> bool:
        """Check if a qubit has an X operator."""
        return self._sim.contains_x(qubit)

    def contains_y(self, qubit: int) -> bool:
        """Check if a qubit has a Y operator."""
        return self._sim.contains_y(qubit)

    def contains_z(self, qubit: int) -> bool:
        """Check if a qubit has a Z operator."""
        return self._sim.contains_z(qubit)

    def add_x(self, qubit: int) -> None:
        """Add an X operator to a qubit."""
        self._sim.add_x(qubit)

    def add_y(self, qubit: int) -> None:
        """Add a Y operator to a qubit."""
        self._sim.add_y(qubit)

    def add_z(self, qubit: int) -> None:
        """Add a Z operator to a qubit."""
        self._sim.add_z(qubit)

    def flip_sign(self) -> None:
        """Flip the sign of the Pauli string."""
        self._sim.flip_sign()

    def flip_img(self, num_is: int) -> None:
        """Add imaginary factors to the phase.

        Args:
            num_is: Number of i factors to add.
        """
        self._sim.flip_img(num_is)

    def add_paulis(self, paulis: dict[str, set[int] | list[int]]) -> None:
        """Add Pauli operators from a dictionary.

        Args:
            paulis: Dictionary with keys "X", "Y", "Z" mapping to sets/lists of qubit indices.
        """
        # Convert lists to sets if needed
        paulis_dict = {}
        for key, value in paulis.items():
            if isinstance(value, list):
                paulis_dict[key] = set(value)
            else:
                paulis_dict[key] = value
        self._sim.add_paulis(paulis_dict)

    def set_faults(self, paulis: dict[str, set[int] | list[int]]) -> None:
        """Set the faults by clearing and then adding new ones.

        Args:
            paulis: Dictionary with keys "X", "Y", "Z" mapping to sets/lists of qubit indices.
        """
        self.reset()
        if paulis:
            self.add_paulis(paulis)

    def weight(self) -> int:
        """Get the weight of the Pauli string (number of non-identity operators)."""
        return self._sim.weight()

    def sign_string(self) -> str:
        """Get the sign string representation."""
        return self._sim.sign_string()

    def sparse_string(self) -> str:
        """Get the sparse string representation."""
        return self._sim.sparse_string()

    def dense_string(self) -> str:
        """Get the dense string representation."""
        return self._sim.dense_string()

    def to_pauli_string(self) -> str:
        """Get the full Pauli string with sign."""
        return self._sim.to_pauli_string()

    def to_dense_string(self) -> str:
        """Get the full dense Pauli string with sign."""
        return self._sim.to_dense_string()

    def fault_string(self) -> str:
        """Get the fault string representation (for compatibility with PauliFaultProp)."""
        return self.to_pauli_string()

    def fault_wt(self) -> int:
        """Get the fault weight (for compatibility with PauliFaultProp)."""
        return self.weight()

    # Clifford gates

    def h(self, qubit: int) -> None:
        """Apply Hadamard gate."""
        self._sim.h(qubit)

    def sz(self, qubit: int) -> None:
        """Apply S gate (sqrt(Z))."""
        self._sim.sz(qubit)

    def sx(self, qubit: int) -> None:
        """Apply sqrt(X) gate."""
        self._sim.sx(qubit)

    def sy(self, qubit: int) -> None:
        """Apply sqrt(Y) gate."""
        self._sim.sy(qubit)

    def cx(self, control: int, target: int) -> None:
        """Apply CNOT/CX gate."""
        self._sim.cx(control, target)

    def cy(self, control: int, target: int) -> None:
        """Apply CY gate."""
        self._sim.cy(control, target)

    def cz(self, control: int, target: int) -> None:
        """Apply CZ gate."""
        self._sim.cz(control, target)

    def swap(self, q1: int, q2: int) -> None:
        """Apply SWAP gate."""
        self._sim.swap(q1, q2)

    def mz(self, qubit: int) -> bool:
        """Measure in Z basis."""
        return self._sim.mz(qubit)

    def is_identity(self) -> bool:
        """Check if this is the identity operator."""
        return self._sim.is_identity()

    def get_sign_bool(self) -> bool:
        """Get the sign as a boolean (False for +, True for -)."""
        return self._sim.get_sign()

    def get_img_value(self) -> int:
        """Get the imaginary component (0 for real, 1 for imaginary)."""
        return self._sim.get_img()

    def __str__(self) -> str:
        """String representation."""
        return str(self._sim)

    def __repr__(self) -> str:
        """Representation string."""
        return repr(self._sim)
