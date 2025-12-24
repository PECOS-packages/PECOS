# Copyright 2018 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Quantum state representation for Pauli fault propagation simulator.

This module provides the quantum state representation for the Pauli fault propagation simulator, implementing
efficient Pauli frame tracking and stabilizer tableau management for fast stabilizer circuit simulation.
"""

# Gate bindings require consistent interfaces even if not all parameters are used.

from __future__ import annotations

from typing import TYPE_CHECKING

from pecos_rslib.simulators import PauliProp as PauliPropRs

from pecos.simulators.gate_syms import alt_symbols
from pecos.simulators.pauliprop import bindings
from pecos.simulators.pauliprop.logical_sign import find_logical_signs
from pecos.simulators.sim_class_types import PauliPropagation

if TYPE_CHECKING:
    from pecos.circuits import QuantumCircuit
    from pecos.circuits.quantum_circuit import ParamGateCollection


class PauliProp(PauliPropagation):
    r"""A simulator that evolves Pauli faults through Clifford circuits.

    The unitary evolution of a Pauli follows :math:`PC = CP' \Leftrightarrow P' = C^{\dagger} P C`, where :math:`P` and
    :math:`P'` are Pauli operators and :math:`C` is a Clifford operator.

    Attributes:
    ----------
        num_qubits(int): Number of qubits.
        faults (Dict[str, Set[int]]):
        bindings (Dict[str, Callable]):

    """

    def __init__(self, *, num_qubits: int, track_sign: bool = False) -> None:
        """Initialize a PauliProp state.

        Args:
            num_qubits (int): Number of qubits in the system.
            track_sign (bool): Whether to track the global phase/sign.

        Returns: None

        """
        super().__init__()

        self.num_qubits = num_qubits
        self.track_sign = track_sign

        # Use Rust backend
        self._backend = PauliPropRs(num_qubits, track_sign=track_sign)

        # Set up optimized bindings for gates available in Rust backend
        self._setup_optimized_bindings()

        # Fall back to Python implementations for gates not in Rust
        for gate, func in bindings.gate_dict.items():
            if gate not in self.bindings:
                self.bindings[gate] = func

        # Add alternative symbols
        for k, v in alt_symbols.items():
            if v in self.bindings:
                self.bindings[k] = self.bindings[v]

    def _setup_optimized_bindings(self) -> None:
        """Set up direct bindings to Rust backend for supported gates."""
        self.bindings = {}
        backend = self._backend  # Local reference to avoid attribute lookup

        # Single-qubit gates - location is always an int
        self.bindings["H"] = lambda _s, q, **_p: backend.h(q)
        self.bindings["SX"] = lambda _s, q, **_p: backend.sx(q)
        self.bindings["SY"] = lambda _s, q, **_p: backend.sy(q)
        self.bindings["SZ"] = lambda _s, q, **_p: backend.sz(q)

        # Two-qubit gates - location is always a tuple
        self.bindings["CX"] = lambda _s, qs, **_p: backend.cx(
            qs[0],
            qs[1],
        )
        self.bindings["CY"] = lambda _s, qs, **_p: backend.cy(
            qs[0],
            qs[1],
        )
        self.bindings["CZ"] = lambda _s, qs, **_p: backend.cz(
            qs[0],
            qs[1],
        )
        self.bindings["SWAP"] = lambda _s, qs, **_p: backend.swap(
            qs[0],
            qs[1],
        )

        # Note: X, Y, Z are Pauli operators, not gates to apply to the state,
        # so they should still use the Python implementations

    @property
    def faults(self) -> dict:
        """Get the current faults dictionary."""
        return self._backend.faults

    @faults.setter
    def faults(self, value: dict) -> None:
        """Set the faults dictionary."""
        self._backend.set_faults(value)

    @property
    def sign(self) -> int:
        """Get the sign (0 for +, 1 for -)."""
        return 1 if self._backend.get_sign_bool() else 0

    @sign.setter
    def sign(self, value: int) -> None:
        """Set the sign."""
        # Reset sign to 0, then flip if needed
        current = self.sign
        if current != value:
            self._backend.flip_sign()

    @property
    def img(self) -> int:
        """Get the imaginary component (0 or 1)."""
        return self._backend.get_img_value()

    @img.setter
    def img(self, value: int) -> None:
        """Set the imaginary component."""
        # Determine how many flips needed to get to target value
        current = self.img
        if current != value:
            # If current is 0 and we want 1, flip once
            # If current is 1 and we want 0, flip 3 times (or once more to cycle back)
            if value == 1 and current == 0:
                self._backend.flip_img(1)
            elif value == 0 and current == 1:
                self._backend.flip_img(3)

    def flip_sign(self) -> None:
        """Flip the sign of the Pauli string."""
        self._backend.flip_sign()

    def flip_img(self, num_is: int) -> None:
        """Flip the imaginary component based on number of i factors.

        Args:
            num_is: Number of imaginary factors to add.
        """
        self._backend.flip_img(num_is)

    def logical_sign(self, logical_op: QuantumCircuit) -> int:
        """Find the sign of a logical operator.

        This is equivalent to determining if the faults commute (sign == 0) or
        anticommute (sign == 1) with the logical operator.

        That is, compare the commutation of `logical_op` with `faults.`

        Args:
            logical_op (QuantumCircuit): Quantum circuit representing a logical operator.

        Returns: int - sign.

        """
        return find_logical_signs(self, logical_op)

    def run_circuit(
        self,
        circuit: ParamGateCollection,
        removed_locations: set[int] | (set[tuple[int, ...]] | None) = None,
        *,
        _apply_faults: bool = False,
    ) -> None:
        """Used to apply a quantum circuit to a state, whether the circuit represents an fault or ideal circuit.

        Args:
            circuit: A class representing a circuit. # TODO: Shouldn't this also include QuantumCircuit
            removed_locations: A set of qudit locations that correspond to
                ideal gates that should be removed.
            _apply_faults: Whether to apply the `circuit` as a Pauli fault (True) or as a Clifford to update the
                faults (False).

        Returns: None

        """
        circuit_type = circuit.metadata.get("circuit_type")

        if circuit_type in {"faults", "recovery"}:
            self.add_faults(circuit)
            return None
        if not self._backend.is_identity():
            # Only apply gates if there are faults to act on
            return super().run_circuit(circuit, removed_locations)
        return None

        # need to return output?

    def add_faults(
        self,
        circuit: QuantumCircuit | ParamGateCollection,
        *,
        minus: bool = False,
    ) -> None:
        """A methods to add faults to the state.

        Args:
            circuit (Union[QuantumCircuit, ParamGateCollection]): A quantum circuit representing Pauli faults.
            minus (bool): Whether to flip the sign when adding faults.

        Returns: None

        """
        if self.track_sign and minus:
            self.flip_sign()

        for elem in circuit.items():
            if len(elem) == 2:
                symbol, locations = elem
            else:
                symbol, locations, _ = elem

            if symbol in {"X", "Y", "Z"}:
                # Convert locations to a dict for add_paulis
                paulis_dict = {symbol: locations}
                self._backend.add_paulis(paulis_dict)
            elif symbol != "I":
                msg = f"Got {symbol}. Can only handle Pauli errors."
                raise Exception(msg)

    def get_str(self) -> str:
        """Get string representation of the Pauli fault state.

        Returns:
            String representation with sign and Pauli operators.
        """
        return self._backend.to_dense_string()

    def fault_str_sign(self, *, strip: bool = False) -> str:
        """Get the sign component of the fault string.

        Args:
            strip: If True, strip leading/trailing whitespace.

        Returns:
            String representation of the sign component.
        """
        sign_str = self._backend.sign_string()

        # Convert to the expected format
        if sign_str == "+":
            fault_str = "+ "
        elif sign_str == "-":
            fault_str = "- "
        elif sign_str == "+i":
            fault_str = "+i"
        elif sign_str == "-i":
            fault_str = "-i"
        else:
            fault_str = sign_str

        if strip:
            fault_str = fault_str.strip()

        return fault_str

    def fault_str_operator(self) -> str:
        """Get the operator component of the fault string.

        Returns:
            String representation of the Pauli operators.
        """
        # Get the dense string and remove the sign part
        full_str = self._backend.to_dense_string()
        # Remove the sign prefix (+, -, +i, -i)
        if full_str.startswith(("+i", "-i")):
            return full_str[2:]
        if full_str.startswith(("+", "-")):
            return full_str[1:]
        return full_str

    def fault_string(self) -> str:
        """Get the complete fault string with sign and operators.

        Returns:
            Complete string representation of the fault state.
        """
        # Use the backend's string representation but format it for compatibility
        backend_str = self._backend.to_dense_string()
        # Ensure there's a space after the sign if no 'i'
        if backend_str.startswith("+") and not backend_str.startswith("+i"):
            return "+ " + backend_str[1:]
        if backend_str.startswith("-") and not backend_str.startswith("-i"):
            return "- " + backend_str[1:]
        return backend_str

    def fault_wt(self) -> int:
        """Get the weight of the fault (number of non-identity operators).

        Returns:
            Total weight of X, Y, and Z operators.
        """
        return self._backend.weight()

    def __str__(self) -> str:
        """Return string representation of the Pauli fault state."""
        faults = self.faults
        return "{{'X': {}, 'Y': {}, 'Z': {}}}".format(
            faults["X"],
            faults["Y"],
            faults["Z"],
        )
