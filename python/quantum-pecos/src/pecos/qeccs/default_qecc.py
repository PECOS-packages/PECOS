# Copyright 2018 The PECOS Developers
# Copyright 2018 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract
# DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Contains the default implementation of the QECC protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, TypeVar

from pecos.circuit_converters.checks2circuit import Check2Circuits
from pecos.protocols import QECCProtocol
from pecos.qeccs.plot import plot_qecc

if TYPE_CHECKING:
    from collections.abc import Generator

    from pecos.protocols import LogicalGateProtocol, LogicalInstructionProtocol
    from pecos.typing import QECCGateParams, QECCInstrParams, QECCParams

T = TypeVar("T")


class QECCElement(Protocol):
    """Protocol for QECC elements that have symbol and params attributes."""

    symbol: str
    params: dict[str, Any]


class DefaultQECC:
    """Default QECC class providing standard implementations.

    A QECC's role is to output circuits for logical gates. Logical gates are a collection of logical instructions.
    Logical instructions might be called multiple times. To save time and memory, logical instructions are saved to
    the QECC instance.

    This class provides default implementations of the QECCProtocol interface.
    QECC implementations can inherit from this class to get the standard behavior,
    or implement the QECCProtocol directly for custom behavior.

    Warnings:
    --------
        Do not change the value of ``params`` once set during initialization.
    """

    def __init__(self, **qecc_params: QECCParams) -> None:
        """Initialize the default QECC base class.

        Args:
            **qecc_params: QECC parameters including:
                - distance: The code distance
                - circuit_compiler: Circuit compiler instance (default: Check2Circuits())
                - mapping: Optional qubit mapping (default: NoMap())
        """
        # Give name for others classes to identify this code
        # --------------------------------------------------
        self.name = None  # Name that identifies to other what QECC this is.

        # QECC parameters:
        # ----------------
        self.qecc_params = qecc_params  # The QECC's parameters such as distance.

        # D
        self.distance = None
        # K
        self.num_logical_qudits = None

        # Number of Qudits and Syndromes
        # ------------------------------
        # N
        self.num_data_qudits = None

        self.num_ancilla_qudits = None  # Total number of ancillas (usually == num_syndromes but not necessarily)
        # Number of qubits (num_qudits) is set as an attribute below.

        # Sets of qudit ids
        # -----------------
        self.qudit_set = set()  # Set of qudit ids used internally in the QECC
        self.data_qudit_set = set()
        self.ancilla_qudit_set = set()

        self.layout = {}  # A dictionary of qudit id => (x, y, ...)
        self.position2qudit = {}
        self.lattice_dimensions = {}  # Dimensions of the physical layout
        self.sides = {}  # Describes the geometry of the qecc so decoders can understand the QECC's shape

        self.sym2gate_class = {}  # symbol => logical gate class
        self.sym2instruction_class = {}  # symbol => logical instruction class

        self.circuit_compiler = qecc_params.get("circuit_compiler", Check2Circuits())

        # logical gates and instructions are stored in the QECC for reference

        self.instr_set = set()  # Instances of instructions that have been created.
        self.gate_set = set()  # Instances of gates that have been created.

        # Mapping
        # -------
        # Maps qudit id to new qudit id.
        self.mapping = self.qecc_params.get("mapping", NoMap())

    @property
    def num_qudits(self) -> int:
        """Get the total number of qudits (data + ancilla).

        Returns:
            Total number of qudits in the QECC.
        """
        return self.num_data_qudits + self.num_ancilla_qudits

    @property
    def available_gates(self) -> list[str]:
        """Get list of available logical gate symbols.

        Returns:
            List of gate symbols that can be applied to this QECC.
        """
        return list(self.sym2gate_class.keys())

    @property
    def available_instructions(self) -> list[str]:
        """Get list of available logical instruction symbols.

        Returns:
            List of instruction symbols that can be applied to this QECC.
        """
        return list(self.sym2instruction_class.keys())

    def plot(self, figsize: tuple[int, int] = (9, 9)) -> None:
        """Default plotter of the QECC.

        Args:
        ----
            figsize(tuple of int): The size of the plotted figure as (width, height).

        """
        plot_qecc(self, figsize)

    def gate(self, symbol: str, **gate_params: QECCGateParams) -> LogicalGateProtocol:
        """Returns a logical gate object.

        Args:
        ----
            symbol(str): The symbol/name of the gate to retrieve or create.
            **gate_params: Additional parameters to pass to the gate constructor.

        """
        # Recognize special symbol prefix
        if symbol.startswith("ideal "):
            # An ideal logical gate is one that has no errors and all random outcomes are forced to be zero.
            gate_params["error_free"] = True
            gate_params["forced_outcome"] = False
            symbol = symbol.replace("ideal ", "")

        gotten_gate = self._retrieve_element(symbol, gate_params, self.gate_set)

        # If None are found create a new one
        if gotten_gate is None:
            gotten_gate = self.sym2gate_class[symbol](self, symbol, **gate_params)
            self.gate_set.add(gotten_gate)

            # Create logical instructions
            # ---------------------------
            for instr_symbol in gotten_gate.instr_symbols:
                instr = self.instruction(instr_symbol, **gate_params)
                gotten_gate.circuits.append(instr.circuit)
                gotten_gate.instr_instances.append(instr)

        return gotten_gate

    def instruction(
        self,
        symbol: str,
        **instr_params: QECCInstrParams,
    ) -> LogicalInstructionProtocol:
        """Gets logical instruction given a string and parameters.

        Args:
        ----
            symbol: The symbol/name of the instruction to retrieve or create.
            **instr_params: Additional parameters to pass to the instruction constructor.

        Returns:
            LogicalInstructionProtocol: The requested logical instruction.

        """
        gotten_instr = self._retrieve_element(symbol, instr_params, self.instr_set)

        # If no instruction has been found corresponding to the symbol:
        if gotten_instr is None:
            instr_class = self.sym2instruction_class[symbol]

            gotten_instr = instr_class(self, symbol, **instr_params)
            self.instr_set.add(gotten_instr)

        return gotten_instr

    @staticmethod
    def _retrieve_element(
        symbol: str,
        params: dict[str, Any],
        element_set: set[QECCElement],
    ) -> QECCElement | None:
        """Retrieve an element from a set.

        Args:
        ----
            symbol: The symbol/identifier of the element to retrieve.
            params: Parameters that the element should match.
            element_set: The set of elements to search through.
            symbol(str):
            gate_params(dict):
            element_set(set):

        """
        gotten_element = None
        for element in element_set:
            if params == element.params and symbol == element.symbol:
                gotten_element = element
                break

        return gotten_element

    def _data_id_iter(self) -> Generator[int, None, None]:
        """Assigns qudit ids. Also, records qudit id in the sets self."""
        while True:
            qudit_id = max(self.qudit_set, default=-1) + 1
            self.qudit_set.add(qudit_id)
            self.data_qudit_set.add(qudit_id)

            if len(self.data_qudit_set) > self.num_data_qudits:
                msg = "Number of data qudits requested exceeds number expected."
                raise Exception(msg)

            yield qudit_id

    def _ancilla_id_iter(self) -> Generator[int, None, None]:
        """Assigns qudit ids. Also, records qudit id in the sets self."""
        last_ancilla_id = None

        while True:
            if len(self.ancilla_qudit_set) == self.num_ancilla_qudits:
                print("Requesting more qudits then expected assuming last ancilla id.")
                yield last_ancilla_id
            else:
                qudit_id = max(self.qudit_set, default=-1) + 1
                last_ancilla_id = qudit_id
                self.qudit_set.add(qudit_id)
                self.ancilla_qudit_set.add(qudit_id)

                yield qudit_id

    def _add_node(self, x: int, y: int, iter_ids: Generator[int, None, None]) -> None:
        nid = next(iter_ids)

        self.layout[nid] = (x, y)
        self.position2qudit[x, y] = nid

    def __eq__(self, other: object) -> bool:
        """Check equality with another QECC."""
        # Check if other implements the QECCProtocol
        if not isinstance(other, QECCProtocol):
            return NotImplemented
        return (self.name, self.qecc_params) == (other.name, other.qecc_params)

    def __ne__(self, other: object) -> bool:
        """Check inequality with another QECC."""
        return not (self == other)

    def __hash__(self) -> int:
        """Return hash of the QECC based on name and parameters."""
        return hash((self.name, tuple(sorted(self.qecc_params.items()))))


class NoMap:
    """Default Mapping: item -> item."""

    def __init__(self) -> None:
        """Initialize the NoMap identity mapping."""

    def __getitem__(self, item: T) -> T:
        """Return the item unchanged (identity mapping)."""
        return item
