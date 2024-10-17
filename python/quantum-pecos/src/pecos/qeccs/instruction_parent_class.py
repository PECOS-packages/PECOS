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

"""Contains the parent classes for logical instructions."""

from pecos.qeccs.helper_functions import make_hashable_params
from pecos.qeccs.plot import plot_instr


class LogicalInstruction:
    """A parent class for logical instructions.

    Logical instructions are circuits that
    """

    def __init__(self, qecc, symbol, **params) -> None:
        """Args:
        ----
            qecc(QECC):
            symbol(str):
            **params:
        """
        self.symbol = symbol
        self.qecc = qecc  # The QECC object this instruction belongs to.
        self.params = params
        self.gate_params = (
            self.params
        )  # Parameters used in defining the logical instruction.
        # TODO: should this be the same as the gate parameters?
        self.abstract_circuit = None  # Abstract representation of the circuit.
        self.circuit = None  # Compiled circuit.

        # The following assumes the role of ancilla and data qudits stays fixed during the instruction
        self.data_qudit_set = (
            self.qecc.data_qudit_set
        )  # set of qudit ids corresponding to data qudits.
        self.ancilla_qudit_set = (
            self.qecc.ancilla_qudit_set
        )  # set of qudit ids corresponding to ancilla qudits.
        # The ancilla set may differ from qecc. (might be a subset)

        self.params["data_qudit_set"] = self.data_qudit_set
        self.params["ancilla_qudit_set"] = self.ancilla_qudit_set

        # Logical operations
        # These are the expected initial and final logical operations
        self.initial_logical_ops = {}
        self.final_logical_ops = {}

        self.params_tuple = make_hashable_params(params)  # Used for hashing.

    def plot(self, **kwargs):
        """Creates a plot of the logical instruction.

        Returns: None

        """
        plot_instr(self, **kwargs)

    def _compile_circuit(self, abstract_circuit, *args, **kwargs):
        """Create `circuit` instance from `abstract_circuit` instance for the logical instruction.

        If the instruction already has a `circuit` instance, do not bother compiling.
        """
        compiler = self.qecc.circuit_compiler
        compiler = self.params.get("circuit_compiler", compiler)

        self.circuit = compiler.compile(self, abstract_circuit, *args, **kwargs)

    def items(self):
        """Yields: Yields the."""
        if self.circuit:
            return self.circuit.items()
        else:
            msg = ""
            raise Exception(msg)

    def __str__(self) -> str:
        return f"[{self.qecc.name} {self.qecc.qecc_params}] - Logical instruction: '{self.symbol}' {self.params}"

    def __hash__(self):
        # The instruction is unique. A hash can be used to identify it.
        return hash(("instr", self.symbol, self.params_tuple))

    def __eq__(self, other):
        return (self.symbol, self.params_tuple, True) == (
            other.symbol,
            other.params_tuple,
            hasattr(other, "circuit"),
        )

    def __ne__(self, other):
        return not (self == other)
