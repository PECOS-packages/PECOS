#  =========================================================================  #
#   Copyright 2018 National Technology & Engineering Solutions of Sandia,
#   LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS,
#   the U.S. Government retains certain rights in this software.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#  =========================================================================  #

"""
Contains the parent classes for logical instructions.
"""
from .plot import plot_instr
from .helper_functions import make_hashable_params


class LogicalInstruction(object):
    """
    A parent class for logical instructions.

    Logical instructions are circuits that
    """

    def __init__(self, qecc, symbol, **gate_params):
        """

        Args:
            qecc(QECC):
            symbol(str):
            **params:
        """

        self.symbol = symbol
        self.qecc = qecc  # The QECC object this instruction belongs to.
        self.gate_params = gate_params  # Parameters used in defining the logical instruction.
        self.abstract_circuit = None  # Abstract representation of the circuit.
        self.circuit = None  # Compiled circuit.

        # The following assumes the role of ancilla and data qudits stays fixed during the instruction
        self.data_qudit_set = self.qecc.data_qudit_set  # set of qudit ids corresponding to data qudits.
        self.ancilla_qudit_set = self.qecc.ancilla_qudit_set  # set of qudit ids corresponding to ancilla qudits.
        # The ancilla set may differ from qecc. (might be a subset)

        # Logical operations
        # These are the expected initial and final logical operations
        self.initial_logical_ops = {}
        self.final_logical_ops = {}

        self.gate_params_tuple = make_hashable_params(gate_params)  # Used for hashing.

    def plot(self, **kwargs):
        """Creates a plot of the logical instruction.

        Returns: None

        """

        plot_instr(self, **kwargs)

    def _compile_circuit(self, abstract_circuit, *args, **kwargs):
        """
        Create `circuit` instance from `abstract_circuit` instance for the logical instruction.

        If the instruction already has a `circuit` instance, do not bother compiling.
        """

        compiler = self.qecc.circuit_compiler
        compiler = self.gate_params.get('circuit_compiler', compiler)

        # self.circuit = compiler.compile(self.abstract_circuit, mapping=self.qecc.mapping, *args, **kwargs)
        self.circuit = compiler.compile(self, abstract_circuit, *args, **kwargs)

    def items(self, params=True):
        """

        Args:
            params (bool): Whether to return circuit parameters.

        Yields: Yields the

        """

        if self.circuit:
            return self.circuit.items(params=params)
        else:
            raise Exception('')

    def __str__(self):

        return "[%s %s] - Logical instruction: '%s' %s" % (self.qecc.name, self.qecc.qecc_params, self.symbol,
                                                           self.gate_params)

    def __hash__(self):
        # The instruction is unique. A hash can be used to identify it.
        return hash(('instr', self.symbol, self.gate_params_tuple))

    def __eq__(self, other):
        return (self.symbol, self.gate_params_tuple, True) == (other.symbol, other.gate_params_tuple,
                                                               hasattr(other, 'circuit'))

    def __ne__(self, other):
        return not(self == other)
