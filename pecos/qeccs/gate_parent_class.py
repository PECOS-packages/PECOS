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
Contains the parent classes for logical gates.
"""
from .helper_functions import make_hashable_params
from .helper_functions import expected_params


class LogicalGate(object):
    """
    A parent class for logical gates.

    The main role of logical gates is to identify the sequence of logical instructions the gate is made out of.

    This class has methods to return a list of logical instruction objects and a list of circuits based on a list of
    instruction symbols, which is an attribute of the class.
    """

    def __init__(self, qecc, symbol, **gate_params):

        self.symbol = symbol
        self.qecc = qecc  # The qecc the gate is a member of.
        self.gate_params = gate_params  # Gate parameters
        self.instr_symbols = None
        self.instr_instances = []
        self.circuits = []  # The circuits of the logical instructions. (Either instr instances or a QuantumCircuit or
        # something with the same methods as a QuantumCircuit.)
        self.error_free = gate_params.get('error_free', False)  # Whether errors should occur for this gate.
        self.random_outcome = gate_params.get('random_outcome', True)  # Whether the measurements are random
        # (if True-> force -1)
        # Can choose 0 or 1.

        self.qecc_params_tuple = make_hashable_params(qecc.qecc_params)  # Used for hashing.
        self.gate_params_tuple = make_hashable_params(gate_params)  # Used for hashing.

    def final_instr(self):
        """
        Gives the final Logical Instruction instance.

        Returns:

        """

        return self.instr_instances[-1]

    def final_logical_stabs(self):
        """
        Gives the final_logical_ops dict.
        """
        return self.instr_instances[-1].final_logical_ops

    def iter_physical_ticks(self):
        """
        Used to loop through all the ideal physical ticks.

        Returns:

        """

        for instr_index, instr_circuit in enumerate(self.circuits):
            for tick_index in range(len(instr_circuit)):
                gates = instr_circuit[tick_index]

                yield instr_index, tick_index, gates

    def expected_params(self, params, expected_set):
        expected_params(params, expected_set)

    def __hash__(self):
        # Added so the logical gate can be a key (gate symbol) in a ``QuantumCircuit``.

        # These uniquely identify the logical and do not change.
        return hash(('gate', self.symbol, self.qecc_params_tuple, self.gate_params_tuple))

    def __eq__(self, other):
        return (self.symbol, self.qecc_params_tuple,
                self.gate_params_tuple, True) == (other.symbol, other.qecc_params_tuple, other.gate_params_tuple,
                                                  hasattr(other, 'instr_symbols'))

    def __ne__(self, other):
        return not(self == other)

    def __str__(self):

        return "Logical gate: '%s' params=%s - QECC: %s params=%s - Instructions: %s" % (self.symbol, self.gate_params,
                                                                                         self.qecc.name,
                                                                                         self.qecc.qecc_params,
                                                                                         self.instr_symbols)
