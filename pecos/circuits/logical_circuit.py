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
Provides class to represent a logical circuit.
"""

from .quantum_circuit import QuantumCircuit


class LogicalCircuit(QuantumCircuit):
    """
    Data structure used to represent a logical circuit.
    """

    def __init__(self, layout=None, suppress_warning=True, **params):

        self.layout = layout
        if self.layout is not None:
            self.qudit_set = set(self.layout.keys())
            self.num_qudits = len(self.qudit_set)
        else:
            self.num_qudits = None
            self.qudit_set = None

        self.suppress_warning = suppress_warning

        super().__init__(**params)

    def append(self, logical_gate, gate_locations=None, **params):
        """

        Args:
            logical_gate:
            gate_locations:
            **params:

        Returns:

        """

        if gate_locations is None and not isinstance(logical_gate, dict):
            super().append(logical_gate, frozenset([None]), **params)
        else:
            super().append(logical_gate, gate_locations, **params)

        if self.layout is None:

            qecc = logical_gate.qecc

            self.layout = qecc.layout
            self.num_qudits = qecc.num_qudits
            self.qudit_set = qecc.qudit_set

            if not self.suppress_warning:
                print('No layout given. The layout of the first QECC is assumed.')

        else:
            if isinstance(logical_gate, dict):
                for gate, _ in logical_gate.items():
                    # TODO: Probably not the best... need total number of qudits
                    qecc = gate.qecc
                    if self.num_qudits < qecc.num_qudits:
                        raise Exception('Number of QECC qudits greater than those in assumed layout (%s < %s)' %
                                        (self.num_qudits, qecc.num_qudits))

                    if self.qudit_set is None:
                        raise Exception('Qudit set should be set!')
            else:
                qecc = logical_gate.qecc
                if self.num_qudits < qecc.num_qudits:
                    raise Exception('Number of QECC qudits greater than those in assumed layout (%s < %s)' %
                                    (self.num_qudits, qecc.num_qudits))

                if self.qudit_set is None:
                    raise Exception('Qudit set should be set!')

    def update(self, symbol, locations=None, tick=-1, emptyappend=False, **params):
        raise NotImplementedError('!!!')

    def discard(self, locations, tick=-1):
        raise NotImplementedError('!!!')

    def iter_ticks(self):
        """An iterator for looping over the various quantum circuits comprising this data structure."""

        for logical_tick in range(len(self)):
            for logical_gate, _, _ in self.items(tick=logical_tick):
                for instr_index, instr_circuit in enumerate(logical_gate.circuits):
                    params = {'logical_circuit_params': self.params, 'gate': logical_gate, }
                    params.update(instr_circuit.params)

                    for tick in range(len(instr_circuit)):
                        tick_gates = instr_circuit[tick]
                        time = (logical_tick, instr_index, tick)
                        yield tick_gates, time, params

    def __iter__(self):

        for element in self._ticks:
            for gate, _, _ in element.items():
                yield gate

    def __str__(self):

        return "LogicalCircuit(%s)" % self._ticks

    def __repr__(self):

        return self.__str__()

    def __getitem__(self, tick):
        """Returns tick when instance[index] is used.

        Args:
            tick(int): Tick index of ``self._ticks``.

        Returns:

        """

        if isinstance(tick, int):

            return self._ticks[tick]
        else:
            logical_tick, instr_index, tick = tick
            tick_circuit = self[logical_tick]
            # (logical_tick, instr_index, tick)
