# -*- coding: utf-8 -*-

#  =========================================================================  #
#   Copyright 2018 Ciarán Ryan-Anderson
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

from typing import Set, Tuple, Union
from .._parent_sim_classes import BaseSim
from . import bindings
from ...circuits import QuantumCircuit
from ...circuits.quantum_circuit import ParamGateCollection


class PauliFaultProp(BaseSim):
    r"""
    A simulator that evolves Pauli faults through Clifford circuits.

    The unitary evolution of a Pauli follows :math:`PC = CP' \Leftrightarrow P' = C^{\dagger} P C`, where :math:`P` and
    :math:`P'` are Pauli operators and :math:`C` is a Clifford operator.

    Attributes:
        num_qubits(int): Number of qubits.
        faults (Dict[str, Set[int]]):
        gate_dict (Dict[str, Callable]):

    """

    def __init__(self, num_qubits: int) -> None:
        """

        Args:
            num_qubits (int):

        Returns: None

        """

        super().__init__()

        self.num_qubits = num_qubits
        self.faults = {
            'X': set(),
            'Y': set(),
            'Z': set(),
        }
        # Here we will encode Y as the qubit id in faults_x and faults_z

        self.gate_dict = bindings.gate_dict

    def run_circuit(self,
                    tick_circuit: ParamGateCollection,
                    removed_locations: Union[Set[int], Set[Tuple[int, ...]], None] = None,
                    apply_faults: bool = False) -> None:
        """
        Used to apply a quantum circuit to a state, whether the circuit represents an fault or ideal circuit.

        Args:
            tick_circuit (ParamGateCollection): A class representing a tick.
            removed_locations (Union[Set[int], Set[Tuple[int, ...]], None]): A set of qudit locations that correspond to
                ideal gates that should be removed.
            apply_faults (bool): Whether to apply the `circuit` as a Pauli fault (True) or as a Clifford to update the
                faults (False).

        Returns: None

        """

        if (tick_circuit.circuit.metadata.get('circuit_type') == 'faults' or
                tick_circuit.circuit.metadata.get('circuit_type') == 'recovery'):
            self.add_faults(tick_circuit)
        else:
            return super().run_circuit(tick_circuit, removed_locations)

    def add_faults(self, circuit: Union[QuantumCircuit, ParamGateCollection]) -> None:
        """
        A methods to add faults to the state.

        Args:
            circuit (Union[QuantumCircuit, ParamGateCollection]): A quantum circuit representing Pauli faults.

        Returns: None

        """

        for symbol, locations, params in circuit.items():
            if symbol in ['X', 'Y', 'Z'] and not params:
                self.faults[symbol].update(locations)
            else:
                raise Exception('Can only handle Pauli errors.')

    def __str__(self):
        return '{\'X\': %s, \'Y\': %s, \'Z\': %s}' % (self.faults['X'], self.faults['Y'], self.faults['Z'])
