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

from .standard import Standard


class ConvertingRunner(Standard):
    """
    This class represents a standard model for running quantum circuits and adding in errors.
    """

    def __init__(self, seed=None, simulator=None, converter=None):
        """

        Args:
            seed:
        """

        super().__init__(seed, simulator)

        self.converter = converter

    def run_circuit(self, state, circuit, give_output=False, removed_locations=None, gate_dict=None):
        """
        Apply a ``QuantumCircuit`` directly to a state without output.

        Args:
            state:
            circuit:
            give_output:
            removed_locations:
            gate_dict:

        Returns:

        """

        # Note: The circuit could be a tick or a full circuit.

        if self.converter:
            new_circuit = self.converter.compile(circuit)
        else:
            new_circuit = circuit

        result = super().run_circuit(state, new_circuit, give_output=give_output, removed_locations=removed_locations,
                                     gate_dict=gate_dict)

        return result
