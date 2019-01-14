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


try:
    import qcgpu
    from . import bindings
    has_sim = True

except ModuleNotFoundError:
    has_sim = False


class State:

    gate_dict = bindings.gate_dict

    def __init__(self, num_qubits):
        """
        Initializes a quantum state/register.
        """

        if not has_sim:
            raise Exception("ProjectQ must be installed to use this class.")

        if not isinstance(num_qubits, int):
            raise Exception('``num_qubits`` should be of type ``int.``')

        self.num_qubits = num_qubits

        self.state = qcgpu.State(num_qubits)

    def run_gate(self, symbol, locations, **gate_kwargs):
        """

        Args:
            symbol:
            locations:
            **gate_kwargs: A dictionary specifying extra parameters for the gate.

        Returns:

        """

        output = {}
        for location in locations:
            results = self.gate_dict[symbol](self.state, location, **gate_kwargs)

            if results:
                output[location] = results

        return output
