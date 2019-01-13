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
    # from . import bindings

except ModuleNotFoundError:
    has_sim = False


class State(object):

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

        self.gate_dict = {
            'H': self.state.h,
            'X': self.state.x,
            'Y': self.state.y,
            'Z': self.state.z,
            'S': self.state.s,
            'T': self.state.t,
            'Q': self.state.sqrt_x,
            'CNOT': self.state.cnot,
            'TOFFOLI': self.state.toffoli,
            # 'U': None,  # TODO: Add this gate...
            # 'CU: None, # TODO: Add this gate...,

        }

    def run_gate(self, symbol, locations, **gate_kwargs):
        """

        Args:
            symbol:
            locations:
            **gate_kwargs: A dictionary specifying extra parameters for the gate.

        Returns:

        """

        angles = gate_kwargs.get('angles', ())

        # TODO: use the apply all method instead...

        output = {}
        for location in locations:
            if isinstance(location, int):
                results = self.gate_dict[symbol](self, location, *angles)
            else:
                results = self.gate_dict[symbol](self, *location, *angles)

            if results:
                output[location] = results

        return output
