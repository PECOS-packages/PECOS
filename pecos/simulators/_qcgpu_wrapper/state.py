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

from .. import BaseSim

try:
    import qcgpu
    from . import bindings
    has_sim = True

except ModuleNotFoundError:
    has_sim = False


class QCQPUSim(BaseSim):
    """
    A wrapper for the qcqpu package: https://qcgpu.github.io/
    """

    a = 0

    def __init__(self, num_qubits):
        """
        Initializes the stabilizer state.

        Args:
            num_qubits (int): Number of qubits being represented.

        Returns:

        """

        if not has_sim:
            raise Exception("ProjectQ must be installed to use this class.")

        if not isinstance(num_qubits, int):
            raise Exception('``num_qubits`` should be of type ``int.``')

        super().__init__()

        self.gate_dict = bindings.gate_dict
        self.num_qubits = num_qubits

        self.state = qcgpu.State(num_qubits)
