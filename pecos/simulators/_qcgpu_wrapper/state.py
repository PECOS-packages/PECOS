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

from ..sim_class_types import StateVector
import qcgpu
from . import bindings


class QCQPUSim(StateVector):
    """
    A wrapper for the qcqpu package: https://qcgpu.github.io/
    """

    def __init__(self, num_qubits):
        """
        Initializes the stabilizer state.

        Args:
            num_qubits (int): Number of qubits being represented.

        Returns:

        """

        if not isinstance(num_qubits, int):
            raise Exception('``num_qubits`` should be of type ``int.``')

        super().__init__()

        self.bindings = bindings.gate_dict
        self.num_qubits = num_qubits

        self.state = qcgpu.State(num_qubits)
