# Copyright 2024 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

from numpy.typing import ArrayLike
from pyquest import Register

from pecos.simulators.quest import bindings
from pecos.simulators.sim_class_types import StateVector


class QuEST(StateVector):
    """Wrapper of QuEST statevector simulator"""

    def __init__(self, num_qubits) -> None:
        """
        Initializes the state vector.

        Args:
            num_qubits (int): Number of qubits being represented.
        """

        if not isinstance(num_qubits, int):
            msg = "``num_qubits`` should be of type ``int``."
            raise TypeError(msg)

        super().__init__()

        self.bindings = bindings.gate_dict
        self.num_qubits = num_qubits

        self.quest_state = Register(num_qubits)
        self.reset()

    def reset(self):
        """Reset the quantum state for another run without reinitializing."""
        # Initialize state vector to |0>
        self.quest_state[:] = 0
        self.quest_state[0] = 1
        return self

    @property
    def vector(self) -> ArrayLike:
        return self.quest_state[:]
