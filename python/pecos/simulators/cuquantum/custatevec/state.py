# Copyright 2022 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

from . import bindings
from ..cuconn import cq

try:
    from typing import Self
except:
    from typing_extensions import Self


class CuStateVec:
    def __init__(self, num_qubits: int) -> None:
        if not isinstance(num_qubits, int):
            raise Exception("``num_qubits`` should be of type ``int.``")

        self.num_qubits = num_qubits
        self.workspace = cq.CuStatevecWorkspace()
        self.statevec = cq.StateVector(num_qubits)
        self.statevec.init_on_device()
        self.workspace_gates = []

        # result = state.bindings[symbol](state, location, **params)
        self.bindings = bindings.gate_dict

    def get_probs(self) -> None:
        self.statevec.read_from_device()
        return self.statevec.get_probabilities(self.workspace)[0]

    def reset(self) -> Self:
        """Reset the quantum state for another run without reinitializing."""
        self.statevec.free_on_device()
        self.statevec.init_on_device()

        return self

    def __del__(self):
        wqs = list(self.workspace_gates)
        for g in wqs:
            g.free_on_device()
            self.workspace_gates.remove(g)

        self.statevec.free_on_device()
        del self.statevec
        del self.workspace
