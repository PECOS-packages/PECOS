# Copyright 2019 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

from typing import Any


class MakeFunc:
    """Converts ProjectQ gate to a function."""

    def __init__(self, gate, angle=False) -> None:
        """Args:
        ----
            gate:
        """
        self.gate = gate
        self.angle = angle

    def func(self, state, qubits, **params: Any):
        if isinstance(qubits, int):
            qs = state.qids[qubits]
        else:
            qs = []
            for q in qubits:
                qs.append(state.qids[q])

        if self.angle:
            self.gate(params["angle"]) | qs
        else:
            self.gate | qs
