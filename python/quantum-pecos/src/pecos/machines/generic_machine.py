# Copyright 2023 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

from __future__ import annotations

from typing import TYPE_CHECKING

from pecos.machines.machine_abc import Machine
from pecos.reps.pypmir.op_types import QOp

if TYPE_CHECKING:
    from pecos.reps.pypmir.op_types import MOp


class GenericMachine(Machine):
    """Represents generic, abstract machine."""

    def __init__(self, num_qubits: int | None = None) -> None:
        super().__init__(num_qubits=num_qubits)
        self.qubit_set = set()
        self.leaked_qubits = set()
        self.lost_qubits = set()

    def reset(self) -> None:
        """Reset state to initialization state."""
        self.leaked_qubits.clear()
        self.lost_qubits.clear()

    def init(self, num_qubits: int | None = None) -> None:
        self.num_qubits = num_qubits
        self.qubit_set = set(range(num_qubits))

    def shot_reinit(self) -> None:
        self.reset()

    def process(self, op_buffer: list[QOp | MOp]) -> list:
        for op in op_buffer:
            if "mop" in op.name:
                print("MOP >", op)

        return op_buffer

    def leak(self, qubits: set[int]) -> list[QOp]:
        """Starts tracking qubits as leaked qubits and calls the quantum simulation appropriately to trigger leakage."""
        self.leaked_qubits |= qubits
        return [QOp(name="Init", args=list(qubits))]

    def unleak(self, qubits: set[int]) -> list[QOp]:
        """Untrack qubits as leaked qubits and calls the quantum simulation appropriately to trigger leakage."""
        self.leaked_qubits -= qubits
        return []

    def meas_leaked(self, qubits: set[int]) -> list[QOp]:
        self.leaked_qubits -= qubits
        return [
            QOp(name="Init -Z", args=list(qubits)),
        ]
