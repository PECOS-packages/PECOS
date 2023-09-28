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

from .machine_abc import Machine
from ..reps.pypmir.op_types import QOp


class GenericMachine(Machine):
    """Represents generic, abstract machine."""

    def __init__(self, error_model=None, num_qubits=None):
        super().__init__(error_model, num_qubits)
        self.leaked_qubits = None
        self.lost_qubits = None

    def reset(self) -> None:
        """Reset state to initialization state."""
        self.leaked_qubits = set()
        self.lost_qubits = set()

    def init(self, machine_params=None, num_qubits=None):
        if machine_params:
            self.machine_params = machine_params

        self.num_qubits = num_qubits

    def shot_reinit(self):
        self.reset()

    def process_op(self, op):
        pass

    def process(self, op_buffer: list) -> list:

        for op in op_buffer:
            if "mop" in op:
                print("MOP >", op)

        return op_buffer

    def leak(self, qubits: set):
        """Starts tracking qubits as leaked qubits and calls the quantum simulation appropriately to trigger leakage."""
        self.leaked_qubits |= qubits
        return [QOp(name="Init", args=list(qubits), metadata={})]

    def unleak(self, qubits: set):
        """Untrack qubits as leaked qubits and calls the quantum simulation appropriately to trigger leakage."""
        self.leaked_qubits -= qubits

    def meas_leaked(self, qubits: set):
        # measuring = self.leaked_qubits & set(op.args)
        self.leaked_qubits -= qubits
        noisy_ops = [
            QOp(name="Init -Z", args=list(qubits), metadata={}),
        ]
        return noisy_ops

