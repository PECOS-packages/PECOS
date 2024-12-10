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

import abc


class Machine(metaclass=abc.ABCMeta):
    def __init__(
        self,
        machine_params: dict | None = None,
        num_qubits: int | None = None,
        metadata: dict | None = None,
        pos: dict | None = None,
    ) -> None:
        self.machine_params = machine_params
        if self.machine_params is not None:
            self.machine_params = dict(self.machine_params)
        self.num_qubits = num_qubits
        self.metadata = metadata
        self.pos = pos

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset state to initialization state."""

    @abc.abstractmethod
    def init(self, num_qubits: int | None = None) -> None:
        pass

    @abc.abstractmethod
    def shot_reinit(self) -> None:
        """Run all code needed at the beginning of each shot, e.g., resetting state."""

    @abc.abstractmethod
    def process(self, op_buffer: list) -> list:
        pass
