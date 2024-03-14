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
from abc import ABCMeta


class ErrorModel(metaclass=ABCMeta):
    def __init__(self, error_params: dict) -> None:
        self.error_params = dict(error_params)
        self.machine = None
        self.num_qubits = None

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset state to initialization state."""

    def init(self, num_qubits, machine=None):
        self.machine = machine
        self.num_qubits = num_qubits

    @abc.abstractmethod
    def shot_reinit(self, *args, **kwargs) -> None:
        """Run all code needed at the beginning of each shot, e.g., resetting state."""

    @abc.abstractmethod
    def process(self, qops: list, **kwargs) -> list | None:
        pass
