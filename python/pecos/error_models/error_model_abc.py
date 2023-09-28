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

import abc
from abc import ABCMeta
from typing import Union


class ErrorModel(metaclass=ABCMeta):

    def __init__(self):
        self.error_params = None
        self.machine = None
        self.num_qubits = None

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset state to initialization state."""
        pass

    def init(self, error_params, num_qubits, machine=None):
        self.machine = machine
        self.error_params = error_params
        self.num_qubits = num_qubits

    @abc.abstractmethod
    def shot_reinit(self, *args, **kwargs) -> None:
        """Run all code needed at the beginning of each shot, e.g., resetting state."""
        pass

    @abc.abstractmethod
    def process(self, qops: list, **kwargs) -> Union[list, None]:
        pass
