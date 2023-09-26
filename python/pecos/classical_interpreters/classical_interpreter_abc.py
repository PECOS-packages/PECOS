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
from typing import Any, Generator, Optional, Sequence


class ClassicalInterpreter(metaclass=abc.ABCMeta):

    def __init__(self) -> None:
        self.program = None
        self.foreign_obj = None

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset state to initialization state."""
        pass

    @abc.abstractmethod
    def init(self, program: Any, foreign_classical_obj: Optional[object] = None) -> int:
        pass

    @abc.abstractmethod
    def optimize(self, machine=None, error_model=None, qsim=None) -> None:
        pass

    @abc.abstractmethod
    def shot_reinit(self) -> None:
        pass

    @abc.abstractmethod
    def execute(self, sequence: Optional[Sequence]) -> Generator:
        pass

    @abc.abstractmethod
    def recieve_results(self, qsim_results):
        pass

    @abc.abstractmethod
    def results(self) -> dict:
        """Dumps program final results"""
        pass
