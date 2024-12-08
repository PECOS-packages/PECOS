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


class OpProcessor(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def reset(self) -> None:
        """Reset state to initialization state."""

    @abc.abstractmethod
    def init(self) -> None:
        pass

    @abc.abstractmethod
    def shot_reinit(self) -> None:
        pass

    @abc.abstractmethod
    def process(self, buffered_ops: list) -> list:
        pass

    @abc.abstractmethod
    def process_meas(self, measurements: dict) -> dict:
        pass
