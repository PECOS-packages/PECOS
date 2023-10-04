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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


class ForeignObject(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def init(self) -> None:
        """Initialize object before a set of simulations."""

    @abc.abstractmethod
    def new_instance(self) -> None:
        """Create new instance/internal state."""

    @abc.abstractmethod
    def get_funcs(self) -> list[str]:
        """Get a list of function names available from the object."""

    @abc.abstractmethod
    def exec(self, func_name: str, args: Sequence) -> tuple:
        """Execute a function given a list of arguments."""
