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

from __future__ import annotations

from pecos.circuits.hyqc.fund import Expression


class Var(Expression):
    """Type for variables."""

    def __init__(self, symbol: str | None = None) -> None:
        super().__init__()

        self.symbol = symbol

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        if self.symbol is not None:
            repr_str = f"{repr_str}:{self.symbol}"
        return f"<{repr_str} at {hex(id(self))}>"


class CVar(Var):
    """Type for classical variables."""


class QVar(Var):
    """Type for quantum variables."""
