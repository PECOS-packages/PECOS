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

from pecos.circuits.hyqc import cops
from pecos.circuits.hyqc.vars import CVar


class Int(CVar):
    def __init__(self, size: int, symbol: str | None = None) -> None:
        """Representation for a collection of bits.

        Todo:
        ----

        Args:
        ----
            symbol:
            size:
        """
        super().__init__(symbol)
        self.size = size

    def __getitem__(self, item):
        return Bit(self, item)

    def __xor__(self, other):
        """A ^ b."""
        return cops.XOR(self, other)

    def __and__(self, other):
        """A & b."""
        return cops.AND(self, other)

    def __or__(self, other):
        """A | b."""
        return cops.OR(self, other)

    def __eq__(self, other):
        """A == b."""
        return cops.EQUIV(self, other)

    def __ne__(self, other):
        """A != b."""
        return cops.NEQUIV(self, other)

    def __lt__(self, other):
        """A < b."""
        return cops.LT(self, other)

    def __gt__(self, other):
        """A > b."""
        return cops.GT(self, other)

    def __le__(self, other):
        """A <= b."""
        return cops.LE(self, other)

    def __ge__(self, other):
        """A >= b."""
        return cops.GE(self, other)

    def __add__(self, other):
        """A + b."""

    def __sub__(self, other):
        """A - b."""

    def __rshift__(self, other):
        """A >> b."""

    def __lshift__(self, other):
        """A << b."""

    def __invert__(self): ...

    def __ixor__(self, other):
        """A ^= b."""

    def __iand__(self, other):
        """A &= b."""

    def __ior__(self, other):
        """A |= b."""


class Bit(Int):
    def __init__(self, ints: Int, idx: int, symbol: str | None = None) -> None:
        if symbol is None:
            symbol = f"{ints.symbol}[{idx}]"

        super().__init__(size=1, symbol=symbol)
        self.int = ints
        self.index = idx

    def __str__(self) -> str:
        return f"{self.symbol}"

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} {self.index} of {self.int.__repr__()[1:-1]}>"
        )
