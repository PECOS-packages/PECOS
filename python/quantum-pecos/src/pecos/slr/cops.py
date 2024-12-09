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

from pecos.slr.fund import Expression

# ruff: noqa: F811


class PyCOp:
    def __isub__(self, other):
        raise NotImplementedError

    def __iadd__(self, other):
        raise NotImplementedError

    def __imul__(self, other):
        raise NotImplementedError

    def __idiv__(self, other):
        raise NotImplementedError

    def __ifloordiv__(self, other):
        raise NotImplementedError

    def __imod__(self, other):
        raise NotImplementedError

    def __ipow__(self, other):
        raise NotImplementedError

    def __irshift__(self, other):
        raise NotImplementedError

    def __ilshift__(self, other):
        raise NotImplementedError

    def __ior__(self, other):
        raise NotImplementedError

    def __ixor__(self, other):
        raise NotImplementedError

    def __invert__(self):
        """~a"""

        return NOT(self)

    def __neg__(self):
        """-a"""

        return NEG(self)

    def __xor__(self, other):
        """a ^ b"""

        return XOR(self, other)

    def __and__(self, other):
        """a & b"""

        return AND(self, other)

    def __or__(self, other):
        """a | b"""

        return OR(self, other)

    def __eq__(self, other):
        """a == b"""

        return EQUIV(self, other)

    def __ne__(self, other):
        """a != b"""

        return NEQUIV(self, other)

    def __lt__(self, other):
        """a < b"""

        return LT(self, other)

    def __gt__(self, other):
        """a > b"""
        return GT(self, other)

    def __le__(self, other):
        """a <= b"""
        return LE(self, other)

    def __ge__(self, other):
        """a >= b"""
        return GE(self, other)

    def __add__(self, other):
        """a + b"""
        return PLUS(self, other)

    def __sub__(self, other):
        """a - b"""
        return MINUS(self, other)

    def __rshift__(self, other):
        """a >> b"""
        return RSHIFT(self, other)

    def __lshift__(self, other):
        """a << b"""
        return LSHIFT(self, other)


class COp(Expression, PyCOp):
    """Classical operation"""


class BinOp(COp):
    """Binary Operation"""

    symbol = "@"

    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __repr__(self):
        return f"{self.__class__.__name__}({self.left}, {self.right})"


class AssignmentOp(BinOp):
    """Assignment Operation"""


class SET(AssignmentOp):
    """Set a variable to value of an expression (right)."""


class CompOp(COp):
    """Comparison Operation"""

    symbol = "@@"

    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __repr__(self):
        return f"{self.__class__.__name__}({self.left}, {self.right})"


class UnaryOp(COp):
    """Unary Operation"""

    symbol = "#"

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"{self.__class__.__name__}({self.value})"


class NOT(UnaryOp):
    symbol = "~"


class NEG(UnaryOp):
    symbol = "-"


class XOR(BinOp):
    symbol = "^"


class AND(BinOp):
    symbol = "&"


class OR(BinOp):
    symbol = "|"


class PLUS(BinOp):
    symbol = "+"


class MINUS(BinOp):
    symbol = "-"


class RSHIFT(BinOp):
    symbol = ">>"


class LSHIFT(BinOp):
    symbol = "<<"


class EQUIV(CompOp):
    symbol = "=="


class NEQUIV(CompOp):
    symbol = "!="


class LT(CompOp):
    symbol = "<"


class GT(CompOp):
    symbol = ">"


class LE(CompOp):
    symbol = "<="


class GE(CompOp):
    symbol = ">="


class MUL(BinOp):
    symbol = "*"


class DIV(BinOp):
    symbol = "/"
