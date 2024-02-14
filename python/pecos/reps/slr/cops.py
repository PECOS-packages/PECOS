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

from pecos.slr.fund import Expression


class COp(Expression):
    """Classical operation"""


class BinOp(COp):
    """Binary Operation"""

    def __init__(self, left, right):
        self.left = left
        self.right = right


class Assign(BinOp):
    def qasm(self):
        return f"{self.left} = {self.right};"


class CompOp(COp):
    """Comparison Operation"""

    def __init__(self, left, right):
        self.left = left
        self.right = right


class UnaryOp(COp):
    """Unary Operation"""

    def __init__(self, value):
        self.value = value


class NOT(UnaryOp):
    """bitwise ~a"""


class XOR(BinOp):
    """bitwise a ^ b"""


class AND(BinOp):
    """bitwise a & b"""


class OR(BinOp):
    """bitwise a | b"""


class PLUS(BinOp):
    """int a + b"""


class MINUS(BinOp):
    """int a - b"""


class EQUIV(CompOp):
    """bool a == b"""

    def qasm(self):
        return f"{self.left} == {self.right}"


class NEQUIV(CompOp):
    """bool a != b"""

    def qasm(self):
        return f"{self.left} != {self.right}"


class LT(CompOp):
    """bool a < b"""


class GT(CompOp):
    """bool a > b"""


class LE(CompOp):
    """bool a <= b"""


class GE(CompOp):
    """bool a > b"""


class PyCOp:
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

    def __sub__(self, other):
        """a - b"""

    def __rshift__(self, other):
        """a >> b"""

    def __lshift__(self, other):
        """a << b"""

    def __invert__(self): ...

    def __ixor__(self, other):
        """a ^= b"""

    def __iand__(self, other):
        """a &= b"""

    def __ior__(self, other):
        """a |= b"""
