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

from pecos.circuits.hyqc.fund import Expression


class COp(Expression):
    """Classical operation."""


class BinOp(COp):
    """Binary Operation."""

    def __init__(self, left, right) -> None:
        self.left = left
        self.right = right


class CompOp(COp):
    """Comparison Operation."""

    def __init__(self, left, right) -> None:
        self.left = left
        self.right = right


class UnaryOp(COp):
    """Unary Operation."""

    def __init__(self, value) -> None:
        self.value = value


class NOT(UnaryOp):
    """bitwise ~a."""


class XOR(BinOp):
    """bitwise a ^ b."""


class AND(BinOp):
    """bitwise a & b."""


class OR(BinOp):
    """bitwise a | b."""


class PLUS(BinOp):
    """int a + b."""


class MINUS(BinOp):
    """int a - b."""


class EQUIV(CompOp):
    """bool a == b."""


class NEQUIV(CompOp):
    """bool a != b."""


class LT(CompOp):
    """bool a < b."""


class GT(CompOp):
    """bool a > b."""


class LE(CompOp):
    """bool a <= b."""


class GE(CompOp):
    """bool a > b."""
