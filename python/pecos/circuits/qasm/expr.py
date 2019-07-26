# Copyright 2021 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.


class BinaryOp:
    def __init__(self, sym, left, right) -> None:
        self.sym = sym
        self.left = left
        self.right = right

    def __str__(self) -> str:
        return f"{str(self.left)} {self.sym} {str(self.right)}"


class Assign(BinaryOp):
    """Assignment of a variable or number to a variable."""

    def __init__(self, left, right) -> None:
        super().__init__("=", left, right)


class Equiv(BinaryOp):
    """Equivalence between two variables."""

    def __init__(self, left, right) -> None:
        super().__init__("==", left, right)


class NE(BinaryOp):
    """Equivalence between two variables."""

    def __init__(self, left, right) -> None:
        super().__init__("!=", left, right)


class GT(BinaryOp):
    """>."""

    def __init__(self, left, right) -> None:
        super().__init__(">", left, right)


class GE(BinaryOp):
    """>=."""

    def __init__(self, left, right) -> None:
        super().__init__(">=", left, right)


class LT(BinaryOp):
    """>."""

    def __init__(self, left, right) -> None:
        super().__init__(">", left, right)


class LE(BinaryOp):
    """>=."""

    def __init__(self, left, right) -> None:
        super().__init__(">=", left, right)


class XOR(BinaryOp):
    """Exclusive OR."""

    def __init__(self, left, right) -> None:
        super().__init__("^", left, right)
