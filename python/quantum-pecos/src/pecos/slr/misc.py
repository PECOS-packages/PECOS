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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pecos.slr.vars import Elem, QReg, Qubit, Reg

from pecos.slr.block import Block
from pecos.slr.fund import Statement


class Barrier(Statement):
    def __init__(self, *qregs: QReg | tuple[QReg] | Qubit) -> None:
        self.qregs = qregs


class Comment(Statement):
    """A comment for human readability of output qasm."""

    def __init__(self, *txt, space: bool = True, newline: bool = True) -> None:
        self.space = space
        self.newline = newline
        self.txt = "\n".join(txt)


class Parallel(Block):
    """A block that indicates the contained statements can be executed in parallel.

    This is a hint to the compiler/simulator that the operations within this block
    are independent and can be executed simultaneously.
    """

    def __init__(self, *statements: Statement) -> None:
        super().__init__()
        self.extend(*statements)


class Permute(Statement):
    """Permutes the indices that the elements of the register so that Reg[i] now refers to Reg[j]."""

    def __init__(
        self,
        elems_i: list[Elem] | Reg,
        elems_f: list[Elem] | Reg,
        *,
        comment: bool = True,
    ) -> None:
        self.elems_i = elems_i
        self.elems_f = elems_f
        self.comment = comment


class Return(Statement):
    """Explicitly declares which variables a block returns.

    This operation is similar to Python's return statement and works in conjunction with
    the block_returns annotation (similar to Python's -> type annotation).

    Example:
        from pecos.slr import Block, QReg
        from pecos.slr.types import Array, QubitType
        from pecos.slr.misc import Return

        class MyBlock(Block):
            # Type annotation (like -> Type)
            block_returns = (Array[QubitType, 2], Array[QubitType, 7])

            def __init__(self, data, ancilla):
                super().__init__()
                # ... operations ...
                # Explicit return statement
                self.extend(Return(ancilla, data))
    """

    def __init__(self, *return_vars) -> None:
        """Initialize Return operation with variables to return.

        Args:
            *return_vars: Variables to return, in order. Can be QReg, Qubit, Bit, or other variables.
        """
        self.return_vars = return_vars
