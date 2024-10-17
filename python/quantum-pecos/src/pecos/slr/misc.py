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

from pecos.slr.fund import Statement


class Barrier(Statement):
    def __init__(self, *qregs: QReg | tuple[QReg] | Qubit):
        self.qregs = qregs


class Comment(Statement):
    """A comment for human readability of output qasm."""

    def __init__(self, *txt, space: bool = True, newline: bool = True):
        self.space = space
        self.newline = newline
        self.txt = "\n".join(txt)


class Permute(Statement):
    """Permutes the indices that the elements of the register so that Reg[i] now refers to Reg[j]."""

    def __init__(
        self,
        elems_i: list[Elem] | Reg,
        elems_f: list[Elem] | Reg,
        *,
        comment: bool = True,
    ):
        self.elems_i = elems_i
        self.elems_f = elems_f
        self.comment = comment
