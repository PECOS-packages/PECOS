# Copyright 2024 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""List type definitions for PyPHIR intermediate representation.

This module defines specialized list types for PyPHIR (Python PECOS Medium-level Intermediate Representation) including
typed lists for instructions, operations, and other quantum circuit elements.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pecos.reps.pyphir.instr_type import Instr
from pecos.reps.pyphir.op_types import Op, QOp
from pecos.typed_list import TypedList

if TYPE_CHECKING:
    from collections.abc import Iterable


class InstrList(TypedList):
    """A list of general Instructions include Ops, Blocks, and Data."""

    _type = Instr

    def __init__(self, data: Iterable[Instr] | None = None) -> None:
        """Initialize an InstrList.

        Args:
            data: Optional iterable of Instr objects to initialize the list.
        """
        super().__init__(self._type, data)
        self.metadata = None


class OpList(InstrList):
    """A list of Operations, e.g., QOp, MOp,EMOp, etc.."""

    _type = Op

    def __init__(self, data: Iterable[Op] | None = None) -> None:
        """Initialize an OpList.

        Args:
            data: Optional iterable of Op objects to initialize the list.
        """
        super().__init__(data)


class QOpList(OpList):
    """A list of just QOps."""

    _type = QOp

    def __init__(self, data: Iterable[QOp] | None = None) -> None:
        """Initialize a QOpList.

        Args:
            data: Optional iterable of QOp (quantum operation) objects to initialize the list.
        """
        super().__init__(data)
