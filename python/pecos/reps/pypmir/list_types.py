from __future__ import annotations

from typing import TYPE_CHECKING

from pecos.reps.pypmir.instr_type import Instr
from pecos.reps.pypmir.op_types import Op, QOp
from pecos.typed_list import TypedList

if TYPE_CHECKING:
    from collections.abc import Iterable

# from pecos.reps.pypmir.op_types import Op, QOp, COp, MOp, EMOp
# from pecos.reps.pypmir.block_types import Block
# from pecos.reps.pypmir.data_types import Data


class InstrList(TypedList):
    """A list of general Instructions include Ops, Blocks, and Data."""

    _type = Instr

    def __init__(self, data: Iterable[Instr] | None = None) -> None:
        super().__init__(self._type, data)
        self.metadata = None


class OpList(InstrList):
    """A list of Operations, e.g., QOp, MOp,EMOp, etc.."""

    _type = Op

    def __init__(self, data: Iterable[Op] | None = None) -> None:
        super().__init__(data)


class QOpList(OpList):
    """A list of just QOps."""

    _type = QOp

    def __init__(self, data: Iterable[QOp] | None = None) -> None:
        super().__init__(data)
