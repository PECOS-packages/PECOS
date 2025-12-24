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

"""Block type definitions for PyPHIR intermediate representation.

This module defines block structures for PyPHIR (Python PECOS Medium-level Intermediate Representation) including
conditional blocks and control flow structures for quantum circuit execution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pecos.reps.pyphir.instr_type import Instr

if TYPE_CHECKING:
    from pecos.reps.pyphir.op_types import COp, Op, QOp


class Block(Instr):
    """General block type."""


class SeqBlock(Block):
    """A generic sequence block.

    This is not meant to indicate parallelism or lack of it but is just a way to structure operations/blocks.
    """

    def __init__(self, ops: list[Op | Block], metadata: dict | None = None) -> None:
        """Initialize a sequence block.

        Args:
            ops: List of operations or blocks to be executed in sequence.
            metadata: Optional metadata dictionary.
        """
        super().__init__(metadata=metadata)
        self.ops = ops


class QParallelBlock(SeqBlock):
    """A block to indicate that a collection of QOps are applied in parallel."""

    def __init__(self, ops: list[QOp], metadata: dict | None = None) -> None:
        """Initialize a parallel quantum operations block.

        Args:
            ops: List of quantum operations to be applied in parallel.
            metadata: Optional metadata dictionary.
        """
        super().__init__(ops=ops, metadata=metadata)


class IfBlock(Block):
    """If/else block."""

    def __init__(
        self,
        condition: COp,
        true_branch: list[Op],
        false_branch: list[Op] | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Initialize an if/else block.

        Args:
            condition: Classical operation that evaluates to a boolean condition.
            true_branch: List of operations to execute if condition is true.
            false_branch: Optional list of operations to execute if condition is false.
            metadata: Optional metadata dictionary.
        """
        super().__init__(metadata=metadata)
        self.condition = condition
        self.true_branch = true_branch
        self.false_branch = false_branch
