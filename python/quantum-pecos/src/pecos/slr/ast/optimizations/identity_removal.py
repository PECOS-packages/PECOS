# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Identity removal optimization pass.

Removes rotation gates that are equivalent to identity (angle = 0 or 2*pi*n).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from pecos.slr.ast.nodes import (
    ForStmt,
    GateOp,
    IfStmt,
    LiteralExpr,
    ParallelBlock,
    Program,
    RepeatStmt,
    WhileStmt,
)
from pecos.slr.ast.optimizations.base import OptimizationPass, OptimizationResult
from pecos.slr.ast.optimizations.gate_properties import is_rotation_gate

if TYPE_CHECKING:
    from pecos.slr.ast.nodes import (
        Statement,
    )


class IdentityRemovalPass(OptimizationPass):
    """Remove rotation gates that are equivalent to identity.

    Rotation gates with angle = 0 or angle = 2*pi*n are identity operations
    and can be safely removed.

    Supported removals:
    - RX(0), RY(0), RZ(0)
    - RZZ(0)
    - Any rotation with angle that is a multiple of 2*pi

    Example:
        # Before optimization
        RZ(0, q[0])

        # After optimization
        (empty - gate removed)
    """

    # Tolerance for floating-point comparison
    IDENTITY_ANGLE_TOLERANCE = 1e-10

    @property
    def name(self) -> str:
        return "identity_removal"

    def optimize(self, program: Program) -> OptimizationResult:
        """Remove identity rotation gates from the program."""
        optimized_body, count = self._optimize_statements(program.body)

        new_program = Program(
            name=program.name,
            declarations=program.declarations,
            body=optimized_body,
            returns=program.returns,
            allocator=program.allocator,
            location=program.location,
        )

        return OptimizationResult(
            program=new_program,
            gates_removed=count,
            passes_applied=[self.name],
        )

    def _optimize_statements(
        self,
        statements: tuple[Statement, ...],
    ) -> tuple[tuple[Statement, ...], int]:
        """Remove identity gates from a sequence of statements."""
        result: list[Statement] = []
        removed = 0

        for stmt in statements:
            # Handle control flow recursively
            if isinstance(stmt, IfStmt):
                optimized, count = self._optimize_if(stmt)
                result.append(optimized)
                removed += count
                continue

            if isinstance(stmt, WhileStmt):
                optimized, count = self._optimize_while(stmt)
                result.append(optimized)
                removed += count
                continue

            if isinstance(stmt, ForStmt):
                optimized, count = self._optimize_for(stmt)
                result.append(optimized)
                removed += count
                continue

            if isinstance(stmt, RepeatStmt):
                optimized, count = self._optimize_repeat(stmt)
                result.append(optimized)
                removed += count
                continue

            if isinstance(stmt, ParallelBlock):
                optimized, count = self._optimize_parallel(stmt)
                result.append(optimized)
                removed += count
                continue

            # Check if this is an identity rotation gate
            if isinstance(stmt, GateOp) and self._is_identity(stmt):
                removed += 1
                continue

            result.append(stmt)

        return tuple(result), removed

    def _is_identity(self, gate: GateOp) -> bool:
        """Check if a rotation gate is equivalent to identity.

        A rotation is identity if:
        1. It's a rotation gate (RX, RY, RZ, RZZ)
        2. It has exactly one parameter (the angle)
        3. The angle is a literal that equals 0 or a multiple of 2*pi
        """
        if not is_rotation_gate(gate.gate):
            return False

        if len(gate.params) != 1:
            return False

        angle = gate.params[0]
        if not isinstance(angle, LiteralExpr):
            # Can't evaluate symbolic angles at compile time
            return False

        value = angle.value
        if not isinstance(value, (int, float)):
            return False

        # Check for 0
        if abs(value) < self.IDENTITY_ANGLE_TOLERANCE:
            return True

        # Check for multiples of 2*pi
        normalized = value % (2 * math.pi)
        return (
            abs(normalized) < self.IDENTITY_ANGLE_TOLERANCE
            or abs(normalized - 2 * math.pi) < self.IDENTITY_ANGLE_TOLERANCE
        )

    def _optimize_if(self, stmt: IfStmt) -> tuple[IfStmt, int]:
        """Recursively optimize an if statement."""
        then_body, then_count = self._optimize_statements(stmt.then_body)
        else_body, else_count = self._optimize_statements(stmt.else_body) if stmt.else_body else ((), 0)

        optimized = IfStmt(
            condition=stmt.condition,
            then_body=then_body,
            else_body=else_body,
            location=stmt.location,
        )
        return optimized, then_count + else_count

    def _optimize_while(self, stmt: WhileStmt) -> tuple[WhileStmt, int]:
        """Recursively optimize a while statement."""
        body, count = self._optimize_statements(stmt.body)

        optimized = WhileStmt(
            condition=stmt.condition,
            body=body,
            location=stmt.location,
        )
        return optimized, count

    def _optimize_for(self, stmt: ForStmt) -> tuple[ForStmt, int]:
        """Recursively optimize a for statement."""
        body, count = self._optimize_statements(stmt.body)

        optimized = ForStmt(
            variable=stmt.variable,
            start=stmt.start,
            stop=stmt.stop,
            step=stmt.step,
            body=body,
            location=stmt.location,
        )
        return optimized, count

    def _optimize_repeat(self, stmt: RepeatStmt) -> tuple[RepeatStmt, int]:
        """Recursively optimize a repeat statement."""
        body, count = self._optimize_statements(stmt.body)

        optimized = RepeatStmt(
            count=stmt.count,
            body=body,
            location=stmt.location,
        )
        return optimized, count

    def _optimize_parallel(self, stmt: ParallelBlock) -> tuple[ParallelBlock, int]:
        """Recursively optimize a parallel block."""
        body, count = self._optimize_statements(stmt.body)

        optimized = ParallelBlock(
            body=body,
            location=stmt.location,
        )
        return optimized, count
