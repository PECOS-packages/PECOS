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

"""Rotation merging optimization pass.

Merges consecutive rotation gates of the same type on the same qubits.
For example: RX(a) + RX(b) = RX(a+b).
"""

from __future__ import annotations

from pecos.slr.ast.nodes import (
    BinaryExpr,
    BinaryOp,
    ForStmt,
    GateOp,
    IfStmt,
    LiteralExpr,
    ParallelBlock,
    Program,
    RepeatStmt,
    Statement,
    WhileStmt,
)
from pecos.slr.ast.optimizations.base import OptimizationPass, OptimizationResult
from pecos.slr.ast.optimizations.gate_properties import is_rotation_gate, targets_match


class RotationMergingPass(OptimizationPass):
    """Merge consecutive rotation gates on the same qubits.

    Rotation gates with the same type acting on the same qubits can be
    merged by adding their angles: R(a) * R(b) = R(a+b).

    Supported merges:
    - RX(a) + RX(b) -> RX(a+b)
    - RY(a) + RY(b) -> RY(a+b)
    - RZ(a) + RZ(b) -> RZ(a+b)
    - RZZ(a) + RZZ(b) -> RZZ(a+b)

    When both angles are literals, the sum is computed at compile time.
    Otherwise, a symbolic BinaryExpr is created.

    Example:
        # Before optimization
        RX(0.5, q[0]), RX(0.3, q[0])

        # After optimization
        RX(0.8, q[0])
    """

    @property
    def name(self) -> str:
        return "rotation_merging"

    def optimize(self, program: Program) -> OptimizationResult:
        """Merge consecutive rotation gates in the program."""
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
            gates_merged=count,
            passes_applied=[self.name],
        )

    def _optimize_statements(self, statements: tuple[Statement, ...]) -> tuple[tuple[Statement, ...], int]:
        """Merge consecutive rotation gates in a sequence of statements."""
        result: list[Statement] = []
        merged = 0
        i = 0

        while i < len(statements):
            stmt = statements[i]

            # Handle control flow recursively
            if isinstance(stmt, IfStmt):
                optimized, count = self._optimize_if(stmt)
                result.append(optimized)
                merged += count
                i += 1
                continue

            if isinstance(stmt, WhileStmt):
                optimized, count = self._optimize_while(stmt)
                result.append(optimized)
                merged += count
                i += 1
                continue

            if isinstance(stmt, ForStmt):
                optimized, count = self._optimize_for(stmt)
                result.append(optimized)
                merged += count
                i += 1
                continue

            if isinstance(stmt, RepeatStmt):
                optimized, count = self._optimize_repeat(stmt)
                result.append(optimized)
                merged += count
                i += 1
                continue

            if isinstance(stmt, ParallelBlock):
                optimized, count = self._optimize_parallel(stmt)
                result.append(optimized)
                merged += count
                i += 1
                continue

            # Try to merge with next rotation
            if (
                isinstance(stmt, GateOp)
                and is_rotation_gate(stmt.gate)
                and i + 1 < len(statements)
                and isinstance(statements[i + 1], GateOp)
            ):
                next_stmt = statements[i + 1]
                if self._can_merge(stmt, next_stmt):
                    merged_gate = self._merge_rotations(stmt, next_stmt)
                    result.append(merged_gate)
                    merged += 1
                    i += 2
                    continue

            result.append(stmt)
            i += 1

        return tuple(result), merged

    def _can_merge(self, gate1: GateOp, gate2: GateOp) -> bool:
        """Check if two rotation gates can be merged.

        Gates can be merged if:
        1. They are the same rotation gate type
        2. They act on the same qubits
        3. Both have exactly one parameter (the angle)
        """
        return (
            gate1.gate == gate2.gate
            and is_rotation_gate(gate1.gate)
            and targets_match(gate1, gate2)
            and len(gate1.params) == 1
            and len(gate2.params) == 1
        )

    def _merge_rotations(self, gate1: GateOp, gate2: GateOp) -> GateOp:
        """Create a new rotation gate with merged angle.

        If both angles are literals, computes the sum at compile time.
        Otherwise, creates a symbolic BinaryExpr for the sum.
        """
        angle1 = gate1.params[0]
        angle2 = gate2.params[0]

        # Try to evaluate at compile time if both are literals
        if isinstance(angle1, LiteralExpr) and isinstance(angle2, LiteralExpr):
            merged_angle = LiteralExpr(value=angle1.value + angle2.value)
        else:
            # Create symbolic sum
            merged_angle = BinaryExpr(
                op=BinaryOp.ADD,
                left=angle1,
                right=angle2,
            )

        return GateOp(
            gate=gate1.gate,
            targets=gate1.targets,
            params=(merged_angle,),
            location=gate1.location,
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
