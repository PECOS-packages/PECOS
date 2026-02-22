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

"""Base classes for AST optimization passes.

Optimization passes transform AST Programs to simplify quantum circuits.
They return new, optimized Programs (immutable transformation).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pecos.slr.ast.nodes import (
    ForStmt,
    GateOp,
    IfStmt,
    ParallelBlock,
    Program,
    RepeatStmt,
    WhileStmt,
)

if TYPE_CHECKING:
    from pecos.slr.ast.nodes import (
        Statement,
    )


@dataclass
class OptimizationResult:
    """Result of an optimization pass.

    Attributes:
        program: The optimized Program.
        gates_removed: Number of gates removed by the pass.
        gates_merged: Number of gate pairs merged into single gates.
        passes_applied: Names of optimization passes that were applied.
    """

    program: Program
    gates_removed: int = 0
    gates_merged: int = 0
    passes_applied: list[str] = field(default_factory=list)

    @property
    def total_optimizations(self) -> int:
        """Total number of optimizations performed."""
        return self.gates_removed + self.gates_merged


class OptimizationPass(ABC):
    """Base class for all AST optimization passes.

    Optimization passes transform an AST Program and return a new,
    optimized Program. They are composable and can be chained together.

    Subclasses must implement:
    - name: A human-readable name for the pass
    - optimize: The main optimization logic
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this pass."""
        ...

    @abstractmethod
    def optimize(self, program: Program) -> OptimizationResult:
        """Apply this optimization pass to a program.

        Args:
            program: The AST Program to optimize.

        Returns:
            OptimizationResult with the optimized program and statistics.
        """
        ...


class StatementListOptimizer(OptimizationPass):
    """Base class for passes that optimize sequences of statements.

    These passes scan consecutive statements looking for optimization
    opportunities (e.g., gate cancellation). They automatically handle
    recursion into control flow constructs.

    Subclasses must implement:
    - name: A human-readable name for the pass
    - _should_cancel: Determine if two consecutive gates should cancel/merge
    """

    def optimize(self, program: Program) -> OptimizationResult:
        """Recursively optimize all statement lists in the program."""
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
        """Optimize a sequence of statements.

        Scans for consecutive gate operations that can be cancelled or merged.
        Recursively optimizes inside control flow constructs.

        Returns:
            Tuple of (optimized_statements, count_of_gates_removed)
        """
        result: list[Statement] = []
        removed = 0
        i = 0

        while i < len(statements):
            stmt = statements[i]

            # Handle control flow recursively
            if isinstance(stmt, IfStmt):
                optimized, count = self._optimize_if(stmt)
                result.append(optimized)
                removed += count
                i += 1
                continue

            if isinstance(stmt, WhileStmt):
                optimized, count = self._optimize_while(stmt)
                result.append(optimized)
                removed += count
                i += 1
                continue

            if isinstance(stmt, ForStmt):
                optimized, count = self._optimize_for(stmt)
                result.append(optimized)
                removed += count
                i += 1
                continue

            if isinstance(stmt, RepeatStmt):
                optimized, count = self._optimize_repeat(stmt)
                result.append(optimized)
                removed += count
                i += 1
                continue

            if isinstance(stmt, ParallelBlock):
                optimized, count = self._optimize_parallel(stmt)
                result.append(optimized)
                removed += count
                i += 1
                continue

            # Check for cancellation with next gate
            if isinstance(stmt, GateOp) and i + 1 < len(statements) and isinstance(statements[i + 1], GateOp):
                next_stmt = statements[i + 1]
                if self._should_cancel(stmt, next_stmt):
                    # Skip both gates
                    removed += 2
                    i += 2
                    continue

            result.append(stmt)
            i += 1

        return tuple(result), removed

    @abstractmethod
    def _should_cancel(self, gate1: GateOp, gate2: GateOp) -> bool:
        """Determine if two consecutive gates should cancel.

        Args:
            gate1: The first gate.
            gate2: The second gate (immediately following gate1).

        Returns:
            True if the gates should be removed (cancelled).
        """
        ...

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
