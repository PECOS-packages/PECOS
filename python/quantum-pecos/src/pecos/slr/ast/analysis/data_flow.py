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

"""AST-based data flow analysis.

This module tracks how quantum and classical values flow through a program,
particularly tracking measurement results and their usage.

The key insight is distinguishing between:
1. Operations BEFORE measurement (don't require unpacking)
2. Operations AFTER measurement that use the SAME qubit (require unpacking)
3. Operations AFTER measurement that use DIFFERENT qubits (don't require unpacking)

Example:
    from pecos.slr.ast import slr_to_ast
    from pecos.slr.ast.analysis import analyze_data_flow

    ast = slr_to_ast(program)
    result = analyze_data_flow(ast)

    # Check if array needs unpacking
    if result.array_requires_unpacking("q"):
        print("Array q requires unpacking for proper data flow")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pecos.slr.ast.nodes import (
    AssignOp,
    BinaryExpr,
    BitExpr,
    BitRef,
    ForStmt,
    GateOp,
    IfStmt,
    MeasureOp,
    ParallelBlock,
    PrepareOp,
    RepeatStmt,
    UnaryExpr,
    WhileStmt,
)

if TYPE_CHECKING:
    from pecos.slr.ast.nodes import (
        Expression,
        Program,
        Statement,
    )


@dataclass
class ValueUse:
    """Represents a use of a value (qubit or classical bit)."""

    allocator: str
    index: int
    position: int  # Position in operation sequence
    operation_type: str  # "gate", "measurement", "preparation", "condition"
    is_consuming: bool = False  # True if this use consumes the value


@dataclass
class DataFlowInfo:
    """Information about data flow for a single array element."""

    allocator: str
    index: int
    is_classical: bool

    uses: list[ValueUse] = field(default_factory=list)
    consumed_at: list[int] = field(default_factory=list)
    replaced_at: list[int] = field(default_factory=list)

    def add_use(
        self,
        position: int,
        operation_type: str,
        *,
        is_consuming: bool = False,
    ) -> None:
        """Add a use of this value."""
        use = ValueUse(
            allocator=self.allocator,
            index=self.index,
            position=position,
            operation_type=operation_type,
            is_consuming=is_consuming,
        )
        self.uses.append(use)

        if is_consuming:
            self.consumed_at.append(position)

    def add_replacement(self, position: int) -> None:
        """Mark that this value is replaced at a position (e.g., Prep)."""
        self.replaced_at.append(position)

    def has_use_after_consumption(self) -> bool:
        """Check if this element is used after being consumed.

        This is the key analysis for determining if unpacking is needed.
        """
        if not self.consumed_at:
            return False

        first_consumption = min(self.consumed_at)

        for use in self.uses:
            if use.position > first_consumption and use.position not in self.replaced_at:
                # Check if there's a replacement between consumption and this use
                replacements_between = [r for r in self.replaced_at if first_consumption < r < use.position]
                if not replacements_between:
                    return True

        return False

    def requires_unpacking(self) -> bool:
        """Determine if this element requires unpacking based on data flow."""
        if self.is_classical:
            return False
        return self.has_use_after_consumption()


@dataclass
class DataFlowResult:
    """Complete data flow analysis result."""

    element_flows: dict[tuple[str, int], DataFlowInfo] = field(default_factory=dict)
    conditional_accesses: set[tuple[str, int]] = field(default_factory=set)

    def get_or_create_flow(
        self,
        allocator: str,
        index: int,
        is_classical: bool,
    ) -> DataFlowInfo:
        """Get or create data flow info for an array element."""
        key = (allocator, index)
        if key not in self.element_flows:
            self.element_flows[key] = DataFlowInfo(
                allocator=allocator,
                index=index,
                is_classical=is_classical,
            )
        return self.element_flows[key]

    def elements_requiring_unpacking(self) -> set[tuple[str, int]]:
        """Get the set of array elements that require unpacking."""
        return {key for key, flow in self.element_flows.items() if flow.requires_unpacking()}

    def array_requires_unpacking(self, allocator: str) -> bool:
        """Check if an array requires unpacking based on data flow."""
        return any(key[0] == allocator and flow.requires_unpacking() for key, flow in self.element_flows.items())


class DataFlowAnalyzer:
    """Analyzes data flow in AST programs using recursive descent."""

    def __init__(self) -> None:
        self.position = 0
        self.in_conditional = False
        self.result = DataFlowResult()

    def analyze(self, program: Program) -> DataFlowResult:
        """Analyze data flow in a program.

        Args:
            program: The AST Program to analyze.

        Returns:
            DataFlowResult containing all data flow information.
        """
        self.position = 0
        self.in_conditional = False
        self.result = DataFlowResult()

        for stmt in program.body:
            self._analyze_statement(stmt)

        return self.result

    def _analyze_statement(self, stmt: Statement) -> None:
        """Analyze a single statement using recursive descent."""
        if isinstance(stmt, MeasureOp):
            self._analyze_measure(stmt)
        elif isinstance(stmt, PrepareOp):
            self._analyze_prepare(stmt)
        elif isinstance(stmt, GateOp):
            self._analyze_gate(stmt)
        elif isinstance(stmt, AssignOp):
            self._analyze_assign(stmt)
        elif isinstance(stmt, IfStmt):
            self._analyze_if(stmt)
        elif isinstance(stmt, WhileStmt):
            self._analyze_while(stmt)
        elif isinstance(stmt, ForStmt):
            self._analyze_for(stmt)
        elif isinstance(stmt, RepeatStmt):
            self._analyze_repeat(stmt)
        elif isinstance(stmt, ParallelBlock):
            self._analyze_parallel(stmt)

        self.position += 1

    def _analyze_measure(self, node: MeasureOp) -> None:
        """Analyze measurement: consumes quantum values, creates classical."""
        for i, target in enumerate(node.targets):
            flow = self.result.get_or_create_flow(
                target.allocator,
                target.index,
                is_classical=False,
            )
            flow.add_use(self.position, "measurement", is_consuming=True)

            # Record classical output if present
            if i < len(node.results):
                result = node.results[i]
                c_flow = self.result.get_or_create_flow(
                    result.register,
                    result.index,
                    is_classical=True,
                )
                c_flow.add_use(self.position, "measurement_result", is_consuming=False)

    def _analyze_prepare(self, node: PrepareOp) -> None:
        """Analyze preparation: replaces quantum values."""
        if node.slots is not None:
            for slot in node.slots:
                flow = self.result.get_or_create_flow(
                    node.allocator,
                    slot,
                    is_classical=False,
                )
                flow.add_use(self.position, "preparation", is_consuming=False)
                flow.add_replacement(self.position)

    def _analyze_gate(self, node: GateOp) -> None:
        """Analyze gate: uses quantum values."""
        for target in node.targets:
            flow = self.result.get_or_create_flow(
                target.allocator,
                target.index,
                is_classical=False,
            )

            if self.in_conditional:
                flow.add_use(self.position, "conditional_gate", is_consuming=False)
                self.result.conditional_accesses.add((target.allocator, target.index))
            else:
                flow.add_use(self.position, "gate", is_consuming=False)

    def _analyze_assign(self, node: AssignOp) -> None:
        """Analyze assignment."""
        if isinstance(node.target, BitRef):
            flow = self.result.get_or_create_flow(
                node.target.register,
                node.target.index,
                is_classical=True,
            )
            flow.add_use(self.position, "assignment", is_consuming=False)

        self._analyze_expression(node.value)

    def _analyze_expression(self, expr: Expression) -> None:
        """Analyze expression for data flow."""
        if isinstance(expr, BitExpr):
            flow = self.result.get_or_create_flow(
                expr.ref.register,
                expr.ref.index,
                is_classical=True,
            )
            flow.add_use(self.position, "expression", is_consuming=False)
            if self.in_conditional:
                self.result.conditional_accesses.add(
                    (expr.ref.register, expr.ref.index),
                )
        elif isinstance(expr, BinaryExpr):
            self._analyze_expression(expr.left)
            self._analyze_expression(expr.right)
        elif isinstance(expr, UnaryExpr):
            self._analyze_expression(expr.operand)

    def _analyze_if(self, node: IfStmt) -> None:
        """Analyze if statement."""
        # Analyze condition
        prev_conditional = self.in_conditional
        self.in_conditional = True

        self._analyze_expression(node.condition)

        # Analyze then block
        for stmt in node.then_body:
            self._analyze_statement(stmt)

        # Analyze else block
        if node.else_body:
            for stmt in node.else_body:
                self._analyze_statement(stmt)

        self.in_conditional = prev_conditional

    def _analyze_while(self, node: WhileStmt) -> None:
        """Analyze while loop."""
        self._analyze_expression(node.condition)
        for stmt in node.body:
            self._analyze_statement(stmt)

    def _analyze_for(self, node: ForStmt) -> None:
        """Analyze for loop."""
        for stmt in node.body:
            self._analyze_statement(stmt)

    def _analyze_repeat(self, node: RepeatStmt) -> None:
        """Analyze repeat loop."""
        for stmt in node.body:
            self._analyze_statement(stmt)

    def _analyze_parallel(self, node: ParallelBlock) -> None:
        """Analyze parallel block."""
        for stmt in node.body:
            self._analyze_statement(stmt)


def analyze_data_flow(program: Program) -> DataFlowResult:
    """Convenience function to analyze data flow in an AST program.

    Args:
        program: The AST Program to analyze.

    Returns:
        DataFlowResult containing data flow information.
    """
    analyzer = DataFlowAnalyzer()
    return analyzer.analyze(program)
