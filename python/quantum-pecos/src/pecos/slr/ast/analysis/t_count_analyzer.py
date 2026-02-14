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

"""T-count and T-depth analysis for quantum circuits.

T gates are the most expensive gates in fault-tolerant quantum computing,
requiring magic state distillation. This analyzer provides T-count and T-depth
metrics critical for resource estimation.

Example:
    from pecos.slr.ast import slr_to_ast
    from pecos.slr.ast.analysis import analyze_t_count

    ast = slr_to_ast(program)
    result = analyze_t_count(ast)

    print(f"T-count: {result.t_count}")
    print(f"T-depth: {result.t_depth}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pecos.slr.ast.nodes import (
    ForStmt,
    GateKind,
    GateOp,
    IfStmt,
    MeasureOp,
    ParallelBlock,
    PrepareOp,
    RepeatStmt,
    WhileStmt,
)

if TYPE_CHECKING:
    from pecos.slr.ast.nodes import (
        Program,
        Statement,
    )

# T gates (expensive in fault-tolerant QEC)
T_GATES = frozenset({GateKind.T, GateKind.Tdg})


@dataclass
class TCountResult:
    """Result of T-count analysis."""

    # Total T and Tdg gate count
    t_count: int = 0

    # T-depth (maximum T gates on any qubit's path)
    t_depth: int = 0

    # Breakdown by gate type
    breakdown: dict[str, int] = field(default_factory=dict)

    # List of T gate locations (for detailed analysis)
    t_gate_locations: list[tuple[str, int, int]] = field(default_factory=list)

    def __str__(self) -> str:
        return f"T-count: {self.t_count}, T-depth: {self.t_depth}"


class TCountAnalyzer:
    """Analyzes T-count and T-depth using recursive descent.

    T-depth is calculated as the maximum number of T gates on any
    qubit's critical path through the circuit.
    """

    def __init__(self) -> None:
        # T-depth for each qubit slot
        self.qubit_t_depth: dict[tuple[str, int], int] = {}
        self.max_t_depth = 0
        self.t_count = 0
        self.breakdown: dict[str, int] = {}
        self.t_gate_locations: list[tuple[str, int, int]] = []

    def analyze(self, program: Program) -> TCountResult:
        """Analyze T-count and T-depth of a program.

        Args:
            program: The AST Program to analyze.

        Returns:
            TCountResult with T-count and T-depth information.
        """
        self.qubit_t_depth = {}
        self.max_t_depth = 0
        self.t_count = 0
        self.breakdown = {}
        self.t_gate_locations = []

        for stmt in program.body:
            self._analyze_statement(stmt)

        return TCountResult(
            t_count=self.t_count,
            t_depth=self.max_t_depth,
            breakdown=dict(self.breakdown),
            t_gate_locations=list(self.t_gate_locations),
        )

    def _get_qubit_t_depth(self, allocator: str, index: int) -> int:
        """Get current T-depth for a qubit."""
        return self.qubit_t_depth.get((allocator, index), 0)

    def _set_qubit_t_depth(self, allocator: str, index: int, t_depth: int) -> None:
        """Set T-depth for a qubit."""
        self.qubit_t_depth[(allocator, index)] = t_depth
        if t_depth > self.max_t_depth:
            self.max_t_depth = t_depth

    def _analyze_statement(self, stmt: Statement) -> None:
        """Analyze a statement using recursive descent."""
        if isinstance(stmt, GateOp):
            self._analyze_gate(stmt)
        elif isinstance(stmt, MeasureOp | PrepareOp):
            pass  # No T-depth contribution
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

    def _analyze_gate(self, node: GateOp) -> None:
        """Analyze a gate's T-count/T-depth contribution."""
        if node.gate in T_GATES:
            # Increment T-count
            self.t_count += 1
            gate_name = node.gate.name
            self.breakdown[gate_name] = self.breakdown.get(gate_name, 0) + 1

            # Record location if available
            if node.location:
                self.t_gate_locations.append(
                    (node.gate.name, node.location.line, node.location.column),
                )

            # Update T-depth: T-depth increases on the path through this qubit
            for target in node.targets:
                current_depth = self._get_qubit_t_depth(target.allocator, target.index)
                self._set_qubit_t_depth(
                    target.allocator,
                    target.index,
                    current_depth + 1,
                )

    def _analyze_if(self, node: IfStmt) -> None:
        """Analyze an if statement.

        Takes maximum T-depth from both branches.
        """
        state_before = dict(self.qubit_t_depth)
        depth_before = self.max_t_depth
        count_before = self.t_count
        breakdown_before = dict(self.breakdown)

        # Analyze then branch
        for stmt in node.then_body:
            self._analyze_statement(stmt)

        state_after_then = dict(self.qubit_t_depth)
        depth_after_then = self.max_t_depth
        count_after_then = self.t_count
        breakdown_after_then = dict(self.breakdown)

        # Reset and analyze else branch
        self.qubit_t_depth = dict(state_before)
        self.max_t_depth = depth_before
        self.t_count = count_before
        self.breakdown = dict(breakdown_before)

        if node.else_body:
            for stmt in node.else_body:
                self._analyze_statement(stmt)

        # Merge: take maximum from each branch for T-depth
        for key in set(state_after_then.keys()) | set(self.qubit_t_depth.keys()):
            then_depth = state_after_then.get(key, 0)
            else_depth = self.qubit_t_depth.get(key, 0)
            self.qubit_t_depth[key] = max(then_depth, else_depth)

        self.max_t_depth = max(depth_after_then, self.max_t_depth)

        # For T-count, take maximum (conservative estimate for conditional)
        self.t_count = max(count_after_then, self.t_count)

        # Merge breakdown with max values
        all_gates = set(breakdown_after_then.keys()) | set(self.breakdown.keys())
        for gate in all_gates:
            then_count = breakdown_after_then.get(gate, 0)
            else_count = self.breakdown.get(gate, 0)
            self.breakdown[gate] = max(then_count, else_count)

    def _analyze_while(self, node: WhileStmt) -> None:
        """Analyze a while loop (single iteration, conservative)."""
        for stmt in node.body:
            self._analyze_statement(stmt)

    def _analyze_for(self, node: ForStmt) -> None:
        """Analyze a for loop (single iteration, conservative)."""
        for stmt in node.body:
            self._analyze_statement(stmt)

    def _analyze_repeat(self, node: RepeatStmt) -> None:
        """Analyze a repeat loop (all iterations)."""
        for _ in range(node.count):
            for stmt in node.body:
                self._analyze_statement(stmt)

    def _analyze_parallel(self, node: ParallelBlock) -> None:
        """Analyze a parallel block."""
        for stmt in node.body:
            self._analyze_statement(stmt)


def analyze_t_count(program: Program) -> TCountResult:
    """Convenience function to analyze T-count and T-depth.

    Args:
        program: The AST Program to analyze.

    Returns:
        TCountResult with T-count and T-depth information.
    """
    analyzer = TCountAnalyzer()
    return analyzer.analyze(program)
