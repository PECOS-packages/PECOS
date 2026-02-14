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

"""AST circuit depth analysis.

This module analyzes the depth of a quantum circuit, tracking when each
qubit is available for the next operation.

Example:
    from pecos.slr.ast import slr_to_ast
    from pecos.slr.ast.analysis import analyze_depth

    ast = slr_to_ast(program)
    result = analyze_depth(ast)

    print(f"Circuit depth: {result.depth}")
    print(f"Critical path: {result.critical_path}")
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pecos.slr.ast.nodes import (
    ForStmt,
    GateKind,
    GateOp,
    IfStmt,
    MeasureOp,
    ParallelBlock,
    PrepareOp,
    Program,
    RepeatStmt,
    SlotRef,
    Statement,
    WhileStmt,
)


@dataclass
class DepthResult:
    """Result of circuit depth analysis."""

    # Maximum depth (longest path through the circuit)
    depth: int = 0

    # Depth at which each qubit becomes available
    qubit_depths: dict[tuple[str, int], int] = field(default_factory=dict)

    # Operations on the critical path
    critical_path: list[str] = field(default_factory=list)

    # Two-qubit gate depth (often more relevant for error rates)
    two_qubit_depth: int = 0

    def __str__(self) -> str:
        return f"Depth: {self.depth} (2Q depth: {self.two_qubit_depth})"


class DepthAnalyzer:
    """Analyzes circuit depth using recursive descent.

    Tracks when each qubit becomes available and computes the overall
    circuit depth based on operation dependencies.
    """

    def __init__(self) -> None:
        # When each qubit slot becomes available (depth level)
        self.qubit_available: dict[tuple[str, int], int] = {}
        self.max_depth = 0
        self.max_2q_depth = 0
        self.critical_ops: list[str] = []

    def analyze(self, program: Program) -> DepthResult:
        """Analyze depth of a program.

        Args:
            program: The AST Program to analyze.

        Returns:
            DepthResult with depth information.
        """
        self.qubit_available = {}
        self.max_depth = 0
        self.max_2q_depth = 0
        self.critical_ops = []

        # Process all statements
        for stmt in program.body:
            self._analyze_statement(stmt)

        return DepthResult(
            depth=self.max_depth,
            qubit_depths=dict(self.qubit_available),
            critical_path=self.critical_ops,
            two_qubit_depth=self.max_2q_depth,
        )

    def _get_qubit_depth(self, allocator: str, index: int) -> int:
        """Get current depth level of a qubit."""
        return self.qubit_available.get((allocator, index), 0)

    def _set_qubit_depth(self, allocator: str, index: int, depth: int) -> None:
        """Set depth level of a qubit."""
        self.qubit_available[(allocator, index)] = depth
        if depth > self.max_depth:
            self.max_depth = depth

    def _analyze_statement(self, stmt: Statement) -> None:
        """Analyze a statement using recursive descent."""
        if isinstance(stmt, GateOp):
            self._analyze_gate(stmt)
        elif isinstance(stmt, MeasureOp):
            self._analyze_measure(stmt)
        elif isinstance(stmt, PrepareOp):
            self._analyze_prepare(stmt)
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
        """Analyze a gate operation's depth contribution."""
        # Find when all target qubits are available
        start_depth = 0
        for target in node.targets:
            qubit_depth = self._get_qubit_depth(target.allocator, target.index)
            start_depth = max(start_depth, qubit_depth)

        # Gate executes at start_depth, completes at start_depth + 1
        new_depth = start_depth + 1

        # Update all target qubits
        for target in node.targets:
            self._set_qubit_depth(target.allocator, target.index, new_depth)

        # Track two-qubit depth
        if node.gate.arity >= 2:
            if new_depth > self.max_2q_depth:
                self.max_2q_depth = new_depth

        # Track critical path
        if new_depth == self.max_depth:
            self.critical_ops.append(node.gate.name)

    def _analyze_measure(self, node: MeasureOp) -> None:
        """Analyze a measurement's depth contribution."""
        for target in node.targets:
            qubit_depth = self._get_qubit_depth(target.allocator, target.index)
            new_depth = qubit_depth + 1
            self._set_qubit_depth(target.allocator, target.index, new_depth)

    def _analyze_prepare(self, node: PrepareOp) -> None:
        """Analyze a preparation's depth contribution."""
        if node.slots is not None:
            for slot in node.slots:
                qubit_depth = self._get_qubit_depth(node.allocator, slot)
                new_depth = qubit_depth + 1
                self._set_qubit_depth(node.allocator, slot, new_depth)

    def _analyze_if(self, node: IfStmt) -> None:
        """Analyze an if statement.

        Takes maximum depth from both branches.
        """
        # Save state
        state_before = dict(self.qubit_available)
        depth_before = self.max_depth

        # Analyze then branch
        for stmt in node.then_body:
            self._analyze_statement(stmt)

        state_after_then = dict(self.qubit_available)
        depth_after_then = self.max_depth

        # Reset and analyze else branch
        self.qubit_available = dict(state_before)
        self.max_depth = depth_before

        if node.else_body:
            for stmt in node.else_body:
                self._analyze_statement(stmt)

        # Merge: take maximum depth from each branch
        for key in set(state_after_then.keys()) | set(self.qubit_available.keys()):
            then_depth = state_after_then.get(key, 0)
            else_depth = self.qubit_available.get(key, 0)
            self.qubit_available[key] = max(then_depth, else_depth)

        self.max_depth = max(depth_after_then, self.max_depth)

    def _analyze_while(self, node: WhileStmt) -> None:
        """Analyze a while loop (single iteration)."""
        for stmt in node.body:
            self._analyze_statement(stmt)

    def _analyze_for(self, node: ForStmt) -> None:
        """Analyze a for loop (single iteration)."""
        for stmt in node.body:
            self._analyze_statement(stmt)

    def _analyze_repeat(self, node: RepeatStmt) -> None:
        """Analyze a repeat loop (all iterations)."""
        for _ in range(node.count):
            for stmt in node.body:
                self._analyze_statement(stmt)

    def _analyze_parallel(self, node: ParallelBlock) -> None:
        """Analyze a parallel block.

        In a parallel block, independent operations can run simultaneously.
        """
        # For parallel blocks, we analyze all statements but they execute
        # at the same depth level for non-conflicting qubits
        for stmt in node.body:
            self._analyze_statement(stmt)


def analyze_depth(program: Program) -> DepthResult:
    """Convenience function to analyze circuit depth.

    Args:
        program: The AST Program to analyze.

    Returns:
        DepthResult with depth information.
    """
    analyzer = DepthAnalyzer()
    return analyzer.analyze(program)
