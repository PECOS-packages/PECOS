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

"""AST-based analyzer for qubit usage patterns.

This module analyzes how qubits are used in a program to:
- Classify qubits as DATA or ANCILLA based on usage patterns
- Track lifetimes and access patterns
- Provide allocation recommendations

Example:
    from pecos.slr.ast import slr_to_ast
    from pecos.slr.ast.analysis import analyze_qubit_usage

    ast = slr_to_ast(program)
    result = analyze_qubit_usage(ast)

    for name, stats in result.allocator_stats.items():
        print(f"{name}: {stats.classify_role()}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

from pecos.slr.ast.nodes import (
    AllocatorDecl,
    ForStmt,
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


class QubitRole(Enum):
    """Role classification for quantum allocators."""

    DATA = auto()  # Long-lived data qubits
    ANCILLA = auto()  # Short-lived ancilla qubits
    UNKNOWN = auto()  # Not yet classified


@dataclass
class QubitUsageStats:
    """Statistics about how a quantum allocator is used."""

    name: str
    capacity: int

    # Usage patterns
    measurement_count: int = 0
    reset_count: int = 0
    gate_count: int = 0

    # Lifetime tracking
    first_use_position: int | None = None
    last_use_position: int | None = None
    measurement_positions: list[int] = field(default_factory=list)
    reset_positions: list[int] = field(default_factory=list)

    # Access patterns
    individual_accesses: set[int] = field(default_factory=set)
    full_array_accesses: int = 0

    @property
    def lifetime(self) -> int:
        """Calculate the lifetime of this allocator."""
        if self.first_use_position is None or self.last_use_position is None:
            return 0
        return self.last_use_position - self.first_use_position

    @property
    def measure_reset_ratio(self) -> float:
        """Ratio of measurements+resets to total operations."""
        total_ops = self.measurement_count + self.reset_count + self.gate_count
        if total_ops == 0:
            return 0.0
        return (self.measurement_count + self.reset_count) / total_ops

    @property
    def individual_access_ratio(self) -> float:
        """Ratio of individual element accesses to capacity."""
        if self.capacity == 0:
            return 0.0
        return len(self.individual_accesses) / self.capacity

    def classify_role(self) -> QubitRole:
        """Classify the role of this allocator based on usage patterns."""
        # Explicit ancilla naming patterns
        name_lower = self.name.lower()
        if any(p in name_lower for p in ["ancilla", "anc", "syndrome", "flag"]):
            return QubitRole.ANCILLA

        # Explicit data naming patterns
        if any(p in name_lower for p in ["data", "logical", "code"]):
            return QubitRole.DATA

        # High measure/reset ratio suggests ancilla
        if self.measure_reset_ratio > 0.7:
            return QubitRole.ANCILLA

        # Short lifetime with measurements suggests ancilla
        if self.lifetime < 10 and self.measurement_count > 0:
            return QubitRole.ANCILLA

        # Default to data for long-lived qubits
        if self.lifetime > 20:
            return QubitRole.DATA

        return QubitRole.UNKNOWN


@dataclass
class QubitUsageResult:
    """Result of qubit usage analysis."""

    allocator_stats: dict[str, QubitUsageStats] = field(default_factory=dict)

    def get_allocation_recommendations(self) -> dict[str, dict]:
        """Get allocation recommendations based on usage analysis."""
        recommendations = {}

        for name, stats in self.allocator_stats.items():
            role = stats.classify_role()

            if role == QubitRole.ANCILLA:
                recommendations[name] = {
                    "allocation": "dynamic",
                    "reason": f"High measure/reset ratio ({stats.measure_reset_ratio:.2f})",
                    "keep_packed": False,
                    "pre_allocate": False,
                }
            elif role == QubitRole.DATA:
                recommendations[name] = {
                    "allocation": "static",
                    "reason": f"Low measure/reset ratio ({stats.measure_reset_ratio:.2f})",
                    "keep_packed": True,
                    "pre_allocate": True,
                }
            else:
                recommendations[name] = {
                    "allocation": "static",
                    "reason": "Unknown usage pattern",
                    "keep_packed": True,
                    "pre_allocate": True,
                }

        return recommendations


class QubitUsageAnalyzer:
    """Analyzes qubit usage patterns using recursive descent."""

    def __init__(self) -> None:
        self.stats: dict[str, QubitUsageStats] = {}
        self.position = 0

    def analyze(self, program: Program) -> QubitUsageResult:
        """Analyze qubit usage in a program.

        Args:
            program: The AST Program to analyze.

        Returns:
            QubitUsageResult with statistics for each allocator.
        """
        self.stats = {}
        self.position = 0

        # Collect allocator declarations
        for decl in program.declarations:
            if isinstance(decl, AllocatorDecl):
                self.stats[decl.name] = QubitUsageStats(
                    name=decl.name,
                    capacity=decl.capacity,
                )

        if program.allocator:
            self.stats[program.allocator.name] = QubitUsageStats(
                name=program.allocator.name,
                capacity=program.allocator.capacity,
            )

        # Analyze statements
        for stmt in program.body:
            self._analyze_statement(stmt)

        return QubitUsageResult(allocator_stats=self.stats)

    def _analyze_statement(self, stmt: Statement) -> None:
        """Analyze a single statement using recursive descent."""
        if isinstance(stmt, MeasureOp):
            self._analyze_measure(stmt)
        elif isinstance(stmt, PrepareOp):
            self._analyze_prepare(stmt)
        elif isinstance(stmt, GateOp):
            self._analyze_gate(stmt)
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
        """Analyze measurement operations."""
        for target in node.targets:
            if target.allocator in self.stats:
                stats = self.stats[target.allocator]
                stats.measurement_count += 1
                stats.measurement_positions.append(self.position)
                stats.individual_accesses.add(target.index)
                self._update_lifetime(stats)

    def _analyze_prepare(self, node: PrepareOp) -> None:
        """Analyze prepare/reset operations."""
        if node.allocator in self.stats:
            stats = self.stats[node.allocator]
            if node.slots is not None:
                stats.reset_count += len(node.slots)
                for slot in node.slots:
                    stats.individual_accesses.add(slot)
            else:
                stats.reset_count += 1
                stats.full_array_accesses += 1
            stats.reset_positions.append(self.position)
            self._update_lifetime(stats)

    def _analyze_gate(self, node: GateOp) -> None:
        """Analyze gate operations."""
        for target in node.targets:
            if target.allocator in self.stats:
                stats = self.stats[target.allocator]
                stats.gate_count += 1
                stats.individual_accesses.add(target.index)
                self._update_lifetime(stats)

    def _analyze_if(self, node: IfStmt) -> None:
        """Analyze if statement."""
        for stmt in node.then_body:
            self._analyze_statement(stmt)
        if node.else_body:
            for stmt in node.else_body:
                self._analyze_statement(stmt)

    def _analyze_while(self, node: WhileStmt) -> None:
        """Analyze while loop."""
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

    def _update_lifetime(self, stats: QubitUsageStats) -> None:
        """Update lifetime tracking for an allocator."""
        if stats.first_use_position is None:
            stats.first_use_position = self.position
        stats.last_use_position = self.position


def analyze_qubit_usage(program: Program) -> QubitUsageResult:
    """Convenience function to analyze qubit usage in an AST program.

    Args:
        program: The AST Program to analyze.

    Returns:
        QubitUsageResult with usage statistics.
    """
    analyzer = QubitUsageAnalyzer()
    return analyzer.analyze(program)
