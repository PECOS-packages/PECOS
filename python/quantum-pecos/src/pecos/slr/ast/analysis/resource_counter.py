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

"""AST resource counting analysis.

This module counts quantum resources used in an AST program:
- Number of qubits (allocator slots used)
- Number of classical bits
- Gate counts by type
- Measurement count
- Preparation count

Example:
    from pecos.slr.ast import slr_to_ast
    from pecos.slr.ast.analysis import count_resources

    ast = slr_to_ast(program)
    resources = count_resources(ast)

    print(f"Qubits: {resources.qubit_count}")
    print(f"Gates: {resources.total_gates}")
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pecos.slr.ast.nodes import (
    AllocatorDecl,
    ForStmt,
    GateOp,
    IfStmt,
    MeasureOp,
    ParallelBlock,
    PrepareOp,
    RegisterDecl,
    RepeatStmt,
    WhileStmt,
)

if TYPE_CHECKING:
    from pecos.slr.ast.nodes import (
        GateKind,
        Program,
        Statement,
    )


@dataclass
class ResourceCount:
    """Resource counts for an AST program."""

    # Qubit resources
    qubit_count: int = 0
    qubits_by_allocator: dict[str, int] = field(default_factory=dict)

    # Classical resources
    classical_bit_count: int = 0
    bits_by_register: dict[str, int] = field(default_factory=dict)

    # Gate counts
    gate_counts: Counter[GateKind] = field(default_factory=Counter)

    # Operation counts
    measurement_count: int = 0
    preparation_count: int = 0

    @property
    def total_gates(self) -> int:
        """Total number of gate operations."""
        return sum(self.gate_counts.values())

    @property
    def single_qubit_gates(self) -> int:
        """Count of single-qubit gates."""
        return sum(count for gate, count in self.gate_counts.items() if gate.arity == 1)

    @property
    def two_qubit_gates(self) -> int:
        """Count of two-qubit gates."""
        return sum(count for gate, count in self.gate_counts.items() if gate.arity == 2)

    def __str__(self) -> str:
        lines = [
            f"Qubits: {self.qubit_count}",
            f"Classical bits: {self.classical_bit_count}",
            f"Total gates: {self.total_gates}",
            f"  Single-qubit: {self.single_qubit_gates}",
            f"  Two-qubit: {self.two_qubit_gates}",
            f"Measurements: {self.measurement_count}",
            f"Preparations: {self.preparation_count}",
        ]
        return "\n".join(lines)


class ResourceCounter:
    """Counts quantum resources in an AST program using recursive descent."""

    def __init__(self) -> None:
        self.result = ResourceCount()

    def count(self, program: Program) -> ResourceCount:
        """Count resources in a program.

        Args:
            program: The AST Program to analyze.

        Returns:
            ResourceCount with all resource counts.
        """
        self.result = ResourceCount()

        # Count declarations
        for decl in program.declarations:
            self._count_declaration(decl)

        if program.allocator:
            self._count_allocator(program.allocator)

        # Count operations in body
        for stmt in program.body:
            self._count_statement(stmt)

        return self.result

    def _count_declaration(self, decl: AllocatorDecl | RegisterDecl) -> None:
        """Count resources from a declaration."""
        if isinstance(decl, AllocatorDecl):
            self._count_allocator(decl)
        elif isinstance(decl, RegisterDecl):
            self._count_register(decl)

    def _count_allocator(self, decl: AllocatorDecl) -> None:
        """Count qubits from an allocator declaration."""
        self.result.qubit_count += decl.capacity
        self.result.qubits_by_allocator[decl.name] = decl.capacity

    def _count_register(self, decl: RegisterDecl) -> None:
        """Count classical bits from a register declaration."""
        self.result.classical_bit_count += decl.size
        self.result.bits_by_register[decl.name] = decl.size

    def _count_statement(self, stmt: Statement) -> None:
        """Count resources in a statement using recursive descent."""
        if isinstance(stmt, GateOp):
            self._count_gate(stmt)
        elif isinstance(stmt, MeasureOp):
            self._count_measure(stmt)
        elif isinstance(stmt, PrepareOp):
            self._count_prepare(stmt)
        elif isinstance(stmt, IfStmt):
            self._count_if(stmt)
        elif isinstance(stmt, WhileStmt):
            self._count_while(stmt)
        elif isinstance(stmt, ForStmt):
            self._count_for(stmt)
        elif isinstance(stmt, RepeatStmt):
            self._count_repeat(stmt)
        elif isinstance(stmt, ParallelBlock):
            self._count_parallel(stmt)
        # Other statements don't contribute to resource counts

    def _count_gate(self, node: GateOp) -> None:
        """Count a gate operation."""
        self.result.gate_counts[node.gate] += 1

    def _count_measure(self, node: MeasureOp) -> None:
        """Count a measurement operation."""
        self.result.measurement_count += len(node.targets)

    def _count_prepare(self, node: PrepareOp) -> None:
        """Count a preparation operation."""
        if node.slots is not None:
            self.result.preparation_count += len(node.slots)
        else:
            # PZ all - would need allocator info to count exactly
            self.result.preparation_count += 1

    def _count_if(self, node: IfStmt) -> None:
        """Count resources in an if statement.

        Note: Counts both branches since either could execute.
        """
        for stmt in node.then_body:
            self._count_statement(stmt)
        if node.else_body:
            for stmt in node.else_body:
                self._count_statement(stmt)

    def _count_while(self, node: WhileStmt) -> None:
        """Count resources in a while loop.

        Note: Only counts one iteration.
        """
        for stmt in node.body:
            self._count_statement(stmt)

    def _count_for(self, node: ForStmt) -> None:
        """Count resources in a for loop.

        Note: Only counts one iteration.
        """
        for stmt in node.body:
            self._count_statement(stmt)

    def _count_repeat(self, node: RepeatStmt) -> None:
        """Count resources in a repeat loop.

        Note: Multiplies by repeat count for accurate resource estimation.
        """
        # Count body once
        body_result = ResourceCount()
        temp_counter = ResourceCounter()
        temp_counter.result = body_result

        for stmt in node.body:
            temp_counter._count_statement(stmt)

        # Multiply by repeat count
        for gate, count in body_result.gate_counts.items():
            self.result.gate_counts[gate] += count * node.count
        self.result.measurement_count += body_result.measurement_count * node.count
        self.result.preparation_count += body_result.preparation_count * node.count

    def _count_parallel(self, node: ParallelBlock) -> None:
        """Count resources in a parallel block."""
        for stmt in node.body:
            self._count_statement(stmt)


def count_resources(program: Program) -> ResourceCount:
    """Convenience function to count resources in an AST program.

    Args:
        program: The AST Program to analyze.

    Returns:
        ResourceCount with all resource counts.
    """
    counter = ResourceCounter()
    return counter.count(program)
