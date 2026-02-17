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

"""AST-based dependency analyzer.

This module analyzes dependencies in an AST program:
- Which allocators are used by which statements
- Which registers are used
- Variable dependencies for code generation

Example:
    from pecos.slr.ast import slr_to_ast
    from pecos.slr.ast.analysis import analyze_dependencies

    ast = slr_to_ast(program)
    result = analyze_dependencies(ast)

    print(f"Allocators used: {result.allocators_used}")
    print(f"Registers used: {result.registers_used}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pecos.slr.ast.nodes import (
    AllocatorDecl,
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
    RegisterDecl,
    RepeatStmt,
    UnaryExpr,
    VarExpr,
    WhileStmt,
)

if TYPE_CHECKING:
    from pecos.slr.ast.nodes import (
        Expression,
        Program,
        SlotRef,
        Statement,
    )


@dataclass
class DependencyResult:
    """Result of dependency analysis."""

    # Allocators and their capacities
    allocators_used: dict[str, int] = field(default_factory=dict)

    # Registers and their sizes
    registers_used: dict[str, int] = field(default_factory=dict)

    # Variables used in expressions
    variables_used: set[str] = field(default_factory=set)

    # Mapping of allocator name to list of slot indices accessed
    slot_accesses: dict[str, set[int]] = field(default_factory=dict)

    # Mapping of register name to list of bit indices accessed
    bit_accesses: dict[str, set[int]] = field(default_factory=dict)

    def get_parameter_list(self) -> list[tuple[str, str]]:
        """Get parameters needed for function signature.

        Returns:
            List of (name, type_string) tuples.
        """
        params = []

        # Add allocators as qubit arrays
        for name, capacity in sorted(self.allocators_used.items()):
            params.append((name, f"array[qubit, {capacity}]"))

        # Add registers as bool arrays
        for name, size in sorted(self.registers_used.items()):
            params.append((name, f"array[bool, {size}]"))

        return params


class DependencyAnalyzer:
    """Analyzes dependencies in an AST program using recursive descent."""

    def __init__(self) -> None:
        self.result = DependencyResult()

    def analyze(self, program: Program) -> DependencyResult:
        """Analyze dependencies in a program.

        Args:
            program: The AST Program to analyze.

        Returns:
            DependencyResult with dependency information.
        """
        self.result = DependencyResult()

        # Collect declarations
        for decl in program.declarations:
            if isinstance(decl, AllocatorDecl):
                self.result.allocators_used[decl.name] = decl.capacity
                self.result.slot_accesses[decl.name] = set()
            elif isinstance(decl, RegisterDecl):
                self.result.registers_used[decl.name] = decl.size
                self.result.bit_accesses[decl.name] = set()

        if program.allocator:
            self.result.allocators_used[program.allocator.name] = program.allocator.capacity
            self.result.slot_accesses[program.allocator.name] = set()

        # Analyze statements
        for stmt in program.body:
            self._analyze_statement(stmt)

        return self.result

    def _analyze_statement(self, stmt: Statement) -> None:
        """Analyze a single statement using recursive descent."""
        if isinstance(stmt, GateOp):
            self._analyze_gate(stmt)
        elif isinstance(stmt, MeasureOp):
            self._analyze_measure(stmt)
        elif isinstance(stmt, PrepareOp):
            self._analyze_prepare(stmt)
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

    def _analyze_gate(self, node: GateOp) -> None:
        """Analyze gate operation dependencies."""
        for target in node.targets:
            self._record_slot_access(target)

    def _analyze_measure(self, node: MeasureOp) -> None:
        """Analyze measurement dependencies."""
        for target in node.targets:
            self._record_slot_access(target)
        for result in node.results:
            self._record_bit_access(result)

    def _analyze_prepare(self, node: PrepareOp) -> None:
        """Analyze prepare dependencies."""
        if node.allocator not in self.result.allocators_used:
            # Allocator used but not declared - record it
            self.result.allocators_used[node.allocator] = 0
            self.result.slot_accesses[node.allocator] = set()

        if node.slots is not None:
            for slot in node.slots:
                self.result.slot_accesses[node.allocator].add(slot)

    def _analyze_assign(self, node: AssignOp) -> None:
        """Analyze assignment dependencies."""
        if isinstance(node.target, BitRef):
            self._record_bit_access(node.target)
        self._analyze_expression(node.value)

    def _analyze_expression(self, expr: Expression) -> None:
        """Analyze expression dependencies."""
        if isinstance(expr, VarExpr):
            self.result.variables_used.add(expr.name)
        elif isinstance(expr, BitExpr):
            self._record_bit_access(expr.ref)
        elif isinstance(expr, BinaryExpr):
            self._analyze_expression(expr.left)
            self._analyze_expression(expr.right)
        elif isinstance(expr, UnaryExpr):
            self._analyze_expression(expr.operand)

    def _analyze_if(self, node: IfStmt) -> None:
        """Analyze if statement."""
        self._analyze_expression(node.condition)
        for stmt in node.then_body:
            self._analyze_statement(stmt)
        if node.else_body:
            for stmt in node.else_body:
                self._analyze_statement(stmt)

    def _analyze_while(self, node: WhileStmt) -> None:
        """Analyze while loop."""
        self._analyze_expression(node.condition)
        for stmt in node.body:
            self._analyze_statement(stmt)

    def _analyze_for(self, node: ForStmt) -> None:
        """Analyze for loop."""
        self.result.variables_used.add(node.variable)
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

    def _record_slot_access(self, ref: SlotRef) -> None:
        """Record a slot access."""
        if ref.allocator not in self.result.allocators_used:
            self.result.allocators_used[ref.allocator] = 0
            self.result.slot_accesses[ref.allocator] = set()
        self.result.slot_accesses[ref.allocator].add(ref.index)

    def _record_bit_access(self, ref: BitRef) -> None:
        """Record a bit access."""
        if ref.register not in self.result.registers_used:
            self.result.registers_used[ref.register] = 0
            self.result.bit_accesses[ref.register] = set()
        self.result.bit_accesses[ref.register].add(ref.index)


def analyze_dependencies(program: Program) -> DependencyResult:
    """Convenience function to analyze dependencies in an AST program.

    Args:
        program: The AST Program to analyze.

    Returns:
        DependencyResult with dependency information.
    """
    analyzer = DependencyAnalyzer()
    return analyzer.analyze(program)
