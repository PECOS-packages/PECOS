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

"""Visitor pattern support for SLR AST.

This module provides:
- AstVisitor protocol defining the visitor interface
- BaseVisitor with default traversal behavior
- Utility functions for common traversal patterns
"""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar

if TYPE_CHECKING:
    from pecos.slr.ast.nodes import (
        AllocatorDecl,
        AllocatorTypeExpr,
        ArrayTypeExpr,
        AssignOp,
        BarrierOp,
        BinaryExpr,
        BitExpr,
        BitRef,
        BitTypeExpr,
        CommentOp,
        ForStmt,
        GateOp,
        IfStmt,
        LiteralExpr,
        MeasureOp,
        ParallelBlock,
        PermuteOp,
        PrepareOp,
        Program,
        QubitTypeExpr,
        RegisterDecl,
        RepeatStmt,
        ReturnOp,
        SlotRef,
        UnaryExpr,
        VarExpr,
        WhileStmt,
    )

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)


class AstVisitor(Protocol[T_co]):
    """Protocol defining the visitor interface for AST nodes.

    Implement this protocol to create custom AST visitors for
    analysis, transformation, or code generation.
    """

    # Program structure
    def visit_program(self, node: Program) -> T_co: ...

    def visit_allocator_decl(self, node: AllocatorDecl) -> T_co: ...

    def visit_register_decl(self, node: RegisterDecl) -> T_co: ...

    # Statements
    def visit_gate(self, node: GateOp) -> T_co: ...

    def visit_prepare(self, node: PrepareOp) -> T_co: ...

    def visit_measure(self, node: MeasureOp) -> T_co: ...

    def visit_assign(self, node: AssignOp) -> T_co: ...

    def visit_barrier(self, node: BarrierOp) -> T_co: ...

    def visit_comment(self, node: CommentOp) -> T_co: ...

    def visit_return(self, node: ReturnOp) -> T_co: ...

    def visit_permute(self, node: PermuteOp) -> T_co: ...

    # Control flow
    def visit_if(self, node: IfStmt) -> T_co: ...

    def visit_while(self, node: WhileStmt) -> T_co: ...

    def visit_for(self, node: ForStmt) -> T_co: ...

    def visit_repeat(self, node: RepeatStmt) -> T_co: ...

    def visit_parallel(self, node: ParallelBlock) -> T_co: ...

    # References
    def visit_slot_ref(self, node: SlotRef) -> T_co: ...

    def visit_bit_ref(self, node: BitRef) -> T_co: ...

    # Expressions
    def visit_literal(self, node: LiteralExpr) -> T_co: ...

    def visit_var(self, node: VarExpr) -> T_co: ...

    def visit_bit_expr(self, node: BitExpr) -> T_co: ...

    def visit_binary(self, node: BinaryExpr) -> T_co: ...

    def visit_unary(self, node: UnaryExpr) -> T_co: ...

    # Types
    def visit_qubit_type(self, node: QubitTypeExpr) -> T_co: ...

    def visit_bit_type(self, node: BitTypeExpr) -> T_co: ...

    def visit_array_type(self, node: ArrayTypeExpr) -> T_co: ...

    def visit_allocator_type(self, node: AllocatorTypeExpr) -> T_co: ...


class BaseVisitor(ABC, Generic[T]):
    """Base visitor with default traversal behavior.

    Provides default implementations that visit all children.
    Override specific visit methods to customize behavior.

    Usage:
        class MyVisitor(BaseVisitor[str]):
            def visit_gate(self, node: GateOp) -> str:
                return f"Gate: {node.gate.name}"

            def default_result(self) -> str:
                return ""

            def combine_results(self, results: list[str]) -> str:
                return "\\n".join(results)
    """

    def visit(self, node) -> T:
        """Dispatch to the appropriate visit method."""
        return node.accept(self)

    def visit_children(self, node) -> list[T]:
        """Visit all children and collect results."""
        return [self.visit(child) for child in node.children()]

    def default_result(self) -> T:
        """Default result when no specific handling. Must override."""
        msg = "Subclasses must implement default_result()"
        raise NotImplementedError(msg)

    def combine_results(self, results: list[T]) -> T:
        """Combine multiple child results. Default returns last or default."""
        return results[-1] if results else self.default_result()

    # Program structure

    def visit_program(self, node: Program) -> T:
        results = self.visit_children(node)
        return self.combine_results(results)

    def visit_allocator_decl(self, _node: AllocatorDecl) -> T:
        return self.default_result()

    def visit_register_decl(self, _node: RegisterDecl) -> T:
        return self.default_result()

    # Statements

    def visit_gate(self, node: GateOp) -> T:
        results = self.visit_children(node)
        return self.combine_results(results)

    def visit_prepare(self, _node: PrepareOp) -> T:
        return self.default_result()

    def visit_measure(self, node: MeasureOp) -> T:
        results = self.visit_children(node)
        return self.combine_results(results)

    def visit_assign(self, node: AssignOp) -> T:
        results = self.visit_children(node)
        return self.combine_results(results)

    def visit_barrier(self, _node: BarrierOp) -> T:
        return self.default_result()

    def visit_comment(self, _node: CommentOp) -> T:
        return self.default_result()

    def visit_return(self, node: ReturnOp) -> T:
        results = self.visit_children(node)
        return self.combine_results(results)

    def visit_permute(self, _node: PermuteOp) -> T:
        return self.default_result()

    # Control flow

    def visit_if(self, node: IfStmt) -> T:
        results = self.visit_children(node)
        return self.combine_results(results)

    def visit_while(self, node: WhileStmt) -> T:
        results = self.visit_children(node)
        return self.combine_results(results)

    def visit_for(self, node: ForStmt) -> T:
        results = self.visit_children(node)
        return self.combine_results(results)

    def visit_repeat(self, node: RepeatStmt) -> T:
        results = self.visit_children(node)
        return self.combine_results(results)

    def visit_parallel(self, node: ParallelBlock) -> T:
        results = self.visit_children(node)
        return self.combine_results(results)

    # References

    def visit_slot_ref(self, _node: SlotRef) -> T:
        return self.default_result()

    def visit_bit_ref(self, _node: BitRef) -> T:
        return self.default_result()

    # Expressions

    def visit_literal(self, _node: LiteralExpr) -> T:
        return self.default_result()

    def visit_var(self, _node: VarExpr) -> T:
        return self.default_result()

    def visit_bit_expr(self, node: BitExpr) -> T:
        results = self.visit_children(node)
        return self.combine_results(results)

    def visit_binary(self, node: BinaryExpr) -> T:
        results = self.visit_children(node)
        return self.combine_results(results)

    def visit_unary(self, node: UnaryExpr) -> T:
        results = self.visit_children(node)
        return self.combine_results(results)

    # Types

    def visit_qubit_type(self, _node: QubitTypeExpr) -> T:
        return self.default_result()

    def visit_bit_type(self, _node: BitTypeExpr) -> T:
        return self.default_result()

    def visit_array_type(self, node: ArrayTypeExpr) -> T:
        results = self.visit_children(node)
        return self.combine_results(results)

    def visit_allocator_type(self, _node: AllocatorTypeExpr) -> T:
        return self.default_result()


class VoidVisitor(BaseVisitor[None]):
    """Visitor that returns None (for side-effect only traversal)."""

    def default_result(self) -> None:
        return None

    def combine_results(self, _results: list[None]) -> None:
        return None


class CollectingVisitor(BaseVisitor[list[T]], Generic[T]):
    """Visitor that collects items into a list."""

    def default_result(self) -> list[T]:
        return []

    def combine_results(self, results: list[list[T]]) -> list[T]:
        combined: list[T] = []
        for r in results:
            combined.extend(r)
        return combined
