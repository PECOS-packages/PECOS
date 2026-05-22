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
        AllocatorArg,
        AllocatorDecl,
        AllocatorTypeExpr,
        ArrayTypeExpr,
        AssignOp,
        BarrierOp,
        BinaryExpr,
        BitBundleArg,
        BitExpr,
        BitRef,
        BitTypeExpr,
        BlockCall,
        BlockDecl,
        BlockInput,
        CommentOp,
        ForStmt,
        GateOp,
        IfStmt,
        LiteralExpr,
        MeasureOp,
        ParallelBlock,
        PermuteOp,
        PrepareOp,
        PrintOp,
        Program,
        QubitBundleArg,
        QubitTypeExpr,
        RegisterDecl,
        RepeatStmt,
        ReturnOp,
        SingleBitArg,
        SingleQubitArg,
        SlotRef,
        UnaryExpr,
        VarExpr,
        WhileStmt,
    )

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)

# Node-class-name -> BaseVisitor method name. This is the single source of
# truth for visitor dispatch (replaces the per-node `accept()` double
# dispatch). Keyed by `type(node).__name__` so no runtime import of the
# node classes is needed (avoids an import cycle with nodes.py).
# `test_ast_visitor` asserts every concrete AstNode subclass is registered
# here, so a new node without an entry fails loudly in tests.
_DISPATCH: dict[str, str] = {
    "Program": "visit_program",
    "AllocatorDecl": "visit_allocator_decl",
    "RegisterDecl": "visit_register_decl",
    "GateOp": "visit_gate",
    "PrepareOp": "visit_prepare",
    "MeasureOp": "visit_measure",
    "AssignOp": "visit_assign",
    "BarrierOp": "visit_barrier",
    "CommentOp": "visit_comment",
    "ReturnOp": "visit_return",
    "PrintOp": "visit_print",
    "PermuteOp": "visit_permute",
    "IfStmt": "visit_if",
    "WhileStmt": "visit_while",
    "ForStmt": "visit_for",
    "RepeatStmt": "visit_repeat",
    "ParallelBlock": "visit_parallel",
    "BlockInput": "visit_block_input",
    "BlockDecl": "visit_block_decl",
    "BlockCall": "visit_block_call",
    "AllocatorArg": "visit_allocator_arg",
    "SingleQubitArg": "visit_single_qubit_arg",
    "SingleBitArg": "visit_single_bit_arg",
    "QubitBundleArg": "visit_qubit_bundle_arg",
    "BitBundleArg": "visit_bit_bundle_arg",
    "SlotRef": "visit_slot_ref",
    "BitRef": "visit_bit_ref",
    "LiteralExpr": "visit_literal",
    "VarExpr": "visit_var",
    "BitExpr": "visit_bit_expr",
    "BinaryExpr": "visit_binary",
    "UnaryExpr": "visit_unary",
    "QubitTypeExpr": "visit_qubit_type",
    "BitTypeExpr": "visit_bit_type",
    "ArrayTypeExpr": "visit_array_type",
    "AllocatorTypeExpr": "visit_allocator_type",
}


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

    def visit_print(self, node: PrintOp) -> T_co: ...

    # Reusable blocks
    def visit_block_decl(self, node: BlockDecl) -> T_co: ...

    def visit_block_input(self, node: BlockInput) -> T_co: ...

    def visit_block_call(self, node: BlockCall) -> T_co: ...

    # BlockCall argument bindings
    def visit_allocator_arg(self, node: AllocatorArg) -> T_co: ...

    def visit_single_qubit_arg(self, node: SingleQubitArg) -> T_co: ...

    def visit_single_bit_arg(self, node: SingleBitArg) -> T_co: ...

    def visit_qubit_bundle_arg(self, node: QubitBundleArg) -> T_co: ...

    def visit_bit_bundle_arg(self, node: BitBundleArg) -> T_co: ...

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
        """Dispatch to the appropriate visit method by node type.

        Centralized dispatch (replaces per-node `accept()` double
        dispatch): nodes carry no visitor coupling. `_DISPATCH` maps a
        node class name to its `visit_*` method; the lookup is
        late-bound via `getattr` so subclass overrides still apply.

        Resolution walks `type(node).__mro__` and uses the first
        registered ancestor. This preserves the old inherited-`accept()`
        semantics: a user subclass of a concrete node (e.g.
        `class MyGate(GateOp)`) inherited `GateOp.accept` and dispatched
        to `visit_gate`; MRO lookup reproduces that exactly.
        """
        for cls in type(node).__mro__:
            method = _DISPATCH.get(cls.__name__)
            if method is not None:
                return getattr(self, method)(node)
        msg = (
            f"BaseVisitor: no visit method registered for AST node "
            f"{type(node).__name__!r} (add it to _DISPATCH in "
            f"pecos.slr.ast.visitor)"
        )
        raise TypeError(msg)

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

    def visit_print(self, node: PrintOp) -> T:
        results = self.visit_children(node)
        return self.combine_results(results)

    # Reusable blocks

    def visit_block_decl(self, node: BlockDecl) -> T:
        results = self.visit_children(node)
        return self.combine_results(results)

    def visit_block_input(self, node: BlockInput) -> T:
        results = self.visit_children(node)
        return self.combine_results(results)

    def visit_block_call(self, node: BlockCall) -> T:
        results = self.visit_children(node)
        return self.combine_results(results)

    # BlockCall argument bindings

    def visit_allocator_arg(self, _node: AllocatorArg) -> T:
        return self.default_result()

    def visit_single_qubit_arg(self, node: SingleQubitArg) -> T:
        results = self.visit_children(node)
        return self.combine_results(results)

    def visit_single_bit_arg(self, node: SingleBitArg) -> T:
        results = self.visit_children(node)
        return self.combine_results(results)

    def visit_qubit_bundle_arg(self, node: QubitBundleArg) -> T:
        results = self.visit_children(node)
        return self.combine_results(results)

    def visit_bit_bundle_arg(self, node: BitBundleArg) -> T:
        results = self.visit_children(node)
        return self.combine_results(results)

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
