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

"""AST pretty-printing to SLR-like syntax.

This module provides functions to convert AST nodes back to human-readable
SLR-like syntax for debugging and display.

Example:
    >>> from pecos.slr import Main, QReg
    >>> from pecos.slr.qeclib import qubit as qb
    >>> from pecos.slr.ast import slr_to_ast
    >>> from pecos.slr.ast.pretty_print import pretty_print
    >>>
    >>> prog = Main(q := QReg("q", 2), qb.H(q[0]), qb.CX(q[0], q[1]))
    >>> ast = slr_to_ast(prog)
    >>> print(pretty_print(ast))
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pecos.slr.ast.nodes import (
    AllocatorDecl,
    BinaryOp,
    BitRef,
    Expression,
    RegisterDecl,
    UnaryOp,
)
from pecos.slr.ast.visitor import BaseVisitor

if TYPE_CHECKING:
    from typing_extensions import Self

    from pecos.slr.ast.nodes import (
        AssignOp,
        BarrierOp,
        BinaryExpr,
        BitExpr,
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
        RepeatStmt,
        ReturnOp,
        SlotRef,
        Statement,
        UnaryExpr,
        VarExpr,
        WhileStmt,
    )

# Operator symbols for pretty-printing
_BINARY_OP_SYMBOLS: dict[BinaryOp, str] = {
    BinaryOp.ADD: "+",
    BinaryOp.SUB: "-",
    BinaryOp.MUL: "*",
    BinaryOp.DIV: "/",
    BinaryOp.EQ: "==",
    BinaryOp.NE: "!=",
    BinaryOp.LT: "<",
    BinaryOp.LE: "<=",
    BinaryOp.GT: ">",
    BinaryOp.GE: ">=",
    BinaryOp.AND: "and",
    BinaryOp.OR: "or",
    BinaryOp.XOR: "^",
    BinaryOp.LSHIFT: "<<",
    BinaryOp.RSHIFT: ">>",
}

_UNARY_OP_SYMBOLS: dict[UnaryOp, str] = {
    UnaryOp.NOT: "not ",
    UnaryOp.NEG: "-",
}


class AstPrettyPrinter(BaseVisitor[str]):
    """Pretty-print AST nodes to SLR-like syntax."""

    def __init__(self, *, indent: str = "    "):
        """Initialize printer.

        Args:
            indent: String to use for indentation (default: 4 spaces).
        """
        self._indent = indent
        self._level = 0

    def print(self, program: Program) -> str:
        """Print a program to SLR-like syntax.

        Args:
            program: The AST program to print.

        Returns:
            String representation in SLR-like syntax.
        """
        return self.visit_program(program)

    def _indented(self, line: str) -> str:
        """Apply current indentation level to a line."""
        return f"{self._indent * self._level}{line}"

    def increment_level(self) -> None:
        """Increment indentation level."""
        self._level += 1

    def decrement_level(self) -> None:
        """Decrement indentation level."""
        self._level -= 1

    def _with_indent(self) -> _IndentContext:
        """Context manager to increase indent level."""
        return _IndentContext(self)

    # Program and declarations

    def visit_program(self, node: Program) -> str:
        """Visit program node."""
        lines = ["Main("]
        self._level += 1

        # Allocator
        if node.allocator:
            lines.append(
                self._indented(
                    f'{node.allocator.name} := QReg("{node.allocator.name}", {node.allocator.capacity}),',
                ),
            )

        # Additional declarations
        for decl in node.declarations:
            if isinstance(decl, AllocatorDecl):
                if decl.parent:
                    lines.append(
                        self._indented(
                            f'{decl.name} := QReg("{decl.name}", {decl.capacity}, parent={decl.parent}),',
                        ),
                    )
                else:
                    lines.append(
                        self._indented(
                            f'{decl.name} := QReg("{decl.name}", {decl.capacity}),',
                        ),
                    )
            elif isinstance(decl, RegisterDecl):
                lines.append(
                    self._indented(f'{decl.name} := CReg("{decl.name}", {decl.size}),'),
                )

        # Body statements
        lines.extend(self._indented(f"{self.format_statement(stmt)},") for stmt in node.body)

        self._level -= 1
        lines.append(")")
        return "\n".join(lines)

    def visit_allocator_decl(self, node: AllocatorDecl) -> str:
        """Visit allocator declaration."""
        if node.parent:
            return f'{node.name} := QReg("{node.name}", {node.capacity}, parent={node.parent})'
        return f'{node.name} := QReg("{node.name}", {node.capacity})'

    def visit_register_decl(self, node: RegisterDecl) -> str:
        """Visit register declaration."""
        return f'{node.name} := CReg("{node.name}", {node.size})'

    # Statements

    def format_statement(self, stmt: Statement) -> str:
        """Format a statement."""
        return stmt.accept(self)

    def visit_gate(self, node: GateOp) -> str:
        """Visit gate operation."""
        gate_name = node.gate.name
        targets = ", ".join(self.visit_slot_ref(t) for t in node.targets)

        if node.params:
            params = ", ".join(self.format_expression(p) for p in node.params)
            return f"qb.{gate_name}[{params}]({targets})"
        return f"qb.{gate_name}({targets})"

    def visit_prepare(self, node: PrepareOp) -> str:
        """Visit prepare operation."""
        if node.slots is None:
            return f"{node.allocator}.prepare_all()"
        if len(node.slots) == 1:
            return f"{node.allocator}.prepare({node.slots[0]})"
        slots = ", ".join(str(s) for s in node.slots)
        return f"{node.allocator}.prepare({slots})"

    def visit_measure(self, node: MeasureOp) -> str:
        """Visit measure operation."""
        targets = ", ".join(self.visit_slot_ref(t) for t in node.targets)
        if node.results:
            results = ", ".join(self.visit_bit_ref(r) for r in node.results)
            return f"Measure({targets}) >> ({results})"
        return f"Measure({targets})"

    def visit_assign(self, node: AssignOp) -> str:
        """Visit assignment."""
        target = self.visit_bit_ref(node.target) if isinstance(node.target, BitRef) else node.target
        value = self.format_expression(node.value)
        return f"{target} = {value}"

    def visit_barrier(self, node: BarrierOp) -> str:
        """Visit barrier."""
        if node.allocators:
            allocs = ", ".join(node.allocators)
            return f"Barrier({allocs})"
        return "Barrier()"

    def visit_comment(self, node: CommentOp) -> str:
        """Visit comment."""
        return f"# {node.text}"

    def visit_return(self, node: ReturnOp) -> str:
        """Visit return."""
        if node.values:
            vals = ", ".join(self.format_expression(v) if isinstance(v, Expression) else str(v) for v in node.values)
            return f"Return({vals})"
        return "Return()"

    def visit_permute(self, node: PermuteOp) -> str:
        """Visit permute."""
        sources = ", ".join(node.sources)
        targets = ", ".join(node.targets)
        return f"Permute([{sources}], [{targets}])"

    # Control flow

    def visit_if(self, node: IfStmt) -> str:
        """Visit if statement."""
        cond = self.format_expression(node.condition)
        lines = [f"If({cond}).Then("]

        self._level += 1
        lines.extend(self._indented(f"{self.format_statement(stmt)},") for stmt in node.then_body)
        self._level -= 1

        if node.else_body:
            lines.append(").Else(")
            self._level += 1
            lines.extend(self._indented(f"{self.format_statement(stmt)},") for stmt in node.else_body)
            self._level -= 1

        lines.append(")")
        return "\n".join(lines)

    def visit_while(self, node: WhileStmt) -> str:
        """Visit while statement."""
        cond = self.format_expression(node.condition)
        lines = [f"While({cond}).block("]

        self._level += 1
        lines.extend(self._indented(f"{self.format_statement(stmt)},") for stmt in node.body)
        self._level -= 1

        lines.append(")")
        return "\n".join(lines)

    def visit_for(self, node: ForStmt) -> str:
        """Visit for statement."""
        start = self.format_expression(node.start)
        stop = self.format_expression(node.stop)
        if node.step:
            step = self.format_expression(node.step)
            lines = [f"For({node.variable}, {start}, {stop}, {step}).block("]
        else:
            lines = [f"For({node.variable}, {start}, {stop}).block("]

        self._level += 1
        lines.extend(self._indented(f"{self.format_statement(stmt)},") for stmt in node.body)
        self._level -= 1

        lines.append(")")
        return "\n".join(lines)

    def visit_repeat(self, node: RepeatStmt) -> str:
        """Visit repeat statement."""
        lines = [f"Repeat(cond={node.count}).block("]

        self._level += 1
        lines.extend(self._indented(f"{self.format_statement(stmt)},") for stmt in node.body)
        self._level -= 1

        lines.append(")")
        return "\n".join(lines)

    def visit_parallel(self, node: ParallelBlock) -> str:
        """Visit parallel block."""
        lines = ["Parallel("]

        self._level += 1
        lines.extend(self._indented(f"{self.format_statement(stmt)},") for stmt in node.body)
        self._level -= 1

        lines.append(")")
        return "\n".join(lines)

    # References

    def visit_slot_ref(self, node: SlotRef) -> str:
        """Visit slot reference."""
        return f"{node.allocator}[{node.index}]"

    def visit_bit_ref(self, node: BitRef) -> str:
        """Visit bit reference."""
        return f"{node.register}[{node.index}]"

    # Expressions

    def format_expression(self, expr: Expression) -> str:
        """Format an expression."""
        return expr.accept(self)

    def visit_literal(self, node: LiteralExpr) -> str:
        """Visit literal expression."""
        if isinstance(node.value, bool):
            return "True" if node.value else "False"
        if isinstance(node.value, float):
            # Format float nicely
            if node.value == int(node.value):
                return str(int(node.value))
            return str(node.value)
        return str(node.value)

    def visit_var(self, node: VarExpr) -> str:
        """Visit variable expression."""
        return node.name

    def visit_bit_expr(self, node: BitExpr) -> str:
        """Visit bit expression."""
        return self.visit_bit_ref(node.ref)

    def visit_binary(self, node: BinaryExpr) -> str:
        """Visit binary expression."""
        left = self.format_expression(node.left)
        right = self.format_expression(node.right)
        op = _BINARY_OP_SYMBOLS.get(node.op, str(node.op))
        return f"({left} {op} {right})"

    def visit_unary(self, node: UnaryExpr) -> str:
        """Visit unary expression."""
        operand = self.format_expression(node.operand)
        op = _UNARY_OP_SYMBOLS.get(node.op, str(node.op))
        return f"{op}{operand}"

    # Type expressions (not commonly used in output)

    def visit_qubit_type(self, _node: object) -> str:
        """Visit qubit type."""
        return "Qubit"

    def visit_bit_type(self, _node: object) -> str:
        """Visit bit type."""
        return "Bit"

    def visit_array_type(self, node) -> str:
        """Visit array type."""
        element = node.element.accept(self)
        return f"Array[{element}, {node.size}]"

    def visit_allocator_type(self, node) -> str:
        """Visit allocator type."""
        return f"Allocator[{node.capacity}]"


class _IndentContext:
    """Context manager for indentation.

    This is an internal helper that accesses the printer's private methods.
    """

    def __init__(self, printer: AstPrettyPrinter):
        self._printer = printer

    def __enter__(self) -> Self:
        self._printer.increment_level()
        return self

    def __exit__(self, *_args: object) -> None:
        self._printer.decrement_level()


def pretty_print(program: Program, *, indent: str = "    ") -> str:
    """Pretty-print an AST Program to SLR-like syntax.

    Args:
        program: The AST Program to print.
        indent: String to use for indentation (default: 4 spaces).

    Returns:
        String representation in SLR-like syntax.

    Example:
        >>> from pecos.slr.ast.pretty_print import pretty_print
        >>> print(pretty_print(ast))
        Main(
            q := QReg("q", 2),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
        )
    """
    printer = AstPrettyPrinter(indent=indent)
    return printer.print(program)


def format_statement(stmt: Statement, *, indent: str = "    ") -> str:
    """Format a single statement to SLR-like syntax.

    Args:
        stmt: The statement to format.
        indent: String to use for indentation.

    Returns:
        String representation of the statement.
    """
    printer = AstPrettyPrinter(indent=indent)
    return printer.format_statement(stmt)


def format_expression(expr: Expression) -> str:
    """Format an expression to SLR-like syntax.

    Args:
        expr: The expression to format.

    Returns:
        String representation of the expression.
    """
    printer = AstPrettyPrinter()
    return printer.format_expression(expr)
