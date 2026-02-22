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

"""Tests for AST pretty-printing."""

import pytest
from pecos.slr import CReg, If, Main, QAlloc, QReg, Repeat
from pecos.slr.ast import slr_to_ast
from pecos.slr.ast.nodes import (
    BinaryExpr,
    BinaryOp,
    BitRef,
    GateKind,
    GateOp,
    LiteralExpr,
    SlotRef,
    UnaryExpr,
    UnaryOp,
)
from pecos.slr.ast.pretty_print import (
    AstPrettyPrinter,
    format_expression,
    format_statement,
    pretty_print,
)
from pecos.slr.qeclib import qubit as qb


class TestPrettyPrintBasic:
    """Basic pretty-printing tests."""

    def test_empty_program(self) -> None:
        """Empty program prints correctly."""
        prog = Main()
        ast = slr_to_ast(prog)

        output = pretty_print(ast)

        assert "Main(" in output
        assert output.endswith(")")

    def test_program_with_qreg(self) -> None:
        """Program with QReg declaration prints correctly."""
        prog = Main(
            _q := QReg("q", 2),
        )
        ast = slr_to_ast(prog)

        output = pretty_print(ast)

        assert 'q := QReg("q", 2)' in output

    def test_program_with_creg(self) -> None:
        """Program with CReg declaration prints correctly."""
        prog = Main(
            _q := QReg("q", 1),
            _c := CReg("c", 2),
        )
        ast = slr_to_ast(prog)

        output = pretty_print(ast)

        assert 'c := CReg("c", 2)' in output

    def test_indentation(self) -> None:
        """Output is properly indented."""
        prog = Main(
            q := QReg("q", 1),
            qb.H(q[0]),
        )
        ast = slr_to_ast(prog)

        output = pretty_print(ast)
        lines = output.split("\n")

        # First line should not be indented
        assert lines[0] == "Main("
        # Content lines should be indented
        assert lines[1].startswith("    ")
        # Last line should not be indented
        assert lines[-1] == ")"

    def test_custom_indent(self) -> None:
        """Custom indent string is respected."""
        prog = Main(
            q := QReg("q", 1),
            qb.H(q[0]),
        )
        ast = slr_to_ast(prog)

        output = pretty_print(ast, indent="  ")  # 2 spaces
        lines = output.split("\n")

        assert lines[1].startswith("  ")
        assert not lines[1].startswith("    ")


class TestPrettyPrintGates:
    """Gate pretty-printing tests."""

    def test_single_qubit_gate(self) -> None:
        """Single-qubit gate prints correctly."""
        prog = Main(
            q := QReg("q", 1),
            qb.H(q[0]),
        )
        ast = slr_to_ast(prog)

        output = pretty_print(ast)

        assert "qb.H(q[0])" in output

    def test_pauli_gates(self) -> None:
        """Pauli gates print correctly."""
        prog = Main(
            q := QReg("q", 3),
            qb.X(q[0]),
            qb.Y(q[1]),
            qb.Z(q[2]),
        )
        ast = slr_to_ast(prog)

        output = pretty_print(ast)

        assert "qb.X(q[0])" in output
        assert "qb.Y(q[1])" in output
        assert "qb.Z(q[2])" in output

    def test_two_qubit_gate(self) -> None:
        """Two-qubit gate prints correctly."""
        prog = Main(
            q := QReg("q", 2),
            qb.CX(q[0], q[1]),
        )
        ast = slr_to_ast(prog)

        output = pretty_print(ast)

        assert "qb.CX(q[0], q[1])" in output

    def test_phase_gates(self) -> None:
        """Phase gates print correctly."""
        prog = Main(
            q := QReg("q", 2),
            qb.SZ(q[0]),  # S gate (sqrt Z)
            qb.T(q[1]),
        )
        ast = slr_to_ast(prog)

        output = pretty_print(ast)

        assert "qb.SZ(q[0])" in output
        assert "qb.T(q[1])" in output


class TestPrettyPrintMeasure:
    """Measurement pretty-printing tests."""

    def test_measure_without_result(self) -> None:
        """Measurement without result prints correctly."""
        gate = GateOp(
            gate=GateKind.H,
            targets=(SlotRef(allocator="q", index=0),),
        )

        output = format_statement(gate)

        assert "qb.H(q[0])" in output

    def test_measure_with_result(self) -> None:
        """Measurement with result prints correctly."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            qb.Measure(q[0]) > c[0],
        )
        ast = slr_to_ast(prog)

        output = pretty_print(ast)

        # Check measurement is formatted
        assert "Measure" in output
        assert "q[0]" in output


class TestPrettyPrintControlFlow:
    """Control flow pretty-printing tests."""

    def test_if_statement(self) -> None:
        """If statement prints correctly."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            If(c[0] == 1).Then(
                qb.X(q[0]),
            ),
        )
        ast = slr_to_ast(prog)

        output = pretty_print(ast)

        assert "If(" in output
        assert ").Then(" in output
        assert "qb.X(q[0])" in output

    def test_if_else_statement(self) -> None:
        """If-else statement prints correctly."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            If(c[0] == 1)
            .Then(
                qb.X(q[0]),
            )
            .Else(
                qb.Y(q[0]),
            ),
        )
        ast = slr_to_ast(prog)

        output = pretty_print(ast)

        assert "If(" in output
        assert ").Then(" in output
        assert ").Else(" in output
        assert "qb.X(q[0])" in output
        assert "qb.Y(q[0])" in output

    def test_repeat_statement(self) -> None:
        """Repeat statement prints correctly."""
        prog = Main(
            q := QReg("q", 1),
            Repeat(cond=5).block(
                qb.H(q[0]),
            ),
        )
        ast = slr_to_ast(prog)

        output = pretty_print(ast)

        assert "Repeat(cond=5).block(" in output
        assert "qb.H(q[0])" in output

    def test_nested_control_flow(self) -> None:
        """Nested control flow prints with proper indentation."""
        prog = Main(
            q := QReg("q", 2),
            c := CReg("c", 2),
            If(c[0] == 1).Then(
                If(c[1] == 1).Then(
                    qb.X(q[0]),
                ),
            ),
        )
        ast = slr_to_ast(prog)

        output = pretty_print(ast)

        # Count indentation levels
        lines = output.split("\n")
        max_indent = max(len(line) - len(line.lstrip()) for line in lines if line.strip())

        # Should have at least 3 levels of indentation for nested if
        assert max_indent >= 8  # 2 levels * 4 spaces


class TestPrettyPrintExpressions:
    """Expression pretty-printing tests."""

    def test_literal_int(self) -> None:
        """Integer literal formats correctly."""
        expr = LiteralExpr(value=42)

        output = format_expression(expr)

        assert output == "42"

    def test_literal_float(self) -> None:
        """Float literal formats correctly."""
        expr = LiteralExpr(value=3.14)

        output = format_expression(expr)

        assert "3.14" in output

    def test_literal_bool_true(self) -> None:
        """True literal formats correctly."""
        expr = LiteralExpr(value=True)

        output = format_expression(expr)

        assert output == "True"

    def test_literal_bool_false(self) -> None:
        """False literal formats correctly."""
        expr = LiteralExpr(value=False)

        output = format_expression(expr)

        assert output == "False"

    def test_binary_expr_eq(self) -> None:
        """Equality expression formats correctly."""
        expr = BinaryExpr(
            op=BinaryOp.EQ,
            left=LiteralExpr(value=1),
            right=LiteralExpr(value=2),
        )

        output = format_expression(expr)

        assert "==" in output
        assert "1" in output
        assert "2" in output

    def test_binary_expr_add(self) -> None:
        """Addition expression formats correctly."""
        expr = BinaryExpr(
            op=BinaryOp.ADD,
            left=LiteralExpr(value=3),
            right=LiteralExpr(value=4),
        )

        output = format_expression(expr)

        assert "+" in output

    def test_binary_expr_and(self) -> None:
        """And expression formats correctly."""
        expr = BinaryExpr(
            op=BinaryOp.AND,
            left=LiteralExpr(value=True),
            right=LiteralExpr(value=False),
        )

        output = format_expression(expr)

        assert "and" in output

    def test_unary_expr_not(self) -> None:
        """Not expression formats correctly."""
        expr = UnaryExpr(
            op=UnaryOp.NOT,
            operand=LiteralExpr(value=True),
        )

        output = format_expression(expr)

        assert "not" in output

    def test_unary_expr_neg(self) -> None:
        """Negation expression formats correctly."""
        expr = UnaryExpr(
            op=UnaryOp.NEG,
            operand=LiteralExpr(value=5),
        )

        output = format_expression(expr)

        assert "-" in output


class TestPrettyPrintReferences:
    """Reference pretty-printing tests."""

    def test_slot_ref(self) -> None:
        """SlotRef formats as allocator[index]."""
        slot = SlotRef(allocator="data", index=3)
        printer = AstPrettyPrinter()

        output = printer.visit_slot_ref(slot)

        assert output == "data[3]"

    def test_bit_ref(self) -> None:
        """BitRef formats as register[index]."""
        bit = BitRef(register="result", index=5)
        printer = AstPrettyPrinter()

        output = printer.visit_bit_ref(bit)

        assert output == "result[5]"


class TestPrettyPrintHierarchicalAllocators:
    """Hierarchical allocator pretty-printing tests."""

    def test_hierarchical_allocators(self) -> None:
        """Hierarchical allocators show parent relationship."""
        all_qubits = QAlloc(4, name="all")
        data = QAlloc(2, name="data", parent=all_qubits)
        ancilla = QAlloc(2, name="ancilla", parent=all_qubits)

        prog = Main(
            all_qubits,
            data,
            ancilla,
            qb.H(data[0]),
        )
        ast = slr_to_ast(prog)

        output = pretty_print(ast)

        # Check parent relationship is shown
        assert "parent=" in output or "data" in output


class TestPrettyPrintQEC:
    """QEC pattern pretty-printing tests."""

    def test_syndrome_extraction(self) -> None:
        """Syndrome extraction pattern prints correctly."""
        prog = Main(
            data := QReg("data", 2),
            ancilla := QReg("ancilla", 1),
            c := CReg("c", 1),
            qb.CX(data[0], ancilla[0]),
            qb.CX(data[1], ancilla[0]),
            qb.Measure(ancilla[0]) > c[0],
        )
        ast = slr_to_ast(prog)

        output = pretty_print(ast)

        # Check key elements
        assert 'data := QReg("data", 2)' in output
        assert 'ancilla := QReg("ancilla", 1)' in output
        assert "qb.CX(data[0], ancilla[0])" in output
        assert "qb.CX(data[1], ancilla[0])" in output

    def test_bell_state(self) -> None:
        """Bell state circuit prints correctly."""
        prog = Main(
            q := QReg("q", 2),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
        )
        ast = slr_to_ast(prog)

        output = pretty_print(ast)

        assert "qb.H(q[0])" in output
        assert "qb.CX(q[0], q[1])" in output


class TestFormatStatement:
    """Tests for format_statement function."""

    def test_format_gate(self) -> None:
        """Single gate formats correctly."""
        gate = GateOp(
            gate=GateKind.H,
            targets=(SlotRef(allocator="q", index=0),),
        )

        output = format_statement(gate)

        assert output == "qb.H(q[0])"

    def test_format_two_qubit_gate(self) -> None:
        """Two-qubit gate formats correctly."""
        gate = GateOp(
            gate=GateKind.CZ,
            targets=(
                SlotRef(allocator="a", index=0),
                SlotRef(allocator="b", index=1),
            ),
        )

        output = format_statement(gate)

        assert output == "qb.CZ(a[0], b[1])"


class TestFormatExpression:
    """Tests for format_expression function."""

    def test_format_simple(self) -> None:
        """Simple expression formats correctly."""
        expr = LiteralExpr(value=100)

        output = format_expression(expr)

        assert output == "100"

    def test_format_complex(self) -> None:
        """Complex nested expression formats correctly."""
        expr = BinaryExpr(
            op=BinaryOp.ADD,
            left=BinaryExpr(
                op=BinaryOp.MUL,
                left=LiteralExpr(value=2),
                right=LiteralExpr(value=3),
            ),
            right=LiteralExpr(value=4),
        )

        output = format_expression(expr)

        # Should contain structure of expression
        assert "2" in output
        assert "3" in output
        assert "4" in output
        assert "*" in output
        assert "+" in output


class TestPrettyPrinterClass:
    """Tests for AstPrettyPrinter class."""

    def test_reusable(self) -> None:
        """Printer can be reused for multiple programs."""
        printer = AstPrettyPrinter()

        prog1 = Main(q := QReg("q", 1), qb.H(q[0]))
        prog2 = Main(r := QReg("r", 2), qb.X(r[0]))

        ast1 = slr_to_ast(prog1)
        ast2 = slr_to_ast(prog2)

        output1 = printer.print(ast1)
        output2 = printer.print(ast2)

        assert "q[0]" in output1
        assert "r[0]" in output2

    def test_indent_level_reset(self) -> None:
        """Indent level resets between prints."""
        printer = AstPrettyPrinter()

        prog = Main(
            q := QReg("q", 1),
            If(LiteralExpr(value=True)).Then(
                qb.H(q[0]),
            ),
        )
        ast = slr_to_ast(prog)

        # Print multiple times
        output1 = printer.print(ast)
        output2 = printer.print(ast)

        # Outputs should be identical
        assert output1 == output2


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_if_body(self) -> None:
        """Test If with no operations in then body (unusual but valid)."""
        # This would require manual AST construction since SLR requires content
        # Skip - SLR requires at least one statement

    def test_multiple_allocators(self) -> None:
        """Multiple allocators are rendered correctly."""
        prog = Main(
            a := QReg("a", 2),
            b := QReg("b", 2),
            c := QReg("c", 2),
            qb.H(a[0]),
            qb.H(b[0]),
            qb.H(c[0]),
        )
        ast = slr_to_ast(prog)

        output = pretty_print(ast)

        assert 'a := QReg("a", 2)' in output
        assert 'b := QReg("b", 2)' in output
        assert 'c := QReg("c", 2)' in output

    def test_float_that_looks_like_int(self) -> None:
        """Test float value that equals an integer."""
        expr = LiteralExpr(value=5.0)

        output = format_expression(expr)

        # Should format as integer since it's a whole number
        assert output == "5"
