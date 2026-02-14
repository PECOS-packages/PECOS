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

import math

from pecos.slr import CReg, If, Main, QReg, Repeat
from pecos.slr.ast import slr_to_ast
from pecos.slr.ast.nodes import (
    AllocatorDecl,
    BinaryExpr,
    BinaryOp,
    BitRef,
    GateKind,
    GateOp,
    IfStmt,
    LiteralExpr,
    MeasureOp,
    Program,
    RegisterDecl,
    RepeatStmt,
    SlotRef,
    UnaryExpr,
    UnaryOp,
)
from pecos.slr.ast.pretty_print import format_expression, format_statement, pretty_print
from pecos.slr.qeclib import qubit as qb


class TestPrettyPrintBasic:
    """Basic pretty-printing tests."""

    def test_simple_program(self):
        """Simple program prints correctly."""
        prog = Main(
            q := QReg("q", 2),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
        )

        ast = slr_to_ast(prog)
        result = pretty_print(ast)

        assert 'q := QReg("q", 2)' in result
        assert "qb.H(q[0])" in result
        assert "qb.CX(q[0], q[1])" in result
        assert result.startswith("Main(")
        assert result.endswith(")")

    def test_empty_program(self):
        """Empty program prints correctly."""
        prog = Program(
            name="test",
            allocator=AllocatorDecl(name="q", capacity=1),
        )

        result = pretty_print(prog)

        assert "Main(" in result
        assert 'q := QReg("q", 1)' in result

    def test_with_classical_register(self):
        """Program with classical register prints correctly."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            qb.H(q[0]),
        )

        ast = slr_to_ast(prog)
        result = pretty_print(ast)

        assert 'c := CReg("c", 1)' in result


class TestPrettyPrintGates:
    """Gate pretty-printing tests."""

    def test_single_qubit_gates(self):
        """Single-qubit gates print correctly."""
        prog = Main(
            q := QReg("q", 1),
            qb.H(q[0]),
            qb.X(q[0]),
            qb.Y(q[0]),
            qb.Z(q[0]),
        )

        ast = slr_to_ast(prog)
        result = pretty_print(ast)

        assert "qb.H(q[0])" in result
        assert "qb.X(q[0])" in result
        assert "qb.Y(q[0])" in result
        assert "qb.Z(q[0])" in result

    def test_two_qubit_gates(self):
        """Two-qubit gates print correctly."""
        prog = Main(
            q := QReg("q", 2),
            qb.CX(q[0], q[1]),
            qb.CZ(q[0], q[1]),
        )

        ast = slr_to_ast(prog)
        result = pretty_print(ast)

        assert "qb.CX(q[0], q[1])" in result
        assert "qb.CZ(q[0], q[1])" in result

    def test_rotation_gates(self):
        """Rotation gates with parameters print correctly."""
        prog = Main(
            q := QReg("q", 1),
            qb.RZ[0.5](q[0]),
            qb.RX[math.pi](q[0]),
        )

        ast = slr_to_ast(prog)
        result = pretty_print(ast)

        assert "qb.RZ[0.5](q[0])" in result
        # Pi value should be formatted
        assert "qb.RX[" in result


class TestPrettyPrintControlFlow:
    """Control flow pretty-printing tests."""

    def test_if_statement(self):
        """If statement prints correctly."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            If(c[0] == 1).Then(
                qb.H(q[0]),
            ),
        )

        ast = slr_to_ast(prog)
        result = pretty_print(ast)

        assert "If(" in result
        assert ".Then(" in result
        assert "qb.H(q[0])" in result

    def test_repeat_statement(self):
        """Repeat statement prints correctly."""
        prog = Main(
            q := QReg("q", 1),
            Repeat(cond=5).block(
                qb.X(q[0]),
            ),
        )

        ast = slr_to_ast(prog)
        result = pretty_print(ast)

        assert "Repeat(cond=5).block(" in result
        assert "qb.X(q[0])" in result

    def test_nested_control_flow(self):
        """Nested control flow prints correctly."""
        prog = Program(
            name="test",
            allocator=AllocatorDecl(name="q", capacity=1),
            declarations=(RegisterDecl(name="c", size=1),),
            body=(
                RepeatStmt(
                    count=3,
                    body=(
                        IfStmt(
                            condition=BinaryExpr(
                                op=BinaryOp.EQ,
                                left=LiteralExpr(value=1),
                                right=LiteralExpr(value=1),
                            ),
                            then_body=(GateOp(gate=GateKind.H, targets=(SlotRef(allocator="q", index=0),)),),
                        ),
                    ),
                ),
            ),
        )

        result = pretty_print(prog)

        assert "Repeat(cond=3).block(" in result
        assert "If(" in result
        # Check indentation is present
        lines = result.split("\n")
        assert any(line.startswith("        ") for line in lines)  # Nested indentation


class TestPrettyPrintExpressions:
    """Expression pretty-printing tests."""

    def test_binary_expressions(self):
        """Binary expressions print correctly."""
        expr = BinaryExpr(op=BinaryOp.EQ, left=LiteralExpr(value=1), right=LiteralExpr(value=0))

        result = format_expression(expr)

        assert result == "(1 == 0)"

    def test_arithmetic_expressions(self):
        """Arithmetic expressions print correctly."""
        expr = BinaryExpr(op=BinaryOp.ADD, left=LiteralExpr(value=2), right=LiteralExpr(value=3))

        result = format_expression(expr)

        assert result == "(2 + 3)"

    def test_unary_expressions(self):
        """Unary expressions print correctly."""
        expr = UnaryExpr(op=UnaryOp.NEG, operand=LiteralExpr(value=5))

        result = format_expression(expr)

        assert result == "-5"

    def test_not_expression(self):
        """Not expression prints correctly."""
        expr = UnaryExpr(op=UnaryOp.NOT, operand=LiteralExpr(value=True))

        result = format_expression(expr)

        assert result == "not True"

    def test_comparison_operators(self):
        """Comparison operators print correctly."""
        ops = [
            (BinaryOp.LT, "<"),
            (BinaryOp.LE, "<="),
            (BinaryOp.GT, ">"),
            (BinaryOp.GE, ">="),
            (BinaryOp.NE, "!="),
        ]

        for op, symbol in ops:
            expr = BinaryExpr(op=op, left=LiteralExpr(value=1), right=LiteralExpr(value=2))
            result = format_expression(expr)
            assert symbol in result


class TestPrettyPrintStatements:
    """Statement pretty-printing tests."""

    def test_gate_statement(self):
        """Gate statement formats correctly."""
        stmt = GateOp(gate=GateKind.H, targets=(SlotRef(allocator="q", index=0),))

        result = format_statement(stmt)

        assert result == "qb.H(q[0])"

    def test_measure_statement(self):
        """Measure statement formats correctly."""
        stmt = MeasureOp(
            targets=(SlotRef(allocator="q", index=0),),
            results=(BitRef(register="c", index=0),),
        )

        result = format_statement(stmt)

        assert "Measure(q[0])" in result
        assert "c[0]" in result

    def test_rotation_gate_statement(self):
        """Rotation gate statement formats correctly."""
        stmt = GateOp(
            gate=GateKind.RZ,
            targets=(SlotRef(allocator="q", index=0),),
            params=(LiteralExpr(value=0.25),),
        )

        result = format_statement(stmt)

        assert result == "qb.RZ[0.25](q[0])"


class TestPrettyPrintIndentation:
    """Indentation tests."""

    def test_default_indentation(self):
        """Default indentation is 4 spaces."""
        prog = Main(
            q := QReg("q", 1),
            qb.H(q[0]),
        )

        ast = slr_to_ast(prog)
        result = pretty_print(ast)

        lines = result.split("\n")
        # Find a content line (not Main( or ))
        content_lines = [l for l in lines if l.strip() and not l.strip().startswith("Main") and l.strip() != ")"]
        assert len(content_lines) > 0
        # Should start with 4 spaces
        assert content_lines[0].startswith("    ")

    def test_custom_indentation(self):
        """Custom indentation works."""
        prog = Main(
            q := QReg("q", 1),
            qb.H(q[0]),
        )

        ast = slr_to_ast(prog)
        result = pretty_print(ast, indent="  ")  # 2 spaces

        lines = result.split("\n")
        content_lines = [l for l in lines if l.strip() and not l.strip().startswith("Main") and l.strip() != ")"]
        assert len(content_lines) > 0
        # Should start with 2 spaces, not 4
        assert content_lines[0].startswith("  ")
        assert not content_lines[0].startswith("    ")


class TestPrettyPrintMultipleAllocators:
    """Multiple allocator tests."""

    def test_nested_allocators(self):
        """Nested allocators print correctly."""
        prog = Program(
            name="test",
            allocator=AllocatorDecl(name="q", capacity=4),
            declarations=(AllocatorDecl(name="data", capacity=2, parent="q"),),
            body=(),
        )

        result = pretty_print(prog)

        assert 'q := QReg("q", 4)' in result
        assert "parent=q" in result


class TestPrettyPrintRoundTrip:
    """Tests that verify output structure matches expected patterns."""

    def test_bell_state(self):
        """Bell state preparation prints nicely."""
        prog = Main(
            q := QReg("q", 2),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
        )

        ast = slr_to_ast(prog)
        result = pretty_print(ast)

        expected_parts = [
            "Main(",
            'q := QReg("q", 2)',
            "qb.H(q[0])",
            "qb.CX(q[0], q[1])",
            ")",
        ]

        for part in expected_parts:
            assert part in result

    def test_ghz_state(self):
        """GHZ state preparation prints nicely."""
        prog = Main(
            q := QReg("q", 3),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
            qb.CX(q[1], q[2]),
        )

        ast = slr_to_ast(prog)
        result = pretty_print(ast)

        assert "qb.H(q[0])" in result
        assert "qb.CX(q[0], q[1])" in result
        assert "qb.CX(q[1], q[2])" in result
