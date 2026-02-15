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

"""Tests for AST comparison and diff utilities."""

import pytest
from pecos.slr import CReg, If, Main, QAlloc, QReg, Repeat
from pecos.slr.ast import slr_to_ast
from pecos.slr.ast.compare import (
    AstComparator,
    AstDiff,
    ast_equal,
    compare_ast,
    nodes_equal,
)
from pecos.slr.ast.nodes import (
    AllocatorDecl,
    BinaryExpr,
    BinaryOp,
    BitRef,
    GateKind,
    GateOp,
    LiteralExpr,
    Program,
    SlotRef,
    SourceLocation,
)
from pecos.slr.qeclib import qubit as qb


class TestAstEqual:
    """Tests for ast_equal function."""

    def test_identical_programs(self) -> None:
        """Identical programs compare as equal."""
        prog1 = Main(
            q := QReg("q", 2),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
        )
        prog2 = Main(
            r := QReg("q", 2),
            qb.H(r[0]),
            qb.CX(r[0], r[1]),
        )

        ast1 = slr_to_ast(prog1)
        ast2 = slr_to_ast(prog2)

        assert ast_equal(ast1, ast2)

    def test_different_gates(self) -> None:
        """Programs with different gates compare as not equal."""
        prog1 = Main(
            q := QReg("q", 1),
            qb.H(q[0]),
        )
        prog2 = Main(
            q := QReg("q", 1),
            qb.X(q[0]),
        )

        ast1 = slr_to_ast(prog1)
        ast2 = slr_to_ast(prog2)

        assert not ast_equal(ast1, ast2)

    def test_different_allocator_sizes(self) -> None:
        """Programs with different allocator sizes compare as not equal."""
        prog1 = Main(_q := QReg("q", 2))
        prog2 = Main(_q := QReg("q", 3))

        ast1 = slr_to_ast(prog1)
        ast2 = slr_to_ast(prog2)

        assert not ast_equal(ast1, ast2)

    def test_different_body_length(self) -> None:
        """Programs with different body lengths compare as not equal."""
        prog1 = Main(
            q := QReg("q", 2),
            qb.H(q[0]),
        )
        prog2 = Main(
            q := QReg("q", 2),
            qb.H(q[0]),
            qb.X(q[1]),
        )

        ast1 = slr_to_ast(prog1)
        ast2 = slr_to_ast(prog2)

        assert not ast_equal(ast1, ast2)

    def test_empty_programs_equal(self) -> None:
        """Empty programs compare as equal."""
        prog1 = Main()
        prog2 = Main()

        ast1 = slr_to_ast(prog1)
        ast2 = slr_to_ast(prog2)

        assert ast_equal(ast1, ast2)

    def test_ignore_name_option(self) -> None:
        """Test that ignore_name option works."""
        ast1 = Program(name="program1", declarations=(), body=())
        ast2 = Program(name="program2", declarations=(), body=())

        # With ignore_name=True (default False)
        assert not ast_equal(ast1, ast2, ignore_name=False)
        assert ast_equal(ast1, ast2, ignore_name=True)


class TestCompareAst:
    """Tests for compare_ast function."""

    def test_returns_ast_diff(self) -> None:
        """compare_ast returns AstDiff object."""
        prog = Main(_q := QReg("q", 1))
        ast = slr_to_ast(prog)

        diff = compare_ast(ast, ast)

        assert isinstance(diff, AstDiff)

    def test_equal_programs_diff(self) -> None:
        """Equal programs produce diff with equal=True and no differences."""
        prog1 = Main(
            q := QReg("q", 2),
            qb.H(q[0]),
        )
        prog2 = Main(
            q := QReg("q", 2),
            qb.H(q[0]),
        )

        ast1 = slr_to_ast(prog1)
        ast2 = slr_to_ast(prog2)

        diff = compare_ast(ast1, ast2)

        assert diff.equal
        assert len(diff.differences) == 0

    def test_different_programs_diff(self) -> None:
        """Different programs produce diff with differences."""
        prog1 = Main(
            q := QReg("q", 1),
            qb.H(q[0]),
        )
        prog2 = Main(
            q := QReg("q", 1),
            qb.X(q[0]),
        )

        ast1 = slr_to_ast(prog1)
        ast2 = slr_to_ast(prog2)

        diff = compare_ast(ast1, ast2)

        assert not diff.equal
        assert len(diff.differences) > 0

    def test_diff_contains_path_info(self) -> None:
        """Diff contains path information to differences."""
        prog1 = Main(
            q := QReg("q", 1),
            qb.H(q[0]),
        )
        prog2 = Main(
            q := QReg("q", 1),
            qb.X(q[0]),
        )

        ast1 = slr_to_ast(prog1)
        ast2 = slr_to_ast(prog2)

        diff = compare_ast(ast1, ast2)

        # Should contain path to the difference
        diff_text = str(diff)
        assert "body" in diff_text or "gate" in diff_text

    def test_diff_str_representation(self) -> None:
        """Diff has readable string representation."""
        prog1 = Main(_q := QReg("q", 1))
        prog2 = Main(_q := QReg("q", 2))

        ast1 = slr_to_ast(prog1)
        ast2 = slr_to_ast(prog2)

        diff = compare_ast(ast1, ast2)

        # Should have readable string output
        diff_str = str(diff)
        assert "differ" in diff_str.lower() or "mismatch" in diff_str.lower()

    def test_equal_diff_str_representation(self) -> None:
        """Equal diff string contains 'equal'."""
        prog = Main(_q := QReg("q", 1))
        ast = slr_to_ast(prog)

        diff = compare_ast(ast, ast)

        assert "equal" in str(diff).lower()


class TestAstDiff:
    """Tests for AstDiff dataclass."""

    def test_bool_conversion_true(self) -> None:
        """AstDiff converts to True when equal."""
        diff = AstDiff(equal=True, differences=[])

        assert bool(diff) is True
        # Can use in if statement
        passed = bool(diff)
        assert passed

    def test_bool_conversion_false(self) -> None:
        """AstDiff converts to False when not equal."""
        diff = AstDiff(equal=False, differences=["some difference"])

        assert bool(diff) is False

    def test_str_equal(self) -> None:
        """Equal diff string contains 'equal'."""
        diff = AstDiff(equal=True, differences=[])

        output = str(diff)

        assert "equal" in output.lower()

    def test_str_with_differences(self) -> None:
        """Diff string lists all differences."""
        diff = AstDiff(equal=False, differences=["diff1", "diff2"])

        output = str(diff)

        assert "diff1" in output
        assert "diff2" in output
        assert "2 difference" in output


class TestNodesEqual:
    """Tests for nodes_equal function."""

    def test_slot_refs_equal(self) -> None:
        """Identical SlotRefs compare as equal."""
        a = SlotRef(allocator="q", index=0)
        b = SlotRef(allocator="q", index=0)

        assert nodes_equal(a, b)

    def test_slot_refs_different_allocator(self) -> None:
        """SlotRefs with different allocators compare as not equal."""
        a = SlotRef(allocator="q", index=0)
        b = SlotRef(allocator="r", index=0)

        assert not nodes_equal(a, b)

    def test_slot_refs_different_index(self) -> None:
        """SlotRefs with different indices compare as not equal."""
        a = SlotRef(allocator="q", index=0)
        b = SlotRef(allocator="q", index=1)

        assert not nodes_equal(a, b)

    def test_bit_refs_equal(self) -> None:
        """Identical BitRefs compare as equal."""
        a = BitRef(register="c", index=2)
        b = BitRef(register="c", index=2)

        assert nodes_equal(a, b)

    def test_gate_ops_equal(self) -> None:
        """Identical GateOps compare as equal."""
        a = GateOp(
            gate=GateKind.H,
            targets=(SlotRef(allocator="q", index=0),),
        )
        b = GateOp(
            gate=GateKind.H,
            targets=(SlotRef(allocator="q", index=0),),
        )

        assert nodes_equal(a, b)

    def test_gate_ops_different_kind(self) -> None:
        """GateOps with different kinds compare as not equal."""
        a = GateOp(
            gate=GateKind.H,
            targets=(SlotRef(allocator="q", index=0),),
        )
        b = GateOp(
            gate=GateKind.X,
            targets=(SlotRef(allocator="q", index=0),),
        )

        assert not nodes_equal(a, b)

    def test_literal_exprs_equal(self) -> None:
        """Identical LiteralExprs compare as equal."""
        a = LiteralExpr(value=42)
        b = LiteralExpr(value=42)

        assert nodes_equal(a, b)

    def test_literal_exprs_different(self) -> None:
        """LiteralExprs with different values compare as not equal."""
        a = LiteralExpr(value=42)
        b = LiteralExpr(value=43)

        assert not nodes_equal(a, b)

    def test_binary_exprs_equal(self) -> None:
        """Identical BinaryExprs compare as equal."""
        a = BinaryExpr(
            op=BinaryOp.ADD,
            left=LiteralExpr(value=1),
            right=LiteralExpr(value=2),
        )
        b = BinaryExpr(
            op=BinaryOp.ADD,
            left=LiteralExpr(value=1),
            right=LiteralExpr(value=2),
        )

        assert nodes_equal(a, b)

    def test_binary_exprs_different_op(self) -> None:
        """BinaryExprs with different operators compare as not equal."""
        a = BinaryExpr(
            op=BinaryOp.ADD,
            left=LiteralExpr(value=1),
            right=LiteralExpr(value=2),
        )
        b = BinaryExpr(
            op=BinaryOp.SUB,
            left=LiteralExpr(value=1),
            right=LiteralExpr(value=2),
        )

        assert not nodes_equal(a, b)


class TestSourceLocationHandling:
    """Tests for source location handling in comparison."""

    def test_ignores_location_by_default(self) -> None:
        """Source locations are ignored by default."""
        loc1 = SourceLocation(line=1, column=1)
        loc2 = SourceLocation(line=99, column=99)

        a = SlotRef(allocator="q", index=0, location=loc1)
        b = SlotRef(allocator="q", index=0, location=loc2)

        assert nodes_equal(a, b)

    def test_can_compare_locations(self) -> None:
        """Locations can be compared when ignore_location=False."""
        loc1 = SourceLocation(line=1, column=1)
        loc2 = SourceLocation(line=99, column=99)

        a = SlotRef(allocator="q", index=0, location=loc1)
        b = SlotRef(allocator="q", index=0, location=loc2)

        assert not nodes_equal(a, b, ignore_location=False)

    def test_locations_equal_when_same(self) -> None:
        """Same locations compare as equal."""
        loc = SourceLocation(line=10, column=5)

        a = SlotRef(allocator="q", index=0, location=loc)
        b = SlotRef(allocator="q", index=0, location=loc)

        assert nodes_equal(a, b, ignore_location=False)


class TestAstComparator:
    """Tests for AstComparator class."""

    def test_reusable(self) -> None:
        """Comparator can be reused for multiple comparisons."""
        comparator = AstComparator()

        prog1 = Main(q := QReg("q", 1), qb.H(q[0]))
        prog2 = Main(q := QReg("q", 1), qb.X(q[0]))
        prog3 = Main(q := QReg("q", 1), qb.H(q[0]))

        ast1 = slr_to_ast(prog1)
        ast2 = slr_to_ast(prog2)
        ast3 = slr_to_ast(prog3)

        diff1 = comparator.compare(ast1, ast2)
        diff2 = comparator.compare(ast1, ast3)

        assert not diff1.equal
        assert diff2.equal

    def test_ignore_location_option(self) -> None:
        """Comparator with ignore_location=False detects location differences."""
        comparator = AstComparator(ignore_location=False)

        loc1 = SourceLocation(line=1, column=1)
        loc2 = SourceLocation(line=2, column=2)

        ast1 = Program(name="test", declarations=(), body=(), location=loc1)
        ast2 = Program(name="test", declarations=(), body=(), location=loc2)

        diff = comparator.compare(ast1, ast2)

        assert not diff.equal

    def test_ignore_name_option(self) -> None:
        """Comparator with ignore_name=True ignores program names."""
        comparator = AstComparator(ignore_name=True)

        ast1 = Program(name="program1", declarations=(), body=())
        ast2 = Program(name="program2", declarations=(), body=())

        diff = comparator.compare(ast1, ast2)

        assert diff.equal


class TestComplexComparisons:
    """Tests for complex AST comparisons."""

    def test_nested_control_flow(self) -> None:
        """Nested control flow compares correctly."""
        prog1 = Main(
            q := QReg("q", 2),
            c := CReg("c", 2),
            If(c[0] == 1).Then(
                If(c[1] == 1).Then(
                    qb.X(q[0]),
                ),
            ),
        )
        prog2 = Main(
            q := QReg("q", 2),
            c := CReg("c", 2),
            If(c[0] == 1).Then(
                If(c[1] == 1).Then(
                    qb.X(q[0]),
                ),
            ),
        )

        ast1 = slr_to_ast(prog1)
        ast2 = slr_to_ast(prog2)

        assert ast_equal(ast1, ast2)

    def test_nested_control_flow_different(self) -> None:
        """Nested control flow with differences is detected."""
        prog1 = Main(
            q := QReg("q", 2),
            c := CReg("c", 2),
            If(c[0] == 1).Then(
                If(c[1] == 1).Then(
                    qb.X(q[0]),
                ),
            ),
        )
        prog2 = Main(
            q := QReg("q", 2),
            c := CReg("c", 2),
            If(c[0] == 1).Then(
                If(c[1] == 1).Then(
                    qb.Y(q[0]),  # Different gate
                ),
            ),
        )

        ast1 = slr_to_ast(prog1)
        ast2 = slr_to_ast(prog2)

        diff = compare_ast(ast1, ast2)

        assert not diff.equal
        # Should identify the nested difference
        assert any("gate" in d.lower() or "body" in d.lower() for d in diff.differences)

    def test_repeat_blocks_equal(self) -> None:
        """Repeat blocks with same count and body compare as equal."""
        prog1 = Main(
            q := QReg("q", 1),
            Repeat(cond=5).block(
                qb.H(q[0]),
            ),
        )
        prog2 = Main(
            q := QReg("q", 1),
            Repeat(cond=5).block(
                qb.H(q[0]),
            ),
        )

        ast1 = slr_to_ast(prog1)
        ast2 = slr_to_ast(prog2)

        assert ast_equal(ast1, ast2)

    def test_repeat_different_count(self) -> None:
        """Repeat blocks with different counts compare as not equal."""
        prog1 = Main(
            q := QReg("q", 1),
            Repeat(cond=5).block(
                qb.H(q[0]),
            ),
        )
        prog2 = Main(
            q := QReg("q", 1),
            Repeat(cond=10).block(
                qb.H(q[0]),
            ),
        )

        ast1 = slr_to_ast(prog1)
        ast2 = slr_to_ast(prog2)

        assert not ast_equal(ast1, ast2)

    def test_hierarchical_allocators(self) -> None:
        """Hierarchical allocators compare correctly."""
        all1 = QAlloc(4, name="all")
        data1 = QAlloc(2, name="data", parent=all1)

        all2 = QAlloc(4, name="all")
        data2 = QAlloc(2, name="data", parent=all2)

        prog1 = Main(all1, data1, qb.H(data1[0]))
        prog2 = Main(all2, data2, qb.H(data2[0]))

        ast1 = slr_to_ast(prog1)
        ast2 = slr_to_ast(prog2)

        assert ast_equal(ast1, ast2)

    def test_qec_syndrome_extraction(self) -> None:
        """QEC syndrome extraction pattern compares correctly."""
        prog1 = Main(
            data := QReg("data", 2),
            ancilla := QReg("ancilla", 1),
            c := CReg("c", 1),
            qb.CX(data[0], ancilla[0]),
            qb.CX(data[1], ancilla[0]),
            qb.Measure(ancilla[0]) > c[0],
        )
        prog2 = Main(
            data := QReg("data", 2),
            ancilla := QReg("ancilla", 1),
            c := CReg("c", 1),
            qb.CX(data[0], ancilla[0]),
            qb.CX(data[1], ancilla[0]),
            qb.Measure(ancilla[0]) > c[0],
        )

        ast1 = slr_to_ast(prog1)
        ast2 = slr_to_ast(prog2)

        assert ast_equal(ast1, ast2)


class TestEdgeCases:
    """Edge case tests."""

    def test_none_handling(self) -> None:
        """Test that None values are handled correctly."""
        decl1 = AllocatorDecl(name="q", capacity=2, parent=None)
        decl2 = AllocatorDecl(name="q", capacity=2, parent=None)

        assert nodes_equal(decl1, decl2)

    def test_none_vs_value(self) -> None:
        """Test None vs actual value."""
        decl1 = AllocatorDecl(name="q", capacity=2, parent=None)
        decl2 = AllocatorDecl(name="q", capacity=2, parent="all")

        assert not nodes_equal(decl1, decl2)

    def test_empty_tuples(self) -> None:
        """Test comparison of empty tuples."""
        ast1 = Program(name="test", declarations=(), body=())
        ast2 = Program(name="test", declarations=(), body=())

        assert ast_equal(ast1, ast2)

    def test_different_tuple_lengths(self) -> None:
        """Test tuples of different lengths."""
        gate1 = GateOp(
            gate=GateKind.CX,
            targets=(
                SlotRef(allocator="q", index=0),
                SlotRef(allocator="q", index=1),
            ),
        )
        gate2 = GateOp(
            gate=GateKind.H,
            targets=(SlotRef(allocator="q", index=0),),
        )

        assert not nodes_equal(gate1, gate2)

    def test_bool_values(self) -> None:
        """Test boolean value comparison."""
        expr1 = LiteralExpr(value=True)
        expr2 = LiteralExpr(value=True)
        expr3 = LiteralExpr(value=False)

        assert nodes_equal(expr1, expr2)
        assert not nodes_equal(expr1, expr3)

    def test_float_values(self) -> None:
        """Test float value comparison."""
        expr1 = LiteralExpr(value=3.14159)
        expr2 = LiteralExpr(value=3.14159)
        expr3 = LiteralExpr(value=2.71828)

        assert nodes_equal(expr1, expr2)
        assert not nodes_equal(expr1, expr3)

    def test_int_vs_float(self) -> None:
        """Test that int and float of same value are different types."""
        expr1 = LiteralExpr(value=5)
        expr2 = LiteralExpr(value=5.0)

        # They're different types, so should not be equal
        # (Python int vs float)
        # Actually in Python 5 == 5.0 is True, so this depends on implementation
        # The comparator checks type(a) is not type(b), so they'd be different
        assert not nodes_equal(expr1, expr2)


class TestSerializationRoundtripComparison:
    """Test that serialization round-trip produces equal ASTs."""

    def test_serialization_preserves_equality(self) -> None:
        """Serialization round-trip preserves AST equality."""
        from pecos.slr.ast.serialize import ast_to_json, json_to_ast

        prog = Main(
            q := QReg("q", 2),
            c := CReg("c", 1),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
            qb.Measure(q[0]) > c[0],
        )
        ast = slr_to_ast(prog)

        # Round-trip through JSON
        json_str = ast_to_json(ast)
        restored = json_to_ast(json_str)

        # Should be equal
        assert ast_equal(ast, restored)

    def test_double_serialization_equal(self) -> None:
        """Double serialization round-trip preserves equality."""
        from pecos.slr.ast.serialize import ast_to_json, json_to_ast

        prog = Main(
            q := QReg("q", 2),
            If(LiteralExpr(value=True)).Then(
                qb.H(q[0]),
            ),
        )
        ast = slr_to_ast(prog)

        # Double round-trip
        restored1 = json_to_ast(ast_to_json(ast))
        restored2 = json_to_ast(ast_to_json(restored1))

        assert ast_equal(restored1, restored2)
