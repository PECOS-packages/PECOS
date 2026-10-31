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

"""Tests for AST serialization and deserialization."""

import json

import pytest
from pecos.slr import CReg, If, Main, QAlloc, QReg, Repeat
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
    SourceLocation,
)
from pecos.slr.ast.serialize import ast_to_dict, ast_to_json, dict_to_ast, json_to_ast
from pecos.slr.qeclib import qubit as qb


class TestAstToDict:
    """Tests for ast_to_dict function."""

    def test_simple_program(self) -> None:
        """Simple program converts to dict correctly."""
        prog = Main(
            q := QReg("q", 2),
            qb.H(q[0]),
        )
        ast = slr_to_ast(prog)

        data = ast_to_dict(ast)

        assert data["_type"] == "Program"
        assert data["name"] == "Main"  # Program name matches class name
        assert len(data["declarations"]) == 1
        assert len(data["body"]) == 1

    def test_gate_op_serialization(self) -> None:
        """GateOp serializes with gate kind and targets."""
        prog = Main(
            q := QReg("q", 2),
            qb.CX(q[0], q[1]),
        )
        ast = slr_to_ast(prog)

        data = ast_to_dict(ast)

        gate_data = data["body"][0]
        assert gate_data["_type"] == "GateOp"
        assert gate_data["gate"]["_enum"] == "GateKind"
        assert gate_data["gate"]["value"] == "CX"
        assert len(gate_data["targets"]) == 2

    def test_slot_ref_serialization(self) -> None:
        """SlotRef serializes with allocator and index."""
        slot = SlotRef(allocator="q", index=5)

        data = ast_to_dict(slot)

        assert data["_type"] == "SlotRef"
        assert data["allocator"] == "q"
        assert data["index"] == 5

    def test_bit_ref_serialization(self) -> None:
        """BitRef serializes with register and index."""
        bit = BitRef(register="c", index=3)

        data = ast_to_dict(bit)

        assert data["_type"] == "BitRef"
        assert data["register"] == "c"
        assert data["index"] == 3

    def test_literal_expr_serialization(self) -> None:
        """LiteralExpr serializes with value."""
        expr = LiteralExpr(value=42)

        data = ast_to_dict(expr)

        assert data["_type"] == "LiteralExpr"
        assert data["value"] == 42

    def test_binary_expr_serialization(self) -> None:
        """BinaryExpr serializes with operator and operands."""
        expr = BinaryExpr(
            op=BinaryOp.EQ,
            left=LiteralExpr(value=1),
            right=LiteralExpr(value=2),
        )

        data = ast_to_dict(expr)

        assert data["_type"] == "BinaryExpr"
        assert data["op"]["_enum"] == "BinaryOp"
        assert data["op"]["value"] == "EQ"


class TestDictToAst:
    """Tests for dict_to_ast function."""

    def test_simple_program_roundtrip(self) -> None:
        """Program survives dict roundtrip."""
        prog = Main(
            q := QReg("q", 2),
            qb.H(q[0]),
        )
        ast = slr_to_ast(prog)

        data = ast_to_dict(ast)
        restored = dict_to_ast(data)

        assert isinstance(restored, Program)
        assert restored.name == ast.name
        assert len(restored.declarations) == len(ast.declarations)
        assert len(restored.body) == len(ast.body)

    def test_gate_kind_preserved(self) -> None:
        """GateKind enum is preserved through roundtrip."""
        prog = Main(
            q := QReg("q", 2),
            qb.CX(q[0], q[1]),
        )
        ast = slr_to_ast(prog)

        data = ast_to_dict(ast)
        restored = dict_to_ast(data)

        gate_op = restored.body[0]
        assert isinstance(gate_op, GateOp)
        assert gate_op.gate == GateKind.CX

    def test_slot_ref_roundtrip(self) -> None:
        """SlotRef survives dict roundtrip."""
        slot = SlotRef(allocator="data", index=7)

        data = ast_to_dict(slot)
        restored = dict_to_ast(data)

        assert isinstance(restored, SlotRef)
        assert restored.allocator == "data"
        assert restored.index == 7

    def test_binary_op_preserved(self) -> None:
        """BinaryOp enum is preserved through roundtrip."""
        expr = BinaryExpr(
            op=BinaryOp.LT,
            left=LiteralExpr(value=5),
            right=LiteralExpr(value=10),
        )

        data = ast_to_dict(expr)
        restored = dict_to_ast(data)

        assert isinstance(restored, BinaryExpr)
        assert restored.op == BinaryOp.LT

    def test_unknown_type_raises_error(self) -> None:
        """Unknown type raises ValueError."""
        data = {"_type": "UnknownNodeType"}

        with pytest.raises(ValueError, match="Unknown node type"):
            dict_to_ast(data)

    def test_missing_type_raises_error(self) -> None:
        """Missing _type field raises ValueError."""
        data = {"name": "test"}

        with pytest.raises(ValueError, match="missing '_type' field"):
            dict_to_ast(data)


class TestJsonSerialization:
    """Tests for JSON serialization functions."""

    def test_ast_to_json_produces_valid_json(self) -> None:
        """ast_to_json produces parseable JSON."""
        prog = Main(
            q := QReg("q", 2),
            qb.H(q[0]),
        )
        ast = slr_to_ast(prog)

        json_str = ast_to_json(ast)

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
        assert parsed["_type"] == "Program"

    def test_json_roundtrip_basic(self) -> None:
        """Basic program survives JSON roundtrip."""
        prog = Main(
            q := QReg("q", 3),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
            qb.CZ(q[1], q[2]),
        )
        ast = slr_to_ast(prog)

        json_str = ast_to_json(ast)
        restored = json_to_ast(json_str)

        assert restored.name == ast.name
        assert len(restored.body) == len(ast.body)

    def test_json_roundtrip_with_creg(self) -> None:
        """Program with CReg survives JSON roundtrip."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            qb.Measure(q[0]) > c[0],
        )
        ast = slr_to_ast(prog)

        json_str = ast_to_json(ast)
        restored = json_to_ast(json_str)

        # Check measure op preserved
        measure_op = restored.body[0]
        assert isinstance(measure_op, MeasureOp)
        assert len(measure_op.results) == 1

    def test_json_roundtrip_with_if_statement(self) -> None:
        """Program with If statement survives JSON roundtrip."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            If(c[0] == 1).Then(
                qb.X(q[0]),
            ),
        )
        ast = slr_to_ast(prog)

        json_str = ast_to_json(ast)
        restored = json_to_ast(json_str)

        if_stmt = restored.body[0]
        assert isinstance(if_stmt, IfStmt)
        assert len(if_stmt.then_body) == 1

    def test_json_roundtrip_with_repeat(self) -> None:
        """Program with Repeat survives JSON roundtrip."""
        prog = Main(
            q := QReg("q", 1),
            Repeat(cond=5).block(
                qb.H(q[0]),
            ),
        )
        ast = slr_to_ast(prog)

        json_str = ast_to_json(ast)
        restored = json_to_ast(json_str)

        repeat_stmt = restored.body[0]
        assert isinstance(repeat_stmt, RepeatStmt)
        assert repeat_stmt.count == 5

    def test_json_to_ast_non_program_raises_error(self) -> None:
        """json_to_ast raises error for non-Program JSON."""
        slot = SlotRef(allocator="q", index=0)
        json_str = json.dumps(ast_to_dict(slot))

        with pytest.raises(ValueError, match="Expected Program"):
            json_to_ast(json_str)

    def test_json_compact_output(self) -> None:
        """JSON can be output without indentation."""
        prog = Main(_q := QReg("q", 1))
        ast = slr_to_ast(prog)

        json_str = ast_to_json(ast, indent=None)

        # Compact output should have no newlines
        assert "\n" not in json_str


class TestComplexRoundtrip:
    """Complex round-trip tests combining multiple features."""

    def test_full_qec_pattern(self) -> None:
        """Test a QEC syndrome extraction pattern."""
        prog = Main(
            data := QReg("data", 2),
            ancilla := QReg("ancilla", 1),
            c := CReg("c", 1),
            qb.CX(data[0], ancilla[0]),
            qb.CX(data[1], ancilla[0]),
            qb.Measure(ancilla[0]) > c[0],
        )
        ast = slr_to_ast(prog)

        json_str = ast_to_json(ast)
        restored = json_to_ast(json_str)

        # Verify structure
        assert len(restored.declarations) == 3  # 2 QReg + 1 CReg
        assert len(restored.body) == 3  # 2 CX + 1 Measure

        # Verify gate types
        assert all(isinstance(s, (GateOp, MeasureOp)) for s in restored.body)

    def test_nested_control_flow(self) -> None:
        """Test nested if statements."""
        prog = Main(
            q := QReg("q", 2),
            c := CReg("c", 2),
            If(c[0] == 1)
            .Then(
                If(c[1] == 1)
                .Then(
                    qb.X(q[0]),
                )
                .Else(
                    qb.Y(q[0]),
                ),
            )
            .Else(
                qb.Z(q[0]),
            ),
        )
        ast = slr_to_ast(prog)

        json_str = ast_to_json(ast)
        restored = json_to_ast(json_str)

        outer_if = restored.body[0]
        assert isinstance(outer_if, IfStmt)
        assert len(outer_if.then_body) == 1
        assert len(outer_if.else_body) == 1

        inner_if = outer_if.then_body[0]
        assert isinstance(inner_if, IfStmt)

    def test_hierarchical_allocators(self) -> None:
        """Test hierarchical allocator serialization."""
        all_qubits = QAlloc(4, name="all")
        data_alloc = QAlloc(2, name="data", parent=all_qubits)
        ancilla_alloc = QAlloc(2, name="ancilla", parent=all_qubits)

        prog = Main(
            all_qubits,
            data_alloc,
            ancilla_alloc,
            qb.H(data_alloc[0]),
            qb.CX(data_alloc[0], ancilla_alloc[0]),
        )
        ast = slr_to_ast(prog)

        json_str = ast_to_json(ast)
        restored = json_to_ast(json_str)

        # Find allocator declarations
        allocators = [d for d in restored.declarations if isinstance(d, AllocatorDecl)]
        assert len(allocators) == 3

        # Check parent relationships preserved
        data_decl = next(d for d in allocators if d.name == "data")
        assert data_decl.parent == "all"

        ancilla_decl = next(d for d in allocators if d.name == "ancilla")
        assert ancilla_decl.parent == "all"

    def test_double_roundtrip_identical(self) -> None:
        """Test that double round-trip produces identical JSON."""
        prog = Main(
            q := QReg("q", 2),
            c := CReg("c", 1),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
            qb.Measure(q[0]) > c[0],
            If(c[0] == 1).Then(
                qb.X(q[1]),
            ),
        )
        ast = slr_to_ast(prog)

        # First round-trip
        json1 = ast_to_json(ast)
        restored1 = json_to_ast(json1)

        # Second round-trip
        json2 = ast_to_json(restored1)
        restored2 = json_to_ast(json2)

        # Third round-trip
        json3 = ast_to_json(restored2)

        # All JSON should be identical
        assert json1 == json2 == json3


class TestSourceLocation:
    """Tests for source location preservation."""

    def test_source_location_roundtrip(self) -> None:
        """Test that source locations are preserved."""
        loc = SourceLocation(line=10, column=5, file="test.py")
        slot = SlotRef(allocator="q", index=0, location=loc)

        data = ast_to_dict(slot)
        restored = dict_to_ast(data)

        assert restored.location is not None
        assert restored.location.line == 10
        assert restored.location.column == 5
        assert restored.location.file == "test.py"

    def test_source_location_optional(self) -> None:
        """Test that missing source location is handled."""
        slot = SlotRef(allocator="q", index=0)

        data = ast_to_dict(slot)
        restored = dict_to_ast(data)

        assert restored.location is None


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_program(self) -> None:
        """Empty program survives JSON roundtrip."""
        prog = Main()
        ast = slr_to_ast(prog)

        json_str = ast_to_json(ast)
        restored = json_to_ast(json_str)

        assert len(restored.declarations) == 0
        assert len(restored.body) == 0

    def test_register_decl_roundtrip(self) -> None:
        """RegisterDecl round-trips through dict (no is_result field post-3b)."""
        decl = RegisterDecl(name="scratch", size=4)

        data = ast_to_dict(decl)
        restored = dict_to_ast(data)

        assert isinstance(restored, RegisterDecl)
        assert restored.name == "scratch"
        assert restored.size == 4
        assert "is_result" not in data

    def test_allocator_without_parent(self) -> None:
        """Test AllocatorDecl without parent."""
        decl = AllocatorDecl(name="q", capacity=5)

        data = ast_to_dict(decl)
        restored = dict_to_ast(data)

        assert isinstance(restored, AllocatorDecl)
        assert restored.parent is None

    def test_boolean_literal_preserved(self) -> None:
        """Test that boolean literals are preserved correctly."""
        expr_true = LiteralExpr(value=True)
        expr_false = LiteralExpr(value=False)

        data_true = ast_to_dict(expr_true)
        data_false = ast_to_dict(expr_false)

        restored_true = dict_to_ast(data_true)
        restored_false = dict_to_ast(data_false)

        assert restored_true.value is True
        assert restored_false.value is False
        assert isinstance(restored_true.value, bool)
        assert isinstance(restored_false.value, bool)

    def test_float_literal_preserved(self) -> None:
        """Test that float literals are preserved."""
        expr = LiteralExpr(value=3.14159)

        data = ast_to_dict(expr)
        restored = dict_to_ast(data)

        assert restored.value == 3.14159
        assert isinstance(restored.value, float)
