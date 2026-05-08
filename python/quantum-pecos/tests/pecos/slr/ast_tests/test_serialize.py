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

"""Tests for AST serialization."""

import json
import math

import pytest
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
    SourceLocation,
)
from pecos.slr.ast.serialize import ast_to_dict, ast_to_json, dict_to_ast, json_to_ast
from pecos.slr.qeclib import qubit as qb


class TestAstToDict:
    """Tests for ast_to_dict conversion."""

    def test_simple_program(self) -> None:
        """Simple program converts to dict."""
        prog = Program(
            name="test",
            allocator=AllocatorDecl(name="q", capacity=2),
            body=(GateOp(gate=GateKind.H, targets=(SlotRef(allocator="q", index=0),)),),
        )

        result = ast_to_dict(prog)

        assert result["_type"] == "Program"
        assert result["name"] == "test"
        assert result["allocator"]["_type"] == "AllocatorDecl"
        assert result["allocator"]["name"] == "q"
        assert result["allocator"]["capacity"] == 2

    def test_gate_kind_enum(self) -> None:
        """GateKind enum serializes correctly."""
        gate = GateOp(
            gate=GateKind.CX,
            targets=(SlotRef(allocator="q", index=0), SlotRef(allocator="q", index=1)),
        )

        result = ast_to_dict(gate)

        assert result["gate"]["_enum"] == "GateKind"
        assert result["gate"]["value"] == "CX"

    def test_binary_op_enum(self) -> None:
        """BinaryOp enum serializes correctly."""
        expr = BinaryExpr(
            op=BinaryOp.EQ,
            left=LiteralExpr(value=1),
            right=LiteralExpr(value=0),
        )

        result = ast_to_dict(expr)

        assert result["op"]["_enum"] == "BinaryOp"
        assert result["op"]["value"] == "EQ"

    def test_tuple_fields(self) -> None:
        """Tuple fields serialize to lists."""
        prog = Program(
            name="test",
            allocator=AllocatorDecl(name="q", capacity=2),
            body=(
                GateOp(gate=GateKind.H, targets=(SlotRef(allocator="q", index=0),)),
                GateOp(gate=GateKind.X, targets=(SlotRef(allocator="q", index=1),)),
            ),
        )

        result = ast_to_dict(prog)

        assert isinstance(result["body"], list)
        assert len(result["body"]) == 2

    def test_source_location(self) -> None:
        """SourceLocation serializes correctly."""
        gate = GateOp(
            gate=GateKind.H,
            targets=(SlotRef(allocator="q", index=0),),
            location=SourceLocation(line=10, column=5, file="test.py"),
        )

        result = ast_to_dict(gate)

        assert result["location"]["_type"] == "SourceLocation"
        assert result["location"]["line"] == 10
        assert result["location"]["column"] == 5
        assert result["location"]["file"] == "test.py"

    def test_none_values(self) -> None:
        """None values serialize as null."""
        prog = Program(
            name="test",
            allocator=AllocatorDecl(name="q", capacity=1),
        )

        result = ast_to_dict(prog)

        assert result["location"] is None

    def test_rotation_gate_params(self) -> None:
        """Parameterized gate params serialize correctly."""
        gate = GateOp(
            gate=GateKind.RZ,
            targets=(SlotRef(allocator="q", index=0),),
            params=(LiteralExpr(value=0.5),),
        )

        result = ast_to_dict(gate)

        assert len(result["params"]) == 1
        assert result["params"][0]["_type"] == "LiteralExpr"
        assert result["params"][0]["value"] == 0.5


class TestDictToAst:
    """Tests for dict_to_ast conversion."""

    def test_simple_program(self) -> None:
        """Simple dict converts to program."""
        data = {
            "_type": "Program",
            "name": "test",
            "declarations": [],
            "body": [],
            "returns": [],
            "allocator": {
                "_type": "AllocatorDecl",
                "name": "q",
                "capacity": 2,
                "parent": None,
                "location": None,
            },
            "location": None,
        }

        result = dict_to_ast(data)

        assert isinstance(result, Program)
        assert result.name == "test"
        assert result.allocator.name == "q"
        assert result.allocator.capacity == 2

    def test_gate_kind_enum(self) -> None:
        """GateKind enum deserializes correctly."""
        data = {
            "_type": "GateOp",
            "gate": {"_enum": "GateKind", "value": "CX"},
            "targets": [
                {"_type": "SlotRef", "allocator": "q", "index": 0, "location": None},
                {"_type": "SlotRef", "allocator": "q", "index": 1, "location": None},
            ],
            "params": [],
            "location": None,
        }

        result = dict_to_ast(data)

        assert isinstance(result, GateOp)
        assert result.gate == GateKind.CX

    def test_binary_op_enum(self) -> None:
        """BinaryOp enum deserializes correctly."""
        data = {
            "_type": "BinaryExpr",
            "op": {"_enum": "BinaryOp", "value": "EQ"},
            "left": {"_type": "LiteralExpr", "value": 1, "location": None},
            "right": {"_type": "LiteralExpr", "value": 0, "location": None},
            "location": None,
        }

        result = dict_to_ast(data)

        assert isinstance(result, BinaryExpr)
        assert result.op == BinaryOp.EQ

    def test_missing_type_raises(self) -> None:
        """Missing _type raises ValueError."""
        data = {"name": "test"}

        with pytest.raises(ValueError, match="_type"):
            dict_to_ast(data)

    def test_unknown_type_raises(self) -> None:
        """Unknown _type raises ValueError."""
        data = {"_type": "UnknownNode"}

        with pytest.raises(ValueError, match="Unknown node type"):
            dict_to_ast(data)


class TestJsonRoundTrip:
    """Tests for JSON round-trip serialization."""

    def test_simple_circuit(self) -> None:
        """Simple circuit round-trips correctly."""
        prog = Main(
            q := QReg("q", 2),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
        )

        ast = slr_to_ast(prog)
        json_str = ast_to_json(ast)
        restored = json_to_ast(json_str)

        assert restored.name == ast.name
        assert len(restored.body) == len(ast.body)

    def test_rotation_gates(self) -> None:
        """Rotation gates with float params round-trip."""
        prog = Main(
            q := QReg("q", 1),
            qb.RZ[0.5](q[0]),
            qb.RX[math.pi](q[0]),
        )

        ast = slr_to_ast(prog)
        json_str = ast_to_json(ast)
        restored = json_to_ast(json_str)

        # Find RZ gate
        rz_gates = [s for s in restored.body if isinstance(s, GateOp) and s.gate == GateKind.RZ]
        assert len(rz_gates) == 1
        assert rz_gates[0].params[0].value == 0.5

    def test_measurement(self) -> None:
        """Measurement with classical register round-trips."""
        prog = Program(
            name="test",
            allocator=AllocatorDecl(name="q", capacity=2),
            declarations=(RegisterDecl(name="c", size=2),),
            body=(
                GateOp(gate=GateKind.H, targets=(SlotRef(allocator="q", index=0),)),
                MeasureOp(
                    targets=(
                        SlotRef(allocator="q", index=0),
                        SlotRef(allocator="q", index=1),
                    ),
                    results=(
                        BitRef(register="c", index=0),
                        BitRef(register="c", index=1),
                    ),
                ),
            ),
        )

        json_str = ast_to_json(prog)
        restored = json_to_ast(json_str)

        assert len(restored.declarations) == 1
        assert isinstance(restored.declarations[0], RegisterDecl)
        assert restored.declarations[0].name == "c"

    def test_control_flow(self) -> None:
        """Control flow structures round-trip."""
        prog = Program(
            name="test",
            allocator=AllocatorDecl(name="q", capacity=1),
            declarations=(RegisterDecl(name="c", size=1),),
            body=(
                RepeatStmt(
                    count=3,
                    body=(
                        GateOp(
                            gate=GateKind.H,
                            targets=(SlotRef(allocator="q", index=0),),
                        ),
                    ),
                ),
                IfStmt(
                    condition=BinaryExpr(
                        op=BinaryOp.EQ,
                        left=LiteralExpr(value=1),
                        right=LiteralExpr(value=1),
                    ),
                    then_body=(
                        GateOp(
                            gate=GateKind.X,
                            targets=(SlotRef(allocator="q", index=0),),
                        ),
                    ),
                ),
            ),
        )

        json_str = ast_to_json(prog)
        restored = json_to_ast(json_str)

        assert isinstance(restored.body[0], RepeatStmt)
        assert restored.body[0].count == 3
        assert isinstance(restored.body[1], IfStmt)

    def test_nested_allocators(self) -> None:
        """Nested allocators round-trip."""
        prog = Program(
            name="test",
            allocator=AllocatorDecl(name="q", capacity=4),
            declarations=(AllocatorDecl(name="data", capacity=2, parent="q"),),
            body=(),
        )

        json_str = ast_to_json(prog)
        restored = json_to_ast(json_str)

        assert restored.declarations[0].parent == "q"

    def test_from_slr_if_statement(self) -> None:
        """If statement from SLR round-trips."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            If(c[0] == 1).Then(
                qb.H(q[0]),
            ),
        )

        ast = slr_to_ast(prog)
        json_str = ast_to_json(ast)
        restored = json_to_ast(json_str)

        # Find the if statement
        if_stmts = [s for s in restored.body if isinstance(s, IfStmt)]
        assert len(if_stmts) == 1

    def test_from_slr_repeat(self) -> None:
        """Repeat from SLR round-trips."""
        prog = Main(
            q := QReg("q", 1),
            Repeat(cond=5).block(
                qb.X(q[0]),
            ),
        )

        ast = slr_to_ast(prog)
        json_str = ast_to_json(ast)
        restored = json_to_ast(json_str)

        repeat_stmts = [s for s in restored.body if isinstance(s, RepeatStmt)]
        assert len(repeat_stmts) == 1
        assert repeat_stmts[0].count == 5


class TestJsonFormat:
    """Tests for JSON output format."""

    def test_compact_output(self) -> None:
        """Compact JSON has no extra whitespace."""
        prog = Program(
            name="test",
            allocator=AllocatorDecl(name="q", capacity=1),
        )

        json_str = ast_to_json(prog, indent=None)

        assert "\n" not in json_str

    def test_indented_output(self) -> None:
        """Indented JSON is human-readable."""
        prog = Program(
            name="test",
            allocator=AllocatorDecl(name="q", capacity=1),
        )

        json_str = ast_to_json(prog, indent=2)

        assert "\n" in json_str
        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["_type"] == "Program"

    def test_json_is_valid(self) -> None:
        """Output is valid JSON."""
        prog = Main(
            q := QReg("q", 3),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
            qb.CX(q[1], q[2]),
        )

        ast = slr_to_ast(prog)
        json_str = ast_to_json(ast)

        # Should not raise
        data = json.loads(json_str)
        assert isinstance(data, dict)


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_program(self) -> None:
        """Empty program round-trips."""
        prog = Program(
            name="empty",
            allocator=AllocatorDecl(name="q", capacity=1),
        )

        json_str = ast_to_json(prog)
        restored = json_to_ast(json_str)

        assert restored.name == "empty"
        assert len(restored.body) == 0

    def test_all_gate_kinds(self) -> None:
        """All gate kinds can be serialized."""
        for gate_kind in GateKind:
            gate = GateOp(
                gate=gate_kind,
                targets=tuple(SlotRef(allocator="q", index=i) for i in range(gate_kind.arity)),
                params=(LiteralExpr(value=0.5),) if gate_kind.is_parameterized else (),
            )

            result = ast_to_dict(gate)
            assert result["gate"]["value"] == gate_kind.name

    def test_all_binary_ops(self) -> None:
        """All binary operators can be serialized."""
        for op in BinaryOp:
            expr = BinaryExpr(
                op=op,
                left=LiteralExpr(value=1),
                right=LiteralExpr(value=2),
            )

            result = ast_to_dict(expr)
            assert result["op"]["value"] == op.name

    def test_float_precision(self) -> None:
        """Float values maintain precision."""
        prog = Program(
            name="test",
            allocator=AllocatorDecl(name="q", capacity=1),
            body=(
                GateOp(
                    gate=GateKind.RZ,
                    targets=(SlotRef(allocator="q", index=0),),
                    params=(LiteralExpr(value=3.141592653589793),),  # math.pi
                ),
            ),
        )

        json_str = ast_to_json(prog)
        restored = json_to_ast(json_str)

        rz = restored.body[0]
        assert isinstance(rz, GateOp)
        assert abs(rz.params[0].value - math.pi) < 1e-15
