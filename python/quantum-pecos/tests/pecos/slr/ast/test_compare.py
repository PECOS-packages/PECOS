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

"""Tests for AST comparison."""

import math

from pecos.slr import CReg, If, Main, QReg, Repeat
from pecos.slr.ast import slr_to_ast
from pecos.slr.ast.compare import ast_equal, compare_ast, nodes_equal
from pecos.slr.ast.nodes import (
    AllocatorDecl,
    GateKind,
    GateOp,
    LiteralExpr,
    Program,
    RegisterDecl,
    SlotRef,
    SourceLocation,
)
from pecos.slr.qeclib import qubit as qb


class TestAstEqual:
    """Tests for ast_equal function."""

    def test_identical_programs(self):
        """Identical programs are equal."""
        prog1 = Main(
            q := QReg("q", 2),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
        )
        prog2 = Main(
            q := QReg("q", 2),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
        )

        ast1 = slr_to_ast(prog1)
        ast2 = slr_to_ast(prog2)

        assert ast_equal(ast1, ast2)

    def test_different_gates(self):
        """Programs with different gates are not equal."""
        prog1 = Main(q := QReg("q", 1), qb.H(q[0]))
        prog2 = Main(q := QReg("q", 1), qb.X(q[0]))

        ast1 = slr_to_ast(prog1)
        ast2 = slr_to_ast(prog2)

        assert not ast_equal(ast1, ast2)

    def test_different_allocator_sizes(self):
        """Programs with different allocator sizes are not equal."""
        prog1 = Program(
            name="test",
            allocator=AllocatorDecl(name="q", capacity=2),
        )
        prog2 = Program(
            name="test",
            allocator=AllocatorDecl(name="q", capacity=3),
        )

        assert not ast_equal(prog1, prog2)

    def test_different_body_length(self):
        """Programs with different body lengths are not equal."""
        prog1 = Main(q := QReg("q", 1), qb.H(q[0]))
        prog2 = Main(q := QReg("q", 1), qb.H(q[0]), qb.X(q[0]))

        ast1 = slr_to_ast(prog1)
        ast2 = slr_to_ast(prog2)

        assert not ast_equal(ast1, ast2)

    def test_ignore_location(self):
        """Location differences are ignored by default."""
        gate1 = GateOp(
            gate=GateKind.H,
            targets=(SlotRef(allocator="q", index=0),),
            location=SourceLocation(line=1, column=1),
        )
        gate2 = GateOp(
            gate=GateKind.H,
            targets=(SlotRef(allocator="q", index=0),),
            location=SourceLocation(line=10, column=5),
        )

        assert nodes_equal(gate1, gate2, ignore_location=True)
        assert not nodes_equal(gate1, gate2, ignore_location=False)

    def test_ignore_name(self):
        """Program name can be ignored."""
        prog1 = Program(name="prog1", allocator=AllocatorDecl(name="q", capacity=1))
        prog2 = Program(name="prog2", allocator=AllocatorDecl(name="q", capacity=1))

        assert ast_equal(prog1, prog2, ignore_name=True)
        assert not ast_equal(prog1, prog2, ignore_name=False)


class TestCompareAst:
    """Tests for compare_ast function."""

    def test_equal_returns_no_differences(self):
        """Equal programs have empty differences list."""
        prog1 = Main(q := QReg("q", 1), qb.H(q[0]))
        prog2 = Main(q := QReg("q", 1), qb.H(q[0]))

        ast1 = slr_to_ast(prog1)
        ast2 = slr_to_ast(prog2)

        diff = compare_ast(ast1, ast2)

        assert diff.equal
        assert len(diff.differences) == 0

    def test_different_returns_differences(self):
        """Different programs have non-empty differences list."""
        prog1 = Program(
            name="test",
            allocator=AllocatorDecl(name="q", capacity=2),
            body=(GateOp(gate=GateKind.H, targets=(SlotRef(allocator="q", index=0),)),),
        )
        prog2 = Program(
            name="test",
            allocator=AllocatorDecl(name="q", capacity=2),
            body=(GateOp(gate=GateKind.X, targets=(SlotRef(allocator="q", index=0),)),),
        )

        diff = compare_ast(prog1, prog2)

        assert not diff.equal
        assert len(diff.differences) > 0
        # Should identify the gate difference
        assert any("gate" in d.lower() or "mismatch" in d.lower() for d in diff.differences)

    def test_diff_string_representation(self):
        """Diff has useful string representation."""
        prog1 = Program(name="test", allocator=AllocatorDecl(name="q", capacity=1))
        prog2 = Program(name="test", allocator=AllocatorDecl(name="q", capacity=2))

        diff = compare_ast(prog1, prog2)

        result_str = str(diff)
        assert "differ" in result_str
        assert "capacity" in result_str

    def test_equal_diff_string(self):
        """Equal diff has nice string."""
        prog = Program(name="test", allocator=AllocatorDecl(name="q", capacity=1))

        diff = compare_ast(prog, prog)

        assert "equal" in str(diff).lower()

    def test_diff_bool_conversion(self):
        """Diff can be used as boolean."""
        prog = Program(name="test", allocator=AllocatorDecl(name="q", capacity=1))

        diff = compare_ast(prog, prog)

        # Should be truthy when equal
        assert diff
        assert bool(diff) is True


class TestNodesEqual:
    """Tests for nodes_equal function."""

    def test_gate_nodes_equal(self):
        """Gate nodes with same properties are equal."""
        gate1 = GateOp(gate=GateKind.H, targets=(SlotRef(allocator="q", index=0),))
        gate2 = GateOp(gate=GateKind.H, targets=(SlotRef(allocator="q", index=0),))

        assert nodes_equal(gate1, gate2)

    def test_gate_nodes_different_target(self):
        """Gate nodes with different targets are not equal."""
        gate1 = GateOp(gate=GateKind.H, targets=(SlotRef(allocator="q", index=0),))
        gate2 = GateOp(gate=GateKind.H, targets=(SlotRef(allocator="q", index=1),))

        assert not nodes_equal(gate1, gate2)

    def test_rotation_gates_equal(self):
        """Rotation gates with same params are equal."""
        gate1 = GateOp(
            gate=GateKind.RZ,
            targets=(SlotRef(allocator="q", index=0),),
            params=(LiteralExpr(value=0.5),),
        )
        gate2 = GateOp(
            gate=GateKind.RZ,
            targets=(SlotRef(allocator="q", index=0),),
            params=(LiteralExpr(value=0.5),),
        )

        assert nodes_equal(gate1, gate2)

    def test_rotation_gates_different_params(self):
        """Rotation gates with different params are not equal."""
        gate1 = GateOp(
            gate=GateKind.RZ,
            targets=(SlotRef(allocator="q", index=0),),
            params=(LiteralExpr(value=0.5),),
        )
        gate2 = GateOp(
            gate=GateKind.RZ,
            targets=(SlotRef(allocator="q", index=0),),
            params=(LiteralExpr(value=0.25),),
        )

        assert not nodes_equal(gate1, gate2)


class TestComplexCircuits:
    """Tests with complex circuits."""

    def test_bell_state_equal(self):
        """Bell state circuits are equal."""
        prog1 = Main(
            q := QReg("q", 2),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
        )
        prog2 = Main(
            q := QReg("q", 2),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
        )

        ast1 = slr_to_ast(prog1)
        ast2 = slr_to_ast(prog2)

        assert ast_equal(ast1, ast2)

    def test_ghz_different_size(self):
        """GHZ circuits with different sizes are not equal."""
        prog1 = Main(
            q := QReg("q", 3),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
            qb.CX(q[1], q[2]),
        )
        prog2 = Main(
            q := QReg("q", 4),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
            qb.CX(q[1], q[2]),
            qb.CX(q[2], q[3]),
        )

        ast1 = slr_to_ast(prog1)
        ast2 = slr_to_ast(prog2)

        assert not ast_equal(ast1, ast2)

    def test_circuit_with_control_flow(self):
        """Circuits with control flow can be compared."""
        prog1 = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            If(c[0] == 1).Then(
                qb.H(q[0]),
            ),
        )
        prog2 = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            If(c[0] == 1).Then(
                qb.H(q[0]),
            ),
        )

        ast1 = slr_to_ast(prog1)
        ast2 = slr_to_ast(prog2)

        assert ast_equal(ast1, ast2)

    def test_circuit_different_control_flow(self):
        """Circuits with different control flow are not equal."""
        prog1 = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            If(c[0] == 1).Then(
                qb.H(q[0]),
            ),
        )
        prog2 = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            If(c[0] == 0).Then(  # Different condition
                qb.H(q[0]),
            ),
        )

        ast1 = slr_to_ast(prog1)
        ast2 = slr_to_ast(prog2)

        assert not ast_equal(ast1, ast2)

    def test_repeat_loops_equal(self):
        """Repeat loops with same count are equal."""
        prog1 = Main(
            q := QReg("q", 1),
            Repeat(cond=5).block(
                qb.X(q[0]),
            ),
        )
        prog2 = Main(
            q := QReg("q", 1),
            Repeat(cond=5).block(
                qb.X(q[0]),
            ),
        )

        ast1 = slr_to_ast(prog1)
        ast2 = slr_to_ast(prog2)

        assert ast_equal(ast1, ast2)

    def test_repeat_loops_different_count(self):
        """Repeat loops with different counts are not equal."""
        prog1 = Main(
            q := QReg("q", 1),
            Repeat(cond=5).block(
                qb.X(q[0]),
            ),
        )
        prog2 = Main(
            q := QReg("q", 1),
            Repeat(cond=10).block(
                qb.X(q[0]),
            ),
        )

        ast1 = slr_to_ast(prog1)
        ast2 = slr_to_ast(prog2)

        assert not ast_equal(ast1, ast2)


class TestSerializationRoundTrip:
    """Test comparison with serialization round-trip."""

    def test_serialized_equal_to_original(self):
        """Serialized and restored AST equals original."""
        from pecos.slr.ast.serialize import ast_to_json, json_to_ast

        prog = Main(
            q := QReg("q", 3),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
            qb.CX(q[1], q[2]),
        )

        ast = slr_to_ast(prog)
        json_str = ast_to_json(ast)
        restored = json_to_ast(json_str)

        # Ignoring name because slr_to_ast may generate different names
        assert ast_equal(ast, restored, ignore_name=True)

    def test_double_serialization(self):
        """Double serialization produces equal result."""
        from pecos.slr.ast.serialize import ast_to_json, json_to_ast

        prog = Main(
            q := QReg("q", 2),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
        )

        ast = slr_to_ast(prog)

        # First round-trip
        json1 = ast_to_json(ast)
        restored1 = json_to_ast(json1)

        # Second round-trip
        json2 = ast_to_json(restored1)
        restored2 = json_to_ast(json2)

        assert ast_equal(restored1, restored2, ignore_name=True)
