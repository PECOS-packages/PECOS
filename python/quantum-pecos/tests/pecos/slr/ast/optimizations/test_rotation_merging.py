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

"""Tests for rotation merging optimization pass."""

import math

from pecos.slr import CReg, If, Main, QReg, Repeat
from pecos.slr.ast import slr_to_ast
from pecos.slr.ast.nodes import GateKind, GateOp, LiteralExpr
from pecos.slr.ast.optimizations import RotationMergingPass
from pecos.slr.qeclib import qubit as qb


class TestRotationMergingBasic:
    """Basic rotation merging tests."""

    def test_rz_rz_merges(self):
        """RZ+RZ on same qubit merges."""
        prog = Main(
            q := QReg("q", 1),
            qb.RZ[0.5](q[0]),
            qb.RZ[0.3](q[0]),
        )

        ast = slr_to_ast(prog)
        result = RotationMergingPass().optimize(ast)

        assert len(result.program.body) == 1
        assert result.gates_merged == 1

        # Check the merged gate has correct angle
        gate = result.program.body[0]
        assert isinstance(gate, GateOp)
        assert gate.gate == GateKind.RZ
        assert len(gate.params) == 1
        assert isinstance(gate.params[0], LiteralExpr)
        assert abs(gate.params[0].value - 0.8) < 1e-10

    def test_rx_rx_merges(self):
        """RX+RX on same qubit merges."""
        prog = Main(
            q := QReg("q", 1),
            qb.RX[math.pi / 4](q[0]),
            qb.RX[math.pi / 4](q[0]),
        )

        ast = slr_to_ast(prog)
        result = RotationMergingPass().optimize(ast)

        assert len(result.program.body) == 1
        assert result.gates_merged == 1

        gate = result.program.body[0]
        assert isinstance(gate, GateOp)
        assert gate.gate == GateKind.RX
        assert isinstance(gate.params[0], LiteralExpr)
        assert abs(gate.params[0].value - math.pi / 2) < 1e-10

    def test_ry_ry_merges(self):
        """RY+RY on same qubit merges."""
        prog = Main(
            q := QReg("q", 1),
            qb.RY[0.1](q[0]),
            qb.RY[0.2](q[0]),
        )

        ast = slr_to_ast(prog)
        result = RotationMergingPass().optimize(ast)

        assert len(result.program.body) == 1
        assert result.gates_merged == 1

        gate = result.program.body[0]
        assert isinstance(gate, GateOp)
        assert gate.gate == GateKind.RY
        assert isinstance(gate.params[0], LiteralExpr)
        assert abs(gate.params[0].value - 0.3) < 1e-10


class TestRotationMergingNoMerge:
    """Tests where rotations should NOT merge."""

    def test_different_rotation_types_no_merge(self):
        """Different rotation types do not merge."""
        prog = Main(
            q := QReg("q", 1),
            qb.RX[0.5](q[0]),
            qb.RZ[0.3](q[0]),
        )

        ast = slr_to_ast(prog)
        result = RotationMergingPass().optimize(ast)

        assert len(result.program.body) == 2
        assert result.gates_merged == 0

    def test_different_qubits_no_merge(self):
        """Rotations on different qubits do not merge."""
        prog = Main(
            q := QReg("q", 2),
            qb.RZ[0.5](q[0]),
            qb.RZ[0.3](q[1]),
        )

        ast = slr_to_ast(prog)
        result = RotationMergingPass().optimize(ast)

        assert len(result.program.body) == 2
        assert result.gates_merged == 0

    def test_interleaved_rotations_no_merge(self):
        """Interleaved rotations do not merge."""
        prog = Main(
            q := QReg("q", 1),
            qb.RZ[0.5](q[0]),
            qb.H(q[0]),  # Separates the RZ gates
            qb.RZ[0.3](q[0]),
        )

        ast = slr_to_ast(prog)
        result = RotationMergingPass().optimize(ast)

        assert len(result.program.body) == 3
        assert result.gates_merged == 0


class TestRotationMergingControlFlow:
    """Rotation merging inside control flow."""

    def test_merge_inside_if(self):
        """Rotations merge inside if statements."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            If(c[0] == 1).Then(
                qb.RZ[0.5](q[0]),
                qb.RZ[0.3](q[0]),
            ),
        )

        ast = slr_to_ast(prog)
        result = RotationMergingPass().optimize(ast)

        assert len(result.program.body) == 1  # IfStmt
        assert result.gates_merged == 1

    def test_merge_inside_repeat(self):
        """Rotations merge inside repeat blocks."""
        prog = Main(
            q := QReg("q", 1),
            Repeat(cond=3).block(
                qb.RX[0.1](q[0]),
                qb.RX[0.2](q[0]),
            ),
        )

        ast = slr_to_ast(prog)
        result = RotationMergingPass().optimize(ast)

        assert len(result.program.body) == 1  # RepeatStmt
        assert result.gates_merged == 1


class TestRotationMergingMultiple:
    """Multiple rotation merging tests."""

    def test_three_rotations_merge_to_one(self):
        """Three consecutive rotations merge to one (requires multiple passes)."""
        prog = Main(
            q := QReg("q", 1),
            qb.RZ[0.1](q[0]),
            qb.RZ[0.2](q[0]),
            qb.RZ[0.3](q[0]),
        )

        ast = slr_to_ast(prog)

        # First pass merges first two
        result1 = RotationMergingPass().optimize(ast)
        assert len(result1.program.body) == 2
        assert result1.gates_merged == 1

        # Second pass merges the result with third
        result2 = RotationMergingPass().optimize(result1.program)
        assert len(result2.program.body) == 1
        assert result2.gates_merged == 1

        # Final angle should be 0.6
        gate = result2.program.body[0]
        assert isinstance(gate, GateOp)
        assert isinstance(gate.params[0], LiteralExpr)
        assert abs(gate.params[0].value - 0.6) < 1e-10
