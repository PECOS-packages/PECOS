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

"""Tests for gate cancellation optimization pass."""

from pecos.slr import CReg, If, Main, QReg, Repeat
from pecos.slr.ast import slr_to_ast
from pecos.slr.ast.nodes import GateKind, GateOp
from pecos.slr.ast.optimizations import GateCancellationPass
from pecos.slr.qeclib import qubit as qb


class TestGateCancellationBasic:
    """Basic gate cancellation tests."""

    def test_x_x_cancels(self):
        """X-X on same qubit cancels."""
        prog = Main(
            q := QReg("q", 1),
            qb.X(q[0]),
            qb.X(q[0]),
        )

        ast = slr_to_ast(prog)
        result = GateCancellationPass().optimize(ast)

        assert len(result.program.body) == 0
        assert result.gates_removed == 2

    def test_h_h_cancels(self):
        """H-H on same qubit cancels."""
        prog = Main(
            q := QReg("q", 1),
            qb.H(q[0]),
            qb.H(q[0]),
        )

        ast = slr_to_ast(prog)
        result = GateCancellationPass().optimize(ast)

        assert len(result.program.body) == 0
        assert result.gates_removed == 2

    def test_y_y_cancels(self):
        """Y-Y on same qubit cancels."""
        prog = Main(
            q := QReg("q", 1),
            qb.Y(q[0]),
            qb.Y(q[0]),
        )

        ast = slr_to_ast(prog)
        result = GateCancellationPass().optimize(ast)

        assert len(result.program.body) == 0
        assert result.gates_removed == 2

    def test_z_z_cancels(self):
        """Z-Z on same qubit cancels."""
        prog = Main(
            q := QReg("q", 1),
            qb.Z(q[0]),
            qb.Z(q[0]),
        )

        ast = slr_to_ast(prog)
        result = GateCancellationPass().optimize(ast)

        assert len(result.program.body) == 0
        assert result.gates_removed == 2


class TestGateCancellationTwoQubit:
    """Two-qubit gate cancellation tests."""

    def test_cx_cx_cancels(self):
        """CX-CX on same qubits cancels."""
        prog = Main(
            q := QReg("q", 2),
            qb.CX(q[0], q[1]),
            qb.CX(q[0], q[1]),
        )

        ast = slr_to_ast(prog)
        result = GateCancellationPass().optimize(ast)

        assert len(result.program.body) == 0
        assert result.gates_removed == 2

    def test_cz_cz_cancels(self):
        """CZ-CZ on same qubits cancels."""
        prog = Main(
            q := QReg("q", 2),
            qb.CZ(q[0], q[1]),
            qb.CZ(q[0], q[1]),
        )

        ast = slr_to_ast(prog)
        result = GateCancellationPass().optimize(ast)

        assert len(result.program.body) == 0
        assert result.gates_removed == 2

    def test_cx_different_order_no_cancel(self):
        """CX with swapped control/target does not cancel."""
        prog = Main(
            q := QReg("q", 2),
            qb.CX(q[0], q[1]),
            qb.CX(q[1], q[0]),  # Different order
        )

        ast = slr_to_ast(prog)
        result = GateCancellationPass().optimize(ast)

        assert len(result.program.body) == 2
        assert result.gates_removed == 0


class TestGateCancellationNoCancel:
    """Tests where gates should NOT cancel."""

    def test_x_x_different_qubits_no_cancel(self):
        """X-X on different qubits does not cancel."""
        prog = Main(
            q := QReg("q", 2),
            qb.X(q[0]),
            qb.X(q[1]),
        )

        ast = slr_to_ast(prog)
        result = GateCancellationPass().optimize(ast)

        assert len(result.program.body) == 2
        assert result.gates_removed == 0

    def test_x_h_no_cancel(self):
        """Different gate types do not cancel."""
        prog = Main(
            q := QReg("q", 1),
            qb.X(q[0]),
            qb.H(q[0]),
        )

        ast = slr_to_ast(prog)
        result = GateCancellationPass().optimize(ast)

        assert len(result.program.body) == 2
        assert result.gates_removed == 0

    def test_non_self_inverse_no_cancel(self):
        """Non-self-inverse gates (S, T) do not cancel with themselves."""
        prog = Main(
            q := QReg("q", 1),
            qb.SZ(q[0]),  # S gate
            qb.SZ(q[0]),  # S gate (S*S != I)
        )

        ast = slr_to_ast(prog)
        result = GateCancellationPass().optimize(ast)

        assert len(result.program.body) == 2
        assert result.gates_removed == 0


class TestGateCancellationControlFlow:
    """Gate cancellation inside control flow."""

    def test_cancellation_inside_if(self):
        """Gates cancel inside if statements."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            If(c[0] == 1).Then(
                qb.X(q[0]),
                qb.X(q[0]),
            ),
        )

        ast = slr_to_ast(prog)
        result = GateCancellationPass().optimize(ast)

        # The if statement remains, but its body is empty
        assert len(result.program.body) == 1  # IfStmt still present
        assert result.gates_removed == 2

    def test_cancellation_inside_repeat(self):
        """Gates cancel inside repeat blocks."""
        prog = Main(
            q := QReg("q", 1),
            Repeat(cond=3).block(
                qb.H(q[0]),
                qb.H(q[0]),
            ),
        )

        ast = slr_to_ast(prog)
        result = GateCancellationPass().optimize(ast)

        assert len(result.program.body) == 1  # RepeatStmt still present
        assert result.gates_removed == 2

    def test_no_cancel_across_control_flow(self):
        """Gates do not cancel across control flow boundaries."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            qb.X(q[0]),
            If(c[0] == 1).Then(
                qb.H(q[0]),
            ),
            qb.X(q[0]),  # Cannot cancel with first X
        )

        ast = slr_to_ast(prog)
        result = GateCancellationPass().optimize(ast)

        # Both X gates and the If should remain
        assert result.gates_removed == 0


class TestGateCancellationMultiple:
    """Multiple cancellation tests."""

    def test_multiple_cancellations(self):
        """Multiple pairs cancel correctly."""
        prog = Main(
            q := QReg("q", 1),
            qb.X(q[0]),
            qb.X(q[0]),
            qb.H(q[0]),
            qb.H(q[0]),
        )

        ast = slr_to_ast(prog)
        result = GateCancellationPass().optimize(ast)

        assert len(result.program.body) == 0
        assert result.gates_removed == 4

    def test_interleaved_no_cancel(self):
        """Interleaved gates on same qubit do not cancel."""
        prog = Main(
            q := QReg("q", 1),
            qb.X(q[0]),
            qb.H(q[0]),
            qb.X(q[0]),  # Not consecutive with first X
        )

        ast = slr_to_ast(prog)
        result = GateCancellationPass().optimize(ast)

        assert len(result.program.body) == 3
        assert result.gates_removed == 0

    def test_four_of_same_gate(self):
        """Four of the same gate reduces to zero."""
        prog = Main(
            q := QReg("q", 1),
            qb.X(q[0]),
            qb.X(q[0]),
            qb.X(q[0]),
            qb.X(q[0]),
        )

        ast = slr_to_ast(prog)
        result = GateCancellationPass().optimize(ast)

        assert len(result.program.body) == 0
        assert result.gates_removed == 4

    def test_three_of_same_gate(self):
        """Three of the same gate leaves one."""
        prog = Main(
            q := QReg("q", 1),
            qb.X(q[0]),
            qb.X(q[0]),
            qb.X(q[0]),
        )

        ast = slr_to_ast(prog)
        result = GateCancellationPass().optimize(ast)

        assert len(result.program.body) == 1
        assert result.gates_removed == 2
        # Remaining gate should be X
        remaining = result.program.body[0]
        assert isinstance(remaining, GateOp)
        assert remaining.gate == GateKind.X
