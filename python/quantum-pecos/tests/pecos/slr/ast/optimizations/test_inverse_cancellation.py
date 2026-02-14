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

"""Tests for inverse cancellation optimization pass."""

from pecos.slr import CReg, If, Main, QReg, Repeat
from pecos.slr.ast import slr_to_ast
from pecos.slr.ast.optimizations import InverseCancellationPass
from pecos.slr.qeclib import qubit as qb


class TestInverseCancellationBasic:
    """Basic inverse cancellation tests."""

    def test_s_sdg_cancels(self):
        """S-Sdg on same qubit cancels."""
        prog = Main(
            q := QReg("q", 1),
            qb.SZ(q[0]),  # S gate
            qb.SZdg(q[0]),  # Sdg gate
        )

        ast = slr_to_ast(prog)
        result = InverseCancellationPass().optimize(ast)

        assert len(result.program.body) == 0
        assert result.gates_removed == 2

    def test_sdg_s_cancels(self):
        """Sdg-S on same qubit cancels (order reversed)."""
        prog = Main(
            q := QReg("q", 1),
            qb.SZdg(q[0]),  # Sdg gate
            qb.SZ(q[0]),  # S gate
        )

        ast = slr_to_ast(prog)
        result = InverseCancellationPass().optimize(ast)

        assert len(result.program.body) == 0
        assert result.gates_removed == 2

    def test_t_tdg_cancels(self):
        """T-Tdg on same qubit cancels."""
        prog = Main(
            q := QReg("q", 1),
            qb.T(q[0]),
            qb.Tdg(q[0]),
        )

        ast = slr_to_ast(prog)
        result = InverseCancellationPass().optimize(ast)

        assert len(result.program.body) == 0
        assert result.gates_removed == 2

    def test_tdg_t_cancels(self):
        """Tdg-T on same qubit cancels."""
        prog = Main(
            q := QReg("q", 1),
            qb.Tdg(q[0]),
            qb.T(q[0]),
        )

        ast = slr_to_ast(prog)
        result = InverseCancellationPass().optimize(ast)

        assert len(result.program.body) == 0
        assert result.gates_removed == 2


class TestInverseCancellationSqrt:
    """Square root gate inverse cancellation tests."""

    def test_sx_sxdg_cancels(self):
        """SX-SXdg on same qubit cancels."""
        prog = Main(
            q := QReg("q", 1),
            qb.SX(q[0]),
            qb.SXdg(q[0]),
        )

        ast = slr_to_ast(prog)
        result = InverseCancellationPass().optimize(ast)

        assert len(result.program.body) == 0
        assert result.gates_removed == 2

    def test_sy_sydg_cancels(self):
        """SY-SYdg on same qubit cancels."""
        prog = Main(
            q := QReg("q", 1),
            qb.SY(q[0]),
            qb.SYdg(q[0]),
        )

        ast = slr_to_ast(prog)
        result = InverseCancellationPass().optimize(ast)

        assert len(result.program.body) == 0
        assert result.gates_removed == 2


class TestInverseCancellationNoCancel:
    """Tests where inverse pairs should NOT cancel."""

    def test_s_t_no_cancel(self):
        """S-T are not inverses."""
        prog = Main(
            q := QReg("q", 1),
            qb.SZ(q[0]),  # S gate
            qb.T(q[0]),
        )

        ast = slr_to_ast(prog)
        result = InverseCancellationPass().optimize(ast)

        assert len(result.program.body) == 2
        assert result.gates_removed == 0

    def test_s_sdg_different_qubits_no_cancel(self):
        """S-Sdg on different qubits does not cancel."""
        prog = Main(
            q := QReg("q", 2),
            qb.SZ(q[0]),
            qb.SZdg(q[1]),
        )

        ast = slr_to_ast(prog)
        result = InverseCancellationPass().optimize(ast)

        assert len(result.program.body) == 2
        assert result.gates_removed == 0


class TestInverseCancellationControlFlow:
    """Inverse cancellation inside control flow."""

    def test_cancellation_inside_if(self):
        """Inverse pairs cancel inside if statements."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            If(c[0] == 1).Then(
                qb.SZ(q[0]),
                qb.SZdg(q[0]),
            ),
        )

        ast = slr_to_ast(prog)
        result = InverseCancellationPass().optimize(ast)

        assert len(result.program.body) == 1  # IfStmt still present
        assert result.gates_removed == 2

    def test_cancellation_inside_repeat(self):
        """Inverse pairs cancel inside repeat blocks."""
        prog = Main(
            q := QReg("q", 1),
            Repeat(cond=5).block(
                qb.T(q[0]),
                qb.Tdg(q[0]),
            ),
        )

        ast = slr_to_ast(prog)
        result = InverseCancellationPass().optimize(ast)

        assert len(result.program.body) == 1  # RepeatStmt still present
        assert result.gates_removed == 2


class TestInverseCancellationMultiple:
    """Multiple inverse cancellation tests."""

    def test_multiple_inverse_pairs(self):
        """Multiple inverse pairs cancel correctly."""
        prog = Main(
            q := QReg("q", 1),
            qb.SZ(q[0]),
            qb.SZdg(q[0]),
            qb.T(q[0]),
            qb.Tdg(q[0]),
        )

        ast = slr_to_ast(prog)
        result = InverseCancellationPass().optimize(ast)

        assert len(result.program.body) == 0
        assert result.gates_removed == 4

    def test_interleaved_no_cancel(self):
        """Interleaved inverse pairs do not cancel."""
        prog = Main(
            q := QReg("q", 1),
            qb.SZ(q[0]),
            qb.T(q[0]),
            qb.SZdg(q[0]),  # Not consecutive with S
        )

        ast = slr_to_ast(prog)
        result = InverseCancellationPass().optimize(ast)

        assert len(result.program.body) == 3
        assert result.gates_removed == 0
