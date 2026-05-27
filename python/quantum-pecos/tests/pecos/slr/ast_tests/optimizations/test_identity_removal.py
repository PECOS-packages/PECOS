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

"""Tests for identity removal optimization pass."""

import math

from pecos.slr import CReg, If, Main, QReg, Repeat, rad
from pecos.slr.ast import slr_to_ast
from pecos.slr.ast.optimizations import IdentityRemovalPass
from pecos.slr.qeclib import qubit as qb


class TestIdentityRemovalBasic:
    """Basic identity removal tests."""

    def test_rz_zero_removed(self) -> None:
        """RZ(0) is removed."""
        prog = Main(
            q := QReg("q", 1),
            qb.RZ(rad(0), q[0]),
        )

        ast = slr_to_ast(prog)
        result = IdentityRemovalPass().optimize(ast)

        assert len(result.program.body) == 0
        assert result.gates_removed == 1

    def test_rx_zero_removed(self) -> None:
        """RX(0) is removed."""
        prog = Main(
            q := QReg("q", 1),
            qb.RX(rad(0), q[0]),
        )

        ast = slr_to_ast(prog)
        result = IdentityRemovalPass().optimize(ast)

        assert len(result.program.body) == 0
        assert result.gates_removed == 1

    def test_ry_zero_removed(self) -> None:
        """RY(0) is removed."""
        prog = Main(
            q := QReg("q", 1),
            qb.RY(rad(0), q[0]),
        )

        ast = slr_to_ast(prog)
        result = IdentityRemovalPass().optimize(ast)

        assert len(result.program.body) == 0
        assert result.gates_removed == 1

    def test_rz_2pi_removed(self) -> None:
        """RZ(2*pi) is removed."""
        prog = Main(
            q := QReg("q", 1),
            qb.RZ(rad(2 * math.pi), q[0]),
        )

        ast = slr_to_ast(prog)
        result = IdentityRemovalPass().optimize(ast)

        assert len(result.program.body) == 0
        assert result.gates_removed == 1

    def test_rz_4pi_removed(self) -> None:
        """RZ(4*pi) is removed (multiple of 2*pi)."""
        prog = Main(
            q := QReg("q", 1),
            qb.RZ(rad(4 * math.pi), q[0]),
        )

        ast = slr_to_ast(prog)
        result = IdentityRemovalPass().optimize(ast)

        assert len(result.program.body) == 0
        assert result.gates_removed == 1


class TestIdentityRemovalNoRemove:
    """Tests where gates should NOT be removed."""

    def test_rz_nonzero_not_removed(self) -> None:
        """RZ(0.5) is not removed."""
        prog = Main(
            q := QReg("q", 1),
            qb.RZ(rad(0.5), q[0]),
        )

        ast = slr_to_ast(prog)
        result = IdentityRemovalPass().optimize(ast)

        assert len(result.program.body) == 1
        assert result.gates_removed == 0

    def test_rz_pi_not_removed(self) -> None:
        """RZ(pi) is not removed."""
        prog = Main(
            q := QReg("q", 1),
            qb.RZ(rad(math.pi), q[0]),
        )

        ast = slr_to_ast(prog)
        result = IdentityRemovalPass().optimize(ast)

        assert len(result.program.body) == 1
        assert result.gates_removed == 0

    def test_non_rotation_not_removed(self) -> None:
        """Non-rotation gates are not affected."""
        prog = Main(
            q := QReg("q", 1),
            qb.H(q[0]),
            qb.X(q[0]),
        )

        ast = slr_to_ast(prog)
        result = IdentityRemovalPass().optimize(ast)

        assert len(result.program.body) == 2
        assert result.gates_removed == 0


class TestIdentityRemovalControlFlow:
    """Identity removal inside control flow."""

    def test_removal_inside_if(self) -> None:
        """Identity gates removed inside if statements."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            If(c[0] == 1).Then(
                qb.RZ(rad(0), q[0]),
                qb.H(q[0]),
            ),
        )

        ast = slr_to_ast(prog)
        result = IdentityRemovalPass().optimize(ast)

        assert len(result.program.body) == 1  # IfStmt
        assert result.gates_removed == 1

    def test_removal_inside_repeat(self) -> None:
        """Identity gates removed inside repeat blocks."""
        prog = Main(
            q := QReg("q", 1),
            Repeat(cond=3).block(
                qb.RX(rad(0), q[0]),
            ),
        )

        ast = slr_to_ast(prog)
        result = IdentityRemovalPass().optimize(ast)

        assert len(result.program.body) == 1  # RepeatStmt
        assert result.gates_removed == 1


class TestIdentityRemovalMultiple:
    """Multiple identity removal tests."""

    def test_multiple_identity_gates(self) -> None:
        """Multiple identity gates are removed."""
        prog = Main(
            q := QReg("q", 1),
            qb.RZ(rad(0), q[0]),
            qb.H(q[0]),
            qb.RX(rad(0), q[0]),
            qb.RY(rad(2 * math.pi), q[0]),
        )

        ast = slr_to_ast(prog)
        result = IdentityRemovalPass().optimize(ast)

        assert len(result.program.body) == 1  # Only H remains
        assert result.gates_removed == 3

    def test_mixed_with_nonidentity(self) -> None:
        """Identity gates removed among non-identity gates."""
        prog = Main(
            q := QReg("q", 1),
            qb.RZ(rad(0), q[0]),  # Removed
            qb.RZ(rad(0.5), q[0]),  # Kept
            qb.RX(rad(0), q[0]),  # Removed
            qb.RX(rad(0.5), q[0]),  # Kept
        )

        ast = slr_to_ast(prog)
        result = IdentityRemovalPass().optimize(ast)

        assert len(result.program.body) == 2
        assert result.gates_removed == 2
