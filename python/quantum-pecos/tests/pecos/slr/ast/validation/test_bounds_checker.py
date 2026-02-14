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

"""Tests for bounds checker validation."""

from pecos.slr import CReg, If, Main, QReg, Repeat
from pecos.slr.ast import slr_to_ast
from pecos.slr.ast.nodes import (
    AllocatorDecl,
    BitRef,
    GateKind,
    GateOp,
    MeasureOp,
    Program,
    RegisterDecl,
    SlotRef,
)
from pecos.slr.ast.validation import BoundsChecker, check_bounds
from pecos.slr.qeclib import qubit as qb


class TestBoundsCheckerValid:
    """Tests for valid bounds."""

    def test_valid_qubit_indices(self) -> None:
        """Valid qubit indices pass."""
        prog = Main(
            q := QReg("q", 3),
            qb.H(q[0]),
            qb.X(q[1]),
            qb.Y(q[2]),
        )

        ast = slr_to_ast(prog)
        result = check_bounds(ast)

        assert result.valid is True
        assert len(result.errors) == 0

    def test_valid_two_qubit_gates(self) -> None:
        """Valid two-qubit gate indices."""
        prog = Main(
            q := QReg("q", 2),
            qb.CX(q[0], q[1]),
        )

        ast = slr_to_ast(prog)
        result = check_bounds(ast)

        assert result.valid is True


class TestBoundsCheckerQubitErrors:
    """Tests for qubit bound errors."""

    def test_qubit_index_out_of_bounds(self) -> None:
        """Qubit index exceeds capacity."""
        # Create AST directly with out-of-bounds index
        prog = Program(
            name="test",
            allocator=AllocatorDecl(name="q", capacity=2),
            body=(
                GateOp(
                    gate=GateKind.H,
                    targets=(SlotRef(allocator="q", index=5),),  # Out of bounds
                ),
            ),
        )

        result = check_bounds(prog)

        assert result.valid is False
        assert len(result.errors) == 1
        assert "out of bounds" in result.errors[0].message
        assert result.errors[0].code == "E103"

    def test_negative_qubit_index(self) -> None:
        """Negative qubit index."""
        prog = Program(
            name="test",
            allocator=AllocatorDecl(name="q", capacity=2),
            body=(
                GateOp(
                    gate=GateKind.X,
                    targets=(SlotRef(allocator="q", index=-1),),
                ),
            ),
        )

        result = check_bounds(prog)

        assert result.valid is False
        assert "Negative" in result.errors[0].message
        assert result.errors[0].code == "E102"

    def test_unknown_allocator(self) -> None:
        """Reference to unknown allocator."""
        prog = Program(
            name="test",
            allocator=AllocatorDecl(name="q", capacity=2),
            body=(
                GateOp(
                    gate=GateKind.H,
                    targets=(SlotRef(allocator="unknown", index=0),),
                ),
            ),
        )

        result = check_bounds(prog)

        assert result.valid is False
        assert "Unknown allocator" in result.errors[0].message
        assert result.errors[0].code == "E101"


class TestBoundsCheckerBitErrors:
    """Tests for classical bit bound errors."""

    def test_bit_index_out_of_bounds(self) -> None:
        """Classical bit index exceeds register size."""
        prog = Program(
            name="test",
            allocator=AllocatorDecl(name="q", capacity=2),
            declarations=(RegisterDecl(name="c", size=2),),
            body=(
                MeasureOp(
                    targets=(SlotRef(allocator="q", index=0),),
                    results=(BitRef(register="c", index=5),),  # Out of bounds
                ),
            ),
        )

        result = check_bounds(prog)

        assert result.valid is False
        assert "out of bounds" in result.errors[0].message
        assert result.errors[0].code == "E106"

    def test_unknown_register(self) -> None:
        """Reference to unknown register."""
        prog = Program(
            name="test",
            allocator=AllocatorDecl(name="q", capacity=2),
            body=(
                MeasureOp(
                    targets=(SlotRef(allocator="q", index=0),),
                    results=(BitRef(register="unknown", index=0),),
                ),
            ),
        )

        result = check_bounds(prog)

        assert result.valid is False
        assert "Unknown register" in result.errors[0].message
        assert result.errors[0].code == "E104"


class TestBoundsCheckerControlFlow:
    """Bounds checking in control flow."""

    def test_bounds_inside_if(self) -> None:
        """Bounds checked inside if statements."""
        prog = Main(
            q := QReg("q", 2),
            c := CReg("c", 1),
            If(c[0] == 1).Then(
                qb.H(q[0]),
                qb.H(q[1]),
            ),
        )

        ast = slr_to_ast(prog)
        result = check_bounds(ast)

        assert result.valid is True

    def test_bounds_inside_repeat(self) -> None:
        """Bounds checked inside repeat loops."""
        prog = Main(
            q := QReg("q", 2),
            Repeat(cond=3).block(
                qb.CX(q[0], q[1]),
            ),
        )

        ast = slr_to_ast(prog)
        result = check_bounds(ast)

        assert result.valid is True


class TestBoundsCheckerClass:
    """Tests for BoundsChecker class."""

    def test_checker_reuse(self) -> None:
        """Checker can be reused."""
        checker = BoundsChecker()

        prog1 = Program(
            name="test1",
            allocator=AllocatorDecl(name="q", capacity=2),
            body=(GateOp(gate=GateKind.H, targets=(SlotRef(allocator="q", index=0),)),),
        )

        prog2 = Program(
            name="test2",
            allocator=AllocatorDecl(name="q", capacity=1),
            body=(GateOp(gate=GateKind.H, targets=(SlotRef(allocator="q", index=5),)),),
        )

        result1 = checker.validate(prog1)
        result2 = checker.validate(prog2)

        assert result1.valid is True
        assert result2.valid is False

    def test_passes_applied(self) -> None:
        """Pass name is tracked."""
        prog = Main(q := QReg("q", 1), qb.H(q[0]))
        ast = slr_to_ast(prog)

        result = check_bounds(ast)

        assert "bounds_checker" in result.passes_applied
