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

"""Tests for type checker validation."""

import math

from pecos.slr import CReg, If, Main, QReg, Repeat
from pecos.slr.ast import slr_to_ast
from pecos.slr.ast.nodes import (
    AllocatorDecl,
    BitRef,
    GateKind,
    GateOp,
    LiteralExpr,
    MeasureOp,
    Program,
    RegisterDecl,
    RepeatStmt,
    SlotRef,
)
from pecos.slr.ast.validation import TypeChecker, check_types
from pecos.slr.qeclib import qubit as qb


class TestTypeCheckerValid:
    """Tests for valid types."""

    def test_valid_gates(self) -> None:
        """Valid gate types."""
        prog = Main(
            q := QReg("q", 2),
            qb.H(q[0]),
            qb.X(q[1]),
            qb.CX(q[0], q[1]),
        )

        ast = slr_to_ast(prog)
        result = check_types(ast)

        assert result.valid is True
        assert len(result.errors) == 0

    def test_valid_rotation_gates(self) -> None:
        """Valid rotation gate with angle parameter."""
        prog = Main(
            q := QReg("q", 1),
            qb.RZ[0.5](q[0]),
            qb.RX[math.pi](q[0]),
        )

        ast = slr_to_ast(prog)
        result = check_types(ast)

        assert result.valid is True


class TestTypeCheckerArityErrors:
    """Tests for gate arity errors."""

    def test_single_qubit_gate_wrong_arity(self) -> None:
        """Single-qubit gate with wrong number of targets."""
        prog = Program(
            name="test",
            allocator=AllocatorDecl(name="q", capacity=2),
            body=(
                GateOp(
                    gate=GateKind.H,
                    targets=(
                        SlotRef(allocator="q", index=0),
                        SlotRef(allocator="q", index=1),  # H takes 1 qubit
                    ),
                ),
            ),
        )

        result = check_types(prog)

        assert result.valid is False
        assert "expects 1 qubit(s), got 2" in result.errors[0].message
        assert result.errors[0].code == "E201"

    def test_two_qubit_gate_wrong_arity(self) -> None:
        """Two-qubit gate with wrong number of targets."""
        prog = Program(
            name="test",
            allocator=AllocatorDecl(name="q", capacity=3),
            body=(
                GateOp(
                    gate=GateKind.CX,
                    targets=(SlotRef(allocator="q", index=0),),  # CX takes 2 qubits
                ),
            ),
        )

        result = check_types(prog)

        assert result.valid is False
        assert "expects 2 qubit(s), got 1" in result.errors[0].message


class TestTypeCheckerParameterErrors:
    """Tests for gate parameter errors."""

    def test_rotation_gate_missing_param(self) -> None:
        """Rotation gate without required parameter."""
        prog = Program(
            name="test",
            allocator=AllocatorDecl(name="q", capacity=1),
            body=(
                GateOp(
                    gate=GateKind.RZ,
                    targets=(SlotRef(allocator="q", index=0),),
                    params=(),  # RZ requires an angle
                ),
            ),
        )

        result = check_types(prog)

        assert result.valid is False
        assert "requires an angle parameter" in result.errors[0].message
        assert result.errors[0].code == "E202"

    def test_non_numeric_angle(self) -> None:
        """Rotation gate with non-numeric angle."""
        prog = Program(
            name="test",
            allocator=AllocatorDecl(name="q", capacity=1),
            body=(
                GateOp(
                    gate=GateKind.RZ,
                    targets=(SlotRef(allocator="q", index=0),),
                    params=(LiteralExpr(value="not-a-number"),),  # String, not numeric
                ),
            ),
        )

        result = check_types(prog)

        assert result.valid is False
        assert "Expected numeric value" in result.errors[0].message
        assert result.errors[0].code == "E203"

    def test_extra_params_warning(self) -> None:
        """Non-parameterized gate with parameters gives warning."""
        prog = Program(
            name="test",
            allocator=AllocatorDecl(name="q", capacity=1),
            body=(
                GateOp(
                    gate=GateKind.H,
                    targets=(SlotRef(allocator="q", index=0),),
                    params=(LiteralExpr(value=0.5),),  # H doesn't take params
                ),
            ),
        )

        result = check_types(prog)

        assert result.valid is True  # Warning, not error
        assert len(result.warnings) == 1
        assert "does not take parameters" in result.warnings[0].message


class TestTypeCheckerMeasurement:
    """Tests for measurement type checking."""

    def test_valid_measurement(self) -> None:
        """Valid measurement with matching targets and results."""
        prog = Program(
            name="test",
            allocator=AllocatorDecl(name="q", capacity=2),
            declarations=(RegisterDecl(name="c", size=2),),
            body=(
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

        result = check_types(prog)

        assert result.valid is True

    def test_mismatched_measurement_count(self) -> None:
        """Measurement with mismatched target and result count."""
        prog = Program(
            name="test",
            allocator=AllocatorDecl(name="q", capacity=2),
            declarations=(RegisterDecl(name="c", size=1),),
            body=(
                MeasureOp(
                    targets=(
                        SlotRef(allocator="q", index=0),
                        SlotRef(allocator="q", index=1),
                    ),
                    results=(BitRef(register="c", index=0),),  # Only 1 result for 2 qubits
                ),
            ),
        )

        result = check_types(prog)

        assert result.valid is False
        assert "2 qubit target(s) but 1 result" in result.errors[0].message


class TestTypeCheckerControlFlow:
    """Type checking in control flow."""

    def test_negative_repeat_count(self) -> None:
        """Negative repeat count."""
        prog = Program(
            name="test",
            allocator=AllocatorDecl(name="q", capacity=1),
            body=(
                RepeatStmt(
                    count=-5,
                    body=(
                        GateOp(
                            gate=GateKind.H,
                            targets=(SlotRef(allocator="q", index=0),),
                        ),
                    ),
                ),
            ),
        )

        result = check_types(prog)

        assert result.valid is False
        assert "cannot be negative" in result.errors[0].message

    def test_valid_if_statement(self) -> None:
        """Valid if statement."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            If(c[0] == 1).Then(
                qb.H(q[0]),
            ),
        )

        ast = slr_to_ast(prog)
        result = check_types(ast)

        assert result.valid is True


class TestTypeCheckerClass:
    """Tests for TypeChecker class."""

    def test_checker_reuse(self) -> None:
        """Checker can be reused."""
        checker = TypeChecker()

        prog1 = Main(q := QReg("q", 1), qb.H(q[0]))
        prog2 = Program(
            name="test",
            allocator=AllocatorDecl(name="q", capacity=2),
            body=(
                GateOp(
                    gate=GateKind.H,
                    targets=(
                        SlotRef(allocator="q", index=0),
                        SlotRef(allocator="q", index=1),
                    ),
                ),
            ),
        )

        ast1 = slr_to_ast(prog1)

        result1 = checker.validate(ast1)
        result2 = checker.validate(prog2)

        assert result1.valid is True
        assert result2.valid is False

    def test_passes_applied(self) -> None:
        """Pass name is tracked."""
        prog = Main(q := QReg("q", 1), qb.H(q[0]))
        ast = slr_to_ast(prog)

        result = check_types(ast)

        assert "type_checker" in result.passes_applied
