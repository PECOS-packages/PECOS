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

"""Tests for AST qubit state validator."""

import pytest

from pecos.slr import CReg, If, Main, QReg, Repeat
from pecos.slr.ast import slr_to_ast
from pecos.slr.ast.analysis import (
    AstQubitStateValidator,
    ValidationSlotState,
    validate_ast_qubit_states,
)
from pecos.slr.qeclib import qubit as qb


class TestQubitStateValidatorBasic:
    """Basic validation tests."""

    def test_gate_without_prep_fails_strict(self):
        """Gate on unprepared qubit should fail in strict mode."""
        prog = Main(
            q := QReg("q", 1),
            qb.H(q[0]),  # No prep before gate
        )
        ast = slr_to_ast(prog)

        violations = validate_ast_qubit_states(ast, strict=True)

        assert len(violations) == 1
        assert violations[0].allocator == "q"
        assert violations[0].index == 0
        assert "H" in violations[0].message

    def test_gate_without_prep_passes_nonstrict(self):
        """Gate on unprepared qubit should pass in non-strict mode."""
        prog = Main(
            q := QReg("q", 1),
            qb.H(q[0]),  # No prep, but non-strict assumes prepared
        )
        ast = slr_to_ast(prog)

        violations = validate_ast_qubit_states(ast, strict=False)

        assert len(violations) == 0

    def test_gate_with_prep_passes(self):
        """Gate on prepared qubit should pass."""
        prog = Main(
            q := QReg("q", 1),
            qb.Prep(q[0]),
            qb.H(q[0]),
        )
        ast = slr_to_ast(prog)

        violations = validate_ast_qubit_states(ast, strict=True)

        assert len(violations) == 0

    def test_multiple_gates_after_prep(self):
        """Multiple gates after prep should all pass."""
        prog = Main(
            q := QReg("q", 1),
            qb.Prep(q[0]),
            qb.H(q[0]),
            qb.X(q[0]),
            qb.Z(q[0]),
        )
        ast = slr_to_ast(prog)

        violations = validate_ast_qubit_states(ast, strict=True)

        assert len(violations) == 0


class TestQubitStateValidatorMeasurement:
    """Measurement state transition tests."""

    def test_gate_after_measure_fails(self):
        """Gate after measurement should fail (qubit becomes unprepared)."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            qb.Prep(q[0]),
            qb.H(q[0]),
            qb.Measure(q[0]) > c[0],
            qb.X(q[0]),  # Gate after measure
        )
        ast = slr_to_ast(prog)

        violations = validate_ast_qubit_states(ast, strict=True)

        assert len(violations) == 1
        assert violations[0].allocator == "q"
        assert "X" in violations[0].message

    def test_reprep_after_measure_passes(self):
        """Re-preparation after measurement allows gates."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            qb.Prep(q[0]),
            qb.H(q[0]),
            qb.Measure(q[0]) > c[0],
            qb.Prep(q[0]),  # Re-prep
            qb.X(q[0]),
        )
        ast = slr_to_ast(prog)

        violations = validate_ast_qubit_states(ast, strict=True)

        assert len(violations) == 0


class TestQubitStateValidatorMultiQubit:
    """Multi-qubit operation tests."""

    def test_two_qubit_gate_both_prepared(self):
        """Two-qubit gate with both qubits prepared should pass."""
        prog = Main(
            q := QReg("q", 2),
            qb.Prep(q[0]),
            qb.Prep(q[1]),
            qb.CX(q[0], q[1]),
        )
        ast = slr_to_ast(prog)

        violations = validate_ast_qubit_states(ast, strict=True)

        assert len(violations) == 0

    def test_two_qubit_gate_one_unprepared(self):
        """Two-qubit gate with one qubit unprepared should fail."""
        prog = Main(
            q := QReg("q", 2),
            qb.Prep(q[0]),
            # q[1] not prepared
            qb.CX(q[0], q[1]),
        )
        ast = slr_to_ast(prog)

        violations = validate_ast_qubit_states(ast, strict=True)

        assert len(violations) == 1
        assert violations[0].index == 1

    def test_two_qubit_gate_both_unprepared(self):
        """Two-qubit gate with both qubits unprepared should have two violations."""
        prog = Main(
            q := QReg("q", 2),
            qb.CX(q[0], q[1]),
        )
        ast = slr_to_ast(prog)

        violations = validate_ast_qubit_states(ast, strict=True)

        assert len(violations) == 2


class TestQubitStateValidatorControlFlow:
    """Control flow tests."""

    def test_if_branch_prep_not_sufficient(self):
        """Prep in only one branch is not sufficient."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            If(c[0] == 1).Then(
                qb.Prep(q[0]),
            ),
            # After if, q[0] might not be prepared (else branch doesn't prep)
            qb.H(q[0]),
        )
        ast = slr_to_ast(prog)

        violations = validate_ast_qubit_states(ast, strict=True)

        assert len(violations) == 1

    def test_if_both_branches_prep(self):
        """Prep in both branches is sufficient."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            If(c[0] == 1).Then(
                qb.Prep(q[0]),
            ).Else(
                qb.Prep(q[0]),
            ),
            qb.H(q[0]),  # Safe - prepared in both branches
        )
        ast = slr_to_ast(prog)

        violations = validate_ast_qubit_states(ast, strict=True)

        assert len(violations) == 0

    def test_if_body_uses_prep_from_before(self):
        """If body can use prep from before the if."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            qb.Prep(q[0]),
            If(c[0] == 1).Then(
                qb.H(q[0]),
            ),
        )
        ast = slr_to_ast(prog)

        violations = validate_ast_qubit_states(ast, strict=True)

        assert len(violations) == 0

    def test_repeat_uses_prep_from_before(self):
        """Repeat body can use prep from before."""
        prog = Main(
            q := QReg("q", 1),
            qb.Prep(q[0]),
            Repeat(cond=3).block(
                qb.H(q[0]),
            ),
        )
        ast = slr_to_ast(prog)

        violations = validate_ast_qubit_states(ast, strict=True)

        assert len(violations) == 0


class TestQubitStateValidatorQEC:
    """QEC pattern tests."""

    def test_syndrome_extraction_pattern(self):
        """Standard syndrome extraction pattern should pass."""
        prog = Main(
            data := QReg("data", 2),
            ancilla := QReg("ancilla", 1),
            c := CReg("c", 1),
            # Prep data qubits
            qb.Prep(data[0]),
            qb.Prep(data[1]),
            # Prep ancilla
            qb.Prep(ancilla[0]),
            # Syndrome extraction
            qb.CX(data[0], ancilla[0]),
            qb.CX(data[1], ancilla[0]),
            # Measure ancilla
            qb.Measure(ancilla[0]) > c[0],
        )
        ast = slr_to_ast(prog)

        violations = validate_ast_qubit_states(ast, strict=True)

        assert len(violations) == 0

    def test_repeated_syndrome_extraction(self):
        """Repeated syndrome extraction needs re-prep of ancilla."""
        prog = Main(
            data := QReg("data", 2),
            ancilla := QReg("ancilla", 1),
            c := CReg("c", 1),
            # Prep everything
            qb.Prep(data[0]),
            qb.Prep(data[1]),
            qb.Prep(ancilla[0]),
            # First round
            qb.CX(data[0], ancilla[0]),
            qb.CX(data[1], ancilla[0]),
            qb.Measure(ancilla[0]) > c[0],
            # Second round - need to re-prep ancilla
            qb.Prep(ancilla[0]),
            qb.CX(data[0], ancilla[0]),
            qb.CX(data[1], ancilla[0]),
            qb.Measure(ancilla[0]) > c[0],
        )
        ast = slr_to_ast(prog)

        violations = validate_ast_qubit_states(ast, strict=True)

        assert len(violations) == 0


class TestValidatorClass:
    """Tests for the validator class itself."""

    def test_validator_reusable(self):
        """Validator can be reused for multiple programs."""
        validator = AstQubitStateValidator(strict=True)

        prog1 = Main(
            q := QReg("q", 1),
            qb.H(q[0]),  # Violation
        )
        ast1 = slr_to_ast(prog1)

        prog2 = Main(
            q := QReg("q", 1),
            qb.Prep(q[0]),
            qb.H(q[0]),  # No violation
        )
        ast2 = slr_to_ast(prog2)

        violations1 = validator.validate(ast1)
        violations2 = validator.validate(ast2)

        assert len(violations1) == 1
        assert len(violations2) == 0

    def test_violation_string_representation(self):
        """Violation should have a useful string representation."""
        prog = Main(
            q := QReg("q", 1),
            qb.H(q[0]),
        )
        ast = slr_to_ast(prog)

        violations = validate_ast_qubit_states(ast, strict=True)

        assert len(violations) == 1
        violation_str = str(violations[0])
        assert "q[0]" in violation_str
        assert "H" in violation_str
        assert "unprepared" in violation_str.lower()
