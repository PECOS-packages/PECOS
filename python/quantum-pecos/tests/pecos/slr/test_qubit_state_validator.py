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

"""Tests for QubitStateValidator - compile-time detection of unprepared qubit usage."""

import pytest

from pecos.slr import CReg, If, Main, QReg
from pecos.slr.gen_codes.guppy.qubit_state_validator import (
    QubitStateValidator,
    StateViolation,
    validate_qubit_states,
)
from pecos.slr.qeclib import qubit as qb


class TestQubitStateValidatorBasic:
    """Basic validation tests."""

    def test_gate_without_prep_strict_mode(self):
        """In strict mode, gate without prep is a violation."""
        prog = Main(
            q := QReg("q", 2),
            qb.H(q[0]),  # No prep before H - should be violation
        )

        violations = validate_qubit_states(prog, strict=True)

        assert len(violations) == 1
        assert violations[0].array_name == "q"
        assert violations[0].index == 0
        assert "H" in violations[0].gate_name
        assert "unprepared" in violations[0].message.lower()

    def test_gate_without_prep_non_strict_mode(self):
        """In non-strict mode (legacy), qubits start prepared."""
        prog = Main(
            q := QReg("q", 2),
            qb.H(q[0]),  # No prep but non-strict mode - OK
        )

        variable_context = {"q": prog.vars.get("q")}
        violations = validate_qubit_states(prog, variable_context, strict=False)

        assert len(violations) == 0

    def test_prep_then_gate_is_valid(self):
        """Prep followed by gate is valid."""
        prog = Main(
            q := QReg("q", 2),
            qb.Prep(q[0]),
            qb.H(q[0]),  # Prep before H - valid
        )

        violations = validate_qubit_states(prog, strict=True)

        assert len(violations) == 0

    def test_measure_then_gate_is_violation(self):
        """Gate after measurement without re-prep is violation."""
        prog = Main(
            q := QReg("q", 2),
            c := CReg("c", 2),
            qb.Prep(q[0]),
            qb.Measure(q[0]) > c[0],
            qb.H(q[0]),  # After measure, no re-prep - violation
        )

        violations = validate_qubit_states(prog, strict=True)

        assert len(violations) == 1
        assert violations[0].array_name == "q"
        assert violations[0].index == 0

    def test_measure_reprep_gate_is_valid(self):
        """Measure, re-prep, then gate is valid."""
        prog = Main(
            q := QReg("q", 2),
            c := CReg("c", 2),
            qb.Prep(q[0]),
            qb.Measure(q[0]) > c[0],
            qb.Prep(q[0]),  # Re-prep after measure
            qb.H(q[0]),  # Now valid
        )

        violations = validate_qubit_states(prog, strict=True)

        assert len(violations) == 0


class TestQubitStateValidatorMultiQubit:
    """Tests with multiple qubits."""

    def test_independent_qubit_tracking(self):
        """Each qubit's state is tracked independently."""
        prog = Main(
            q := QReg("q", 3),
            qb.Prep(q[0]),
            qb.Prep(q[1]),
            # q[2] not prepared
            qb.H(q[0]),  # OK
            qb.H(q[1]),  # OK
            qb.H(q[2]),  # Violation - not prepared
        )

        violations = validate_qubit_states(prog, strict=True)

        assert len(violations) == 1
        assert violations[0].index == 2

    def test_two_qubit_gate_both_prepared(self):
        """Two-qubit gate requires both qubits prepared."""
        prog = Main(
            q := QReg("q", 2),
            qb.Prep(q[0]),
            qb.Prep(q[1]),
            qb.CX(q[0], q[1]),  # Both prepared - valid
        )

        violations = validate_qubit_states(prog, strict=True)

        assert len(violations) == 0

    def test_two_qubit_gate_one_unprepared(self):
        """Two-qubit gate with one unprepared qubit is violation."""
        prog = Main(
            q := QReg("q", 2),
            qb.Prep(q[0]),
            # q[1] not prepared
            qb.CX(q[0], q[1]),  # q[1] unprepared - violation
        )

        violations = validate_qubit_states(prog, strict=True)

        assert len(violations) == 1
        assert violations[0].index == 1

    def test_two_qubit_gate_both_unprepared(self):
        """Two-qubit gate with both unprepared is two violations."""
        prog = Main(
            q := QReg("q", 2),
            qb.CX(q[0], q[1]),  # Both unprepared - two violations
        )

        violations = validate_qubit_states(prog, strict=True)

        assert len(violations) == 2


class TestQubitStateValidatorConditionals:
    """Tests with conditional blocks."""

    def test_if_block_both_branches_prepare(self):
        """If both branches prepare, qubit is prepared after."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            If(c[0] == 1).Then(
                qb.Prep(q[0]),
            ).Else(
                qb.Prep(q[0]),
            ),
            qb.H(q[0]),  # Prepared in both branches - valid
        )

        variable_context = {"q": prog.vars.get("q"), "c": prog.vars.get("c")}
        violations = validate_qubit_states(prog, variable_context, strict=True)

        assert len(violations) == 0

    def test_if_block_only_then_prepares(self):
        """If only then branch prepares, qubit may be unprepared after."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            If(c[0] == 1).Then(
                qb.Prep(q[0]),
            ),
            # No else - q[0] may not be prepared
            qb.H(q[0]),  # May be unprepared - violation
        )

        variable_context = {"q": prog.vars.get("q"), "c": prog.vars.get("c")}
        violations = validate_qubit_states(prog, variable_context, strict=True)

        # Should detect violation - qubit not prepared in else branch
        assert len(violations) >= 1


class TestQubitStateValidatorQECPattern:
    """Tests for typical QEC patterns."""

    def test_syndrome_extraction_pattern(self):
        """Typical syndrome extraction: prep, use, measure, re-prep cycle."""
        prog = Main(
            data := QReg("data", 2),
            ancilla := QReg("ancilla", 1),
            c := CReg("c", 1),
            # Initialize data qubits
            qb.Prep(data[0]),
            qb.Prep(data[1]),
            # Syndrome extraction round 1
            qb.Prep(ancilla[0]),
            qb.CX(data[0], ancilla[0]),
            qb.CX(data[1], ancilla[0]),
            qb.Measure(ancilla[0]) > c[0],
            # Syndrome extraction round 2
            qb.Prep(ancilla[0]),  # Re-prep after measure
            qb.CX(data[0], ancilla[0]),
            qb.CX(data[1], ancilla[0]),
            qb.Measure(ancilla[0]) > c[0],
        )

        violations = validate_qubit_states(prog, strict=True)

        assert len(violations) == 0

    def test_missing_reprep_in_qec_cycle(self):
        """Detect missing re-prep in QEC cycle."""
        prog = Main(
            data := QReg("data", 1),
            ancilla := QReg("ancilla", 1),
            c := CReg("c", 1),
            # Initialize
            qb.Prep(data[0]),
            qb.Prep(ancilla[0]),
            # Round 1
            qb.CX(data[0], ancilla[0]),
            qb.Measure(ancilla[0]) > c[0],
            # Round 2 - MISSING re-prep of ancilla
            qb.CX(data[0], ancilla[0]),  # ancilla[0] is unprepared - violation
        )

        violations = validate_qubit_states(prog, strict=True)

        assert len(violations) == 1
        assert violations[0].array_name == "ancilla"
        assert violations[0].index == 0


class TestStateViolation:
    """Tests for StateViolation dataclass."""

    def test_string_representation(self):
        """StateViolation has readable string representation."""
        violation = StateViolation(
            array_name="q",
            index=2,
            position=5,
            gate_name="H",
            message="Test message",
        )

        s = str(violation)
        assert "q[2]" in s
        assert "position 5" in s
        assert "Test message" in s


class TestValidatorClass:
    """Tests for QubitStateValidator class directly."""

    def test_validator_reusable(self):
        """Validator can be reused for multiple blocks."""
        validator = QubitStateValidator(strict=True)

        prog1 = Main(
            q := QReg("q", 1),
            qb.H(q[0]),
        )

        prog2 = Main(
            q := QReg("q", 1),
            qb.Prep(q[0]),
            qb.H(q[0]),
        )

        violations1 = validator.validate(prog1)
        violations2 = validator.validate(prog2)

        assert len(violations1) == 1
        assert len(violations2) == 0

    def test_strict_flag(self):
        """Strict flag controls initial state assumption."""
        prog = Main(
            q := QReg("q", 1),
            qb.H(q[0]),
        )

        strict_validator = QubitStateValidator(strict=True)
        non_strict_validator = QubitStateValidator(strict=False)

        variable_context = {"q": prog.vars.get("q")}

        strict_violations = strict_validator.validate(prog)
        non_strict_violations = non_strict_validator.validate(prog, variable_context)

        assert len(strict_violations) == 1
        assert len(non_strict_violations) == 0
