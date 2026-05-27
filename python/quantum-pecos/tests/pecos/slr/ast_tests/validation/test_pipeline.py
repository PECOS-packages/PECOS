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

"""Tests for validation pipeline."""

import math

from pecos.slr import CReg, If, Main, QReg, rad
from pecos.slr.ast import slr_to_ast
from pecos.slr.ast.nodes import (
    AllocatorDecl,
    GateKind,
    GateOp,
    Program,
    SlotRef,
)
from pecos.slr.ast.validation import (
    BoundsChecker,
    TypeChecker,
    ValidationPipeline,
    ValidationResult,
    create_default_pipeline,
    validate,
)
from pecos.slr.qeclib import qubit as qb


class TestValidateFunction:
    """Tests for the validate() convenience function."""

    def test_validate_valid_program(self) -> None:
        """Valid program passes all checks."""
        prog = Main(
            q := QReg("q", 2),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
        )

        ast = slr_to_ast(prog)
        result = validate(ast)

        assert result.valid is True
        assert len(result.errors) == 0

    def test_validate_with_rotation(self) -> None:
        """Program with rotation gates passes."""
        prog = Main(
            q := QReg("q", 1),
            qb.RZ(rad(0.5), q[0]),
            qb.RX(rad(math.pi), q[0]),
        )

        ast = slr_to_ast(prog)
        result = validate(ast)

        assert result.valid is True

    def test_validate_complex_circuit(self) -> None:
        """Complex circuit with all features."""
        prog = Main(
            q := QReg("q", 3),
            c := CReg("c", 1),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
            qb.CX(q[1], q[2]),
            qb.RZ(rad(0.5), q[0]),
            If(c[0] == 1).Then(
                qb.X(q[0]),
            ),
        )

        ast = slr_to_ast(prog)
        result = validate(ast)

        assert result.valid is True


class TestValidationPipeline:
    """Tests for ValidationPipeline class."""

    def test_empty_pipeline(self) -> None:
        """Empty pipeline returns valid result."""
        pipeline = ValidationPipeline([])

        prog = Main(q := QReg("q", 1), qb.H(q[0]))
        ast = slr_to_ast(prog)

        result = pipeline.validate(ast)

        assert result.valid is True
        assert len(result.passes_applied) == 0

    def test_custom_pipeline(self) -> None:
        """Custom pipeline with specific passes."""
        pipeline = ValidationPipeline([BoundsChecker()])

        prog = Main(q := QReg("q", 1), qb.H(q[0]))
        ast = slr_to_ast(prog)

        result = pipeline.validate(ast)

        assert result.valid is True
        assert "bounds_checker" in result.passes_applied

    def test_add_pass(self) -> None:
        """Adding passes to pipeline."""
        pipeline = ValidationPipeline()
        pipeline.add_pass(BoundsChecker())
        pipeline.add_pass(TypeChecker())

        prog = Main(q := QReg("q", 1), qb.H(q[0]))
        ast = slr_to_ast(prog)

        result = pipeline.validate(ast)

        assert result.valid is True
        assert len(result.passes_applied) == 2

    def test_pipeline_accumulates_errors(self) -> None:
        """Pipeline accumulates errors from all passes."""
        pipeline = ValidationPipeline([BoundsChecker(), TypeChecker()])

        # Create program with multiple types of errors
        prog = Program(
            name="test",
            allocator=AllocatorDecl(name="q", capacity=2),
            body=(
                GateOp(
                    gate=GateKind.H,
                    targets=(SlotRef(allocator="q", index=5),),  # Bounds error
                ),
                GateOp(
                    gate=GateKind.H,
                    targets=(
                        SlotRef(allocator="q", index=0),
                        SlotRef(allocator="q", index=1),  # Arity error
                    ),
                ),
            ),
        )

        result = pipeline.validate(prog)

        assert result.valid is False
        assert len(result.errors) >= 2  # At least one from each pass


class TestDefaultPipeline:
    """Tests for create_default_pipeline()."""

    def test_default_pipeline_runs_all(self) -> None:
        """Default pipeline runs all standard passes."""
        pipeline = create_default_pipeline()

        prog = Main(q := QReg("q", 1), qb.H(q[0]))
        ast = slr_to_ast(prog)

        result = pipeline.validate(ast)

        assert result.valid is True
        assert "bounds_checker" in result.passes_applied
        assert "type_checker" in result.passes_applied
        assert "allocation_validator" in result.passes_applied


class TestValidationResult:
    """Tests for ValidationResult class."""

    def test_result_merge(self) -> None:
        """Merging validation results."""
        result1 = ValidationResult(valid=True, passes_applied=["pass1"])
        result2 = ValidationResult(valid=False, errors=[], passes_applied=["pass2"])

        merged = result1.merge(result2)

        assert merged.valid is False
        assert "pass1" in merged.passes_applied
        assert "pass2" in merged.passes_applied

    def test_result_string_valid(self) -> None:
        """String representation for valid result."""
        result = ValidationResult(valid=True)
        assert "Valid" in str(result)

    def test_result_string_invalid(self) -> None:
        """String representation for invalid result."""
        result = ValidationResult(valid=False)
        assert "Invalid" in str(result)

    def test_error_count(self) -> None:
        """Error count property."""
        from pecos.slr.ast.validation import ValidationError

        result = ValidationResult(
            valid=False,
            errors=[
                ValidationError(message="error1"),
                ValidationError(message="error2"),
            ],
        )

        assert result.error_count == 2

    def test_warning_count(self) -> None:
        """Warning count property."""
        from pecos.slr.ast.validation import Severity, ValidationError

        result = ValidationResult(
            valid=True,
            warnings=[
                ValidationError(message="warn1", severity=Severity.WARNING),
            ],
        )

        assert result.warning_count == 1
