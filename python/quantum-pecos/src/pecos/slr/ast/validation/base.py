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

"""Base classes for AST validation passes.

This module provides the foundation for building validation passes that
check AST programs for errors and warnings.

Example:
    from pecos.slr.ast.validation import ValidationResult, ValidationError

    class MyValidator(ValidationPass):
        @property
        def name(self) -> str:
            return "my_validator"

        def validate(self, program: Program) -> ValidationResult:
            errors = []
            # ... validation logic ...
            return ValidationResult(valid=len(errors) == 0, errors=errors)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pecos.slr.ast.nodes import Program, SourceLocation


class Severity(Enum):
    """Severity level for validation messages."""

    ERROR = auto()
    WARNING = auto()
    INFO = auto()


@dataclass
class ValidationError:
    """A single validation error or warning.

    Attributes:
        message: Human-readable description of the issue.
        location: Source location where the issue was found.
        severity: How severe the issue is (error, warning, info).
        code: Machine-readable error code (e.g., "E001", "W001").
    """

    message: str
    location: SourceLocation | None = None
    severity: Severity = Severity.ERROR
    code: str = ""

    def __str__(self) -> str:
        severity_str = self.severity.name.lower()
        if self.location:
            return f"{self.location}: {severity_str}: {self.message}"
        return f"{severity_str}: {self.message}"


@dataclass
class ValidationResult:
    """Result of running validation passes.

    Attributes:
        valid: Whether the program is valid (no errors).
        errors: List of errors found.
        warnings: List of warnings found.
        info: List of informational messages.
        passes_applied: Names of validation passes that were run.
    """

    valid: bool = True
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[ValidationError] = field(default_factory=list)
    info: list[ValidationError] = field(default_factory=list)
    passes_applied: list[str] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        """Total number of errors."""
        return len(self.errors)

    @property
    def warning_count(self) -> int:
        """Total number of warnings."""
        return len(self.warnings)

    def merge(self, other: ValidationResult) -> ValidationResult:
        """Merge another result into this one.

        Returns:
            A new ValidationResult with combined results.
        """
        return ValidationResult(
            valid=self.valid and other.valid,
            errors=self.errors + other.errors,
            warnings=self.warnings + other.warnings,
            info=self.info + other.info,
            passes_applied=self.passes_applied + other.passes_applied,
        )

    def __str__(self) -> str:
        if self.valid:
            return f"Valid ({self.warning_count} warnings)"
        return f"Invalid: {self.error_count} errors, {self.warning_count} warnings"


class ValidationPass(ABC):
    """Abstract base class for validation passes.

    Each validation pass checks for specific types of issues in the AST.
    Passes can be composed to form a complete validation pipeline.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this validation pass."""
        ...

    @abstractmethod
    def validate(self, program: Program) -> ValidationResult:
        """Validate a program.

        Args:
            program: The AST Program to validate.

        Returns:
            ValidationResult with any errors/warnings found.
        """
        ...


class ValidationPipeline:
    """A pipeline of validation passes.

    Runs multiple validation passes in sequence and combines their results.
    """

    def __init__(self, passes: list[ValidationPass] | None = None) -> None:
        """Initialize the pipeline.

        Args:
            passes: List of validation passes to run.
        """
        self.passes = passes or []

    def add_pass(self, pass_: ValidationPass) -> None:
        """Add a validation pass to the pipeline.

        Args:
            pass_: The validation pass to add.
        """
        self.passes.append(pass_)

    def validate(self, program: Program) -> ValidationResult:
        """Run all validation passes on a program.

        Args:
            program: The AST Program to validate.

        Returns:
            Combined ValidationResult from all passes.
        """
        result = ValidationResult()

        for pass_ in self.passes:
            pass_result = pass_.validate(program)
            result = result.merge(pass_result)

        return result
