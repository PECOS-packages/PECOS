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

"""AST validation passes.

This package provides validation passes for AST programs, checking for:
- Bounds errors (qubit/bit indices)
- Type errors (gate parameters, arity)
- Allocation errors (allocator references, hierarchy)

Example:
    from pecos.slr import Main, QReg
    from pecos.slr.qeclib import qubit as qb
    from pecos.slr.ast import slr_to_ast
    from pecos.slr.ast.validation import validate

    prog = Main(
        q := QReg("q", 2),
        qb.H(q[0]),
        qb.CX(q[0], q[1]),
    )
    ast = slr_to_ast(prog)
    result = validate(ast)

    if result.valid:
        print("Program is valid")
    else:
        for error in result.errors:
            print(error)
"""

from pecos.slr.ast.nodes import Program
from pecos.slr.ast.validation.allocation_validator import (
    AllocationValidator,
    validate_allocations,
)
from pecos.slr.ast.validation.base import (
    Severity,
    ValidationError,
    ValidationPass,
    ValidationPipeline,
    ValidationResult,
)
from pecos.slr.ast.validation.bounds_checker import (
    BoundsChecker,
    check_bounds,
)
from pecos.slr.ast.validation.type_checker import (
    TypeChecker,
    check_types,
)


def validate(program: Program) -> ValidationResult:
    """Run all validation passes on a program.

    This is the main entry point for validation. It runs all standard
    validation passes and returns combined results.

    Args:
        program: The AST Program to validate.

    Returns:
        ValidationResult with all errors and warnings found.

    Example:
        result = validate(ast)
        if not result.valid:
            for error in result.errors:
                print(error)
    """
    pipeline = ValidationPipeline(
        [
            AllocationValidator(),
            BoundsChecker(),
            TypeChecker(),
        ]
    )
    return pipeline.validate(program)


def create_default_pipeline() -> ValidationPipeline:
    """Create the default validation pipeline.

    Returns:
        ValidationPipeline with all standard validation passes.
    """
    return ValidationPipeline(
        [
            AllocationValidator(),
            BoundsChecker(),
            TypeChecker(),
        ]
    )


__all__ = [
    # Base classes
    "Severity",
    "ValidationError",
    "ValidationPass",
    "ValidationPipeline",
    "ValidationResult",
    # Validators
    "AllocationValidator",
    "BoundsChecker",
    "TypeChecker",
    # Convenience functions
    "check_bounds",
    "check_types",
    "create_default_pipeline",
    "validate",
    "validate_allocations",
]
