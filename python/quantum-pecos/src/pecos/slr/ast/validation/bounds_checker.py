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

"""Bounds checking validation pass.

This module validates that qubit and classical bit indices are within
the bounds of their respective registers/allocators.

Example:
    from pecos.slr.ast import slr_to_ast
    from pecos.slr.ast.validation import BoundsChecker

    ast = slr_to_ast(program)
    result = BoundsChecker().validate(ast)

    if not result.valid:
        for error in result.errors:
            print(error)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pecos.slr.ast.nodes import (
    AllocatorDecl,
    AssignOp,
    BitExpr,
    BitRef,
    ForStmt,
    GateOp,
    IfStmt,
    MeasureOp,
    ParallelBlock,
    PrepareOp,
    RegisterDecl,
    RepeatStmt,
    WhileStmt,
)
from pecos.slr.ast.validation.base import (
    Severity,
    ValidationError,
    ValidationPass,
    ValidationResult,
)

if TYPE_CHECKING:
    from pecos.slr.ast.nodes import (
        Expression,
        Program,
        SlotRef,
        Statement,
    )


class BoundsChecker(ValidationPass):
    """Validates that indices are within bounds.

    Checks:
    - Qubit slot indices are within allocator capacity
    - Classical bit indices are within register size
    - Negative indices are flagged as errors
    """

    @property
    def name(self) -> str:
        return "bounds_checker"

    def __init__(self) -> None:
        self.allocator_sizes: dict[str, int] = {}
        self.register_sizes: dict[str, int] = {}
        self.errors: list[ValidationError] = []
        self.warnings: list[ValidationError] = []

    def validate(self, program: Program) -> ValidationResult:
        """Validate bounds in a program.

        Args:
            program: The AST Program to validate.

        Returns:
            ValidationResult with any bounds errors found.
        """
        self.allocator_sizes = {}
        self.register_sizes = {}
        self.errors = []
        self.warnings = []

        # Collect declarations
        if program.allocator:
            self.allocator_sizes[program.allocator.name] = program.allocator.capacity

        for decl in program.declarations:
            if isinstance(decl, AllocatorDecl):
                self.allocator_sizes[decl.name] = decl.capacity
            elif isinstance(decl, RegisterDecl):
                self.register_sizes[decl.name] = decl.size

        # Validate statements
        for stmt in program.body:
            self._validate_statement(stmt)

        return ValidationResult(
            valid=len(self.errors) == 0,
            errors=self.errors,
            warnings=self.warnings,
            passes_applied=[self.name],
        )

    def _validate_statement(self, stmt: Statement) -> None:
        """Validate a statement using recursive descent."""
        if isinstance(stmt, GateOp):
            self._validate_gate(stmt)
        elif isinstance(stmt, MeasureOp):
            self._validate_measure(stmt)
        elif isinstance(stmt, PrepareOp):
            self._validate_prepare(stmt)
        elif isinstance(stmt, AssignOp):
            self._validate_assign(stmt)
        elif isinstance(stmt, IfStmt):
            self._validate_if(stmt)
        elif isinstance(stmt, WhileStmt):
            self._validate_while(stmt)
        elif isinstance(stmt, ForStmt):
            self._validate_for(stmt)
        elif isinstance(stmt, RepeatStmt):
            self._validate_repeat(stmt)
        elif isinstance(stmt, ParallelBlock):
            self._validate_parallel(stmt)

    def _validate_slot_ref(self, slot: SlotRef) -> None:
        """Validate a slot reference is within bounds."""
        if slot.allocator not in self.allocator_sizes:
            self.errors.append(
                ValidationError(
                    message=f"Unknown allocator '{slot.allocator}'",
                    location=slot.location,
                    severity=Severity.ERROR,
                    code="E101",
                ),
            )
            return

        capacity = self.allocator_sizes[slot.allocator]

        if slot.index < 0:
            self.errors.append(
                ValidationError(
                    message=f"Negative qubit index {slot.index} in allocator '{slot.allocator}'",
                    location=slot.location,
                    severity=Severity.ERROR,
                    code="E102",
                ),
            )
        elif slot.index >= capacity:
            self.errors.append(
                ValidationError(
                    message=f"Qubit index {slot.index} out of bounds for allocator "
                    f"'{slot.allocator}' (capacity={capacity})",
                    location=slot.location,
                    severity=Severity.ERROR,
                    code="E103",
                ),
            )

    def _validate_bit_ref(self, ref: BitRef) -> None:
        """Validate a bit reference is within bounds."""
        if ref.register not in self.register_sizes:
            self.errors.append(
                ValidationError(
                    message=f"Unknown register '{ref.register}'",
                    location=ref.location,
                    severity=Severity.ERROR,
                    code="E104",
                ),
            )
            return

        size = self.register_sizes[ref.register]

        if ref.index < 0:
            self.errors.append(
                ValidationError(
                    message=f"Negative bit index {ref.index} in register '{ref.register}'",
                    location=ref.location,
                    severity=Severity.ERROR,
                    code="E105",
                ),
            )
        elif ref.index >= size:
            self.errors.append(
                ValidationError(
                    message=f"Bit index {ref.index} out of bounds for register '{ref.register}' (size={size})",
                    location=ref.location,
                    severity=Severity.ERROR,
                    code="E106",
                ),
            )

    def _validate_gate(self, node: GateOp) -> None:
        """Validate gate operation bounds."""
        for target in node.targets:
            self._validate_slot_ref(target)

    def _validate_measure(self, node: MeasureOp) -> None:
        """Validate measurement bounds."""
        for target in node.targets:
            self._validate_slot_ref(target)
        for result in node.results:
            self._validate_bit_ref(result)

    def _validate_prepare(self, node: PrepareOp) -> None:
        """Validate prepare bounds."""
        if node.allocator not in self.allocator_sizes:
            self.errors.append(
                ValidationError(
                    message=f"Unknown allocator '{node.allocator}'",
                    location=node.location,
                    severity=Severity.ERROR,
                    code="E101",
                ),
            )
            return

        if node.slots is not None:
            capacity = self.allocator_sizes[node.allocator]
            for slot in node.slots:
                if slot < 0:
                    self.errors.append(
                        ValidationError(
                            message=f"Negative slot index {slot} in prepare for '{node.allocator}'",
                            location=node.location,
                            severity=Severity.ERROR,
                            code="E102",
                        ),
                    )
                elif slot >= capacity:
                    self.errors.append(
                        ValidationError(
                            message=f"Slot index {slot} out of bounds for allocator "
                            f"'{node.allocator}' (capacity={capacity})",
                            location=node.location,
                            severity=Severity.ERROR,
                            code="E103",
                        ),
                    )

    def _validate_assign(self, node: AssignOp) -> None:
        """Validate assignment bounds."""
        if isinstance(node.target, BitRef):
            self._validate_bit_ref(node.target)
        self._validate_expression(node.value)

    def _validate_expression(self, expr: Expression) -> None:
        """Validate expression for bit references."""
        if isinstance(expr, BitExpr):
            self._validate_bit_ref(expr.ref)

    def _validate_if(self, node: IfStmt) -> None:
        """Validate if statement."""
        self._validate_expression(node.condition)
        for stmt in node.then_body:
            self._validate_statement(stmt)
        if node.else_body is not None:
            for stmt in node.else_body:
                self._validate_statement(stmt)

    def _validate_while(self, node: WhileStmt) -> None:
        """Validate while loop."""
        self._validate_expression(node.condition)
        for stmt in node.body:
            self._validate_statement(stmt)

    def _validate_for(self, node: ForStmt) -> None:
        """Validate for loop."""
        for stmt in node.body:
            self._validate_statement(stmt)

    def _validate_repeat(self, node: RepeatStmt) -> None:
        """Validate repeat loop."""
        for stmt in node.body:
            self._validate_statement(stmt)

    def _validate_parallel(self, node: ParallelBlock) -> None:
        """Validate parallel block."""
        for stmt in node.body:
            self._validate_statement(stmt)


def check_bounds(program: Program) -> ValidationResult:
    """Convenience function to check bounds.

    Args:
        program: The AST Program to check.

    Returns:
        ValidationResult with any bounds errors found.
    """
    checker = BoundsChecker()
    return checker.validate(program)
