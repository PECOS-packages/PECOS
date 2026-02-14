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

"""Allocation validation pass.

This module validates allocator consistency in AST programs, checking:
- All referenced allocators are declared
- Allocator hierarchy is consistent
- No duplicate allocator names
- Parent allocators exist when referenced

Example:
    from pecos.slr.ast import slr_to_ast
    from pecos.slr.ast.validation import AllocationValidator

    ast = slr_to_ast(program)
    result = AllocationValidator().validate(ast)

    if not result.valid:
        for error in result.errors:
            print(error)
"""

from __future__ import annotations

from pecos.slr.ast.nodes import (
    AllocatorDecl,
    AssignOp,
    ForStmt,
    GateOp,
    IfStmt,
    MeasureOp,
    ParallelBlock,
    PrepareOp,
    Program,
    RepeatStmt,
    Statement,
    WhileStmt,
)
from pecos.slr.ast.validation.base import (
    Severity,
    ValidationError,
    ValidationPass,
    ValidationResult,
)


class AllocationValidator(ValidationPass):
    """Validates allocator consistency in AST programs.

    Checks:
    - All SlotRefs reference declared allocators
    - No duplicate allocator names
    - Parent allocators exist when referenced
    - Allocator capacity is positive
    """

    @property
    def name(self) -> str:
        return "allocation_validator"

    def __init__(self) -> None:
        self.allocators: dict[str, AllocatorDecl] = {}
        self.errors: list[ValidationError] = []
        self.warnings: list[ValidationError] = []
        self.referenced_allocators: set[str] = set()

    def validate(self, program: Program) -> ValidationResult:
        """Validate allocator consistency in a program.

        Args:
            program: The AST Program to validate.

        Returns:
            ValidationResult with any allocator errors found.
        """
        self.allocators = {}
        self.errors = []
        self.warnings = []
        self.referenced_allocators = set()

        # Collect and validate declarations
        self._collect_allocators(program)

        # Validate references in statements
        for stmt in program.body:
            self._validate_statement(stmt)

        # Check for unused allocators
        for name in self.allocators:
            if name not in self.referenced_allocators:
                self.warnings.append(
                    ValidationError(
                        message=f"Allocator '{name}' is declared but never used",
                        location=self.allocators[name].location,
                        severity=Severity.WARNING,
                        code="W301",
                    )
                )

        return ValidationResult(
            valid=len(self.errors) == 0,
            errors=self.errors,
            warnings=self.warnings,
            passes_applied=[self.name],
        )

    def _collect_allocators(self, program: Program) -> None:
        """Collect and validate allocator declarations."""
        # Base allocator
        if program.allocator:
            self._add_allocator(program.allocator)

        # Additional allocators in declarations
        for decl in program.declarations:
            if isinstance(decl, AllocatorDecl):
                self._add_allocator(decl)

        # Validate parent references
        for name, decl in self.allocators.items():
            if decl.parent and decl.parent not in self.allocators:
                self.errors.append(
                    ValidationError(
                        message=f"Allocator '{name}' references unknown parent '{decl.parent}'",
                        location=decl.location,
                        severity=Severity.ERROR,
                        code="E302",
                    )
                )

        # Check for cycles in parent hierarchy
        self._check_parent_cycles()

    def _add_allocator(self, decl: AllocatorDecl) -> None:
        """Add an allocator declaration."""
        if decl.name in self.allocators:
            self.errors.append(
                ValidationError(
                    message=f"Duplicate allocator name '{decl.name}'",
                    location=decl.location,
                    severity=Severity.ERROR,
                    code="E301",
                )
            )
        else:
            self.allocators[decl.name] = decl

            # Validate capacity
            if decl.capacity <= 0:
                self.errors.append(
                    ValidationError(
                        message=f"Allocator '{decl.name}' has non-positive capacity: {decl.capacity}",
                        location=decl.location,
                        severity=Severity.ERROR,
                        code="E303",
                    )
                )

    def _check_parent_cycles(self) -> None:
        """Check for cycles in the allocator parent hierarchy."""
        for name in self.allocators:
            visited: set[str] = set()
            current = name

            while current:
                if current in visited:
                    self.errors.append(
                        ValidationError(
                            message=f"Cycle detected in allocator hierarchy starting at '{name}'",
                            location=self.allocators[name].location,
                            severity=Severity.ERROR,
                            code="E304",
                        )
                    )
                    break

                visited.add(current)
                decl = self.allocators.get(current)
                if decl:
                    current = decl.parent
                else:
                    break

    def _validate_statement(self, stmt: Statement) -> None:
        """Validate a statement using recursive descent."""
        if isinstance(stmt, GateOp):
            self._validate_gate(stmt)
        elif isinstance(stmt, MeasureOp):
            self._validate_measure(stmt)
        elif isinstance(stmt, PrepareOp):
            self._validate_prepare(stmt)
        elif isinstance(stmt, AssignOp):
            pass  # No allocator references
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

    def _validate_allocator_ref(self, allocator: str, location: object = None) -> None:
        """Validate a reference to an allocator."""
        self.referenced_allocators.add(allocator)

        if allocator not in self.allocators:
            self.errors.append(
                ValidationError(
                    message=f"Reference to undeclared allocator '{allocator}'",
                    location=location if hasattr(location, "line") else None,  # type: ignore[arg-type]
                    severity=Severity.ERROR,
                    code="E305",
                )
            )

    def _validate_gate(self, node: GateOp) -> None:
        """Validate gate allocator references."""
        for target in node.targets:
            self._validate_allocator_ref(target.allocator, target.location)

    def _validate_measure(self, node: MeasureOp) -> None:
        """Validate measurement allocator references."""
        for target in node.targets:
            self._validate_allocator_ref(target.allocator, target.location)

    def _validate_prepare(self, node: PrepareOp) -> None:
        """Validate prepare allocator reference."""
        self._validate_allocator_ref(node.allocator, node.location)

    def _validate_if(self, node: IfStmt) -> None:
        """Validate if statement."""
        for stmt in node.then_body:
            self._validate_statement(stmt)
        if node.else_body is not None:
            for stmt in node.else_body:
                self._validate_statement(stmt)

    def _validate_while(self, node: WhileStmt) -> None:
        """Validate while loop."""
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


def validate_allocations(program: Program) -> ValidationResult:
    """Convenience function to validate allocations.

    Args:
        program: The AST Program to validate.

    Returns:
        ValidationResult with any allocation errors found.
    """
    validator = AllocationValidator()
    return validator.validate(program)
