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

"""Type checking validation pass.

This module validates type correctness in AST programs, including:
- Gate parameter types (angles should be numeric)
- Gate arity (correct number of qubit targets)
- Measurement result types

Example:
    from pecos.slr.ast import slr_to_ast
    from pecos.slr.ast.validation import TypeChecker

    ast = slr_to_ast(program)
    result = TypeChecker().validate(ast)

    if not result.valid:
        for error in result.errors:
            print(error)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pecos.slr.ast.nodes import (
    AssignOp,
    BinaryExpr,
    BitExpr,
    ForStmt,
    GateKind,
    GateOp,
    IfStmt,
    LiteralExpr,
    MeasureOp,
    ParallelBlock,
    PrepareOp,
    RepeatStmt,
    UnaryExpr,
    VarExpr,
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
        Statement,
    )

# Gates that require parameters
PARAMETERIZED_GATES = frozenset({GateKind.RX, GateKind.RY, GateKind.RZ, GateKind.RZZ})

# Expected arity for each gate
GATE_ARITY: dict[GateKind, int] = {
    # Single-qubit gates
    GateKind.X: 1,
    GateKind.Y: 1,
    GateKind.Z: 1,
    GateKind.H: 1,
    GateKind.S: 1,
    GateKind.Sdg: 1,
    GateKind.T: 1,
    GateKind.Tdg: 1,
    GateKind.SX: 1,
    GateKind.SY: 1,
    GateKind.SZ: 1,
    GateKind.SXdg: 1,
    GateKind.SYdg: 1,
    GateKind.SZdg: 1,
    GateKind.RX: 1,
    GateKind.RY: 1,
    GateKind.RZ: 1,
    GateKind.F: 1,
    GateKind.Fdg: 1,
    GateKind.F4: 1,
    GateKind.F4dg: 1,
    # Two-qubit gates
    GateKind.CX: 2,
    GateKind.CY: 2,
    GateKind.CZ: 2,
    GateKind.CH: 2,
    GateKind.SXX: 2,
    GateKind.SYY: 2,
    GateKind.SZZ: 2,
    GateKind.SXXdg: 2,
    GateKind.SYYdg: 2,
    GateKind.SZZdg: 2,
    GateKind.RZZ: 2,
}


class TypeChecker(ValidationPass):
    """Validates type correctness in AST programs.

    Checks:
    - Gate parameter types are numeric
    - Gate arity matches expected qubit count
    - Non-parameterized gates don't have parameters
    """

    @property
    def name(self) -> str:
        return "type_checker"

    def __init__(self) -> None:
        self.errors: list[ValidationError] = []
        self.warnings: list[ValidationError] = []

    def validate(self, program: Program) -> ValidationResult:
        """Validate types in a program.

        Args:
            program: The AST Program to validate.

        Returns:
            ValidationResult with any type errors found.
        """
        self.errors = []
        self.warnings = []

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
            pass  # No type checking needed
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

    def _validate_gate(self, node: GateOp) -> None:
        """Validate gate type correctness."""
        # Check arity
        expected_arity = GATE_ARITY.get(node.gate, node.gate.arity)
        actual_arity = len(node.targets)

        if actual_arity != expected_arity:
            self.errors.append(
                ValidationError(
                    message=f"Gate {node.gate.name} expects {expected_arity} qubit(s), got {actual_arity}",
                    location=node.location,
                    severity=Severity.ERROR,
                    code="E201",
                ),
            )

        # Check parameters
        if node.gate in PARAMETERIZED_GATES:
            if not node.params:
                self.errors.append(
                    ValidationError(
                        message=f"Gate {node.gate.name} requires an angle parameter",
                        location=node.location,
                        severity=Severity.ERROR,
                        code="E202",
                    ),
                )
            else:
                for param in node.params:
                    self._validate_numeric_expression(
                        param,
                        f"angle parameter for {node.gate.name}",
                    )
        else:
            if node.params:
                self.warnings.append(
                    ValidationError(
                        message=f"Gate {node.gate.name} does not take parameters, "
                        f"but {len(node.params)} were provided",
                        location=node.location,
                        severity=Severity.WARNING,
                        code="W201",
                    ),
                )

    def _validate_numeric_expression(self, expr: Expression, context: str) -> None:
        """Validate that an expression is numeric."""
        if isinstance(expr, LiteralExpr):
            if not isinstance(expr.value, int | float):
                self.errors.append(
                    ValidationError(
                        message=f"Expected numeric value for {context}, got {type(expr.value).__name__}",
                        location=expr.location,
                        severity=Severity.ERROR,
                        code="E203",
                    ),
                )
        elif isinstance(expr, VarExpr):
            # Variables could be numeric - can't check at compile time
            pass
        elif isinstance(expr, BinaryExpr):
            self._validate_numeric_expression(expr.left, context)
            self._validate_numeric_expression(expr.right, context)
        elif isinstance(expr, UnaryExpr):
            self._validate_numeric_expression(expr.operand, context)
        elif isinstance(expr, BitExpr):
            self.errors.append(
                ValidationError(
                    message=f"Expected numeric expression for {context}, got bit expression",
                    location=expr.location,
                    severity=Severity.ERROR,
                    code="E204",
                ),
            )

    def _validate_measure(self, node: MeasureOp) -> None:
        """Validate measurement type correctness."""
        # Check that we have matching targets and results if results are specified
        if node.results and len(node.targets) != len(node.results):
            self.errors.append(
                ValidationError(
                    message=f"Measurement has {len(node.targets)} qubit target(s) "
                    f"but {len(node.results)} result location(s)",
                    location=node.location,
                    severity=Severity.ERROR,
                    code="E205",
                ),
            )

    def _validate_assign(self, node: AssignOp) -> None:
        """Validate assignment type correctness."""
        # The value expression should be valid
        self._validate_expression(node.value)

    def _validate_expression(self, expr: Expression) -> None:
        """Validate general expression (for conditions, etc.)."""
        if isinstance(expr, BinaryExpr):
            self._validate_expression(expr.left)
            self._validate_expression(expr.right)
        elif isinstance(expr, UnaryExpr):
            self._validate_expression(expr.operand)
        # LiteralExpr, VarExpr, BitExpr are all valid expressions

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
        self._validate_numeric_expression(node.start, "for loop start")
        self._validate_numeric_expression(node.stop, "for loop stop")
        if node.step:
            self._validate_numeric_expression(node.step, "for loop step")
        for stmt in node.body:
            self._validate_statement(stmt)

    def _validate_repeat(self, node: RepeatStmt) -> None:
        """Validate repeat loop."""
        if node.count < 0:
            self.errors.append(
                ValidationError(
                    message=f"Repeat count cannot be negative: {node.count}",
                    location=node.location,
                    severity=Severity.ERROR,
                    code="E206",
                ),
            )
        for stmt in node.body:
            self._validate_statement(stmt)

    def _validate_parallel(self, node: ParallelBlock) -> None:
        """Validate parallel block."""
        for stmt in node.body:
            self._validate_statement(stmt)


def check_types(program: Program) -> ValidationResult:
    """Convenience function to check types.

    Args:
        program: The AST Program to check.

    Returns:
        ValidationResult with any type errors found.
    """
    checker = TypeChecker()
    return checker.validate(program)
