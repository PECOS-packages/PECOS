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

"""AST-based qubit state validation.

This module validates that quantum gates are only applied to prepared qubit slots,
detecting compile-time errors when gates are applied to unprepared/measured qubits.

The validation follows the two-state model:
- UNPREPARED: Initial state or after measurement - cannot apply gates
- PREPARED: After preparation - ready for gate operations

Example:
    from pecos.slr.ast import slr_to_ast
    from pecos.slr.ast.analysis import validate_ast_qubit_states

    ast = slr_to_ast(program)
    violations = validate_ast_qubit_states(ast)

    if violations:
        for v in violations:
            print(f"Error: {v}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

from pecos.slr.ast.nodes import (
    AllocatorDecl,
    ForStmt,
    GateOp,
    IfStmt,
    MeasureOp,
    ParallelBlock,
    PrepareOp,
    RepeatStmt,
    WhileStmt,
)

if TYPE_CHECKING:
    from pecos.slr.ast.nodes import (
        GateKind,
        Program,
        Statement,
    )


class ValidationSlotState(Enum):
    """State of a qubit slot for validation purposes."""

    UNPREPARED = auto()  # Initial state or after measurement
    PREPARED = auto()  # After preparation, ready for gates


@dataclass
class StateViolation:
    """A validation error: gate applied to unprepared slot."""

    allocator: str
    index: int
    position: int
    gate: GateKind
    location: str | None = None

    @property
    def message(self) -> str:
        """Human-readable error message."""
        return (
            f"Gate '{self.gate.name}' applied to unprepared qubit "
            f"{self.allocator}[{self.index}]. Call Prep() before applying gates."
        )

    def __str__(self) -> str:
        loc = f" at {self.location}" if self.location else ""
        return f"{self.allocator}[{self.index}]{loc}: {self.message}"


@dataclass
class QubitStateTracker:
    """Tracks the preparation state of qubit slots through program execution."""

    # Map from (allocator_name, index) to current state
    slot_states: dict[tuple[str, int], ValidationSlotState] = field(
        default_factory=dict,
    )

    # Collected violations
    violations: list[StateViolation] = field(default_factory=list)

    # Position counter for tracking operation order
    position: int = 0

    def get_state(self, allocator: str, index: int) -> ValidationSlotState:
        """Get the current state of a slot. Defaults to UNPREPARED."""
        return self.slot_states.get((allocator, index), ValidationSlotState.UNPREPARED)

    def mark_prepared(self, allocator: str, index: int) -> None:
        """Mark a slot as prepared (after Prep operation)."""
        self.slot_states[(allocator, index)] = ValidationSlotState.PREPARED

    def mark_unprepared(self, allocator: str, index: int) -> None:
        """Mark a slot as unprepared (after measurement)."""
        self.slot_states[(allocator, index)] = ValidationSlotState.UNPREPARED

    def validate_gate(
        self,
        allocator: str,
        index: int,
        gate: GateKind,
        location: str | None = None,
    ) -> bool:
        """Validate that a gate can be applied to this slot.

        Returns True if valid, False if violation detected.
        """
        state = self.get_state(allocator, index)
        if state == ValidationSlotState.UNPREPARED:
            self.violations.append(
                StateViolation(
                    allocator=allocator,
                    index=index,
                    position=self.position,
                    gate=gate,
                    location=location,
                ),
            )
            return False
        return True

    def has_violations(self) -> bool:
        """Check if any violations were detected."""
        return len(self.violations) > 0

    def get_violations(self) -> list[StateViolation]:
        """Get all detected violations."""
        return self.violations.copy()

    def copy_state(self) -> dict[tuple[str, int], ValidationSlotState]:
        """Copy the current slot states for branch analysis."""
        return dict(self.slot_states)


class AstQubitStateValidator:
    """Validates qubit state requirements in AST programs using recursive descent.

    Validates that:
    1. Gates are only applied to prepared qubits
    2. Measurements transition qubits to unprepared
    3. Preparations transition qubits to prepared

    Usage:
        validator = AstQubitStateValidator()
        violations = validator.validate(ast_program)

        if violations:
            for v in violations:
                print(f"Error: {v}")
    """

    def __init__(self, *, strict: bool = True) -> None:
        """Initialize the validator.

        Args:
            strict: If True, all qubits start unprepared and must be explicitly prepared.
                   If False, qubits are assumed prepared initially (legacy compatibility).
        """
        self.strict = strict
        self.tracker = QubitStateTracker()

    def validate(self, program: Program) -> list[StateViolation]:
        """Validate qubit states in a program.

        Args:
            program: The AST Program to validate.

        Returns:
            List of StateViolation objects for any detected errors.
        """
        self.tracker = QubitStateTracker()

        # In non-strict mode, mark all declared qubits as prepared initially
        if not self.strict:
            self._initialize_prepared(program)

        # Validate all statements in the program body
        for stmt in program.body:
            self._validate_statement(stmt)

        return self.tracker.get_violations()

    def _initialize_prepared(self, program: Program) -> None:
        """Mark all declared qubits as initially prepared (legacy mode)."""
        for decl in program.declarations:
            if isinstance(decl, AllocatorDecl):
                for i in range(decl.capacity):
                    self.tracker.mark_prepared(decl.name, i)

        if program.allocator:
            for i in range(program.allocator.capacity):
                self.tracker.mark_prepared(program.allocator.name, i)

    def _validate_statement(self, stmt: Statement) -> None:
        """Validate a single statement using recursive descent."""
        if isinstance(stmt, PrepareOp):
            self._validate_prepare(stmt)
        elif isinstance(stmt, MeasureOp):
            self._validate_measure(stmt)
        elif isinstance(stmt, GateOp):
            self._validate_gate(stmt)
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
        # Other statements (Assign, Barrier, Comment, Return) don't affect qubit state

    def _validate_prepare(self, node: PrepareOp) -> None:
        """Handle preparation: transitions qubits to prepared."""
        if node.slots is not None:
            for slot in node.slots:
                self.tracker.mark_prepared(node.allocator, slot)
        self.tracker.position += 1

    def _validate_measure(self, node: MeasureOp) -> None:
        """Handle measurement: transitions qubits to unprepared."""
        for target in node.targets:
            self.tracker.mark_unprepared(target.allocator, target.index)
        self.tracker.position += 1

    def _validate_gate(self, node: GateOp) -> None:
        """Validate gate: all target qubits must be prepared."""
        location = None
        if node.location:
            location = f"{node.location.file}:{node.location.line}"

        for target in node.targets:
            self.tracker.validate_gate(
                target.allocator,
                target.index,
                node.gate,
                location,
            )

        self.tracker.position += 1

    def _validate_if(self, node: IfStmt) -> None:
        """Validate an if statement with branch merging.

        For if/else, we need to be conservative: a qubit is only considered
        prepared after the if block if it's prepared in BOTH branches.
        """
        # Save state before if
        state_before = self.tracker.copy_state()

        # Validate then block
        for stmt in node.then_body:
            self._validate_statement(stmt)

        # Save state after then
        state_after_then = self.tracker.copy_state()

        # Reset to state before if and validate else
        self.tracker.slot_states = dict(state_before)

        if node.else_body:
            for stmt in node.else_body:
                self._validate_statement(stmt)

        state_after_else = self.tracker.copy_state()

        # Merge states: only prepared if prepared in BOTH branches
        merged_state: dict[tuple[str, int], ValidationSlotState] = {}
        all_keys = set(state_after_then.keys()) | set(state_after_else.keys())

        for key in all_keys:
            then_state = state_after_then.get(key, ValidationSlotState.UNPREPARED)
            else_state = state_after_else.get(key, ValidationSlotState.UNPREPARED)

            if then_state == ValidationSlotState.PREPARED and else_state == ValidationSlotState.PREPARED:
                merged_state[key] = ValidationSlotState.PREPARED
            else:
                merged_state[key] = ValidationSlotState.UNPREPARED

        self.tracker.slot_states = merged_state

    def _validate_while(self, node: WhileStmt) -> None:
        """Validate a while loop."""
        for stmt in node.body:
            self._validate_statement(stmt)

    def _validate_for(self, node: ForStmt) -> None:
        """Validate a for loop."""
        for stmt in node.body:
            self._validate_statement(stmt)

    def _validate_repeat(self, node: RepeatStmt) -> None:
        """Validate a repeat loop."""
        for stmt in node.body:
            self._validate_statement(stmt)

    def _validate_parallel(self, node: ParallelBlock) -> None:
        """Validate a parallel block."""
        for stmt in node.body:
            self._validate_statement(stmt)


def validate_ast_qubit_states(
    program: Program,
    *,
    strict: bool = True,
) -> list[StateViolation]:
    """Convenience function to validate qubit states in an AST program.

    Args:
        program: The AST Program to validate.
        strict: If True, qubits must be explicitly prepared before use.

    Returns:
        List of StateViolation objects for any detected errors.
    """
    validator = AstQubitStateValidator(strict=strict)
    return validator.validate(program)
