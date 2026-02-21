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

"""Qubit state validation for SLR programs.

This module validates that quantum gates are only applied to prepared qubit slots,
detecting compile-time errors when gates are applied to unprepared/measured qubits.

The validation follows the two-state model from the QAlloc design:
- UNPREPARED: Initial state or after measurement - cannot apply gates
- PREPARED: After preparation - ready for gate operations

Validation errors are collected and can be reported as compile-time errors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pecos.slr import Block as SLRBlock


class ValidationSlotState(Enum):
    """State of a qubit slot for validation purposes."""

    UNPREPARED = auto()  # Initial state or after measurement
    PREPARED = auto()  # After preparation, ready for gates


@dataclass
class StateViolation:
    """A validation error: gate applied to unprepared slot."""

    array_name: str
    index: int
    position: int
    gate_name: str
    message: str

    def __str__(self) -> str:
        return f"{self.array_name}[{self.index}] at position {self.position}: {self.message}"


@dataclass
class QubitStateTracker:
    """Tracks the preparation state of qubit slots through program execution.

    Used to validate that gates are only applied to prepared qubits.
    """

    # Map from (array_name, index) to current state
    slot_states: dict[tuple[str, int], ValidationSlotState] = field(
        default_factory=dict,
    )

    # Collected violations
    violations: list[StateViolation] = field(default_factory=list)

    # Position counter for tracking operation order
    position: int = 0

    def get_state(self, array_name: str, index: int) -> ValidationSlotState:
        """Get the current state of a slot. Defaults to UNPREPARED."""
        return self.slot_states.get((array_name, index), ValidationSlotState.UNPREPARED)

    def mark_prepared(self, array_name: str, index: int) -> None:
        """Mark a slot as prepared (after Prep/Init/Reset)."""
        self.slot_states[(array_name, index)] = ValidationSlotState.PREPARED

    def mark_unprepared(self, array_name: str, index: int) -> None:
        """Mark a slot as unprepared (after measurement)."""
        self.slot_states[(array_name, index)] = ValidationSlotState.UNPREPARED

    def validate_gate(self, array_name: str, index: int, gate_name: str) -> bool:
        """Validate that a gate can be applied to this slot.

        Returns True if valid, False if violation detected.
        """
        state = self.get_state(array_name, index)
        if state == ValidationSlotState.UNPREPARED:
            self.violations.append(
                StateViolation(
                    array_name=array_name,
                    index=index,
                    position=self.position,
                    gate_name=gate_name,
                    message=f"Gate '{gate_name}' applied to unprepared qubit. "
                    f"Call prepare() before applying gates.",
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

    def clear_violations(self) -> None:
        """Clear all violations."""
        self.violations.clear()


class QubitStateValidator:
    """Validates qubit state requirements in SLR programs.

    Walks through the program operations and validates that:
    1. Gates are only applied to prepared qubits
    2. Measurements transition qubits to unprepared
    3. Preparations transition qubits to prepared

    Usage:
        validator = QubitStateValidator()
        violations = validator.validate(block, variable_context)

        if violations:
            for v in violations:
                print(f"Error: {v}")
    """

    # Operations that prepare qubits
    PREPARATION_OPS = frozenset({"Prep", "Init", "Reset", "PrepZ", "PrepX", "PrepY"})

    # Operations that consume/measure qubits
    MEASUREMENT_OPS = frozenset({"Measure", "MeasZ", "MeasX", "MeasY"})

    def __init__(self, *, strict: bool = True):
        """Initialize the validator.

        Args:
            strict: If True, all qubits start unprepared and must be explicitly prepared.
                   If False, qubits are assumed prepared initially (legacy compatibility).
        """
        self.strict = strict
        self.tracker = QubitStateTracker()

    def validate(
        self,
        block: SLRBlock,
        variable_context: dict[str, Any] | None = None,
    ) -> list[StateViolation]:
        """Validate qubit states in a block.

        Args:
            block: The SLR block to validate.
            variable_context: Optional context of variables (QReg, CReg, etc.).

        Returns:
            List of StateViolation objects for any detected errors.
        """
        self.tracker = QubitStateTracker()
        variable_context = variable_context or {}

        # In non-strict mode, mark all known qubits as prepared initially
        if not self.strict:
            self._initialize_prepared(variable_context)

        # Validate all operations
        if hasattr(block, "ops"):
            for op in block.ops:
                self._validate_operation(op, variable_context)
                self.tracker.position += 1

        return self.tracker.get_violations()

    def _initialize_prepared(self, variable_context: dict[str, Any]) -> None:
        """Mark all qubits as initially prepared (legacy mode)."""
        for var in variable_context.values():
            if hasattr(var, "size") and hasattr(var, "sym"):
                # Check if it's a quantum register
                var_type = type(var).__name__
                if var_type in ("QReg", "QAlloc"):
                    for i in range(var.size):
                        self.tracker.mark_prepared(var.sym, i)

    def _validate_operation(
        self,
        op: Any,
        variable_context: dict[str, Any],
    ) -> None:
        """Validate a single operation."""
        op_name = type(op).__name__

        if op_name in self.MEASUREMENT_OPS:
            self._handle_measurement(op)
        elif op_name in self.PREPARATION_OPS:
            self._handle_preparation(op)
        elif op_name == "If":
            self._validate_if_block(op, variable_context)
        elif op_name in ("For", "While", "Repeat"):
            self._validate_loop_block(op, variable_context)
        elif op_name == "Parallel":
            self._validate_parallel_block(op, variable_context)
        elif hasattr(op, "qargs"):
            # This is a quantum gate
            self._validate_gate(op)
        elif hasattr(op, "ops"):
            # Nested block - recurse
            for nested_op in op.ops:
                self._validate_operation(nested_op, variable_context)

    def _handle_measurement(self, op: Any) -> None:
        """Handle measurement: transitions qubits to unprepared."""
        if hasattr(op, "qargs") and op.qargs:
            for qarg in op.qargs:
                if self._has_reg_and_index(qarg):
                    self.tracker.mark_unprepared(qarg.reg.sym, qarg.index)

    def _handle_preparation(self, op: Any) -> None:
        """Handle preparation: transitions qubits to prepared."""
        if hasattr(op, "qargs") and op.qargs:
            for qarg in op.qargs:
                if self._has_reg_and_index(qarg):
                    self.tracker.mark_prepared(qarg.reg.sym, qarg.index)

    def _validate_gate(self, op: Any) -> None:
        """Validate a quantum gate: all qubits must be prepared."""
        gate_name = type(op).__name__
        if hasattr(op, "qargs") and op.qargs:
            for qarg in op.qargs:
                if self._has_reg_and_index(qarg):
                    self.tracker.validate_gate(qarg.reg.sym, qarg.index, gate_name)

    def _validate_if_block(
        self,
        if_block: Any,
        variable_context: dict[str, Any],
    ) -> None:
        """Validate an if block.

        For if/else, we need to be conservative: a qubit is only considered
        prepared after the if block if it's prepared in BOTH branches.
        """
        # Save state before if
        state_before = dict(self.tracker.slot_states)

        # Validate then block
        if hasattr(if_block, "ops"):
            for op in if_block.ops:
                self._validate_operation(op, variable_context)

        # Save state after then
        state_after_then = dict(self.tracker.slot_states)

        # Reset to state before if and validate else
        self.tracker.slot_states = dict(state_before)

        if hasattr(if_block, "else_block") and if_block.else_block and hasattr(if_block.else_block, "ops"):
            for op in if_block.else_block.ops:
                self._validate_operation(op, variable_context)

        state_after_else = dict(self.tracker.slot_states)

        # Merge states: only prepared if prepared in BOTH branches
        merged_state = {}
        all_keys = set(state_after_then.keys()) | set(state_after_else.keys())
        for key in all_keys:
            then_state = state_after_then.get(key, ValidationSlotState.UNPREPARED)
            else_state = state_after_else.get(key, ValidationSlotState.UNPREPARED)
            # Only prepared if prepared in both branches
            if then_state == ValidationSlotState.PREPARED and else_state == ValidationSlotState.PREPARED:
                merged_state[key] = ValidationSlotState.PREPARED
            else:
                merged_state[key] = ValidationSlotState.UNPREPARED

        self.tracker.slot_states = merged_state

    def _validate_loop_block(
        self,
        loop_block: Any,
        variable_context: dict[str, Any],
    ) -> None:
        """Validate a loop block.

        For loops, we validate the body once but assume the state after
        the loop could be any state that occurs during the loop.
        """
        # Validate loop body
        if hasattr(loop_block, "ops"):
            for op in loop_block.ops:
                self._validate_operation(op, variable_context)

    def _validate_parallel_block(
        self,
        parallel_block: Any,
        variable_context: dict[str, Any],
    ) -> None:
        """Validate a parallel block."""
        if hasattr(parallel_block, "ops"):
            for op in parallel_block.ops:
                self._validate_operation(op, variable_context)

    def _has_reg_and_index(self, qarg: Any) -> bool:
        """Check if a qubit argument has reg and index attributes."""
        return hasattr(qarg, "reg") and hasattr(qarg.reg, "sym") and hasattr(qarg, "index")


def validate_qubit_states(
    block: SLRBlock,
    variable_context: dict[str, Any] | None = None,
    *,
    strict: bool = True,
) -> list[StateViolation]:
    """Convenience function to validate qubit states in a block.

    Args:
        block: The SLR block to validate.
        variable_context: Optional context of variables.
        strict: If True, qubits must be explicitly prepared before use.

    Returns:
        List of StateViolation objects for any detected errors.
    """
    validator = QubitStateValidator(strict=strict)
    return validator.validate(block, variable_context)
