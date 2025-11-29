"""Qubit allocation optimizer for Guppy code generation.

This module analyzes qubit usage patterns to determine when qubits can be
allocated locally rather than pre-allocated, making the generated code more
idiomatic and potentially more efficient.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class AllocationStrategy(Enum):
    """Different allocation strategies for qubits."""

    PRE_ALLOCATE = "pre_allocate"  # Allocate all qubits upfront
    LOCAL_ALLOCATE = "local_allocate"  # Allocate when first used
    FUNCTION_SCOPED = "function_scoped"  # Allocate within function scope


@dataclass
class QubitUsage:
    """Tracks how a qubit is used throughout the program."""

    first_use_line: int
    last_use_line: int
    consumption_line: int | None = None
    uses_in_loops: set[int] = field(default_factory=set)
    uses_in_conditionals: set[int] = field(default_factory=set)
    reused_after_consumption: bool = False
    used_in_multiple_scopes: bool = False

    @property
    def lifetime_span(self) -> int:
        """Number of lines this qubit is active."""
        return self.last_use_line - self.first_use_line

    @property
    def is_short_lived(self) -> bool:
        """True if qubit has a short lifetime (< 10 lines)."""
        return self.lifetime_span < 10

    @property
    def is_consumed_early(self) -> bool:
        """True if qubit is consumed and not reused."""
        return self.consumption_line is not None and not self.reused_after_consumption


@dataclass
class AllocationDecision:
    """Decision about how to allocate a specific qubit array."""

    array_name: str
    original_size: int
    strategy: AllocationStrategy
    local_elements: set[int] = field(
        default_factory=set,
    )  # Which elements to allocate locally
    reused_elements: set[int] = field(
        default_factory=set,
    )  # Which elements are reused after consumption (need replacement)
    reasons: list[str] = field(default_factory=list)


class AllocationOptimizer:
    """Analyzes qubit usage patterns and suggests optimized allocation strategies."""

    def __init__(self):
        self.qubit_usage: dict[str, dict[int, QubitUsage]] = (
            {}
        )  # array_name -> index -> usage
        self.current_line = 0
        self.scope_stack: list[str] = ["main"]

    def analyze_program(self, main_block) -> dict[str, AllocationDecision]:
        """Analyze a program and return allocation decisions."""
        self.qubit_usage.clear()
        self.current_line = 0
        self.scope_stack = ["main"]

        # First pass: collect usage information
        self._analyze_block(main_block)

        # Second pass: make allocation decisions
        decisions = {}
        for array_name, elements in self.qubit_usage.items():
            decision = self._make_allocation_decision(array_name, elements)
            decisions[array_name] = decision

        return decisions

    def _analyze_block(self, block) -> None:
        """Analyze a block and track qubit usage."""
        if hasattr(block, "vars"):
            for var in block.vars:
                if type(var).__name__ == "QReg":
                    # Initialize usage tracking for this array
                    if var.sym not in self.qubit_usage:
                        self.qubit_usage[var.sym] = {}
                    for i in range(var.size):
                        if i not in self.qubit_usage[var.sym]:
                            self.qubit_usage[var.sym][i] = QubitUsage(
                                first_use_line=float("inf"),  # Will be set on first use
                                last_use_line=0,
                            )

        if hasattr(block, "ops"):
            for op in block.ops:
                self._analyze_operation(op)

    def _analyze_operation(self, op) -> None:
        """Analyze a single operation."""
        self.current_line += 1
        op_type = type(op).__name__

        if hasattr(op, "qargs") and op.qargs:
            for qarg in op.qargs:
                # Handle tuple arguments (e.g., CX gates with (control, target) pairs)
                if isinstance(qarg, tuple):
                    for sub_qarg in qarg:
                        self._record_qubit_use(sub_qarg, op_type)
                else:
                    self._record_qubit_use(qarg, op_type)

        # Handle measurements specially
        if op_type == "Measure" and hasattr(op, "qargs") and op.qargs:
            for qarg in op.qargs:
                self._record_qubit_consumption(qarg)

        # Recurse into nested operations
        if op_type == "If":
            self._enter_scope("if")
            if hasattr(op, "ops"):
                for nested_op in op.ops:
                    self._analyze_operation(nested_op)
            if hasattr(op, "else_block") and op.else_block:
                self._enter_scope("else")
                if hasattr(op.else_block, "ops"):
                    for nested_op in op.else_block.ops:
                        self._analyze_operation(nested_op)
                self._exit_scope()
            self._exit_scope()

        elif op_type in ["While", "For"]:
            self._enter_scope("loop")
            if hasattr(op, "ops"):
                for nested_op in op.ops:
                    self._analyze_operation(nested_op)
            self._exit_scope()

        # Handle any other blocks (PrepRUS, PrepEncodingNonFTZero, etc.)
        elif hasattr(op, "ops") and hasattr(op, "vars"):
            # This is a custom block - analyze its operations
            self._analyze_block(op)

    def _record_qubit_use(self, qarg, op_type: str) -> None:
        """Record that a qubit is being used."""
        _ = op_type  # Currently unused, kept for future use
        array_name, index = self._extract_qubit_ref(qarg)
        if array_name:
            if index is not None:
                # Single element usage
                if (
                    array_name in self.qubit_usage
                    and index in self.qubit_usage[array_name]
                ):
                    usage = self.qubit_usage[array_name][index]

                    # Check if this qubit was already consumed (measured)
                    # If so, this is a reuse after consumption
                    if (
                        usage.consumption_line is not None
                        and usage.consumption_line < self.current_line
                    ):
                        usage.reused_after_consumption = True

                    # Update first/last use
                    if usage.first_use_line == float("inf"):
                        usage.first_use_line = self.current_line
                    usage.last_use_line = self.current_line

                    # Track scope usage
                    current_scope = self.scope_stack[-1]
                    if current_scope == "loop":
                        usage.uses_in_loops.add(self.current_line)
                    elif current_scope in ["if", "else"]:
                        usage.uses_in_conditionals.add(self.current_line)

                    # Check if used across multiple scopes
                    if len(self.scope_stack) > 1:
                        usage.used_in_multiple_scopes = True
            # Full array usage - mark all elements as used
            elif array_name in self.qubit_usage:
                for idx in self.qubit_usage[array_name]:
                    usage = self.qubit_usage[array_name][idx]
                    if usage.first_use_line == float("inf"):
                        usage.first_use_line = self.current_line
                    usage.last_use_line = self.current_line

                    # Track scope usage for each element
                    current_scope = self.scope_stack[-1]
                    if current_scope == "loop":
                        usage.uses_in_loops.add(self.current_line)
                    elif current_scope in ["if", "else"]:
                        usage.uses_in_conditionals.add(self.current_line)

                    # Check if used across multiple scopes
                    if len(self.scope_stack) > 1:
                        usage.used_in_multiple_scopes = True

    def _record_qubit_consumption(self, qarg) -> None:
        """Record that a qubit is being consumed (measured)."""
        array_name, index = self._extract_qubit_ref(qarg)
        if array_name and index is not None:
            usage = self.qubit_usage[array_name][index]

            # Check if this is a reuse after previous consumption
            if usage.consumption_line is not None:
                usage.reused_after_consumption = True
            else:
                usage.consumption_line = self.current_line

    def _extract_qubit_ref(self, qarg) -> tuple[str | None, int | None]:
        """Extract array name and index from a qubit reference."""
        if hasattr(qarg, "reg") and hasattr(qarg.reg, "sym"):
            array_name = qarg.reg.sym
            if hasattr(qarg, "index"):
                return array_name, qarg.index
            return array_name, None  # Full array reference
        if hasattr(qarg, "sym") and hasattr(qarg, "size"):
            # Full array measurement
            return qarg.sym, None
        return None, None

    def _enter_scope(self, scope_type: str) -> None:
        """Enter a new scope."""
        self.scope_stack.append(scope_type)

    def _exit_scope(self) -> None:
        """Exit the current scope."""
        if len(self.scope_stack) > 1:
            self.scope_stack.pop()

    def _make_allocation_decision(
        self,
        array_name: str,
        elements: dict[int, QubitUsage],
    ) -> AllocationDecision:
        """Make allocation decision for a qubit array."""
        decision = AllocationDecision(
            array_name=array_name,
            original_size=len(elements),
            strategy=AllocationStrategy.PRE_ALLOCATE,
        )

        # Analyze each element
        short_lived_elements = set()
        early_consumed_elements = set()
        single_scope_elements = set()

        for index, usage in elements.items():
            if usage.first_use_line == float("inf"):
                # Never used in any operations
                decision.reasons.append(
                    f"Element {index} allocated but not used in operations",
                )
                continue

            if usage.is_short_lived:
                short_lived_elements.add(index)

            if usage.is_consumed_early:
                early_consumed_elements.add(index)

            if not usage.used_in_multiple_scopes and not usage.uses_in_loops:
                single_scope_elements.add(index)

        # Make decision based on analysis
        local_candidates = (
            short_lived_elements & early_consumed_elements & single_scope_elements
        )

        if len(local_candidates) > 0:
            # For now, only use partial optimization to avoid breaking existing functionality
            if len(local_candidates) < len(elements):
                # Partial local allocation
                decision.strategy = AllocationStrategy.FUNCTION_SCOPED
                decision.local_elements = local_candidates
                decision.reasons.append(
                    f"Elements {local_candidates} can be allocated locally",
                )
            else:
                # Full local allocation
                decision.strategy = AllocationStrategy.LOCAL_ALLOCATE
                decision.local_elements = local_candidates
                decision.reasons.append(
                    "All elements are short-lived and consumed early",
                )

        # Additional heuristics
        # Track which elements are reused after consumption (need replacement)
        for index, usage in elements.items():
            if usage.reused_after_consumption:
                decision.reused_elements.add(index)

        if decision.reused_elements:
            decision.strategy = AllocationStrategy.PRE_ALLOCATE
            decision.reasons.append("Some elements are reused after consumption")
            decision.local_elements.clear()

        return decision

    def generate_optimization_report(
        self,
        decisions: dict[str, AllocationDecision],
    ) -> str:
        """Generate a human-readable optimization report."""
        lines = ["=== Qubit Allocation Optimization Report ===", ""]

        for array_name, decision in decisions.items():
            lines.append(f"Array: {array_name} (size: {decision.original_size})")
            lines.append(f"  Strategy: {decision.strategy.value}")

            if decision.local_elements:
                lines.append(f"  Local elements: {sorted(decision.local_elements)}")

            lines.extend(f"  - {reason}" for reason in decision.reasons)

            lines.append("")

        return "\n".join(lines)
