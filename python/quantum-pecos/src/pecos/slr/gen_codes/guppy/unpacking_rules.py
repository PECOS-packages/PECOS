"""Rule-based decision tree for array unpacking in Guppy code generation.

This module provides a cleaner, more maintainable approach to deciding when arrays
need to be unpacked, replacing the complex heuristic logic with explicit rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pecos.slr.gen_codes.guppy.ir_analyzer import ArrayAccessInfo


class UnpackingReason(Enum):
    """Enumeration of reasons why an array might need unpacking."""

    # Required unpacking (semantic necessity)
    INDIVIDUAL_QUANTUM_MEASUREMENT = (
        auto()
    )  # Measuring individual qubits requires unpacking
    OPERATIONS_AFTER_MEASUREMENT = (
        auto()
    )  # Using qubits after measurement requires replacement
    CONDITIONAL_ELEMENT_ACCESS = (
        auto()
    )  # Accessing elements conditionally requires unpacking

    # Optional unpacking (code quality)
    MULTIPLE_INDIVIDUAL_ACCESSES = (
        auto()
    )  # Multiple element accesses cleaner when unpacked
    PARTIAL_ARRAY_USAGE = auto()  # Not all elements used together

    # No unpacking needed
    FULL_ARRAY_ONLY = auto()  # Only full array operations (e.g., measure_array)
    SINGLE_ELEMENT_ONLY = auto()  # Only one element accessed (use direct indexing)
    NO_INDIVIDUAL_ACCESS = auto()  # No individual element access


class UnpackingDecision(Enum):
    """Decision outcome for array unpacking."""

    MUST_UNPACK = auto()  # Semantically required
    SHOULD_UNPACK = auto()  # Improves code quality
    SHOULD_NOT_UNPACK = auto()  # Better to keep as array
    MUST_NOT_UNPACK = auto()  # Would cause errors


@dataclass
class DecisionResult:
    """Result of unpacking decision with reasoning."""

    decision: UnpackingDecision
    reason: UnpackingReason
    explanation: str

    @property
    def should_unpack(self) -> bool:
        """Whether the array should be unpacked."""
        return self.decision in (
            UnpackingDecision.MUST_UNPACK,
            UnpackingDecision.SHOULD_UNPACK,
        )


class UnpackingDecisionTree:
    """Rule-based decision tree for determining if an array needs unpacking.

    This replaces the complex heuristic logic in ArrayAccessInfo.needs_unpacking
    with an explicit, testable decision tree.

    Decision rules are applied in order of priority:
    1. Check for conditions that REQUIRE unpacking (semantic necessity)
    2. Check for conditions that FORBID unpacking (would cause errors)
    3. Check for conditions where unpacking IMPROVES code quality
    4. Default to not unpacking (prefer simpler code)
    """

    def decide(self, info: ArrayAccessInfo) -> DecisionResult:
        """Determine if an array should be unpacked based on access patterns.

        Args:
            info: Information about how the array is accessed

        Returns:
            DecisionResult with the decision and reasoning
        """
        # Rule 1: Full array operations forbid unpacking
        if info.full_array_accesses:
            return DecisionResult(
                decision=UnpackingDecision.MUST_NOT_UNPACK,
                reason=UnpackingReason.FULL_ARRAY_ONLY,
                explanation=(
                    f"Array '{info.array_name}' has full-array operations "
                    f"(e.g., measure_array) at positions {info.full_array_accesses}. "
                    "Unpacking would prevent these operations."
                ),
            )

        # Rule 2: No individual access means no unpacking needed
        if not info.has_individual_access:
            return DecisionResult(
                decision=UnpackingDecision.SHOULD_NOT_UNPACK,
                reason=UnpackingReason.NO_INDIVIDUAL_ACCESS,
                explanation=(
                    f"Array '{info.array_name}' has no individual element access. "
                    "Keeping as array."
                ),
            )

        # Rule 3: Operations after measurement REQUIRES unpacking (quantum arrays only)
        # This is because measured qubits are consumed and need to be replaced
        if not info.is_classical and info.has_operations_between:
            return DecisionResult(
                decision=UnpackingDecision.MUST_UNPACK,
                reason=UnpackingReason.OPERATIONS_AFTER_MEASUREMENT,
                explanation=(
                    f"Quantum array '{info.array_name}' has operations on qubits "
                    "after measurement. This requires unpacking to handle qubit "
                    "replacement correctly."
                ),
            )

        # Rule 4: Individual quantum measurements REQUIRE unpacking
        # This avoids MoveOutOfSubscriptError when measuring from array indices
        if not info.is_classical and info.elements_consumed:
            return DecisionResult(
                decision=UnpackingDecision.MUST_UNPACK,
                reason=UnpackingReason.INDIVIDUAL_QUANTUM_MEASUREMENT,
                explanation=(
                    f"Quantum array '{info.array_name}' has individual element "
                    f"measurements (indices: {sorted(info.elements_consumed)}). "
                    "This requires unpacking to avoid MoveOutOfSubscriptError."
                ),
            )

        # Rule 5: Conditional element access REQUIRES unpacking
        # Elements accessed in conditionals need to be separate variables
        # NEW: Use precise element-level tracking if available
        if (
            hasattr(info, "conditionally_accessed_elements")
            and info.conditionally_accessed_elements
        ):
            # Use precise tracking - only unpack if conditionally accessed elements
            # are also individually accessed
            conditional_and_accessed = (
                info.conditionally_accessed_elements & info.element_accesses
            )
            if conditional_and_accessed:
                return DecisionResult(
                    decision=UnpackingDecision.MUST_UNPACK,
                    reason=UnpackingReason.CONDITIONAL_ELEMENT_ACCESS,
                    explanation=(
                        f"Array '{info.array_name}' has elements "
                        f"{sorted(conditional_and_accessed)} accessed in conditional "
                        "blocks. This requires unpacking for proper control flow handling."
                    ),
                )
        elif info.has_conditionals_between:
            # Fallback to old heuristic if precise tracking not available
            return DecisionResult(
                decision=UnpackingDecision.MUST_UNPACK,
                reason=UnpackingReason.CONDITIONAL_ELEMENT_ACCESS,
                explanation=(
                    f"Array '{info.array_name}' has elements accessed in conditional "
                    "blocks. This requires unpacking for proper control flow handling."
                ),
            )

        # Rule 6: Single element access should use direct indexing (no unpack)
        # This avoids PlaceNotUsedError when unpacking all but using only one
        if len(info.element_accesses) == 1:
            return DecisionResult(
                decision=UnpackingDecision.SHOULD_NOT_UNPACK,
                reason=UnpackingReason.SINGLE_ELEMENT_ONLY,
                explanation=(
                    f"Array '{info.array_name}' has only one element accessed "
                    f"(index {next(iter(info.element_accesses))}). "
                    "Using direct array indexing instead of unpacking."
                ),
            )

        # Rule 7: Classical arrays with multiple individual accesses should unpack
        # This produces cleaner code (e.g., c0, c1 instead of c[0], c[1])
        if info.is_classical and len(info.element_accesses) > 1:
            return DecisionResult(
                decision=UnpackingDecision.SHOULD_UNPACK,
                reason=UnpackingReason.MULTIPLE_INDIVIDUAL_ACCESSES,
                explanation=(
                    f"Classical array '{info.array_name}' has multiple individual "
                    f"element accesses ({len(info.element_accesses)} elements). "
                    "Unpacking produces cleaner code."
                ),
            )

        # Rule 8: Partial array usage (not all elements accessed)
        # If accessing most elements individually, unpacking may be clearer
        if not info.all_elements_accessed and info.has_individual_access:
            # Only unpack if accessing a significant portion (> 50%)
            access_ratio = len(info.element_accesses) / info.size
            if access_ratio > 0.5:
                return DecisionResult(
                    decision=UnpackingDecision.SHOULD_UNPACK,
                    reason=UnpackingReason.PARTIAL_ARRAY_USAGE,
                    explanation=(
                        f"Array '{info.array_name}' has {len(info.element_accesses)} "
                        f"of {info.size} elements accessed individually "
                        f"({access_ratio:.0%}). Unpacking for clarity."
                    ),
                )
            return DecisionResult(
                decision=UnpackingDecision.SHOULD_NOT_UNPACK,
                reason=UnpackingReason.PARTIAL_ARRAY_USAGE,
                explanation=(
                    f"Array '{info.array_name}' has only {len(info.element_accesses)} "
                    f"of {info.size} elements accessed individually "
                    f"({access_ratio:.0%}). Keeping as array."
                ),
            )

        # Default: Don't unpack (prefer simpler code)
        return DecisionResult(
            decision=UnpackingDecision.SHOULD_NOT_UNPACK,
            reason=UnpackingReason.NO_INDIVIDUAL_ACCESS,
            explanation=(
                f"Array '{info.array_name}' does not meet criteria for unpacking. "
                "Keeping as array for simpler code."
            ),
        )


def should_unpack_array(info: ArrayAccessInfo, *, verbose: bool = False) -> bool:
    """Convenience function to determine if an array should be unpacked.

    Args:
        info: Information about how the array is accessed
        verbose: If True, print the decision reasoning

    Returns:
        True if the array should be unpacked, False otherwise
    """
    decision_tree = UnpackingDecisionTree()
    result = decision_tree.decide(info)

    if verbose:
        print(f"Array '{info.array_name}' unpacking decision:")
        print(f"  Decision: {result.decision.name}")
        print(f"  Reason: {result.reason.name}")
        print(f"  Explanation: {result.explanation}")

    return result.should_unpack
