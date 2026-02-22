# Copyright 2024 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Generic stabilizer check framework.

This module provides code-agnostic abstractions for stabilizer checks
that can be used across different QEC codes (surface codes, color codes, etc.).

The framework supports:
- Arbitrary Pauli operators (X, Y, Z) on data qubits
- Weight-2 to weight-n stabilizers
- Both bulk and boundary stabilizers
- Flexible ancilla allocation
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum


class PauliType(Enum):
    """Pauli operator type."""

    X = "X"
    Y = "Y"
    Z = "Z"

    def __str__(self) -> str:
        """Return string representation."""
        return self.value


@dataclass(frozen=True)
class PauliOperator:
    """A Pauli operator on a specific qubit.

    Attributes:
        qubit: The qubit index this operator acts on
        pauli: The Pauli type (X, Y, or Z)
    """

    qubit: int
    pauli: PauliType

    def __str__(self) -> str:
        """Return string representation (e.g., 'X0', 'Z3')."""
        return f"{self.pauli.value}{self.qubit}"


@dataclass(frozen=True)
class StabilizerCheck:
    """A generic stabilizer check.

    This represents a stabilizer measurement that can be used in any
    CSS or non-CSS quantum error correction code.

    Attributes:
        index: Unique identifier for this check
        paulis: Sequence of Pauli operators defining the stabilizer
        color: Optional color for color codes (red, green, blue)
        is_boundary: Whether this is a boundary stabilizer
    """

    index: int
    paulis: tuple[PauliOperator, ...]
    color: str | None = None
    is_boundary: bool = False

    @classmethod
    def from_string(
        cls,
        index: int,
        pauli_string: str,
        qubits: Sequence[int],
        *,
        color: str | None = None,
        is_boundary: bool = False,
    ) -> "StabilizerCheck":
        """Create a stabilizer check from a Pauli string.

        Args:
            index: Check index
            pauli_string: String like "XXXX", "ZZZZ", or "XYZX"
            qubits: Qubit indices the operators act on
            color: Optional color for color codes
            is_boundary: Whether this is a boundary stabilizer

        Returns:
            StabilizerCheck instance

        Raises:
            ValueError: If pauli_string length doesn't match qubits length
        """
        if len(pauli_string) == 1:
            pauli_string = pauli_string * len(qubits)

        if len(pauli_string) != len(qubits):
            msg = f"Pauli string length ({len(pauli_string)}) must match number of qubits ({len(qubits)})"
            raise ValueError(
                msg,
            )

        paulis = tuple(PauliOperator(q, PauliType(p)) for p, q in zip(pauli_string, qubits, strict=False))

        return cls(
            index=index,
            paulis=paulis,
            color=color,
            is_boundary=is_boundary,
        )

    @classmethod
    def x_check(
        cls,
        index: int,
        qubits: Sequence[int],
        *,
        is_boundary: bool = False,
    ) -> "StabilizerCheck":
        """Create an X-type stabilizer check."""
        return cls.from_string(index, "X", qubits, is_boundary=is_boundary)

    @classmethod
    def z_check(
        cls,
        index: int,
        qubits: Sequence[int],
        *,
        is_boundary: bool = False,
    ) -> "StabilizerCheck":
        """Create a Z-type stabilizer check."""
        return cls.from_string(index, "Z", qubits, is_boundary=is_boundary)

    @property
    def weight(self) -> int:
        """Number of qubits this check acts on."""
        return len(self.paulis)

    @property
    def qubits(self) -> tuple[int, ...]:
        """Qubit indices this check acts on."""
        return tuple(p.qubit for p in self.paulis)

    @property
    def pauli_string(self) -> str:
        """Get the Pauli string representation."""
        return "".join(p.pauli.value for p in self.paulis)

    def is_css(self) -> bool:
        """Check if this is a CSS stabilizer (all same Pauli type)."""
        if not self.paulis:
            return True
        first = self.paulis[0].pauli
        return all(p.pauli == first for p in self.paulis)

    def get_controlled_gate(self, pauli: PauliType) -> str:
        """Get the controlled gate name for a Pauli type."""
        return {
            PauliType.X: "cx",
            PauliType.Y: "cy",
            PauliType.Z: "cz",
        }[pauli]


@dataclass
class CheckSchedule:
    """A schedule for measuring multiple stabilizer checks.

    Organizes checks into rounds that can be measured in parallel,
    respecting qubit constraints.

    Attributes:
        rounds: List of rounds, each containing checks that can run in parallel
    """

    rounds: list[list[StabilizerCheck]]

    @classmethod
    def sequential(cls, checks: Sequence[StabilizerCheck]) -> "CheckSchedule":
        """Create a sequential schedule (one check per round)."""
        return cls(rounds=[[c] for c in checks])

    @classmethod
    def parallel_by_color(
        cls,
        checks: Sequence[StabilizerCheck],
    ) -> "CheckSchedule":
        """Create a schedule that parallelizes checks by color."""
        by_color: dict[str | None, list[StabilizerCheck]] = {}
        for check in checks:
            color = check.color
            if color not in by_color:
                by_color[color] = []
            by_color[color].append(check)

        max_len = max(len(group) for group in by_color.values())

        rounds = []
        for i in range(max_len):
            round_checks = [color_checks[i] for color_checks in by_color.values() if i < len(color_checks)]
            if round_checks:
                rounds.append(round_checks)

        return cls(rounds=rounds)

    def total_checks(self) -> int:
        """Total number of checks in the schedule."""
        return sum(len(r) for r in self.rounds)

    def num_rounds(self) -> int:
        """Number of rounds in the schedule."""
        return len(self.rounds)
