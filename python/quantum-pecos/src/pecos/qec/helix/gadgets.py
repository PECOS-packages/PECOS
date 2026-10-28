# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""C4-Helix memory gadgets from Berthusen et al., arXiv:2609.03194.

Use the first four independent weight-8 rows per type in paper order, modulo
that type's five C4 rows. Rounds alternate variants a/b; the swap-out frame
X(d3)^mz Z(d2)^mx remains unapplied. The 62 slots reserve 20 data, ten C4
ancillas, and four dedicated cat qubits for each of eight weight-8 checks.
"""

from dataclasses import dataclass
from typing import ClassVar

from pecos_rslib import PauliStabilizerGroup, PauliString

from pecos.qec.circuit_steps import CircuitStep, OpType
from pecos.qec.helix.code import c4_helix_code


def validate_basis(basis: str) -> None:
    """Reject a memory or check basis other than X or Z."""
    if basis not in {"X", "Z"}:
        msg = f"Expected basis X or Z, got {basis!r}"
        raise ValueError(msg)


def _index(value: int, limit: int, name: str) -> int:
    if not 0 <= value < limit:
        msg = f"Invalid {name} {value!r}; expected 0..{limit - 1}"
        raise ValueError(msg)
    return value


@dataclass(frozen=True)
class HelixAllocation:
    """Dedicated slots for data, C4 swap-out ancillas, and weight-8 cats."""

    NUM_SLOTS: ClassVar[int] = 62

    def data(self, slot: int) -> int:
        """Return data slot 0..19."""
        return _index(slot, 20, "slot")

    def c4_plus(self, block: int) -> int:
        """Return the C4 plus ancilla slot for block 0..4."""
        return 20 + 2 * _index(block, 5, "block")

    def c4_zero(self, block: int) -> int:
        """Return the C4 zero ancilla slot for block 0..4."""
        return self.c4_plus(block) + 1

    def check(self, basis: str, index: int) -> tuple[int, int, int, int]:
        """Return (a, f1, f2, f3) for weight-8 check 0..3."""
        validate_basis(basis)
        base = (30 if basis == "X" else 46) + 4 * _index(index, 4, "index")
        return base, base + 1, base + 2, base + 3


def chosen_weight8_supports(basis: str) -> tuple[tuple[int, ...], ...]:
    """Choose independent weight-8 checks in paper order modulo the C4 rows."""
    validate_basis(basis)
    rows = [row for row in c4_helix_code().stabilizers if set(row) <= {"I", basis}]
    group = PauliStabilizerGroup([PauliString.from_str(row) for row in rows if row.count(basis) == 4])
    chosen = []
    for row in rows:
        if row.count(basis) == 8:
            pauli = PauliString.from_str(row)
            if not group.contains(pauli):
                group.add_generator(pauli)
                chosen.append(tuple(q for q, axis in enumerate(row) if axis == basis))
                if len(chosen) == 4:
                    break
    return tuple(chosen)


def c4_round_steps(block: int, variant: str, round_index: int) -> list[CircuitStep]:
    """Swap out two data slots and measure C4 signs, leaving corrections unapplied."""
    allocation = HelixAllocation()
    plus, zero = allocation.c4_plus(block), allocation.c4_zero(block)
    if variant not in {"a", "b"}:
        msg = f"Expected variant a or b, got {variant!r}"
        raise ValueError(msg)
    slots = c4_helix_code().blocks[block]
    d0, d1, d2, d3 = slots if variant == "a" else (*slots[2:], *slots[:2])
    steps = [CircuitStep(OpType.PREP, [plus]), CircuitStep(OpType.H, [plus]), CircuitStep(OpType.PREP, [zero])]
    pairs = [(d0, zero), (plus, d1), (d1, zero), (plus, d0), (d2, zero), (plus, d3), (zero, d3), (d2, plus)]
    steps.extend(CircuitStep(OpType.CX, list(pair)) for pair in pairs)
    steps.extend(
        [
            CircuitStep(OpType.RELABEL, [d2, plus]),
            CircuitStep(OpType.RELABEL, [d3, zero]),
            CircuitStep(OpType.H, [plus]),
            CircuitStep(OpType.MEASURE, [plus], f"c4x:{block}:{round_index}"),
            CircuitStep(OpType.MEASURE, [zero], f"c4z:{block}:{round_index}"),
        ],
    )
    return steps


def weight8_check_steps(basis: str, index: int, round_index: int) -> list[CircuitStep]:
    """Measure a weight-8 check with three flags; Z is the Hadamard dual of X."""
    ancillas = HelixAllocation().check(basis, index)
    a, f1, f2, f3 = ancillas
    support = chosen_weight8_supports(basis)[index]
    steps = []
    for k, qubit in enumerate(ancillas):
        steps.append(CircuitStep(OpType.PREP, [qubit]))
        if (k == 0) == (basis == "X"):
            steps.append(CircuitStep(OpType.H, [qubit]))
    pairs = [(a, f1), (a, f2), (f1, f3)]
    pairs.extend(zip(ancillas * 2, support, strict=True))
    pairs.extend([(a, f3), (f1, f2), (a, f1)])
    steps.extend(CircuitStep(OpType.CX, [c, t] if basis == "X" else [t, c]) for c, t in pairs)
    for k, qubit in enumerate(ancillas):
        if (k == 0) == (basis == "X"):
            steps.append(CircuitStep(OpType.H, [qubit]))
        label = (
            f"{basis.lower()}8:{index}:{round_index}" if k == 0 else f"flag:{basis.lower()}{index}.{k}:{round_index}"
        )
        steps.append(CircuitStep(OpType.MEASURE, [qubit], label))
    return steps


def memory_steps(basis: str, rounds: int) -> list[CircuitStep]:
    """Prepare data, alternate C4 variants with X/Z checks, and read out all slots."""
    validate_basis(basis)
    if rounds < 1:
        msg = f"Expected rounds >= 1, got {rounds!r}"
        raise ValueError(msg)
    steps = [CircuitStep(OpType.PREP, [q]) for q in range(20)]
    if basis == "X":
        steps.extend(CircuitStep(OpType.H, [q]) for q in range(20))
    for round_index in range(rounds):
        for block in range(5):
            steps.extend(c4_round_steps(block, "a" if round_index % 2 == 0 else "b", round_index))
        for check_basis in ("X", "Z"):
            for index in range(4):
                steps.extend(weight8_check_steps(check_basis, index, round_index))
    for slot in range(20):
        if basis == "X":
            steps.append(CircuitStep(OpType.H, [slot]))
        steps.append(CircuitStep(OpType.MEASURE, [slot], f"data:{slot}"))
    return steps
