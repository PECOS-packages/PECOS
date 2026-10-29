# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""C4-Helix memory and logical gate gadgets from Berthusen et al., arXiv:2609.03194.

Use the first four independent weight-8 rows per type in paper order, modulo
that type's five C4 rows. Rounds alternate variants a/b; the swap-out frame
X(d3)^mz Z(d2)^mx remains unapplied. The 76 slots reserve 20 data, ten C4
ancillas, four dedicated cat qubits for each of eight weight-8 checks, and
two flags for each of seven physical CZs. Memory uses only the first 62 slots.
"""

from dataclasses import dataclass
from typing import ClassVar

from pecos_rslib import PauliStabilizerGroup, PauliString

from pecos.qec.circuit_steps import CircuitStep, OpType
from pecos.qec.helix.code import c4_helix_code
from pecos.qec.helix.logical import c4_helix_gates, logical_token_ops


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
    """Dedicated slots for data, C4 swap-out ancillas, weight-8 cats, and CZ flags."""

    NUM_SLOTS: ClassVar[int] = 76

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

    def cz_flags(self, index: int) -> tuple[int, int]:
        """Return (zero_flag, plus_flag) for physical CZ 0..6."""
        base = 62 + 2 * _index(index, 7, "index")
        return base, base + 1


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


def c4_round_steps(block: int, variant: str) -> list[CircuitStep]:
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
            CircuitStep(OpType.MEASURE, [plus], f"c4x:{block}"),
            CircuitStep(OpType.MEASURE, [zero], f"c4z:{block}"),
        ],
    )
    return steps


def weight8_check_steps(basis: str, index: int) -> list[CircuitStep]:
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
        label = f"{basis.lower()}8:{index}" if k == 0 else f"flag:{basis.lower()}{index}.{k}"
        steps.append(CircuitStep(OpType.MEASURE, [qubit], label))
    return steps


def memory_gadgets(basis: str, rounds: int) -> list[tuple[str, list[CircuitStep]]]:
    """Return named gadget step lists in memory execution order."""
    validate_basis(basis)
    if rounds < 1:
        msg = f"Expected rounds >= 1, got {rounds!r}"
        raise ValueError(msg)
    steps = [CircuitStep(OpType.PREP, [q]) for q in range(20)]
    if basis == "X":
        steps.extend(CircuitStep(OpType.H, [q]) for q in range(20))
    gadgets = [(f"prepare_{basis.lower()}", steps)]
    for round_index in range(rounds):
        variant = "a" if round_index % 2 == 0 else "b"
        gadgets.extend(syndrome_round_gadgets(variant))
    steps = []
    for slot in range(20):
        if basis == "X":
            steps.append(CircuitStep(OpType.H, [slot]))
        steps.append(CircuitStep(OpType.MEASURE, [slot], f"data:{slot}"))
    gadgets.append((f"readout_{basis.lower()}", steps))
    return gadgets


def memory_steps(basis: str, rounds: int) -> list[CircuitStep]:
    """Prepare data, alternate C4 variants with X/Z checks, and read out all slots."""
    return [step for _, steps in memory_gadgets(basis, rounds) for step in steps]


def syndrome_round_gadgets(variant: str) -> list[tuple[str, list[CircuitStep]]]:
    """Return one C4 swap-out round followed by the eight flagged weight-8 checks."""
    gadgets = [(f"c4_{block}_{variant}", c4_round_steps(block, variant)) for block in range(5)]
    for basis in ("X", "Z"):
        gadgets.extend((f"weight8_{basis.lower()}_{index}", weight8_check_steps(basis, index)) for index in range(4))
    return gadgets


def logical_gate_steps(token: str) -> list[CircuitStep]:
    """Expand a paper gate or nonidentity canonical logical Pauli into slot steps.

    Every physical CZ uses the two-flag gadget of CSD eq:cz_flag, also shown
    in Berthusen et al., arXiv:2609.03194, Fig. ft_phase_gate(b).
    """
    ops = logical_token_ops(token)
    if token not in c4_helix_gates():
        return [
            CircuitStep(op, list(qubits))
            for axis, qubits in ops
            for op in ({"x": (OpType.X,), "z": (OpType.Z,), "y": (OpType.X, OpType.Z)}[axis])
        ]
    steps = []
    cz_index = 0
    allocation = HelixAllocation()
    single = {"swap": OpType.RELABEL, "h": OpType.H, "sz": OpType.SZ, "szdg": OpType.SZDG}
    for op, qubits in ops:
        if op in single:
            steps.append(CircuitStep(single[op], list(qubits)))
        elif op == "cz":
            q1, q2 = qubits
            zero, plus = allocation.cz_flags(cz_index)
            steps.extend(
                [
                    CircuitStep(OpType.PREP, [zero]),
                    CircuitStep(OpType.PREP, [plus]),
                    CircuitStep(OpType.H, [plus]),
                    CircuitStep(OpType.CX, [q1, zero]),
                    CircuitStep(OpType.CX, [plus, q2]),
                    CircuitStep(OpType.CZ, [q1, q2]),
                    CircuitStep(OpType.CZ, [q1, plus]),
                    CircuitStep(OpType.CX, [q1, zero]),
                    CircuitStep(OpType.CX, [plus, q2]),
                    CircuitStep(OpType.MEASURE, [zero], f"czf:{cz_index}.z"),
                    CircuitStep(OpType.H, [plus]),
                    CircuitStep(OpType.MEASURE, [plus], f"czf:{cz_index}.x"),
                ],
            )
            cz_index += 1
        else:
            msg = f"Unsupported logical gate operation {op!r} in {token!r}"
            raise ValueError(msg)
    return steps
