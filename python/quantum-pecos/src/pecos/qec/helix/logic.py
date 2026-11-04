# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""C4-Helix logical circuits from Berthusen et al., arXiv:2609.03194.

Prepare, extract syndrome, apply gates, and read out in the prepared basis.
Only S0S1 is followed by another syndrome round; variants alternate a/b.
The net symplectic action must preserve the prepared logical span, so readout
is deterministic. Discovery conjugates are pulled back through the inverse
logical action to the boundary just after preparation, using dense products.
For two S0S1 gates, full-noise X distance is the owner-accepted floor of four
(2026-10-29): round faults feeding a gate spread through its CZs. Gate faults
alone give a weight-six upper bound (a stim witness); full-noise Z bounds
are 5 <= d <= 6.
"""

from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING

from pecos_rslib import PauliString

from pecos.qec import discover_detectors
from pecos.qec.helix.code import c4_helix_code
from pecos.qec.helix.compile import parse_signed_logical_pauli
from pecos.qec.helix.gadgets import (
    HelixAllocation,
    logical_gate_steps,
    memory_gadgets,
    syndrome_round_gadgets,
    validate_basis,
)
from pecos.qec.helix.logical import (
    LogicalGate,
    logical_pauli_string,
    logical_token_ops,
    symplectic_action,
)
from pecos.qec.helix.lowering import helix_result_refs, lower_steps
from pecos.qec.helix.memory import set_discovered_metadata

if TYPE_CHECKING:
    from pecos.qec.circuit_steps import CircuitStep
    from pecos.qec.helix.compile import Matrix
    from pecos.quantum import TickCircuit


def _token_ops(tokens: tuple[str, ...]) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Concatenate physical operations in token order, validating every token."""
    return tuple(op for token in tokens for op in logical_token_ops(token))


def _checked_action(basis: str, tokens: tuple[str, ...]) -> Matrix:
    """Return the word's net symplectic action, rejecting words with random readout."""
    validate_basis(basis)
    matrix = symplectic_action(c4_helix_code(), LogicalGate("word", _token_ops(tokens)))
    prepared = range(2) if basis == "X" else range(2, 4)
    opposite = range(2, 4) if basis == "X" else range(2)
    if any(matrix[row][col] for row in opposite for col in prepared):
        msg = f"Random readout for basis {basis!r} and tokens {tokens!r}"
        raise ValueError(msg)
    return matrix


def logic_gadgets(basis: str, tokens: tuple[str, ...]) -> list[tuple[str, list[CircuitStep]]]:
    """Return named gadgets, rejecting words with random readout in the given basis."""
    _checked_action(basis, tokens)
    memory = memory_gadgets(basis, 1)
    gadgets = memory[:-1]
    round_index = 1
    for name in tokens:
        gadgets.append((name, logical_gate_steps(name)))
        if name == "S0S1":
            gadgets.extend(syndrome_round_gadgets("a" if round_index % 2 == 0 else "b"))
            round_index += 1
    gadgets.append(memory[-1])
    return gadgets


def logic_conjugates(basis: str, tokens: tuple[str, ...]) -> list[tuple[int, str]]:
    """Pull readout conjugates back to preparation as dense sign-free Pauli products.

    Each injection flips only its paired observable. Using a forward image or
    overwriting overlapping factors can silently lower the discovered distance.
    """
    matrix = _checked_action(basis, tokens)
    code = c4_helix_code()
    stabilizers = [PauliString.from_str(row) for row in code.stabilizers]
    candidates = []
    for letters in product("IXYZ", repeat=code.k):
        spelling = "".join(f"{letter}{j}" for j, letter in enumerate(letters) if letter != "I") or "I"
        candidates.append((spelling, parse_signed_logical_pauli("+" + spelling, code.k)[1]))
    conjugates = []
    for index in range(2) if basis == "Z" else range(2, 4):
        target = tuple(int(j == index) for j in range(4))
        spelling = next(
            spelling
            for spelling, bits in candidates
            if tuple(sum(a * b for a, b in zip(row, bits, strict=True)) % 2 for row in matrix) == target
        )
        dense = logical_pauli_string(spelling)
        pauli = PauliString.from_str(dense)
        if not all(pauli.commutes_with(row) for row in stabilizers):
            msg = f"Nonlogical conjugate for token word {tokens!r}: {dense}"
            raise ValueError(msg)
        conjugates.append((1 if basis == "Z" else 2, dense))
    return conjugates


def logic_circuit(basis: str, tokens: tuple[str, ...]) -> TickCircuit:
    """Build an ASAP logic circuit with discovered detectors and signed references."""
    steps = [step for _, gadget in logic_gadgets(basis, tokens) for step in gadget]
    circuit = lower_steps(steps, HelixAllocation.NUM_SLOTS).tick_circuit
    records = {ref: record for record, ref in enumerate(helix_result_refs(circuit))}
    code = c4_helix_code()
    logicals = code.logical_z if basis == "Z" else code.logical_x
    observables = [[records[(f"data:{q}", 0)] for q, axis in enumerate(row) if axis != "I"] for row in logicals]
    found = discover_detectors(circuit, observables, logic_conjugates(basis, tokens))
    set_discovered_metadata(circuit, found)
    return circuit
