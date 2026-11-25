# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Two C4-Helix blocks with transversal CNOT and discovered detectors.

Prepare both blocks, extract syndrome, apply CX from A to B on every data
slot, extract syndrome again, and read out. ASAP packing adds no barriers.
The accepted basis pairs ZZ, XX and ZX have deterministic logical readout.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pecos.qec import discover_detectors
from pecos.qec.circuit_steps import CircuitStep, OpType
from pecos.qec.helix.code import c4_helix_code
from pecos.qec.helix.gadgets import HelixAllocation, memory_gadgets, syndrome_round_gadgets, validate_basis
from pecos.qec.helix.lowering import helix_result_refs, lower_steps
from pecos.qec.helix.memory import discovered_definitions, set_discovered_metadata

if TYPE_CHECKING:
    from pecos_rslib.qec import DiscoveredDetectors

    from pecos.qec.dem_spec import Detector, Observable
    from pecos.quantum import TickCircuit


def _check_pair(basis_a: str, basis_b: str) -> None:
    """Validate both bases and reject the pair with random logical readout."""
    validate_basis(basis_a)
    validate_basis(basis_b)
    if basis_a + basis_b == "XZ":
        msg = (
            f"Random readout for basis pair {basis_a + basis_b!r}: "
            "CNOT makes a Bell pair; accepted pairs are ZZ, XX, ZX"
        )
        raise ValueError(msg)


def _block_gadget(block: str, gadget: tuple[str, list[CircuitStep]]) -> tuple[str, list[CircuitStep]]:
    """Place a single-block gadget in its allocation and namespace its labels."""
    name, steps = gadget
    offset = 0 if block == "A" else HelixAllocation.NUM_SLOTS
    placed = []
    for step in steps:
        label = f"{block}.{step.label}" if step.label else step.label
        placed.append(CircuitStep(step.op_type, [offset + q for q in step.qubits], label))
    return f"{block}_{name}", placed


def cnot_gadgets(basis_a: str, basis_b: str) -> list[tuple[str, list[CircuitStep]]]:
    """Return prepare, a rounds, transversal CNOT, b rounds, and readout gadgets."""
    _check_pair(basis_a, basis_b)
    memories = {block: memory_gadgets(basis, 1) for block, basis in (("A", basis_a), ("B", basis_b))}
    gadgets = [_block_gadget(block, memories[block][0]) for block in "AB"]
    gadgets.extend(_block_gadget(block, gadget) for block in "AB" for gadget in syndrome_round_gadgets("a"))
    gadgets.append(("cnot", [CircuitStep(OpType.CX, [q, HelixAllocation.NUM_SLOTS + q]) for q in range(20)]))
    gadgets.extend(_block_gadget(block, gadget) for block in "AB" for gadget in syndrome_round_gadgets("b"))
    gadgets.extend(_block_gadget(block, memories[block][-1]) for block in "AB")
    return gadgets


def cnot_conjugates(basis_a: str, basis_b: str) -> list[tuple[int, str]]:
    """Return dense paired injections at preparation boundaries in A0, A1, B0, B1 order.

    Each conjugate anticommutes with its own observable pulled back through
    the CNOT (an involution, so pull-back equals push-forward) and commutes
    with the other three. A conjugate has factors on both blocks only when
    both share a basis, so one boundary is valid on both: 1 for Z, 2 for X.
    """
    _check_pair(basis_a, basis_b)
    factor_blocks = {"ZZ": ("AB", "B"), "XX": ("A", "AB"), "ZX": ("A", "B")}[basis_a + basis_b]
    code = c4_helix_code()
    conjugates = []
    for basis, blocks in zip((basis_a, basis_b), factor_blocks, strict=True):
        for row in code.logical_x if basis == "Z" else code.logical_z:
            dense = "".join(
                row.ljust(HelixAllocation.NUM_SLOTS, "I") if block in blocks else "I" * HelixAllocation.NUM_SLOTS
                for block in "AB"
            )
            conjugates.append((1 if basis == "Z" else 2, dense))
    return conjugates


def _cnot_discovery(basis_a: str, basis_b: str) -> tuple[TickCircuit, DiscoveredDetectors]:
    """Discover both public representations on the same ASAP circuit."""
    steps = [step for _, gadget in cnot_gadgets(basis_a, basis_b) for step in gadget]
    circuit = lower_steps(steps, 2 * HelixAllocation.NUM_SLOTS).tick_circuit
    records = {ref: record for record, ref in enumerate(helix_result_refs(circuit))}
    code = c4_helix_code()
    observables = []
    for block, basis in (("A", basis_a), ("B", basis_b)):
        logicals = code.logical_z if basis == "Z" else code.logical_x
        observables.extend(
            [records[(f"{block}.data:{q}", 0)] for q, axis in enumerate(row) if axis != "I"] for row in logicals
        )
    found = discover_detectors(circuit, observables, cnot_conjugates(basis_a, basis_b))
    return circuit, found


def cnot_circuit(basis_a: str, basis_b: str) -> TickCircuit:
    """Build the two-block ASAP CNOT circuit with discovered parity metadata."""
    circuit, found = _cnot_discovery(basis_a, basis_b)
    set_discovered_metadata(circuit, found)
    return circuit


def cnot_definitions(basis_a: str, basis_b: str) -> tuple[list[Detector], list[Observable]]:
    """Return typed discovered parities with observable reference metadata."""
    circuit, found = _cnot_discovery(basis_a, basis_b)
    return discovered_definitions(circuit, found)
