# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Slot-aware step-ordered and ASAP lowering for C4-Helix (arXiv:2609.03194).

RELABEL implements swap-out by exchanging slot bindings without emitting gates.
Gadgets choose dedicated ancillas and leave their corrections unapplied; this
lowering preserves physical-qubit order and reports actual measurement order.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pecos.qec.circuit_steps import OpType
from pecos.qec.measurement_layout import abstract_result_refs
from pecos.quantum import TickCircuit

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from pecos.qec.circuit_steps import CircuitStep
    from pecos.qec.measurement_layout import ResultRef


@dataclass(frozen=True)
class LoweredHelixCircuit:
    """A physical circuit and the final slot-to-qubit map."""

    tick_circuit: TickCircuit
    final_layout: tuple[int, ...]


def lower_steps(
    steps: Iterable[CircuitStep],
    num_slots: int,
    *,
    schedule: str = "asap",
) -> LoweredHelixCircuit:
    """Pack gates ASAP per physical qubit or emit one gate per tick in step order.

    Every gate step has one slot, except CX and RELABEL, which have two.
    Measurements carry scalar labels. TICK is a barrier only for ASAP packing;
    neither TICK nor COMMENT emits a gate or tick.
    """
    if schedule not in {"asap", "step"}:
        msg = f"Invalid Helix schedule {schedule!r}; expected asap or step"
        raise ValueError(msg)
    if num_slots < 0:
        msg = f"Invalid num_slots {num_slots!r}"
        raise ValueError(msg)
    circuit = TickCircuit()
    layout = list(range(num_slots))
    last = [-1] * num_slots
    gates = {OpType.PREP: "pz", OpType.H: "h", OpType.X: "x", OpType.Z: "z", OpType.CX: "cx", OpType.MEASURE: "mz"}
    for step in steps:
        op = step.op_type
        if op == OpType.COMMENT:
            continue
        if op == OpType.TICK:
            last = [max(last, default=-1)] * num_slots
            continue
        if op == OpType.RELABEL:
            step.__post_init__()  # Re-run construction validation: steps are mutable.
        elif op not in gates:
            msg = f"Unsupported Helix operation: {op.name} in step {step!r}"
            raise ValueError(msg)
        arity = 2 if op in {OpType.CX, OpType.RELABEL} else 1
        if len(step.qubits) != arity or len(set(step.qubits)) != arity:
            msg = f"Invalid qubits for {op.name} step: {step.qubits!r}"
            raise ValueError(msg)
        if any(not 0 <= slot < num_slots for slot in step.qubits):
            msg = f"Slots out of range for num_slots={num_slots}: {step.qubits!r}"
            raise ValueError(msg)
        if op == OpType.RELABEL:
            first, second = step.qubits
            layout[first], layout[second] = layout[second], layout[first]
            continue
        qubits = [layout[slot] for slot in step.qubits]
        tick_index = circuit.num_ticks() if schedule == "step" else max(last[q] for q in qubits) + 1
        while circuit.num_ticks() <= tick_index:
            circuit.tick()
        tick = circuit.tick_at(tick_index)
        if op == OpType.CX:
            tick.cx([tuple(qubits)])
        elif op == OpType.MEASURE:
            tick.add_gate("MZ", qubits).metas({"label": step.label})
        else:
            getattr(tick, gates[op])(qubits)
        for qubit in qubits:
            last[qubit] = tick_index
    return LoweredHelixCircuit(circuit, tuple(layout))


def helix_result_classifier() -> Callable[[str, Callable[[str], object]], ResultRef]:
    """Create a shared-reader classifier for the complete round-free Helix grammar."""

    def classify(label: str, _meta: Callable[[str], object]) -> ResultRef:
        if re.fullmatch(r"(?:c4[xz]:[0-4]|[xz]8:[0-3]|flag:[xz][0-3]\.[1-3]|data:(?:[0-9]|1[0-9]))", label):
            return ("scalar", label)
        msg = f"Invalid Helix measurement label {label!r}"
        raise ValueError(msg)

    return classify


def helix_result_refs(tc: TickCircuit) -> list[tuple[str, int]]:
    """Read (tag, occurrence) references in the circuit's measurement record order."""
    occurrences: Counter[str] = Counter()
    refs = []
    for ref in abstract_result_refs(tc, helix_result_classifier, family="helix"):
        tag = ref[1]
        refs.append((tag, occurrences[tag]))
        occurrences[tag] += 1
    return refs
