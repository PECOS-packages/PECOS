# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Slot-aware ASAP lowering for C4-Helix (arXiv:2609.03194).

RELABEL implements swap-out by exchanging slot bindings without emitting gates.
Gadgets choose dedicated ancillas and leave their corrections unapplied; this
lowering preserves physical-qubit order and reports actual measurement order.
"""

from collections.abc import Iterable
from dataclasses import dataclass

from pecos.qec.circuit_steps import CircuitStep, OpType
from pecos.quantum import TickCircuit


@dataclass(frozen=True)
class LoweredHelixCircuit:
    """A physical circuit, its record labels, and the final slot-to-qubit map."""

    tick_circuit: TickCircuit
    record_labels: tuple[str, ...]
    final_layout: tuple[int, ...]


def lower_steps(steps: Iterable[CircuitStep], num_slots: int) -> LoweredHelixCircuit:
    """Pack gates ASAP per physical qubit, respecting explicit global TICK barriers.

    Every gate step has one slot, except CX and RELABEL, which have two.
    Measurements carry scalar labels. COMMENT steps do not affect scheduling.
    """
    if num_slots < 0:
        msg = f"Invalid num_slots {num_slots!r}"
        raise ValueError(msg)
    circuit = TickCircuit()
    layout = list(range(num_slots))
    last = [-1] * num_slots
    labels: list[dict[int, str]] = []
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
        tick_index = max(last[q] for q in qubits) + 1
        while circuit.num_ticks() <= tick_index:
            circuit.tick()
            labels.append({})
        tick = circuit.tick_at(tick_index)
        if op == OpType.CX:
            tick.cx([tuple(qubits)])
        else:
            getattr(tick, gates[op])(qubits)
        for qubit in qubits:
            last[qubit] = tick_index
        if op == OpType.MEASURE:
            labels[tick_index][qubits[0]] = step.label
    emission = iter(circuit.measurement_emission())
    record_labels = tuple(tick_labels[int(next(emission)[0])] for tick_labels in labels for _ in tick_labels)
    return LoweredHelixCircuit(circuit, record_labels, tuple(layout))
