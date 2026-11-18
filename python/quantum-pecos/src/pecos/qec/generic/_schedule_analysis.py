# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Backend-independent source-order contracts and operation identities."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pecos.qec.generic.scheduled import ScheduledCircuit


def _rescheduling_contract(circuit: ScheduledCircuit) -> tuple:
    """Compare each wire within unitary regions and retain boundary moments exactly."""
    regions, wires = [], [[] for _ in range(circuit.num_qubits)]
    for moment in circuit.moments:
        if any(op.name in {"R", "M", "RELABEL"} for op in moment):
            regions.append((tuple(tuple(w) for w in wires), moment))
            wires = [[] for _ in wires]
        else:
            for op in moment:
                for q in op.qubits:
                    wires[q].append(op)
    regions.append((tuple(tuple(w) for w in wires), ()))
    return (
        circuit.num_qubits,
        circuit.detectors,
        circuit.observables,
        circuit.role_names,
        circuit.measurement_names,
        tuple(regions),
    )


def _source_listing(circuit: ScheduledCircuit) -> tuple:
    return tuple((op, op.origin) for moment in circuit.moments for op in moment)


def _source_placements(circuit: ScheduledCircuit) -> tuple[dict, list[frozenset]]:
    # Per-role occurrence ordinals survive disjoint source repacking. They are
    # source operation identities, not program handles or runtime site IDs.
    occurrences = Counter()
    rows, groups = {}, []
    for moment, operations in enumerate(circuit.moments):
        members = []
        for operation, op in enumerate(operations):
            anchor = op.qubits[0]
            identity = f"q[{anchor}]/op[{occurrences[anchor]}]"
            rows[identity] = {
                "gate": op.name,
                "roles": list(op.qubits),
                "role_names": [circuit.role_names[q] if circuit.role_names else f"q[{q}]" for q in op.qubits],
                "source_origin": op.origin.id if op.origin else None,
                "moment": moment,
                "operation": operation,
                "listing_index": len(rows),
            }
            members.append(identity)
            occurrences.update(op.qubits)
        groups.append(frozenset(members))
    return rows, groups
