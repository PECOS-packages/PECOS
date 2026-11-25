# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Native TickCircuit lowering of named, scheduled QEC experiments."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pecos.qec.generic.scheduled import ScheduledCircuit
    from pecos.quantum import TickCircuit


def lower_tick_circuit(plan: ScheduledCircuit) -> TickCircuit:
    """Preserve selected physical moments, carriers, and measurement identities."""
    from pecos.quantum import TickCircuit  # noqa: PLC0415 - optional native backend

    if any(not p.records for p in (*plan.detectors, *plan.observables)):
        msg = "TickCircuit detector/observable definitions require nonempty parities"
        raise ValueError(msg)
    circuit = TickCircuit()
    carriers = list(range(plan.num_qubits))
    names = plan.role_names or tuple(f"q[{i}]" for i in range(plan.num_qubits))
    rows = iter(plan.inspect())
    relabels = []
    event = 0
    single = {
        "R": "pz",
        "H": "h",
        "S": "sz",
        "S_DAG": "szdg",
        "X": "x",
        "Z": "z",
        "C_XYZ": "f",
        "C_ZYX": "fdg",
        "SQRT_X": "sx",
    }
    for index, moment in enumerate(plan.moments):
        tick = circuit.tick() if not moment or any(op.name != "RELABEL" for op in moment) else None
        origins = []
        for op in moment:
            row = next(rows)
            physical = [carriers[q] for q in op.qubits]
            if op.name == "RELABEL":
                a, b = op.qubits
                carriers[a], carriers[b] = carriers[b], carriers[a]
                relabels.append({"moment": index, "id": row.id, "roles": list(row.roles), "before": physical})
                continue
            if op.name == "M":
                tick.mz_with_ids(physical, [event])
                event += 1
            elif op.name in {"CX", "CZ"}:
                getattr(tick, op.name.lower())([tuple(physical)])
            else:
                getattr(tick, single[op.name])(physical)
            origins.append({"id": row.id, "roles": list(row.roles), "carriers": physical, "gate": op.name})
        if tick is not None:
            circuit.set_tick_meta(tick.index(), "qec_moment", index)
            circuit.set_tick_meta(tick.index(), "qec_operations", origins)
    emissions = circuit.measurement_emission()
    positions = {identity: pos for pos, (_, identity) in enumerate(emissions)}
    if set(positions) != set(range(event)) or len(emissions) != event:
        msg = "TickCircuit did not retain canonical measurement identities"
        raise RuntimeError(msg)
    for parity in plan.detectors:
        circuit.add_detector([positions[i] - event for i in parity.records], coords=parity.coords, label=parity.label)
    for index, parity in enumerate(plan.observables):
        # Native labels L<n> select decoder IDs; authored labels are opaque names.
        label = parity.label
        if label.startswith("L") and label != f"L{index}":
            label = f"qec:{label}"
        circuit.add_observable(
            [positions[i] - event for i in parity.records],
            observable_id=index,
            label=label,
        )
    for key, parities in (("detectors", plan.detectors), ("observables", plan.observables)):
        if any(p.constant for p in parities):
            entries = json.loads(circuit.get_meta(key))
            for entry in entries:
                entry["parity_constant"] = int(parities[entry["id"]].constant)
            circuit.set_meta(key, json.dumps(entries))
    circuit.metas(
        {
            "qec_schedule": plan.schedule.value,
            "qec_roles": list(names),
            "qec_measurement_names": list(plan.measurement_aliases),
            "qec_measurement_tags": list(plan.measurement_aliases.values()),
            "qec_observable_labels": [p.label for p in plan.observables],
            "qec_output_carriers": carriers,
            "qec_relabels": relabels,
            "qec_citations": list(plan.citations),
        },
    )
    return circuit


def evaluate_tick_circuit(
    circuit: TickCircuit,
    record: Sequence[int | bool],
) -> tuple[tuple[bool, ...], tuple[bool, ...]]:
    """Evaluate native emitted records including explicit parity constants.

    Results follow the ID order of circuit_definitions(). Its support-only
    native definitions do not themselves apply constants. This helper combines
    those resolved positions with the serialized parity_constant metadata.
    """
    from pecos.qec.dem_spec import parity_constants  # noqa: PLC0415

    definitions = circuit.circuit_definitions()
    if len(record) != definitions["num_measurements"] or any(value not in (0, 1) for value in record):
        msg = "Expected one Boolean value per emitted measurement"
        raise ValueError(msg)
    result = []
    for key in ("detectors", "observables"):
        constants = parity_constants(circuit.get_meta(key), key[:-1])
        result.append(
            tuple(
                bool((sum(record[i] for i in entry["measurements"]) % 2) ^ constants.get(entry["id"], 0))
                for entry in definitions[key]
            ),
        )
    return tuple(result)
