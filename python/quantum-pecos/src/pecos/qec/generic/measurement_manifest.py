# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Portable source metadata for tagged raw measurements and derived QEC parities."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pecos.qec.generic.scheduled import ScheduledCircuit


def measurement_manifest(plan: ScheduledCircuit) -> dict:
    """Describe a static output layout without compiling or changing the circuit.

    Tags are unique within one generated experiment; runtime columns carry shot
    occurrences. Carriers and moments describe the selected source schedule,
    not a compiled runtime's site assignments or clocks. Parity definitions are
    separate from raw events and retain their explicit affine constants.
    """
    aliases = plan.measurement_aliases
    rows = [row for row in plan.inspect() if row.measurement is not None]
    roles = plan.role_names or tuple(f"q[{i}]" for i in range(plan.num_qubits))
    role_indices = {name: i for i, name in enumerate(roles)}
    events = [
        {
            "index": index,
            "tag": aliases[row.measurement],
            "name": row.measurement,
            "value_type": "bool",
            "role": row.roles[0],
            "role_index": role_indices[row.roles[0]],
            "reference_carrier": row.carriers[0],
            "source_moment": row.moment,
            "source_operation": row.id,
            "origin": None if row.source is None else {**asdict(row.source), "id": row.source.id},
        }
        for index, row in enumerate(rows)
    ]
    derived = {}
    for kind, parities in (("detectors", plan.detectors), ("observables", plan.observables)):
        derived[kind] = [
            {
                "index": index,
                "label": parity.label,
                "measurement_tags": [events[i]["tag"] for i in parity.records],
                "constant": parity.constant,
                **({"coords": None if parity.coords is None else list(parity.coords)} if kind == "detectors" else {}),
            }
            for index, parity in enumerate(parities)
        ]
    return {
        "format": "pecos.scheduled-measurements",
        "version": 1,
        "schedule": plan.schedule.value,
        "measurement_count": plan.num_measurements,
        "raw_events": events,
        "derived_parities": derived,
        "source_citations": list(plan.citations),
        "tag_scope": "Unique raw event within one generated experiment; each result column contains shots",
        "parity_semantics": "XOR of measurement_tags and constant; no implicit ideal-baseline subtraction",
        "source_location_scope": "Source schedule moments, roles and reference carriers; not runtime sites or clocks",
        "compiled_measurement_binding_verified": False,
        "physical_schedule_verified": False,
        "noise_equivalence_verified": False,
    }
