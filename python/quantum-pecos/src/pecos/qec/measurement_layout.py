# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Read abstract measurements as scalar tags or array-element result references.

A result reference is ``("scalar", tag)`` or ``("array", tag, element)``.
Each code family supplies its label vocabulary as a classifier while sharing
this one reader and its measurement-emission checks.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from pecos_rslib.quantum import TickCircuit

ResultRef = tuple[str, str] | tuple[str, str, int]


def abstract_result_refs(
    abstract_tc: TickCircuit,
    new_classifier: Callable[[], Callable[[str, Callable[[str], object]], ResultRef]],
    *,
    family: str,
) -> list[ResultRef]:
    """Classify each single-qubit measurement in tick and gate-batch order.

    ``new_classifier`` is called once per walk, so a classifier that keeps state
    (such as per-round ordinals) always starts fresh; a used instance cannot be
    passed by mistake.
    """
    classify = new_classifier()
    refs: list[ResultRef] = []
    measured_qubits: list[int] = []
    measurement_gate_types = {"MZ", "MX", "MPZ", "MeasureFree"}
    for tick_index in range(abstract_tc.num_ticks()):
        tick = abstract_tc.get_tick(tick_index)
        if tick is None:
            continue
        for gate_index, gate in enumerate(tick.gate_batches()):
            gate_type = str(getattr(gate, "gate_type", "")).rsplit(".", maxsplit=1)[-1]
            if gate_type not in measurement_gate_types:
                continue
            qubits = list(gate.qubits)
            measured_qubits.extend(qubits)
            meta = partial(abstract_tc.get_gate_meta, tick_index, gate_index)
            label = str(meta("label") or "")
            if len(qubits) != 1:
                msg = (
                    f"{family} measurement label {label!r} names one measurement but its gate "
                    f"measures {len(qubits)} qubits"
                )
                raise ValueError(msg)
            refs.append(classify(label, meta))
    emission_qubits = [qubit for qubit, _ in abstract_tc.measurement_emission()]
    if len(refs) != len(emission_qubits) or measured_qubits != emission_qubits:
        msg = f"{family} measurement result refs disagree with the shared reader's measurement emission"
        raise ValueError(msg)
    return refs
