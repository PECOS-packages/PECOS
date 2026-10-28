# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""C4-Helix memory circuits with discovered detectors (arXiv:2609.03194).

The gadgets select four independent weight-8 checks per type in paper order,
alternate C4 variants a/b, reserve 62 slots, and leave swap-out corrections
unapplied. Discovery accounts for their frame and projective preparation signs.
"""

import json

from pecos.qec import discover_detectors
from pecos.qec.helix.code import c4_helix_code
from pecos.qec.helix.gadgets import HelixAllocation, memory_steps, validate_basis
from pecos.qec.helix.lowering import lower_steps
from pecos.quantum import TickCircuit


def memory_conjugates(basis: str) -> list[tuple[int, str]]:
    """Return paired (before_tick, dense Pauli) logical injections for memory.

    Z memory uses logical X at boundary 1, after data PREP. X memory uses
    logical Z at boundary 2, after PREP and H. At these boundaries data slots
    still name physical qubits 0..19. The order matches the two observables.
    """
    validate_basis(basis)
    code = c4_helix_code()
    return [(1 if basis == "Z" else 2, row) for row in (code.logical_x if basis == "Z" else code.logical_z)]


def memory_circuit(basis: str, rounds: int) -> TickCircuit:
    """Build memory with absolute-record metadata and logical-free detectors.

    ``memory_conjugates(basis)`` supplies the paired logical injections used
    by discovery; detector references must all be zero for this metadata format.
    """
    lowered = lower_steps(memory_steps(basis, rounds), HelixAllocation.NUM_SLOTS)
    circuit = lowered.tick_circuit
    records = {label: record for record, label in enumerate(lowered.record_labels)}
    code = c4_helix_code()
    logicals = code.logical_z if basis == "Z" else code.logical_x
    observables = [[records[f"data:{q}"] for q, axis in enumerate(row) if axis != "I"] for row in logicals]
    found = discover_detectors(circuit, observables, memory_conjugates(basis))
    detectors = []
    for index, (detector_records, reference) in enumerate(found.detectors):
        if reference != 0:
            msg = f"Detector {index} has nonzero reference {reference}: {detector_records}"
            raise ValueError(msg)
        detectors.append({"id": index, "records": detector_records})
    circuit.set_meta("detectors", json.dumps(detectors))
    circuit.set_meta(
        "observables",
        json.dumps(
            [
                {"id": index, "records": obs_records, "reference": ref}
                for index, (obs_records, ref) in enumerate(found.observables)
            ],
        ),
    )
    circuit.set_meta("num_measurements", str(len(lowered.record_labels)))
    circuit.set_meta("num_detectors", str(len(detectors)))
    return circuit
