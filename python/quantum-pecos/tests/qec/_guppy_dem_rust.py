# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Convert validated facade inputs at the Rust preparation boundary."""

from __future__ import annotations

from pecos._engine_builders import _configure_selene_runtime
from pecos.qec.dem import _guppy_dem_evidence
from pecos.qec.dem_spec import Detector, Observable, _definition_record
from pecos_rslib.qec import _prepare_guppy_dem, _SeleneRuntimeSpec


def prepare(program, num_qubits, *, seed=7, settings=None, detectors=None, observables=None, analysis=None, **options):
    definitions = dict(options)
    for role, values, expected in (("detectors", detectors, Detector), ("observables", observables, Observable)):
        if values is not None:
            definitions[role] = [_definition_record(spec, expected, role) for spec in values]
    qis, evidence = _guppy_dem_evidence(
        program,
        [*(detectors or []), *(observables or [])],
        settings,
        definitions,
        analysis=analysis,
    )
    runtime = _configure_selene_runtime(_SeleneRuntimeSpec(), None)
    return _prepare_guppy_dem(qis, num_qubits, seed, runtime, settings=settings, **definitions, **evidence)


def circuit_gates(circuit):
    return [
        (tick, gate.gate_type.name, gate.qubits, gate.angles, gate.params, gate.meas_ids)
        for tick, gate in circuit.gate_batches()
    ]
