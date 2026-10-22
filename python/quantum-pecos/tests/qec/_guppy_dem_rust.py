# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Convert validated facade inputs at the Rust preparation boundary."""

from __future__ import annotations

import json

import pecos
from pecos._engine_builders import _configure_selene_runtime
from pecos.qec.dem import (
    _GENERATOR_LAYOUT_ATTR,
    _certificate_carrier,
    _preflight_guppy_static_schedule,
    _result_tags_present,
)
from pecos.qec.dem_spec import Detector, RecordRef
from pecos_rslib import Qis
from pecos_rslib.qec import _prepare_guppy_dem, _SeleneRuntimeSpec


def definition_record(spec):
    selector = spec.inferred_output
    references = (
        None
        if selector is not None
        else [
            -ref.offset if isinstance(ref, RecordRef) else (ref.tag, ref.occurrence, ref.element) for ref in spec.refs
        ]
    )
    return (
        spec.id,
        None,
        references,
        None if selector is None else (selector.tag, selector.element),
        None if not isinstance(spec, Detector) or spec.coords is None else list(spec.coords),
        json.dumps(dict(spec.metadata or {})),
    )


def prepare(program, num_qubits, *, seed=7, settings=None, detectors=None, observables=None, analysis=None, **options):
    specs = [*(detectors or []), *(observables or [])]
    tags = sorted({ref.tag for spec in specs for ref in spec.refs if not isinstance(ref, RecordRef)})
    json_tags = _result_tags_present(options.get("detectors_json", "[]"), options.get("observables_json", "[]"))
    inferred = any(spec.inferred_output is not None for spec in specs) or (
        settings is not None
        and (
            (detectors is None and "detectors_json" not in options)
            or (observables is None and "observables_json" not in options)
        )
    )
    evidence = {}
    if not inferred or tags or json_tags:
        layout, checked = _preflight_guppy_static_schedule(program, required_tags=tags, json_result_tags=json_tags)
        if analysis is None:
            analysis = checked
        if layout is None:
            evidence["frontend_static_schedule"] = "python_hugr_preflight"
        else:
            digest, _ = getattr(_certificate_carrier(program), _GENERATOR_LAYOUT_ATTR)
            evidence["trusted_generator_layout"] = (analysis.hugr_bytes, list(layout), digest)
    qis = Qis(analysis.qis) if analysis is not None else pecos.Guppy(program)._to_program()
    runtime = _configure_selene_runtime(_SeleneRuntimeSpec(), None)
    return _prepare_guppy_dem(
        qis,
        num_qubits,
        seed,
        runtime,
        settings=settings,
        detectors=None if detectors is None else [definition_record(spec) for spec in detectors],
        observables=None if observables is None else [definition_record(spec) for spec in observables],
        **evidence,
        **options,
    )


def circuit_gates(circuit):
    return [
        (tick, gate.gate_type.name, gate.qubits, gate.angles, gate.params, gate.meas_ids)
        for tick, gate in circuit.gate_batches()
    ]
