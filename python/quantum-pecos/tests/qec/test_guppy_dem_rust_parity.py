# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Named parity matrix against the current Python Guppy DEM facade."""

from __future__ import annotations

import hashlib
import json
import re

import pecos
import pytest
from pecos import NoiseParameters
from pecos.guppy_gen import get_num_qubits, make_surface_code, make_surface_memory
from pecos.programs import Qis
from pecos.qec import (
    DemInferenceSettings,
    Detector,
    GuppyDemBuilder,
    Observable,
    rec,
    result_ref,
    surface_memory_dem_spec,
)
from pecos.qec import dem as dem_module
from pecos.qec.surface import SurfacePatch

from ._guppy_dem_rust import circuit_gates, prepare
from .test_dem_inference_builder import _inferred_program, _selected_program
from .test_from_guppy_dem import _single_measurement
from .test_guppy_dem_build import _mixed_supported_result_occurrences, _scrambled_tagged_measurements
from .test_guppy_dem_builder import _tagged_two_qubit_program
from .test_guppy_dem_rust_bindings import _reference_one

NOISE = {"p1": 0.001, "p2": 0.01, "p_meas": 0.02, "p_prep": 0.001}
ROWS = [
    "explicit-records",
    "explicit-result-references",
    "explicit-bare-tags",
    "explicit-json-result-tags",
    "inferred-permuted-raw-ids",
    "mixed-positional",
    "mixed-named",
    "generator-certified",
    "gadget-generator-certified",
    "sparse-measurement-ids",
    "permuted-measurement-ids",
    "repeated-result-calls-and-holes",
    "bell-parity",
    "random-detector-admission",
    "random-observable-admission",
    "reference-one",
]


def _case(name):
    program, qubits = _tagged_two_qubit_program, 2
    options = {"detectors": [Detector(rec[-2])], "observables": [Observable(rec[-1])]}
    if name == "explicit-result-references":
        options = {"detectors": [Detector(result_ref("m0"))], "observables": [Observable(result_ref("m1", element=0))]}
    elif name == "explicit-bare-tags":
        options = {"detectors": [Detector("m0")], "observables": [Observable("m1")]}
    elif name == "explicit-json-result-tags":
        options = {
            "detectors_json": '[{"id":0,"result_tags":["m0"]}]',
            "observables_json": '[{"id":0,"result_tags":["m1"]}]',
            "num_measurements": 2,
        }
    elif name == "inferred-permuted-raw-ids":
        program, qubits = _inferred_program, 3
        options = {"settings": DemInferenceSettings().with_validation_rows(8)}
    elif name in {"mixed-positional", "mixed-named"}:
        program, qubits = _selected_program, 3
        explicit = rec[-2] if name == "mixed-positional" else result_ref("m1")
        options = {
            "detectors": [Detector(explicit), Detector.inferred("parities", element=0)],
            "observables": [Observable.inferred("logical", id=70), Observable(rec[-3], id=2)],
        }
    elif name in {"generator-certified", "gadget-generator-certified"}:
        program = (
            make_surface_code(3, 1, "Z")
            if name == "generator-certified"
            else make_surface_memory(SurfacePatch.create(distance=3), 1, "Z")
        )
        qubits = get_num_qubits(3)
        detectors, observables = surface_memory_dem_spec(3, 1, "Z")
        options = {"detectors": detectors, "observables": observables}
    elif name == "sparse-measurement-ids":
        # The current facade defaults to the measurement-ID span, so sparse IDs
        # require an explicit emission count. Both paths receive the same count.
        options = {
            "detectors_json": '[{"id":0,"meas_ids":[7]}]',
            "observables_json": '[{"id":0,"meas_ids":[3]}]',
            "num_measurements": 2,
        }
    elif name == "repeated-result-calls-and-holes":
        program, qubits = _mixed_supported_result_occurrences, 1
        options = {"detectors": [Detector(result_ref("same", occurrence=1))], "observables": []}
    elif name in {"bell-parity", "random-detector-admission", "random-observable-admission"}:
        program = _scrambled_tagged_measurements
        options = {"detectors": [Detector("a", "b")], "observables": [Observable("a", "b")]}
        if name == "random-detector-admission":
            options["detectors"] = [Detector("a")]
        elif name == "random-observable-admission":
            options["observables"] = [Observable("b")]
    elif name == "reference-one":
        program, qubits = _reference_one, 1
        options = {"detectors": [Detector("one")], "observables": [Observable("one", id=70)]}
    return program, qubits, options


def _python_build(program, qubits, options, noise=NOISE):
    builder = GuppyDemBuilder().with_program(program).with_qubits(qubits).with_seed(7)
    for key, value in options.items():
        method = "with_inference" if key == "settings" else f"with_{key}"
        getattr(builder, method)(value)
    return builder.with_noise(NoiseParameters(**noise)).build()


def _reidentify_qis(analysis, ids):
    # Use QIS's explicit integer result-slot intrinsic to exercise the existing
    # fixtures with sparse/permuted identities while preserving their quantum gates.
    pattern = r"(?m)^(\s*)(%[\w.]+) = (?:tail )?call i64 @___lazy_measure\(i64 ([^\)]+)\).*$"
    matches = list(re.finditer(pattern, analysis.qis))
    assert len(matches) == len(ids)
    identities = iter(ids)

    def replace(match):
        indent, result, qubit = match.groups()
        identity = next(identities)
        return (
            f"{indent}{result} = add i64 0, {identity}\n"
            f"{indent}call i32 @__quantum__qis__m__body(i64 {qubit}, i64 {result})"
        )

    analysis.qis = re.sub(pattern, replace, analysis.qis) + "\ndeclare i32 @__quantum__qis__m__body(i64, i64)\n"
    return analysis


def _fingerprint_v2(prepared):
    # These fixtures have integral coordinates. Rust Display writes 0.0 as 0;
    # v2 also includes the computed reference fields and a new version number.
    def numbers(value):
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, dict):
            return {key: numbers(item) for key, item in value.items()}
        if isinstance(value, list):
            return [numbers(item) for item in value]
        return value

    payload = {
        "version": 2,
        "detectors": json.loads(prepared.circuit.get_meta("detectors")),
        "observables": json.loads(prepared.circuit.get_meta("observables")),
        "runtime_measurement_order": prepared.measurement_ids,
        "named_result_measurements": prepared.named_result_measurements,
    }
    return hashlib.sha256(
        json.dumps(numbers(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _assert_schema(python, rust):
    assert circuit_gates(rust.circuit) == circuit_gates(python.circuit)
    assert rust.measurement_ids == [entry.meas_id for entry in python.measurement_ledger]
    assert rust.measurement_ledger == [entry.to_dict() for entry in python.measurement_ledger]
    assert rust.named_result_measurements == dict(python._result_ids_by_tag)
    assert rust.definition_origins == list(python.definition_origins)
    for role in ("detectors", "observables"):
        expected = python.circuit.circuit_definitions()[role]
        actual = getattr(rust, role)
        assert actual == rust.circuit.circuit_definitions()[role]
        assert [(item["id"], sorted(item["measurements"])) for item in actual] == [
            (item["id"], sorted(item["measurements"])) for item in expected
        ]
        # Rust computes exact noiseless references; the current facade leaves
        # unspecified references absent. This changes metadata, not the supports.
        assert all(item["reference"] in (0, 1) for item in actual)
        for actual_definition, old_definition in zip(actual, expected, strict=True):
            if old_definition["reference"] is not None:
                assert actual_definition["reference"] == old_definition["reference"]
            else:
                assert actual_definition["reference"] in (0, 1)  # Previously absent, now computed.
    assert rust.schema_fingerprint == _fingerprint_v2(rust)
    assert rust.schema_fingerprint != python.schema_fingerprint  # v2 intentionally replaces Python's v1 digest.
    assert rust.circuit.get_meta("dem_schema_fingerprint") == rust.schema_fingerprint
    if python.inference is not None:
        assert rust.audit["inference"] == python.inference.audit
        for key in (
            "raw_measurement_ids",
            "detector_supports",
            "observable_supports",
            "detector_labels",
            "observable_labels",
        ):
            expected = json.loads(json.dumps(getattr(python.inference, key)))
            assert json.loads(json.dumps(rust.inference[key])) == expected


@pytest.mark.parametrize("name", ROWS, ids=ROWS)
def test_guppy_dem_parity_matrix(name, monkeypatch) -> None:
    program, qubits, options = _case(name)
    analysis = None
    if name in {"sparse-measurement-ids", "permuted-measurement-ids"}:
        layout, analysis = dem_module._preflight_guppy_static_schedule(program, required_tags=[])
        assert layout is None
        ids = [7, 3] if name == "sparse-measurement-ids" else [1, 0]
        analysis = _reidentify_qis(analysis, ids)
        monkeypatch.setattr(dem_module, "_preflight_guppy_static_schedule", lambda *_args, **_kwargs: (None, analysis))
    python = _python_build(program, qubits, options)
    if name in {"random-detector-admission", "random-observable-admission"}:
        # The old facade admits random readouts; Rust requires deterministic
        # noiseless detector and observable parities before any noisy build.
        role = "detector" if name == "random-detector-admission" else "observable"
        with pytest.raises(ValueError, match=f"{role} 0 has a random noiseless parity"):
            prepare(program, qubits, analysis=analysis, **options)
        assert python.dem.num_detectors == 1
        return
    rust = prepare(program, qubits, analysis=analysis, **options)
    assert rust.build_dem(**NOISE).dem.to_string() == python.dem.to_string()
    _assert_schema(python, rust)
    if name in {"sparse-measurement-ids", "permuted-measurement-ids"}:
        assert rust.measurement_ids == ids
        assert rust.named_result_measurements == {}  # Explicit result-slot QIS is not compiler-certified for tags.
    if name == "repeated-result-calls-and-holes":
        assert rust.named_result_measurements == {"same": (None, 0)}
        assert rust.evaluate_results({"same": [1, 0]}) == ([0], 0)
    if name in {"generator-certified", "gadget-generator-certified"}:
        assert rust.audit["static_schedule_source"] == "generator_layout_v2_program_bound"
    executable = Qis(analysis.qis) if analysis is not None else program
    results = (
        pecos.sim(executable)
        .classical(pecos.selene_engine())
        .quantum(pecos.stabilizer())
        .qubits(qubits)
        .seed(17)
        .run(4)
    )
    columns = results.to_shot_map().to_dict()
    # The binding consumes columns after Python has normalized scalar bits.
    columns = {
        tag: [list(map(int, row)) if isinstance(row, (list, tuple)) else [int(row)] for row in rows]
        for tag, rows in columns.items()
    }
    raw_expected = [([1], 1 << 70)] * 4 if name == "reference-one" else [([0] * len(rust.detectors), 0)] * 4
    if rust.named_result_measurements:
        assert rust.result_column_parities(columns) == python.evaluate_result_columns(columns) == raw_expected
        # Relative evaluators subtract the computed reference-one parity; the
        # current Python facade's evaluators return raw parities.
        assert rust.evaluate_result_columns(columns) == [([0] * len(rust.detectors), 0)] * 4
        for shot in range(4):
            values = {tag: rows[shot] for tag, rows in columns.items()}
            by_id = {
                identity: values[tag][index]
                for tag, identities in rust.named_result_measurements.items()
                if tag in values
                for index, identity in enumerate(identities)
                if identity is not None
            }
            assert rust.result_parities(values) == python.evaluate_results(values) == raw_expected[shot]
            assert rust.measurement_parities(by_id) == python.evaluate_measurements(by_id) == raw_expected[shot]
            if set(by_id) == set(rust.measurement_ids):
                record = [by_id[identity] for identity in rust.measurement_ids]
                assert (
                    rust.runtime_record_parities(record) == python.evaluate_runtime_record(record) == raw_expected[shot]
                )
    else:
        # Both columns are noiseless zero in the reidentified two-qubit fixture.
        for shot in range(4):
            record = [columns["m0"][shot][0], columns["m1"][shot][0]]
            assert rust.evaluate_runtime_record(record) == python.evaluate_runtime_record(record) == ([0], 0)
            by_id = dict(zip(rust.measurement_ids, record, strict=True))
            assert rust.evaluate_measurements(by_id) == python.evaluate_measurements(by_id) == ([0], 0)


def test_rust_message_format_and_zero_rate_idle_target_difference() -> None:
    program, qubits, options = _case("repeated-result-calls-and-holes")
    options["detectors"] = [Detector(result_ref("same", occurrence=0))]
    with pytest.raises(ValueError) as python_error:
        _python_build(program, qubits, options)
    with pytest.raises(ValueError) as rust_error:
        prepare(program, qubits, **options)
    # Rust Debug quotes strings with double quotes; Python repr uses single quotes.
    assert "'same'" in str(python_error.value)
    assert '"same"' in str(rust_error.value)
    assert "not a direct scalar measurement" in str(python_error.value)
    assert "not a direct scalar measurement" in str(rust_error.value)
    options = {"detectors": [Detector(rec[-1])]}
    with pytest.raises(ValueError, match="idle"):
        (
            GuppyDemBuilder()
            .with_program(_single_measurement)
            .with_qubits(1)
            .with_detectors(options["detectors"])
            .with_noise(NoiseParameters(**NOISE).with_p_idle_linear(0.0))
            .build()
        )
    prepared = prepare(_single_measurement, 1, **options)
    # Rust's idle-target admission checks active rates; a zero rate has no target requirement.
    assert (
        prepared.build_dem(p_idle_linear_rate=0.0, **NOISE).dem.to_string()
        == prepared.build_dem(**NOISE).dem.to_string()
    )
