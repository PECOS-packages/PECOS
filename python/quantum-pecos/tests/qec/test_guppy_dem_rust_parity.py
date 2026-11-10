# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Facade regressions for QIR slots, occurrence identities, references, and schema v2."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
from pecos import NoiseParameters, Qis, pecos_rslib, selene_engine, sim
from pecos.qec import Detector, GuppyDemBuilder, Observable, rec, result_ref
from pecos.qec import dem as dem_module
from pecos.qec.dem import _preflight_guppy_static_schedule

from qec.test_from_guppy_dem import _single_measurement
from qec.test_guppy_dem_build import _mixed_supported_result_occurrences, _scrambled_tagged_measurements
from qec.test_guppy_dem_builder import _tagged_two_qubit_program
from qec.test_guppy_dem_rust_bindings import _reference_one

NOISE = {"p1": 0.001, "p2": 0.01, "p_meas": 0.02, "p_prep": 0.001}
ROWS = ["sparse-result-slots", "permuted-result-slots", "reference-one"]


def _case(name):
    program, qubits = _tagged_two_qubit_program, 2
    options = {"detectors": [Detector(rec[-2])], "observables": [Observable(rec[-1])]}
    if name == "sparse-result-slots":
        options = {
            "detectors_json": '[{"id":0,"meas_ids":[0]}]',
            "observables_json": '[{"id":0,"meas_ids":[1]}]',
            "num_measurements": 2,
        }
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


def _qir_slot_program(slots):
    # Follow selene_request_tests.rs and test_tracing._identity_program:
    # QIR program qubits and result slots use QIR gates and QIR result reads.
    return f"""
        declare void @__quantum__qis__reset__body(i64)
        declare void @__quantum__qis__cx__body(i64, i64)
        declare i32 @__quantum__qis__m__body(i64, i64)
        declare i32 @__quantum__rt__result_get_one(i64)
        define i64 @qmain(i64 %shot) {{
            call void @__quantum__qis__reset__body(i64 0)
            call void @__quantum__qis__reset__body(i64 1)
            call void @__quantum__qis__cx__body(i64 0, i64 1)
            %m0 = call i32 @__quantum__qis__m__body(i64 0, i64 {slots[0]})
            %v0 = call i32 @__quantum__rt__result_get_one(i64 {slots[0]})
            %m1 = call i32 @__quantum__qis__m__body(i64 1, i64 {slots[1]})
            %v1 = call i32 @__quantum__rt__result_get_one(i64 {slots[1]})
            ret i64 0
        }}
    """


def _fingerprint_v2(build):
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
        "detectors": json.loads(build.circuit.get_meta("detectors")),
        "observables": json.loads(build.circuit.get_meta("observables")),
        "runtime_measurement_order": [entry.meas_id for entry in build.measurement_ledger],
        "named_result_measurements": {
            ref.tag: tuple(item.meas_id for item in build.measurement_ledger if ref in item.result_refs)
            for entry in build.measurement_ledger
            for ref in entry.result_refs
        },
    }
    return hashlib.sha256(
        json.dumps(numbers(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(),
    ).hexdigest()


@pytest.mark.parametrize("name", ROWS, ids=ROWS)
def test_guppy_dem_facade_regressions(name, monkeypatch) -> None:
    program, qubits, options = _case(name)
    analysis = None
    if name in {"sparse-result-slots", "permuted-result-slots"}:
        layout, analysis, _ = _preflight_guppy_static_schedule(program, required_tags=[])
        assert layout is None
        slots = [7, 3] if name == "sparse-result-slots" else [1, 0]
        analysis.qis = _qir_slot_program(slots)
        monkeypatch.setattr(
            dem_module,
            "_preflight_guppy_static_schedule",
            lambda *_args, **_kwargs: (None, analysis, None),
        )
    built = _python_build(program, qubits, options)
    assert built.schema_fingerprint == _fingerprint_v2(built)
    assert built.circuit.get_meta("dem_schema_fingerprint") == built.schema_fingerprint
    if analysis is not None:
        assert [entry.meas_id for entry in built.measurement_ledger] == [0, 1]
        assert all(entry.result_refs == () for entry in built.measurement_ledger)
        assert json.loads(built.detectors_json)[0]["meas_ids"] == [0]
        assert json.loads(built.observables_json)[0]["meas_ids"] == [1]
        assert built.evaluate_runtime_record([0, 1]) == ([0], 1)
        assert built.evaluate_measurements({0: 1, 1: 0}) == ([1], 0)
        # The runtime retains caller-selected slots in raw result names;
        # preparation and evaluation use dense measurement occurrence IDs.
        raw = (
            sim(Qis(analysis.qis))
            .classical(selene_engine("selene_simple_runtime"))
            .quantum(pecos_rslib.state_vector())
            .qubits(qubits)
            .workers(1)
            .seed(7)
            .run(1)
            .to_dict()
        )
        assert raw == {f"measurement_{slot}": [0] for slot in slots}
    else:
        assert built.measurement_ledger[0].result_refs == (result_ref("one"),)
        assert json.loads(built.detectors_json)[0]["reference"] == 1
        assert json.loads(built.observables_json)[0]["reference"] == 1
        assert built.evaluate_runtime_record([True]) == ([0], 0)
        assert built.evaluate_measurements({0: True}) == ([0], 0)
        assert built.evaluate_results({"one": True}) == ([0], 0)
        assert built.evaluate_result_columns({"one": [True, False]}) == [([0], 0), ([1], 1 << 70)]
        assert built.runtime_record_parities([True]) == ([1], 1 << 70)
        assert built.measurement_parities({0: True}) == ([1], 1 << 70)
        assert built.result_parities({"one": True}) == ([1], 1 << 70)
        assert built.result_column_parities({"one": [True, False]}) == [([1], 1 << 70), ([0], 0)]
        assert built.evaluate_result_columns({"one": []}) == built.result_column_parities({"one": []}) == []


@pytest.mark.parametrize("role", ["detector", "observable"])
def test_facade_rejects_random_noiseless_parity(role) -> None:
    options = {"detectors": [Detector("a", "b")], "observables": [Observable("a", "b")]}
    options[f"{role}s"] = [Detector("a")] if role == "detector" else [Observable("b")]
    with pytest.raises(ValueError, match=f"{role} 0 has a random noiseless parity"):
        _python_build(_scrambled_tagged_measurements, 2, options)


def test_rust_message_format_and_zero_rate_idle_target() -> None:
    with pytest.raises(ValueError, match=r'"same".*not a direct scalar measurement'):
        _python_build(_mixed_supported_result_occurrences, 1, {"detectors": [Detector(result_ref("same"))]})
    builder = GuppyDemBuilder().with_program(_single_measurement).with_qubits(1).with_detectors([Detector(rec[-1])])
    baseline = builder.build().dem.to_string()
    assert (
        builder.with_noise(NoiseParameters(p_meas=0.001, p_prep=0.001).with_p_idle_linear(0.0)).build().dem.to_string()
        == baseline
    )


@pytest.mark.parametrize("role", ["detector", "observable"])
def test_facade_rejects_disagreeing_declared_references(role) -> None:
    options = {"detectors": [], "observables": []}
    definition = Detector if role == "detector" else Observable
    options[f"{role}s"] = [definition("one", metadata={"reference": 0})]
    with pytest.raises(ValueError, match="reference"):
        _python_build(_reference_one, 1, options)


def test_every_facade_evaluator_validates_python_bits() -> None:
    build = _python_build(
        _reference_one,
        1,
        {"detectors": [Detector("one")], "observables": [Observable("one", id=70)]},
    )
    methods = (
        (build.evaluate_runtime_record, build.runtime_record_parities, lambda bit: [bit]),
        (build.evaluate_measurements, build.measurement_parities, lambda bit: {0: bit}),
        (build.evaluate_results, build.result_parities, lambda bit: {"one": bit}),
        (build.evaluate_result_columns, build.result_column_parities, lambda bit: {"one": [[bit]]}),
    )
    for relative, raw, carrier in methods:
        for bit in (True, 1, np.int64(1)):
            expected = ([0], 0)
            raw_expected = ([1], 1 << 70)
            if relative == build.evaluate_result_columns:
                expected, raw_expected = [expected], [raw_expected]
            assert relative(carrier(bit)) == expected
            assert raw(carrier(bit)) == raw_expected
        for value in (0.0, 1.0, np.float64(1.0), 2, -1, "0", None):
            for evaluate in (relative, raw):
                with pytest.raises(ValueError, match="must be bool, 0, or 1"):
                    evaluate(carrier(value))
    assert build.evaluate_measurements({0: True, 99: object()}) == ([0], 0)
    assert build.measurement_parities({0: True, 99: object()}) == ([1], 1 << 70)
    assert build.evaluate_results({"one": np.array([1], dtype=np.int64), "unbound": object()}) == ([0], 0)
    assert build.dem is build.dem
    assert build.circuit.get_meta("num_measurements") == "1"
    snapshot = build.circuit
    snapshot.clear()
    snapshot.set_meta("detectors", "[]")
    assert build.circuit.num_measurements() == 1
    assert json.loads(build.detectors_json)[0]["reference"] == 1
