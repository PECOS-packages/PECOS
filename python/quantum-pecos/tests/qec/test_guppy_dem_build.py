# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Adversarial measurement-identity tests for typed Guppy DEM builds."""

from __future__ import annotations

import json

import pytest
from guppylang import guppy
from guppylang.std.builtins import array, result
from guppylang.std.quantum import cx, h, measure, qubit
from pecos.qec import Detector, Observable, build_dem_from_guppy, rec, result_ref
from pecos.qec.dem_spec import _resolve_dem_specs
from pecos.qec.surface.decode import _replay_qis_trace_chunks_into_tick_circuit
from pecos_rslib.quantum import TickCircuit


@guppy
def _scrambled_tagged_measurements() -> None:
    qa = qubit()
    qb = qubit()
    h(qb)
    cx(qb, qa)
    a = measure(qa)
    b = measure(qb)
    result("b", b)
    result("a", a)


@guppy
def _transformed_measurement_result() -> None:
    q = qubit()
    measured = measure(q)
    result("not_m", not measured)


@guppy
def _aggregate_measurement_result() -> None:
    qa = qubit()
    qb = qubit()
    a = measure(qa)
    b = measure(qb)
    result("pair", array(a, b))


def _reordered_trace() -> tuple[TickCircuit, list[dict[str, object]]]:
    circuit = TickCircuit()
    circuit.tick().mz_with_ids([2, 0, 1], [2, 0, 1])
    traces = [
        {"name": "a", "values": [False], "result_ids": [0]},
        {"name": "b", "values": [False], "result_ids": [1]},
        {"name": "c", "values": [False], "result_ids": [2]},
    ]
    return circuit, traces


def test_rec_and_result_refs_resolve_identically_after_runtime_reorder() -> None:
    circuit, traces = _reordered_trace()
    via_records = _resolve_dem_specs(
        [Detector(rec[-3], rec[-1])],
        [Observable(rec[-2])],
        circuit=circuit,
        result_traces=traces,
    )
    via_results = _resolve_dem_specs(
        [Detector(result_ref("a"), result_ref("c"))],
        [Observable(result_ref("b"))],
        circuit=circuit,
        result_traces=traces,
    )

    assert via_records.detectors_json == via_results.detectors_json
    assert via_records.observables_json == via_results.observables_json
    assert via_records.schema_fingerprint == via_results.schema_fingerprint
    assert json.loads(via_records.detectors_json) == [{"id": 0, "meas_ids": [0, 2]}]
    assert [entry.meas_id for entry in via_records.ledger] == [2, 0, 1]


def test_real_guppy_rec_and_result_ref_builds_are_byte_identical() -> None:
    noise = {"p1": 0.01, "p2": 0.02, "p_meas": 0.1, "p_prep": 0.0}
    via_records = build_dem_from_guppy(
        _scrambled_tagged_measurements,
        num_qubits=2,
        detectors=[Detector(rec[-2])],
        observables=[Observable(rec[-1])],
        **noise,
    )
    via_results = build_dem_from_guppy(
        _scrambled_tagged_measurements,
        num_qubits=2,
        detectors=[Detector(result_ref("a"))],
        observables=[Observable(result_ref("b"))],
        **noise,
    )

    assert via_records.detectors_json == via_results.detectors_json
    assert via_records.observables_json == via_results.observables_json
    assert via_records.schema_fingerprint == via_results.schema_fingerprint
    assert via_records.dem.to_string() == via_results.dem.to_string()


def test_trace_once_build_evaluates_runtime_and_rejects_uncertified_named_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fake_trace(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return _reordered_trace()

    monkeypatch.setattr(
        "pecos.qec.surface.decode.trace_guppy_into_tick_circuit_with_result_traces",
        fake_trace,
    )
    build = build_dem_from_guppy(
        object(),
        num_qubits=3,
        detectors=[
            Detector(rec[-3], rec[-1]),
            Detector(rec[-3], rec[-1]),
        ],
        observables=[Observable(rec[-2])],
        p1=0.0,
        p2=0.0,
        p_meas=0.1,
        p_prep=0.0,
    )

    assert calls == 1
    assert build.audit["runtime_measurement_order"] == [2, 0, 1]
    assert build.audit["runtime_order_mismatch_count"] == 3
    assert build.evaluate_runtime_record([1, 0, 1]) == ([1, 1], 1)
    with pytest.raises(ValueError, match="missing referenced MeasId"):
        build.evaluate_results({"a": [0], "b": [1], "c": [1]})


def test_rec_build_evaluates_compiler_certified_scalar_results() -> None:
    build = build_dem_from_guppy(
        _scrambled_tagged_measurements,
        num_qubits=2,
        detectors=[Detector(rec[-2])],
        observables=[Observable(rec[-1])],
        p1=0.0,
        p2=0.0,
        p_meas=0.1,
        p_prep=0.0,
    )

    assert build.evaluate_results({"a": [0], "b": [1]}) == ([0], 1)


def test_rec_build_rejects_transformed_scalar_result_as_measurement_provenance() -> None:
    build = build_dem_from_guppy(
        _transformed_measurement_result,
        num_qubits=1,
        detectors=[Detector(rec[-1])],
        p1=0.0,
        p2=0.0,
        p_meas=0.1,
        p_prep=0.0,
    )

    with pytest.raises(ValueError, match="missing referenced MeasId"):
        build.evaluate_results({"not_m": [1]})


def test_rec_build_rejects_aggregate_array_as_measurement_provenance() -> None:
    build = build_dem_from_guppy(
        _aggregate_measurement_result,
        num_qubits=2,
        detectors=[Detector(rec[-2])],
        p1=0.0,
        p2=0.0,
        p_meas=0.1,
        p_prep=0.0,
    )

    with pytest.raises(ValueError, match="missing referenced MeasId"):
        build.evaluate_results({"pair": [0, 1]})


def test_result_ref_rejects_ambiguous_or_missing_runtime_provenance() -> None:
    circuit, _ = _reordered_trace()
    ambiguous = [{"name": "computed", "values": [True], "result_ids": []}]

    with pytest.raises(ValueError, match="absent from the runtime trace"):
        _resolve_dem_specs(
            [Detector(result_ref("computed"))],
            [],
            circuit=circuit,
            result_traces=ambiguous,
        )


def test_result_ref_rejects_array_element_provenance() -> None:
    circuit, _ = _reordered_trace()
    array_trace = [{"name": "pair", "values": [False, False], "result_ids": [0, 1]}]

    with pytest.raises(ValueError, match="array-valued result_ref provenance"):
        _resolve_dem_specs(
            [Detector(result_ref("pair", element=0))],
            [],
            circuit=circuit,
            result_traces=array_trace,
        )


def test_scalar_result_provenance_takes_precedence_over_aggregate_arrays() -> None:
    circuit, _ = _reordered_trace()
    traces = [
        {"name": "a", "values": [False], "result_ids": [0]},
        {"name": "b", "values": [False], "result_ids": [1]},
        {"name": "aggregate", "values": [False, False], "result_ids": [1, 0]},
        {"name": "final", "values": [False], "result_ids": [2]},
    ]
    schema = _resolve_dem_specs(
        [Detector(rec[-3], rec[-1])],
        [],
        circuit=circuit,
        result_traces=traces,
    )

    assert dict(schema.result_ids_by_tag) == {"a": (0,), "b": (1,), "final": (2,)}


def test_rec_rejects_non_dense_runtime_measurement_identities() -> None:
    circuit = TickCircuit()
    circuit.tick().mz_with_ids([0, 1], [4, 9])

    with pytest.raises(ValueError, match=r"rec\[\.\.\.\].*dense"):
        _resolve_dem_specs(
            [Detector(rec[-1])],
            [],
            circuit=circuit,
            result_traces=[],
        )


def test_audited_build_disables_raw_measurement_id_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def fake_trace(*_args, **kwargs):
        seen.update(kwargs)
        return _reordered_trace()

    monkeypatch.setattr(
        "pecos.qec.surface.decode.trace_guppy_into_tick_circuit_with_result_traces",
        fake_trace,
    )
    build_dem_from_guppy(
        object(),
        num_qubits=3,
        detectors=[Detector(rec[-1])],
        p1=0.0,
        p2=0.0,
        p_meas=0.0,
        p_prep=0.0,
    )

    assert seen["allow_raw_measurement_id_fallback"] is False


def test_audited_replay_rejects_entirely_raw_runtime_trace() -> None:
    chunks = [
        {
            "operations": [
                {"AllocateQubit": {"id": 0}},
                {"Quantum": {"Measure": [0, 0]}},
            ],
        },
    ]

    with pytest.raises(ValueError, match="does not contain lowered_quantum_ops"):
        _replay_qis_trace_chunks_into_tick_circuit(
            chunks,
            allow_raw_measurement_id_fallback=False,
        )
