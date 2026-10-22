# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Reusable Rust preparation and the value-only Python binding contract."""

import inspect
import json

import pytest
from guppylang import guppy
from guppylang.std.builtins import result
from guppylang.std.quantum import cx, h, measure, qubit, x
from pecos._traced_circuit import normalize_traced_tick_circuit
from pecos.qec import DemInferenceSettings, Detector, Observable, rec, result_ref
from pecos_rslib.qec import DetectorErrorModel

from ._guppy_dem_rust import circuit_gates, prepare
from .test_dem_inference_builder import _inferred_program
from .test_from_guppy_dem import _single_measurement


@guppy
def _reference_one() -> None:
    q = qubit()
    x(q)
    result("one", measure(q).read())


@guppy
def _idle_phase_readout() -> None:
    q0, q1 = qubit(), qubit()
    h(q0)
    cx(q1, q0)
    h(q0)
    result("m0", measure(q0).read())
    result("m1", measure(q1).read())


@pytest.mark.parametrize("settings", [None, DemInferenceSettings()], ids=["explicit", "unused-settings"])
def test_prepare_without_noise_and_rebuild_with_distinct_noise(settings) -> None:
    prepared = prepare(_single_measurement, 1, detectors=[Detector(rec[-1])], observables=[], settings=settings)
    assert prepared.measurement_ids == [0]
    assert prepared.detectors == prepared.circuit.circuit_definitions()["detectors"]
    assert prepared.detectors[0]["measurements"] == [0]
    assert prepared.detectors[0]["reference"] == 0
    assert prepared.observables == []
    assert prepared.inference is None
    assert prepared.audit["idle_noise_residuals"] == []
    assert prepared.definition_origins == [{"role": "detector", "id": 0, "source": "explicit"}]
    assert prepared.audit["static_schedule_source"] == "python_hugr_preflight"
    low = prepared.build_dem(p_meas=0.01)
    high = prepared.build_dem(p_meas=0.2)
    assert low.dem.to_string() != high.dem.to_string()
    for built in (low, high):
        assert isinstance(built.dem, DetectorErrorModel)
        assert built.prepared.schema_fingerprint == prepared.schema_fingerprint
        assert built.prepared.detectors == prepared.detectors
        assert built.prepared.measurement_ledger == prepared.measurement_ledger
    assert low.idle_noise_residuals == low.dem.idle_noise_residuals == []
    assert low.residual_summary(0.0) == low.residual_summary(1.0) == []
    for threshold in (-0.1, 1.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite and between 0 and 1"):
            low.residual_summary(threshold)
    expected = list(inspect.signature(DetectorErrorModel.from_circuit).parameters.values())[1:]
    assert list(inspect.signature(prepared.build_dem).parameters.values()) == expected


def test_circuit_snapshots_are_detached_and_failure_does_not_poison_preparation() -> None:
    prepared = prepare(_single_measurement, 1, detectors=[Detector(rec[-1])])
    before = prepared.build_dem().dem.to_string()
    gates = circuit_gates(prepared.circuit)
    snapshot = prepared.circuit
    snapshot.clear()
    snapshot.tick().x([0])
    snapshot.set_meta("detectors", "[]")
    assert circuit_gates(prepared.circuit) == gates
    assert prepared.circuit.get_meta("detectors") != "[]"
    with pytest.raises(ValueError, match="idle-noise parameters have no idle gates to attach to"):
        prepared.build_dem(p_idle=0.1)
    assert prepared.build_dem().dem.to_string() == before
    assert prepared.build_dem(p_meas=0.2).prepared.schema_fingerprint == prepared.schema_fingerprint


@pytest.mark.parametrize(
    "noise",
    [
        pytest.param({"idle_rz": 0.2}, id="coherent-idle"),
        pytest.param({"p1_gate_rates": {"H": 0.02}, "p2_gate_rates": {"SZZ": 0.03}}, id="per-gate"),
        pytest.param(
            {
                "p_meas_crosstalk_local": 0.02,
                "p_meas_crosstalk_model": {"0->1": 0.4},
                "measurement_crosstalk_dem_mode": "omitted",
            },
            id="measurement-crosstalk",
        ),
        pytest.param(
            {"p1_weights": {"X": 1.0}, "p2_weights": {"XX": 1.0}, "p_idle_x_linear_rate": 0.03},
            id="weighted-gate-and-idle",
        ),
    ],
)
def test_noise_forwarding_matches_from_circuit(noise) -> None:
    prepared = prepare(
        _idle_phase_readout, 2, detectors=[Detector(rec[-2])], observables=[Observable(rec[-1])], idle_after_2q=1.0
    )
    if "p1_gate_rates" in noise:
        counts = prepared.circuit.gate_counts_by_type()
        one = next(
            gate.gate_type.name
            for _, gate in prepared.circuit.gate_batches()
            if gate.gate_type.name in {"H", "SZ", "SZdg", "SX", "SXdg", "SY", "SYdg", "X", "Y", "Z"}
        )
        two = next(gate.gate_type.name for _, gate in prepared.circuit.gate_batches() if gate.is_two_qubit())
        assert one in counts and two in counts
        noise = {"p1_gate_rates": {one: 0.02}, "p2_gate_rates": {two: 0.03}}
    actual = prepared.build_dem(**noise)
    expected = DetectorErrorModel.from_circuit(prepared.circuit, **noise)
    assert actual.dem.to_string() == expected.to_string()
    assert actual.dem.num_contributions == expected.num_contributions
    assert actual.idle_noise_residuals == expected.idle_noise_residuals
    if "p_meas_crosstalk_local" in noise:
        # This capture has no payload locations. Omission is valid; requesting
        # replay must forward both the mode and the positive local rate.
        for overrides, message in (
            ({"measurement_crosstalk_dem_mode": "exact_deterministic"}, "no MeasCrosstalkLocalPayload locations"),
            ({"p_meas_crosstalk_model": {"0->1": 1.1}}, "transition rows must sum to <= 1"),
        ):
            options = noise | overrides
            with pytest.raises(ValueError, match=message) as direct_error:
                DetectorErrorModel.from_circuit(prepared.circuit, **options)
            with pytest.raises(ValueError, match=message) as prepared_error:
                prepared.build_dem(**options)
            assert str(prepared_error.value) == str(direct_error.value)
    if "idle_rz" in noise:
        # Coherent idle sources are tracked even when they have no detector or
        # observable effect and therefore leave the aggregated DEM text unchanged.
        baseline = prepared.build_dem().dem
        assert actual.dem.to_string() == baseline.to_string()
        assert actual.dem.num_contributions > baseline.num_contributions


def test_reference_one_through_every_evaluator_and_raw_variant() -> None:
    prepared = prepare(
        _reference_one, 1, detectors=[Detector(result_ref("one"))], observables=[Observable("one", id=70)]
    )
    assert prepared.detectors[0]["reference"] == prepared.observables[0]["reference"] == 1
    for schema in (prepared, prepared.build_dem().prepared):
        assert schema.evaluate_runtime_record([1]) == ([0], 0)
        assert schema.evaluate_measurements({0: 1}) == ([0], 0)
        assert schema.evaluate_results({"one": [1]}) == ([0], 0)
        assert schema.evaluate_result_columns({"one": [[1], [0]]}) == [([0], 0), ([1], 1 << 70)]
        assert schema.runtime_record_parities([1]) == ([1], 1 << 70)
        assert schema.measurement_parities({0: 1}) == ([1], 1 << 70)
        assert schema.result_parities({"one": [1]}) == ([1], 1 << 70)
        assert schema.result_column_parities({"one": [[1], [0]]}) == [([1], 1 << 70), ([0], 0)]
        assert schema.evaluate_result_columns({"one": []}) == schema.result_column_parities({"one": []}) == []


def test_inference_data_keeps_labels_counts_and_source_values() -> None:
    prepared = prepare(_inferred_program, 3, settings=DemInferenceSettings().with_validation_rows(8))
    data = prepared.inference
    assert data["raw_measurement_ids"] == [2, 0, 1]
    assert data["raw_tag"] == "raw measurements"
    assert data["detector_labels"] == [("DETECTOR", 0)]
    assert data["observable_labels"] == [("obs", 0)]
    assert data["validation_rows"] == 8
    assert data["probe_shots"] >= 3 + 8
    assert data["provenance_shots"] == 32
    assert data["raw_binding"] == "probe_correlated_result_ids"
    assert data["schedule_validation"] == "empirical_trace_comparison"
    assert data["schedule_validation_shots"] >= data["probe_shots"]
    assert all("values" in trace and len(trace["values"]) == trace["width"] for trace in data["source_result_traces"])
    assert [identity for trace in data["result_traces"] for identity in trace["result_ids"]] == [2, 0, 1]
    assert prepared.detectors == prepared.circuit.circuit_definitions()["detectors"]


def test_num_measurements_constraints() -> None:
    with pytest.raises(ValueError, match="cannot be combined with typed definitions"):
        prepare(_single_measurement, 1, detectors=[Detector(rec[-1])], num_measurements=1)
    with pytest.raises(ValueError, match="cannot be combined with inference"):
        prepare(_inferred_program, 3, settings=DemInferenceSettings(), num_measurements=3)
    with pytest.raises(ValueError, match="num_measurements=2 disagrees with the 1 measurement"):
        prepare(_single_measurement, 1, detectors_json='[{"id":0,"records":[-2]}]', num_measurements=2)
    assert prepare(_single_measurement, 1, detectors_json='[{"id":0,"records":[-1]}]', num_measurements=1).detectors[0][
        "measurements"
    ] == [0]


def test_normalization_preserves_context_and_forwards_the_pass_option() -> None:
    with pytest.raises(
        TypeError, match=r"binding test: expected a TickCircuit with callable normalize_traced_tick_circuit\(\)"
    ):
        normalize_traced_tick_circuit(object(), context="binding test")
    from pecos_rslib.quantum import TickCircuit

    circuit = TickCircuit()
    circuit.tick().h([0])
    circuit.tick().h([0])
    circuit.tick().mz([0])
    normalize_traced_tick_circuit(circuit, simplify_single_qubit_clifford_chains=False)
    assert circuit.gate_counts_by_type()["H"] == 2
    assert circuit.measurement_emission() == [(0, 0)]
    normalize_traced_tick_circuit(circuit)
    assert circuit.gate_counts_by_type().get("H", 0) == 0


def test_converted_measurement_id_records_preserve_metadata() -> None:
    from pecos._engine_builders import _configure_selene_runtime
    from pecos.qec.dem import _preflight_guppy_static_schedule
    from pecos_rslib import Qis
    from pecos_rslib.qec import _prepare_guppy_dem, _SeleneRuntimeSpec

    _, analysis = _preflight_guppy_static_schedule(_single_measurement, required_tags=[])
    runtime = _configure_selene_runtime(_SeleneRuntimeSpec(), None)
    arguments = (Qis(analysis.qis), 1, 0, runtime)
    detector = (None, [0], None, None, [1.0, 2.0, 3.0], '{"label":"readout","note":"α"}')
    observable = (70, [0], None, None, None, "{}")
    prepared = _prepare_guppy_dem(
        *arguments,
        detectors=[detector],
        observables=[observable],
        frontend_static_schedule="python_hugr_preflight",
    )
    assert prepared.detectors == [
        {
            "id": 0,
            "measurements": [0],
            "reference": 0,
            "coords": [1.0, 2.0, 3.0],
            "label": "readout",
            "fault_sector": None,
        },
    ]
    assert prepared.observables == [{"id": 70, "measurements": [0], "reference": 0, "label": None}]
    assert json.loads(prepared.circuit.get_meta("detectors"))[0]["note"] == "α"
    with pytest.raises(ValueError, match="exactly one source"):
        _prepare_guppy_dem(*arguments, detectors=[(None, [0], [1], None, None, "{}")])
    with pytest.raises(ValueError, match="may only be supplied once"):
        _prepare_guppy_dem(*arguments, detectors=[detector], detectors_json="[]")
    with pytest.raises(ValueError, match="static schedule"):
        _prepare_guppy_dem(*arguments, detectors=[detector])
