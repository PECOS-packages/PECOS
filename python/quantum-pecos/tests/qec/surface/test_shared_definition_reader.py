# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Consumers resolve ids and records through the circuit's shared reader."""

import json
from pathlib import Path

import pytest
from pecos._traced_circuit import measurement_ids_in_execution_order
from pecos.qec.dem_spec import _measurement_ids_in_runtime_order
from pecos.qec.surface import (
    SurfacePatch,
    extract_detection_events_and_observables,
    generate_tick_circuit_from_patch,
    get_detector_descriptors_from_tick_circuit,
    get_measurement_order_from_tick_circuit,
    get_observable_descriptors_from_tick_circuit,
    tick_circuit_to_stim,
)
from pecos.qec.surface.circuit_builder import generate_dem_from_tick_circuit_via_pauli_frame
from pecos.testing import simulate_tick_circuit
from pecos_rslib.quantum import TickCircuit


@pytest.fixture
def scrambled():
    circuit = TickCircuit()
    circuit.tick().mz_with_ids([0, 1], [17, 9])
    circuit.set_meta("num_measurements", "2")
    return circuit


@pytest.mark.parametrize("references", [{"meas_ids": [9]}, {"records": [1]}, {"records": [1], "meas_ids": [9]}])
def test_extraction_uses_positions(scrambled, references):
    scrambled.set_meta("detectors", json.dumps([{"id": 0, **references}]))
    assert extract_detection_events_and_observables(scrambled, [[0, 1]]) == ([[0]], [[]])


def test_extraction_annotation_only(scrambled):
    scrambled.detector([(0, 0, 1)])
    scrambled.observable([(0, 0, 1)])
    assert extract_detection_events_and_observables(scrambled, [[0, 1]]) == ([[0]], [[0]])


def test_extraction_sparse_ids_and_multiplicity(scrambled):
    scrambled.set_meta("detectors", '[{"id":8,"records":[1]},{"id":3,"records":[0]},{"id":2,"records":[1,1]}]')
    scrambled.set_meta("observables", '[{"id":7,"records":[1]},{"id":4,"records":[0]}]')
    assert extract_detection_events_and_observables(scrambled, [[1, 1]]) == ([[3, 8]], [[4, 7]])
    with pytest.raises(ValueError, match=r"row has length 1.*num_measurements=2"):
        extract_detection_events_and_observables(scrambled, [[1]])


def test_stim_resolves_scrambled_id(scrambled):
    scrambled.set_meta("detectors", '[{"id":0,"meas_ids":[17],"coords":[0,0,0]}]')
    assert tick_circuit_to_stim(scrambled) == "M 0 1\nDETECTOR(0.0, 0.0, 0.0) rec[-2]"


def test_stim_detector_id_order(scrambled):
    scrambled.set_meta("detectors", '[{"id":1,"records":[1]},{"id":0,"records":[0]}]')
    assert tick_circuit_to_stim(scrambled) == "M 0 1\nDETECTOR rec[-2]\nDETECTOR rec[-1]"


def test_stim_annotation_only_observable(scrambled):
    scrambled.observable([(0, 0, 1)])
    assert tick_circuit_to_stim(scrambled) == "M 0 1\nOBSERVABLE_INCLUDE(0) rec[-1]"


def test_measurement_walkers_include_mx_and_mpz():
    circuit = TickCircuit()
    circuit.tick().mz_with_ids([4], [17])
    circuit.tick().mx([2])
    circuit.tick().add_gate("MPZ", [6, 1])
    circuit.assign_missing_meas_ids()
    assert measurement_ids_in_execution_order(circuit) == [17, 18, 19, 20]
    assert _measurement_ids_in_runtime_order(circuit) == [17, 18, 19, 20]
    assert get_measurement_order_from_tick_circuit(circuit) == [4, 2, 6, 1]


def test_runtime_walker_rejects_duplicates_and_unstamped_records():
    circuit = TickCircuit()
    circuit.tick().mz_with_ids([0], [7])
    circuit.tick().mz_with_ids([1], [7])
    with pytest.raises(ValueError, match="duplicate MeasId"):
        _measurement_ids_in_runtime_order(circuit)
    unstamped = TickCircuit()
    unstamped.tick().add_gate("MX", [0])
    with pytest.raises(ValueError, match="emission position 0 carries no MeasId"):
        measurement_ids_in_execution_order(unstamped)


def test_simulation_absolute_record():
    circuit = TickCircuit()
    circuit.tick().pz([0])
    circuit.tick().x([0])
    circuit.tick().mz_with_ids([0], [17])
    circuit.set_meta("detectors", '[{"id":8,"records":[0]}]')
    circuit.set_meta("observables", '[{"id":3,"records":[0]}]')
    assert simulate_tick_circuit(circuit) == ([1], 1, {3: 1})


def test_pauli_frame_generator_warns_at_caller(scrambled):
    scrambled.set_meta("detectors", '[{"id":0,"records":[-1],"coords":[0,0,0]}]')
    scrambled.set_meta("num_measurements", "2")
    with pytest.warns(DeprecationWarning, match="will be removed.*DetectorErrorModel.from_circuit") as warnings:
        generate_dem_from_tick_circuit_via_pauli_frame(scrambled)
    assert warnings[0].filename == __file__


def test_annotation_only_observable_descriptor(scrambled):
    scrambled.observable([(0, 0, 1)], label="ignored by descriptors")
    patch = SurfacePatch.create(distance=3)
    assert get_observable_descriptors_from_tick_circuit(scrambled, patch) == [
        {
            "id": 0,
            "observable_id": 0,
            "basis": "Z",
            "records": [-1],
            "logical_type": "Z",
            "data_qubits": [0, 1, 2],
            "data_qubit_positions": [[0, 0], [0, 1], [0, 2]],
            "weight": 3,
            "support_axis": "horizontal",
        },
    ]


def test_detector_descriptor_requires_coords(scrambled):
    scrambled.detector([(0, 0, 1)])
    with pytest.raises(ValueError, match="Detector 0 requires coordinates"):
        get_detector_descriptors_from_tick_circuit(scrambled, SurfacePatch.create(distance=3))


@pytest.mark.parametrize(("basis", "budget"), [("Z", None), ("X", None), ("Z", 2)])
def test_surface_descriptors_match_pre_reader_baseline(basis, budget):
    expected = json.loads((Path(__file__).parent / "data" / "reader_descriptor_baseline.json").read_text())
    patch = SurfacePatch.create(distance=3)
    circuit = generate_tick_circuit_from_patch(patch, num_rounds=1, basis=basis, ancilla_budget=budget)
    assert circuit.get_meta("detector_descriptors") is None
    assert circuit.get_meta("observable_descriptors") is None
    baseline = expected[f"{basis}-{budget}"]
    assert get_detector_descriptors_from_tick_circuit(circuit, patch) == baseline["detectors"]
    assert get_observable_descriptors_from_tick_circuit(circuit, patch) == baseline["observables"]
    assert circuit.get_meta("detector_descriptors") == json.dumps(baseline["detectors"])
    assert circuit.get_meta("observable_descriptors") == json.dumps(baseline["observables"])
    assert "DETECTOR" in tick_circuit_to_stim(circuit)


@pytest.mark.parametrize("gate", ["MeasureFree", "MX", "MPZ"])
def test_simulation_checks_reader_count_before_replay(gate):
    circuit = TickCircuit()
    circuit.tick().add_gate(gate, [0])
    circuit.set_meta("num_measurements", "0")
    with pytest.raises(ValueError, match="num_measurements=0 disagrees with the 1 measurement"):
        simulate_tick_circuit(circuit)


@pytest.mark.parametrize("consumer", ["extract", "stim", "simulate", "detectors", "observables"])
def test_consumers_reject_disagreeing_sources(scrambled, consumer):
    scrambled.detector([(0, 0, 1)])
    scrambled.set_meta("detectors", '[{"id":0,"records":[0],"coords":[0,0,0]}]')
    patch = SurfacePatch.create(distance=3)
    function, args = {
        "extract": (extract_detection_events_and_observables, (scrambled, [[0, 1]])),
        "stim": (tick_circuit_to_stim, (scrambled,)),
        "simulate": (simulate_tick_circuit, (scrambled,)),
        "detectors": (get_detector_descriptors_from_tick_circuit, (scrambled, patch)),
        "observables": (get_observable_descriptors_from_tick_circuit, (scrambled, patch)),
    }[consumer]
    with pytest.raises(ValueError, match="differ from annotation positions"):
        function(*args)


def test_descriptor_cache_is_returned_without_rereading(scrambled):
    scrambled.set_meta("detector_descriptors", '[{"cached":"detector"}]')
    scrambled.set_meta("observable_descriptors", '[{"cached":"observable"}]')
    scrambled.set_meta("detectors", 123)
    patch = SurfacePatch.create(distance=3)
    assert get_detector_descriptors_from_tick_circuit(scrambled, patch) == [{"cached": "detector"}]
    assert get_observable_descriptors_from_tick_circuit(scrambled, patch) == [{"cached": "observable"}]


@pytest.mark.parametrize("records", [[-2], [0], [-1]])
def test_dem_nonpositional_ids_need_no_measurement_order(records):
    from pecos.qec.surface import generate_dem_from_tick_circuit

    # Only qubit 0 is prepared, so only the measurement stamped 17 sees the
    # preparation error and the two measurements give different DEMs.
    circuit = TickCircuit()
    circuit.tick().pz([0])
    circuit.tick().mz_with_ids([0, 1], [17, 9])
    circuit.set_meta("num_measurements", "2")

    def dem(references):
        circuit.set_meta("detectors", json.dumps([{"id": 0, **references}]))
        circuit.set_meta("observables", json.dumps([{"id": 0, **references}]))
        return generate_dem_from_tick_circuit(circuit, p_meas=0.125, p_prep=0.25)

    # Emission order is [17, 9]: -2 and 0 name id 17, and -1 names id 9.
    meas_id, other_id = (9, 17) if records == [-1] else (17, 9)
    records_dem = dem({"records": records})
    assert dem({"meas_ids": [other_id]}) != dem({"meas_ids": [meas_id]})
    assert records_dem == dem({"meas_ids": [meas_id]})


@pytest.mark.parametrize("basis", ["Z", "X"])
def test_surface_dems_match_pre_removal_baseline(basis):
    from pecos.qec.surface import NoiseParameters, generate_dem_from_tick_circuit
    from pecos.qec.surface.decode import build_native_sampler, generate_circuit_level_dem_from_builder

    expected = json.loads((Path(__file__).parent / "data" / "reader_dem_baseline.json").read_text())[basis]
    patch = SurfacePatch.create(distance=3)
    params = {"p1": 0.001, "p2": 0.002, "p_meas": 0.003, "p_prep": 0.004}
    noise = NoiseParameters(**params)
    circuit = generate_tick_circuit_from_patch(patch, num_rounds=2, basis=basis, ancilla_budget=2)
    assert generate_dem_from_tick_circuit(circuit, **params) == expected["direct"]
    assert generate_circuit_level_dem_from_builder(patch, 2, noise, basis, ancilla_budget=2) == expected["cached"]
    assert build_native_sampler(patch, 2, noise, basis, ancilla_budget=2).dem_string == expected["sampler"]


def test_logical_dem_matches_pre_removal_baseline(monkeypatch):
    from pecos.qec.surface import LogicalCircuitBuilder

    expected = json.loads((Path(__file__).parent / "data" / "reader_dem_baseline.json").read_text())["logical"]
    builder = LogicalCircuitBuilder()
    builder.add_patch(SurfacePatch.create(distance=3), "A")
    builder.add_memory("A", rounds=2, basis="Z")
    monkeypatch.setattr(builder, "_build_structured_dem_from_cached_slices", lambda **_kwargs: None)
    assert builder.build_dem(p1=0.001, p2=0.002, p_meas=0.003, p_prep=0.004) == expected


@pytest.mark.parametrize("references", [{"records": [0]}, {"meas_ids": [17]}, {"records": [0], "meas_ids": [17]}])
def test_audit_schema_uses_resolved_positions(scrambled, references):
    from pecos.qec.dem_spec import _resolved_schema_from_validated_json

    detectors = json.dumps([{"id": "D8", "records": [-1, -1]}, {"detector_id": "D3", **references}])
    observables = json.dumps([{"observable_id": "L4", **references}])
    scrambled.set_meta("detectors", detectors)
    scrambled.set_meta("observables", observables)
    schema = _resolved_schema_from_validated_json(detectors, observables, circuit=scrambled, result_traces=[])
    assert schema.detector_meas_ids == ((17,), (9, 9))
    assert schema.observable_meas_ids == ((4, (17,)),)
    assert [(entry.meas_id, entry.runtime_record_index) for entry in schema.ledger] == [(17, 0), (9, 1)]


@pytest.mark.parametrize("kind", ["detectors", "observables"])
def test_remap_requires_definition_and_position(scrambled, kind):
    from pecos.qec.surface.decode import _copy_surface_tick_circuit_metadata

    scrambled.set_meta(kind, '[{"id":0,"meas_ids":[9]}]')
    target = TickCircuit()
    with pytest.raises(ValueError, match="emission position 1 is missing"):
        _copy_surface_tick_circuit_metadata(scrambled, target, measurement_index_remap={0: 100})
    descriptor_key = "detector_descriptors" if kind == "detectors" else "observable_descriptors"
    scrambled.set_meta(descriptor_key, '[{"id":1,"records":[-1]}]')
    with pytest.raises(ValueError, match="entry id 1 has no source definition"):
        _copy_surface_tick_circuit_metadata(scrambled, target, measurement_index_remap={0: 100, 1: 200})


def test_remap_descriptors_use_annotation_definitions(scrambled):
    from pecos.qec.surface.decode import _copy_surface_tick_circuit_metadata

    scrambled.detector([(0, 0, 1)])
    scrambled.observable([(0, 0, 0)])
    scrambled.set_meta("detector_descriptors", '[{"id":0,"records":[99],"weight":2}]')
    scrambled.set_meta("observable_descriptors", '[{"id":0,"meas_ids":[99],"basis":"Z"}]')
    target = TickCircuit()
    _copy_surface_tick_circuit_metadata(scrambled, target, measurement_index_remap={0: 100, 1: 200})
    assert json.loads(target.get_meta("detector_descriptors")) == [{"id": 0, "meas_ids": [200], "weight": 2}]
    assert json.loads(target.get_meta("observable_descriptors")) == [{"id": 0, "meas_ids": [100], "basis": "Z"}]


@pytest.mark.parametrize("consumer", ["remap", "audit", "topology"])
def test_decode_readers_reject_disagreeing_sources(scrambled, consumer, monkeypatch):
    from pecos.qec.dem_spec import _resolved_schema_from_validated_json
    from pecos.qec.surface import decode

    scrambled.detector([(0, 0, 1)])
    scrambled.set_meta("detectors", '[{"id":0,"records":[0]}]')
    monkeypatch.setattr(decode, "_build_surface_tick_circuit_for_native_model", lambda *_args, **_kwargs: scrambled)
    from pecos.qec.surface.decode import (
        _copy_surface_tick_circuit_metadata,
        _surface_native_topology,
        _surface_patch_cache_key,
    )

    patch_key = _surface_patch_cache_key(SurfacePatch.create(distance=3))
    function, args, kwargs = {
        "remap": (
            _copy_surface_tick_circuit_metadata,
            (scrambled, TickCircuit()),
            {"measurement_index_remap": {0: 100, 1: 200}},
        ),
        "audit": (
            _resolved_schema_from_validated_json,
            (scrambled.get_meta("detectors"), "[]"),
            {"circuit": scrambled, "result_traces": []},
        ),
        "topology": (_surface_native_topology, (patch_key, 1, "Z", None, "abstract", False), {}),
    }[consumer]
    with pytest.raises(ValueError, match="differ from annotation positions"):
        function(*args, **kwargs)


def test_native_topology_passes_resolved_offsets_to_pauli_lookup(scrambled, monkeypatch):
    from types import SimpleNamespace

    from pecos.qec.surface import decode
    from pecos.qec.surface._twirl_config import TwirlConfig
    from pecos_rslib import qec

    scrambled.set_meta("detectors", '[{"id":1,"records":[0,0]},{"id":0,"meas_ids":[9]}]')
    scrambled.set_meta("observables", '[{"id":0,"records":[0],"meas_ids":[17]}]')
    expected_detectors = [[-1], [-2, -2]]
    lookup_calls = []

    def capture_lookup(_dag, detectors, observables):
        lookup_calls.append((detectors, observables))
        return SimpleNamespace(num_pauli_sites=0)

    monkeypatch.setattr(qec, "PauliFrameLookup", SimpleNamespace(from_circuit=capture_lookup))
    monkeypatch.setattr(decode, "_build_surface_tick_circuit_for_native_model", lambda *_args, **_kwargs: scrambled)
    patch = SurfacePatch.create(distance=3)
    from pecos.qec.surface.decode import _surface_native_topology, _surface_patch_cache_key

    topology = _surface_native_topology(
        _surface_patch_cache_key(patch),
        1,
        "Z",
        None,
        "abstract",
        False,
        twirl=TwirlConfig(),
    )
    assert lookup_calls == [(expected_detectors, [[-2]])]
    assert topology.num_measurements == 2
    assert topology.num_detectors == len(expected_detectors)
    assert topology.num_observables == 1
