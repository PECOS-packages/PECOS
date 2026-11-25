# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Fault-sector metadata follows family keys, including H and fold boundaries."""

import json

import pytest
import stim
from pecos.qec.dem_spec import surface_memory_dem_spec
from pecos.qec.surface import LogicalCircuitBuilder, SurfacePatch
from pecos.qec.surface.decode import _surface_native_topology, _surface_patch_cache_key
from pecos.qec.surface.logical_circuit import LogicalGateType
from pecos.qec.surface.teleportation_decision import t_teleportation_decision_model
from pecos_rslib import TickCircuit
from pecos_rslib.decoders import pymatching
from pecos_rslib.qec import DetectorErrorModel, SampleBatch

from qec.surface.test_logical_builder_gadgets import make_builder


@pytest.mark.parametrize("source", ["abstract", "traced_qis"])
@pytest.mark.parametrize("basis", ["X", "Z"])
def test_surface_fault_sectors_and_native_forward(source, basis):
    patch = SurfacePatch.create(distance=3)
    topology = _surface_native_topology(_surface_patch_cache_key(patch), 2, basis, None, source, False)
    detectors = json.loads(topology.detectors_json)
    assert detectors
    assert {d["fault_sector"] for d in detectors} == {"X", "Z"}
    assert all(d["fault_sector"] == ("X" if d["coords"][1] == 0 else "Z") for d in detectors)


@pytest.mark.parametrize(
    "shape",
    [
        "d3_mem_X",
        "d3_mem_Z",
        "d3_h_z_to_x",
        "d3_h_x_to_z",
        "d3_h_zero_final",
        "d3_hh",
        "d3_h_cx_h",
        "d2_h_even",
        "d5_h",
        "d3_cx_zz",
        "d3_cx_xx",
    ],
)
def test_logical_fault_sectors_follow_code_family(shape):
    builder = make_builder(shape)
    circuit = builder.to_tick_circuit()
    swapped = dict.fromkeys((patch for op in builder.operations for patch in op.patches), False)
    segment_swaps = {}
    segment = 0
    for op in builder.operations:
        if op.gate_type == LogicalGateType.TRANSVERSAL_H:
            for patch in op.patches:
                swapped[patch] = not swapped[patch]
        if op.is_segment:
            for patch in op.patches:
                segment_swaps[patch, segment] = swapped[patch]
            segment += 1
    keys = json.loads(circuit.get_meta("measurement_keys"))["stabilizer"]
    by_record = {
        ordinal: ({"X": "Z", "Z": "X"}[family] if segment_swaps[patch, segment] else family)
        for patch, family, _, segment, _, ordinal in keys
    }
    detectors = json.loads(circuit.get_meta("detectors"))
    for detector in detectors:
        # Map the latest syndrome record back through its own measurement frame.
        current = max(record for record in detector["meas_ids"] if record in by_record)
        expected = by_record[current]
        if shape == "d3_h_zero_final" and any(record not in by_record for record in detector["meas_ids"]):
            # The final X data readout after H reads the code's Z family.
            expected = "Z"
        assert detector["fault_sector"] == expected


@pytest.mark.parametrize("shape", ["d3_fold_s_mid", "d3_fold_s_first", "d3_fold_s_last", "d3_fold_pair_x", "d3_h_fold"])
def test_fold_involved_fault_sectors_unknown(shape):
    builder = make_builder(shape)
    circuit = builder.to_tick_circuit()
    keys = json.loads(circuit.get_meta("measurement_keys"))["stabilizer"]
    fold_segments = set()
    segment = 0
    for op in builder.operations:
        if op.is_segment:
            if op.fold:
                fold_segments.update((patch, segment) for patch in op.patches)
            segment += 1
    fold_records = {ordinal for patch, _, _, seg, _, ordinal in keys if (patch, seg) in fold_segments}
    assert fold_records
    detectors = json.loads(circuit.get_meta("detectors"))
    for detector in detectors:
        if fold_records.intersection(detector["meas_ids"]):
            assert detector["fault_sector"] is None
        else:
            assert detector["fault_sector"] in {"X", "Z"}
    assert any(d["fault_sector"] is None for d in detectors)


def test_surface_spec_fault_sector_survival():
    detectors, _ = surface_memory_dem_spec(3, 2, "Z")
    assert {d.metadata["fault_sector"] for d in detectors} == {"X", "Z"}


def test_teleportation_filter_fault_sector_survival():
    model = t_teleportation_decision_model(SurfacePatch.create(distance=3), rounds_before=1)
    detectors = json.loads(model.circuit.get_meta("detectors"))
    assert detectors
    assert all(d["fault_sector"] in {"X", "Z"} for d in detectors)


def test_surface_descriptor_fault_sector_survival():
    from pecos.qec.surface.circuit_builder import (
        generate_tick_circuit_from_patch,
        get_detector_descriptors_from_tick_circuit,
    )

    patch = SurfacePatch.create(distance=3)
    circuit = generate_tick_circuit_from_patch(patch, 2, "Z")
    expected = [d["fault_sector"] for d in circuit.circuit_definitions()["detectors"]]
    descriptors = get_detector_descriptors_from_tick_circuit(circuit, patch)
    assert [d["fault_sector"] for d in descriptors] == expected
    assert get_detector_descriptors_from_tick_circuit(circuit, patch) == descriptors


@pytest.mark.parametrize("dagger", [False, True])
@pytest.mark.parametrize("final_rounds", [0, 2])
def test_physical_sz_boundary_fault_sectors_unknown(dagger, final_rounds):
    from pecos.qec.surface import LogicalCircuitBuilder

    builder = LogicalCircuitBuilder()
    builder.add_patch(SurfacePatch.create(distance=3), "A")
    builder.add_memory("A", 2, "Z")
    if dagger:
        builder.add_transversal_szdg("A")
    else:
        builder.add_transversal_sz("A")
    builder.add_memory("A", final_rounds, "Z")
    circuit = builder.to_tick_circuit()
    keys = json.loads(circuit.get_meta("measurement_keys"))
    affected = {ordinal for _, _, _, segment, rnd, ordinal in keys["stabilizer"] if segment == 1 and rnd == 0}
    if final_rounds == 0:
        affected.update(ordinal for _, _, ordinal in keys["data"])
    detectors = json.loads(circuit.get_meta("detectors"))
    for detector in detectors:
        if affected.intersection(detector["meas_ids"]):
            assert detector["fault_sector"] is None
        else:
            assert detector["fault_sector"] in {"X", "Z"}
    assert any(d["fault_sector"] is None for d in detectors)


def test_physical_sz_phase_uncertainty_follows_transversal_cx():
    from pecos.qec.surface import LogicalCircuitBuilder

    patch = SurfacePatch.create(distance=3)
    builder = LogicalCircuitBuilder()
    builder.add_patch(patch, "A")
    builder.add_patch(
        SurfacePatch.create(distance=3),
        "B",
        qubit_offset=patch.geometry.num_qubits,
        coord_offset=(8.0, 0.0),
    )
    builder.add_memory(["A", "B"], 2, {"A": "Z", "B": "Z"})
    builder.add_transversal_sz("A")
    builder.add_transversal_cx("A", "B")
    builder.add_memory(["A", "B"], 2, {"A": "Z", "B": "Z"})
    circuit = builder.to_tick_circuit()
    keys = json.loads(circuit.get_meta("measurement_keys"))
    last_segment = max(entry[3] for entry in keys["stabilizer"] if entry[0] == "B")
    affected_b = {
        entry[5] for entry in keys["stabilizer"] if entry[0] == "B" and entry[3] == last_segment and entry[4] == 0
    }
    detectors = json.loads(circuit.get_meta("detectors"))
    touching_b = [d for d in detectors if affected_b.intersection(d["meas_ids"])]
    assert touching_b
    assert all(d["fault_sector"] is None for d in touching_b)


@pytest.mark.parametrize("policy", ["global_axis_cycle_f", "checkerboard_xzzx", "checkerboard_zxxz"])
@pytest.mark.parametrize("basis", ["X", "Z"])
def test_clifford_deformed_surface_keeps_undeformed_family_sectors(policy, basis):
    from pecos.qec.surface.circuit_builder import generate_tick_circuit_from_patch

    def sectors(clifford_frame_policy):
        circuit = generate_tick_circuit_from_patch(
            SurfacePatch.create(distance=3),
            num_rounds=2,
            basis=basis,
            interaction_basis="szz",
            clifford_frame_policy=clifford_frame_policy,
        )
        return [d["fault_sector"] for d in json.loads(circuit.get_meta("detectors"))]

    deformed = sectors(policy)
    assert deformed == sectors(None)
    assert set(deformed) == {"X", "Z"}


def _h_memory_builder():
    builder = LogicalCircuitBuilder()
    builder.add_patch(SurfacePatch.create(distance=3), "A")
    builder.add_memory("A", 3, "Z")
    builder.add_transversal_h("A")
    builder.add_memory("A", 3, "X")
    return builder


def test_logical_fault_sectors_constant_across_h():
    circuit = _h_memory_builder().to_tick_circuit()
    detectors = json.loads(circuit.get_meta("detectors"))
    before = {tuple(d["coords"][:2]): d["fault_sector"] for d in detectors if d["coords"][2] == 2}
    after = {tuple(d["coords"][:2]): d["fault_sector"] for d in detectors if d["coords"][2] == 3}
    assert len(before) == len(after) == 8
    assert set(before.values()) == {"X", "Z"}
    assert before == after


def test_logical_h_single_faults_decode_with_code_family_sectors():
    model = DetectorErrorModel.from_circuit(
        _h_memory_builder().to_tick_circuit(),
        p1=0.004,
        p2=0.004,
        p_meas=0.004,
        p_prep=0.004,
    )
    raw = stim.DetectorErrorModel(model.to_string()).flattened()
    mechanisms = []
    truths_by_detectors = {}
    for instruction in raw:
        if instruction.type != "error":
            continue
        targets = instruction.targets_copy()
        detectors = frozenset(t.val for t in targets if t.is_relative_detector_id())
        truth = sum(1 << t.val for t in targets if t.is_logical_observable_id())
        mechanisms.append((detectors, truth))
        truths_by_detectors.setdefault(detectors, set()).add(truth)
    unambiguous = [(ds, truth) for ds, truth in mechanisms if len(truths_by_detectors[ds]) == 1]
    assert unambiguous
    shots = [[int(d in ds) for d in range(raw.num_detectors)] for ds, _ in unambiguous]
    batch = SampleBatch(shots, [truth for _, truth in unambiguous], num_observables=raw.num_observables)
    result = batch.decode(model, pymatching(correlated=False), workers=1, predictions=True)
    assert result.num_errors == 0


def test_measurement_classes_manual_binding():
    from pecos_rslib import TickCircuit
    from pecos_rslib.qec import DagFaultAnalyzer, DemBuilder

    circuit = TickCircuit()
    circuit.tick().mz([0, 1])
    influence = DagFaultAnalyzer(circuit.to_dag_circuit()).build_influence_map()
    dem = (
        DemBuilder(influence)
        .with_detectors_json('[{"id":0,"records":[0]},{"id":1,"records":[1]}]')
        .with_measurement_classes_json(
            json.dumps(
                [
                    {"block": "a", "family": "X", "round": -1},
                    {"block": "a", "family": "Z", "round": 0},
                ],
            ),
        )
        .build()
    )
    assert dem.detector_fault_sectors() == [(0, "X"), (1, "Z")]


def test_measurement_classes_circuit_attribute():
    from pecos_rslib import TickCircuit

    circuit = TickCircuit()
    circuit.tick().mz([0, 1])
    circuit.set_meta("detectors", '[{"id":0,"records":[0,1],"fault_sector":null}]')
    circuit.set_meta(
        "measurement_classes",
        json.dumps([{"block": "a", "family": "X", "round": -1}] * 2),
    )
    dem = DetectorErrorModel.from_circuit(circuit, p1=0.0, p2=0.0, p_meas=0.0, p_prep=0.0)
    assert dem.detector_fault_sectors() == [(0, "X")]
    assert circuit.circuit_definitions()["measurement_classes"] == [
        {"block": "a", "family": "X", "round": -1},
        {"block": "a", "family": "X", "round": -1},
    ]


def test_bb_measurement_classes_forwarded_by_tick_dem_generator(monkeypatch):
    import pecos.qec
    from pecos.qec import bb_memory_circuit
    from pecos.qec.surface.circuit_builder import generate_dem_from_tick_circuit

    real_builder = pecos.qec.DemBuilder
    builders = []

    def capture_builder(influence):
        builder = real_builder(influence)
        builders.append(builder)
        return builder

    monkeypatch.setattr(pecos.qec, "DemBuilder", capture_builder)
    circuit = bb_memory_circuit(6, 6, [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)], 2, "Z")
    generate_dem_from_tick_circuit(circuit, p1=0.0, p2=0.0, p_meas=0.001, p_prep=0.0)
    assert circuit.get_meta("measurement_classes") is not None
    sectors = [sector for _, sector in builders[0].build().detector_fault_sectors()]
    assert set(sectors) == {"X", "Z"}
    for detector, sector in zip(json.loads(circuit.get_meta("detectors")), sectors, strict=True):
        assert sector == ("X" if detector["label"].startswith("X") else "Z")


@pytest.fixture
def classified_circuit():
    from pecos_rslib import TickCircuit

    circuit = TickCircuit()
    circuit.tick().pz([0])
    circuit.tick().mz([0])
    circuit.set_meta("detectors", '[{"id":0,"records":[0]}]')
    circuit.set_meta("observables", '[{"id":0,"records":[0]}]')
    circuit.set_meta("measurement_classes", '[{"block":"memory","family":"Z","round":0}]')
    circuit.set_meta("num_measurements", "1")
    return circuit


@pytest.fixture
def capture_dem_builders(monkeypatch):
    import pecos_rslib.qec

    real_builder = pecos_rslib.qec.DemBuilder
    builders = []

    def capture(influence):
        builder = real_builder(influence)
        builders.append(builder)
        return builder

    monkeypatch.setattr(pecos_rslib.qec, "DemBuilder", capture)
    return builders


def test_with_circuit_metadata_replaces_all_four_attributes(classified_circuit):
    from pecos_rslib.qec import DagFaultAnalyzer, DemBuilder

    influence = DagFaultAnalyzer(classified_circuit.to_dag_circuit()).build_influence_map()
    builder = (
        DemBuilder(influence)
        .with_detectors_json("[]")
        .with_observables_json("[]")
        .with_measurement_classes_json("[]")
        .with_num_measurements(99)
    )
    assert builder.with_circuit_metadata(classified_circuit) is builder
    dem = builder.build()
    assert dem.detector_fault_sectors() == [(0, "Z")]
    assert dem.num_observables == 1


@pytest.mark.parametrize("blank", [None, "", " \n"], ids=["absent", "empty", "whitespace"])
def test_with_circuit_metadata_preserves_all_four_settings(blank):
    from pecos_rslib import TickCircuit
    from pecos_rslib.qec import DagFaultAnalyzer, DemBuilder

    circuit = TickCircuit()
    if blank is not None:
        for attribute in ("detectors", "observables", "num_measurements"):
            circuit.set_meta(attribute, blank)
    # An empty map permits a declarative count; preserving it is observable
    # because record 1 and the two-entry class list require a count of two.
    influence = DagFaultAnalyzer(TickCircuit().to_dag_circuit()).build_influence_map()
    dem = (
        DemBuilder(influence)
        .with_detectors_json('[{"id":0,"records":[1]}]')
        .with_observables_json('[{"id":0,"records":[1]}]')
        .with_measurement_classes_json('[null,{"block":"a","family":"X","round":0}]')
        .with_num_measurements(2)
        .with_circuit_metadata(circuit)
        .build()
    )
    assert dem.detector_fault_sectors() == [(0, "X")]
    assert dem.num_observables == 1


@pytest.mark.parametrize("attribute", ["detectors", "observables", "measurement_classes", "num_measurements"])
@pytest.mark.parametrize("value", [7, 1.5, True, [], {}])
def test_with_circuit_metadata_rejects_non_strings(classified_circuit, attribute, value):
    from pecos_rslib.qec import DagFaultAnalyzer, DemBuilder

    classified_circuit.set_meta(attribute, value)
    influence = DagFaultAnalyzer(classified_circuit.to_dag_circuit()).build_influence_map()
    with pytest.raises(ValueError, match=f'attribute "{attribute}" must be a string, got '):
        DemBuilder(influence).with_circuit_metadata(classified_circuit)


@pytest.mark.parametrize("value", ["-1", "1.0", "two", "184467440737095516160"])
def test_with_circuit_metadata_rejects_invalid_count(classified_circuit, value):
    from pecos_rslib.qec import DagFaultAnalyzer, DemBuilder

    classified_circuit.set_meta("num_measurements", value)
    influence = DagFaultAnalyzer(classified_circuit.to_dag_circuit()).build_influence_map()
    with pytest.raises(ValueError, match="num_measurements") as error:
        DemBuilder(influence).with_circuit_metadata(classified_circuit)
    assert value in str(error.value)


def test_with_circuit_metadata_changes_nothing_when_rejected(classified_circuit):
    from pecos_rslib.qec import DagFaultAnalyzer, DemBuilder

    classified_circuit.set_meta("num_measurements", "two")
    influence = DagFaultAnalyzer(classified_circuit.to_dag_circuit()).build_influence_map()
    # Detector id 1 differs from the circuit's id 0, so a leaked detectors attribute
    # changes the id, and a leaked classes attribute adds a label.
    builder = DemBuilder(influence).with_detectors_json('[{"id":1,"records":[0]}]').with_observables_json("[]")
    with pytest.raises(ValueError, match="num_measurements"):
        builder.with_circuit_metadata(classified_circuit)
    dem = builder.build()
    assert dem.detector_fault_sectors() == [(1, None)]
    assert dem.num_observables == 0


def annotated_circuit(attribute_kind, attribute_value, *, labelled=False):
    """Build a one-detector, one-observable annotated circuit with one attribute under test."""
    circuit = TickCircuit()
    circuit.tick().pz([0, 1])
    measurements = list(circuit.tick().mz([0]))
    circuit.tick().mz([1])
    label = "annotation" if labelled else None
    circuit.detector(measurements, label=label)
    circuit.observable(measurements, label=label)
    # The other kind's attribute is valid, so only the kind under test can fail.
    other = "observables" if attribute_kind == "detectors" else "detectors"
    circuit.set_meta(other, '[{"id":0,"records":[0]}]')
    if attribute_value is not None:
        circuit.set_meta(attribute_kind, attribute_value)
    return circuit


@pytest.mark.parametrize(
    "value",
    [None, "", " \n", "[]"],
    ids=["absent", "empty", "whitespace", "empty-list"],
)
@pytest.mark.parametrize("kind", ["detectors", "observables"])
def test_with_circuit_metadata_rejects_annotations_with_empty_attribute(kind, value):
    from pecos_rslib.qec import DagFaultAnalyzer, DemBuilder

    circuit = annotated_circuit(kind, value)
    influence = DagFaultAnalyzer(circuit.to_dag_circuit()).build_influence_map()
    builder = DemBuilder(influence)
    with pytest.raises(ValueError, match=f'{kind[:-1]} annotations but its "{kind}" attribute defines none'):
        builder.with_circuit_metadata(circuit)
    dem = builder.build()
    assert dem.detector_fault_sectors() == []
    assert dem.num_observables == 0

    advised = DetectorErrorModel.from_circuit(circuit, p1=0.01, p2=0.01, p_meas=0.01, p_prep=0.01)
    assert advised.num_detectors == 1
    assert advised.num_observables == 1


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ('[{"id":0,"records":[0]},{"id":1,"records":[0]}]', "ids differ"),
        ('[{"id":7,"records":[0]}]', "ids differ"),
        ('[{"id":0,"records":[1]}]', "differ from annotation positions"),
        ('[{"id":0,"records":[0],"label":"attribute"}]', "conflicting labels"),
        ('[{"id":0,"records":[0]},{"id":0,"records":[1]}]', "is repeated in metadata"),
        ('[{"id":0,"records":[9]}]', "out of range"),
    ],
    ids=["surplus", "other-id", "other-record", "label", "duplicate-id", "bad-record"],
)
@pytest.mark.parametrize("kind", ["detectors", "observables"])
def test_with_circuit_metadata_applies_the_from_circuit_rules(kind, value, message):
    from pecos_rslib.qec import DagFaultAnalyzer, DemBuilder

    circuit = annotated_circuit(kind, value, labelled=True)
    influence = DagFaultAnalyzer(circuit.to_dag_circuit()).build_influence_map()
    builder = DemBuilder(influence)
    with pytest.raises(ValueError, match=message):
        builder.with_circuit_metadata(circuit)
    with pytest.raises(ValueError, match=message):
        DetectorErrorModel.from_circuit(circuit, p1=0.01, p2=0.01, p_meas=0.01, p_prep=0.01)
    dem = builder.build()
    assert dem.detector_fault_sectors() == []
    assert dem.num_observables == 0


@pytest.mark.parametrize("kind", ["detectors", "observables"])
def test_with_circuit_metadata_rejects_partial_attribute(kind):
    from pecos_rslib.qec import DagFaultAnalyzer, DemBuilder

    circuit = annotated_circuit(kind, '[{"id":0,"records":[0]}]')
    measurements = list(circuit.tick().mz([0]))
    add_annotation = circuit.detector if kind == "detectors" else circuit.observable
    add_annotation(measurements)
    influence = DagFaultAnalyzer(circuit.to_dag_circuit()).build_influence_map()
    with pytest.raises(ValueError, match="ids differ"):
        DemBuilder(influence).with_circuit_metadata(circuit)


def test_with_circuit_metadata_rejects_duplicate_ids_without_annotations(classified_circuit):
    from pecos_rslib.qec import DagFaultAnalyzer, DemBuilder

    classified_circuit.set_meta("detectors", '[{"id":0,"records":[0]},{"id":0,"records":[0]}]')
    influence = DagFaultAnalyzer(classified_circuit.to_dag_circuit()).build_influence_map()
    with pytest.raises(ValueError, match="detector id 0 is repeated in metadata"):
        DemBuilder(influence).with_circuit_metadata(classified_circuit)


def test_with_circuit_metadata_keeps_one_copy_of_a_map_annotation():
    from pecos_rslib.qec import DagFaultAnalyzer, DemBuilder, InfluenceBuilder

    circuit = TickCircuit()
    circuit.tick().pz([0, 1])
    circuit.tick().mz([0])
    circuit.observable(list(circuit.tick().mz([1])))
    circuit.set_meta("detectors", '[{"id":0,"records":[0]},{"id":1,"records":[1]}]')
    circuit.set_meta("observables", '[{"id":0,"records":[1]}]')
    dag = circuit.to_dag_circuit()
    influence = DagFaultAnalyzer(dag).build_influence_map()
    influence.merge_dem_outputs_from(InfluenceBuilder(dag).with_circuit_annotations().build())
    dem = DemBuilder(influence).with_noise(0.01, 0.01, 0.01, 0.01).with_circuit_metadata(circuit).build()
    assert "D1 L0" in dem.to_string()


def measured_circuit(*gates):
    circuit = TickCircuit()
    circuit.tick().pz([0, 1])
    for gate in gates:
        gate(circuit.tick())
    return circuit


def z_on(qubit):
    return lambda tick: tick.mz([qubit])


def x_on(qubit):
    return lambda tick: tick.mx([qubit])


def cx(tick):
    tick.cx([(0, 1)])


@pytest.mark.parametrize(
    ("map_gates", "message"),
    [
        ((z_on(1), z_on(0)), r"record 0: map has node \d+ qubit 1 basis 0.*circuit has node \d+ qubit 0"),
        ((z_on(0),), r"record 1: map has no record, circuit has node \d+ qubit 1"),
        ((z_on(0), z_on(1), z_on(0)), r"record 2: map has node \d+ qubit 0.*circuit has no record"),
        ((), r"record 0: map has no record, circuit has node \d+ qubit 0"),
        ((z_on(0), x_on(1)), r"record 1: map has node \d+ qubit 1 basis 1.*build the influence map"),
        ((cx, z_on(0), z_on(1)), r"record 0: map has node \d+ qubit 0"),
    ],
    ids=["reversed", "shorter", "longer", "empty", "other-basis", "extra-gate-first"],
)
def test_with_circuit_metadata_rejects_map_of_another_circuit(map_gates, message):
    from pecos_rslib.qec import DagFaultAnalyzer, DemBuilder

    circuit = measured_circuit(z_on(0), z_on(1))
    circuit.set_meta("detectors", '[{"id":0,"records":[0]}]')
    influence = DagFaultAnalyzer(measured_circuit(*map_gates).to_dag_circuit()).build_influence_map()
    with pytest.raises(ValueError, match=message):
        DemBuilder(influence).with_circuit_metadata(circuit)


def test_with_circuit_metadata_rejects_map_with_other_measurement_ids():
    from pecos_rslib.qec import DagFaultAnalyzer, DemBuilder

    def stamped(ids):
        circuit = TickCircuit()
        circuit.tick().pz([0, 1])
        circuit.tick().mz_with_ids([0, 1], ids)
        return circuit

    circuit = stamped([1, 0])
    circuit.set_meta("detectors", '[{"id":0,"meas_ids":[0]}]')
    influence = DagFaultAnalyzer(stamped([0, 1]).to_dag_circuit()).build_influence_map()
    with pytest.raises(ValueError, match="record 0"):
        DemBuilder(influence).with_circuit_metadata(circuit)


def test_with_circuit_metadata_resolves_ids_the_dag_minted():
    from pecos_rslib.qec import DagFaultAnalyzer, DemBuilder

    circuit = TickCircuit()
    circuit.tick().pz([0, 1])
    circuit.tick().mz_with_ids([0], [5])
    circuit.tick().add_gate("MZ", [1])
    # Id 6 is out of range as an ordinal, so only the DAG's minted id resolves it.
    circuit.set_meta("detectors", '[{"id":0,"meas_ids":[6]}]')
    influence = DagFaultAnalyzer(circuit.to_dag_circuit()).build_influence_map()
    dem = DemBuilder(influence).with_noise(0.01, 0.01, 0.01, 0.01).with_circuit_metadata(circuit).build()
    expected = DetectorErrorModel.from_circuit(circuit, p1=0.01, p2=0.01, p_meas=0.01, p_prep=0.01)
    assert "error" in expected.to_string()
    assert dem.to_string() == expected.to_string()


@pytest.mark.parametrize("declared", ["0", "2"])
def test_with_circuit_metadata_rejects_count_differing_from_circuit(classified_circuit, declared):
    from pecos_rslib.qec import DagFaultAnalyzer, DemBuilder

    classified_circuit.set_meta("num_measurements", declared)
    influence = DagFaultAnalyzer(classified_circuit.to_dag_circuit()).build_influence_map()
    with pytest.raises(ValueError, match=f"num_measurements={declared} disagrees with the 1"):
        DemBuilder(influence).with_circuit_metadata(classified_circuit)


def test_with_circuit_metadata_rejects_count_on_empty_circuit():
    from pecos_rslib.qec import DagFaultAnalyzer, DemBuilder

    circuit = TickCircuit()
    circuit.set_meta("num_measurements", "5")
    circuit.set_meta("detectors", '[{"id":0,"records":[3]}]')
    influence = DagFaultAnalyzer(circuit.to_dag_circuit()).build_influence_map()
    with pytest.raises(ValueError, match="num_measurements=5 disagrees with the 0"):
        DemBuilder(influence).with_circuit_metadata(circuit)


@pytest.mark.parametrize("kind", ["detectors", "observables"])
def test_with_circuit_metadata_requires_only_the_annotated_kind(kind):
    from pecos_rslib.qec import DagFaultAnalyzer, DemBuilder

    circuit = TickCircuit()
    circuit.tick().pz([0])
    measurements = list(circuit.tick().mz([0]))
    add_annotation = circuit.detector if kind == "detectors" else circuit.observable
    add_annotation(measurements)
    circuit.set_meta(kind, '[{"id":0,"records":[0]}]')
    influence = DagFaultAnalyzer(circuit.to_dag_circuit()).build_influence_map()
    dem = DemBuilder(influence).with_circuit_metadata(circuit).build()
    assert (dem.num_detectors, dem.num_observables) == ((1, 0) if kind == "detectors" else (0, 1))


def test_with_circuit_metadata_accepts_annotations_with_attributes(classified_circuit):
    from pecos_rslib.qec import DagFaultAnalyzer, DemBuilder

    measurements = list(classified_circuit.tick().mz([0]))
    classified_circuit.detector(measurements)
    classified_circuit.observable(measurements)
    classified_circuit.set_meta("detectors", '[{"id":0,"records":[1]}]')
    classified_circuit.set_meta("observables", '[{"id":0,"records":[1]}]')
    classified_circuit.set_meta("measurement_classes", '[null,{"block":"memory","family":"Z","round":1}]')
    classified_circuit.set_meta("num_measurements", "2")
    influence = DagFaultAnalyzer(classified_circuit.to_dag_circuit()).build_influence_map()
    dem = DemBuilder(influence).with_circuit_metadata(classified_circuit).build()
    assert dem.detector_fault_sectors() == [(0, "Z")]
    assert dem.num_observables == 1


@pytest.mark.parametrize("blank", ["", " \n"], ids=["empty", "whitespace"])
def test_generate_dem_rejects_blank_detectors(classified_circuit, blank):
    from pecos.qec.surface.circuit_builder import generate_dem_from_tick_circuit

    classified_circuit.set_meta("detectors", blank)
    with pytest.raises(ValueError, match=r"TickCircuit must have detector metadata.*DetectorErrorModel\.from_circuit"):
        generate_dem_from_tick_circuit(classified_circuit, p1=0.01, p2=0.01, p_meas=0.01, p_prep=0.01)


@pytest.mark.parametrize("count", [None, "", " \n"], ids=["absent", "empty", "whitespace"])
def test_generate_dem_takes_an_undeclared_count_from_the_map(count):
    from pecos.qec.surface.circuit_builder import generate_dem_from_tick_circuit

    def generate(declared):
        circuit = TickCircuit()
        circuit.tick().pz([0])
        circuit.tick().mz([0])
        circuit.set_meta("detectors", '[{"id":0,"records":[-1]}]')
        if declared is not None:
            circuit.set_meta("num_measurements", declared)
        return generate_dem_from_tick_circuit(circuit, p1=0.01, p2=0.01, p_meas=0.01, p_prep=0.01)

    assert "D0" in generate(count)
    assert generate(count) == generate("1")


@pytest.mark.parametrize("blank", ["", " \n"], ids=["empty", "whitespace"])
def test_with_circuit_metadata_rejects_blank_classes(classified_circuit, blank):
    from pecos_rslib.qec import DagFaultAnalyzer, DemBuilder

    classified_circuit.set_meta("measurement_classes", blank)
    influence = DagFaultAnalyzer(classified_circuit.to_dag_circuit()).build_influence_map()
    builder = DemBuilder(influence).with_circuit_metadata(classified_circuit)
    with pytest.raises(ValueError, match="measurement_classes: malformed JSON"):
        builder.build()


@pytest.mark.parametrize("attribute", ["detectors", "observables"])
def test_with_circuit_metadata_rejects_malformed_definitions(classified_circuit, attribute):
    from pecos_rslib.qec import DagFaultAnalyzer, DemBuilder

    classified_circuit.set_meta(attribute, "not JSON")
    influence = DagFaultAnalyzer(classified_circuit.to_dag_circuit()).build_influence_map()
    with pytest.raises(ValueError, match=f"invalid circuit definitions: {attribute[:-1]} metadata"):
        DemBuilder(influence).with_circuit_metadata(classified_circuit)


def test_with_circuit_metadata_leaves_malformed_classes_to_build(classified_circuit):
    from pecos_rslib.qec import DagFaultAnalyzer, DemBuilder

    classified_circuit.set_meta("measurement_classes", "not JSON")
    influence = DagFaultAnalyzer(classified_circuit.to_dag_circuit()).build_influence_map()
    builder = DemBuilder(influence).with_circuit_metadata(classified_circuit)
    with pytest.raises(ValueError, match="measurement_classes: malformed JSON"):
        builder.build()


def test_fit_dem_from_simulation_forwards_classes(classified_circuit, capture_dem_builders):
    from pecos.qec.analysis import fit_dem_from_simulation
    from pecos_rslib_exp import depolarizing

    fit_dem_from_simulation(classified_circuit, depolarizing().p_meas(0.01), 8)
    assert len(capture_dem_builders) == 1
    assert capture_dem_builders[0].build().detector_fault_sectors() == [(0, "Z")]


def test_build_adaptive_dem_forwards_classes(classified_circuit, capture_dem_builders):
    from pecos.qec.analysis import build_adaptive_dem

    build_adaptive_dem(classified_circuit, {"p_meas": 0.01})
    assert len(capture_dem_builders) == 1
    assert capture_dem_builders[0].build().detector_fault_sectors() == [(0, "Z")]


@pytest.mark.parametrize("site", ["fit_dem_from_simulation", "build_adaptive_dem"])
@pytest.mark.parametrize("attribute", ["detectors", "observables", "num_measurements"])
@pytest.mark.parametrize("blank", [None, "", " \n"], ids=["absent", "empty", "whitespace"])
def test_analysis_requires_metadata(site, attribute, blank):
    from pecos.qec import analysis
    from pecos_rslib import TickCircuit
    from pecos_rslib_exp import depolarizing

    circuit = TickCircuit()
    for name, value in [("detectors", "[]"), ("observables", "[]"), ("num_measurements", "0")]:
        if name != attribute:
            circuit.set_meta(name, value)
        elif blank is not None:
            circuit.set_meta(name, blank)
    args = (circuit, depolarizing(), 1) if site == "fit_dem_from_simulation" else (circuit, {"p_meas": 0.01})
    with pytest.raises(ValueError, match=f"requires {attribute} metadata.*reads attributes"):
        getattr(analysis, site)(*args)


def test_logical_structured_dem_forwards_classes(monkeypatch, capture_dem_builders):
    from pecos.qec import bb_memory_circuit

    circuit = bb_memory_circuit(6, 6, [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)], 2, "Z")
    monkeypatch.setattr(LogicalCircuitBuilder, "to_tick_circuit", lambda _self: circuit)
    LogicalCircuitBuilder().build_dem()
    assert len(capture_dem_builders) == 1
    sectors = capture_dem_builders[0].build().detector_fault_sectors()
    assert {sector for _, sector in sectors} == {"X", "Z"}
    for detector, (_, sector) in zip(json.loads(circuit.get_meta("detectors")), sectors, strict=True):
        assert sector == ("X" if detector["label"].startswith("X") else "Z")


def test_bb_resolved_definitions_carry_dem_labels():
    from pecos.qec import bb_memory_circuit

    circuit = bb_memory_circuit(6, 6, [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)], 2, "Z")
    raw = json.loads(circuit.get_meta("detectors"))
    assert raw
    assert all("fault_sector" not in detector for detector in raw)
    dem = DetectorErrorModel.from_circuit(circuit, p1=0.0, p2=0.0, p_meas=0.001, p_prep=0.0)
    sectors = dem.detector_fault_sectors()
    assert {sector for _, sector in sectors} == {"X", "Z"}
    assert [(d["id"], d["fault_sector"]) for d in circuit.circuit_definitions()["detectors"]] == sectors
