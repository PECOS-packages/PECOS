# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Corrected SZ parities, symbolic crossings, and downstream observable contracts."""

import json
from dataclasses import replace

import pytest
import stim
from pecos.qec.surface import LogicalCircuitBuilder, SurfacePatch, extract_detection_events_and_observables
from pecos.qec.surface.circuit_builder import get_observable_descriptors_from_tick_circuit, tick_circuit_to_stim
from pecos.qec.surface.logical_circuit import LogicalGateType, LogicalOp, _logical_readout_flow
from pecos.testing import simulate_tick_circuit
from pecos_rslib.qec import LogicalAlgorithmDecoder, LogicalCircuitDecoder


class ReadoutProbe(LogicalCircuitBuilder):
    """Expose symbolic policy without changing the physical gate sequence."""

    def omit_sz_correction(self):
        self._operations = [replace(op, teleportation_type=None) if op.teleportation else op for op in self._operations]

    def output_ids(self):
        return self._assembled_dem_output_ids()

    def cached_model(self):
        return self._build_structured_dem_from_cached_slices(p1=0.001, p2=0.001, p_meas=0.001, p_prep=0.0)


def readout_builder(distance=3, *, dz=None, gate="SZ", prepared="X", readout="Y", sparse=False):
    patch = SurfacePatch.create(dx=distance, dz=dz or distance)
    builder = ReadoutProbe()
    labels = ["R", "M", "D", "A"] if sparse else ["D", "A"]
    for index, label in enumerate(labels):
        builder.add_patch(patch, label, qubit_offset=index * patch.geometry.num_qubits)
    if sparse:
        builder.add_memory("R", 2, "X")
        builder.add_memory("R", 2, "Z")  # Random readout reserves id 0.
        builder.add_memory("M", 2, "Z")  # Existing memory keeps id 1.
    builder.add_memory("D", 2, prepared)
    if gate == "T":
        builder.add_t_teleportation_placeholder("D", "A", 2, 2)
    else:
        builder.add_sz_via_teleportation("D", "A", 2, 2)
    builder.add_memory("D", 2, readout)
    return builder, patch


def fold_oracle(distance, *, dagger=False):
    builder = LogicalCircuitBuilder()
    builder.add_patch(SurfacePatch.create(distance), "D")
    builder.add_memory("D", 2, "X")
    builder.add_logical_sz("D", dagger=dagger)
    builder.add_memory("D", 2, "Y")
    return builder.to_tick_circuit()


@pytest.mark.parametrize("distance", [3, 5])
def test_teleported_mask_and_noiseless_flips(distance):
    builder, patch = readout_builder(distance)
    tc = builder.to_tick_circuit()
    observables = json.loads(tc.get_meta("observables"))
    assert len(observables) == 1
    observable = observables[0]
    assert observable["id"] == 0
    keys = json.loads(tc.get_meta("measurement_keys"))
    data = [
        ordinal for label, qubit, ordinal in keys["data"] if label == "D" and qubit in range(0, distance**2, distance)
    ]
    fold = [
        ordinal
        for label, family, _, segment, _, ordinal in keys["stabilizer"]
        if (label, family, segment) == ("D", "Z", 3)
    ]
    readout = json.loads(tc.get_meta("teleportation_readouts"))[0]
    ancilla = readout["meas_ids"]
    resource = readout["resource_sign_meas_ids"]
    assert len(data) == len(ancilla) == distance
    assert len(fold) == len(resource) == (distance**2 - 1) // 2
    assert {family for label, family, _, segment, rnd, ordinal in keys["stabilizer"] if ordinal in resource} == {
        "X",
        "Z",
    }
    assert observable["meas_ids"] == data + fold + ancilla + resource
    assert len(set(observable["meas_ids"])) == {3: 14, 5: 34}[distance]
    assert observable["records"] == [index - tc.num_measurements() for index in observable["meas_ids"]]
    assert readout["resource_sign_reference"] == 0
    descriptors = get_observable_descriptors_from_tick_circuit(tc, patch)
    assert len(descriptors) == 1
    assert descriptors[0]["records"] == observable["records"]
    assert descriptors[0]["basis"] == descriptors[0]["logical_type"] == "X"
    assert descriptors[0]["data_qubits"] == list(range(0, distance**2, distance))

    # Each raw subset is random; only the complete corrected mask is deterministic.
    subset_parities = [set(), set()]
    for seed in range(8):
        measurements, fired, raw = simulate_tick_circuit(tc, seed)
        assert fired == 0
        assert raw == {0: 0}
        for seen, records in zip(subset_parities, (ancilla, resource), strict=True):
            seen.add(sum(measurements[index] for index in records) % 2)
    assert subset_parities == [{0, 1}, {0, 1}]

    for circuit, expected_raw in ((tc, 0), (fold_oracle(distance), 0), (fold_oracle(distance, dagger=True), 1)):
        exported = stim.Circuit(tick_circuit_to_stim(circuit))
        for seed in range(4):
            samples = exported.compile_sampler(seed=seed).sample(64)
            events, flips = extract_detection_events_and_observables(circuit, samples)
            assert events == flips == [[] for _ in range(64)]
            events, flips = exported.compile_detector_sampler(seed=seed).sample(64, separate_observables=True)
            assert not events.any()
            assert not flips.any()
            assert simulate_tick_circuit(circuit, seed)[2] == {0: expected_raw}


@pytest.mark.parametrize("distance", [3, 5])
def test_signed_accessor_distinguishes_phase_inverse(distance):
    for dagger in (False, True):
        tc = fold_oracle(distance, dagger=dagger)
        assert json.loads(tc.get_meta("observables"))[0]["reference"] == int(dagger)
        for seed in range(4):
            measurements, _, raw = simulate_tick_circuit(tc, seed)
            assert raw == {0: int(dagger)}
            assert extract_detection_events_and_observables(tc, [measurements])[1] == [[]]


@pytest.mark.parametrize("distance", [3, 5])
def test_detectors_and_existing_ids_unchanged(distance):
    builder, _ = readout_builder(distance, sparse=True)
    corrected = builder.to_tick_circuit()
    builder.omit_sz_correction()
    baseline = builder.to_tick_circuit()
    assert corrected.get_meta("detectors") == baseline.get_meta("detectors")
    assert len(json.loads(corrected.get_meta("detectors"))) == {3: 148, 5: 444}[distance]
    assert corrected.get_meta("measurement_keys") == baseline.get_meta("measurement_keys")
    before = json.loads(baseline.get_meta("observables"))
    after = json.loads(corrected.get_meta("observables"))
    assert [entry["id"] for entry in before] == [1]
    assert [entry["id"] for entry in after] == [1, 2]
    assert after[:1] == before


@pytest.mark.parametrize(("distance", "dz"), [(3, 3), (5, 5), (3, 5), (5, 3), (5, 7)])
def test_z_readout_unchanged(distance, dz):
    builder, _ = readout_builder(distance, dz=dz, prepared="Z", readout="Z")
    tc = builder.to_tick_circuit()
    builder.omit_sz_correction()
    baseline = builder.to_tick_circuit()
    assert tc.get_meta("observables") == baseline.get_meta("observables")
    assert len(json.loads(tc.get_meta("observables"))[0]["records"]) == dz
    assert simulate_tick_circuit(tc)[2] == {0: 0}


def test_t_readout_unchanged():
    builder, _ = readout_builder(gate="T")
    tc = builder.to_tick_circuit()
    assert json.loads(tc.get_meta("observables")) == []
    assert builder.output_ids() == []
    assert stim.Circuit(builder.to_stim()).detector_error_model().num_observables == 0
    assert "resource_sign_records" not in json.loads(tc.get_meta("teleportation_readouts"))[0]


@pytest.mark.parametrize(("dx", "dz"), [(3, 3), (3, 5), (5, 3), (5, 7)])
def test_multiple_teleportations_downstream(dx, dz):
    builder, _ = multiple_teleportations_builder(dx, dz)
    tc = builder.to_tick_circuit()
    assert builder.output_ids() == [entry["id"] for entry in json.loads(tc.get_meta("observables"))] == [0]
    model = stim.DetectorErrorModel(builder.build_dem())
    assert model.num_observables == 1
    exported = stim.Circuit(builder.to_stim())
    assert exported.detector_error_model().num_observables == 1
    _, flips = exported.compile_detector_sampler(seed=0).sample(64, separate_observables=True)
    assert flips.shape == (64, 1)
    assert not flips.any()
    descriptor = builder.build_algorithm_descriptor()
    assert descriptor["num_observables"] == 1
    assert descriptor["num_frame_slots"] == 6
    for use_stim_dem in (False, True):
        _, decoder = builder.build_decoder(inner_decoder="pymatching", use_stim_dem=use_stim_dem)
        assert decoder.num_observables() == 1
    sampler, decoder, _ = builder.build_sampler_and_decoder()
    assert sampler.num_observables == decoder.num_observables() == 1


@pytest.mark.parametrize("distance", [3, 5])
def test_teleported_observable_has_distance_one_mechanism(distance):
    builder, _ = readout_builder(distance)
    model = stim.DetectorErrorModel(builder.build_dem(p1=0.001, p2=0.001, p_meas=0.001))
    undetected = [
        instruction
        for instruction in model.flattened()
        if instruction.type == "error"
        and instruction.targets_copy() == [stim.target_logical_observable_id(0)]
        and instruction.args_copy()[0] > 0
    ]
    assert len(undetected) == 1


@pytest.mark.parametrize("distance", [3, 5])
def test_sparse_downstream_consumers(distance):
    builder, patch = readout_builder(distance, sparse=True)
    tc = builder.to_tick_circuit()
    assert builder.output_ids() == [1, 2]
    descriptors = get_observable_descriptors_from_tick_circuit(tc, patch)
    assert [entry["observable_id"] for entry in descriptors] == [1, 2]
    assert [entry["basis"] for entry in descriptors] == ["Z", "X"]
    model = stim.DetectorErrorModel(builder.build_dem())
    assert model.num_observables == 3
    circuit = stim.Circuit(builder.to_stim(p1=0.001, p2=0.001, p_meas=0.001))
    assert circuit.detector_error_model().num_observables == 3
    _, flips = circuit.compile_detector_sampler(seed=0).sample(4, separate_observables=True)
    assert flips.shape == (4, 3)
    assert not flips[:, 0].any()
    descriptor = builder.build_algorithm_descriptor()
    assert descriptor["num_observables"] == 3
    assert descriptor["num_frame_slots"] == 8
    for decoder_type in (LogicalAlgorithmDecoder, LogicalCircuitDecoder):
        kwargs = {"budget": "unlimited"} if decoder_type is LogicalCircuitDecoder else {}
        decoder = decoder_type(descriptor, **kwargs)
        assert decoder.decode([0] * model.num_detectors) == 0
    for use_stim_dem in (False, True):
        _, decoder = builder.build_decoder(inner_decoder="pymatching", use_stim_dem=use_stim_dem)
        assert decoder.num_observables() == 3
        assert decoder.decode([0] * model.num_detectors) == 0
    sampler, decoder, _ = builder.build_sampler_and_decoder()
    assert sampler.num_observables == decoder.num_observables() == 3
    for seed in range(4):
        _, flips = sampler.sample(seed)
        assert len(flips) == 3
        assert not flips[0]


@pytest.mark.parametrize("gate", ["SZ", "T"])
def test_teleportations_stay_out_of_cached_models(gate):
    # This alternating memory/CX/memory layout would otherwise qualify for both
    # repeated-CX and mixed-H/CX caches; no Y-readout fold masks the exclusion.
    builder = ReadoutProbe()
    patch = SurfacePatch.create(3)
    builder.add_patch(patch, "D")
    builder.add_patch(patch, "A", qubit_offset=patch.geometry.num_qubits)
    if gate == "SZ":
        builder.add_sz_via_teleportation("D", "A", 2, 2)
    else:
        builder.add_t_teleportation_placeholder("D", "A", 2, 2)
    assert builder.cached_model() is None


@pytest.mark.parametrize(
    ("patch", "kind", "prepared", "expected"),
    [
        ("D", "X", "X", (False, (), ())),
        ("D", "Z", "Z", (True, (), ())),
        ("A", "X", "X", (True, (), ())),
        ("B", "X", "X", (True, (), ())),
    ],
)
def test_walk_control_only(patch, kind, prepared, expected):
    operations = [
        LogicalOp(LogicalGateType.MEMORY, ["D", "A", "B"], rounds=2, basis=prepared),
        LogicalOp(LogicalGateType.TRANSVERSAL_CX, ["D", "A"], teleportation=True, teleportation_type="SZ"),
        LogicalOp(LogicalGateType.MEMORY, ["D", "A", "B"], rounds=2, basis=kind),
    ]
    assert _logical_readout_flow(operations, 1, patch, kind) == expected


@pytest.mark.parametrize("teleportation_type", ["SZ", "T", None])
def test_walk_replaces_conjugation_and_returns_symbols(teleportation_type):
    operations = [
        LogicalOp(LogicalGateType.MEMORY, ["D", "A"], rounds=2, per_patch_basis={"D": "X", "A": "Y"}),
        LogicalOp(
            LogicalGateType.TRANSVERSAL_CX,
            ["D", "A"],
            teleportation=True,
            teleportation_type=teleportation_type,
        ),
        LogicalOp(LogicalGateType.FOLD_SZ, ["D"], rounds=1, fold="SZdg"),
        LogicalOp(LogicalGateType.MEMORY, ["D", "A"], rounds=2, basis="X"),
    ]
    expected = (True, (("D", 1),), ("A",)) if teleportation_type == "SZ" else (False, (), ())
    assert _logical_readout_flow(operations, 2, "D", "X") == expected


def multiple_teleportations_builder(dx, dz):
    builder, patch = readout_builder(dx, dz=dz, readout="X")
    builder.add_patch(patch, "B", qubit_offset=2 * patch.geometry.num_qubits)
    builder.add_sz_via_teleportation("D", "B", 2, 2)
    builder.add_memory("D", 2, "X")
    return builder, patch


@pytest.mark.parametrize(("dx", "dz", "constant"), [(3, 3, 0), (3, 5, 1), (5, 3, 1), (5, 7, 1), (3, 7, 0)])
def test_multiple_teleportations_x_readout(dx, dz, constant):
    builder, _ = multiple_teleportations_builder(dx, dz)
    tc = builder.to_tick_circuit()
    observables = json.loads(tc.get_meta("observables"))
    assert len(observables) == 1
    readouts = json.loads(tc.get_meta("teleportation_readouts"))
    assert [entry["resource_sign_reference"] for entry in readouts] == [constant, constant]
    keys = json.loads(tc.get_meta("measurement_keys"))
    data = [ordinal for label, qubit, ordinal in keys["data"] if label == "D" and qubit % dz == 0]
    corrections = []
    for readout in reversed(readouts):
        corrections.extend(readout["meas_ids"])
        corrections.extend(readout["resource_sign_meas_ids"])
    assert len(data) == dx
    assert observables[0]["meas_ids"] == data + corrections
    assert observables[0]["reference"] == 1
    for seed in range(8):
        measurements, fired, raw = simulate_tick_circuit(tc, seed)
        assert fired == 0
        assert raw == {0: 1}
        assert extract_detection_events_and_observables(tc, [measurements])[1] == [[]]


@pytest.mark.parametrize("swapped", [False, True])
def test_builder_descriptors_follow_patch_orientation(swapped):
    builder = LogicalCircuitBuilder()
    patch = SurfacePatch.create(3)
    builder.add_patch(patch, "D")
    builder.add_memory("D", 2, "Z" if swapped else "X")
    if swapped:
        builder.add_transversal_h("D")
    builder.add_memory("D", 2, "X")
    tc = builder.to_tick_circuit()
    descriptor = get_observable_descriptors_from_tick_circuit(tc, patch)[0]
    assert descriptor["basis"] == "X"
    assert descriptor["logical_type"] == ("Z" if swapped else "X")
    assert descriptor["data_qubits"] == ([0, 1, 2] if swapped else [0, 3, 6])
