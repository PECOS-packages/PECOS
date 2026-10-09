# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Fold builder parity-space oracles and the known X-sector distance reduction."""

import json
import re
from dataclasses import replace

import pytest
import stim
from pecos.qec import DetectorErrorModel
from pecos.qec.surface import LogicalCircuitBuilder, SurfacePatch, extract_detection_events_and_observables
from pecos.qec.surface.circuit_builder import tick_circuit_to_stim
from pecos.qec.surface.logical_circuit import (
    LogicalGateType,
    LogicalOp,
    PatchState,
    _CircuitGenerator,
    _logical_readout_flow,
)
from pecos.qec.surface.patch import PatchOrientation
from pecos.testing import deterministic_parity_basis, group_contains, simulate_tick_circuit, stabilizer_generators_after
from pecos_rslib.qec import LogicalCircuitDecoder


class FoldMapProbe(_CircuitGenerator):
    """Expose geometry validation without emitting an invalid physical gadget."""

    def fold_maps(self, label):
        return self._fold_check_maps(label)


def fold_builder(shape: str, distance: int = 3) -> LogicalCircuitBuilder:
    """Exercise folds at segment edges, in either frame, and around a CX."""
    patch = SurfacePatch.create(distance=distance)
    builder = LogicalCircuitBuilder()
    builder.add_patch(patch, "A")
    if shape in {"mem_Z_zero_final", "h_zero_final"}:
        builder.add_memory("A", 2, "Z")
        if shape == "h_zero_final":
            builder.add_transversal_h("A")
        builder.add_memory("A", 0, "X" if shape == "h_zero_final" else "Z")
    elif shape == "sign_cx":
        builder.add_patch(patch, "B", qubit_offset=patch.geometry.num_qubits)
        builder.add_memory("A", 1, "X")
        builder.add_memory("B", 1, "Z")
        builder.add_logical_szdg("A")
        builder.add_transversal_cx("A", "B")
        builder.add_logical_szdg("B")
        builder.add_transversal_cx("A", "B")
        builder.add_logical_szdg("B")
        builder.add_memory(["A", "B"], 1, "X")
    elif shape.startswith("cx_"):
        builder.add_patch(patch, "B", qubit_offset=patch.geometry.num_qubits)
        builder.add_memory(["A", "B"], 2, "Z")
        if shape == "cx_before":
            builder.add_logical_sz("A")
        builder.add_transversal_cx("A", "B")
        if shape == "cx_after":
            builder.add_logical_sz("A")
        builder.add_memory(["A", "B"], 2, "Z")
    elif shape == "h_fold":
        builder.add_memory("A", 2, "X")
        builder.add_transversal_h("A")
        builder.add_logical_sz("A")
        builder.add_memory("A", 2, "Z")
    elif shape.startswith("pair"):
        builder.add_memory("A", 1, "X")
        builder.add_logical_sz("A")
        if shape == "pair_separated":
            builder.add_memory("A", 2, "X")
        builder.add_logical_sz("A", dagger=shape not in {"pair_sz_sz", "pair_sz_sz_h"})
        if shape == "pair_sz_sz_h":
            builder.add_transversal_h("A")
        builder.add_memory("A", 1, "Z" if shape == "pair_sz_sz_h" else "X")
    else:
        before, after = {"first": (0, 2), "mid": (1, 1), "last": (2, 0), "single_x": (1, 1)}[shape]
        basis = "X" if shape == "single_x" else "Z"
        builder.add_memory("A", before, basis)
        builder.add_logical_sz("A")
        builder.add_memory("A", after, basis)
    return builder


def _rank(rows: list[int]) -> int:
    pivots = {}
    for row in rows:
        value = row
        while value:
            pivot = value.bit_length() - 1
            if pivot not in pivots:
                pivots[pivot] = value
                break
            value ^= pivots[pivot]
    return len(pivots)


def _mask(records: list[int]) -> int:
    result = 0
    for record in records:
        result ^= 1 << record
    return result


RANK_SHAPES = [
    (shape, 3)
    for shape in (
        "mem_Z_zero_final",
        "h_zero_final",
        "first",
        "mid",
        "last",
        "pair_adjacent",
        "pair_separated",
        "h_fold",
        "cx_before",
        "cx_after",
        "single_x",
    )
] + [("pair_adjacent", 5), ("pair_separated", 5)]


@pytest.mark.parametrize(("shape", "distance"), RANK_SHAPES)
def test_fold_parity_space(shape, distance):
    """Every emitted parity lies in, and together they span, the noiseless space."""
    tc = fold_builder(shape, distance).to_tick_circuit()
    circuit = stim.Circuit(tick_circuit_to_stim(tc))
    shots = circuit.compile_sampler(seed=0).sample(2048)
    space = deterministic_parity_basis(shots)
    emitted = [
        _mask(entry["meas_ids"]) for key in ("detectors", "observables") for entry in json.loads(tc.get_meta(key))
    ]
    rank = _rank(emitted)
    assert _rank([*space, *emitted]) == len(space), f"{shape}: emitted parity outside deterministic space"
    assert rank == len(space), f"{shape}: dimension={len(space)}, emitted rank={rank}"
    assert len(space) == circuit.count_determined_measurements()
    for seed in range(8):
        dets, obs = circuit.compile_detector_sampler(seed=seed).sample(256, separate_observables=True)
        assert not dets.any()
        assert not obs.any()


@pytest.mark.parametrize(("shape", "count"), [("mid", 1), ("single_x", 0), ("pair_adjacent", 1)])
def test_fold_observables(shape, count):
    tc = fold_builder(shape).to_tick_circuit()
    observables = json.loads(tc.get_meta("observables"))
    assert len(observables) == count
    if not count:
        return
    keys = json.loads(tc.get_meta("measurement_keys"))
    fold_z = {
        ordinal
        for label, family, index, segment, rnd, ordinal in keys["stabilizer"]
        if family == "Z" and segment in ({1, 2} if shape == "pair_adjacent" else {1})
    }
    selected = set(observables[0]["meas_ids"])
    assert (selected & fold_z) == (fold_z if shape == "pair_adjacent" else set())
    circuit = stim.Circuit(tick_circuit_to_stim(tc))
    for seed in range(8):
        shots = circuit.compile_sampler(seed=seed).sample(256)
        assert not (shots[:, sorted(selected)].sum(axis=1) % 2).any()


@pytest.mark.parametrize(
    ("shape", "distance", "expected"),
    [
        ("mid", 3, 3),
        ("pair_adjacent", 3, 2),
        pytest.param("mid", 5, 5, marks=pytest.mark.slow),
        ("pair_adjacent", 5, 4),
    ],
)
def test_fold_fault_distance(shape, distance, expected):
    """Pin the known reduction to d-1 for adjacent X-prepared SZ/SZdg rounds."""
    tc = fold_builder(shape, distance).to_tick_circuit()
    dem = DetectorErrorModel.from_circuit(tc, p1=0.001, p2=0.001, p_meas=0.001, p_prep=0.001)
    distances = dem.per_observable_fault_distances(distance)
    assert len(distances) == 1
    assert distances[0] is not None
    assert distances[0].distance == expected


def test_fold_descriptor():
    descriptor = fold_builder("mid").build_algorithm_descriptor()
    assert descriptor["boundary_gates"] == [[{"type": "SZGate", "x_obs_bit": 0, "z_obs_bit": 1}], []]
    assert len(descriptor["segments"]) == len(descriptor["boundary_gates"]) + 1 == 3
    assert descriptor["num_frame_slots"] == 2
    assert descriptor["num_observables"] == 1
    assert sum(segment["num_detectors"] for segment in descriptor["segments"]) == 24
    for segment in descriptor["segments"]:
        # ``num_detectors`` counts the segment's own commit detectors; the
        # segment DEM also carries the look-behind and look-ahead halo.
        assert stim.DetectorErrorModel(segment["dem"]).num_detectors == segment["num_window_detectors"]


def test_fold_dagger_descriptor_matches_s():
    """Sign-free Pauli frame updates and DEMs agree for SZ/SZ and SZ/SZdg."""
    dagger = fold_builder("pair_adjacent")
    phase = fold_builder("pair_sz_sz")
    assert dagger.to_tick_circuit().num_measurements() == 41
    dem = stim.DetectorErrorModel(dagger.build_dem())
    assert dem.num_detectors == 32
    assert dem.num_observables == 1
    assert dagger.build_algorithm_descriptor() == phase.build_algorithm_descriptor()


@pytest.mark.parametrize(
    ("dimensions", "message"),
    [
        ({"dx": 3, "dz": 5}, "square"),
        ({"distance": 3, "rotated": False}, "rotated"),
        ({"distance": 1}, "distance at least 2"),
    ],
)
def test_fold_geometry_rejections(dimensions, message):
    builder = LogicalCircuitBuilder()
    builder.add_patch(SurfacePatch.create(**dimensions), "A")
    builder.add_memory("A", 1)
    with pytest.raises(ValueError, match=f"Fold-transversal SZ requires.*{message}"):
        builder.add_logical_sz("A")


@pytest.mark.parametrize("before_preparation", [False, True])
def test_fold_lifetime_rejections(before_preparation):
    builder = LogicalCircuitBuilder()
    builder.add_patch(SurfacePatch.create(3), "A")
    if before_preparation:
        builder.add_logical_sz("A")
        builder.add_memory("A", 1)
    else:
        builder.add_memory("A", 1)
        builder.add_logical_sz("A")
    message = "precedes.*first MEMORY preparation" if before_preparation else "executes after final data measurement"
    with pytest.raises(ValueError, match=message):
        builder.to_tick_circuit()


def test_parity_space_known_samples():
    # The first bit is fixed at one; the other two are correlated random bits.
    assert deterministic_parity_basis([[1, 0, 0], [1, 1, 1]] * 2) == (1, 6)
    assert deterministic_parity_basis([[], []]) == ()
    for shots in ([], [[0], [0, 1]], [[2]]):
        with pytest.raises(ValueError, match="deterministic_parity_basis requires"):
            deterministic_parity_basis(shots)


@pytest.mark.parametrize("physical_s", [False, True])
def test_fold_readout_y_guards(physical_s):
    """A Y term cannot cross physical SZ or close at product Y preparation."""
    builder = LogicalCircuitBuilder()
    builder.add_patch(SurfacePatch.create(3), "A")
    builder.add_memory("A", 1, "X" if physical_s else "Y")
    if physical_s:
        builder.add_transversal_sz("A")
    builder.add_logical_sz("A")
    builder.add_memory("A", 1, "X")
    tc = builder.to_tick_circuit()
    assert json.loads(tc.get_meta("observables")) == []
    for seed in range(8):
        _, fired, observables = simulate_tick_circuit(tc, seed)
        assert fired == 0
        assert observables == {}


@pytest.mark.parametrize(("gate", "fold"), [(LogicalGateType.FOLD_SZ, None), (LogicalGateType.MEMORY, "SZ")])
def test_fold_op_identity_assertion(gate, fold):
    with pytest.raises(AssertionError, match="Fold identity must match FOLD_SZ"):
        LogicalOp(gate, ["A"], rounds=1, fold=fold)


@pytest.mark.parametrize("rounds", [0, 2])
def test_fold_op_round_assertion(rounds):
    with pytest.raises(AssertionError, match="Fold segments require exactly one round"):
        LogicalOp(LogicalGateType.FOLD_SZ, ["A"], rounds=rounds, fold="SZ")


def test_logical_readout_combines_x_z_on_same_patch():
    """The middle CX maps X_A Y_B to Y_A Z_B; the earlier fold maps Y_A to X_A."""
    operations = [
        LogicalOp(LogicalGateType.MEMORY, ["A", "B"], rounds=1, per_patch_basis={"A": "X", "B": "Z"}),
        LogicalOp(LogicalGateType.FOLD_SZ, ["A"], rounds=1, fold="SZ"),
        LogicalOp(LogicalGateType.TRANSVERSAL_CX, ["A", "B"]),
        LogicalOp(LogicalGateType.FOLD_SZ, ["B"], rounds=1, fold="SZ"),
        LogicalOp(LogicalGateType.TRANSVERSAL_CX, ["A", "B"]),
        LogicalOp(LogicalGateType.MEMORY, ["A", "B"], rounds=1, basis="X"),
    ]
    assert _logical_readout_flow(operations, 3, "A", "X") == (True, (("B", 2), ("A", 1)))


@pytest.mark.parametrize("family", ["X", "Z"])
@pytest.mark.parametrize("malformation", ["duplicate", "weight", "interior_axis"])
def test_fold_check_map_bounds(family, malformation):
    patch = SurfacePatch.create(3)
    checks = patch.geometry.x_stabilizers if family == "X" else patch.geometry.z_stabilizers
    if malformation == "duplicate":
        checks[1] = replace(checks[1], data_qubits=checks[0].data_qubits)
        message = f"Fold {family} check centres must be unique"
    elif malformation == "weight":
        checks[0] = replace(checks[0], data_qubits=(0,))
        message = "Fold check weights must be 2 or 4"
    else:
        checks[0] = replace(checks[0], data_qubits=(1, 4) if family == "X" else (3, 4))
        axis = "X" if family == "X" else "Y"
        message = f"Fold boundary {axis} axis must start at a patch edge"
    generator = FoldMapProbe({"A": PatchState(patch, "A")}, [])
    with pytest.raises(AssertionError, match=message):
        generator.fold_maps("A")


@pytest.mark.parametrize("shots", [[[0, 0]], [[0, 0], [1, 1]], [[0]]])
def test_parity_basis_requires_enough_shots(shots):
    with pytest.raises(ValueError, match=r"deterministic_parity_basis requires at least width \+ 1 shots"):
        deterministic_parity_basis(iter(shots))


def test_fold_matching_skips_hyperedges():
    """Pin the documented matching-route limitation without changing the decoder."""
    builders = []
    for fold in (False, True):
        builder = LogicalCircuitBuilder()
        builder.add_patch(SurfacePatch.create(3), "A")
        builder.add_memory("A", 1, "Z")
        if fold:
            builder.add_logical_sz("A")
        else:
            builder.add_memory("A", 1, "Z")
        builder.add_memory("A", 1, "Z")
        builders.append(builder)
    skipped = []
    for builder in builders:
        _, decoder = builder.build_decoder(inner_decoder="pymatching")
        skipped.append(sum(hyperedges for _, hyperedges in decoder.subgraph_diagnostics()))
    assert skipped[0] == 0
    assert skipped[1] == 12


@pytest.mark.parametrize(
    ("shape", "raw_parity"),
    [("pair_sz_sz", 1), ("pair_adjacent", 0), ("pair_sz_sz_h", 1), ("sign_cx", 1)],
)
def test_fold_reference_parity(shape, raw_parity):
    """Raw parity retains the logical sign; Stim reports flips from its reference."""
    tc = fold_builder(shape).to_tick_circuit()
    circuit = stim.Circuit(tick_circuit_to_stim(tc))
    for seed in range(8):
        _, fired, observables = simulate_tick_circuit(tc, seed)
        assert fired == 0
        assert observables == {0: raw_parity}
        samples = circuit.compile_sampler(seed=seed).sample(256)
        events, raw_observables = extract_detection_events_and_observables(tc, samples)
        assert events == [[] for _ in range(256)]
        assert raw_observables == [[0] if raw_parity else [] for _ in range(256)]
        detectors, flips = circuit.compile_detector_sampler(seed=seed).sample(256, separate_observables=True)
        assert not detectors.any()
        assert not flips.any()


@pytest.mark.parametrize("fold", ["", "T", "SZDG"])
def test_fold_op_variant_assertion(fold):
    with pytest.raises(AssertionError, match="Fold variant must be SZ or SZdg"):
        LogicalOp(LogicalGateType.FOLD_SZ, ["A"], rounds=1, fold=fold)


Y_ROUND_PAIRS = [(3, before, after) for before in range(3) for after in range(3)] + [(5, 0, 0), (5, 1, 1)]
EMPTY_COMMIT = (
    r"segment 0 \(patch 'A'\) has an empty commit region: .*; descriptor commit regions need at least one round"
)


def y_builder(patch, before, after, *, explicit=False, swapped=False, shared=False, cx=False):
    """Build the user program or its public-API composition oracle."""
    builder = LogicalCircuitBuilder()
    builder.add_patch(patch, "A")
    if shared:
        builder.add_patch(patch, "B", qubit_offset=patch.geometry.num_qubits)
    labels = ["A", "B"] if shared else "A"
    if before is not None:
        builder.add_memory(labels, before, {"A": "Z" if swapped else "X", "B": "Z"})
        if swapped:
            builder.add_transversal_h("A")
        builder.add_logical_sz("A")
        if cx:
            builder.add_transversal_cx("B", "A")
    if explicit:
        if before is None:
            builder.add_memory("A", 0, "Y")
        builder.add_logical_szdg("A")
    builder.add_memory(labels, after, {"A": "X" if explicit else "Y", "B": "Z"})
    return builder


def assert_y_composition(builder, explicit, *, empty_commit):
    """Compare all compilation products, including repeated calls on one builder."""
    expected = explicit.to_tick_circuit()
    expected_dem = explicit.build_dem()
    if empty_commit:
        with pytest.raises(ValueError, match=EMPTY_COMMIT) as expected_error:
            explicit.build_algorithm_descriptor()
    else:
        expected_descriptor = explicit.build_algorithm_descriptor()
    for _ in range(2):
        actual = builder.to_tick_circuit()
        assert tick_circuit_to_stim(actual) == tick_circuit_to_stim(expected)
        for key in ("detectors", "observables", "num_measurements", "measurement_keys", "teleportation_readouts"):
            assert actual.get_meta(key) == expected.get_meta(key), key
        assert builder.build_dem() == expected_dem
        if empty_commit:
            with pytest.raises(ValueError, match=EMPTY_COMMIT) as actual_error:
                builder.build_algorithm_descriptor()
            assert str(actual_error.value) == str(expected_error.value)
        else:
            assert builder.build_algorithm_descriptor() == expected_descriptor


@pytest.mark.parametrize("orientation", list(PatchOrientation))
@pytest.mark.parametrize("distance", [3, 5])
@pytest.mark.parametrize("rounds", [0, 1, 2])
@pytest.mark.parametrize("first", [False, True])
def test_y_readout_composition(distance, orientation, rounds, first):
    patch = SurfacePatch.create(distance, orientation=orientation)
    before = None if first else 1
    builder = y_builder(patch, before, rounds)
    explicit = y_builder(patch, before, rounds, explicit=True)
    assert_y_composition(builder, explicit, empty_commit=first)


@pytest.mark.parametrize("rounds", [0, 1, 2])
def test_y_product_readout(rounds):
    builder = LogicalCircuitBuilder()
    builder.add_patch(SurfacePatch.create(3), "A")
    builder.add_memory("A", rounds, "Y")
    tc = builder.to_tick_circuit()
    assert json.loads(tc.get_meta("observables")) == []
    assert len(json.loads(tc.get_meta("detectors"))) == 4 + 8 * rounds
    if rounds:
        circuit = stim.Circuit(tick_circuit_to_stim(tc))
        space = deterministic_parity_basis(circuit.compile_sampler(seed=0).sample(2048))
        emitted = [
            _mask(entry["meas_ids"]) for key in ("detectors", "observables") for entry in json.loads(tc.get_meta(key))
        ]
        assert _rank([*space, *emitted]) == len(space)
        assert _rank(emitted) == 4 + 8 * rounds
        assert len(space) == _rank(emitted) + 1 == circuit.count_determined_measurements()
    assert builder.build_dem()
    with pytest.raises(ValueError, match=EMPTY_COMMIT):
        builder.build_algorithm_descriptor()
    for seed in range(8):
        _, fired, observables = simulate_tick_circuit(tc, seed)
        assert fired == 0
        assert observables == {}


@pytest.mark.parametrize(("distance", "before", "after"), Y_ROUND_PAIRS)
def test_y_readout_grid_composition(distance, before, after):
    patch = SurfacePatch.create(distance)
    assert_y_composition(
        y_builder(patch, before, after),
        y_builder(patch, before, after, explicit=True),
        empty_commit=before == 0,
    )


@pytest.mark.parametrize(("distance", "before", "after"), Y_ROUND_PAIRS)
def test_y_readout_support_and_parity(distance, before, after):
    patch = SurfacePatch.create(distance)
    builder = y_builder(patch, before, after)
    tc = builder.to_tick_circuit()
    observables = json.loads(tc.get_meta("observables"))
    assert len(observables) == 1
    keys = json.loads(tc.get_meta("measurement_keys"))
    fold_z = {
        ordinal for _, family, _, segment, _, ordinal in keys["stabilizer"] if family == "Z" and segment in {1, 2}
    }
    final_x = {ordinal for _, qubit, ordinal in keys["data"] if qubit in patch.geometry.logical_x.data_qubits}
    assert set(observables[0]["meas_ids"]) == final_x | fold_z
    assert len(fold_z) == distance**2 - 1
    for seed in range(8):
        _, fired, values = simulate_tick_circuit(tc, seed)
        assert fired == 0
        assert values == {0: 0}


@pytest.mark.parametrize("orientation", list(PatchOrientation))
@pytest.mark.parametrize("distance", [3, 5])
@pytest.mark.parametrize("rounds", [0, 1, 2])
def test_y_shared_first_readout(distance, orientation, rounds):
    patch = SurfacePatch.create(distance, orientation=orientation)
    builder = y_builder(patch, None, rounds, shared=True)
    assert_y_composition(builder, y_builder(patch, None, rounds, shared=True, explicit=True), empty_commit=True)
    tc = builder.to_tick_circuit()
    # A consumes ID 0 even though its product-Y preparation cannot close the walk.
    assert [obs["id"] for obs in json.loads(tc.get_meta("observables"))] == [1]
    for seed in range(8):
        _, fired, values = simulate_tick_circuit(tc, seed)
        assert fired == 0
        assert values == {1: 0}


@pytest.mark.parametrize("orientation", list(PatchOrientation))
@pytest.mark.parametrize("distance", [3, 5])
def test_y_shared_cx_readout(distance, orientation):
    patch = SurfacePatch.create(distance, orientation=orientation)
    builder = y_builder(patch, 1, 1, shared=True, cx=True)
    explicit = y_builder(patch, 1, 1, shared=True, cx=True, explicit=True)
    assert_y_composition(builder, explicit, empty_commit=False)
    tc = builder.to_tick_circuit()
    assert [obs["id"] for obs in json.loads(tc.get_meta("observables"))] == [0, 1]
    for seed in range(8):
        _, fired, values = simulate_tick_circuit(tc, seed)
        assert fired == 0
        assert values == {0: 0, 1: 0}


def test_y_readout_stabilizer_oracle():
    patch = SurfacePatch.create(3)
    tc = y_builder(patch, 1, 1).to_tick_circuit()
    assert tc.num_measurements() == 41
    keys = json.loads(tc.get_meta("measurement_keys"))
    fold_z = [ordinal for _, family, _, segment, _, ordinal in keys["stabilizer"] if family == "Z" and segment == 1]
    ancillas = set(range(patch.geometry.num_data, patch.geometry.num_qubits))
    readouts = [
        tick
        for tick in range(tc.num_ticks())
        if any(g.gate_type.name == "MZ" and ancillas.intersection(g.qubits) for g in tc.get_tick(tick).gate_batches())
    ]
    lx, lz = set(patch.geometry.logical_x.data_qubits), set(patch.geometry.logical_z.data_qubits)
    body = "".join(
        "Y" if q in lx & lz else "X" if q in lx else "Z" if q in lz else "I" for q in range(patch.geometry.num_qubits)
    )
    parities = set()
    for seed in range(8):
        measurements, fired, observables = simulate_tick_circuit(tc, seed)
        parity = sum(measurements[index] for index in fold_z) % 2
        parities.add(parity)
        generators = stabilizer_generators_after(tc, readouts[1] + 1, seed=seed)
        assert group_contains(generators, ("-" if parity else "+") + body)
        assert not group_contains(generators, ("+" if parity else "-") + body)
        assert fired == 0
        assert observables == {0: 0}
    assert parities == {0, 1}


@pytest.mark.parametrize("distance", [3, 5])
@pytest.mark.parametrize("swapped", [False, True])
def test_y_readout_parity_space(distance, swapped):
    tc = y_builder(SurfacePatch.create(distance), 1, 1, swapped=swapped).to_tick_circuit()
    assert len(json.loads(tc.get_meta("observables"))) == 1
    circuit = stim.Circuit(tick_circuit_to_stim(tc))
    space = deterministic_parity_basis(circuit.compile_sampler(seed=0).sample(2048))
    emitted = [
        _mask(entry["meas_ids"]) for key in ("detectors", "observables") for entry in json.loads(tc.get_meta(key))
    ]
    assert _rank([*space, *emitted]) == len(space)
    assert _rank(emitted) == len(space) == circuit.count_determined_measurements()
    for seed in range(8):
        _, fired, values = simulate_tick_circuit(tc, seed)
        assert fired == 0
        assert values == {0: 0}


@pytest.mark.parametrize(("distance", "after", "expected"), [(3, 0, 2), (3, 1, 2), (5, 0, 3), (5, 1, 4)])
def test_y_readout_fault_distance(distance, after, expected):
    tc = y_builder(SurfacePatch.create(distance), 1, after).to_tick_circuit()
    dem = DetectorErrorModel.from_circuit(tc, p1=0.001, p2=0.001, p_meas=0.001, p_prep=0.001)
    distances = dem.per_observable_fault_distances(distance)
    assert len(distances) == 1
    assert distances[0] is not None
    assert distances[0].distance == expected


@pytest.mark.parametrize(
    ("dimensions", "message"),
    [
        ({"dx": 3, "dz": 5}, "Fold-transversal SZ requires a square patch (dx=dz), got dx=3, dz=5"),
        ({"distance": 3, "rotated": False}, "Fold-transversal SZ requires a rotated patch"),
        ({"distance": 1}, "Fold-transversal SZ requires distance at least 2"),
    ],
)
def test_y_readout_geometry_rejections(dimensions, message):
    builder = LogicalCircuitBuilder()
    builder.add_patch(SurfacePatch.create(**dimensions), "A")
    builder.add_memory("A", 1, "Y")
    for build in (builder.to_tick_circuit, builder.build_dem, builder.build_algorithm_descriptor):
        with pytest.raises(ValueError, match=re.escape(message)) as error:
            build()
        assert str(error.value) == f"Y readout on patch 'A' lowers to a fold-transversal SZ: {message}"
        assert isinstance(error.value.__cause__, ValueError)
        assert str(error.value.__cause__) == message


@pytest.mark.parametrize(("distance", "expected_distance"), [(2, 2), (4, 3)])
def test_y_readout_even_distance(distance, expected_distance):
    patch = SurfacePatch.create(distance)
    builder = y_builder(patch, 1, 1)
    assert_y_composition(builder, y_builder(patch, 1, 1, explicit=True), empty_commit=False)
    tc = builder.to_tick_circuit()
    assert len(json.loads(tc.get_meta("observables"))) == 1
    for seed in range(8):
        _, fired, values = simulate_tick_circuit(tc, seed)
        assert fired == 0
        assert values == {0: 0}
    dem = DetectorErrorModel.from_circuit(tc, p1=0.001, p2=0.001, p_meas=0.001, p_prep=0.001)
    distances = dem.per_observable_fault_distances(distance)
    assert len(distances) == 1
    assert distances[0] is not None
    assert distances[0].distance == expected_distance


def test_y_readout_append_after_compile():
    patch = SurfacePatch.create(3)
    builder = y_builder(patch, None, 1)
    builder.to_tick_circuit()
    builder.build_dem()
    builder.add_memory("A", 2, "X")
    explicit = LogicalCircuitBuilder()
    explicit.add_patch(patch, "A")
    explicit.add_memory("A", 1, "Y")
    explicit.add_memory("A", 2, "X")
    assert_y_composition(builder, explicit, empty_commit=False)


class YReadoutProbe(LogicalCircuitBuilder):
    """Expose the lowered operation list to pin serialization and basis normalization."""

    def lowered_operations(self):
        return self._lowered_operations()


@pytest.mark.parametrize("labels", [["A"], ["A", "B"]])
def test_y_shared_serialized_folds(labels):
    patch = SurfacePatch.create(3)
    builder = YReadoutProbe()
    explicit = LogicalCircuitBuilder()
    for index, label in enumerate(labels):
        for program in (builder, explicit):
            program.add_patch(patch, label, qubit_offset=index * patch.geometry.num_qubits)
        explicit.add_memory(label, 0, "Y")
        explicit.add_logical_szdg(label)
    builder.add_memory(labels, 1, "Y")
    explicit.add_memory(labels, 1, "X")
    expected = []
    for label in labels:
        expected.extend(
            [
                LogicalOp(LogicalGateType.MEMORY, [label], rounds=0, basis="Y"),
                LogicalOp(LogicalGateType.FOLD_SZ, [label], rounds=1, fold="SZdg"),
            ],
        )
    expected.append(LogicalOp(LogicalGateType.MEMORY, labels, rounds=1, basis="X"))
    assert builder.lowered_operations() == expected
    assert json.loads(builder.to_tick_circuit().get_meta("observables")) == []
    assert_y_composition(builder, explicit, empty_commit=True)


@pytest.mark.parametrize("shared", [False, True], ids=["h", "mixed"])
def test_y_readout_h_composition(shared):
    patch = SurfacePatch.create(3)
    programs = []
    for explicit in (False, True):
        builder = LogicalCircuitBuilder()
        builder.add_patch(patch, "A")
        if shared:
            builder.add_patch(patch, "B", qubit_offset=patch.geometry.num_qubits)
        labels = ["A", "B"] if shared else "A"
        builder.add_memory(labels, 2, "Z" if shared else "X")
        builder.add_transversal_h("A")
        if explicit:
            builder.add_logical_szdg("A")
        builder.add_memory(labels, 2, {"A": "X" if explicit else "Y", "B": "Z"})
        programs.append(builder)
    assert_y_composition(*programs, empty_commit=False)
    for builder in programs:
        dem = DetectorErrorModel.from_circuit(
            builder.to_tick_circuit(),
            p1=0.001,
            p2=0.001,
            p_meas=0.001,
            p_prep=0.0,
        )
        assert builder.build_dem() == dem.to_string()


@pytest.mark.parametrize("basis", ["X", "Y"])
@pytest.mark.parametrize("buffer", [None, 0, 1, 2])
def test_descriptor_empty_preparation_message(basis, buffer):
    builder = LogicalCircuitBuilder()
    builder.add_patch(SurfacePatch.create(3), "A")
    builder.add_memory("A", 0, basis)
    builder.add_logical_sz("A", dagger=basis == "Y")
    builder.add_memory("A", 2, "X")
    cause = "a zero-round Y preparation before the Y-readout fold" if basis == "Y" else "a zero-round memory segment"
    message = (
        f"segment 0 (patch 'A') has an empty commit region: {cause}; descriptor commit regions need at least one round"
    )
    with pytest.raises(ValueError, match=re.escape(message)) as error:
        builder.build_algorithm_descriptor(buffer=buffer)
    assert str(error.value) == message


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_descriptor_empty_commit_region(basis):
    builder = LogicalCircuitBuilder()
    builder.add_patch(SurfacePatch.create(3), "A")
    builder.add_memory("A", 2, basis)
    builder.add_memory("A", 0, basis)
    builder.add_memory("A", 2, basis)
    message = (
        "segment 1 (patch 'A') has an empty commit region: a zero-round memory segment; "
        "descriptor commit regions need at least one round"
    )
    with pytest.raises(ValueError, match=re.escape(message)) as error:
        builder.build_algorithm_descriptor()
    assert str(error.value) == message


@pytest.mark.parametrize(
    ("shape", "buffer", "counts"),
    [
        ("buffered_zero", 1, [12, 0, 20]),
        ("buffered_zero", 2, [12, 0, 20]),
        ("no_detectors", None, [0, 20]),
        ("no_detectors", 1, [0, 20]),
        ("no_detectors", 2, [0, 20]),
    ],
)
def test_descriptor_non_empty_commit_region_with_zero_detectors(shape, buffer, counts):
    """A non-empty native window is decodable even when its segment owns no detectors."""
    builder = LogicalCircuitBuilder()
    builder.add_patch(SurfacePatch.create(3), "A")
    if shape == "buffered_zero":
        builder.add_memory("A", 2, "Z")
        builder.add_memory("A", 0, "Z")
        builder.add_memory("A", 2, "Z")
    else:
        builder.add_memory("A", 1, "Y")
        builder.add_memory("A", 2, "X")
    descriptor = builder.build_algorithm_descriptor(buffer=buffer)
    assert [segment["num_detectors"] for segment in descriptor["segments"]] == counts
    decoder = LogicalCircuitDecoder(descriptor, budget="unlimited")
    assert decoder.decode([0] * sum(counts)) == 0


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_descriptor_buffer_error_precedes_later_empty_commit_region(basis):
    builder = LogicalCircuitBuilder()
    builder.add_patch(SurfacePatch.create(3), "A")
    builder.add_memory("A", 2, basis)
    builder.add_memory("A", 0, basis)
    builder.add_memory("A", 2, basis)
    message = (
        "buffer=0 is too small for logical segment 0; the source-tracked DEM requires at least 1 look-ahead rounds"
    )
    with pytest.raises(ValueError, match=re.escape(message)) as error:
        builder.build_algorithm_descriptor(buffer=0)
    assert str(error.value) == message


def test_descriptor_validates_buffer_before_lowering():
    builder = LogicalCircuitBuilder()
    builder.add_patch(SurfacePatch.create(dx=3, dz=5), "A")
    builder.add_memory("A", 1, "Y")
    with pytest.raises(ValueError, match="buffer must be non-negative or None"):
        builder.build_algorithm_descriptor(buffer=-1)
