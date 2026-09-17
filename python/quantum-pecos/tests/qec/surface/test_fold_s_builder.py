# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Fold builder parity-space oracles and the known X-sector distance reduction."""

import json
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
from pecos.testing import deterministic_parity_basis, simulate_tick_circuit


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
        builder.add_logical_sdg("A")
        builder.add_transversal_cx("A", "B")
        builder.add_logical_sdg("B")
        builder.add_transversal_cx("A", "B")
        builder.add_logical_sdg("B")
        builder.add_memory(["A", "B"], 1, "X")
    elif shape.startswith("cx_"):
        builder.add_patch(patch, "B", qubit_offset=patch.geometry.num_qubits)
        builder.add_memory(["A", "B"], 2, "Z")
        if shape == "cx_before":
            builder.add_logical_s("A")
        builder.add_transversal_cx("A", "B")
        if shape == "cx_after":
            builder.add_logical_s("A")
        builder.add_memory(["A", "B"], 2, "Z")
    elif shape == "h_fold":
        builder.add_memory("A", 2, "X")
        builder.add_transversal_h("A")
        builder.add_logical_s("A")
        builder.add_memory("A", 2, "Z")
    elif shape.startswith("pair"):
        builder.add_memory("A", 1, "X")
        builder.add_logical_s("A")
        if shape == "pair_separated":
            builder.add_memory("A", 2, "X")
        builder.add_logical_s("A", dagger=shape not in {"pair_s_s", "pair_s_s_h"})
        if shape == "pair_s_s_h":
            builder.add_transversal_h("A")
        builder.add_memory("A", 1, "Z" if shape == "pair_s_s_h" else "X")
    else:
        before, after = {"first": (0, 2), "mid": (1, 1), "last": (2, 0), "single_x": (1, 1)}[shape]
        basis = "X" if shape == "single_x" else "Z"
        builder.add_memory("A", before, basis)
        builder.add_logical_s("A")
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
    """Pin the known reduction to d-1 for adjacent X-prepared S/S-dagger rounds."""
    tc = fold_builder(shape, distance).to_tick_circuit()
    dem = DetectorErrorModel.from_circuit(tc, p1=0.001, p2=0.001, p_meas=0.001, p_prep=0.001)
    distances = dem.per_observable_fault_distances(distance)
    assert len(distances) == 1
    assert distances[0] is not None
    assert distances[0].distance == expected


def test_fold_descriptor():
    descriptor = fold_builder("mid").build_algorithm_descriptor()
    assert descriptor["boundary_gates"] == [[{"type": "SGate", "x_obs_bit": 0, "z_obs_bit": 1}], []]
    assert len(descriptor["segments"]) == len(descriptor["boundary_gates"]) + 1 == 3
    assert descriptor["num_frame_slots"] == 2
    assert descriptor["num_observables"] == 1
    assert sum(segment["num_detectors"] for segment in descriptor["segments"]) == 24
    for segment in descriptor["segments"]:
        # ``num_detectors`` counts the segment's own commit detectors; the
        # segment DEM also carries the look-behind and look-ahead halo.
        assert stim.DetectorErrorModel(segment["dem"]).num_detectors == segment["num_window_detectors"]


def test_fold_dagger_descriptor_matches_s():
    """Sign-free Pauli frame updates and DEMs agree for S/S and S/S-dagger."""
    dagger = fold_builder("pair_adjacent")
    phase = fold_builder("pair_s_s")
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
    with pytest.raises(ValueError, match=f"Fold-transversal S requires.*{message}"):
        builder.add_logical_s("A")


@pytest.mark.parametrize("before_preparation", [False, True])
def test_fold_lifetime_rejections(before_preparation):
    builder = LogicalCircuitBuilder()
    builder.add_patch(SurfacePatch.create(3), "A")
    if before_preparation:
        builder.add_logical_s("A")
        builder.add_memory("A", 1)
    else:
        builder.add_memory("A", 1)
        builder.add_logical_s("A")
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
    """A Y term cannot cross physical S or close at product Y preparation."""
    builder = LogicalCircuitBuilder()
    builder.add_patch(SurfacePatch.create(3), "A")
    builder.add_memory("A", 1, "X" if physical_s else "Y")
    if physical_s:
        builder.add_transversal_sz("A")
    builder.add_logical_s("A")
    builder.add_memory("A", 1, "X")
    tc = builder.to_tick_circuit()
    assert json.loads(tc.get_meta("observables")) == []
    for seed in range(8):
        _, fired, observables = simulate_tick_circuit(tc, seed)
        assert fired == 0
        assert observables == {}


@pytest.mark.parametrize(("gate", "fold"), [(LogicalGateType.FOLD_S, None), (LogicalGateType.MEMORY, "S")])
def test_fold_op_identity_assertion(gate, fold):
    with pytest.raises(AssertionError, match="Fold identity must match FOLD_S"):
        LogicalOp(gate, ["A"], rounds=1, fold=fold)


@pytest.mark.parametrize("rounds", [0, 2])
def test_fold_op_round_assertion(rounds):
    with pytest.raises(AssertionError, match="Fold segments require exactly one round"):
        LogicalOp(LogicalGateType.FOLD_S, ["A"], rounds=rounds, fold="S")


def test_logical_readout_combines_x_z_on_same_patch():
    """The middle CX maps X_A Y_B to Y_A Z_B; the earlier fold maps Y_A to X_A."""
    operations = [
        LogicalOp(LogicalGateType.MEMORY, ["A", "B"], rounds=1, per_patch_basis={"A": "X", "B": "Z"}),
        LogicalOp(LogicalGateType.FOLD_S, ["A"], rounds=1, fold="S"),
        LogicalOp(LogicalGateType.TRANSVERSAL_CX, ["A", "B"]),
        LogicalOp(LogicalGateType.FOLD_S, ["B"], rounds=1, fold="S"),
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
            builder.add_logical_s("A")
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
    [("pair_s_s", 1), ("pair_adjacent", 0), ("pair_s_s_h", 1), ("sign_cx", 1)],
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


@pytest.mark.parametrize("fold", ["", "T", "Sdg"])
def test_fold_op_variant_assertion(fold):
    with pytest.raises(AssertionError, match="Fold variant must be S or SDG"):
        LogicalOp(LogicalGateType.FOLD_S, ["A"], rounds=1, fold=fold)
