# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Fold builder parity-space oracles and the known X-sector distance reduction."""

import json

import pytest
import stim
from pecos.qec import DetectorErrorModel
from pecos.qec.surface import LogicalCircuitBuilder, SurfacePatch
from pecos.qec.surface.circuit_builder import tick_circuit_to_stim
from pecos.testing import deterministic_parity_space


def fold_builder(shape: str, distance: int = 3) -> LogicalCircuitBuilder:
    """Exercise folds at segment edges, in either frame, and around a CX."""
    patch = SurfacePatch.create(distance=distance)
    builder = LogicalCircuitBuilder()
    builder.add_patch(patch, "A")
    if shape.startswith("cx_"):
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
        builder.add_logical_sdg("A")
        builder.add_memory("A", 1, "X")
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
    space = deterministic_parity_space(shots)
    emitted = [
        _mask(entry["meas_ids"]) for key in ("detectors", "observables") for entry in json.loads(tc.get_meta(key))
    ]
    rank = _rank(emitted)
    assert _rank([*space, *emitted]) == len(space), f"{shape}: emitted parity outside deterministic space"
    assert rank == len(space), f"{shape}: dimension={len(space)}, emitted rank={rank}"
    assert len(space) == circuit.count_determined_measurements()
    print(f"RANK {shape} d={distance}: dimension={len(space)}, emitted_rank={rank}")
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
        pytest.param("pair_adjacent", 5, 4, marks=pytest.mark.slow),
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
        assert stim.DetectorErrorModel(segment["dem"]).num_detectors == segment["num_detectors"]


def test_fold_dagger_descriptor_rejected():
    builder = fold_builder("pair_adjacent")
    assert builder.to_tick_circuit().num_measurements() == 41
    assert builder.build_dem()
    message = "Fold S-dagger descriptor unsupported: Rust BoundaryGate has no S-dagger variant"
    with pytest.raises(ValueError, match=message):
        builder.build_algorithm_descriptor()


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
    assert deterministic_parity_space([[1, 0, 0], [1, 1, 1]]) == (1, 6)
    assert deterministic_parity_space([[], []]) == ()
    for shots in ([], [[0], [0, 1]], [[2]]):
        with pytest.raises(ValueError, match="deterministic_parity_space requires"):
            deterministic_parity_space(shots)
