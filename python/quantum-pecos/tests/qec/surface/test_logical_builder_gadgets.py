# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Builder parity and independent physical oracles for surface gadgets."""

import ast
import inspect
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest
import stim
from pecos.guppy_gen.gadget_render import render_gadget_function
from pecos.qec import DetectorErrorModel
from pecos.qec.surface import LogicalCircuitBuilder, SurfacePatch, gadgets, logical_circuit
from pecos.qec.surface.circuit_builder import OpType, SurfaceCircuitStep
from pecos.qec.surface.logical_circuit import (
    LogicalGateType,
    LogicalOp,
    _CircuitGenerator,
    _conjugate_stabilizer_term,
    _logical_readout_is_deterministic,
    _propagate_stabilizer_terms,
)
from pecos.qec.surface.patch import PatchOrientation
from pecos.testing import group_contains, simulate_tick_circuit, stabilizer_generators_after
from pecos_rslib.quantum import TickCircuit

GOLDENS = Path(__file__).parent / "goldens" / "logical_builder"
SHAPES = (
    "dx3dz1_mem_Z",
    "dx1dz3_mem_Z",
    "dx1dz3_mem_X",
    "d3_mem_Z",
    "d3_mem_X",
    "d3_h_z_to_x",
    "d3_h_x_to_z",
    "d3_hh",
    "d3_cx_zz",
    "d3_cx_xx",
    "d3_cx_xz",
    "d3_cx_zx",
    "d3_h_cx_h",
    "d3_sz_teleport_first_op",
    "d3_sz_teleport_memory_first",
    "d3_t_inject_first_op",
    "d3_t_inject_data_memory_first",
    "d5_mem_Z",
    "d5_h",
    "d5_cx_zz",
    "d3_hh_adjacent",
    "d3_cxcx",
    "d3_cx_chain",
    "d3_skip_segment",
    "d3_late_partner_cx",
    "d2_h_even",
)

EXPECTED_OBSERVABLE_COUNTS = {
    "d3_cx_xx": 2,
    "d3_cx_xz": 0,
    "d3_cx_zx": 2,
    "d3_cx_zz": 2,
    "d3_h_cx_h": 2,
    "d3_h_x_to_z": 1,
    "d3_h_z_to_x": 1,
    "d3_hh": 1,
    "d3_mem_X": 1,
    "d3_mem_Z": 1,
    "d3_sz_teleport_first_op": 1,
    "d3_sz_teleport_memory_first": 1,
    "d3_t_inject_data_memory_first": 1,
    "d3_t_inject_first_op": 1,
    "d5_cx_zz": 2,
    "d5_h": 1,
    "d5_mem_Z": 1,
    "dx1dz3_mem_X": 1,
    "dx1dz3_mem_Z": 1,
    "dx3dz1_mem_Z": 1,
    "d3_hh_adjacent": 1,
    "d3_cxcx": 2,
    "d3_cx_chain": 3,
    "d3_skip_segment": 2,
    "d3_late_partner_cx": 2,
    "d2_h_even": 1,
}


assert {path.name for path in GOLDENS.iterdir()} == {
    f"{shape}.{suffix}" for shape in SHAPES for suffix in ("stim", "noisy.stim", "tickmeta.json")
}
assert set(EXPECTED_OBSERVABLE_COUNTS) == set(SHAPES)


class BuilderProbe(LogicalCircuitBuilder):
    """Expose queued operations and patch state for registration assertions."""

    @property
    def operations(self):
        return self._operations

    @property
    def patches(self):
        return self._patches


class GeneratorProbe(_CircuitGenerator):
    """Exercise the lowering boundary and inspect its measurement analysis."""

    def emit_steps(self, step_lists):
        return self._emit_steps(step_lists)

    def ancilla_spatial_coords(self, label, family, index):
        return self._ancilla_spatial_coords(label, family, index)

    def allocation(self, label):
        return self._allocation(label)

    def boundary_detector(self, key, measurement):
        return self._emit_boundary_detector(*key, measurement)

    def round_detectors(self, label, round_index):
        return self._emit_round_detectors(label, round_index, is_first_segment=False)

    @property
    def round_measurements(self):
        return self._stab_meas_by_round

    @property
    def detectors(self):
        return self._det_json


def make_builder(name: str) -> LogicalCircuitBuilder:
    """Reproduce the orchestrator's captured recipes exactly."""
    if name in {"dx3dz1_mem_Z", "dx1dz3_mem_Z", "dx1dz3_mem_X"}:
        dx, dz = (3, 1) if name == "dx3dz1_mem_Z" else (1, 3)
        builder = BuilderProbe()
        builder.add_patch(SurfacePatch.create(dx=dx, dz=dz), "A")
        builder.add_memory("A", 2, name[-1])
        return builder
    patch = SurfacePatch.create(distance=int(name[1]))
    shape = name[3:]
    if shape.startswith("sz_"):
        labels = ["D", "Y"]
    elif shape.startswith("t_"):
        labels = ["D", "A"]
    elif shape == "cx_chain":
        labels = ["A", "B", "C"]
    elif shape in {"skip_segment", "late_partner_cx"}:
        labels = ["A", "B"]
    elif shape.startswith("cx_") or shape == "cxcx":
        labels = ["C", "T"]
    elif shape == "h_cx_h":
        labels = ["A", "B"]
    else:
        labels = ["A"]
    builder = BuilderProbe()
    for i, label in enumerate(labels):
        builder.add_patch(patch, label, qubit_offset=i * (patch.geometry.num_data + patch.geometry.num_ancilla))
    if shape.startswith("mem_"):
        builder.add_memory("A", 3 if name.startswith("d5") else 2, shape[-1])
    elif shape in {"h", "h_z_to_x", "h_x_to_z", "hh"}:
        before, after = ("X", "Z") if shape == "h_x_to_z" else ("Z", "X")
        builder.add_memory("A", 2, before)
        builder.add_transversal_h("A")
        builder.add_memory("A", 2, after)
        if shape == "hh":
            builder.add_transversal_h("A")
            builder.add_memory("A", 2, "Z")
    elif shape in {"hh_adjacent", "h_even"}:
        builder.add_memory("A", 2, "Z")
        builder.add_transversal_h("A")
        if shape == "hh_adjacent":
            builder.add_transversal_h("A")
        builder.add_memory("A", 2, "Z" if shape == "hh_adjacent" else "X")
    elif shape == "cxcx":
        builder.add_memory(labels, 2, "Z")
        builder.add_transversal_cx("C", "T")
        builder.add_transversal_cx("C", "T")
        builder.add_memory(labels, 2, "Z")
    elif shape == "cx_chain":
        builder.add_memory(labels, 2, "X")
        builder.add_transversal_cx("A", "B")
        builder.add_transversal_cx("B", "C")
        builder.add_memory(labels, 2, "X")
    elif shape == "skip_segment":
        for label in ["A", "B", "A", "B"]:
            builder.add_memory(label, 2, "Z")
    elif shape == "late_partner_cx":
        builder.add_memory("B", 2, "Z")
        builder.add_memory("A", 2, "Z")
        builder.add_transversal_cx("A", "B")
        builder.add_memory(labels, 2, "Z")
    elif shape.startswith("cx_"):
        basis = {"C": shape[-2].upper(), "T": shape[-1].upper()}
        builder.add_memory(labels, 2, basis)
        builder.add_transversal_cx("C", "T")
        builder.add_memory(labels, 2, basis)
    elif shape == "h_cx_h":
        builder.add_memory(labels, 2, "Z")
        builder.add_transversal_h("A")
        builder.add_memory(labels, 2, {"A": "X", "B": "Z"})
        builder.add_transversal_h("A")
        builder.add_memory(labels, 2, "Z")
        builder.add_transversal_cx("A", "B")
        builder.add_memory(labels, 2, "Z")
    else:
        if "memory_first" in shape:
            builder.add_memory("D", 2, "Z")
        if shape.startswith("sz_"):
            builder.add_sz_via_teleportation("D", "Y", 2, 2)
            builder.add_memory("D", 2, "Z")
        else:
            builder.add_t_via_injection("D", "A", 2, 2)
    return builder


def golden_outputs(builder: LogicalCircuitBuilder) -> dict[str, str]:
    """Serialize raw metadata, including nested strings and nulls."""
    tc = builder.to_tick_circuit()
    return {
        "stim": builder.to_stim(),
        "noisy.stim": builder.to_stim(p1=0.001, p2=0.01, p_meas=0.005, p_prep=0.002),
        "tickmeta.json": json.dumps(
            {k: tc.get_meta(k) for k in ("detectors", "observables", "num_measurements", "num_detectors", "basis")}
            | {"num_ticks": tc.num_ticks()},
            indent=1,
        ),
    }


@pytest.mark.parametrize("shape", SHAPES)
def test_golden_parity(shape):
    for suffix, actual in golden_outputs(make_builder(shape)).items():
        assert actual.encode() == (GOLDENS / f"{shape}.{suffix}").read_bytes(), suffix


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("seed", range(8))
def test_noiseless_determinism(shape, seed):
    assert simulate_tick_circuit(make_builder(shape).to_tick_circuit(), seed)[1] == 0


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("seed", range(8))
def test_noiseless_observables(shape, seed):
    observables = simulate_tick_circuit(make_builder(shape).to_tick_circuit(), seed)[2]
    assert len(observables) == EXPECTED_OBSERVABLE_COUNTS[shape]
    assert all(value == 0 for value in observables.values())


@pytest.mark.parametrize("shape", SHAPES)
def test_gadget_dependencies(shape, monkeypatch):
    calls = []
    names = (
        "prep_gadget",
        "syndrome_round_gadget",
        "measure_out_gadget",
        "transversal_layer_gadget",
        "transversal_cx_gadget",
    )
    for name in names:
        original = getattr(gadgets, name)

        def record(*args, _name=name, _original=original, **kwargs):
            calls.append((_name, tuple(args[1].data_qubits), kwargs))
            return _original(*args, **kwargs)

        monkeypatch.setattr(gadgets, name, record)
    builder = make_builder(shape)
    builder.to_tick_circuit()
    expected = Counter()
    prepared = set()
    for op in builder.operations:
        if op.gate_type == LogicalGateType.MEMORY:
            for label in op.patches:
                allocation = tuple(GeneratorProbe(builder.patches, []).allocation(label).data_qubits)
                if label not in prepared:
                    expected["prep_gadget", allocation] += 1
                    expected["measure_out_gadget", allocation] += 1
                    prepared.add(label)
                expected["syndrome_round_gadget", allocation] += op.rounds
        else:
            allocation = tuple(GeneratorProbe(builder.patches, []).allocation(op.patches[0]).data_qubits)
            name = (
                "transversal_cx_gadget"
                if op.gate_type == LogicalGateType.TRANSVERSAL_CX
                else "transversal_layer_gadget"
            )
            expected[name, allocation] += 1
    assert Counter((name, allocation) for name, allocation, _ in calls) == expected
    for label in prepared:
        allocation = tuple(GeneratorProbe(builder.patches, []).allocation(label).data_qubits)
        rounds = [
            kw["round_index"] for name, data, kw in calls if name == "syndrome_round_gadget" and data == allocation
        ]
        assert rounds == [
            r
            for op in builder.operations
            if op.gate_type == LogicalGateType.MEMORY and label in op.patches
            for r in range(op.rounds)
        ]


def test_no_manual_gate_emission():
    module = ast.parse(inspect.getsource(logical_circuit))
    for function in (node for node in ast.walk(module) if isinstance(node, ast.FunctionDef)):
        if function.name in {"_emit_steps", "_emit_qalloc_or_reset"}:
            continue
        for node in ast.walk(function):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert not (
                    isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "t"
                    and node.func.attr in {"cx", "h", "sz", "szdg"}
                )


def _pauli(n, *terms):
    body = ["I"] * n
    for letter, qubits in terms:
        for q in qubits:
            body[q] = letter
    return "+" + "".join(body)


def _data_gate_tick(tc, name, data):
    return next(
        i
        for i in range(tc.num_ticks())
        for g in tc.get_tick(i).gate_batches()
        if g.gate_type.name == name and set(g.qubits) == set(data)
    )


def test_y_preparation_stabilizers():
    patch = SurfacePatch.create(3)
    generator = GeneratorProbe({}, [])
    allocation = gadgets.default_allocation(patch)
    generator.emit_steps([gadgets.prep_gadget(patch, allocation, basis="Y").steps])
    group = stabilizer_generators_after(generator.tc, generator.tc.num_ticks())
    for q in allocation.data_qubits:
        assert group_contains(group, _pauli(patch.geometry.num_data, ("Y", [q])))
        assert not group_contains(group, "-" + _pauli(patch.geometry.num_data, ("Y", [q]))[1:])


def test_h_stabilizer_oracle():
    builder = make_builder("d3_h_z_to_x")
    tc = builder.to_tick_circuit()
    patch = builder.patches["A"].patch
    tick = _data_gate_tick(tc, "H", range(patch.geometry.num_data))
    group = stabilizer_generators_after(tc, tick + 1)
    for family, stabs in (("Z", patch.geometry.x_stabilizers), ("X", patch.geometry.z_stabilizers)):
        for stab in stabs:
            pauli = _pauli(patch.geometry.num_qubits, (family, stab.data_qubits))
            # Syndrome signs are random; compare the measured sign before H.
            before = stabilizer_generators_after(tc, tick)
            original = _pauli(patch.geometry.num_qubits, ("X" if family == "Z" else "Z", stab.data_qubits))
            sign = "+" if group_contains(before, original) else "-"
            assert group_contains(before, sign + original[1:])
            assert group_contains(group, sign + pauli[1:])


def test_cx_stabilizer_oracle():
    builder = make_builder("d3_cx_xz")
    tc = builder.to_tick_circuit()
    patch = builder.patches["C"].patch
    nq = patch.geometry.num_qubits
    tick = _data_gate_tick(tc, "CX", [*range(9), *range(nq, nq + 9)])
    before = stabilizer_generators_after(tc, tick)
    after = stabilizer_generators_after(tc, tick + 1)
    lx, lz = patch.geometry.logical_x.data_qubits, patch.geometry.logical_z.data_qubits
    assert group_contains(before, _pauli(2 * nq, ("X", lx)))
    assert group_contains(before, _pauli(2 * nq, ("Z", [nq + q for q in lz])))
    assert group_contains(after, _pauli(2 * nq, ("X", [*lx, *(nq + q for q in lx)])))
    assert group_contains(after, _pauli(2 * nq, ("Z", [*lz, *(nq + q for q in lz)])))


@pytest.mark.parametrize(
    ("shape", "num_observables", "distance"),
    [
        ("d3_h_z_to_x", 1, 3),
        ("d3_cx_zz", 2, 3),
        ("d3_hh_adjacent", 1, 3),
        ("d3_cxcx", 2, 3),
        ("d3_cx_chain", 3, 3),
        ("d3_skip_segment", 2, 3),
        ("d3_late_partner_cx", 2, 3),
        ("d2_h_even", 1, 2),
    ],
)
def test_fault_distance(shape, num_observables, distance):
    dem = DetectorErrorModel.from_circuit(
        make_builder(shape).to_tick_circuit(),
        p1=0.001,
        p2=0.001,
        p_meas=0.001,
        p_prep=0.001,
    )
    distances = dem.per_observable_fault_distances(distance)
    assert len(distances) == num_observables
    assert all(result is not None and result.distance == distance for result in distances)


@pytest.mark.parametrize(("shape", "copies", "expected"), [("d3_skip_segment", 2, 64), ("d3_hh_adjacent", 1, 32)])
def test_boundary_detector_counts(shape, copies, expected):
    memory = make_builder("d3_mem_Z")
    memory.operations[0].rounds = 4
    memory_count = len(json.loads(memory.to_tick_circuit().get_meta("detectors")))
    actual = len(json.loads(make_builder(shape).to_tick_circuit().get_meta("detectors")))
    assert actual == copies * memory_count == expected


def test_post_h_physical_detector_coordinates():
    builder = make_builder("d3_h_z_to_x")
    generator = GeneratorProbe(builder.patches, builder.operations)
    generator.generate()
    state = builder.patches["A"]
    geometry = state.patch.geometry
    measured_keys = {measurement: key for key, measurement in generator.stab_meas.items()}
    post_h = [detector for detector in generator.detectors if detector["coords"][2] >= 2]
    assert len(post_h) == 20
    for detector in post_h:
        records = detector["abs_records"]
        # Round comparisons put the new syndrome first; readout puts it last.
        measurement = records[0] if detector["coords"][2] < 4 else records[-1]
        label, family, index, segment, _ = measured_keys[measurement]
        assert (label, segment) == ("A", 1)
        base_register = geometry.z_stabilizers if family == "X" else geometry.x_stabilizers
        stabilizer = next(stab for stab in base_register if stab.index == index)
        positions = [geometry.id_to_pos[q] for q in stabilizer.data_qubits]
        expected = [
            2 * sum(col for row, col in positions) / len(positions) + state.coord_offset[0],
            2 * sum(row for row, col in positions) / len(positions) + state.coord_offset[1],
        ]
        assert detector["coords"][:2] == expected


def test_even_distance_post_h_comparisons():
    builder = make_builder("d2_h_even")
    geometry = builder.patches["A"].patch.geometry
    assert (len(geometry.x_stabilizers), len(geometry.z_stabilizers)) == (2, 1)
    detectors = json.loads(builder.to_tick_circuit().get_meta("detectors"))
    for time in (2, 3):
        comparisons = [detector for detector in detectors if detector["coords"][2] == time]
        assert len(comparisons) == 3
        assert all(len(detector["meas_ids"]) == 2 for detector in comparisons)


@pytest.mark.parametrize("family", ["X", "Z"])
def test_propagation_hh(family):
    operations = [
        LogicalOp(LogicalGateType.MEMORY, ["A"], rounds=2),
        LogicalOp(LogicalGateType.TRANSVERSAL_H, ["A"]),
        LogicalOp(LogicalGateType.TRANSVERSAL_H, ["A"]),
        LogicalOp(LogicalGateType.MEMORY, ["A"], rounds=2),
    ]
    assert _propagate_stabilizer_terms(operations, 1, ("A", family, 7, family)) == [("A", family, 7, 0)]


@pytest.mark.parametrize("patch", ["C", "T"])
@pytest.mark.parametrize("family", ["X", "Z"])
def test_propagation_cxcx(patch, family):
    operations = [
        LogicalOp(LogicalGateType.MEMORY, ["C", "T"], rounds=2),
        LogicalOp(LogicalGateType.TRANSVERSAL_CX, ["C", "T"]),
        LogicalOp(LogicalGateType.TRANSVERSAL_CX, ["C", "T"]),
        LogicalOp(LogicalGateType.MEMORY, ["C", "T"], rounds=2),
    ]
    assert _propagate_stabilizer_terms(operations, 1, (patch, family, 7, family)) == [(patch, family, 7, 0)]


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        (("A", "X", 7, "X"), [("A", "X", 7, 0), ("B", "X", 7, 0)]),
        (("C", "Z", 7, "Z"), [("C", "Z", 7, 0), ("B", "Z", 7, 0), ("A", "Z", 7, 0)]),
    ],
)
def test_propagation_cx_chain(key, expected):
    operations = [
        LogicalOp(LogicalGateType.MEMORY, ["A", "B", "C"], rounds=2),
        LogicalOp(LogicalGateType.TRANSVERSAL_CX, ["A", "B"]),
        LogicalOp(LogicalGateType.TRANSVERSAL_CX, ["B", "C"]),
        LogicalOp(LogicalGateType.MEMORY, ["A", "B", "C"], rounds=2),
    ]
    assert _propagate_stabilizer_terms(operations, 1, key) == expected


@pytest.mark.parametrize("rounds", [0, 2])
def test_propagation_skipped_segment(rounds):
    operations = [
        LogicalOp(LogicalGateType.MEMORY, ["A"], rounds=2),
        LogicalOp(LogicalGateType.MEMORY, ["A"] if rounds == 0 else ["B"], rounds=rounds),
        LogicalOp(LogicalGateType.MEMORY, ["A"], rounds=2),
    ]
    assert _propagate_stabilizer_terms(operations, 2, ("A", "X", 7, "X")) == [("A", "X", 7, 0)]


def test_propagation_resolves_in_memory_order():
    operations = [
        LogicalOp(LogicalGateType.MEMORY, ["A"], rounds=2),
        LogicalOp(LogicalGateType.MEMORY, ["B"], rounds=2),
        LogicalOp(LogicalGateType.TRANSVERSAL_CX, ["A", "B"]),
        LogicalOp(LogicalGateType.MEMORY, ["A", "B"], rounds=2),
    ]
    assert _propagate_stabilizer_terms(operations, 2, ("A", "X", 7, "X")) == [
        ("B", "X", 7, 1),
        ("A", "X", 7, 0),
    ]


@pytest.mark.parametrize("gate", [LogicalGateType.TRANSVERSAL_SZ, LogicalGateType.TRANSVERSAL_SZdg])
@pytest.mark.parametrize("family", ["X", "Z"])
@pytest.mark.parametrize("gate_patch", ["A", "B"])
def test_propagation_physical_sz(gate, family, gate_patch):
    operations = [
        LogicalOp(LogicalGateType.MEMORY, ["A"], rounds=2),
        LogicalOp(gate, [gate_patch]),
        LogicalOp(LogicalGateType.MEMORY, ["A"], rounds=2),
    ]
    expected = None if family == "X" and gate_patch == "A" else [("A", family, 7, 0)]
    assert _propagate_stabilizer_terms(operations, 1, ("A", family, 7, family)) == expected


@pytest.mark.parametrize(
    ("gate", "pauli", "mapped_pauli"),
    [
        (LogicalGateType.TRANSVERSAL_SZ, "X", "Y"),
        (LogicalGateType.TRANSVERSAL_SZ, "Y", "X"),
        (LogicalGateType.TRANSVERSAL_SZ, "Z", "Z"),
        (LogicalGateType.TRANSVERSAL_SZdg, "X", "Y"),
        (LogicalGateType.TRANSVERSAL_SZdg, "Y", "X"),
        (LogicalGateType.TRANSVERSAL_SZdg, "Z", "Z"),
        (LogicalGateType.TRANSVERSAL_H, "X", "Z"),
        (LogicalGateType.TRANSVERSAL_H, "Y", "Y"),
        (LogicalGateType.TRANSVERSAL_H, "Z", "X"),
    ],
)
@pytest.mark.parametrize("base_family", ["X", "Z"])
def test_check_clifford_images(gate, pauli, mapped_pauli, base_family):
    op = LogicalOp(gate, ["A"])
    assert _conjugate_stabilizer_term(op, ("A", base_family, 7, pauli)) == [
        ("A", base_family, 7, mapped_pauli),
    ]
    assert _conjugate_stabilizer_term(op, ("B", base_family, 7, pauli)) == [("B", base_family, 7, pauli)]


@pytest.mark.parametrize("patch", ["A", "B", "C"])
@pytest.mark.parametrize("base_family", ["X", "Z"])
def test_y_term_at_cx(patch, base_family):
    op = LogicalOp(LogicalGateType.TRANSVERSAL_CX, ["A", "B"])
    expected = [(patch, base_family, 7, "Y")]
    if patch == "A":
        expected.append(("B", base_family, 7, "X"))
    elif patch == "B":
        expected.append(("A", base_family, 7, "Z"))
    assert _conjugate_stabilizer_term(op, (patch, base_family, 7, "Y")) == expected


@pytest.mark.parametrize("base_family", ["X", "Z"])
@pytest.mark.parametrize(("patch", "pauli", "partner"), [("A", "X", "B"), ("B", "Z", "A")])
def test_cx_partner_retains_support(base_family, patch, pauli, partner):
    op = LogicalOp(LogicalGateType.TRANSVERSAL_CX, ["A", "B"])
    assert _conjugate_stabilizer_term(op, (patch, base_family, 7, pauli)) == [
        (patch, base_family, 7, pauli),
        (partner, base_family, 7, pauli),
    ]


@pytest.mark.parametrize("base_family", ["X", "Z"])
@pytest.mark.parametrize("pauli", ["X", "Y", "Z"])
@pytest.mark.parametrize("swapped", [False, True])
def test_propagation_memory_matches_register_type(base_family, pauli, swapped):
    operations = [LogicalOp(LogicalGateType.MEMORY, ["A", "B"], rounds=0)]
    if swapped:
        operations.append(LogicalOp(LogicalGateType.TRANSVERSAL_H, ["A"]))
    operations.extend(
        [
            LogicalOp(LogicalGateType.MEMORY, ["A", "B"], rounds=2),
            LogicalOp(LogicalGateType.TRANSVERSAL_H, ["B"]),
            LogicalOp(LogicalGateType.MEMORY, ["A", "B"], rounds=2),
        ],
    )
    measured_type = ("Z" if base_family == "X" else "X") if swapped else base_family
    expected = [("A", pauli, 7, 1)] if pauli == measured_type else None
    assert _propagate_stabilizer_terms(operations, 2, ("A", base_family, 7, pauli)) == expected


@pytest.mark.parametrize("base_family", ["X", "Z"])
def test_propagation_sz_h_sz_does_not_relabel_support(base_family):
    operations = [
        LogicalOp(LogicalGateType.MEMORY, ["A"], rounds=2, basis="X"),
        LogicalOp(LogicalGateType.TRANSVERSAL_SZ, ["A"]),
        LogicalOp(LogicalGateType.TRANSVERSAL_H, ["A"]),
        LogicalOp(LogicalGateType.TRANSVERSAL_SZ, ["A"]),
        LogicalOp(LogicalGateType.MEMORY, ["A"], rounds=2, basis="X"),
    ]
    pauli = "Z" if base_family == "X" else "X"
    assert _propagate_stabilizer_terms(operations, 1, ("A", base_family, 7, pauli)) is None


@pytest.mark.parametrize("family", ["X", "Z"])
@pytest.mark.parametrize("inverse", [False, True])
def test_propagation_sz_cancellation(family, inverse):
    operations = [
        LogicalOp(LogicalGateType.MEMORY, ["A"], rounds=2, basis="X"),
        LogicalOp(LogicalGateType.TRANSVERSAL_SZ, ["A"]),
        LogicalOp(LogicalGateType.TRANSVERSAL_SZdg if inverse else LogicalGateType.TRANSVERSAL_SZ, ["A"]),
        LogicalOp(LogicalGateType.MEMORY, ["A"], rounds=2, basis="X"),
    ]
    assert _propagate_stabilizer_terms(operations, 1, ("A", family, 7, family)) == [("A", family, 7, 0)]


@pytest.mark.parametrize("inverse", [False, True])
def test_xor_cancellation_after_sz_pair(inverse):
    operations = [
        LogicalOp(LogicalGateType.MEMORY, ["A", "B"], rounds=2, basis="X"),
        LogicalOp(LogicalGateType.TRANSVERSAL_CX, ["A", "B"]),
        LogicalOp(LogicalGateType.TRANSVERSAL_SZ, ["B"]),
        LogicalOp(LogicalGateType.TRANSVERSAL_SZdg if inverse else LogicalGateType.TRANSVERSAL_SZ, ["B"]),
        LogicalOp(LogicalGateType.TRANSVERSAL_CX, ["A", "B"]),
        LogicalOp(LogicalGateType.MEMORY, ["A", "B"], rounds=2, basis="X"),
    ]
    assert _propagate_stabilizer_terms(operations, 1, ("A", "X", 7, "X")) == [("A", "X", 7, 0)]


@pytest.mark.parametrize("patch", ["A", "B"])
@pytest.mark.parametrize("base_family", ["X", "Z"])
def test_y_cxcx_cancellation_at_preparation(patch, base_family):
    operations = [
        LogicalOp(LogicalGateType.MEMORY, ["A", "B"], rounds=0, basis="Y"),
        LogicalOp(LogicalGateType.TRANSVERSAL_CX, ["A", "B"]),
        LogicalOp(LogicalGateType.TRANSVERSAL_CX, ["A", "B"]),
        LogicalOp(LogicalGateType.MEMORY, ["A", "B"], rounds=2),
    ]
    assert _propagate_stabilizer_terms(operations, 1, (patch, base_family, 7, "Y")) == []


def _y_cxcx_builder():
    builder = BuilderProbe()
    for label, offset in [("A", 0), ("B", 17)]:
        builder.add_patch(SurfacePatch.create(3), label, qubit_offset=offset)
    builder.add_memory(["A", "B"], 2, "Z")
    builder.add_transversal_sz("A")
    builder.add_transversal_cx("A", "B")
    builder.add_transversal_cx("A", "B")
    builder.add_transversal_szdg("A")
    builder.add_memory(["A", "B"], 2, "Z")
    return builder


@pytest.mark.parametrize("seed", range(8))
def test_y_cxcx_boundary_detectors(seed):
    tc = _y_cxcx_builder().to_tick_circuit()
    records = [detector["meas_ids"] for detector in json.loads(tc.get_meta("detectors"))]
    assert len(records) == 64
    assert all([32 + index, 16 + index] in records for index in range(4))
    assert simulate_tick_circuit(tc, seed)[1] == 0


def test_y_cxcx_fault_distance():
    dem = DetectorErrorModel.from_circuit(
        _y_cxcx_builder().to_tick_circuit(),
        p1=0.001,
        p2=0.001,
        p_meas=0.001,
        p_prep=0.001,
    )
    distances = dem.per_observable_fault_distances(3)
    assert len(distances) == 2
    assert all(result is not None and result.distance == 3 for result in distances)


@pytest.mark.parametrize("seed", range(8))
def test_zero_round_per_patch_preparation_detectors(seed):
    builder = BuilderProbe()
    for label, offset in [("A", 0), ("B", 17)]:
        builder.add_patch(SurfacePatch.create(3), label, qubit_offset=offset)
    builder.add_memory(["A", "B"], 0, {"A": "Z", "B": "X"})
    builder.add_transversal_cx("A", "B")
    builder.add_memory(["A", "B"], 2, "Z")
    tc = builder.to_tick_circuit()
    detectors = json.loads(tc.get_meta("detectors"))
    assert len(detectors) == 32
    first_round = [detector["meas_ids"] for detector in detectors if detector["coords"][2] == 0]
    # A-Z occupies records 4..7 and B-X records 8..11; these are the only
    # deterministic first-round families. B-Z (12..15) has no singletons.
    assert first_round == [[index] for index in range(4, 12)]
    assert simulate_tick_circuit(tc, seed)[1] == 0


def _sz_builder(shape):
    builder = BuilderProbe()
    builder.add_patch(SurfacePatch.create(3), "A")
    if shape in {"sz_szdg", "sz_sz", "sz_h_sz"}:
        builder.add_memory("A", 2, "X")
        builder.add_transversal_sz("A")
        if shape == "sz_h_sz":
            builder.add_transversal_h("A")
        getattr(builder, "add_transversal_szdg" if shape == "sz_szdg" else "add_transversal_sz")("A")
    else:
        builder.add_memory("A", 0, "Y")
        getattr(builder, "add_transversal_sz" if shape == "y_sz" else "add_transversal_szdg")("A")
    builder.add_memory("A", 2, "X")
    return builder


@pytest.mark.parametrize(
    ("shape", "count"),
    [("y_sz", 16), ("y_szdg", 16), ("sz_szdg", 32), ("sz_sz", 32), ("sz_h_sz", 24)],
)
@pytest.mark.parametrize("seed", range(8))
def test_sz_detectors(shape, count, seed):
    tc = _sz_builder(shape).to_tick_circuit()
    detectors = json.loads(tc.get_meta("detectors"))
    assert len(detectors) == count
    records = [detector["meas_ids"] for detector in detectors]
    for index in range(4):
        if shape in {"sz_szdg", "sz_sz"}:
            assert [16 + index, 8 + index] in records
            assert [20 + index, 12 + index] in records
        elif shape == "sz_h_sz":
            assert all(detector[0] not in range(16, 24) for detector in records)
        else:
            assert [index] in records
    assert simulate_tick_circuit(tc, seed)[1] == 0


def test_even_weight_y_sz_singletons_noiseless_1024_shots():
    tc = _sz_builder("y_sz").to_tick_circuit()
    records = [detector["meas_ids"] for detector in json.loads(tc.get_meta("detectors"))]
    assert len(records) == 16
    assert all([index] in records for index in range(4))
    for seed in range(1024):
        assert simulate_tick_circuit(tc, seed)[1] == 0, seed


@pytest.mark.parametrize("family", ["X", "Z"])
@pytest.mark.parametrize("output", ["to_tick_circuit", "to_dag_circuit", "to_stim"])
def test_odd_weight_check_rejected_at_generation(family, output):
    builder = make_builder("d3_mem_Z")
    geometry = builder.patches["A"].patch.geometry
    stabs = geometry.x_stabilizers if family == "X" else geometry.z_stabilizers
    original = stabs[0]
    stabs[0] = replace(original, data_qubits=original.data_qubits[:-1])
    with pytest.raises(ValueError, match=rf"A.*{family}.*{original.index}.*odd weight"):
        getattr(builder, output)()


@pytest.mark.parametrize("shape", ["y_sz", "y_szdg", "sz_sz", "sz_szdg"])
def test_sz_layers_rule_out_logical_readout(shape):
    """A physical S layer is not a logical gate, so no X readout crosses it.

    The logical-readout walk conservatively emits no observable for these
    shapes even where the layers cancel; the fold-transversal S replaces the
    layer and its readout rule together. Detectors are still built and are
    deterministic (see the noiseless tests for the same shapes).
    """
    tc = _sz_builder(shape).to_tick_circuit()
    assert json.loads(tc.get_meta("observables")) == []
    dem = DetectorErrorModel.from_circuit(tc, p1=0.001, p2=0.001, p_meas=0.001, p_prep=0.001)
    assert dem.per_observable_fault_distances(3) == []


@pytest.mark.parametrize(("rounds", "count"), [(0, 16), (2, 24)])
@pytest.mark.parametrize("seed", range(8))
def test_y_preparation_syndrome_barrier(rounds, count, seed):
    builder = BuilderProbe()
    builder.add_patch(SurfacePatch.create(3), "A")
    builder.add_memory("A", rounds, "Y")
    builder.add_transversal_szdg("A")
    builder.add_memory("A", 2, "X")
    tc = builder.to_tick_circuit()
    records = [detector["meas_ids"] for detector in json.loads(tc.get_meta("detectors"))]
    assert len(records) == count
    for index in range(4):
        current = 8 * rounds + index
        if rounds == 0:
            assert [current] in records
        else:
            assert all(detector[0] != current for detector in records)
    assert simulate_tick_circuit(tc, seed)[1] == 0


@pytest.mark.parametrize("patch", ["A", "B"])
@pytest.mark.parametrize("rounds", [0, 2])
def test_y_term_at_later_memory(patch, rounds):
    operations = [
        LogicalOp(LogicalGateType.MEMORY, ["A"], rounds=0, basis="Y"),
        LogicalOp(LogicalGateType.MEMORY, [patch], rounds=rounds, basis="Y"),
        LogicalOp(LogicalGateType.MEMORY, ["A"], rounds=2, basis="X"),
    ]
    expected = None if patch == "A" and rounds > 0 else []
    assert _propagate_stabilizer_terms(operations, 2, ("A", "X", 7, "Y")) == expected


@pytest.mark.parametrize("basis", ["X", "Y", "Z"])
@pytest.mark.parametrize("family", ["X", "Y", "Z"])
@pytest.mark.parametrize("rounds", [0, 2])
def test_propagation_preparation_resolution(basis, family, rounds):
    operations = [
        LogicalOp(LogicalGateType.MEMORY, ["A"], rounds=rounds, per_patch_basis={"A": basis}),
        LogicalOp(LogicalGateType.MEMORY, ["A"], rounds=2),
    ]
    expected = None
    if rounds and family != "Y":
        expected = [("A", family, 7, 0)]
    elif not rounds and family == basis:
        expected = []
    assert _propagate_stabilizer_terms(operations, 1, ("A", "X" if family == "Y" else family, 7, family)) == expected


@pytest.mark.parametrize("seed", range(8))
def test_zero_round_preparation_breaks_cx_cancellation(seed):
    builder = make_builder("d3_mem_Z")
    builder.add_patch(SurfacePatch.create(3), "B", qubit_offset=17)
    builder.add_transversal_cx("A", "B")
    builder.add_memory("B", 0, "Y")
    builder.add_transversal_cx("A", "B")
    builder.add_memory(["A", "B"], 2, "Z")
    # Exercise propagation below public validation: the first CX also violates
    # the builder's requirement that B be prepared before a transversal gate.
    with pytest.raises(ValueError, match=r"B.*preceding MEMORY"):
        builder.to_tick_circuit()
    generator = GeneratorProbe(builder.patches, builder.operations)
    tc = generator.generate()
    assert len(generator.detectors) == 40
    assert [16, 8] not in [detector["abs_records"] for detector in generator.detectors]
    assert simulate_tick_circuit(tc, seed)[1] == 0


@pytest.mark.parametrize("seed", range(8))
def test_zero_round_preparation_closes_partner_without_record(seed):
    builder = BuilderProbe()
    for label, offset in [("A", 0), ("B", 17)]:
        builder.add_patch(SurfacePatch.create(3), label, qubit_offset=offset)
    builder.add_memory("B", 0, "X")
    builder.add_memory("A", 2, "Z")
    builder.add_transversal_cx("A", "B")
    builder.add_memory(["A", "B"], 2, {"A": "Z", "B": "X"})
    tc = builder.to_tick_circuit()
    detectors = json.loads(tc.get_meta("detectors"))
    assert len(detectors) == 48
    for index in range(4):
        assert [16 + index, 8 + index] in [detector["meas_ids"] for detector in detectors]
    assert simulate_tick_circuit(tc, seed)[1] == 0


@pytest.mark.parametrize("method", ["add_transversal_sz", "add_transversal_szdg"])
@pytest.mark.parametrize("seed", range(8))
def test_physical_sz_across_skipped_segment(method, seed):
    builder = make_builder("d3_mem_Z")
    builder.add_patch(SurfacePatch.create(3), "B", qubit_offset=17)
    getattr(builder, method)("A")
    builder.add_memory("B", 2, "Z")
    builder.add_memory("A", 2, "Z")
    tc = builder.to_tick_circuit()
    assert len(json.loads(tc.get_meta("detectors"))) == 44
    assert simulate_tick_circuit(tc, seed)[1] == 0


@pytest.mark.parametrize(
    ("method", "labels"),
    [
        ("add_transversal_h", ["B"]),
        ("add_transversal_sz", ["B"]),
        ("add_transversal_szdg", ["B"]),
        ("add_transversal_cx", ["A", "B"]),
        ("add_transversal_cx", ["B", "A"]),
    ],
)
@pytest.mark.parametrize("output", ["to_tick_circuit", "to_dag_circuit", "to_stim"])
def test_transversal_before_preparation_rejected(method, labels, output):
    builder = make_builder("d3_mem_Z")
    builder.add_patch(SurfacePatch.create(3), "B", qubit_offset=17)
    getattr(builder, method)(*labels)
    builder.add_memory(["A", "B"], 2, "Z")
    with pytest.raises(ValueError, match=r"B.*preceding MEMORY"):
        getattr(builder, output)()
    assert not builder.patches["B"].x_z_swapped


@pytest.mark.parametrize(
    ("method", "labels"),
    [
        ("add_transversal_h", ["B"]),
        ("add_transversal_sz", ["B"]),
        ("add_transversal_szdg", ["B"]),
        ("add_transversal_cx", ["A", "B"]),
        ("add_transversal_cx", ["B", "A"]),
    ],
)
@pytest.mark.parametrize("output", ["to_tick_circuit", "to_dag_circuit", "to_stim", "build_algorithm_descriptor"])
def test_gate_after_patch_final_memory_rejected(method, labels, output):
    builder = BuilderProbe()
    for label, offset in [("A", 0), ("B", 17)]:
        builder.add_patch(SurfacePatch.create(3), label, qubit_offset=offset)
    builder.add_memory(["A", "B"], 2, "Z")
    getattr(builder, method)(*labels)
    builder.add_memory("A", 2, "Z")
    # Validation must precede the reset, not merely restore state after failing.
    builder.patches["B"].x_z_swapped = True
    gate = method.removeprefix("add_").upper()
    with pytest.raises(ValueError, match=rf"(?i){gate}.*B.*after final data measurement"):
        getattr(builder, output)()
    assert builder.patches["B"].x_z_swapped


@pytest.mark.parametrize("method", ["add_sz_via_teleportation", "add_t_via_injection"])
def test_teleportation_validates_expanded_preparations(method):
    builder = BuilderProbe()
    for label, offset in [("A", 0), ("B", 17)]:
        builder.add_patch(SurfacePatch.create(3), label, qubit_offset=offset)
    getattr(builder, method)("A", "B", 0, 2)
    builder.to_tick_circuit()
    # A malformed expanded helper must be rejected just like a plain CX.
    builder.operations.pop(0)
    with pytest.raises(ValueError, match=r"A.*preceding MEMORY"):
        builder.to_tick_circuit()


@pytest.mark.parametrize("shape", ["d3_skip_segment", "d2_h_even", "d1_mem_Z"])
def test_round_measurement_index(shape, monkeypatch):
    builder = make_builder(shape)
    generator = GeneratorProbe(builder.patches, builder.operations)
    generator.generate()
    grouped = {}
    for key in generator.stab_meas:
        label, _, _, segment, round_index = key
        grouped.setdefault((label, segment, round_index), []).append(key)
    assert generator.round_measurements == grouped
    reads = []

    class RecordingRoundIndex(dict):
        def __getitem__(self, key):
            reads.append(key)
            return super().__getitem__(key)

    class LookupOnlyMeasurements(dict):
        def __iter__(self):
            pytest.fail("Round enumeration must not scan all measurements")

        def items(self):
            pytest.fail("Round enumeration must not scan all measurements")

        def keys(self):
            pytest.fail("Round enumeration must not scan all measurements")

        def values(self):
            pytest.fail("Round enumeration must not scan all measurements")

    generator.stab_meas = LookupOnlyMeasurements(generator.stab_meas)
    monkeypatch.setattr(generator, "_stab_meas_by_round", RecordingRoundIndex(generator.round_measurements))
    expected_reads = []
    for (label, segment, round_index), keys in generator.round_measurements.items():
        # Generation leaves each patch in its final orientation.
        last_segment = max(seg for patch, seg, _ in grouped if patch == label)
        if round_index == 0 or segment != last_segment:
            continue
        generator.segment_idx = segment
        before = len(generator.detectors)
        generator.round_detectors(label, round_index)
        expected_reads.append((label, segment, round_index))
        assert [detector["abs_records"][0] for detector in generator.detectors[before:]] == [
            generator.stab_meas[key] for key in keys
        ]
    assert reads == expected_reads


def test_propagation_unmeasured_partner():
    operations = [
        LogicalOp(LogicalGateType.MEMORY, ["A"], rounds=2),
        LogicalOp(LogicalGateType.TRANSVERSAL_CX, ["A", "B"]),
        LogicalOp(LogicalGateType.MEMORY, ["A", "B"], rounds=2),
    ]
    assert _propagate_stabilizer_terms(operations, 0, ("A", "X", 7, "X")) is None
    assert _propagate_stabilizer_terms(operations, 1, ("A", "X", 7, "X")) is None
    generator = GeneratorProbe(make_builder("d3_mem_Z").patches, operations)
    generator.segment_idx = 1
    generator.stab_meas["A", "X", 7, 0, 1] = 0
    generator.boundary_detector(("A", "X", 7), 1)
    assert generator.detectors == []


@pytest.mark.parametrize("missing", ["family", "index", "partner"])
def test_positive_memory_missing_measurement_raises(missing):
    operations = [
        LogicalOp(LogicalGateType.MEMORY, ["A", "B"], rounds=2),
        LogicalOp(LogicalGateType.TRANSVERSAL_CX, ["A", "B"]),
        LogicalOp(LogicalGateType.MEMORY, ["A", "B"], rounds=2),
    ]
    generator = GeneratorProbe(make_builder("d3_mem_Z").patches, operations)
    generator.segment_idx = 1
    if missing == "index":
        generator.stab_meas["A", "X", 8, 0, 1] = 0
    elif missing == "partner":
        generator.stab_meas["A", "X", 7, 0, 1] = 0
    label = "B" if missing == "partner" else "A"
    with pytest.raises(ValueError, match=rf"{label}.*X.*7.*0"):
        generator.boundary_detector(("A", "X", 7), 1)


@pytest.mark.parametrize("method", ["add_transversal_sz", "add_transversal_szdg"])
def test_physical_sz_layer_dependency(method, monkeypatch):
    builder = make_builder("d3_mem_Z")
    getattr(builder, method)("A")
    builder.add_memory("A", 2, "Z")
    calls = []
    original = gadgets.transversal_layer_gadget

    def record(*args, **kwargs):
        calls.append(kwargs["gate"])
        return original(*args, **kwargs)

    monkeypatch.setattr(gadgets, "transversal_layer_gadget", record)
    tc = builder.to_tick_circuit()
    gate = "SZ" if method == "add_transversal_sz" else "SZDG"
    assert calls == [gate]
    tick = _data_gate_tick(tc, "SZ" if gate == "SZ" else "SZdg", range(9))
    assert tc.get_tick(tick).gate_batches()[0].gate_type.name.upper() == gate


@pytest.mark.parametrize("variant", ["H", "SZ", "SZDG", "CX", "round_swapped", "init_Z_swapped", "init_X_swapped", "Y"])
@pytest.mark.parametrize("renamed", [False, True])
def test_deferred_renderer(variant, renamed):
    patch = SurfacePatch.create(3)
    allocation = gadgets.default_allocation(patch)
    if variant in {"H", "SZ", "SZDG"}:
        gadget = gadgets.transversal_layer_gadget(patch, allocation, gate=variant)
    elif variant == "CX":
        gadget = gadgets.transversal_cx_gadget(patch, allocation, patch, allocation)
    elif variant == "round_swapped":
        gadget = gadgets.syndrome_round_gadget(patch, allocation, round_index=0, x_z_swapped=True)
    elif variant.startswith("init_"):
        gadget = gadgets.init_syndrome_gadget(patch, allocation, basis=variant[5], x_z_swapped=True)
    else:
        gadget = gadgets.prep_gadget(patch, allocation, basis="Y")
    if renamed:
        gadget = replace(gadget, name="custom_function")
    assert gadget.x_z_swapped == (variant in {"round_swapped", "init_Z_swapped", "init_X_swapped"})
    with pytest.raises(NotImplementedError, match=gadget.name):
        render_gadget_function(gadget)


def test_y_readout_unsupported():
    builder = make_builder("d3_mem_Z")
    builder.add_memory("A", 2, "Y")
    with pytest.raises(NotImplementedError, match="Y readout"):
        builder.to_tick_circuit()


@pytest.mark.parametrize("method", ["add_transversal_cx", "add_sz_via_teleportation", "add_t_via_injection"])
@pytest.mark.parametrize("mismatch", ["rectangle", "rotated", "orientation", "stabilizer", "logical"])
def test_static_geometry_registration(method, mismatch):
    first = SurfacePatch.create(3)
    second = SurfacePatch.create(3)
    if mismatch == "rectangle":
        first, second = SurfacePatch.create(dx=3, dz=5), SurfacePatch.create(dx=5, dz=3)
    elif mismatch == "rotated":
        second = SurfacePatch.create(3, rotated=False)
    elif mismatch == "orientation":
        second = SurfacePatch.create(3, orientation=PatchOrientation.Z_TOP_BOTTOM)
    elif mismatch == "stabilizer":
        second.geometry.x_stabilizers[0] = replace(second.geometry.x_stabilizers[0], data_qubits=(0, 8))
    else:
        second.geometry.logical_x = replace(second.geometry.logical_x, data_qubits=(0, 1, 2))
    builder = BuilderProbe()
    builder.add_patch(first, "C")
    builder.add_patch(second, "T", qubit_offset=first.geometry.num_qubits)
    with pytest.raises(ValueError, match="geometry"):
        getattr(builder, method)("C", "T")
    assert builder.operations == []
    with pytest.raises(ValueError, match="geometry"):
        gadgets.transversal_cx_gadget(
            first,
            gadgets.default_allocation(first),
            second,
            gadgets.default_allocation(second),
        )


def test_runtime_orientation_check():
    builder = make_builder("d3_cx_zz")
    builder.operations.insert(1, logical_circuit.LogicalOp(LogicalGateType.TRANSVERSAL_H, ["C"]))
    with pytest.raises(ValueError, match="orientation"):
        builder.to_tick_circuit()


@pytest.mark.parametrize("method", ["add_sz_via_teleportation", "add_t_via_injection"])
def test_fresh_ancilla_atomic(method):
    builder = make_builder("d3_cx_zz")
    before = list(builder.operations)
    with pytest.raises(ValueError, match="fresh"):
        getattr(builder, method)("C", "T")
    assert builder.operations == before


def test_memory_first_y_preparation_gates():
    builder = make_builder("d3_sz_teleport_memory_first")
    tc = builder.to_tick_circuit()
    allocation = GeneratorProbe(builder.patches, []).allocation("Y")
    h_tick = _data_gate_tick(tc, "H", allocation.data_qubits)
    sz_tick = _data_gate_tick(tc, "SZ", allocation.data_qubits)
    ancillas = set(allocation.x_ancilla_qubits + allocation.z_ancilla_qubits)
    first_syndrome = next(
        i for i in range(tc.num_ticks()) for g in tc.get_tick(i).gate_batches() if set(g.qubits) & ancillas
    )
    assert h_tick < sz_tick < first_syndrome


@pytest.mark.parametrize("basis", ["X", "Y"])
def test_late_preparation_detectors(basis):
    patch = SurfacePatch.create(3)
    builder = BuilderProbe()
    builder.add_patch(patch, "A")
    builder.add_patch(patch, "B", qubit_offset=patch.geometry.num_qubits)
    builder.add_memory("A", 2, "Z")
    builder.add_memory("B", 2, basis)
    builder.add_memory("B", 2, "Z")
    generator = GeneratorProbe(builder.patches, builder.operations)
    tc = generator.generate()
    assert len(json.loads(tc.get_meta("observables"))) == 1
    first_records = {
        index: family
        for (label, family, _, seg, rnd), index in generator.stab_meas.items()
        if label == "B" and seg == 1 and rnd == 0
    }
    first_dets = [d for d in generator.detectors if d["coords"][2] == 2.0 and d["coords"][0] >= 8]
    assert len(first_dets) == (4 if basis == "X" else 0)
    assert all(len(d["abs_records"]) == 1 and first_records[d["abs_records"][0]] == "X" for d in first_dets)
    allocation = generator.allocation("B")
    h_tick = _data_gate_tick(generator.tc, "H", allocation.data_qubits)
    if basis == "Y":
        assert _data_gate_tick(generator.tc, "SZ", allocation.data_qubits) == h_tick + 1


def test_disjoint_terminal_bases():
    patch = SurfacePatch.create(3)
    builder = BuilderProbe()
    builder.add_patch(patch, "A")
    builder.add_patch(patch, "B", qubit_offset=patch.geometry.num_qubits)
    builder.add_memory("A", 2, "X")
    builder.add_memory("B", 2, "Z")
    tc = builder.to_tick_circuit()
    assert simulate_tick_circuit(tc)[1] == 0
    assert (
        len(
            [
                g
                for i in range(tc.num_ticks())
                for g in tc.get_tick(i).gate_batches()
                if g.gate_type.name == "H" and set(g.qubits) == set(range(9))
            ],
        )
        == 2
    )
    builder.operations[0].basis = "Y"
    with pytest.raises(NotImplementedError, match="Y readout"):
        builder.to_tick_circuit()


def test_split_zip_padding_and_measurement_labels():
    step = SurfaceCircuitStep
    generator = GeneratorProbe({}, [])
    measurements = generator.emit_steps(
        [
            (
                step(OpType.ALLOC, [0]),
                step(OpType.H, [0]),
                step(OpType.SZ, [0]),
                step(OpType.TICK),
                step(OpType.MEASURE, [0], "sx7"),
            ),
            (
                step(OpType.ALLOC, [1]),
                step(OpType.TICK),
                step(OpType.MEASURE, [1], "sz9"),
                step(OpType.TICK),
                step(OpType.X, [2]),
            ),
        ],
    )
    assert generator.tc.num_ticks() == 5
    assert measurements == {(0, 0): 0, (1, 1): 1}
    assert [g.gate_type.name for g in generator.tc.get_tick(1).gate_batches()] == ["H"]
    assert [g.gate_type.name for g in generator.tc.get_tick(2).gate_batches()] == ["SZ"]


def test_oracle_rejects_unsupported_operation():
    tc = TickCircuit()
    tc.tick().rz(0.1, [0])
    with pytest.raises(NotImplementedError, match="Unsupported replay operation"):
        stabilizer_generators_after(tc, 1)


def test_group_membership_signed_products():
    assert group_contains(("+XX", "+ZZ"), "-YY")
    assert not group_contains(("+XX", "+ZZ"), "+YY")


def test_transversal_h_requires_square():
    patch = SurfacePatch.create(dx=3, dz=5)
    with pytest.raises(ValueError, match="square"):
        gadgets.transversal_layer_gadget(patch, gadgets.default_allocation(patch), gate="H")


@pytest.mark.parametrize("swapped", [False, True])
def test_stabilizer_indices_are_not_list_positions(swapped, monkeypatch):
    original = gadgets.syndrome_round_gadget

    def renamed_round(*args, **kwargs):
        gadget = original(*args, **kwargs)
        return replace(
            gadget,
            steps=tuple(
                replace(step, label="custom") if step.op_type == OpType.MEASURE else step for step in gadget.steps
            ),
        )

    monkeypatch.setattr(gadgets, "syndrome_round_gadget", renamed_round)
    patch = SurfacePatch.create(3)
    patch.geometry.x_stabilizers.reverse()
    patch.geometry.z_stabilizers.reverse()
    builder = BuilderProbe()
    builder.add_patch(patch, "A")
    builder.add_memory("A", 2, "Z")
    if swapped:
        builder.add_transversal_h("A")
        builder.add_memory("A", 2, "X")
    generator = GeneratorProbe(builder.patches, builder.operations)
    tc = generator.generate()
    assert simulate_tick_circuit(tc)[1] == 0
    allocation = generator.allocation("A")
    # Every synthetic index must refer to the physical register named by the key.
    physical_measurements = [
        q
        for i in range(tc.num_ticks())
        for g in tc.get_tick(i).gate_batches()
        if g.gate_type.name == "MZ"
        for q in g.qubits
    ]
    for (_, family, index, seg, _), measurement in generator.stab_meas.items():
        physical_x = (family == "X") != (swapped and seg == 1)
        register = allocation.x_ancilla_qubits if physical_x else allocation.z_ancilla_qubits
        assert physical_measurements[measurement] == register[index]
    gadget = gadgets.syndrome_round_gadget(patch, allocation, round_index=0, x_z_swapped=swapped)
    assert gadget.x_z_swapped is swapped
    expected = (
        allocation.z_ancilla_qubits + allocation.x_ancilla_qubits
        if swapped
        else allocation.x_ancilla_qubits + allocation.z_ancilla_qubits
    )
    for op in (OpType.ALLOC, OpType.MEASURE):
        assert [s.qubits[0] for s in gadget.steps if s.op_type == op] == expected


@pytest.mark.parametrize("basis", ["Z", "X"])
def test_swapped_init_physical_roles(basis):
    patch = SurfacePatch.create(3)
    allocation = gadgets.default_allocation(patch)
    gadget = gadgets.init_syndrome_gadget(patch, allocation, basis=basis, x_z_swapped=True)
    register = allocation.z_ancilla_qubits if basis == "Z" else allocation.x_ancilla_qubits
    assert [s.qubits[0] for s in gadget.steps if s.op_type == OpType.MEASURE] == register
    h_qubits = [s.qubits[0] for s in gadget.steps if s.op_type == OpType.H]
    assert h_qubits == (register * 2 if basis == "Z" else [])
    for step in gadget.steps:
        if step.op_type == OpType.CX:
            assert step.qubits[0 if basis == "Z" else 1] in register


@pytest.mark.parametrize("basis", ["Z", "X"])
def test_swapped_init_matches_round_family(basis):
    patch = SurfacePatch.create(3)
    allocation = gadgets.default_allocation(patch)
    init = gadgets.init_syndrome_gadget(patch, allocation, basis=basis, x_z_swapped=True)
    full_round = gadgets.syndrome_round_gadget(patch, allocation, round_index=0, x_z_swapped=True)
    measured = {step.qubits[0] for step in init.steps if step.op_type == OpType.MEASURE}
    physical_init = [step for step in init.steps if step.op_type not in {OpType.COMMENT, OpType.TICK}]
    physical_family = [
        step
        for step in full_round.steps
        if step.op_type not in {OpType.COMMENT, OpType.TICK} and measured.intersection(step.qubits)
    ]
    assert physical_init == physical_family


def test_unequal_patch_streams_preserve_prep_and_rounds():
    builder = BuilderProbe()
    small, large = SurfacePatch.create(3), SurfacePatch.create(5)
    builder.add_patch(small, "A")
    builder.add_patch(large, "B", qubit_offset=small.geometry.num_qubits)
    builder.add_memory(["A", "B"], 2, {"A": "Z", "B": "Y"})
    builder.add_memory(["A", "B"], 2, "Z")
    tc = builder.to_tick_circuit()
    assert len(json.loads(tc.get_meta("observables"))) == 1
    assert simulate_tick_circuit(tc)[1] == 0
    assert (
        int(tc.get_meta("num_measurements"))
        == 4 * (small.geometry.num_ancilla + large.geometry.num_ancilla)
        + small.geometry.num_data
        + large.geometry.num_data
    )


def test_allocation_order_with_old_and_fresh_patches():
    generator = GeneratorProbe({}, [])
    generator.emit_steps([(SurfaceCircuitStep(OpType.ALLOC, [0]),)])
    generator.emit_steps(
        [
            (SurfaceCircuitStep(OpType.ALLOC, [0]),),
            (SurfaceCircuitStep(OpType.ALLOC, [1]),),
        ],
    )
    assert [(g.gate_type.name, g.qubits) for g in generator.tc.get_tick(1).gate_batches()] == [
        ("PZ", [0]),
        ("QAlloc", [1]),
    ]


@pytest.mark.parametrize("shape", [name for name in SHAPES if "teleport" in name or "inject" in name])
def test_injection_readout_is_separate_from_data_observable(shape):
    builder = make_builder(shape)
    tc = builder.to_tick_circuit()
    observables = json.loads(tc.get_meta("observables"))
    readouts = json.loads(tc.get_meta("injection_readouts"))
    assert len(observables) == len(readouts) == 1
    observable, readout = observables[0], readouts[0]
    assert observable["id"] == 0
    assert readout["basis"] == "Z"
    assert readout["data_patch"] == "D"
    assert readout["ancilla_patch"] == ("Y" if "teleport" in shape else "A")
    assert len(observable["meas_ids"]) == len(readout["meas_ids"]) == 3
    measured_qubits = [
        q
        for i in range(tc.num_ticks())
        for g in tc.get_tick(i).gate_batches()
        if g.gate_type.name == "MZ"
        for q in g.qubits
    ]
    patch = builder.patches["D"].patch
    assert [measured_qubits[i] for i in observable["meas_ids"]] == list(patch.geometry.logical_z.data_qubits)
    ancilla = builder.patches[readout["ancilla_patch"]]
    assert [measured_qubits[i] for i in readout["meas_ids"]] == [
        ancilla.qubit_offset + q for q in ancilla.patch.geometry.logical_z.data_qubits
    ]
    assert readout["records"] == [i - int(tc.get_meta("num_measurements")) for i in readout["meas_ids"]]
    raw_values = set()
    for seed in range(8):
        measurements, _, values = simulate_tick_circuit(tc, seed)
        assert values == {0: 0}
        raw_values.add(sum(measurements[i] for i in readout["meas_ids"]) % 2)
    assert raw_values == {0, 1}
    assert stim.Circuit(builder.to_stim()).detector_error_model().num_observables == 1
    descriptor = builder.build_algorithm_descriptor()
    assert descriptor["injection_readouts"] == [readout | {"data_z_frame_slot": 1, "ancilla_z_frame_slot": 3}]
    assert descriptor["num_observables"] == 1
    assert descriptor["num_frame_slots"] == 4
    if "inject" in shape:
        decision = next(
            g for boundary in descriptor["boundary_gates"] for g in boundary if g["type"] == "TGateInjection"
        )
        assert decision["z_obs_bit"] == 1
        assert decision["ancilla_z_bit"] == 3


def test_nonrotated_conflicting_syndrome_layer_rejected():
    builder = LogicalCircuitBuilder()
    builder.add_patch(SurfacePatch.create(3, rotated=False), "A")
    builder.add_memory("A", 2, "Z")
    with pytest.raises(ValueError, match=r"Gadget layer.*CX round.*repeated CX"):
        builder.to_tick_circuit()


def test_same_type_conflict_rejected_across_subticks():
    generator = GeneratorProbe({}, [])
    with pytest.raises(ValueError, match=r"Gadget layer.*repeated H"):
        generator.emit_steps(
            [
                (
                    SurfaceCircuitStep(OpType.H, [0]),
                    SurfaceCircuitStep(OpType.Z, [0]),
                    SurfaceCircuitStep(OpType.H, [0]),
                ),
            ],
        )
    assert generator.tc.num_ticks() == 0


def test_same_type_on_separate_tick_groups_allowed():
    generator = GeneratorProbe({}, [])
    generator.emit_steps(
        [
            (SurfaceCircuitStep(OpType.H, [0]), SurfaceCircuitStep(OpType.TICK), SurfaceCircuitStep(OpType.H, [0])),
        ],
    )
    assert generator.tc.num_ticks() == 2


def test_multi_qubit_measurement_step_rejected_before_emission():
    generator = GeneratorProbe({}, [])
    with pytest.raises(ValueError, match="MEASURE requires exactly one qubit"):
        generator.emit_steps(
            [
                (SurfaceCircuitStep(OpType.MEASURE, [0, 1], "pair"), SurfaceCircuitStep(OpType.MEASURE, [2], "later")),
            ],
        )
    assert generator.meas_count == 0
    assert generator.tc.num_ticks() == 0


def test_replay_qalloc_resets_to_zero():
    tc = TickCircuit()
    tc.tick().qalloc([0])
    tc.tick().x([0])
    tc.tick().qalloc([0])
    tc.tick().mz([0])
    tc.set_meta("num_measurements", "1")
    assert simulate_tick_circuit(tc)[0] == [0]


@pytest.mark.parametrize("method", ["add_transversal_cx", "add_sz_via_teleportation", "add_t_via_injection"])
def test_geometry_equality_uses_stabilizer_indices(method):
    first, second = SurfacePatch.create(3), SurfacePatch.create(3)
    second.geometry.x_stabilizers.reverse()
    second.geometry.z_stabilizers.reverse()
    assert gadgets.same_static_geometry(first, second)
    builder = BuilderProbe()
    builder.add_patch(first, "C")
    builder.add_patch(second, "T", qubit_offset=first.geometry.num_qubits)
    getattr(builder, method)("C", "T")
    assert builder.operations
    ctrl = gadgets.default_allocation(first)
    target = GeneratorProbe(builder.patches, []).allocation("T")
    assert len(gadgets.transversal_cx_gadget(first, ctrl, second, target).steps) == first.geometry.num_data


@pytest.mark.parametrize("basis", ["", "Q", "XYZ", " X", {"A": "Q"}, {"A": "x", "B": "bad"}])
def test_memory_basis_validation_is_atomic(basis):
    builder = make_builder("d3_mem_Z")
    before = list(builder.operations)
    with pytest.raises(ValueError, match="Unsupported memory basis"):
        builder.add_memory("A", 2, basis)
    assert builder.operations == before


@pytest.mark.parametrize("basis", ["x", "y", "z", {"A": "x"}, {"A": "y"}, {"A": "z"}])
def test_memory_basis_validation_accepts_lowercase(basis):
    builder = BuilderProbe()
    builder.add_patch(SurfacePatch.create(3), "A")
    builder.add_memory("A", 2, basis)
    op = builder.operations[0]
    expected = basis.upper() if isinstance(basis, str) else basis["A"].upper()
    assert op.per_patch_basis.get("A", op.basis) == expected


@pytest.mark.parametrize(
    ("factory", "keyword", "basis"),
    [
        (gadgets.prep_gadget, "basis", "Q"),
        (gadgets.init_syndrome_gadget, "basis", "Y"),
        (gadgets.init_syndrome_gadget, "basis", "Q"),
        (gadgets.measure_out_gadget, "basis", "Q"),
        (gadgets.logical_pauli_gadget, "pauli", "H"),
        (gadgets.logical_pauli_gadget, "pauli", "Y"),
    ],
)
def test_gadget_rejects_unknown_basis(factory, keyword, basis):
    patch = SurfacePatch.create(3)
    with pytest.raises(ValueError, match="Unsupported basis"):
        factory(patch, gadgets.default_allocation(patch), **{keyword: basis})


@pytest.mark.parametrize("pauli", ["X", "Z"])
def test_logical_pauli_requires_operator(pauli):
    patch = SurfacePatch.create(3)
    setattr(patch.geometry, f"logical_{pauli.lower()}", None)
    with pytest.raises(ValueError, match=f"no logical {pauli}"):
        gadgets.logical_pauli_gadget(patch, gadgets.default_allocation(patch), pauli=pauli)


@pytest.mark.parametrize("family", ["X", "Z"])
def test_stabilizer_indices_must_cover_register(family):
    patch = SurfacePatch.create(3)
    stabs = getattr(patch.geometry, f"{family.lower()}_stabilizers")
    stabs[0] = replace(stabs[0], index=len(stabs))
    with pytest.raises(ValueError, match=f"{family} stabilizer indices"):
        gadgets.syndrome_round_gadget(patch, gadgets.default_allocation(patch), round_index=0)


def test_missing_stabilizer_coordinate_is_loud():
    builder = make_builder("d3_mem_Z")
    generator = GeneratorProbe(builder.patches, [])
    with pytest.raises(ValueError, match=r"Patch 'A'.*index 99"):
        generator.ancilla_spatial_coords("A", "X", 99)


@pytest.mark.parametrize("seed", range(8))
def test_sz_ancilla_encoded_y_after_first_syndrome(seed):
    builder = make_builder("d3_sz_teleport_first_op")
    tc = builder.to_tick_circuit()
    allocation = GeneratorProbe(builder.patches, []).allocation("Y")
    ancillas = set(allocation.x_ancilla_qubits + allocation.z_ancilla_qubits)
    first_readout = next(
        i
        for i in range(tc.num_ticks())
        for gate in tc.get_tick(i).gate_batches()
        if gate.gate_type.name == "MZ" and ancillas.intersection(gate.qubits)
    )
    group = stabilizer_generators_after(tc, first_readout + 1, seed=seed)
    assert len(allocation.data_qubits) == 9
    logical_y = _pauli(2 * builder.patches["Y"].patch.geometry.num_qubits, ("Y", allocation.data_qubits))
    assert group_contains(group, logical_y) or group_contains(group, "-" + logical_y[1:])
    patch = builder.patches["Y"].patch
    num_qubits = 2 * patch.geometry.num_qubits
    for family, checks in (("X", patch.geometry.x_stabilizers), ("Z", patch.geometry.z_stabilizers)):
        for check in checks:
            pauli = _pauli(num_qubits, (family, [allocation.data_qubits[q] for q in check.data_qubits]))
            assert group_contains(group, pauli) or group_contains(group, "-" + pauli[1:]), (family, check.index)
    logical_x = {allocation.data_qubits[q] for q in patch.geometry.logical_x.data_qubits}
    logical_z = {allocation.data_qubits[q] for q in patch.geometry.logical_z.data_qubits}
    logical_y = _pauli(
        num_qubits,
        ("Y", logical_x & logical_z),
        ("X", logical_x - logical_z),
        ("Z", logical_z - logical_x),
    )
    assert group_contains(group, logical_y) or group_contains(group, "-" + logical_y[1:])


@pytest.mark.parametrize(("dx", "dz"), [(2, 2), (2, 3), (3, 2)])
def test_sz_teleportation_requires_odd_ancilla(dx, dz):
    builder = BuilderProbe()
    patch = SurfacePatch.create(dx=dx, dz=dz)
    builder.add_patch(patch, "D")
    builder.add_patch(patch, "A", qubit_offset=patch.geometry.num_qubits)
    with pytest.raises(ValueError, match=r"ancilla 'A'.*odd dx and dz.*logical-Y"):
        builder.add_sz_via_teleportation("D", "A", 2, 2)
    assert builder.operations == []


@pytest.mark.parametrize("helper", ["add_sz_via_teleportation", "add_t_via_injection"])
@pytest.mark.parametrize(
    ("method", "args"),
    [
        ("add_memory", ("A", 1, "X")),
        ("add_memory", (["D", "A"], 1, "Z")),
        ("add_transversal_h", ("A",)),
        ("add_transversal_sz", ("A",)),
        ("add_transversal_szdg", ("A",)),
        ("add_transversal_cx", ("D", "A")),
        ("add_transversal_cx", ("A", "D")),
        ("add_sz_via_teleportation", ("A", "B")),
        ("add_t_via_injection", ("A", "B")),
    ],
)
def test_consumed_injection_ancilla_rejects_operations(helper, method, args):
    builder = BuilderProbe()
    patch = SurfacePatch.create(3)
    for i, label in enumerate(("D", "A", "B")):
        builder.add_patch(patch, label, qubit_offset=i * patch.geometry.num_qubits)
    getattr(builder, helper)("D", "A", 2, 2)
    before = list(builder.operations)
    with pytest.raises(ValueError, match=r"ancilla 'A'.*consumed"):
        getattr(builder, method)(*args)
    assert builder.operations == before


@pytest.mark.parametrize("basis", ["Z", "X"])
@pytest.mark.parametrize("seed", range(8))
def test_y_preparation_has_no_final_observable(basis, seed):
    builder = BuilderProbe()
    builder.add_patch(SurfacePatch.create(3), "A")
    builder.add_memory("A", 2, "Y")
    builder.add_memory("A", 2, basis)
    tc = builder.to_tick_circuit()
    _, fired, observables = simulate_tick_circuit(tc, seed)
    assert observables == {}
    assert fired == 0
    assert stim.Circuit(builder.to_stim()).detector_error_model().num_observables == 0


@pytest.mark.parametrize("helper", ["add_sz_via_teleportation", "add_t_via_injection"])
def test_injection_readout_requires_logical_operator(helper):
    builder = BuilderProbe()
    patch = SurfacePatch.create(3)
    patch.geometry.logical_z = None
    builder.add_patch(patch, "D")
    builder.add_patch(patch, "A", qubit_offset=patch.geometry.num_qubits)
    getattr(builder, helper)("D", "A", 2, 2)
    # Keep data in memory so the ancilla reaches its readout first.
    builder.add_memory("D", 2, "Z")
    with pytest.raises(ValueError, match=r"ancilla 'A'.*no logical operator"):
        builder.build_algorithm_descriptor()


def test_empty_tick_groups_preserved_until_stream_end():
    generator = GeneratorProbe({}, [])
    step = SurfaceCircuitStep
    generator.emit_steps(
        [
            (
                step(OpType.COMMENT, label="empty CX layer"),
                step(OpType.TICK),
                step(OpType.X, [0]),
                step(OpType.TICK),
                step(OpType.COMMENT, label="another empty CX layer"),
                step(OpType.TICK),
                step(OpType.COMMENT, label="trailing empty layer"),
                step(OpType.TICK),
            ),
            (
                step(OpType.TICK),
                step(OpType.TICK),
                step(OpType.TICK),
                step(OpType.Z, [1]),
                step(OpType.TICK),
                step(OpType.TICK),
            ),
        ],
    )
    assert generator.tc.num_ticks() == 4
    assert [len(generator.tc.get_tick(i).gate_batches()) for i in range(4)] == [0, 1, 0, 1]


def test_measure_out_y_remains_unsupported():
    patch = SurfacePatch.create(3)
    with pytest.raises(NotImplementedError, match="Y readout"):
        gadgets.measure_out_gadget(patch, gadgets.default_allocation(patch), basis="y")


@pytest.mark.parametrize(
    "op_type",
    [OpType.ALLOC, OpType.H, OpType.SZ, OpType.SZDG, OpType.X, OpType.Z, OpType.CX, OpType.MEASURE],
)
def test_physical_steps_require_qubits_before_emission(op_type):
    generator = GeneratorProbe({}, [])
    with pytest.raises(ValueError, match=f"{op_type.name} requires at least one qubit"):
        generator.emit_steps(
            [
                (SurfaceCircuitStep(OpType.ALLOC, [0]),),
                (SurfaceCircuitStep(op_type),),
            ],
        )
    assert generator.tc.num_ticks() == 0
    assert generator.meas_count == 0


@pytest.mark.parametrize(("y_patch", "readout"), [("C", "Z"), ("T", "X")])
@pytest.mark.parametrize("seed", range(8))
def test_y_preparation_makes_entangled_partner_readout_unreliable(y_patch, readout, seed):
    builder = BuilderProbe()
    patch = SurfacePatch.create(3)
    builder.add_patch(patch, "C")
    builder.add_patch(patch, "T", qubit_offset=patch.geometry.num_qubits)
    basis = {label: "Y" if label == y_patch else readout for label in ("C", "T")}
    builder.add_memory(["C", "T"], 2, basis)
    builder.add_transversal_cx("C", "T")
    builder.add_memory(["C", "T"], 2, readout)
    tc = builder.to_tick_circuit()
    _, fired, observables = simulate_tick_circuit(tc, seed)
    assert fired == 0
    assert observables == {}
    assert stim.Circuit(builder.to_stim()).detector_error_model().num_observables == 0


@pytest.mark.parametrize("family", ["X", "Z"])
def test_detector_coordinates_use_geometry_edited_after_registration(family):
    builder = BuilderProbe()
    patch = SurfacePatch.create(3)
    offset = 11
    coord_offset = (2.0, 4.0)
    builder.add_patch(patch, "A", qubit_offset=offset, coord_offset=coord_offset)
    builder.add_memory("A", 2, family)
    builder.to_tick_circuit()
    checks = patch.geometry.x_stabilizers if family == "X" else patch.geometry.z_stabilizers
    first, second = checks[:2]
    checks[:2] = [replace(first, index=second.index), replace(second, index=first.index)]
    tc = builder.to_tick_circuit()
    measured_qubits = [
        q
        for i in range(tc.num_ticks())
        for gate in tc.get_tick(i).gate_batches()
        if gate.gate_type.name == "MZ"
        for q in gate.qubits
    ]
    allocation = gadgets.default_allocation(patch)
    register = allocation.x_ancilla_qubits if family == "X" else allocation.z_ancilla_qubits
    detectors = json.loads(tc.get_meta("detectors"))
    for check in checks:
        measurement = measured_qubits.index(offset + register[check.index])
        detector = next(d for d in detectors if d["meas_ids"] == [measurement])
        positions = [patch.geometry.id_to_pos[q] for q in check.data_qubits]
        expected = [
            2 * sum(col for row, col in positions) / len(positions) + coord_offset[0],
            2 * sum(row for row, col in positions) / len(positions) + coord_offset[1],
            0.0,
        ]
        assert detector["coords"] == expected


@pytest.mark.parametrize("basis", ["X", "Z"])
@pytest.mark.parametrize("swapped", [False, True])
def test_ordinary_readout_requires_logical_operator(basis, swapped):
    builder = BuilderProbe()
    patch = SurfacePatch.create(3)
    builder.add_patch(patch, "A")
    builder.add_memory("A", 2, "Z")
    if swapped:
        builder.add_transversal_h("A")
    builder.add_memory("A", 2, basis)
    family = ("Z" if basis == "X" else "X") if swapped else basis
    setattr(patch.geometry, f"logical_{family.lower()}", None)
    with pytest.raises(ValueError, match=f"Patch 'A' has no logical operator for {basis} readout"):
        builder.to_tick_circuit()
    assert builder.patches["A"].x_z_swapped is False


def _readout_chain(preparations, gates):
    builder = BuilderProbe()
    patch = SurfacePatch.create(3)
    labels = list(preparations)
    for index, label in enumerate(labels):
        builder.add_patch(patch, label, qubit_offset=index * patch.geometry.num_qubits)
    builder.add_memory(labels, 2, preparations)
    readout = {label: "Z" if basis == "Y" else basis for label, basis in preparations.items()}
    for control, target in gates:
        builder.add_transversal_cx(control, target)
        builder.add_memory(labels, 2, readout)
    return builder


@pytest.mark.parametrize("seed", range(8))
def test_logical_readout_y_chain(seed):
    builder = _readout_chain({"C": "Y", "T": "Z", "U": "Z"}, [("C", "T"), ("T", "U")])
    _, fired, observables = simulate_tick_circuit(builder.to_tick_circuit(), seed)
    assert fired == 0
    assert observables == {}
    assert stim.Circuit(builder.to_stim()).detector_error_model().num_observables == 0


@pytest.mark.parametrize("seed", range(8))
def test_logical_readout_repeated_cx_cancels(seed):
    builder = _readout_chain({"C": "Y", "T": "Z"}, [("C", "T"), ("C", "T")])
    _, fired, observables = simulate_tick_circuit(builder.to_tick_circuit(), seed)
    assert fired == 0
    assert observables == {1: 0}


@pytest.mark.parametrize("seed", range(8))
def test_logical_readout_x_chain(seed):
    builder = _readout_chain({"A": "X", "B": "X", "C": "X"}, [("A", "B"), ("B", "C")])
    _, fired, observables = simulate_tick_circuit(builder.to_tick_circuit(), seed)
    assert fired == 0
    assert observables == {0: 0, 1: 0, 2: 0}


@pytest.mark.parametrize("seed", range(8))
def test_logical_readout_cross_basis(seed):
    builder = _readout_chain({"C": "X", "T": "Z"}, [("C", "T")])
    _, fired, observables = simulate_tick_circuit(builder.to_tick_circuit(), seed)
    assert fired == 0
    assert observables == {}


@pytest.mark.parametrize("seed", range(8))
def test_logical_readout_h_image(seed):
    builder = make_builder("d3_h_z_to_x")
    _, fired, observables = simulate_tick_circuit(builder.to_tick_circuit(), seed)
    assert fired == 0
    assert observables == {0: 0}


@pytest.mark.parametrize("gate", ["add_transversal_sz", "add_transversal_szdg"])
def test_logical_readout_x_crossing_physical_s(gate):
    builder = BuilderProbe()
    builder.add_patch(SurfacePatch.create(3), "A")
    builder.add_memory("A", 2, "X")
    getattr(builder, gate)("A")
    builder.add_memory("A", 2, "X")
    assert json.loads(builder.to_tick_circuit().get_meta("observables")) == []


def _memory_op(bases, rounds=2):
    return LogicalOp(LogicalGateType.MEMORY, list(bases), rounds=rounds, per_patch_basis=bases)


@pytest.mark.parametrize("rounds", [0, 2])
@pytest.mark.parametrize("prepared", ["X", "Y", "Z"])
@pytest.mark.parametrize("readout", ["X", "Z"])
def test_readout_walk_preparation_closure(rounds, prepared, readout):
    operations = [_memory_op({"A": prepared}, rounds), _memory_op({"A": readout})]
    assert _logical_readout_is_deterministic(operations, 1, "A", readout) == (prepared == readout)


@pytest.mark.parametrize(
    ("prepared", "readout", "expected"),
    [
        ("X", "Z", True),
        ("Z", "X", True),
        ("X", "X", False),
        ("Z", "Z", False),
    ],
)
def test_readout_walk_h_image(prepared, readout, expected):
    operations = [
        _memory_op({"A": prepared}),
        LogicalOp(LogicalGateType.TRANSVERSAL_H, ["A"]),
        _memory_op({"A": readout}),
    ]
    assert _logical_readout_is_deterministic(operations, 1, "A", readout) is expected


@pytest.mark.parametrize(
    ("patch", "kind", "prepared", "expected"),
    [
        ("C", "X", {"C": "X", "T": "Z"}, False),
        ("T", "Z", {"C": "X", "T": "Z"}, False),
        ("C", "Z", {"C": "Z", "T": "X"}, True),
        ("T", "X", {"C": "Z", "T": "X"}, True),
        ("C", "X", {"C": "X", "T": "X"}, True),
        ("T", "Z", {"C": "Z", "T": "Z"}, True),
    ],
)
def test_readout_walk_cx_images(patch, kind, prepared, expected):
    operations = [
        _memory_op(prepared),
        LogicalOp(LogicalGateType.TRANSVERSAL_CX, ["C", "T"]),
        _memory_op(prepared),
    ]
    assert _logical_readout_is_deterministic(operations, 1, patch, kind) is expected


@pytest.mark.parametrize("gate", [LogicalGateType.TRANSVERSAL_SZ, LogicalGateType.TRANSVERSAL_SZdg])
@pytest.mark.parametrize("kind", ["X", "Z"])
def test_readout_walk_physical_s_image(gate, kind):
    operations = [_memory_op({"A": kind}), LogicalOp(gate, ["A"]), _memory_op({"A": kind})]
    assert _logical_readout_is_deterministic(operations, 1, "A", kind) == (kind == "Z")


@pytest.mark.parametrize("gate", [LogicalGateType.TRANSVERSAL_H, LogicalGateType.TRANSVERSAL_SZ])
def test_readout_walk_ignores_unrelated_gates(gate):
    operations = [_memory_op({"A": "X"}), LogicalOp(gate, ["B"]), _memory_op({"A": "X"})]
    assert _logical_readout_is_deterministic(operations, 1, "A", "X")


@pytest.mark.parametrize(("patch", "kind"), [("C", "X"), ("T", "Z")])
def test_readout_walk_rejects_dead_partner(patch, kind):
    """This gate-after-readout shape is rejected as invalid on the follow-up branch."""
    operations = [
        _memory_op({"C": kind, "T": kind}),
        LogicalOp(LogicalGateType.TRANSVERSAL_CX, ["C", "T"]),
        _memory_op({patch: kind}),
    ]
    assert not _logical_readout_is_deterministic(operations, 1, patch, kind)


@pytest.mark.parametrize(("patch", "kind"), [("C", "Z"), ("T", "X")])
def test_readout_walk_unaffected_term_does_not_read_dead_partner(patch, kind):
    operations = [
        _memory_op({"C": kind, "T": kind}),
        LogicalOp(LogicalGateType.TRANSVERSAL_CX, ["C", "T"]),
        _memory_op({patch: kind}),
    ]
    assert _logical_readout_is_deterministic(operations, 1, patch, kind)


def test_readout_walk_requires_preparation_for_every_term():
    operations = [
        _memory_op({"C": "X"}),
        LogicalOp(LogicalGateType.TRANSVERSAL_CX, ["C", "T"]),
        _memory_op({"C": "X"}),
    ]
    with pytest.raises(ValueError, match=r"without preparation.*T"):
        _logical_readout_is_deterministic(operations, 1, "C", "X")


@pytest.mark.parametrize("segment", [-1, 1])
def test_readout_walk_requires_existing_segment(segment):
    with pytest.raises(ValueError, match="No memory segment"):
        _logical_readout_is_deterministic([_memory_op({"A": "Z"})], segment, "A", "Z")


def test_readout_walk_requires_patch_in_segment():
    with pytest.raises(ValueError, match="Patch 'B' is not in memory segment"):
        _logical_readout_is_deterministic([_memory_op({"A": "Z"})], 0, "B", "Z")


def test_readout_walk_requires_logical_type():
    with pytest.raises(ValueError, match="Unsupported logical readout type"):
        _logical_readout_is_deterministic([_memory_op({"A": "Z"})], 0, "A", "Y")


@pytest.mark.parametrize("seed", range(8))
def test_zero_round_final_memory_is_not_preparation(seed):
    builder = LogicalCircuitBuilder()
    builder.add_patch(SurfacePatch.create(3), "A")
    builder.add_memory("A", 2, "X")
    builder.add_memory("A", 0, "Z")
    tc = builder.to_tick_circuit()
    assert json.loads(tc.get_meta("observables")) == []
    assert simulate_tick_circuit(tc, seed)[2] == {}


@pytest.mark.parametrize("seed", range(8))
def test_zero_round_first_memory_is_preparation(seed):
    builder = LogicalCircuitBuilder()
    builder.add_patch(SurfacePatch.create(3), "A")
    builder.add_memory("A", 0, "X")
    builder.add_memory("A", 2, "X")
    tc = builder.to_tick_circuit()
    assert [obs["id"] for obs in json.loads(tc.get_meta("observables"))] == [0]
    assert simulate_tick_circuit(tc, seed)[2] == {0: 0}
