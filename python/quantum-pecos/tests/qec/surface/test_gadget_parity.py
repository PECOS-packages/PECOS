# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Independent captured-oracle parity for physical surface gadgets."""

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from pecos.guppy_gen.gadget_render import render_gadget_function, render_surface_gadget_module
from pecos.guppy_gen.protocol_render import render_surface_protocol_module
from pecos.guppy_gen.surface import generate_guppy_source
from pecos.qec.surface import SurfacePatch, TwirlConfig
from pecos.qec.surface.circuit_builder import (
    GuppyRenderer,
    OpType,
    QubitAllocation,
    SurfaceCircuitStep,
    TickCircuitRenderer,
    build_surface_code_circuit,
    generate_tick_circuit_from_patch,
    tick_circuit_to_stim,
)
from pecos.qec.surface.gadgets import (
    default_allocation,
    init_syndrome_gadget,
    logical_pauli_gadget,
    measure_out_gadget,
    memory_gadgets,
    syndrome_round_gadget,
    transversal_cx_gadget,
)
from pecos.qec.surface.schedule import compute_cnot_schedule

GOLDENS = Path(__file__).parent / "goldens" / "gadget_parity"
META_KEYS = ("detectors", "observables", "num_measurements", "num_detectors", "basis")
PHASE_KEYS = ("phase", "syndrome_round", "cx_round")

OPS_NAMES = (
    "ops_d3_X_r0.json",
    "ops_d3_X_r1.json",
    "ops_d3_X_r3.json",
    "ops_d3_Z_r0.json",
    "ops_d3_Z_r1.json",
    "ops_d3_Z_r2_balanced.json",
    "ops_d3_Z_r2_budget2.json",
    "ops_d3_Z_r2_szz.json",
    "ops_d3_Z_r2_twirl_between.json",
    "ops_d3_Z_r2_twirl_gate.json",
    "ops_d3_Z_r3.json",
    "ops_d3_x_lower_r1.json",
    "ops_d3_z_lower_r1.json",
    "ops_d3nonrot_X_r1.json",
    "ops_d5_X_r1.json",
    "ops_d5_X_r3.json",
    "ops_d5_Z_r1.json",
    "ops_d5_Z_r3.json",
    "ops_dx1dz3_Z_r1.json",
    "ops_dx3dz5_Z_r1.json",
)
STIM_NAMES = (
    "stim_d3_X_r0.txt",
    "stim_d3_X_r1.txt",
    "stim_d3_X_r3.txt",
    "stim_d3_Z_r0.txt",
    "stim_d3_Z_r1.txt",
    "stim_d3_Z_r3.txt",
    "stim_d3nonrot_X_r1.txt",
    "stim_d5_X_r1.txt",
    "stim_d5_X_r3.txt",
    "stim_d5_Z_r1.txt",
    "stim_d5_Z_r3.txt",
    "stim_dx1dz3_Z_r1.txt",
    "stim_dx3dz5_Z_r1.txt",
)
GUPPY_NAMES = (
    "guppy_d3.py.txt",
    "guppy_d3nonrot.py.txt",
    "guppy_d5.py.txt",
    "guppy_d7.py.txt",
    "guppy_dx1dz3.py.txt",
    "guppy_dx3dz5.py.txt",
    "guppy_dx5dz3.py.txt",
)
EXPECTED_FILES = {
    "protocol_d3.py.txt",
    *OPS_NAMES,
    *STIM_NAMES,
    *GUPPY_NAMES,
    *(name.replace("stim_", "tickmeta_").replace(".txt", ".json") for name in STIM_NAMES),
    *(name.replace("stim_", "tickphases_").replace(".txt", ".json") for name in STIM_NAMES),
}
assert all((GOLDENS / name).is_file() for name in EXPECTED_FILES)
assert {path.name for path in GOLDENS.iterdir()} == EXPECTED_FILES


def _patch(name: str) -> SurfacePatch:
    if name.startswith("dx"):
        dx, dz = name.removeprefix("dx").split("dz")
        return SurfacePatch.create(dx=int(dx), dz=int(dz))
    if name == "d3nonrot":
        return SurfacePatch.create(distance=3, rotated=False)
    return SurfacePatch.create(distance=int(name[1:]))


def _case(name: str) -> tuple:
    geometry, basis, *parts = name.split("_")
    rounds = int(next(part[1:] for part in parts if part.startswith("r")))
    kwargs = {}
    if "balanced" in parts:
        kwargs["check_plan"] = "cx_balanced_data_v1"
    if "budget2" in parts:
        kwargs["ancilla_budget"] = 2
    if "szz" in parts:
        kwargs["interaction_basis"] = "szz"
    if "twirl" in parts:
        kwargs["twirl"] = TwirlConfig(site_schedule="before_two_qubit_gate") if "gate" in parts else TwirlConfig()
    return _patch(geometry), rounds, basis, kwargs


def _ops(steps: list, allocation: QubitAllocation) -> dict:
    return {
        "allocation": {
            "data": allocation.data_qubits,
            "x_anc": allocation.x_ancilla_qubits,
            "z_anc": allocation.z_ancilla_qubits,
        },
        "ops": [{"op": s.op_type.name, "qubits": s.qubits, "label": s.label} for s in steps],
    }


@pytest.mark.parametrize("path", [GOLDENS / name for name in OPS_NAMES], ids=lambda p: p.stem)
def test_op_parity(path: Path) -> None:
    patch, rounds, basis, kwargs = _case(path.stem.removeprefix("ops_"))
    steps, allocation = build_surface_code_circuit(patch, rounds, basis, **kwargs)
    assert _ops(steps, allocation) == json.loads(path.read_text())


@pytest.mark.parametrize("path", [GOLDENS / name for name in STIM_NAMES], ids=lambda p: p.stem)
def test_tick_parity(path: Path) -> None:
    name = path.stem.removeprefix("stim_")
    patch, rounds, basis, kwargs = _case(name)
    circuit = generate_tick_circuit_from_patch(patch, rounds, basis, **kwargs)
    assert tick_circuit_to_stim(circuit) == path.read_text()
    assert {key: circuit.get_meta(key) for key in META_KEYS} == json.loads(
        (GOLDENS / f"tickmeta_{name}.json").read_text(),
    )
    assert [
        {key: circuit.get_tick_meta(i, key) for key in PHASE_KEYS} for i in range(circuit.num_ticks())
    ] == json.loads((GOLDENS / f"tickphases_{name}.json").read_text())


@pytest.mark.parametrize("path", [GOLDENS / name for name in GUPPY_NAMES], ids=lambda p: p.name)
def test_guppy_parity(path: Path) -> None:
    patch = _patch(path.name.removeprefix("guppy_").removesuffix(".py.txt"))
    source = render_surface_gadget_module(patch)
    assert source == path.read_text()
    assert source == generate_guppy_source(patch)


@pytest.mark.parametrize("geometry", ["d3", "d5", "dx3dz5", "d3nonrot"])
@pytest.mark.parametrize("basis", ["Z", "X"])
def test_standalone_allocation(geometry: str, basis: str) -> None:
    patch = _patch(geometry)
    allocation = default_allocation(patch)
    shifted = QubitAllocation(
        *[
            [q + 100 for q in group]
            for group in (allocation.data_qubits, allocation.x_ancilla_qubits, allocation.z_ancilla_qubits)
        ],
    )
    canonical = memory_gadgets(patch, 2, basis, allocation=allocation)
    relocated = memory_gadgets(patch, 2, basis, allocation=shifted)
    assert [render_gadget_function(g) for g in canonical] == [render_gadget_function(g) for g in relocated]
    renderer = TickCircuitRenderer()
    first = renderer.render([s for g in canonical for s in g.steps], allocation, patch, 2, basis)
    second = renderer.render([s for g in relocated for s in g.steps], shifted, patch, 2, basis)
    for pauli in ("X", "Z"):
        assert render_gadget_function(logical_pauli_gadget(patch, allocation, pauli=pauli)) == render_gadget_function(
            logical_pauli_gadget(patch, shifted, pauli=pauli),
        )
    first_annotations = first.annotations()
    second_annotations = second.annotations()
    assert len(first_annotations) == len(second_annotations)
    for left, right in zip(first_annotations, second_annotations, strict=True):
        assert {k: v for k, v in left.items() if k != "pauli"} == {k: v for k, v in right.items() if k != "pauli"}
        assert left["pauli"].get_phase() == right["pauli"].get_phase()
        assert [(axis, q + 100) for axis, q in left["pauli"].get_paulis()] == right["pauli"].get_paulis()
        assert [q + 100 for q in left["pauli"].qubits()] == right["pauli"].qubits()
    assert first.num_ticks() == second.num_ticks()
    keys = (*META_KEYS, "ancilla_budget")
    assert {k: first.get_meta(k) for k in keys} == {k: second.get_meta(k) for k in keys}
    for i in range(first.num_ticks()):
        assert {k: first.get_tick_meta(i, k) for k in PHASE_KEYS} == {k: second.get_tick_meta(i, k) for k in PHASE_KEYS}
        a = first.get_tick(i).gate_batches()
        b = second.get_tick(i).gate_batches()
        assert len(a) == len(b)
        for gate_index, (left, right) in enumerate(zip(a, b, strict=True)):
            for key in (
                "label",
                "role",
                "stabilizer",
                "stabilizer_kind",
                "stabilizer_index",
                "stabilizer_is_boundary",
                "stabilizer_region",
                "touch_label",
                "interaction_gate",
                "cx_round_0based",
            ):
                assert first.get_gate_meta(i, gate_index, key) == second.get_gate_meta(i, gate_index, key)
            for key in ("ancilla_qubit", "data_qubit"):
                value = first.get_gate_meta(i, gate_index, key)
                assert second.get_gate_meta(i, gate_index, key) == (None if value is None else value + 100)
            assert left.meas_ids == right.meas_ids
            assert left.gate_type == right.gate_type
            assert [int(q) + 100 for q in left.qubits] == [int(q) for q in right.qubits]


def test_compile_rendered_source(tmp_path: Path) -> None:
    """Load actual source using the file loader from test_program_fuzz.py:135."""
    source = render_surface_gadget_module(SurfacePatch.create(distance=3))
    path = tmp_path / "gadget_parity_module.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("gadget_parity_module", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        assert module.make_memory_z(2).compile() is not None
    finally:
        sys.modules.pop(spec.name, None)


def test_guppy_renderer_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    patch = SurfacePatch.create(distance=3)
    steps, allocation = build_surface_code_circuit(patch, 2, "Z")
    assert GuppyRenderer().render(steps, allocation, patch, 2, "Z") == render_surface_gadget_module(patch)

    calls = []

    def render_patch(actual_patch: SurfacePatch) -> str:
        calls.append(actual_patch)
        return render_surface_gadget_module(actual_patch)

    monkeypatch.setattr("pecos.guppy_gen.gadget_render.render_surface_gadget_module", render_patch)
    assert GuppyRenderer().render(steps, allocation, patch, 2, "Z") == render_surface_gadget_module(patch)
    assert calls == [patch]


def test_balanced_cx_plan_parity() -> None:
    patch = SurfacePatch.create(distance=3)
    steps, allocation = build_surface_code_circuit(patch, 2, "Z", check_plan="cx_balanced_data_v1")
    assert _ops(steps, allocation) == json.loads((GOLDENS / "ops_d3_Z_r2_balanced.json").read_text())


@pytest.mark.parametrize("basis", [None, "Z", "X"])
def test_gadget_round_order(basis: str | None) -> None:
    patch = SurfacePatch.create(distance=3)
    allocation = default_allocation(patch)
    order = (3, 1, 0, 2)
    if basis is None:
        gadget = syndrome_round_gadget(patch, allocation, round_index=0, round_order=order)
    else:
        gadget = init_syndrome_gadget(patch, allocation, basis=basis, round_order=order)
    actual_layers = []
    for step in gadget.steps:
        if step.op_type == OpType.COMMENT and step.label.startswith("CX round "):
            actual_layers.append([])
        elif step.op_type == OpType.CX:
            family = step.label[0]
            ancilla, data = step.qubits if family == "X" else reversed(step.qubits)
            ancillas = allocation.x_ancilla_qubits if family == "X" else allocation.z_ancilla_qubits
            actual_layers[-1].append((family, ancillas.index(ancilla), allocation.data_qubits.index(data)))
    default_layers = compute_cnot_schedule(patch)
    expected_layers = [[touch for touch in default_layers[i] if basis is None or touch[0] != basis] for i in order]
    assert actual_layers == expected_layers
    assert actual_layers != [
        [touch for touch in layer if basis is None or touch[0] != basis] for layer in default_layers
    ]


def test_builder_forwards_resolved_round_order(monkeypatch: pytest.MonkeyPatch) -> None:
    patch = SurfacePatch.create(distance=3)
    order = "round-order-3102-v1"
    standard, _ = build_surface_code_circuit(patch, 2, "Z")
    monkeypatch.setattr("pecos.qec.surface.circuit_builder.cnot_round_order_for_check_plan", lambda _plan: order)
    actual, allocation = build_surface_code_circuit(patch, 2, "Z")
    gadgets = memory_gadgets(patch, 2, "Z", allocation=allocation, round_order=order)
    assert actual == [step for gadget in gadgets for step in gadget.steps]
    assert actual != standard


@pytest.mark.parametrize("basis", ["z", "x"])
@pytest.mark.parametrize("name", ["custom_function", "custom_function_swapped"])
def test_gadget_basis_is_independent_of_name(basis: str, name: str) -> None:
    patch = SurfacePatch.create(distance=3)
    allocation = default_allocation(patch)
    gadgets = memory_gadgets(patch, 1, basis, allocation=allocation)
    assert [g.basis for g in gadgets] == [basis.upper(), basis.upper(), None, basis.upper()]
    logical = logical_pauli_gadget(patch, allocation, pauli=basis)
    assert logical.basis == basis.upper()
    for gadget in [*gadgets, logical]:
        renamed = replace(gadget, name=name)
        assert renamed.x_z_swapped is gadget.x_z_swapped is False
        expected = render_gadget_function(gadget)
        expected[1] = expected[1].replace(gadget.name, renamed.name)
        assert render_gadget_function(renamed) == expected


def test_measure_out_rejects_physical_steps_after_data_measurement() -> None:
    patch = SurfacePatch.create(distance=3)
    allocation = default_allocation(patch)
    gadget = measure_out_gadget(patch, allocation, basis="Z")
    ancilla = allocation.x_ancilla_qubits[0]
    gadget = replace(
        gadget,
        steps=(
            *gadget.steps,
            SurfaceCircuitStep(OpType.ALLOC, [ancilla], "ax0"),
            SurfaceCircuitStep(OpType.H, [ancilla], "ax0"),
            SurfaceCircuitStep(OpType.MEASURE, [ancilla], "sx0"),
        ),
    )
    with pytest.raises(ValueError, match=gadget.name):
        render_gadget_function(gadget)


@pytest.mark.parametrize("basis", [None, "Z", "X"])
def test_measurement_membership_uses_allocation(basis: str | None) -> None:
    patch = SurfacePatch.create(distance=3)
    allocation = default_allocation(patch)
    gadget = (
        syndrome_round_gadget(patch, allocation, round_index=0)
        if basis is None
        else init_syndrome_gadget(patch, allocation, basis=basis)
    )
    family = "z" if basis == "X" else "x"
    gadget = replace(
        gadget,
        steps=tuple(
            replace(step, label="custom") if step.op_type == OpType.MEASURE and step.label == f"s{family}0" else step
            for step in gadget.steps
        ),
    )
    lines = render_gadget_function(gadget)
    assert f"    custom = measure(a{family}0).read()" in lines
    results = f"array(custom, s{family}1, s{family}2, s{family}3)"
    if basis is None:
        assert f"    synx = {results}" in lines
        assert "    synz = array(sz0, sz1, sz2, sz3)" in lines
    else:
        assert f"    return {results}" in lines


def test_protocol_source_parity() -> None:
    """The post-fix protocol capture guards rendered source against drift."""
    assert (
        render_surface_protocol_module(SurfacePatch.create(distance=3)) == (GOLDENS / "protocol_d3.py.txt").read_text()
    )


def test_two_register_ancilla_names() -> None:
    patch = SurfacePatch.create(distance=3)
    ctrl = default_allocation(patch)
    tgt = QubitAllocation(
        [q + 100 for q in ctrl.data_qubits],
        [q + 100 for q in ctrl.x_ancilla_qubits],
        [q + 100 for q in ctrl.z_ancilla_qubits],
    )
    gadget = transversal_cx_gadget(patch, ctrl, patch, tgt)
    steps = []
    for scope, allocation in (("ctrl", ctrl), ("tgt", tgt)):
        for family, ancillas in (("x", allocation.x_ancilla_qubits), ("z", allocation.z_ancilla_qubits)):
            for i, ancilla in enumerate(ancillas):
                steps.extend(
                    [
                        SurfaceCircuitStep(OpType.ALLOC, [ancilla]),
                        SurfaceCircuitStep(OpType.H, [ancilla]),
                        SurfaceCircuitStep(OpType.CX, [ancilla, allocation.data_qubits[i]]),
                        SurfaceCircuitStep(OpType.MEASURE, [ancilla], f"{scope}_s{family}{i}"),
                    ],
                )
    lines = render_gadget_function(replace(gadget, steps=tuple(steps)))
    allocated = [line.strip().split(" = ")[0] for line in lines if " = qubit()" in line]
    assert allocated == [
        f"{scope}_a{family}{i}" for scope in ("ctrl", "tgt") for family in ("x", "z") for i in range(4)
    ]
    assert len(set(allocated)) == len(allocated)
    for scope in ("ctrl", "tgt"):
        for family in ("x", "z"):
            for i in range(4):
                name = f"{scope}_a{family}{i}"
                assert f"    h({name})" in lines
                assert f"    cx({name}, {scope}.data[{i}])" in lines
                assert f"    {scope}_s{family}{i} = measure({name}).read()" in lines


def test_unsupported_gadget_operation() -> None:
    patch = SurfacePatch.create(distance=3)
    allocation = default_allocation(patch)
    gadget = logical_pauli_gadget(patch, allocation, pauli="X")
    gadget = replace(gadget, steps=(SurfaceCircuitStep(OpType.F, [allocation.data_qubits[0]]),))
    with pytest.raises(ValueError, match="Unsupported gadget operation: F"):
        render_gadget_function(gadget)


def test_data_allocation_requires_prep() -> None:
    patch = SurfacePatch.create(distance=3)
    allocation = default_allocation(patch)
    gadget = logical_pauli_gadget(patch, allocation, pauli="X")
    gadget = replace(gadget, steps=tuple(SurfaceCircuitStep(OpType.ALLOC, [q]) for q in allocation.data_qubits))
    with pytest.raises(ValueError, match="Data allocation requires a preparation gadget"):
        render_gadget_function(gadget)


@pytest.mark.parametrize("operation", [OpType.ALLOC, OpType.MEASURE])
@pytest.mark.parametrize("qubits", [[0], list(reversed(range(9)))])
def test_incomplete_data_run(operation, qubits) -> None:
    patch = SurfacePatch.create(distance=3)
    gadget = measure_out_gadget(patch, default_allocation(patch), basis="Z")
    gadget = replace(gadget, steps=tuple(SurfaceCircuitStep(operation, [q]) for q in qubits))
    with pytest.raises(ValueError, match=f"{operation.name} must cover all data in register order"):
        render_gadget_function(gadget)


@pytest.mark.parametrize("within_register", [False, True])
def test_non_disjoint_allocations(within_register) -> None:
    patch = SurfacePatch.create(distance=3)
    allocation = default_allocation(patch)
    gadget = syndrome_round_gadget(patch, allocation, round_index=0)
    if within_register:
        bad = replace(allocation, x_ancilla_qubits=[allocation.data_qubits[0], *allocation.x_ancilla_qubits[1:]])
        gadget = replace(gadget, allocations=(bad,))
    else:
        # Construct a valid two-patch gadget first, then inject overlapping IDs.
        target = QubitAllocation([q + 100 for q in allocation.data_qubits], [], [])
        gadget = transversal_cx_gadget(patch, allocation, patch, target)
        gadget = replace(gadget, allocations=(allocation, allocation))
    with pytest.raises(ValueError, match="allocations must be disjoint"):
        render_gadget_function(gadget)
