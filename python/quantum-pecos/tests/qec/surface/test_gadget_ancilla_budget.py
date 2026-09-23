# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Ancilla lifetimes against the builder captured before gadget delegation."""

import gzip
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from pecos.guppy_gen import get_num_qubits
from pecos.guppy_gen.gadget_render import render_gadget_function, render_surface_gadget_module
from pecos.qec.surface import SurfacePatch, gadgets
from pecos.qec.surface._check_plan import (
    ancilla_schedule_for_check_plan,
    cnot_round_order_for_check_plan,
    resolve_surface_check_plan,
)
from pecos.qec.surface.circuit_builder import (
    OpType,
    QubitAllocation,
    SurfaceCircuitStep,
    TickCircuitRenderer,
    build_surface_code_circuit,
)

# Captured to /tmp/pecos-slice5a-original.json before editing the builder or gadgets.
# Compression keeps the complete element-wise oracle small without deriving it from the new path.
CASES = json.loads(gzip.decompress((Path(__file__).parent / "fixtures/ancilla_budget_builder.json.gz").read_bytes()))


@pytest.mark.parametrize("case", CASES, ids=lambda case: "-".join(map(str, case["shape"])))
def test_original_builder_parity(case: dict) -> None:
    distance, rounds, basis, budget, plan = case["shape"]
    patch = SurfacePatch.create(distance=distance)
    resolved = resolve_surface_check_plan(check_plan=plan)
    parts = gadgets.memory_gadgets(
        patch,
        rounds,
        basis,
        ancilla_budget=budget,
        ancilla_schedule=ancilla_schedule_for_check_plan(resolved),
        round_order=cnot_round_order_for_check_plan(resolved),
    )
    expected = case["output"]
    allocation = QubitAllocation(*expected["allocation"])
    steps = [SurfaceCircuitStep(OpType[op], qubits, label) for op, qubits, label in expected["steps"]]
    assert all(part.allocations == (allocation,) for part in parts)
    assert [step for part in parts for step in part.steps] == steps
    assert build_surface_code_circuit(patch, rounds, basis, ancilla_budget=budget, check_plan=plan) == (
        steps,
        allocation,
    )


@pytest.mark.parametrize("distance", [3, 5])
@pytest.mark.parametrize("budget_kind", ["one", "two", "half", "total_minus_one", "total"])
@pytest.mark.parametrize("schedule", ["default", "balanced-data-v1"])
def test_tick_peak_matches_legacy_oracle(distance: int, budget_kind: str, schedule: str) -> None:
    total = distance * distance - 1
    budget = {"one": 1, "two": 2, "half": total // 2, "total_minus_one": total - 1, "total": total}[budget_kind]
    patch = SurfacePatch.create(distance=distance)
    parts = gadgets.memory_gadgets(patch, 2, "Z", ancilla_budget=budget, ancilla_schedule=schedule)
    allocation = parts[0].allocations[0]
    circuit = TickCircuitRenderer().render([s for g in parts for s in g.steps], allocation, patch, 2, "Z")
    live = set()
    peak = 0
    for tick_index in range(circuit.num_ticks()):
        for gate in circuit.get_tick(tick_index).gate_batches():
            qubits = set(gate.qubits)
            if gate.gate_type.name == "QAlloc":
                assert not live & qubits
                live.update(qubits)
                peak = max(peak, len(live))
            elif gate.gate_type.name == "MeasureFree":
                assert qubits <= live
                live.difference_update(qubits)
            else:
                assert qubits <= live
    if schedule == "default":
        assert peak == get_num_qubits(distance, ancilla_budget=budget)
    else:
        # Balanced batches can leave part of the requested capacity unused.
        assert peak == allocation.total
        assert peak <= get_num_qubits(distance, ancilla_budget=budget)


@pytest.mark.parametrize("plan", ["cx_standard_v1", "cx_balanced_data_v1"])
@pytest.mark.parametrize("budget", [1, 2])
def test_guppy_budget_epochs(tmp_path: Path, plan: str, budget: int) -> None:
    patch = SurfacePatch.create(distance=3)
    schedule = ancilla_schedule_for_check_plan(resolve_surface_check_plan(check_plan=plan))
    parts = gadgets.memory_gadgets(patch, 1, "Z", ancilla_budget=budget, ancilla_schedule=schedule)
    for part in parts[1:-1]:
        lines = render_gadget_function(part)
        allocations = [s for s in part.steps if s.op_type == OpType.ALLOC]
        assert [line.strip() for line in lines if " = qubit()" in line] == [f"{s.label} = qubit()" for s in allocations]
        measures = [s for s in part.steps if s.op_type == OpType.MEASURE]
        tag = "init:meas" if part.kind == gadgets.GadgetKind.INIT_SYNDROME else "meas"
        assert [line.strip() for line in lines if "output(" in line] == [
            f'output("{s.label}:{tag}:{i}", {s.label})' for i, s in enumerate(measures)
        ]
        for s in measures:
            assert f"    {s.label} = measure(a{s.label[1:]}).read()" in lines
        if part.kind == gadgets.GadgetKind.SYNDROME_ROUND:
            assert "    synx = array(sx0, sx1, sx2, sx3)" in lines
            assert "    synz = array(sz0, sz1, sz2, sz3)" in lines
    source = render_surface_gadget_module(patch, ancilla_budget=budget, check_plan=plan)
    path = tmp_path / "budget_module.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("budget_module", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        assert module.make_memory_z(2).compile() is not None
        assert module.make_memory_x(2).compile() is not None
    finally:
        sys.modules.pop(spec.name, None)


def test_repeated_register_slot_starts_fresh_epoch() -> None:
    patch = SurfacePatch.create(distance=3)
    allocation = gadgets.default_allocation(patch)
    part = gadgets.syndrome_round_gadget(patch, allocation, round_index=0)
    qubit = allocation.x_ancilla_qubits[0]
    steps = (
        SurfaceCircuitStep(OpType.ALLOC, [qubit], "ax0"),
        SurfaceCircuitStep(OpType.MEASURE, [qubit], "first"),
        SurfaceCircuitStep(OpType.ALLOC, [qubit], "ax0"),
        SurfaceCircuitStep(OpType.MEASURE, [qubit], "second"),
    )
    lines = render_gadget_function(replace(part, steps=steps))
    assert "    ax0 = qubit()" in lines
    assert "    ax0_1 = qubit()" in lines
    assert "    second = measure(ax0_1).read()" in lines
    with pytest.raises(ValueError, match="allocations must be disjoint within a live epoch"):
        render_gadget_function(replace(part, steps=(steps[0], steps[2])))


@pytest.mark.parametrize("budget", [0, -1, 1.5, True, "2"])
@pytest.mark.parametrize("entry", ["allocation", "memory", "init", "syndrome", "module"])
def test_invalid_ancilla_budget_by_name(budget: object, entry: str) -> None:
    patch = SurfacePatch.create(distance=3)
    allocation = gadgets.default_allocation(patch)
    calls = {
        "allocation": lambda: gadgets.default_allocation(patch, ancilla_budget=budget),
        "memory": lambda: gadgets.memory_gadgets(patch, 1, "Z", ancilla_budget=budget),
        "init": lambda: gadgets.init_syndrome_gadget(patch, allocation, basis="Z", ancilla_budget=budget),
        "syndrome": lambda: gadgets.syndrome_round_gadget(patch, allocation, round_index=0, ancilla_budget=budget),
        "module": lambda: render_surface_gadget_module(patch, ancilla_budget=budget),
    }
    with pytest.raises((TypeError, ValueError), match="ancilla_budget"):
        calls[entry]()


def test_non_cx_check_plan_by_name() -> None:
    with pytest.raises(ValueError, match=r"check_plan 'szz_current_v1'.*ancilla_budget"):
        render_surface_gadget_module(SurfacePatch.create(distance=3), ancilla_budget=2, check_plan="szz_current_v1")


def test_budget_requires_matching_allocation() -> None:
    patch = SurfacePatch.create(distance=3)
    with pytest.raises(ValueError, match="allocation exceeds ancilla_budget"):
        gadgets.memory_gadgets(patch, 1, "Z", allocation=gadgets.default_allocation(patch), ancilla_budget=2)
    allocation = gadgets.default_allocation(patch, ancilla_budget=2)
    invalid = replace(allocation, x_ancilla_qubits=[9] * 4, z_ancilla_qubits=[9] * 4)
    with pytest.raises(ValueError, match="ancilla_schedule requires disjoint allocation"):
        gadgets.memory_gadgets(patch, 1, "Z", allocation=invalid, ancilla_budget=2)
