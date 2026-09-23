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
from pecos.qec.surface import SurfacePatch, TwirlConfig, gadgets
from pecos.qec.surface._check_plan import (
    ancilla_schedule_for_check_plan,
    cnot_round_order_for_check_plan,
    resolve_surface_check_plan,
)
from pecos.qec.surface.circuit_builder import (
    GuppyRenderer,
    OpType,
    QubitAllocation,
    SurfaceCircuitStep,
    TickCircuitRenderer,
    build_surface_code_circuit,
)

# Captured from pre-change commit a890a4287. Regenerate only from that commit, not this branch.
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


def _oriented_memory(
    patch: SurfacePatch,
    rounds: int,
    basis: str,
    budget: int,
    schedule: str,
    *,
    x_z_swapped: bool,
) -> list[gadgets.Gadget]:
    allocation = gadgets.default_allocation(patch, ancilla_budget=budget, ancilla_schedule=schedule)
    return [
        gadgets.prep_gadget(patch, allocation, basis=basis),
        gadgets.init_syndrome_gadget(
            patch,
            allocation,
            basis=basis,
            ancilla_budget=budget,
            ancilla_schedule=schedule,
            x_z_swapped=x_z_swapped,
        ),
        *(
            gadgets.syndrome_round_gadget(
                patch,
                allocation,
                round_index=index,
                ancilla_budget=budget,
                ancilla_schedule=schedule,
                x_z_swapped=x_z_swapped,
            )
            for index in range(rounds)
        ),
        gadgets.measure_out_gadget(patch, allocation, basis=basis),
    ]


@pytest.mark.parametrize("distance", [3, 5])
@pytest.mark.parametrize("budget_kind", ["one", "two", "half", "total_minus_one", "total"])
@pytest.mark.parametrize("schedule", ["default", "balanced-data-v1"])
@pytest.mark.parametrize("x_z_swapped", [False, True])
@pytest.mark.parametrize("basis", ["Z", "X"])
def test_tick_peak_matches_legacy_oracle(
    distance: int,
    budget_kind: str,
    schedule: str,
    x_z_swapped: bool,
    basis: str,
) -> None:
    total = distance * distance - 1
    budget = {"one": 1, "two": 2, "half": total // 2, "total_minus_one": total - 1, "total": total}[budget_kind]
    patch = SurfacePatch.create(distance=distance)
    parts = _oriented_memory(patch, 2, basis, budget, schedule, x_z_swapped=x_z_swapped)
    allocation = parts[0].allocations[0]
    circuit = TickCircuitRenderer(add_detectors=False).render(
        [s for g in parts for s in g.steps],
        allocation,
        patch,
        2,
        basis,
    )
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
@pytest.mark.parametrize("x_z_swapped", [False, True])
@pytest.mark.parametrize("basis", ["Z", "X"])
def test_guppy_budget_epochs(tmp_path: Path, plan: str, budget: int, x_z_swapped: bool, basis: str) -> None:
    patch = SurfacePatch.create(distance=3)
    schedule = ancilla_schedule_for_check_plan(resolve_surface_check_plan(check_plan=plan))
    parts = _oriented_memory(patch, 1, basis, budget, schedule, x_z_swapped=x_z_swapped)
    for part in parts[1:-1]:
        lines = render_gadget_function(part)
        allocations = [s for s in part.steps if s.op_type == OpType.ALLOC]
        assert [line.strip() for line in lines if " = qubit()" in line] == [f"{s.label} = qubit()" for s in allocations]
        h_prefix = "az" if x_z_swapped else "ax"
        expected_h = [s.label for s in allocations if s.label.startswith(h_prefix)]
        actual_h = [s.label for s in part.steps if s.op_type == OpType.H]
        assert sorted(actual_h) == sorted(expected_h * 2)
        measures = [s for s in part.steps if s.op_type == OpType.MEASURE]
        tag = "init:meas" if part.kind == gadgets.GadgetKind.INIT_SYNDROME else "meas"
        scope = "swapped:" if x_z_swapped else ""
        assert [line.strip() for line in lines if "output(" in line] == [
            f'output("{scope}{s.label}:{tag}:{i}", {s.label})' for i, s in enumerate(measures)
        ]
        for s in measures:
            assert f"    {s.label} = measure(a{s.label[1:]}).read()" in lines
        if part.kind == gadgets.GadgetKind.SYNDROME_ROUND:
            x_family, z_family = ("z", "x") if x_z_swapped else ("x", "z")
            for field, family in (("synx", x_family), ("synz", z_family)):
                assert f"    {field} = array({', '.join(f's{family}{i}' for i in range(4))})" in lines
    source = render_surface_gadget_module(patch, ancilla_budget=budget, check_plan=plan)
    if x_z_swapped:
        source += "\n\n" + "\n\n".join("\n".join(render_gadget_function(part)) for part in parts[1:-1]) + "\n"
    path = tmp_path / "budget_module.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("budget_module", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        assert module.make_memory_z(2).compile() is not None
        assert module.make_memory_x(2).compile() is not None
        if x_z_swapped:
            assert getattr(module, f"init_{basis.lower()}_basis_swapped").compile_function() is not None
            assert module.syndrome_extraction_swapped.compile_function() is not None
    finally:
        sys.modules.pop(spec.name, None)


def test_repeated_live_allocation_is_rejected() -> None:
    patch = SurfacePatch.create(distance=3)
    allocation = gadgets.default_allocation(patch)
    part = gadgets.syndrome_round_gadget(patch, allocation, round_index=0)
    step = SurfaceCircuitStep(OpType.ALLOC, [allocation.x_ancilla_qubits[0]], "ax0")
    with pytest.raises(ValueError, match="allocations must be disjoint within a live epoch"):
        render_gadget_function(replace(part, steps=(step, step)))


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
    with pytest.raises(ValueError, match="allocation does not match ancilla_budget"):
        gadgets.memory_gadgets(patch, 1, "Z", allocation=gadgets.default_allocation(patch), ancilla_budget=2)
    allocation = gadgets.default_allocation(patch, ancilla_budget=2)
    invalid = replace(allocation, x_ancilla_qubits=[9] * 4, z_ancilla_qubits=[9] * 4)
    with pytest.raises(ValueError, match=r"allocation.*ancilla_schedule"):
        gadgets.memory_gadgets(patch, 1, "Z", allocation=invalid, ancilla_budget=2)


@pytest.mark.parametrize("entry", ["init", "syndrome", "memory"])
@pytest.mark.parametrize("mismatch", ["budget", "schedule"])
def test_allocation_must_match_resolved_budget_and_schedule(entry: str, mismatch: str) -> None:
    patch = SurfacePatch.create(distance=3)
    schedule = "balanced-data-v1" if mismatch == "schedule" else "default"
    allocation = gadgets.default_allocation(patch, ancilla_budget=2, ancilla_schedule=schedule)
    kwargs = {"ancilla_budget": 2} if mismatch == "schedule" else {}
    calls = {
        "init": lambda: gadgets.init_syndrome_gadget(patch, allocation, basis="Z", **kwargs),
        "syndrome": lambda: gadgets.syndrome_round_gadget(patch, allocation, round_index=0, **kwargs),
        "memory": lambda: gadgets.memory_gadgets(patch, 1, "Z", allocation=allocation, **kwargs),
    }
    with pytest.raises(ValueError, match=f"allocation.*ancilla_{mismatch}"):
        calls[entry]()


@pytest.mark.parametrize("schedule", ["default", "balanced-data-v1"])
@pytest.mark.parametrize("budget", [2, None])
def test_matching_allocation_allows_physical_id_remapping(schedule: str, budget: int | None) -> None:
    patch = SurfacePatch.create(distance=3)
    original = gadgets.default_allocation(patch, ancilla_budget=budget, ancilla_schedule=schedule)
    shifted = QubitAllocation(
        *[
            [100 + 3 * q for q in register]
            for register in (original.data_qubits, original.x_ancilla_qubits, original.z_ancilla_qubits)
        ],
    )
    canonical = gadgets.memory_gadgets(
        patch,
        1,
        "Z",
        allocation=original,
        ancilla_budget=budget,
        ancilla_schedule=schedule,
    )
    relocated = gadgets.memory_gadgets(
        patch,
        1,
        "Z",
        allocation=shifted,
        ancilla_budget=budget,
        ancilla_schedule=schedule,
    )
    for original_part, relocated_part in zip(canonical, relocated, strict=True):
        render_gadget_function(original_part)
        render_gadget_function(relocated_part)
        assert (
            tuple(replace(step, qubits=[100 + 3 * q for q in step.qubits]) for step in original_part.steps)
            == relocated_part.steps
        )


@pytest.mark.parametrize("case", CASES, ids=lambda case: "-".join(map(str, case["shape"])))
@pytest.mark.parametrize("site_schedule", ["between_rounds", "before_two_qubit_gate"])
def test_twirled_builder_matches_budgeted_gadgets(case: dict, site_schedule: str) -> None:
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
    steps, allocation = build_surface_code_circuit(
        patch,
        rounds,
        basis,
        ancilla_budget=budget,
        check_plan=plan,
        twirl=TwirlConfig(site_schedule=site_schedule),
    )
    assert allocation == parts[0].allocations[0]
    assert [s for s in steps if s.op_type != OpType.TRACKED_PAULI] == [s for g in parts for s in g.steps]


@pytest.mark.parametrize("plan", ["cx_standard_v1", "cx_balanced_data_v1"])
def test_guppy_renderer_rejects_budgeted_allocation(plan: str) -> None:
    patch = SurfacePatch.create(distance=3)
    steps, allocation = build_surface_code_circuit(patch, 1, "Z", ancilla_budget=2, check_plan=plan)
    with pytest.raises(ValueError, match=r"GuppyRenderer.*ancilla_budget"):
        GuppyRenderer().render(steps, allocation, patch, 1, "Z")


@pytest.mark.parametrize(
    ("budget", "plan", "description"),
    [
        (None, "cx_standard_v1", "8 (one per stabilizer)"),
        (2, "cx_standard_v1", "2 (reused)"),
        (7, "cx_balanced_data_v1", "4 (reused)"),
    ],
)
def test_ancilla_description_and_budget_comment(budget: int | None, plan: str, description: str) -> None:
    source = render_surface_gadget_module(SurfacePatch.create(distance=3), ancilla_budget=budget, check_plan=plan)
    assert f"Ancilla qubits: {description}\n" in source
    if budget is None:
        assert "# Allocate ancilla qubits (one per stabilizer)" in source
        assert "# Reuse ancillas in batches" not in source
    else:
        assert "# Reuse ancillas in batches to respect the live-qubit budget" in source
        assert "# Allocate ancilla qubits (one per stabilizer)" not in source


@pytest.mark.parametrize("label", ["", "ax99", "unidentified"])
def test_allocation_label_must_identify_pool_slot(label: str) -> None:
    part = gadgets.memory_gadgets(SurfacePatch.create(distance=3), 1, "Z", ancilla_budget=1)[2]
    step = next(s for s in part.steps if s.op_type == OpType.ALLOC)
    with pytest.raises(ValueError, match=r"ALLOC label.*must identify an ancilla register slot"):
        render_gadget_function(replace(part, steps=(replace(step, label=label),)))


def test_measurement_requires_live_allocation() -> None:
    part = gadgets.memory_gadgets(SurfacePatch.create(distance=3), 1, "Z", ancilla_budget=1)[2]
    step = next(s for s in part.steps if s.op_type == OpType.MEASURE)
    with pytest.raises(ValueError, match="measurement requires a live ancilla allocation"):
        render_gadget_function(replace(part, steps=(step,)))


@pytest.mark.parametrize("op", [OpType.H, OpType.CX])
@pytest.mark.parametrize("two_registers", [False, True])
def test_use_after_measurement_has_no_live_name(op: OpType, two_registers: bool) -> None:
    patch = SurfacePatch.create(distance=3)
    part = gadgets.memory_gadgets(patch, 1, "Z", ancilla_budget=1)[2]
    alloc = next(s for s in part.steps if s.op_type == OpType.ALLOC)
    measure = next(s for s in part.steps if s.op_type == OpType.MEASURE)
    qubit = alloc.qubits[0]
    qubits = [qubit] if op == OpType.H else [part.allocations[0].data_qubits[0], qubit]
    if two_registers:
        allocation = part.allocations[0]
        target = QubitAllocation(
            *[
                [100 + q for q in register]
                for register in (allocation.data_qubits, allocation.x_ancilla_qubits, allocation.z_ancilla_qubits)
            ],
        )
        part = replace(
            part,
            kind=gadgets.GadgetKind.TWO_PATCH,
            allocations=(*part.allocations, target),
        )
    with pytest.raises(ValueError, match=rf"{part.name}: measured ancilla {qubit} has no live allocation"):
        render_gadget_function(replace(part, steps=(alloc, measure, SurfaceCircuitStep(op, qubits))))


@pytest.mark.parametrize("entry", ["init", "syndrome"])
@pytest.mark.parametrize("budget", [2, None])
def test_swapped_rectangle_rejected_before_emission(
    monkeypatch: pytest.MonkeyPatch,
    entry: str,
    budget: int | None,
) -> None:
    patch = SurfacePatch.create(dx=3, dz=5)
    allocation = gadgets.default_allocation(patch, ancilla_budget=budget)

    def unexpected_emission(*_args, **_kwargs):
        pytest.fail("swapped rectangle reached step construction")

    monkeypatch.setattr(gadgets, "_budgeted_syndrome_steps", unexpected_emission)
    calls = {
        "init": lambda: gadgets.init_syndrome_gadget(
            patch,
            allocation,
            basis="Z",
            x_z_swapped=True,
            ancilla_budget=budget,
        ),
        "syndrome": lambda: gadgets.syndrome_round_gadget(
            patch,
            allocation,
            round_index=0,
            x_z_swapped=True,
            ancilla_budget=budget,
        ),
    }
    with pytest.raises(ValueError, match=f"{entry}.*requires a square patch when x_z_swapped"):
        calls[entry]()


@pytest.mark.parametrize("budget", [None, 1])
def test_empty_ancilla_allocation(budget: int | None) -> None:
    parts = gadgets.memory_gadgets(SurfacePatch.create(distance=1), 1, "Z", ancilla_budget=budget)
    assert all(part.allocations == (QubitAllocation([0], [], []),) for part in parts)
    assert all(step.op_type != OpType.ALLOC for part in parts[1:-1] for step in part.steps)


def test_builder_empty_ancilla_allocation() -> None:
    patch = SurfacePatch.create(distance=1)
    circuits = [build_surface_code_circuit(patch, 2, "Z", ancilla_budget=budget) for budget in (None, 1, 2)]
    assert circuits[0] == circuits[1] == circuits[2]
    steps, allocation = circuits[0]
    assert len(steps) == 47
    assert allocation == QubitAllocation([0], [], [])
    with pytest.raises(ValueError, match="ancilla_budget must be >= 1"):
        build_surface_code_circuit(patch, 2, "Z", ancilla_budget=0)
