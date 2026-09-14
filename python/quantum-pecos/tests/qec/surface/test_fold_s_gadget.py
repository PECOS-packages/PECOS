# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Signed half-cycle and round-flow oracles for the Cartesian fold S gadget."""

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import stim
from pecos.guppy_gen.gadget_render import render_gadget_function, render_surface_gadget_module
from pecos.qec.surface import SurfacePatch, gadgets
from pecos.qec.surface.circuit_builder import (
    DagCircuitRenderer,
    GuppyRenderer,
    OpType,
    QubitAllocation,
    StimRenderer,
    SurfaceCircuitStep,
    TickCircuitRenderer,
    tick_circuit_to_stim,
)
from pecos.quantum import TickCircuit
from pecos.testing import group_contains, stabilizer_generators_after


def _render(patch: SurfacePatch, allocation: QubitAllocation, parts: list[gadgets.Gadget]) -> TickCircuit:
    return TickCircuitRenderer(add_detectors=False).render(
        [step for part in parts for step in part.steps],
        allocation,
        patch,
        2,
        "Z",
    )


def _replayable(source: TickCircuit, start: int = 0, stop: int | None = None) -> TickCircuit:
    """Retain measured ancillas: the oracle supports MZ but not MeasureFree.

    Reallocation resets the retained qubit, so this preserves both the data
    channel and sampled measurement branches of the rendered gadget stream.
    """
    result = TickCircuit()
    for index in range(start, source.num_ticks() if stop is None else stop):
        tick = result.tick()
        for gate in source.get_tick(index).gate_batches():
            if gate.gate_type.name in {"MZ", "MeasureFree"}:
                tick.mz(list(gate.qubits))
            else:
                tick.add_gate(gate.gate_type.name, list(gate.qubits))
    return result


def _pauli(n: int, axis: str, support: list[int]) -> str:
    return "+" + "".join(axis if q in support else "I" for q in range(n))


def _unsigned_contains(group: tuple[str, ...], pauli: str) -> bool:
    return group_contains(group, "+" + pauli[1:]) or group_contains(group, "-" + pauli[1:])


def _rank(group: tuple[str, ...]) -> int:
    """Use the signed membership oracle to select an unsigned independent basis."""
    basis: tuple[str, ...] = ()
    for row in group:
        if not _unsigned_contains(basis, row):
            basis += (row,)
    return len(basis)


def _coordinates(patch: SurfacePatch, allocation: QubitAllocation) -> dict[int, tuple[int, int]]:
    """Independent coordinate formula; locate bulk sites by support bounds."""
    d = patch.dx
    data = {i: (2 * (i % d) + 1, 2 * (d - i // d) - 1) for i in range(d * d)}
    coords = {allocation.data_qubits[i]: xy for i, xy in data.items()}
    for checks, ancillas in (
        (patch.geometry.x_stabilizers, allocation.x_ancilla_qubits),
        (patch.geometry.z_stabilizers, allocation.z_ancilla_qubits),
    ):
        for check in checks:
            if len(check.data_qubits) == 4:
                xs, ys = zip(*(data[q] for q in check.data_qubits), strict=True)
                coords[ancillas[check.index]] = ((min(xs) + max(xs)) // 2, (min(ys) + max(ys)) // 2)
    return coords


def _groups(gadget: gadgets.Gadget) -> list[list[SurfaceCircuitStep]]:
    groups: list[list[SurfaceCircuitStep]] = [[]]
    for step in gadget.steps:
        if step.op_type == OpType.TICK:
            groups.append([])
        elif step.op_type != OpType.COMMENT:
            groups[-1].append(step)
    return groups[:-1]


def _half_cycle(patch: SurfacePatch, round_gadget: gadgets.Gadget) -> tuple[TickCircuit, int]:
    allocation = round_gadget.allocations[0]
    parts = gadgets.memory_gadgets(patch, 2, "Z", allocation=allocation)
    parts[-2] = round_gadget
    tc = _replayable(_render(patch, allocation, parts))
    cx_ticks = [
        i for i in range(tc.num_ticks()) if any(g.gate_type.name == "CX" for g in tc.get_tick(i).gate_batches())
    ]
    # Complementary initialization and round 0 each contribute four CX layers.
    return tc, cx_ticks[9] + 1


@pytest.mark.parametrize("d", [3, 5])
def test_half_cycle_symmetry(d: int) -> None:
    """Transpose is two-way on the code subgroup, excluding the logical generator.

    The pure Z-memory state's Z subgroup has one extra generator: logical Z.
    It equals the span of the transposed X subgroup and propagated logical Z.
    Exterior ancillas factor out as single-qubit stabilizers.
    """
    patch = SurfacePatch.create(distance=d)
    allocation = gadgets.default_allocation(patch)
    tc, half = _half_cycle(patch, gadgets.syndrome_round_gadget(patch, allocation, round_index=1))
    before = stabilizer_generators_after(tc, half)
    coords = _coordinates(patch, allocation)
    exterior = set(range(allocation.total)) - coords.keys()
    for q in exterior:
        axis = "X" if q in allocation.x_ancilla_qubits else "Z"
        assert _unsigned_contains(before, _pauli(allocation.total, axis, [q]))
    block = tuple(row[0] + "".join("I" if q in exterior else p for q, p in enumerate(row[1:])) for row in before)
    assert all(set(row[1:]) <= set("IX") or set(row[1:]) <= set("IZ") for row in block)
    xs = tuple(row for row in block if "X" in row)
    zs = tuple(row for row in block if "Z" in row)
    inverse = {xy: q for q, xy in coords.items()}
    mirrored = tuple(
        _pauli(allocation.total, "Z", [inverse[coords[q][::-1]] for q, p in enumerate(row[1:]) if p == "X"])
        for row in xs
    )
    assert all(_unsigned_contains(before, row) for row in mirrored)
    logical_z = stim.PauliString(_pauli(allocation.total, "Z", list(patch.geometry.logical_z.data_qubits)))
    logical_z = logical_z.after(stim.Circuit(tick_circuit_to_stim(_replayable(tc, half - 2, half))))
    logical = str(logical_z).replace("_", "I")
    span = (*mirrored, logical)
    assert all(_unsigned_contains(zs, row) for row in span)
    assert all(_unsigned_contains(span, row) for row in zs)
    assert _rank(zs) == _rank(mirrored) + 1 == (d * d + (d - 1) ** 2 + 1) // 2


@pytest.mark.parametrize("d", [3, 5])
@pytest.mark.parametrize("dagger", [False, True])
def test_fold_preserves_signed_group(d: int, *, dagger: bool) -> None:
    """Both alternating assignments preserve Z memory; one ancilla flip does not."""
    patch = SurfacePatch.create(distance=d)
    allocation = gadgets.default_allocation(patch)
    gadget = gadgets.fold_s_round_gadget(patch, allocation, round_index=1, dagger=dagger)
    tc, half = _half_cycle(patch, gadget)
    for seed in range(4):
        before = stabilizer_generators_after(tc, half, seed=seed)
        after = stabilizer_generators_after(tc, half + 1, seed=seed)
        assert all(group_contains(before, row) for row in after)
        assert all(group_contains(after, row) for row in before)

    fixed = next(
        s for s in gadget.steps if s.op_type in {OpType.SZ, OpType.SZDG} and s.qubits[0] not in allocation.data_qubits
    )
    flipped = replace(fixed, op_type=OpType.SZDG if fixed.op_type == OpType.SZ else OpType.SZ)
    mutant = replace(gadget, steps=tuple(flipped if s is fixed else s for s in gadget.steps))
    mutated, half = _half_cycle(patch, mutant)
    before = stabilizer_generators_after(mutated, half)
    after = stabilizer_generators_after(mutated, half + 1)
    assert not all(group_contains(before, row) for row in after)


@pytest.mark.parametrize("d", [3, 5])
@pytest.mark.parametrize("swapped", [False, True])
@pytest.mark.parametrize("dagger", [False, True])
def test_round_flow(d: int, *, swapped: bool, dagger: bool) -> None:
    """Pin the exact signed channel and the measured Z-syndrome frame in either orientation."""
    patch = SurfacePatch.create(distance=d)
    allocation = gadgets.default_allocation(patch)
    gadget = gadgets.fold_s_round_gadget(patch, allocation, round_index=1, x_z_swapped=swapped, dagger=dagger)
    tc = _render(patch, allocation, [gadget])
    circuit = stim.Circuit(tick_circuit_to_stim(tc))
    geom = patch.geometry
    x_support = geom.logical_z.data_qubits if swapped else geom.logical_x.data_qubits
    z_support = geom.logical_x.data_qubits if swapped else geom.logical_z.data_qubits
    xl = stim.PauliString(_pauli(allocation.total, "X", list(x_support)))
    zl = stim.PauliString(_pauli(allocation.total, "Z", list(z_support)))
    yl = stim.PauliString("i") * xl * zl
    if dagger:
        yl = -yl
    z_checks = geom.x_stabilizers if swapped else geom.z_stabilizers
    product = stim.PauliString(allocation.total)
    for check in z_checks:
        product *= stim.PauliString(_pauli(allocation.total, "Z", list(check.data_qubits)))
    assert circuit.has_flow(stim.Flow(input=xl, output=yl * product))
    assert not circuit.has_flow(stim.Flow(input=xl, output=-yl * product))
    assert circuit.has_flow(stim.Flow(input=zl, output=zl))
    z_records = list(range(len(z_checks), 2 * len(z_checks)))
    assert circuit.has_flow(stim.Flow(input=xl, output=yl, measurements=z_records))
    for check, record in zip(z_checks, z_records, strict=True):
        operator = stim.PauliString(_pauli(allocation.total, "Z", list(check.data_qubits)))
        assert circuit.has_flow(stim.Flow(output=operator, measurements=[record]))

    # Prepare current +X through actual H and swapped rounds, then certify the Y frame.
    parts = gadgets.memory_gadgets(patch, 1, "Z" if swapped else "X")[:-1]
    if swapped:
        parts.append(gadgets.transversal_layer_gadget(patch, allocation, gate="H"))
        parts.append(gadgets.syndrome_round_gadget(patch, allocation, round_index=0, x_z_swapped=True))
    prefix = _render(patch, allocation, parts)
    full = _render(patch, allocation, [*parts, gadget])
    offset = prefix.num_measurements()
    assert stim.Circuit(tick_circuit_to_stim(full)).has_flow(
        stim.Flow(output=yl, measurements=[offset + r for r in z_records]),
    )


@pytest.mark.parametrize("d", [3, 5])
@pytest.mark.parametrize("swapped", [False, True])
@pytest.mark.parametrize("remap", [False, True])
def test_fold_structure(d: int, *, swapped: bool, remap: bool) -> None:
    """There are (d-1)^2 CZ pairs, d data fixed points, and d-1 ancilla fixed points."""
    patch = SurfacePatch.create(distance=d)
    allocation = gadgets.default_allocation(patch)
    if remap:
        allocation = QubitAllocation(
            *[
                [100 + 3 * q for q in reversed(register)]
                for register in (
                    allocation.data_qubits,
                    allocation.x_ancilla_qubits,
                    allocation.z_ancilla_qubits,
                )
            ],
        )
    folded = gadgets.fold_s_round_gadget(patch, allocation, round_index=2, x_z_swapped=swapped)
    plain = gadgets.syndrome_round_gadget(patch, allocation, round_index=2, x_z_swapped=swapped)
    groups = _groups(folded)
    assert len(groups) == 7
    assert groups[:3] + groups[4:] == _groups(plain)
    fold = groups[3]
    touched = [q for s in fold for q in s.qubits]
    assert len(set(touched)) == len(touched)
    coords = _coordinates(patch, allocation)
    assert set(touched) == coords.keys()
    data = set(allocation.data_qubits)
    pairs = [s.qubits for s in fold if s.op_type == OpType.CZ]
    assert sum(a in data for a, _ in pairs) == d * (d - 1) // 2
    assert sum(a not in data for a, _ in pairs) == (d - 1) * (d - 2) // 2
    assert len(pairs) == (d - 1) ** 2
    assert all(coords[a] == coords[b][::-1] for a, b in pairs)
    phases = [s for s in fold if s.op_type != OpType.CZ]
    assert sum(s.qubits[0] in data for s in phases) == d
    assert sum(s.qubits[0] not in data for s in phases) == d - 1
    for step in phases:
        x, y = coords[step.qubits[0]]
        assert x == y
        assert step.op_type == (OpType.SZ if x % 2 else OpType.SZDG)
    tc = _render(patch, allocation, [folded])
    assert tc.num_ticks() == 9
    assert tc.num_measurements() == d * d - 1
    assert {g.gate_type.name for g in tc.get_tick(4).gate_batches()} == {"CZ", "SZ", "SZdg"}
    assert folded.kind == gadgets.GadgetKind.SYNDROME_ROUND
    assert folded.name == "syndrome_extraction_fold_s" + ("_swapped" if swapped else "")


@pytest.mark.parametrize("renderer_name", ["tick", "stim", "dag", "guppy"])
def test_renderers(renderer_name: str, tmp_path: Path) -> None:
    """Pin native CZ and fixed-point phases, including a compiled Guppy function."""
    patch = SurfacePatch.create(distance=3)
    allocation = gadgets.default_allocation(patch)
    gadget = gadgets.fold_s_round_gadget(patch, allocation, round_index=0)
    steps = list(gadget.steps)
    expected_pairs = [(0, 8), (1, 5), (3, 7), (14, 15)]
    if renderer_name == "guppy":
        lines = render_gadget_function(gadget)
        assert sorted(line.strip() for line in lines if line.strip().startswith("cz(")) == [
            "cz(az1, az2)",
            "cz(surf.data[0], surf.data[8])",
            "cz(surf.data[1], surf.data[5])",
            "cz(surf.data[3], surf.data[7])",
        ]
        assert sorted(line.strip() for line in lines if line.strip().startswith(("s(", "sdg("))) == [
            "s(surf.data[2])",
            "s(surf.data[4])",
            "s(surf.data[6])",
            "sdg(ax1)",
            "sdg(ax2)",
        ]
        memory = GuppyRenderer().render(steps, allocation, patch, 1, "Z")
        assert memory == render_surface_gadget_module(patch)
        assert "fold_s" not in memory
        source = memory + "\nfrom guppylang.std.quantum import cz, s, sdg\n\n" + "\n".join(lines) + "\n"
        path = tmp_path / "fold_module.py"
        path.write_text(source)
        spec = importlib.util.spec_from_file_location("fold_module", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
            assert module.syndrome_extraction_fold_s.compile_function() is not None
        finally:
            sys.modules.pop(spec.name, None)
        return
    if renderer_name == "stim":
        circuit = stim.Circuit(StimRenderer(add_detectors=False).render(steps, allocation, patch, 1, "Z"))
    elif renderer_name == "dag":
        dag = DagCircuitRenderer().render(steps, allocation, patch, 1, "Z")
        circuit = stim.Circuit(tick_circuit_to_stim(dag.to_tick_circuit()))
    else:
        circuit = stim.Circuit(tick_circuit_to_stim(_render(patch, allocation, [gadget])))
    pairs = []
    phases: dict[str, list[int]] = {"S": [], "S_DAG": []}
    for instruction in circuit:
        targets = [t.value for t in instruction.targets_copy()]
        if instruction.name == "CZ":
            pairs.extend(zip(targets[::2], targets[1::2], strict=True))
        elif instruction.name in phases:
            phases[instruction.name].extend(targets)
    assert sorted(pairs) == expected_pairs
    assert sorted(phases["S"]) == [2, 4, 6]
    assert sorted(phases["S_DAG"]) == [10, 11]


def test_dag_rejects_unknown_operation() -> None:
    """Unsupported IR steps must never silently disappear from a DAG."""
    patch = SurfacePatch.create(distance=3)
    with pytest.raises(ValueError, match="Unsupported DagCircuit operation"):
        DagCircuitRenderer().render(
            [SurfaceCircuitStep(OpType.F, [0])],
            gadgets.default_allocation(patch),
            patch,
            0,
            "Z",
        )


@pytest.mark.parametrize(
    ("dimensions", "message"),
    [({"dx": 3, "dz": 5}, "square"), ({"distance": 4}, "odd distance"), ({"distance": 3, "rotated": False}, "rotated")],
)
def test_rejections(dimensions: dict[str, int | bool], message: str) -> None:
    """Reject geometries outside the verified fold construction by name."""
    patch = SurfacePatch.create(**dimensions)
    with pytest.raises(ValueError, match=f"fold_s_round_gadget requires.*{message}"):
        gadgets.fold_s_round_gadget(patch, gadgets.default_allocation(patch), round_index=0)
