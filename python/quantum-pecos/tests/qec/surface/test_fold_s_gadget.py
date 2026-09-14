# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Signed half-cycle and round-flow oracles for the Cartesian fold S gadget."""

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import stim
from pecos.guppy_gen.gadget_render import render_gadget_function
from pecos.qec.surface import SurfacePatch, gadgets
from pecos.qec.surface.circuit_builder import (
    DagCircuitRenderer,
    GuppyRenderer,
    OpType,
    QubitAllocation,
    StimRenderer,
    SurfaceCircuitStep,
    TickCircuitRenderer,
    _analyze_szz_forward_flow,
    _lower_szz_forward_flow_ops,
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


def _coordinates(
    patch: SurfacePatch,
    allocation: QubitAllocation,
    *,
    exterior: bool = False,
) -> dict[int, tuple[int, int]]:
    """Independent coordinate formula; locate bulk sites by support bounds."""
    d = patch.dx
    data = {i: (2 * (i % d) + 1, 2 * (d - i // d) - 1) for i in range(d * d)}
    coords = {allocation.data_qubits[i]: xy for i, xy in data.items()}
    for checks, ancillas in (
        (patch.geometry.x_stabilizers, allocation.x_ancilla_qubits),
        (patch.geometry.z_stabilizers, allocation.z_ancilla_qubits),
    ):
        for check in checks:
            if len(check.data_qubits) == 4 or exterior:
                xs, ys = zip(*(data[q] for q in check.data_qubits), strict=True)
                x, y = (min(xs) + max(xs)) // 2, (min(ys) + max(ys)) // 2
                if len(check.data_qubits) == 2:
                    if min(xs) == max(xs):
                        x += -1 if x == 1 else 1
                    else:
                        y += -1 if y == 1 else 1
                coords[ancillas[check.index]] = (x, y)
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
    if round_gadget.x_z_swapped:
        parts[1] = gadgets.init_syndrome_gadget(patch, allocation, basis="Z", x_z_swapped=True)
        parts[2] = gadgets.syndrome_round_gadget(patch, allocation, round_index=0, x_z_swapped=True)
    parts[-2] = round_gadget
    tc = _replayable(_render(patch, allocation, parts))
    cx_ticks = [
        i for i in range(tc.num_ticks()) if any(g.gate_type.name == "CX" for g in tc.get_tick(i).gate_batches())
    ]
    # Complementary initialization and round 0 each contribute four CX layers.
    return tc, cx_ticks[4 + 4 + 1] + 1


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


@pytest.mark.parametrize("d", [2, 3, 4, 5])
@pytest.mark.parametrize("dagger", [False, True])
@pytest.mark.parametrize("swapped", [False, True])
def test_fold_preserves_signed_group(d: int, *, dagger: bool, swapped: bool) -> None:
    """Both alternating assignments preserve Z memory; one ancilla flip does not."""
    patch = SurfacePatch.create(distance=d)
    allocation = gadgets.default_allocation(patch)
    gadget = gadgets.fold_s_round_gadget(patch, allocation, round_index=1, dagger=dagger, x_z_swapped=swapped)
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


@pytest.mark.parametrize("d", [2, 3, 4, 5])
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
    x_checks = geom.z_stabilizers if swapped else geom.x_stabilizers
    product = stim.PauliString(allocation.total)
    for check in z_checks:
        product *= stim.PauliString(_pauli(allocation.total, "Z", list(check.data_qubits)))
    assert circuit.has_flow(stim.Flow(input=xl, output=yl * product))
    assert not circuit.has_flow(stim.Flow(input=xl, output=-yl * product))
    assert circuit.has_flow(stim.Flow(input=zl, output=zl))
    z_records = list(range(len(x_checks), len(x_checks) + len(z_checks)))
    assert circuit.has_flow(stim.Flow(input=xl, output=yl, measurements=z_records))
    for check, record in zip(z_checks, z_records, strict=True):
        operator = stim.PauliString(_pauli(allocation.total, "Z", list(check.data_qubits)))
        assert circuit.has_flow(stim.Flow(output=operator, measurements=[record]))

    coords = _coordinates(patch, allocation, exterior=True)
    if swapped:
        coords = {q: (y, x) for q, (x, y) in coords.items()}
    x_ancillas, z_ancillas = (
        (allocation.z_ancilla_qubits, allocation.x_ancilla_qubits)
        if swapped
        else (allocation.x_ancilla_qubits, allocation.z_ancilla_qubits)
    )
    z_by_position = {coords[z_ancillas[check.index]]: check for check in z_checks}
    for check in x_checks:
        x, y = coords[x_ancillas[check.index]]
        operator = stim.PauliString(_pauli(allocation.total, "X", list(check.data_qubits)))
        # Only the bottom bulk row reaches the left boundary on the input side.
        input_partner = z_by_position[0, x] if y == 2 and check.weight == 4 else None
        input_operator = operator.copy()
        if input_partner is not None:
            input_operator *= stim.PauliString(_pauli(allocation.total, "Z", list(input_partner.data_qubits)))
        assert circuit.has_flow(stim.Flow(input=input_operator, measurements=[check.index]))
        assert not circuit.has_flow(stim.Flow(input=-input_operator, measurements=[check.index]))
        assert circuit.has_flow(stim.Flow(input=operator, measurements=[check.index])) == (input_partner is None)

        output_partner = z_by_position.get((y + 2, x))
        records = [check.index]
        if output_partner is not None:
            records.append(len(x_checks) + output_partner.index)
        assert circuit.has_flow(stim.Flow(output=operator, measurements=records))
        assert not circuit.has_flow(stim.Flow(output=-operator, measurements=records))
        assert circuit.has_flow(stim.Flow(output=operator, measurements=[check.index])) == (output_partner is None)

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


@pytest.mark.parametrize("d", [2, 3, 4, 5])
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
    assert tc.get_tick_meta(4, "phase") == "fold_s"
    assert tc.get_tick_meta(4, "cx_round") is None
    assert folded.fold == "S"
    assert folded.kind == gadgets.GadgetKind.SYNDROME_ROUND
    assert folded.name == "syndrome_extraction_fold_s" + ("_swapped" if swapped else "")


@pytest.mark.parametrize("renderer_name", ["tick", "stim", "dag", "guppy"])
@pytest.mark.parametrize("dagger", [False, True])
def test_renderers(renderer_name: str, tmp_path: Path, *, dagger: bool) -> None:
    """Pin native CZ and fixed-point phases, including a compiled Guppy function."""
    patch = SurfacePatch.create(distance=3)
    allocation = gadgets.default_allocation(patch)
    gadget = gadgets.fold_s_round_gadget(patch, allocation, round_index=0, dagger=dagger)
    name = "syndrome_extraction_fold_sdg" if dagger else "syndrome_extraction_fold_s"
    variant = "S-dagger" if dagger else "S"
    assert gadget.name == name
    assert gadget.fold == ("SDG" if dagger else "S")
    steps = list(gadget.steps)
    expected_pairs = [(0, 8), (1, 5), (3, 7), (14, 15)]
    if renderer_name == "guppy":
        lines = render_gadget_function(gadget)
        assert f"def {name}(surf: SurfaceCode_3x3) -> Syndrome_3x3:" in lines
        doc = f'    """Extract full syndrome with the fold-transversal logical {variant} between CX layers 2 and 3."""'
        assert doc in lines
        assert doc in render_gadget_function(replace(gadget, name="renamed_round"))
        assert f"    # fold-transversal {variant} layer" in lines
        assert sorted(line.strip() for line in lines if line.strip().startswith("cz(")) == [
            "cz(az1, az2)",
            "cz(surf.data[0], surf.data[8])",
            "cz(surf.data[1], surf.data[5])",
            "cz(surf.data[3], surf.data[7])",
        ]
        data_gate, ancilla_gate = ("sdg", "s") if dagger else ("s", "sdg")
        assert sorted(line.strip() for line in lines if line.strip().startswith(("s(", "sdg("))) == sorted(
            [
                *(f"{data_gate}(surf.data[{q}])" for q in (2, 4, 6)),
                *(f"{ancilla_gate}(ax{q})" for q in (1, 2)),
            ],
        )
        _compile_gadget(patch, gadget, tmp_path)
        return
    if renderer_name == "stim":
        circuit = stim.Circuit(StimRenderer(add_detectors=False).render(steps, allocation, patch, 1, "Z"))
    elif renderer_name == "dag":
        dag = DagCircuitRenderer().render(steps, allocation, patch, 1, "Z")
        circuit = stim.Circuit(tick_circuit_to_stim(dag.to_tick_circuit()))
    else:
        tc = _render(patch, allocation, [gadget])
        assert tc.get_tick_meta(4, "phase") == ("fold_sdg" if dagger else "fold_s")
        assert tc.get_tick_meta(4, "cx_round") is None
        circuit = stim.Circuit(tick_circuit_to_stim(tc))
    pairs = []
    phases: dict[str, list[int]] = {"S": [], "S_DAG": []}
    for instruction in circuit:
        targets = [t.value for t in instruction.targets_copy()]
        if instruction.name == "CZ":
            pairs.extend(zip(targets[::2], targets[1::2], strict=True))
        elif instruction.name in phases:
            phases[instruction.name].extend(targets)
    assert sorted(pairs) == expected_pairs
    assert sorted(phases["S"]) == ([10, 11] if dagger else [2, 4, 6])
    assert sorted(phases["S_DAG"]) == ([2, 4, 6] if dagger else [10, 11])


def _compile_gadget(patch: SurfacePatch, gadget: gadgets.Gadget, tmp_path: Path) -> None:
    memory = GuppyRenderer().render(list(gadget.steps), gadget.allocations[0], patch, 1, "Z")
    assert "fold_s" not in memory
    source = (
        memory + "\nfrom guppylang.std.quantum import cz, s, sdg\n\n" + "\n".join(render_gadget_function(gadget)) + "\n"
    )
    path = tmp_path / "fold_module.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("fold_module", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        assert getattr(module, gadget.name).compile_function() is not None
    finally:
        sys.modules.pop(spec.name, None)


@pytest.mark.parametrize("d", [2, 4])
@pytest.mark.parametrize("swapped", [False, True])
@pytest.mark.parametrize("variant", [None, "S", "SDG"])
def test_even_guppy_syndrome(d: int, variant: str | None, tmp_path: Path, *, swapped: bool) -> None:
    """Unequal X/Z register sizes follow the orientation for plain and fold rounds."""
    patch = SurfacePatch.create(distance=d)
    allocation = gadgets.default_allocation(patch)
    if variant is None:
        gadget = gadgets.syndrome_round_gadget(patch, allocation, round_index=0, x_z_swapped=swapped)
    else:
        gadget = gadgets.fold_s_round_gadget(
            patch,
            allocation,
            round_index=0,
            x_z_swapped=swapped,
            dagger=variant == "SDG",
        )
    syndrome = f"Syndrome_{d}x{d}" + ("_swapped" if swapped else "")
    assert f"    return {syndrome}(synx, synz)" in render_gadget_function(gadget)
    _compile_gadget(patch, gadget, tmp_path)


def test_dag_rejects_unsupported_operation() -> None:
    """Unsupported IR steps must never silently disappear from a DAG."""
    patch = SurfacePatch.create(distance=3)
    with pytest.raises(ValueError, match=r"^Unsupported DagCircuit operation: F$"):
        DagCircuitRenderer().render(
            [SurfaceCircuitStep(OpType.F, [0])],
            gadgets.default_allocation(patch),
            patch,
            0,
            "Z",
        )


@pytest.mark.parametrize(
    ("dimensions", "message"),
    [
        ({"dx": 3, "dz": 5}, "square"),
        ({"distance": 1}, "distance at least 2"),
        ({"distance": 3, "rotated": False}, "rotated"),
    ],
)
def test_rejections(dimensions: dict[str, int | bool], message: str) -> None:
    """Reject geometries outside the verified fold construction by name."""
    patch = SurfacePatch.create(**dimensions)
    with pytest.raises(ValueError, match=f"fold_s_round_gadget requires.*{message}"):
        gadgets.fold_s_round_gadget(patch, gadgets.default_allocation(patch), round_index=0)


@pytest.mark.parametrize("renderer", [TickCircuitRenderer, StimRenderer])
@pytest.mark.parametrize("basis", ["X", "Z"])
def test_fold_detector_annotation_rejected(renderer: type, basis: str) -> None:
    """Memory-template detectors are invalid even when Z memory masks the error."""
    patch = SurfacePatch.create(distance=3)
    allocation = gadgets.default_allocation(patch)
    parts = gadgets.memory_gadgets(patch, 2, basis)
    parts[-2] = gadgets.fold_s_round_gadget(patch, allocation, round_index=1)
    steps = [step for part in parts for step in part.steps]
    message = f"{renderer.__name__}: detectors for fold rounds come from the logical builder route"
    with pytest.raises(ValueError, match=message):
        renderer(add_detectors=True).render(steps, allocation, patch, 2, basis)
    assert renderer(add_detectors=False).render(steps, allocation, patch, 2, basis) is not None


def test_szz_forward_flow_rejects_cz() -> None:
    """A native CZ cannot be counted as an SZZ interaction by the pulse analysis."""
    with pytest.raises(ValueError, match="SZZ forward-flow analysis only supports SZZ/SZZdg two-qubit gates"):
        _analyze_szz_forward_flow([SurfaceCircuitStep(OpType.CZ, [0, 1])])


def test_szz_lowering_rejects_cz() -> None:
    """SZZ lowering must not silently pass native CZ through its pulse model."""
    with pytest.raises(ValueError, match="SZZ forward-flow lowering only supports SZZ/SZZdg two-qubit gates"):
        _lower_szz_forward_flow_ops([SurfaceCircuitStep(OpType.CZ, [0, 1])])


@pytest.mark.parametrize("malformation", ["fractional_x", "fractional_y", "duplicate_centre", "missing_partner"])
def test_fold_geometry_bounds(malformation: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Malformed coordinates fail at the fold boundary with a named error."""
    patch = SurfacePatch.create(distance=3)
    allocation = gadgets.default_allocation(patch)
    checks = patch.geometry.x_stabilizers
    bulk = [check for check in checks if check.weight == 4]
    if malformation in {"fractional_x", "fractional_y"}:
        support = (0, 1, 3, 5) if malformation == "fractional_x" else (0, 1, 3, 7)
        checks[bulk[0].index] = replace(bulk[0], data_qubits=support)
        message = "bulk-centre coordinate sums divisible by 4"
    elif malformation == "duplicate_centre":
        checks[bulk[0].index] = replace(bulk[0], data_qubits=bulk[1].data_qubits)
        message = "distinct data and bulk-ancilla sites"
    else:
        original = gadgets.rotated_id_to_position

        def missing_partner(qubit: int, distance: int) -> tuple[int, int]:
            x, y = original(qubit, distance)
            return (x + 16, y) if qubit == 0 else (x, y)

        monkeypatch.setattr(gadgets, "rotated_id_to_position", missing_partner)
        message = "missing transpose partner"
    with pytest.raises(ValueError, match=f"fold_s_round_gadget.*{message}"):
        gadgets.fold_s_round_gadget(patch, allocation, round_index=0)


@pytest.mark.parametrize("layer_count", [3, 5])
def test_fold_requires_four_cx_layers(layer_count: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """The fold's insertion point is defined only for the four-layer default round."""
    patch = SurfacePatch.create(distance=3)
    allocation = gadgets.default_allocation(patch)
    monkeypatch.setattr(gadgets, "compute_cnot_schedule", lambda *_args, **_kwargs: [[] for _ in range(layer_count)])
    with pytest.raises(ValueError, match="fold_s_round_gadget requires four CX layers in the default schedule"):
        gadgets.fold_s_round_gadget(patch, allocation, round_index=0)
