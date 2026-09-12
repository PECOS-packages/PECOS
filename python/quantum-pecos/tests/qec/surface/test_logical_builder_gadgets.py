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
from pecos.qec.surface.circuit_builder import OpType, QubitAllocation, SurfaceCircuitStep
from pecos.qec.surface.logical_circuit import LogicalGateType, _CircuitGenerator
from pecos.qec.surface.patch import PatchOrientation
from pecos.testing import group_contains, simulate_tick_circuit, stabilizer_generators_after
from pecos_rslib.quantum import TickCircuit

GOLDENS = Path(__file__).parent / "goldens" / "logical_builder"
SHAPES = (
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
)


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

    def allocation(self, label):
        return self._allocation(label)

    @property
    def detectors(self):
        return self._det_json


def make_builder(name: str) -> LogicalCircuitBuilder:
    """Reproduce the orchestrator's captured recipes exactly."""
    patch = SurfacePatch.create(distance=int(name[1]))
    shape = name[3:]
    if shape.startswith("sz_"):
        labels = ["D", "Y"]
    elif shape.startswith("t_"):
        labels = ["D", "A"]
    elif shape.startswith("cx_"):
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


@pytest.mark.parametrize(("shape", "num_observables"), [("d3_h_z_to_x", 1), ("d3_cx_zz", 2)])
def test_fault_distance(shape, num_observables):
    dem = DetectorErrorModel.from_circuit(
        make_builder(shape).to_tick_circuit(),
        p1=0.001,
        p2=0.001,
        p_meas=0.001,
        p_prep=0.001,
    )
    distances = dem.per_observable_fault_distances(3)
    assert len(distances) == num_observables
    assert all(distance is not None and distance.distance == 3 for distance in distances)


@pytest.mark.parametrize("method", ["add_transversal_sz", "add_transversal_szdg"])
def test_physical_sz_layer_dependency(method, monkeypatch):
    builder = make_builder("d3_mem_Z")
    getattr(builder, method)("A")
    calls = []
    original = gadgets.transversal_layer_gadget

    def record(*args, **kwargs):
        calls.append(kwargs["gate"])
        return original(*args, **kwargs)

    monkeypatch.setattr(gadgets, "transversal_layer_gadget", record)
    tc = builder.to_tick_circuit()
    gate = "SZ" if method == "add_transversal_sz" else "SZDG"
    assert calls == [gate]
    assert tc.get_tick(tc.num_ticks() - 1).gate_batches()[0].gate_type.name.upper() == gate


@pytest.mark.parametrize("variant", ["H", "SZ", "SZDG", "CX", "round_swapped", "init_Z_swapped", "init_X_swapped", "Y"])
@pytest.mark.parametrize("renamed", [False, True])
def test_gate_renderer(variant, renamed):
    patch = SurfacePatch.create(3)
    allocation = gadgets.default_allocation(patch)
    if variant in {"H", "SZ", "SZDG"}:
        gadget = gadgets.transversal_layer_gadget(patch, allocation, gate=variant)
    elif variant == "CX":
        target = QubitAllocation(
            [q + 17 for q in allocation.data_qubits],
            [q + 17 for q in allocation.x_ancilla_qubits],
            [q + 17 for q in allocation.z_ancilla_qubits],
        )
        gadget = gadgets.transversal_cx_gadget(patch, allocation, patch, target)
    elif variant == "round_swapped":
        gadget = gadgets.syndrome_round_gadget(patch, allocation, round_index=0, x_z_swapped=True)
    elif variant.startswith("init_"):
        gadget = gadgets.init_syndrome_gadget(patch, allocation, basis=variant[5], x_z_swapped=True)
    else:
        gadget = gadgets.prep_gadget(patch, allocation, basis="Y")
    if renamed:
        gadget = replace(gadget, name="custom_function")
    assert gadget.x_z_swapped == (variant in {"round_swapped", "init_Z_swapped", "init_X_swapped"})
    lines = render_gadget_function(gadget)
    assert lines[1].startswith(f"def {gadget.name}(")
    # Expand constant loops to compare physical gate calls directly with gadget steps.
    function = ast.parse("\n".join(lines)).body[0]
    actual = []

    def collect(nodes, index=None):
        for node in nodes:
            if isinstance(node, ast.For):
                for i in range(ast.literal_eval(node.iter.args[0])):
                    collect(node.body, i)
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                call = node.value
                if isinstance(call.func, ast.Name) and call.func.id in {"h", "s", "sdg", "cx"}:
                    operands = [ast.unparse(arg).replace("[i]", f"[{index}]") for arg in call.args]
                    actual.append((call.func.id, operands))

    collect(function.body)
    names = {}
    for i, alloc in enumerate(gadget.allocations):
        register = ("ctrl.data", "tgt.data")[i] if variant == "CX" else "data" if variant == "Y" else "surf.data"
        names.update({q: f"{register}[{j}]" for j, q in enumerate(alloc.data_qubits)})
        names.update({q: f"ax{j}" for j, q in enumerate(alloc.x_ancilla_qubits)})
        names.update({q: f"az{j}" for j, q in enumerate(alloc.z_ancilla_qubits)})
    gates = {OpType.H: "h", OpType.SZ: "s", OpType.SZDG: "sdg", OpType.CX: "cx"}
    expected = [
        (gates[step.op_type], [names[q] for q in step.qubits]) for step in gadget.steps if step.op_type in gates
    ]
    assert actual == expected


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
    generator.generate()
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
    assert measurements == {(0, "sx7"): 0, (1, "sz9"): 1}
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
def test_stabilizer_indices_are_not_list_positions(swapped):
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
