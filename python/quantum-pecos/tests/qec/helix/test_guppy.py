# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Certified scalar-result Guppy memory acceptance for C4-Helix."""

import ast
from collections import Counter, defaultdict
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import pecos
import pytest
from pecos.guppy_gen._certificate import certify_measurement_layout
from pecos.guppy_gen._module_loader import load_cached_guppy_source
from pecos.guppy_gen.helix import HELIX_MEMORY_NUM_QUBITS, make_helix_memory, render_helix_memory_source
from pecos.qec import build_dem_from_guppy
from pecos.qec.circuit_steps import CircuitStep, OpType
from pecos.qec.dem_spec import result_ref
from pecos.qec.helix import (
    HelixAllocation,
    helix_result_classifier,
    helix_result_refs,
    lower_steps,
    memory_circuit,
    memory_definitions,
    memory_steps,
)
from pecos.qec.helix import memory as memory_module
from pecos.qec.helix.gadgets import memory_gadgets

CERTIFICATE = "__pecos_named_measurement_layout_v2__"
UNIFORM_NOISE = {"p1": 0.001, "p2": 0.001, "p_meas": 0.001, "p_prep": 0.001}
ROUNDS = [1, pytest.param(2, marks=pytest.mark.slow)]


def test_step_schedule() -> None:
    steps = [
        CircuitStep(OpType.H, [0]),
        CircuitStep(OpType.COMMENT, label="ignored"),
        CircuitStep(OpType.TICK),
        CircuitStep(OpType.RELABEL, [0, 1]),
        CircuitStep(OpType.CX, [0, 1]),
        CircuitStep(OpType.MEASURE, [1], "c4x:0"),
        CircuitStep(OpType.MEASURE, [0], "c4z:0"),
    ]
    circuit = lower_steps(steps, 2, schedule="step").tick_circuit
    assert circuit.num_ticks() == 4
    assert [(t, gate.gate_type.name, list(gate.qubits)) for t, gate in circuit.gate_batches()] == [
        (0, "H", [0]),
        (1, "CX", [1, 0]),
        (2, "MZ", [0]),
        (3, "MZ", [1]),
    ]
    assert helix_result_refs(circuit) == [("c4x:0", 0), ("c4z:0", 0)]
    with pytest.raises(ValueError, match="bad-schedule"):
        lower_steps([], 2, schedule="bad-schedule")


@pytest.mark.parametrize(
    "label",
    [
        "",
        "c4x:5",
        "c4z:-1",
        "c4x:0:0",
        "x8:4",
        "z8:00",
        "flag:x0.0",
        "flag:z3.4",
        "flag:y0.1",
        "data:20",
        "data:01",
        "data:1\n",
        "other",
    ],
)
def test_invalid_label(label: str) -> None:
    tc = lower_steps([CircuitStep(OpType.MEASURE, [0], label)], 1).tick_circuit
    with pytest.raises(ValueError, match="label") as error:
        helix_result_refs(tc)
    assert repr(label) in str(error.value)


@pytest.mark.parametrize("basis", ["X", "Z"])
@pytest.mark.parametrize("schedule", ["step", "asap"])
def test_measurement_batches_and_round_order(basis: str, schedule: str) -> None:
    rounds = 3
    steps = memory_steps(basis, rounds)
    tc = lower_steps(steps, HelixAllocation.NUM_SLOTS, schedule=schedule).tick_circuit
    expected = defaultdict(list)
    per_wire = defaultdict(list)
    layout = list(range(HelixAllocation.NUM_SLOTS))
    names = {OpType.PREP: "PZ", OpType.H: "H", OpType.CX: "CX", OpType.MEASURE: "MZ"}
    for step in steps:
        if step.op_type == OpType.RELABEL:
            a, b = step.qubits
            layout[a], layout[b] = layout[b], layout[a]
            continue
        qubits = tuple(layout[q] for q in step.qubits)
        for q in qubits:
            per_wire[q].append((names[step.op_type], qubits if len(qubits) == 2 else (q,), step.label))
        if step.op_type == OpType.MEASURE:
            expected[step.label].append(qubits[0])
    actual = defaultdict(list)
    actual_wires = defaultdict(list)
    for tick in range(tc.num_ticks()):
        batches = tc.get_tick(tick).gate_batches()
        if schedule == "step":
            assert len(batches) == 1
            assert len(batches[0].qubits) == (2 if batches[0].gate_type.name == "CX" else 1)
        for index, gate in enumerate(batches):
            name = gate.gate_type.name
            label = tc.get_gate_meta(tick, index, "label") or ""
            qubits = list(gate.qubits)
            if name == "MZ":
                assert len(qubits) == 1
                actual[label].append(qubits[0])
            for offset, q in enumerate(qubits):
                pair = tuple(qubits[2 * (offset // 2) : 2 * (offset // 2) + 2]) if name == "CX" else (q,)
                actual_wires[q].append((name, pair, label))
    # Per-wire order anchors repeated measurements to their surrounding round gates.
    assert actual_wires == per_wire
    assert actual == expected
    refs = helix_result_refs(tc)
    for tag in expected:
        assert [occurrence for name, occurrence in refs if name == tag] == list(
            range(1 if tag.startswith("data:") else rounds),
        )


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_source_gadget_structure(basis: str) -> None:
    tree = ast.parse(render_helix_memory_source(basis, 2))
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert len([node for node in tree.body if isinstance(node, ast.ClassDef)]) == 1
    assert {f"c4_{block}_{variant}" for block in range(5) for variant in "ab"} <= functions.keys()
    assert {f"weight8_{axis}_{index}" for axis in "xz" for index in range(4)} <= functions.keys()
    factory = functions[f"make_memory_{basis.lower()}"]
    program = factory.body[0]
    assert isinstance(program, ast.FunctionDef)
    calls = [node.value.func.id for node in program.body]
    assert calls == [name for name, _ in memory_gadgets(basis, 2)]
    assert not any(isinstance(node, (ast.For, ast.While)) for node in ast.walk(tree))


@pytest.mark.parametrize("basis", ["X", "Z"])
@pytest.mark.parametrize("rounds", [1, 2])
def test_certificate_and_fresh_factory(basis: str, rounds: int) -> None:
    program = make_helix_memory(basis, rounds)
    another = make_helix_memory(basis, rounds)
    assert program is not another
    circuit = lower_steps(memory_steps(basis, rounds), HelixAllocation.NUM_SLOTS, schedule="step").tick_circuit
    assert getattr(program, CERTIFICATE)[1] == tuple(helix_result_refs(circuit))
    object.__delattr__(program, CERTIFICATE)
    assert hasattr(another, CERTIFICATE)


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_uncertified_and_tampered_programs_rejected(basis: str) -> None:
    rounds = 1
    module = load_cached_guppy_source("helix", render_helix_memory_source(basis, rounds))
    program = module[f"make_memory_{basis.lower()}"]()
    detectors, observables = memory_definitions(basis, rounds)
    kwargs = {
        "num_qubits": HELIX_MEMORY_NUM_QUBITS,
        "detectors": detectors,
        "observables": observables,
        **UNIFORM_NOISE,
    }
    assert not hasattr(program, CERTIFICATE)
    with pytest.raises(ValueError, match="requires a statically straight-line Guppy program"):
        build_dem_from_guppy(program, **kwargs)
    steps = memory_steps(basis, rounds)
    first_x = next(i for i, step in enumerate(steps) if step.label == "c4x:0")
    first_z = next(i for i, step in enumerate(steps) if step.label == "c4z:0")
    steps[first_x], steps[first_z] = steps[first_z], steps[first_x]
    wrong = lower_steps(steps, HelixAllocation.NUM_SLOTS, schedule="step").tick_circuit
    certify_measurement_layout(program, wrong, helix_result_classifier, family="helix")
    with pytest.raises(ValueError, match=r"runtime result trace .*generator-certified layout requires"):
        build_dem_from_guppy(program, **kwargs)


@pytest.mark.parametrize("basis", ["X", "Z"])
@pytest.mark.parametrize("rounds", [1, 2])
def test_typed_definitions_match_discovery(basis: str, rounds: int) -> None:
    tc = memory_circuit(basis, rounds)
    refs = [result_ref(tag, occurrence=occ) for tag, occ in helix_result_refs(tc)]
    detectors, observables = memory_definitions(basis, rounds)
    for kind, definitions in [("detectors", detectors), ("observables", observables)]:
        entries = tc.circuit_definitions()[kind]
        assert len(definitions) == len(entries)
        for index, (definition, entry) in enumerate(zip(definitions, entries, strict=True)):
            assert definition.id == entry["id"] == index
            assert definition.refs == tuple(refs[r] for r in entry["measurements"])


@pytest.mark.parametrize("kind", ["Detector", "Observable"])
def test_nonzero_reference_rejected(monkeypatch: pytest.MonkeyPatch, kind: str) -> None:
    def nonzero(*_args: object) -> SimpleNamespace:
        return SimpleNamespace(
            detectors=[([0], int(kind == "Detector"))],
            observables=[([1], int(kind == "Observable"))],
        )

    monkeypatch.setattr(memory_module, "discover_detectors", nonzero)
    with pytest.raises(ValueError, match=f"{kind} 0 has nonzero reference 1"):
        memory_definitions("Z", 1)


@pytest.mark.parametrize("basis", ["X", "Z"])
@pytest.mark.parametrize("rounds", ROUNDS)
def test_dem(basis: str, rounds: int) -> None:
    program = make_helix_memory(basis, rounds)
    detectors, observables = memory_definitions(basis, rounds)
    build = build_dem_from_guppy(
        program,
        num_qubits=HELIX_MEMORY_NUM_QUBITS,
        detectors=detectors,
        observables=observables,
        **UNIFORM_NOISE,
    )
    assert build.dem.num_detectors == len(detectors) == 42 * rounds
    assert build.dem.num_observables == len(observables) == 2
    assert build.audit["named_result_binding"] == "generator_layout_v2_program_bound"


@pytest.mark.parametrize("basis", ["X", "Z"])
@pytest.mark.parametrize("rounds", ROUNDS)
def test_noiseless_agreement(basis: str, rounds: int) -> None:
    program = make_helix_memory(basis, rounds)
    detectors, observables = memory_definitions(basis, rounds)
    expected = Counter(tag for tag, _ in helix_result_refs(memory_circuit(basis, rounds)))
    shots = 64
    for seed in (3, 19):
        results = (
            pecos.sim(program)
            .classical(pecos.selene_engine())
            .quantum(pecos.stabilizer())
            .qubits(HELIX_MEMORY_NUM_QUBITS)
            .seed(seed)
            .run(shots)
        )
        columns = results.to_shot_map().to_dict()
        assert set(columns) == set(expected)
        by_ref = {}
        for tag, count in expected.items():
            # Check each shot's arity before flattening: totals alone could hide missing rounds.
            assert len(columns[tag]) == shots
            assert all(np.asarray(shot).size == count for shot in columns[tag])
            values = np.asarray(columns[tag]).reshape(shots, count)
            for occurrence in range(count):
                by_ref[result_ref(tag, occurrence=occurrence)] = values[:, occurrence]
        assert len(detectors) == 42 * rounds
        assert len(observables) == 2
        for definition in [*detectors, *observables]:
            parity = sum(by_ref[ref] for ref in definition.refs) % 2
            assert not parity.any(), (basis, rounds, seed, definition)
        # Positive control: the checks project individual data qubits, so a readout that
        # really measures the code varies; one that measured fresh qubits would be constant.
        readouts = [by_ref[result_ref(f"data:{slot}")] for slot in range(20)]
        assert any(len(set(column.tolist())) == 2 for column in readouts), (basis, rounds, seed)


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_guppy_fault_distance(basis: str) -> None:
    detectors, observables = memory_definitions(basis, 1)
    dem = build_dem_from_guppy(
        make_helix_memory(basis, 1),
        num_qubits=HELIX_MEMORY_NUM_QUBITS,
        detectors=detectors,
        observables=observables,
        **UNIFORM_NOISE,
    ).dem
    start = perf_counter()
    distances = dem.per_observable_fault_distances(5)
    elapsed = perf_counter() - start
    print(f"Guppy basis={basis} rounds=1 max_weight=5: {distances}, {elapsed:.3f}s", flush=True)
    assert distances == [None, None], distances
