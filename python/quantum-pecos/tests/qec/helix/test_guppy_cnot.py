# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Certified two-block C4-Helix Guppy programs agree with the abstract CNOT.

No Guppy distance test is needed: equal symptoms imply equal distances, and
``test_cnot_distance.py`` pins the tick side.
"""

import ast
import re
from collections import Counter

import numpy as np
import pecos
import pytest
from pecos.guppy_gen import helix as renderer
from pecos.guppy_gen.helix import (
    HELIX_CNOT_NUM_QUBITS,
    make_helix_cnot,
    render_helix_cnot_source,
    render_helix_memory_source,
)
from pecos.qec import DetectorErrorModel, build_dem_from_guppy
from pecos.qec.circuit_steps import CircuitStep, OpType
from pecos.qec.dem_spec import result_ref
from pecos.qec.helix import (
    HelixAllocation,
    cnot_circuit,
    cnot_definitions,
    cnot_gadgets,
    helix_result_refs,
    lower_steps,
)

from qec.helix.test_guppy import CERTIFICATE, UNIFORM_NOISE
from qec.helix.test_guppy_logic import _symptoms

PAIRS = ["ZZ", "XX", "ZX"]
RUNTIME_PAIRS = [pytest.param(pair, marks=() if pair == "ZZ" else pytest.mark.slow) for pair in PAIRS]


@pytest.mark.parametrize("pair", PAIRS)
def test_source_gadget_structure(pair):
    tree = ast.parse(render_helix_cnot_source(*pair))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    gadgets = cnot_gadgets(*pair)
    names = list(dict(gadgets))
    assert len(gadgets) == 57
    assert len(names) == 41
    assert [node.name for node in functions] == [*names, f"make_cnot_{pair.lower()}"]
    classes = [node.name for node in tree.body if isinstance(node, ast.ClassDef)]
    assert classes == ["HelixCode_20_2_6"]
    program = functions[-1].body[0]
    assert isinstance(program, ast.FunctionDef)
    assert program.name == f"cnot_{pair.lower()}"
    assert [node.value.func.id for node in program.body] == [name for name, _ in gadgets]
    for node, (name, _) in zip(program.body, gadgets, strict=True):
        call = node.value
        assert not call.keywords
        if "_prepare_" in name:
            assert isinstance(node, ast.Assign)
            assert [target.id for target in node.targets] == [name[0].lower()]
            assert call.args == []
        else:
            assert isinstance(node, ast.Expr)
            assert [arg.id for arg in call.args] == (["a", "b"] if name == "cnot" else [name[0].lower()])
    assert [ast.dump(node) for node in program.body[:2]] == [
        ast.dump(ast.parse(f"{block.lower()} = {block}_prepare_{basis.lower()}()").body[0])
        for block, basis in zip("AB", pair, strict=True)
    ]
    cnot = next(node for node in functions if node.name == "cnot")
    assert [ast.dump(node) for node in cnot.body] == [
        ast.dump(ast.parse(f"cx(a.data[{i}], b.data[{i}])").body[0]) for i in range(20)
    ]
    tags = set()
    for block, basis in zip("AB", pair, strict=True):
        memory = ast.parse(render_helix_memory_source(basis, 2))
        originals = {node.name: node for node in memory.body if isinstance(node, ast.FunctionDef)}
        for function in functions:
            if not function.name.startswith(f"{block}_"):
                continue
            for node in ast.walk(function):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "result":
                    tag = node.args[0]
                    assert isinstance(tag, ast.Constant)
                    assert tag.value.startswith(f"{block}.")
                    tags.add(tag.value)
                    tag.value = tag.value[2:]
            assert [ast.dump(node) for node in function.body] == [
                ast.dump(node) for node in originals[function.name[2:]].body
            ]
    assert tags == {tag for tag, _ in helix_result_refs(cnot_circuit(*pair))}
    assert not any(isinstance(node, (ast.For, ast.While)) for node in ast.walk(tree))
    quantum_import = next(
        node for node in tree.body if isinstance(node, ast.ImportFrom) and node.module == "guppylang.std.quantum"
    )
    quantum_names = {"cx", "cz", "discard_array", "h", "measure", "qubit", "s", "sdg", "x", "z"}
    used = {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert [alias.name for alias in quantum_import.names] == sorted(used & quantum_names)


@pytest.mark.parametrize("pair", RUNTIME_PAIRS)
def test_certificate_and_fresh_factory(pair):
    program = make_helix_cnot(*pair)
    another = make_helix_cnot(*pair)
    assert program is not another
    steps = [step for _, gadget in cnot_gadgets(*pair) for step in gadget]
    circuit = lower_steps(steps, 2 * HelixAllocation.NUM_SLOTS, schedule="step").tick_circuit
    assert getattr(program, CERTIFICATE)[1] == tuple(helix_result_refs(circuit))
    object.__delattr__(program, CERTIFICATE)
    assert hasattr(another, CERTIFICATE)


@pytest.mark.parametrize("builder", [render_helix_cnot_source, make_helix_cnot])
@pytest.mark.parametrize("pair", ["XZ", "YZ", "XY", "xZ", "Zz"])
def test_rejected_before_building(builder, pair):
    message = "Random readout for basis pair 'XZ'" if pair == "XZ" else "Expected basis X or Z, got"
    with pytest.raises(ValueError, match=message) as error:
        builder(*pair)
    if pair == "XZ":
        assert "Bell pair" in str(error.value)
        assert "ZZ, XX, ZX" in str(error.value)


@pytest.mark.parametrize(
    ("name", "occurrence", "step", "message"),
    [
        ("A_weight8_x_0", 1, CircuitStep(OpType.H, [0]), "different steps"),
        ("B_c4_0_a", 0, CircuitStep(OpType.H, [5]), "outside block"),
        ("A_c4_0_a", 0, CircuitStep(OpType.H, [81]), "outside block"),
        ("cnot", 0, CircuitStep(OpType.CX, [0, 77]), "not a transversal CX"),
        ("cnot", 0, CircuitStep(OpType.CZ, [0, 76]), "not a transversal CX"),
        ("cnot", 0, CircuitStep(OpType.CX, [20, 96]), "not a transversal CX"),
        ("C_c4_0_a", 0, CircuitStep(OpType.H, [76]), "unknown gadget"),
    ],
    ids=["different-steps", "outside-block-low", "outside-block-high", "cx-target", "cz", "cx-ancilla", "unknown"],
)
def test_renderer_guards(monkeypatch, name, occurrence, step, message):
    gadgets = cnot_gadgets("Z", "Z")
    indices = [index for index, (gadget_name, _) in enumerate(gadgets) if gadget_name == name]
    if indices:
        index = indices[occurrence]
        gadgets[index] = (name, [step, *gadgets[index][1][1:]])
    else:
        gadgets.append((name, [step]))
    monkeypatch.setattr(renderer, "cnot_gadgets", lambda *_args: gadgets)
    with pytest.raises(ValueError, match=rf"{re.escape(name)}: {message}"):
        render_helix_cnot_source("Z", "Z")


@pytest.mark.parametrize("pair", RUNTIME_PAIRS)
def test_dem(pair):
    detectors, observables = cnot_definitions(*pair)
    build = build_dem_from_guppy(
        make_helix_cnot(*pair),
        num_qubits=HELIX_CNOT_NUM_QUBITS,
        detectors=detectors,
        observables=observables,
        **UNIFORM_NOISE,
    )
    assert build.dem.num_detectors == len(detectors) == 168
    assert build.dem.num_observables == len(observables) == 4
    assert build.audit["named_result_binding"] == "generator_layout_v2_program_bound"
    tick_dem = DetectorErrorModel.from_circuit(cnot_circuit(*pair), **UNIFORM_NOISE)
    assert _symptoms(build.dem) == _symptoms(tick_dem)


def _assert_noiseless_agreement(pair):
    program = make_helix_cnot(*pair)
    detectors, observables = cnot_definitions(*pair)
    expected = Counter(tag for tag, _ in helix_result_refs(cnot_circuit(*pair)))
    shots = 64
    for seed in (3, 19):
        results = (
            pecos.sim(program)
            .classical(pecos.selene_engine())
            .quantum(pecos.stabilizer())
            .qubits(HELIX_CNOT_NUM_QUBITS)
            .seed(seed)
            .run(shots)
        )
        columns = results.to_shot_map().to_dict()
        assert set(columns) == set(expected)
        by_ref = {}
        for tag, count in expected.items():
            assert len(columns[tag]) == shots
            assert all(np.asarray(shot).size == count for shot in columns[tag])
            values = np.asarray(columns[tag]).reshape(shots, count)
            for occurrence in range(count):
                by_ref[result_ref(tag, occurrence=occurrence)] = values[:, occurrence]
        for kind, definitions in [("detector", detectors), ("observable", observables)]:
            for definition in definitions:
                parity = sum(by_ref[ref] for ref in definition.refs) % 2
                reference = 0 if kind == "detector" else definition.metadata["reference"]
                assert (parity == reference).all(), f"{kind} parity: {pair}, {seed}, {definition}"
        for block in "AB":
            readouts = [by_ref[result_ref(f"{block}.data:{slot}")] for slot in range(20)]
            assert any(len(set(column.tolist())) == 2 for column in readouts), (pair, block, seed)


@pytest.mark.parametrize("pair", RUNTIME_PAIRS)
def test_noiseless_agreement(pair):
    _assert_noiseless_agreement(pair)


def test_removed_cnot_mutation(monkeypatch):
    def mutated(basis_a, basis_b):
        source = render_helix_cnot_source(basis_a, basis_b)
        call = "        cnot(a, b)\n"
        assert source.count(call) == 1
        return source.replace(call, "")

    monkeypatch.setattr(renderer, "render_helix_cnot_source", mutated)
    with pytest.raises(AssertionError, match="detector parity"):
        _assert_noiseless_agreement("ZZ")


def test_reversed_cnot_mutation(monkeypatch):
    def mutated(basis_a, basis_b):
        source = render_helix_cnot_source(basis_a, basis_b)
        source, count = re.subn(r"cx\(a\.data\[(\d+)\], b\.data\[\1\]\)", r"cx(b.data[\1], a.data[\1])", source)
        assert count == 20
        return source

    monkeypatch.setattr(renderer, "render_helix_cnot_source", mutated)
    with pytest.raises(AssertionError, match="detector parity"):
        _assert_noiseless_agreement("ZZ")


def test_qubit_budget():
    program = make_helix_cnot("Z", "Z")
    with pytest.raises(RuntimeError, match="No more qubits available to allocate"):
        (
            pecos.sim(program)
            .classical(pecos.selene_engine())
            .quantum(pecos.stabilizer())
            .qubits(HELIX_CNOT_NUM_QUBITS - 1)
            .run(2)
        )
