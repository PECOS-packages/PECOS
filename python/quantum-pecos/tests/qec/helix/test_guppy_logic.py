# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Certified C4-Helix Guppy logic with typed, signed measurement parities."""

import ast
import re
from collections import Counter
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import pecos
import pytest
from pecos.guppy_gen import helix as renderer
from pecos.guppy_gen.helix import HELIX_MEMORY_NUM_QUBITS, make_helix_logic, render_helix_logic_source
from pecos.qec import DetectorErrorModel, build_dem_from_guppy
from pecos.qec.circuit_steps import CircuitStep, OpType
from pecos.qec.dem_spec import Detector, Observable, result_ref
from pecos.qec.helix import (
    HelixAllocation,
    c4_helix_gates,
    compile_logical_clifford,
    helix_result_refs,
    logic_circuit,
    logic_definitions,
    logic_gadgets,
    logical_gate_steps,
    lower_steps,
)
from pecos.qec.helix import logic as logic_module

from qec.helix.test_guppy import CERTIFICATE, UNIFORM_NOISE
from qec.helix.test_logic import WORDS

# The following syndrome round exposes a dropped relabel; SWAP alone is a code symmetry.
RELABEL_WORD = ("Z", ("SWAP", "S0S1"))
# CZ around a relabel exposes its signed X readouts (both references are one).
SIGNED_RELABEL_WORD = ("X", ("CZ", "SWAP", "CZ"))
# A compiler-selected Pauli changes the signs relative to the uncorrected word.
FIXUP_WORD = ("X", ("SWAP", "Z1"))
EXTRA_WORDS = [RELABEL_WORD, SIGNED_RELABEL_WORD, FIXUP_WORD]
LOGIC_WORDS = [*WORDS, *EXTRA_WORDS]
PHASE_WORDS = [("Z", ("S0S1",)), ("X", ("S0S1", "S0S1"))]
# Slow runs only after merge, so the words the mutation guards use keep their DEM and
# positive Selene checks in the default lane.
DEFAULT_RUNTIME_WORDS = [*EXTRA_WORDS, *PHASE_WORDS]
RUNTIME_WORDS = [
    pytest.param(basis, word, marks=() if (basis, word) in DEFAULT_RUNTIME_WORDS else pytest.mark.slow)
    for basis, word in LOGIC_WORDS
]


@pytest.mark.parametrize(("basis", "word"), LOGIC_WORDS)
def test_source_gadget_structure(basis, word):
    tree = ast.parse(render_helix_logic_source(basis, word))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    names = Counter(node.name for node in functions)
    assert len([node for node in tree.body if isinstance(node, ast.ClassDef)]) == 1
    assert all(names[f"gate_{token}"] == 1 for token in set(word))
    factory = next(node for node in functions if node.name == f"make_logic_{basis.lower()}")
    program = factory.body[0]
    assert isinstance(program, ast.FunctionDef)
    assert program.name == f"logic_{basis.lower()}"
    calls = [node.value.func.id for node in program.body]
    assert calls == [f"gate_{name}" if name in word else name for name, _ in logic_gadgets(basis, word)]
    assert not any(isinstance(node, (ast.For, ast.While)) for node in ast.walk(tree))
    quantum_import = next(
        node for node in tree.body if isinstance(node, ast.ImportFrom) and node.module == "guppylang.std.quantum"
    )
    imported = [alias.name for alias in quantum_import.names]
    quantum_names = {"cx", "cz", "discard_array", "h", "measure", "qubit", "s", "sdg", "x", "z"}
    used = {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert imported == sorted(used & quantum_names)


@pytest.mark.parametrize(("basis", "word"), LOGIC_WORDS)
def test_certificate_and_fresh_factory(basis, word):
    program = make_helix_logic(basis, word)
    another = make_helix_logic(basis, word)
    assert program is not another
    steps = [step for _, gadget in logic_gadgets(basis, word) for step in gadget]
    circuit = lower_steps(steps, HelixAllocation.NUM_SLOTS, schedule="step").tick_circuit
    assert getattr(program, CERTIFICATE)[1] == tuple(helix_result_refs(circuit))
    object.__delattr__(program, CERTIFICATE)
    assert hasattr(another, CERTIFICATE)


@pytest.mark.parametrize(("basis", "word"), LOGIC_WORDS)
def test_typed_definitions_match_discovery(basis, word):
    circuit = logic_circuit(basis, word)
    refs = [result_ref(tag, occurrence=occ) for tag, occ in helix_result_refs(circuit)]
    detectors, observables = logic_definitions(basis, word)
    for kind, definitions, cls in [("detectors", detectors, Detector), ("observables", observables, Observable)]:
        entries = circuit.circuit_definitions()[kind]
        assert len(definitions) == len(entries)
        for index, (definition, entry) in enumerate(zip(definitions, entries, strict=True)):
            assert isinstance(definition, cls)
            assert definition.id == entry["id"] == index
            assert definition.refs == tuple(refs[r] for r in entry["measurements"])
            if kind == "observables":
                assert definition.metadata == {"reference": entry["reference"]}
                assert entry["reference"] in {0, 1}


@pytest.mark.parametrize("builder", [logic_definitions, logic_circuit])
def test_nonzero_reference_rejected(monkeypatch, builder):
    def nonzero(*_args):
        return SimpleNamespace(detectors=[([0], 1)], observables=[([1], 0)])

    monkeypatch.setattr(logic_module, "discover_detectors", nonzero)
    with pytest.raises(ValueError, match=r"Detector 0 has nonzero reference 1: \[0\]"):
        builder("Z", ("SWAP",))


@pytest.mark.parametrize(
    ("basis", "word", "message"),
    [("X", ("S0S1",), "Random readout"), ("Z", ("unknown",), "Invalid logical gate token 'unknown'")],
)
def test_render_rejects_invalid_word(basis, word, message):
    with pytest.raises(ValueError, match=message):
        render_helix_logic_source(basis, word)


def test_render_rejects_unsupported_operation(monkeypatch):
    gadgets = logic_gadgets("Z", ("SWAP",))
    gadgets[1] = (gadgets[1][0], [CircuitStep(OpType.SX, [0])])
    monkeypatch.setattr(renderer, "logic_gadgets", lambda *_args: gadgets)
    with pytest.raises(ValueError, match="Unsupported Helix Guppy operation: SX"):
        render_helix_logic_source("Z", ("SWAP",))


def test_signed_fixup():
    target = {"X0": "-X1", "X1": "+X0", "Z0": "+Z1", "Z1": "+Z0"}
    basis, word = FIXUP_WORD
    assert compile_logical_clifford(target) == word
    assert word[-1] not in c4_helix_gates()
    assert word[-1] == "Z1"
    _, corrected = logic_definitions(basis, word)
    _, uncorrected = logic_definitions(basis, word[:-1])
    assert [obs.metadata for obs in corrected] != [obs.metadata for obs in uncorrected]
    _, signed = logic_definitions(*SIGNED_RELABEL_WORD)
    assert {obs.id: obs.metadata["reference"] for obs in signed} == {0: 1, 1: 1}


def _guppy_dem(basis, word):
    detectors, observables = logic_definitions(basis, word)
    build = build_dem_from_guppy(
        make_helix_logic(basis, word),
        num_qubits=HELIX_MEMORY_NUM_QUBITS,
        detectors=detectors,
        observables=observables,
        **UNIFORM_NOISE,
    )
    assert build.dem.num_detectors == len(detectors)
    assert build.dem.num_observables == len(observables) == 2
    assert build.audit["named_result_binding"] == "generator_layout_v2_program_bound"
    return build.dem


def _symptoms(dem):
    """Return the distinct (detectors, observables) symptom sets of a DEM's error lines."""
    symptoms = set()
    for line in dem.to_string().splitlines():
        if line.startswith("error("):
            targets = line.split(")", 1)[1].split()
            assert all(t[0] in "DL" for t in targets), line
            symptoms.add((frozenset(t for t in targets if t[0] == "D"), frozenset(t for t in targets if t[0] == "L")))
    return symptoms


@pytest.mark.parametrize(("basis", "word"), RUNTIME_WORDS)
def test_dem(basis, word):
    # Selene's native gate decomposition puts noise on more locations, so probabilities differ;
    # the symptoms must not. This catches a Guppy trace that replays a relabel wrongly.
    tick_dem = DetectorErrorModel.from_circuit(logic_circuit(basis, word), **UNIFORM_NOISE)
    assert _symptoms(_guppy_dem(basis, word)) == _symptoms(tick_dem)


def _assert_noiseless_agreement(basis, word):
    program = make_helix_logic(basis, word)
    detectors, observables = logic_definitions(basis, word)
    expected = Counter(tag for tag, _ in helix_result_refs(logic_circuit(basis, word)))
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
            assert len(columns[tag]) == shots
            assert all(np.asarray(shot).size == count for shot in columns[tag])
            values = np.asarray(columns[tag]).reshape(shots, count)
            for occurrence in range(count):
                by_ref[result_ref(tag, occurrence=occurrence)] = values[:, occurrence]
        for kind, definitions in [("detector", detectors), ("observable", observables)]:
            for definition in definitions:
                parity = sum(by_ref[ref] for ref in definition.refs) % 2
                reference = 0 if kind == "detector" else definition.metadata["reference"]
                assert (parity == reference).all(), f"{kind} parity: {basis}, {word}, {seed}, {definition}"
        # Positive control: genuine data readouts vary after projective preparation.
        readouts = [by_ref[result_ref(f"data:{slot}")] for slot in range(20)]
        assert any(len(set(column.tolist())) == 2 for column in readouts), (basis, word, seed)


@pytest.mark.parametrize(("basis", "word"), RUNTIME_WORDS)
def test_noiseless_agreement(basis, word):
    _assert_noiseless_agreement(basis, word)


@pytest.mark.parametrize(("basis", "word"), PHASE_WORDS)
def test_first_s_mutation(monkeypatch, basis, word):
    def mutated(basis, word):
        source = render_helix_logic_source(basis, word)
        assert "    s(" in source
        source = source.replace(
            "from guppylang.std.quantum import ",
            "from guppylang.std.quantum import sdg, ",
        )
        return source.replace("    s(", "    sdg(", 1)

    monkeypatch.setattr(renderer, "render_helix_logic_source", mutated)
    with pytest.raises(AssertionError, match="detector parity"):
        _assert_noiseless_agreement(basis, word)


def test_drop_fixup_mutation(monkeypatch):
    def mutated(basis, word):
        source = render_helix_logic_source(basis, word)
        call = f"        gate_{word[-1]}(code)\n"
        assert source.count(call) == 1
        return source.replace(call, "")

    monkeypatch.setattr(renderer, "render_helix_logic_source", mutated)
    with pytest.raises(AssertionError, match="observable parity"):
        _assert_noiseless_agreement(*FIXUP_WORD)


@pytest.mark.parametrize(("basis", "word", "kind"), [(*RELABEL_WORD, "detector"), (*SIGNED_RELABEL_WORD, "observable")])
def test_drop_relabel_mutation(monkeypatch, basis, word, kind):
    def mutated(basis, word):
        source = render_helix_logic_source(basis, word)
        # Keep the now empty gate function syntactically valid, without any swaps.
        source, count = re.subn(r"    mem_swap\(code.data\[\d+\], code.data\[\d+\]\)", "    pass", source)
        assert count == sum(step.op_type == OpType.RELABEL for step in logical_gate_steps("SWAP"))
        return source

    monkeypatch.setattr(renderer, "render_helix_logic_source", mutated)
    with pytest.raises(AssertionError, match=f"{kind} parity"):
        _assert_noiseless_agreement(basis, word)


@pytest.mark.parametrize(
    ("basis", "weight"),
    [("X", 3), ("Z", 3), ("X", 4), pytest.param("Z", 4, marks=pytest.mark.slow)],
)
def test_guppy_fault_distance(basis, weight):
    """Equal distances, not equal DEMs: probabilities differ between the two paths (see test_dem)."""
    word = ("S0S1", "S0S1")
    guppy_dem = _guppy_dem(basis, word)
    tick_dem = DetectorErrorModel.from_circuit(logic_circuit(basis, word), **UNIFORM_NOISE)
    values = []
    for label, dem in [("Guppy", guppy_dem), ("tick", tick_dem)]:
        start = perf_counter()
        results = dem.per_observable_fault_distances(weight)
        distances = [None if result is None else result.distance for result in results]
        print(f"{label} basis={basis} max_weight={weight}: {distances}, {perf_counter() - start:.3f}s", flush=True)
        values.append(distances)
    assert values[0] == values[1]
    assert values[0] == ([4, 4] if (basis, weight) == ("X", 4) else [None, None])
