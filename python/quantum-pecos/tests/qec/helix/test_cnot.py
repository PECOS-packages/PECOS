# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Two-block transversal CNOT structure, discovery, and logical injections."""

from collections import Counter
from types import SimpleNamespace

import pytest
import stim
from pecos.qec.circuit_steps import CircuitStep, OpType
from pecos.qec.dem_spec import Detector, Observable, result_ref
from pecos.qec.helix import (
    HelixAllocation,
    c4_helix_code,
    cnot_circuit,
    cnot_conjugates,
    cnot_definitions,
    cnot_gadgets,
    helix_result_refs,
    lower_steps,
)
from pecos.qec.helix import cnot as cnot_module
from pecos.qec.helix.gadgets import memory_gadgets, syndrome_round_gadgets
from pecos.qec.surface.circuit_builder import tick_circuit_to_stim
from pecos.testing import deterministic_parity_basis, simulate_tick_circuit
from pecos_rslib import PauliStabilizerGroup, PauliString

PAIRS = ("ZZ", "XX", "ZX")
BUILDERS = (cnot_gadgets, cnot_conjugates, cnot_circuit, cnot_definitions)


@pytest.mark.parametrize("pair", PAIRS)
def test_gadget_structure(pair):
    gadgets = cnot_gadgets(*pair)
    expected = [f"{block}_prepare_{basis.lower()}" for block, basis in zip("AB", pair, strict=True)]
    for variant in "ab":
        if variant == "b":
            expected.append("cnot")
        for block in "AB":
            expected.extend(f"{block}_c4_{index}_{variant}" for index in range(5))
            expected.extend(f"{block}_weight8_{axis}_{index}" for axis in "xz" for index in range(4))
    expected.extend(f"{block}_readout_{basis.lower()}" for block, basis in zip("AB", pair, strict=True))
    assert [name for name, _ in gadgets] == expected
    assert dict(gadgets)["cnot"] == [CircuitStep(OpType.CX, [q, 76 + q]) for q in range(20)]
    assert not any(step.op_type == OpType.TICK for _, steps in gadgets for step in steps)
    for name, steps in gadgets:
        if name == "cnot":
            continue
        block = name[0]
        offset = 0 if block == "A" else 76
        assert all(offset <= q < offset + 76 for step in steps for q in step.qubits)
        assert all(step.label.startswith(f"{block}.") for step in steps if step.op_type == OpType.MEASURE)
        # Undo only the block transform to check reuse of all preparation/round/readout steps.
        originals = dict(memory_gadgets(pair["AB".index(block)], 1) + syndrome_round_gadgets("b"))
        assert [
            CircuitStep(step.op_type, [q - offset for q in step.qubits], step.label[2:] if step.label else "")
            for step in steps
        ] == originals[name[2:]]


@pytest.mark.parametrize("builder", BUILDERS)
@pytest.mark.parametrize("pair", ["XZ", "YZ", "XY", "xZ", "Zz"])
def test_rejected_before_building(monkeypatch, builder, pair):
    def unexpected(*_args):
        pytest.fail("Invalid basis pairs must be rejected before building anything")

    monkeypatch.setattr(cnot_module, "memory_gadgets", unexpected)
    monkeypatch.setattr(cnot_module, "c4_helix_code", unexpected)
    message = "Random readout for basis pair 'XZ'" if pair == "XZ" else "Expected basis X or Z, got"
    with pytest.raises(ValueError, match=message) as error:
        builder(*pair)
    if pair == "XZ":
        assert "Bell pair" in str(error.value)
        assert "ZZ, XX, ZX" in str(error.value)


@pytest.mark.parametrize("pair", PAIRS)
def test_records_determinism_and_completeness(pair):
    circuit = cnot_circuit(*pair)
    refs = helix_result_refs(circuit)
    round_tags = {f"c4{axis}:{i}" for axis in "xz" for i in range(5)}
    round_tags |= {f"{axis}8:{i}" for axis in "xz" for i in range(4)}
    round_tags |= {f"flag:{axis}{i}.{k}" for axis in "xz" for i in range(4) for k in range(1, 4)}
    expected = Counter({f"{block}.{tag}": 2 for block in "AB" for tag in round_tags})
    expected.update({f"{block}.data:{q}": 1 for block in "AB" for q in range(20)})
    assert Counter(tag for tag, _ in refs) == expected
    assert set(refs) == {(tag, occurrence) for tag, count in expected.items() for occurrence in range(count)}
    assert len(refs) == 208
    definitions = circuit.circuit_definitions()
    assert len(definitions["detectors"]) == 168
    assert [entry["reference"] for entry in definitions["observables"]] == [0] * 4
    for seed in range(8):
        _, fired, observables = simulate_tick_circuit(circuit, seed)
        assert fired == 0
        assert observables == dict.fromkeys(range(4), 0)
    shots = stim.Circuit(tick_circuit_to_stim(circuit)).compile_sampler(seed=17).sample(2048)
    space = deterministic_parity_basis(shots)
    entries = [*definitions["detectors"], *definitions["observables"]]
    for entry in entries:
        assert (shots[:, entry["measurements"]].sum(axis=1) % 2 == entry.get("reference", 0)).all()
    emitted = [sum(1 << record for record in entry["measurements"]) for entry in entries]

    def pauli(mask):
        return PauliString.from_str("".join("Z" if mask >> q & 1 else "I" for q in range(len(refs))))

    group = PauliStabilizerGroup([pauli(mask) for mask in emitted])
    assert group.rank() == len(emitted) == len(space)
    assert all(group.contains(pauli(mask)) for mask in space)


@pytest.mark.parametrize("pair", PAIRS)
def test_conjugates_and_injections(pair):
    code = c4_helix_code()
    idle = "I" * 76
    xs = [row.ljust(76, "I") for row in code.logical_x]
    zs = [row.ljust(76, "I") for row in code.logical_z]
    expected = {
        "ZZ": [(1, x + x) for x in xs] + [(1, idle + x) for x in xs],
        "XX": [(2, z + idle) for z in zs] + [(2, z + z) for z in zs],
        "ZX": [(1, x + idle) for x in xs] + [(2, idle + z) for z in zs],
    }
    assert cnot_conjugates(*pair) == expected[pair]
    for logical, (boundary, dense) in enumerate(cnot_conjugates(*pair)):
        circuit = cnot_circuit(*pair)
        tick = circuit.insert_tick(boundary)
        for q, axis in enumerate(dense):
            if axis != "I":
                getattr(tick, axis.lower())([q])
        _, fired, observables = simulate_tick_circuit(circuit)
        assert fired == 0
        assert observables == {index: int(index == logical) for index in range(4)}


@pytest.mark.parametrize("pair", PAIRS)
def test_conjugate_boundary_follows_preparation(pair):
    circuit = cnot_circuit(*pair)
    conjugates = cnot_conjugates(*pair)
    for block, basis in enumerate(pair):
        boundary = conjugates[2 * block][0]
        assert boundary == (1 if basis == "Z" else 2)
        data = set(range(76 * block, 76 * block + 20))
        gates = [(tick, gate) for tick, gate in circuit.gate_batches() if data & set(gate.qubits)]
        before = Counter(
            (tick, gate.gate_type.name, q) for tick, gate in gates if tick < boundary for q in gate.qubits if q in data
        )
        expected = Counter((0, "PZ", q) for q in data)
        if basis == "X":
            expected.update((1, "H", q) for q in data)
        assert before == expected
        assert min(tick for tick, gate in gates if gate.gate_type.name == "CX") == boundary
    cross_ticks = [
        tick
        for tick, gate in circuit.gate_batches()
        if gate.gate_type.name == "CX"
        for a, b in zip(gate.qubits[::2], gate.qubits[1::2], strict=True)
        if a < 76 <= b
    ]
    assert len(cross_ticks) == 20
    assert min(cross_ticks) == 6


@pytest.mark.parametrize("pair", PAIRS)
def test_typed_definitions_match_discovery(pair):
    circuit = cnot_circuit(*pair)
    refs = [result_ref(tag, occurrence=occurrence) for tag, occurrence in helix_result_refs(circuit)]
    detectors, observables = cnot_definitions(*pair)
    for kind, definitions, cls in [("detectors", detectors, Detector), ("observables", observables, Observable)]:
        entries = circuit.circuit_definitions()[kind]
        assert len(definitions) == len(entries)
        for index, (definition, entry) in enumerate(zip(definitions, entries, strict=True)):
            assert isinstance(definition, cls)
            assert definition.id == entry["id"] == index
            assert definition.refs == tuple(refs[r] for r in entry["measurements"])
            if kind == "observables":
                assert definition.metadata == {"reference": entry["reference"]}


@pytest.mark.parametrize("builder", [cnot_circuit, cnot_definitions])
def test_nonzero_detector_reference_rejected(monkeypatch, builder):
    def nonzero(*_args):
        return SimpleNamespace(detectors=[([0], 1)], observables=[([1], 0)])

    monkeypatch.setattr(cnot_module, "discover_detectors", nonzero)
    with pytest.raises(ValueError, match="Detector 0 has nonzero reference 1"):
        builder("Z", "Z")


@pytest.mark.parametrize("pair", ["ZZ", "XX"])
def test_removed_cnot_fires_original_detectors(pair):
    """Carry parities by tag and occurrence because removing CNOT changes record order.

    Exactly the detectors that compare the two blocks fire. ZX is excluded:
    every stabilizer CNOT mixes into |0>|+> is a prepared eigenstate with a
    deterministic value, so noiseless parities agree with and without it.
    """
    original = cnot_circuit(*pair)
    original_refs = helix_result_refs(original)
    steps = [step for name, gadget in cnot_gadgets(*pair) if name != "cnot" for step in gadget]
    mutated = lower_steps(steps, 2 * HelixAllocation.NUM_SLOTS).tick_circuit
    records = {ref: record for record, ref in enumerate(helix_result_refs(mutated))}
    detectors = original.circuit_definitions()["detectors"]
    carried = [[records[original_refs[r]] for r in entry["measurements"]] for entry in detectors]
    shots = stim.Circuit(tick_circuit_to_stim(mutated)).compile_sampler(seed=31).sample(64)
    fired = {index for index, parity in enumerate(carried) if (shots[:, parity].sum(axis=1) % 2).any()}
    cross_block = {
        index
        for index, entry in enumerate(detectors)
        if {original_refs[r][0][0] for r in entry["measurements"]} == {"A", "B"}
    }
    assert fired
    assert fired == cross_block
