# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Signed gadget, compiled Clifford, and logical detector acceptance."""

from itertools import product
from random import Random

import pytest
import stim
from pecos.qec.circuit_steps import CircuitStep, OpType
from pecos.qec.helix import (
    HelixAllocation,
    LogicalGate,
    c4_helix_code,
    c4_helix_gates,
    compile_logical_clifford,
    helix_result_classifier,
    helix_result_refs,
    logic_circuit,
    logic_conjugates,
    logic_gadgets,
    logical_action,
    logical_gate_steps,
    logical_pauli_ops,
    logical_pauli_string,
    logical_token_ops,
    lower_steps,
    memory_circuit,
    memory_conjugates,
    syndrome_round_gadgets,
)
from pecos.qec.helix import gadgets as gadgets_module
from pecos.qec.helix import logic as logic_module
from pecos.qec.helix import logical as logical_module
from pecos.qec.helix.gadgets import memory_gadgets
from pecos.qec.surface.circuit_builder import tick_circuit_to_stim
from pecos.testing import deterministic_parity_basis, group_contains, simulate_tick_circuit, stabilizer_generators_after
from pecos_rslib import PauliStabilizerGroup, PauliString

from qec.helix.test_helix import physical


@pytest.mark.parametrize("schedule", ["asap", "step"])
def test_phase_lowering(schedule):
    steps = [CircuitStep(OpType.SZ, [0]), CircuitStep(OpType.SZDG, [1]), CircuitStep(OpType.CZ, [0, 1])]
    circuit = lower_steps(steps, 2, schedule=schedule).tick_circuit
    assert [(gate.gate_type.name, list(gate.qubits)) for _, gate in circuit.gate_batches()] == [
        ("SZ", [0]),
        ("SZdg", [1]),
        ("CZ", [0, 1]),
    ]
    assert circuit.num_ticks() == (2 if schedule == "asap" else 3)


@pytest.mark.parametrize(
    ("op", "qubits"),
    [(OpType.SZ, [0, 1]), (OpType.SZDG, []), (OpType.CZ, [0]), (OpType.CZ, [0, 0])],
)
def test_phase_lowering_arity(op, qubits):
    for schedule in ("asap", "step"):
        with pytest.raises(ValueError, match="Invalid qubits"):
            lower_steps([CircuitStep(op, qubits)], 2, schedule=schedule)


def test_flag_labels():
    classify = helix_result_classifier()
    for label in ("czf:0.z", "czf:6.x"):
        assert classify(label, lambda _: None) == ("scalar", label)
    for label in ("czf:7.z", "czf:0.y"):
        with pytest.raises(ValueError, match="label"):
            classify(label, lambda _: None)


def _assert_signed_gate(name, basis):
    code = c4_helix_code()
    steps = memory_gadgets(basis, 1)[0][1] + logical_gate_steps(name)
    lowered = lower_steps(steps, HelixAllocation.NUM_SLOTS, schedule="step")
    circuit = lowered.tick_circuit
    after = stabilizer_generators_after(circuit, circuit.num_ticks())
    width = len(after[0]) - 1
    clifford = physical(c4_helix_gates()[name].ops)
    prepared = [row for row in code.stabilizers if set(row) <= {"I", basis}]
    prepared.extend(code.logical_x if basis == "X" else code.logical_z)
    for row in prepared:
        expected = clifford.apply(PauliString.from_str(row)).to_dense_str(code.n)
        sign = "-" if expected.startswith("-") else "+"
        body = ["I"] * width
        for slot, axis in enumerate(expected.lstrip("+-")):
            body[lowered.final_layout[slot]] = axis
        assert group_contains(after, sign + "".join(body)), (name, basis, row, expected)
    for seed in range(3):
        results = dict(zip(helix_result_refs(circuit), simulate_tick_circuit(circuit, seed)[0], strict=True))
        assert all(value == 0 for (label, _), value in results.items() if label.startswith("czf:"))


@pytest.mark.parametrize(("name", "basis"), product(c4_helix_gates(), ("X", "Z")))
def test_signed_gate(name, basis):
    _assert_signed_gate(name, basis)


@pytest.mark.parametrize("index", [0, 6])
def test_cz_flag_step_sequence(index):
    czs = [qubits for op, qubits in c4_helix_gates()["S0S1"].ops if op == "cz"]
    q1, q2 = czs[index]
    zero, plus = HelixAllocation().cz_flags(index)
    steps = logical_gate_steps("S0S1")
    start = steps.index(CircuitStep(OpType.PREP, [zero]))
    assert steps[start : start + 12] == [
        CircuitStep(OpType.PREP, [zero]),
        CircuitStep(OpType.PREP, [plus]),
        CircuitStep(OpType.H, [plus]),
        CircuitStep(OpType.CX, [q1, zero]),
        CircuitStep(OpType.CX, [plus, q2]),
        CircuitStep(OpType.CZ, [q1, q2]),
        CircuitStep(OpType.CZ, [q1, plus]),
        CircuitStep(OpType.CX, [q1, zero]),
        CircuitStep(OpType.CX, [plus, q2]),
        CircuitStep(OpType.MEASURE, [zero], f"czf:{index}.z"),
        CircuitStep(OpType.H, [plus]),
        CircuitStep(OpType.MEASURE, [plus], f"czf:{index}.x"),
    ]


def test_signed_gate_detects_phase_translation_mutation(monkeypatch):
    """Mutate one translated SZ, leaving the physical oracle's ops untouched."""
    changed = False

    def mutated_step(op, *args, **kwargs):
        nonlocal changed
        if op == OpType.SZ and not changed:
            op = OpType.SZDG
            changed = True
        return CircuitStep(op, *args, **kwargs)

    monkeypatch.setattr(gadgets_module, "CircuitStep", mutated_step)
    with pytest.raises(AssertionError):
        _assert_signed_gate("S0S1", "X")
    assert changed
    changed = False
    _assert_signed_gate("S0S1", "Z")
    assert changed


@pytest.mark.parametrize("op", ["x", "y", "z", "cx"])
def test_paper_gate_rejects_unused_ops(monkeypatch, op):
    bad = {"bad": LogicalGate("bad", ((op, (0,)),))}
    monkeypatch.setattr(gadgets_module, "c4_helix_gates", lambda: bad)
    monkeypatch.setattr(logical_module, "c4_helix_gates", lambda: bad)
    with pytest.raises(ValueError, match=op):
        logical_gate_steps("bad")


@pytest.mark.parametrize("spelling", ["X0X1", "Z0Z1"])
def test_pauli_product_cancels_shared_slot(spelling):
    ops = logical_pauli_ops(spelling)
    assert all(qubits != (3,) for _, qubits in ops)
    assert [qubits[0] for _, qubits in ops] == sorted({qubits[0] for _, qubits in ops})
    assert logical_pauli_ops("I") == ()


@pytest.mark.parametrize("token", ["I", "unknown", "X1X0", "+X0", "X2"])
def test_invalid_token(token):
    for function in (logic_gadgets, logic_conjugates, logic_circuit):
        with pytest.raises(ValueError, match="Invalid logical gate token") as error:
            function("Z", (token,))
        assert repr(token) in str(error.value)


@pytest.mark.parametrize("token", [None, 5, ["X0"]])
def test_non_string_token(token):
    for function in (logic_gadgets, logic_conjugates, logic_circuit):
        with pytest.raises(TypeError, match="must be strings") as error:
            function("Z", (token,))
        assert repr(token) in str(error.value)
    with pytest.raises(TypeError, match="must be strings"):
        logical_gate_steps(token)


def test_compiler_validation():
    identity = {label: "+" + label for label in ("X0", "X1", "Z0", "Z1")}
    assert compile_logical_clifford(identity) == ()
    with pytest.raises(ValueError, match="keys"):
        compile_logical_clifford({"X0": "+X0"})
    with pytest.raises(ValueError, match="logical Pauli"):
        compile_logical_clifford({**identity, "X0": "X0"})
    with pytest.raises(ValueError, match=r"Target.*X0"):
        compile_logical_clifford({**identity, "X0": "+Z0"})


@pytest.mark.parametrize("label", ["X0", "X1", "Z0", "Z1"])
@pytest.mark.parametrize("image", ["X0", None])
def test_compiler_image_errors_name_key(label, image):
    target = {key: "+" + key for key in ("X0", "X1", "Z0", "Z1")}
    target[label] = image
    expected = TypeError if image is None else ValueError
    with pytest.raises(expected, match=f"Invalid image for {label}:") as error:
        compile_logical_clifford(target)
    assert type(error.value.__cause__) is expected


def _dense_product(*rows):
    pauli = PauliString.I()
    for row in rows:
        pauli = pauli * PauliString.from_str(row)
    return pauli.to_dense_str(20).lstrip("+-i")


def test_logical_pauli_string():
    code = c4_helix_code()
    x0, x1 = code.logical_x
    z0, z1 = code.logical_z
    assert logical_pauli_string("I") == "I" * 20
    assert logical_pauli_string("X0X1") == _dense_product(x0, x1)
    assert logical_pauli_string("Z0Z1") == _dense_product(z0, z1)
    assert logical_pauli_string("Y0") == _dense_product(x0, z0)
    assert logical_pauli_string("Z0Y1") == _dense_product(z0, x1, z1)
    # Hand-derived letter by letter from the table rows, independent of PauliString products.
    assert logical_pauli_string("Y0") == "ZIXYIIZZXIIXXIIXIIZZ"
    assert logical_pauli_string("Z0Y1") == "YIZXIIYYZIIZZIIZIIYY"
    for bad in (None, 5):
        with pytest.raises(TypeError, match=repr(bad)):
            logical_pauli_string(bad)
    with pytest.raises(ValueError, match="logical Pauli"):
        logical_pauli_string("X1X0")


def test_pulled_back_conjugates():
    code = c4_helix_code()
    x0, x1 = code.logical_x
    z0, z1 = code.logical_z
    assert logic_conjugates("Z", ("CNOT_SWAP",)) == [(1, x1), (1, _dense_product(x0, x1))]
    assert logic_conjugates("X", ("CNOT_SWAP",)) == [(2, _dense_product(z0, z1)), (2, z0)]
    assert logic_conjugates("Z", ("S0S1",)) == [(1, _dense_product(x0, z0)), (1, _dense_product(x1, z1))]
    expected = [(1, _dense_product(x0, x1)), (1, x1)]
    assert expected == [(1, "XIXIIIXXXIIXXIIXIIXX"), (1, "XIIXIIXXIIIIIIIIIIXX")]
    assert logic_conjugates("Z", ("CNOT_SWAP", "SWAP")) == expected


WORDS = [
    ("Z", ("SWAP",)),
    ("Z", ("CNOT_SWAP",)),
    ("Z", ("CNOT_SWAP", "SWAP")),
    ("Z", ("CZ",)),
    ("Z", ("S0S1",)),
    ("X", ("SWAP",)),
    ("X", ("CNOT_SWAP",)),
    ("X", ("SWAP", "CNOT_SWAP")),
    ("X", ("H_SWAP", "H_SWAP")),
    ("X", ("CZ", "CZ")),
    ("X", ("S0S1", "S0S1")),
]


def _assert_deterministic(circuit, seeds):
    references = {entry["id"]: entry["reference"] for entry in circuit.circuit_definitions()["observables"]}
    for seed in range(seeds):
        _, fired, observables = simulate_tick_circuit(circuit, seed)
        assert fired == 0
        assert observables == references
    return references


@pytest.mark.parametrize(("basis", "tokens"), WORDS)
def test_logic_determinism_and_completeness(basis, tokens):
    circuit = logic_circuit(basis, tokens)
    _assert_deterministic(circuit, 8)
    definitions = circuit.circuit_definitions()
    shots = stim.Circuit(tick_circuit_to_stim(circuit)).compile_sampler(seed=17).sample(2048)
    space = deterministic_parity_basis(shots)
    emitted = [
        sum(1 << r for r in entry["measurements"]) for entry in [*definitions["detectors"], *definitions["observables"]]
    ]
    n = len(helix_result_refs(circuit))

    def pauli(mask):
        return PauliString.from_str("".join("Z" if mask >> q & 1 else "I" for q in range(n)))

    group = PauliStabilizerGroup([pauli(mask) for mask in emitted])
    assert group.rank() == len(emitted) == len(space)
    assert all(group.contains(pauli(mask)) for mask in space)
    for logical, (boundary, dense) in enumerate(logic_conjugates(basis, tokens)):
        injected = logic_circuit(basis, tokens)
        tick = injected.insert_tick(boundary)
        for q, axis in enumerate(dense):
            if axis != "I":
                getattr(tick, axis.lower())([q])
        _, fired, observables = simulate_tick_circuit(injected)
        refs = {entry["id"]: entry["reference"] for entry in definitions["observables"]}
        assert fired == 0
        assert observables == {index: ref ^ (index == logical) for index, ref in refs.items()}


def _token_ops(tokens):
    return tuple(op for token in tokens for op in logical_token_ops(token))


def test_random_signed_sequences(compiled):
    _, words = compiled
    rng = Random(492)
    gates = c4_helix_gates()
    saw_fixup = False
    for _ in range(6):
        word = words[rng.choice(list(words))]
        action = logical_action(c4_helix_code(), LogicalGate("word", _token_ops(word)))
        target = {label: rng.choice("+-") + image[1:] for label, image in action.items()}
        forward = compile_logical_clifford(target)
        ops = _token_ops(forward)
        assert logical_action(c4_helix_code(), LogicalGate("forward", ops)) == target
        inverse_ops = tuple(({"sz": "szdg", "szdg": "sz"}.get(op, op), qs) for op, qs in reversed(ops))
        inverse = compile_logical_clifford(logical_action(c4_helix_code(), LogicalGate("inverse", inverse_ops)))
        for basis in ("X", "Z"):
            assert _assert_deterministic(logic_circuit(basis, forward + inverse), 4) == {0: 0, 1: 0}
        if forward and forward[-1] not in gates:
            saw_fixup = True
            # Drop only the forward correction: the signed inverse remains fixed.
            references = [
                _assert_deterministic(logic_circuit(basis, forward[:-1] + inverse), 4) for basis in ("X", "Z")
            ]
            assert any(any(refs.values()) for refs in references)
    assert saw_fixup


@pytest.mark.parametrize(("basis", "tokens"), [("X", ("S0S1",)), ("X", ("H_SWAP",)), ("Z", ("H_SWAP",))])
def test_random_readout_rejected(basis, tokens):
    for function in (logic_gadgets, logic_conjugates, logic_circuit):
        with pytest.raises(ValueError, match=f"{basis}.*{tokens[0]}"):
            function(basis, tokens)


@pytest.mark.parametrize("tokens", [("S0S1",), ("unknown",), ("I",)])
def test_invalid_word_rejected_before_gadget_expansion(monkeypatch, tokens):
    def unexpected_expansion(*_args):
        pytest.fail("Invalid words must be rejected before expanding gadgets")

    monkeypatch.setattr(logic_module, "logical_gate_steps", unexpected_expansion)
    monkeypatch.setattr(logic_module, "memory_gadgets", unexpected_expansion)
    with pytest.raises(ValueError, match=r"Random readout|Invalid logical gate token"):
        logic_gadgets("X", tokens)


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_empty_word_matches_memory(basis):
    circuit = logic_circuit(basis, ())
    memory = memory_circuit(basis, 1)
    assert circuit.circuit_definitions() == memory.circuit_definitions()
    assert helix_result_refs(circuit) == helix_result_refs(memory)
    assert logic_conjugates(basis, ()) == memory_conjugates(basis)


def test_rounds_and_pauli_expansion():
    word = ("S0S1", "X0", "S0S1", "Y1")
    gadgets = logic_gadgets("Z", word)
    expected = [memory_gadgets("Z", 1)[0], *syndrome_round_gadgets("a")]
    for token, variant in zip(word, ("b", None, "a", None), strict=True):
        expected.append((token, logical_gate_steps(token)))
        if variant:
            expected.extend(syndrome_round_gadgets(variant))
    expected.append(memory_gadgets("Z", 1)[-1])
    assert gadgets == expected
    assert logical_gate_steps("Y1") == [
        CircuitStep(op, list(qs))
        for axis, qs in logical_pauli_ops("Y1")
        for op in {"x": (OpType.X,), "z": (OpType.Z,), "y": (OpType.X, OpType.Z)}[axis]
    ]
