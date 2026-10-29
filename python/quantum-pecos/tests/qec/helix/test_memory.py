# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Memory acceptance gates for C4-Helix, arXiv:2609.03194."""

from collections import Counter
from itertools import product

import pytest
import stim
from pecos.qec.circuit_steps import CircuitStep, OpType
from pecos.qec.helix import (
    HelixAllocation,
    LoweredHelixCircuit,
    c4_helix_code,
    c4_round_steps,
    chosen_weight8_supports,
    helix_result_refs,
    lower_steps,
    memory_circuit,
    memory_conjugates,
    memory_steps,
    weight8_check_steps,
)
from pecos.qec.surface.circuit_builder import tick_circuit_to_stim
from pecos.simulators import PauliProp
from pecos.testing import deterministic_parity_basis, group_contains, simulate_tick_circuit, stabilizer_generators_after
from pecos_rslib import PauliStabilizerGroup, PauliString


@pytest.mark.parametrize("qubits", [[], [0], [0, 0], [0, 1, 2]])
def test_invalid_relabel(qubits: list[int]) -> None:
    with pytest.raises(ValueError, match="RELABEL requires two distinct qubits") as error:
        CircuitStep(OpType.RELABEL, qubits)
    assert repr(qubits) in str(error.value)
    step = CircuitStep(OpType.RELABEL, [0, 1])
    step.qubits = qubits
    with pytest.raises(ValueError, match="RELABEL requires two distinct qubits") as error:
        lower_steps([step], 3)
    assert repr(qubits) in str(error.value)


def _batches(lowered: LoweredHelixCircuit) -> list[list[tuple[str, list[int]]]]:
    tc = lowered.tick_circuit
    return [
        [(gate.gate_type.name, list(gate.qubits)) for gate in tc.get_tick(t).gate_batches()]
        for t in range(tc.num_ticks())
    ]


def test_relabel_and_asap_order() -> None:
    steps = [
        CircuitStep(OpType.PREP, [0]),
        CircuitStep(OpType.H, [0]),
        CircuitStep(OpType.RELABEL, [0, 1]),
        CircuitStep(OpType.X, [1]),
        CircuitStep(OpType.MEASURE, [1], "data:0"),
        CircuitStep(OpType.MEASURE, [0], "data:1"),
    ]
    lowered = lower_steps(steps, 2)
    assert lowered.final_layout == (1, 0)
    assert _batches(lowered) == [[("PZ", [0]), ("MZ", [1])], [("H", [0])], [("X", [0])], [("MZ", [0])]]
    assert helix_result_refs(lowered.tick_circuit) == [("data:1", 0), ("data:0", 0)]
    assert lowered.tick_circuit.gate_count() == 5


def test_barrier_comment_and_measurement_batch_order() -> None:
    lowered = lower_steps(
        [
            CircuitStep(OpType.MEASURE, [5], "data:5"),
            CircuitStep(OpType.COMMENT, label="ignored"),
            CircuitStep(OpType.H, [0]),
            CircuitStep(OpType.MEASURE, [2], "data:2"),
            CircuitStep(OpType.TICK),
            CircuitStep(OpType.Z, [4]),
            CircuitStep(OpType.CX, [0, 4]),
        ],
        6,
    )
    assert _batches(lowered) == [[("MZ", [5]), ("H", [0]), ("MZ", [2])], [("Z", [4])], [("CX", [0, 4])]]
    assert helix_result_refs(lowered.tick_circuit) == [("data:5", 0), ("data:2", 0)]
    assert lowered.final_layout == tuple(range(6))


@pytest.mark.parametrize(
    "op",
    [
        op
        for op in OpType
        if op
        not in {
            OpType.PREP,
            OpType.H,
            OpType.X,
            OpType.Z,
            OpType.CX,
            OpType.MEASURE,
            OpType.RELABEL,
            OpType.COMMENT,
            OpType.TICK,
        }
    ],
)
def test_unsupported_lowering(op: OpType) -> None:
    with pytest.raises(ValueError, match=op.name):
        lower_steps([CircuitStep(op, [0])], 2)


def test_allocation() -> None:
    allocation = HelixAllocation()
    slots = [allocation.data(q) for q in range(20)]
    slots.extend(q for block in range(5) for q in (allocation.c4_plus(block), allocation.c4_zero(block)))
    slots.extend(q for basis in ("X", "Z") for index in range(4) for q in allocation.check(basis, index))
    assert slots == list(range(HelixAllocation.NUM_SLOTS))
    for method, limit in [(allocation.data, 20), (allocation.c4_plus, 5), (allocation.c4_zero, 5)]:
        for bad in (-1, limit):
            with pytest.raises(ValueError, match=str(bad)):
                method(bad)
    for bad in (-1, 4):
        with pytest.raises(ValueError, match=str(bad)):
            allocation.check("X", bad)


@pytest.mark.parametrize("basis", ["x", "Y", "bad"])
def test_invalid_basis(basis: str) -> None:
    for call in [
        lambda: HelixAllocation().check(basis, 0),
        lambda: chosen_weight8_supports(basis),
        lambda: weight8_check_steps(basis, 0),
        lambda: memory_steps(basis, 1),
        lambda: memory_circuit(basis, 1),
        lambda: memory_conjugates(basis),
    ]:
        with pytest.raises(ValueError, match=basis):
            call()


@pytest.mark.parametrize("rounds", [-1, 0])
def test_invalid_rounds(rounds: int) -> None:
    with pytest.raises(ValueError, match=str(rounds)):
        memory_steps("X", rounds)


def test_invalid_variant() -> None:
    with pytest.raises(ValueError, match="other"):
        c4_round_steps(0, "other")


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_check_choice_and_cat_order(basis: str) -> None:
    rows = [row for row in c4_helix_code().stabilizers if set(row) <= {basis, "I"}]
    selected = chosen_weight8_supports(basis)
    assert len(selected) == 4
    group = PauliStabilizerGroup([PauliString.from_str(row) for row in rows if row.count(basis) == 4])
    expected = []
    for row in rows:
        pauli = PauliString.from_str(row)
        if row.count(basis) == 8 and not group.contains(pauli):
            expected.append(tuple(q for q, axis in enumerate(row) if axis != "I"))
            group.add_generator(pauli)
    assert selected == tuple(expected)
    assert group.rank() == 9
    assert all(group.contains(PauliString.from_str(row)) for row in rows)
    for index, support in enumerate(selected):
        a, f1, f2, f3 = HelixAllocation().check(basis, index)
        pairs = [
            (a, f1),
            (a, f2),
            (f1, f3),
            *zip((a, f1, f2, f3) * 2, support, strict=True),
            (a, f3),
            (f1, f2),
            (a, f1),
        ]
        assert [step.qubits for step in weight8_check_steps(basis, index) if step.op_type == OpType.CX] == [
            list(pair if basis == "X" else pair[::-1]) for pair in pairs
        ]


def _c4_input(basis: str, sx: int, sz: int) -> list[CircuitStep]:
    steps = [CircuitStep(OpType.PREP, [q]) for q in range(4)]
    steps.append(CircuitStep(OpType.H, [0]))
    steps.extend(CircuitStep(OpType.CX, [0, q]) for q in (1, 2, 3))
    if basis == "X":
        steps.extend(CircuitStep(OpType.H, [q]) for q in range(4))
    if sx == -1:
        steps.append(CircuitStep(OpType.Z, [3]))
    if sz == -1:
        steps.append(CircuitStep(OpType.X, [3]))
    return steps


def _physical(pauli: str, layout: tuple[int, ...], width: int) -> str:
    body = ["I"] * width
    for slot, axis in enumerate(pauli[1:]):
        body[layout[slot]] = axis
    return pauli[0] + "".join(body)


@pytest.mark.parametrize(("variant", "basis", "sx", "sz"), product(("a", "b"), ("X", "Z"), (1, -1), (1, -1)))
def test_signed_c4_round(variant: str, basis: str, sx: int, sz: int) -> None:
    prepared = _c4_input(basis, sx, sz)
    generators = [
        ("+" if sx == 1 else "-") + "XXXX",
        ("+" if sz == 1 else "-") + "ZZZZ",
        f"+{basis}{basis}II",
        f"+{basis}I{basis}I",
    ]
    initial = lower_steps(prepared, 4)
    before = stabilizer_generators_after(initial.tick_circuit, initial.tick_circuit.num_ticks())
    assert all(group_contains(before, generator) for generator in generators)
    lowered = lower_steps([*prepared, *c4_round_steps(0, variant)], HelixAllocation.NUM_SLOTS)
    tc = lowered.tick_circuit
    measurements, _, _ = simulate_tick_circuit(tc)
    results = dict(zip(helix_result_refs(lowered.tick_circuit), measurements, strict=True))
    mx, mz = results[("c4x:0", 0)], results[("c4z:0", 0)]
    assert (mx, mz) == (int(sx == -1), int(sz == -1))
    after = stabilizer_generators_after(tc, tc.num_ticks())
    width = len(after[0]) - 1
    assert all(group_contains(after, _physical(row, lowered.final_layout, width)) for row in ("+XXXX", "+ZZZZ"))
    group = PauliStabilizerGroup([PauliString.from_str(row) for row in generators])
    d2, d3 = (2, 3) if variant == "a" else (0, 1)
    for mask in range(4, 16):
        logical = group.element(mask).to_dense_str()
        if logical[0] not in "+-":
            logical = "+" + logical
        logical = logical.ljust(5, "I")
        negative = (logical[0] == "-") ^ (mx and logical[d2 + 1] in "XY") ^ (mz and logical[d3 + 1] in "ZY")
        expected = ("-" if negative else "+") + logical[1:]
        assert group_contains(after, _physical(expected, lowered.final_layout, width)), (
            variant,
            basis,
            sx,
            sz,
            logical,
        )


def _hook_violations(
    steps: list[CircuitStep],
    ancillas: tuple[int, ...],
    support: tuple[int, ...],
    basis: str,
) -> list[tuple[int, int, str]]:
    violations = []
    names = {OpType.PREP: "init |0>", OpType.H: "H", OpType.CX: "CNOT", OpType.MEASURE: "measure Z"}
    for location, step in enumerate(steps):
        if step.op_type == OpType.MEASURE:
            continue
        for qubit, pauli in product(ancillas, "XYZ"):
            sim = PauliProp(num_qubits=HelixAllocation.NUM_SLOTS)
            sim.faults = {axis: {qubit} if axis == pauli else set() for axis in "XYZ"}
            flagged = False
            for later in steps[location + 1 :]:
                target = tuple(later.qubits) if later.op_type == OpType.CX else later.qubits[0]
                result = sim.bindings[names[later.op_type]](sim, target)
                if later.op_type == OpType.MEASURE and later.label.startswith("flag:"):
                    flagged |= bool(result)
            faults = sim.faults
            x = (faults["X"] | faults["Y"]) & set(range(20))
            z = (faults["Z"] | faults["Y"]) & set(range(20))
            reduced = (x ^ set(support)) | z if basis == "X" else x | (z ^ set(support))
            if min(len(x | z), len(reduced)) >= 2 and not flagged:
                violations.append((location, qubit, pauli))
    return violations


@pytest.mark.parametrize(("basis", "index"), product(("X", "Z"), range(4)))
def test_weight8_hook_survey(basis: str, index: int) -> None:
    support = chosen_weight8_supports(basis)[index]
    ancillas = HelixAllocation().check(basis, index)
    assert not _hook_violations(weight8_check_steps(basis, index), ancillas, support, basis)
    # Negative control: remove the flags and let a single bare ancilla drive all data.
    a = ancillas[0]
    twin = [CircuitStep(OpType.PREP, [a])]
    if basis == "X":
        twin.append(CircuitStep(OpType.H, [a]))
    twin.extend(CircuitStep(OpType.CX, [a, q] if basis == "X" else [q, a]) for q in support)
    if basis == "X":
        twin.append(CircuitStep(OpType.H, [a]))
    twin.append(CircuitStep(OpType.MEASURE, [a], "syndrome"))
    assert _hook_violations(twin, (a,), support, basis)


@pytest.mark.parametrize(("basis", "rounds"), product(("X", "Z"), (1, 2, 3)))
def test_memory_records_determinism_and_completeness(basis: str, rounds: int) -> None:
    steps = memory_steps(basis, rounds)
    assert not any(step.op_type == OpType.TICK for step in steps)
    lowered = lower_steps(steps, HelixAllocation.NUM_SLOTS)
    layout = list(range(HelixAllocation.NUM_SLOTS))
    physical_labels = {}
    occurrences = Counter()
    for step in steps:
        if step.op_type == OpType.RELABEL:
            a, b = step.qubits
            layout[a], layout[b] = layout[b], layout[a]
        elif step.op_type == OpType.MEASURE:
            ref = (step.label, occurrences[step.label])
            occurrences[step.label] += 1
            physical_labels[ref] = layout[step.qubits[0]]
    assert lowered.final_layout == tuple(layout)
    assert [physical_labels[label] for label in helix_result_refs(lowered.tick_circuit)] == [
        q for q, _ in lowered.tick_circuit.measurement_emission()
    ]
    assert set(helix_result_refs(lowered.tick_circuit)) == {(f"data:{q}", 0) for q in range(20)} | {
        (f"{kind}:{b}", r) for kind in ("c4x", "c4z") for b in range(5) for r in range(rounds)
    } | {(f"{axis}8:{i}", r) for axis in "xz" for i in range(4) for r in range(rounds)} | {
        (f"flag:{axis}{i}.{k}", r) for axis in "xz" for i in range(4) for k in range(1, 4) for r in range(rounds)
    }
    tc = memory_circuit(basis, rounds)
    definitions = tc.circuit_definitions()
    references = {entry["id"]: entry["reference"] for entry in definitions["observables"]}
    assert set(references) == {0, 1}
    for seed in range(8):
        _, fired, observables = simulate_tick_circuit(tc, seed)
        assert fired == 0
        assert observables == references
    shots = stim.Circuit(tick_circuit_to_stim(tc)).compile_sampler(seed=17).sample(2048)
    space = deterministic_parity_basis(shots)
    emitted = [
        sum(1 << record for record in entry["measurements"])
        for entry in [*definitions["detectors"], *definitions["observables"]]
    ]
    n = len(helix_result_refs(tc))
    group = PauliStabilizerGroup(
        [PauliString.from_str("".join("Z" if mask >> q & 1 else "I" for q in range(n))) for mask in emitted],
    )
    assert group.rank() == len(emitted) == len(space)
    assert all(
        group.contains(PauliString.from_str("".join("Z" if mask >> q & 1 else "I" for q in range(n)))) for mask in space
    )


@pytest.mark.parametrize(("basis", "rounds"), product(("X", "Z"), (1, 2, 3)))
def test_memory_logical_free_detectors(basis: str, rounds: int) -> None:
    for logical, (before_tick, pauli) in enumerate(memory_conjugates(basis)):
        tc = memory_circuit(basis, rounds)
        references = {entry["id"]: entry["reference"] for entry in tc.circuit_definitions()["observables"]}
        injection = tc.insert_tick(before_tick)
        for qubit, axis in enumerate(pauli):
            if axis != "I":
                getattr(injection, axis.lower())([qubit])
        _, fired, observables = simulate_tick_circuit(tc)
        assert fired == 0
        assert observables == {index: ref ^ (index == logical) for index, ref in references.items()}


def test_nonzero_detector_reference_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from pecos.qec.helix import memory

    def nonzero_reference(*_args: object) -> SimpleNamespace:
        return SimpleNamespace(detectors=[([0], 0), ([1, 2], 1)], observables=[])

    monkeypatch.setattr(memory, "discover_detectors", nonzero_reference)
    with pytest.raises(ValueError, match=r"Detector 1 has nonzero reference 1: \[1, 2\]"):
        memory_circuit("Z", 1)


def test_memory_alternates_swap_out_slots() -> None:
    relabels = [step.qubits for step in memory_steps("Z", 3) if step.op_type == OpType.RELABEL]
    assert relabels == [
        [4 * block + offset, 20 + 2 * block + ancilla]
        for offsets in ((2, 3), (0, 1), (2, 3))
        for block in range(5)
        for ancilla, offset in enumerate(offsets)
    ]


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_conjugate_injection_tick_follows_data_preparation(basis: str) -> None:
    """The conjugates' boundary must sit after data preparation and before any data CX."""
    before_tick = memory_conjugates(basis)[0][0]
    lowered = lower_steps(memory_steps(basis, 1), HelixAllocation.NUM_SLOTS)
    preparation = {"PZ", "H"} if basis == "X" else {"PZ"}
    data = set(range(20))
    data_gates = [
        (tick, gate.gate_type.name)
        for tick, gate in lowered.tick_circuit.gate_batches()
        if data & {int(q) for q in gate.qubits}
    ]
    assert all(name in preparation for tick, name in data_gates if tick < before_tick)
    assert {name for tick, name in data_gates if tick < before_tick} == preparation
    assert min(tick for tick, name in data_gates if name == "CX") == before_tick


@pytest.mark.parametrize(("basis", "index", "sign"), product(("X", "Z"), range(4), (1, -1)))
def test_signed_weight8_check(basis: str, index: int, sign: int) -> None:
    """A data eigenstate of the check gives its eigenvalue, quiet flags, and is left unchanged."""
    support = chosen_weight8_supports(basis)[index]
    steps = [CircuitStep(OpType.PREP, [q]) for q in range(20)]
    if basis == "X":
        steps.extend(CircuitStep(OpType.H, [q]) for q in range(20))
    if sign == -1:
        steps.append(CircuitStep(OpType.Z if basis == "X" else OpType.X, [support[0]]))
    steps.extend(weight8_check_steps(basis, index))
    lowered = lower_steps(steps, HelixAllocation.NUM_SLOTS)
    tc = lowered.tick_circuit
    results = dict(zip(helix_result_refs(lowered.tick_circuit), simulate_tick_circuit(tc)[0], strict=True))
    assert results[(f"{basis.lower()}8:{index}", 0)] == int(sign == -1)
    assert [value for label, value in results.items() if label[0].startswith("flag:")] == [0, 0, 0]
    after = stabilizer_generators_after(tc, tc.num_ticks())
    width = len(after[0]) - 1
    for q in range(20):
        flipped = sign == -1 and q == support[0]
        body = ["I"] * width
        body[q] = basis
        assert group_contains(after, ("-" if flipped else "+") + "".join(body)), q
