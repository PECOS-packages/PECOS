# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0

"""Acceptance oracle for code-agnostic Clifford detector discovery."""

import pytest
import stim
from pecos.qec import DiscoveredDetectors, discover_detectors
from pecos.qec.helix import c4_helix_code
from pecos.qec.surface import SurfacePatch, build_memory_circuit
from pecos.qec.surface.circuit_builder import tick_circuit_to_stim
from pecos.quantum import TickCircuit
from pecos.testing import deterministic_parity_basis


def _mask(records: list[int]) -> int:
    return sum(1 << record for record in records)


def _rank(rows: list[int]) -> int:
    pivots: dict[int, int] = {}
    for row in rows:
        value = row
        while value:
            pivot = value.bit_length() - 1
            if pivot not in pivots:
                pivots[pivot] = value
                break
            value ^= pivots[pivot]
    return len(pivots)


@pytest.mark.parametrize("distance", [3, 5])
@pytest.mark.parametrize("basis", ["X", "Z"])
@pytest.mark.parametrize("rounds", [3, 6])
def test_surface_memory_matches_hand_detectors(distance: int, basis: str, rounds: int) -> None:
    """Exact locality, all reference bits, completeness, and logical independence."""
    tc = build_memory_circuit(distance=distance, rounds=rounds, basis=basis)
    definitions = tc.circuit_definitions()
    observables = [entry["measurements"] for entry in definitions["observables"]]
    if (distance, basis, rounds) == (3, "Z", 3):
        assert observables == [[28, 29, 30]]
    # These are absolute record indices, unlike the JSON metadata's offsets.
    geometry = SurfacePatch.create(distance=distance).geometry
    logical = geometry.logical_x if basis == "Z" else geometry.logical_z
    assert logical is not None
    # The builder allocates the data register contiguously before the ancillas.
    data_qubits = list(range(distance * distance))
    pauli = "X" if basis == "Z" else "Z"
    conjugates = [(1 if basis == "Z" else 2, " ".join(f"{pauli}{data_qubits[q]}" for q in logical.data_qubits))]
    found = discover_detectors(tc, observables, conjugates)
    assert isinstance(found, DiscoveredDetectors)
    assert isinstance(found.detectors, list)
    assert isinstance(found.observables, list)
    hand = {tuple(sorted(entry["measurements"])) for entry in definitions["detectors"]}
    assert {tuple(records) for records, _ in found.detectors} == hand
    # The observable times any detector it overlaps is an equivalent representative
    # and must preserve the same logical-free detector set.
    for records in [records for records in hand if set(records) & set(observables[0])]:
        alternative = sorted(set(observables[0]) ^ set(records))
        moved = discover_detectors(tc, [alternative], conjugates)
        assert {tuple(found_records) for found_records, _ in moved.detectors} == hand, alternative
    assert len(found.detectors) == len(definitions["detectors"])
    assert [records for records, _ in found.observables] == observables
    assert [max(records) for records, _ in found.detectors] == sorted(max(records) for records, _ in found.detectors)
    detector_masks = [_mask(records) for records, _ in found.detectors]
    observable_masks = [_mask(records) for records, _ in found.observables]
    detector_rank = _rank(detector_masks)
    for observable in observable_masks:
        assert _rank([*detector_masks, observable]) == detector_rank + 1
    circuit = stim.Circuit(tick_circuit_to_stim(tc))
    for seed in (0, 1, 17):
        shots = circuit.compile_sampler(seed=seed).sample(2048)
        assert shots.shape[0] >= shots.shape[1] + 1
        for records, reference in [*found.detectors, *found.observables]:
            assert isinstance(records, list)
            assert records == sorted(set(records))
            assert type(reference) is int
            assert reference in (0, 1)
            assert ((shots[:, records].sum(axis=1) % 2) == reference).all()
        space = deterministic_parity_basis(shots)
        emitted = [*detector_masks, *observable_masks]
        assert _rank(emitted) == len(space)
        assert _rank([*space, *emitted]) == len(space)


def test_binding_defaults_sequences_and_errors() -> None:
    """Default observables, tuple inputs, integer references, and named errors."""
    tc = TickCircuit()
    tc.tick().h([0])
    tc.tick().mz([0])
    tc.tick().x([0])
    tc.tick().mz([0])
    assert discover_detectors(tc).detectors == [([0, 1], 1)]
    assert discover_detectors(tc, ((1,),), ((3, "X0"),)).detectors == []
    with pytest.raises(ValueError, match="observable 1 is non-deterministic: declared record 0"):
        discover_detectors(tc, ((1,), (0,)))
    with pytest.raises(ValueError, match="observable 0 lists record 1 more than once"):
        discover_detectors(tc, ((1, 1),))
    with pytest.raises(ValueError, match="observable 0 is empty"):
        discover_detectors(tc, ((),))
    with pytest.raises(ValueError, match="observable 0 record 2 is out of range for 2 records"):
        discover_detectors(tc, ((2,),))
    for record in (-1, 1 << 100):
        with pytest.raises(ValueError, match=f"observable 0 record {record} must be a nonnegative integer"):
            discover_detectors(tc, ((record,),))


@pytest.mark.parametrize(
    ("observables", "conjugates", "message"),
    [
        (((0,), (1,)), (), "2 observables but 0 conjugates"),
        ((), ((1, "X0"),), "0 observables but 1 conjugates"),
        (((0,),), ((2, "X0"),), "conjugate 0 before_tick 2 is out of range for 2 ticks"),
        (((0,),), ((1, "X2"),), "conjugate 0 qubit 2 is out of range for 2 qubits"),
        (((0,),), ((1, "Z0"),), r"conjugate 0 flips observables \[\]"),
        (((0,), (1,)), ((1, "XX"), (1, "X1")), r"conjugate 0 flips observables \[0, 1\]"),
        (((0,), (0,)), ((1, "X0"), (1, "X0")), r"conjugate 0 flips observables \[0, 1\]"),
        (((0,),), ((1, "not a Pauli"),), "conjugate 0 pauli .* is invalid"),
        (((0,),), ((-1, "X0"),), "conjugate 0 before_tick -1 must be a nonnegative integer"),
    ],
)
def test_conjugate_errors(observables: tuple, conjugates: tuple, message: str) -> None:
    """Pairing, bounds, counts and parsing report the invalid declaration."""
    tc = TickCircuit()
    tc.tick().pz([0, 1])
    tc.tick().mz([0, 1])
    with pytest.raises(ValueError, match=message):
        discover_detectors(tc, observables, conjugates)


@pytest.mark.parametrize("pauli", ["X0 X1 X2", "XXX", "-XXX", "iXXX"])
def test_zero_check_rounds_and_pauli_input(pauli: str) -> None:
    """Sparse/dense strings and phase yield the same cleaned check space."""
    tc = TickCircuit()
    tc.tick().pz([0, 1, 2])
    tc.tick().mz([0, 1, 2])
    assert discover_detectors(tc, [[0]], [(1, pauli)]).detectors == [([0, 1], 0), ([0, 2], 0)]
    with pytest.raises(TypeError):
        discover_detectors(tc, [[0]], [(1, 123)])


def _helix_memory(basis: str, rounds: int) -> tuple[TickCircuit, list[list[int]], list[tuple[int, str]]]:
    code = c4_helix_code()
    tc = TickCircuit()
    prepare = tc.tick()
    if basis == "Z":
        prepare.pz(list(range(code.n)))
    else:
        prepare.px(list(range(code.n)))
    # Five weight-4 rows plus the first four independent weight-8 rows, per type.
    checks = [code.stabilizers[:9], code.stabilizers[10:19]]
    for rows in checks:
        assert _rank([_mask([q for q, p in enumerate(row) if p != "I"]) for row in rows]) == 9
    for _ in range(rounds):
        for kind, rows in zip(("X", "Z"), checks, strict=True):
            start = tc.num_ticks()
            available = [start] * (code.n + 18)
            for index, row in enumerate(rows):
                ancilla = code.n + (0 if kind == "X" else 9) + index
                while tc.num_ticks() <= start:
                    tc.tick()
                if kind == "X":
                    tc.tick_at(start).px([ancilla])
                else:
                    tc.tick_at(start).pz([ancilla])
                available[ancilla] = start + 1
                for qubit, pauli in enumerate(row):
                    if pauli == "I":
                        continue
                    tick = max(available[qubit], available[ancilla])
                    while tc.num_ticks() <= tick:
                        tc.tick()
                    pair = (ancilla, qubit) if kind == "X" else (qubit, ancilla)
                    tc.tick_at(tick).cx([pair])
                    available[qubit] = available[ancilla] = tick + 1
                tick = available[ancilla]
                while tc.num_ticks() <= tick:
                    tc.tick()
                if kind == "X":
                    tc.tick_at(tick).mx([ancilla])
                else:
                    tc.tick_at(tick).mz([ancilla])
    base = tc.num_measurements()
    readout = tc.tick()
    if basis == "Z":
        readout.mz(list(range(code.n)))
    else:
        readout.mx(list(range(code.n)))
    logicals = code.logical_z if basis == "Z" else code.logical_x
    conjugates = code.logical_x if basis == "Z" else code.logical_z
    observables = [[base + q for q, pauli in enumerate(row) if pauli != "I"] for row in logicals]
    return tc, observables, [(1, pauli) for pauli in conjugates]


@pytest.mark.parametrize("basis", ["X", "Z"])
@pytest.mark.parametrize("rounds", [1, 2])
def test_helix_conjugates_flip_only_their_own_observables(basis: str, rounds: int) -> None:
    """Both logical injections preserve every detector in unflagged Helix memory."""
    tc, observables, conjugates = _helix_memory(basis, rounds)
    found = discover_detectors(tc, observables, conjugates)
    for injected_index, (before_tick, pauli) in enumerate(conjugates):
        injected_tc, _, _ = _helix_memory(basis, rounds)
        injection = injected_tc.insert_tick(before_tick)
        for qubit, axis in enumerate(pauli):
            if axis == "X":
                injection.x([qubit])
            elif axis == "Z":
                injection.z([qubit])
        # The surface exporter does not support PX/MX. This circuit uses only
        # these seven gates, translated directly in stored record order.
        gate_names = {"PZ": "R", "PX": "RX", "MZ": "M", "MX": "MX", "CX": "CX", "X": "X", "Z": "Z"}
        injected = stim.Circuit()
        for tick_index in range(injected_tc.num_ticks()):
            for gate in injected_tc.get_tick(tick_index).gate_batches():
                injected.append(gate_names[gate.gate_type.name], [int(q) for q in gate.qubits])
            injected.append("TICK")
        for seed in (0, 17):
            shots = injected.compile_sampler(seed=seed).sample(256)
            for records, reference in found.detectors:
                assert ((shots[:, records].sum(axis=1) % 2) == reference).all(), (injected_index, records)
            for index, (records, reference) in enumerate(found.observables):
                expected = reference ^ (index == injected_index)
                assert ((shots[:, records].sum(axis=1) % 2) == expected).all(), (injected_index, index)
