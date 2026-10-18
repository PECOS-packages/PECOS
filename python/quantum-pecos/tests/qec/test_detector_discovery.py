# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0

"""Acceptance oracle for code-agnostic Clifford detector discovery."""

import pytest
import stim
from pecos.qec import DiscoveredDetectors, discover_detectors
from pecos.qec.surface import build_memory_circuit
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
    found = discover_detectors(tc, observables)
    assert isinstance(found, DiscoveredDetectors)
    assert isinstance(found.detectors, list)
    assert isinstance(found.observables, list)
    assert {tuple(records) for records, _ in found.detectors} == {
        tuple(sorted(entry["measurements"])) for entry in definitions["detectors"]
    }
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
    declared = discover_detectors(tc, ((1,),))
    assert declared.detectors == []
    assert declared.observables == [([0, 1], 1)]
    with pytest.raises(ValueError, match="observable 1 is non-deterministic: declared record 0"):
        discover_detectors(tc, ((1,), (0,)))
    with pytest.raises(ValueError, match="observable 0 is empty"):
        discover_detectors(tc, ((),))
    with pytest.raises(ValueError, match="observable 0 record 2 is out of range for 2 records"):
        discover_detectors(tc, ((2,),))
    for record in (-1, 1 << 100):
        with pytest.raises(ValueError, match=f"observable 0 record {record} must be a nonnegative integer"):
            discover_detectors(tc, ((record,),))
